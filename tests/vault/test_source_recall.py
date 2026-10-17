"""PRD 32 — search inside sources: chunking + cache, chunk search, read_source, auto-recall,
calibration, the raw rename after a model title. Scenario numbers are the PRD's.

The fixture is a generated FAQ-shaped text (~150 k tokens, Title-case sections incl. "Covenant")
filed with a mocked model; a second generated FAQ covers the cross-source checks. Scenario 8
(`live`) files the two real guides from ~/llm-wiki-dev into the temp vault and talks to oMLX.
"""
import os

from unittest.mock import patch

import pytest

from core import settings
from agent import context
from agent import conversation
from agent import prompt
from agent import recall
from agent import turn
from agent import written
from core import util
from memory import identity
from ingest import pipeline
from vault import manifest
from vault import chunks as v_chunks
from vault import focus
from vault import search
from vault import store
from vault import chunks as v_chunks
from tests.conftest import (FALLOUT_SECTIONS, SKYRIM_SECTIONS, FakeClient, both, faq_text,  # noqa: F401
                            fallout, file_faq, seeded_vault, send_setting, summary_json, vault)
from tools import run_tool
from core import llm as core_llm
from ingest import pipeline
from core import toolset


def cache_files() -> list:
    root = os.path.join(store.WIKI_PATH, store.RAW_INDEX_DIR)
    return sorted(os.listdir(root)) if os.path.isdir(root) else []


# --------------------------------------------------------------- 1. chunking + cache


def test_the_fixture_is_faq_sized():
    text = faq_text("Fallout 4", FALLOUT_SECTIONS, seed=4)
    assert 120_000 <= util.tokens_est(text) <= 200_000
    assert "==== Covenant ====" in text


def test_chunks_split_on_paragraphs_at_about_800_tokens_with_overlap():
    text = faq_text("Fallout 4", FALLOUT_SECTIONS, seed=4)
    chunks = v_chunks.chunk_text(text)
    assert chunks == v_chunks.chunk_text(text)                       # deterministic
    assert [c["n"] for c in chunks] == list(range(1, len(chunks) + 1))
    sizes = [util.tokens_est(c["text"]) for c in chunks]
    assert max(sizes) <= v_chunks.CHUNK_TOKENS + v_chunks.CHUNK_OVERLAP + 10
    assert sorted(sizes)[len(sizes) // 2] >= v_chunks.CHUNK_TOKENS * 0.6
    for c in chunks:
        assert text[c["start"]:c["start"] + 40] == c["text"][:40]   # offsets point at the text
        assert c["start"] == 0 or text[c["start"] - 1] in "\n "     # a line or word boundary
    # the head of a chunk is the tail of the previous one: overlap
    overlaps = [chunks[i + 1]["text"][:60] in chunks[i]["text"] for i in range(len(chunks) - 1)]
    assert sum(overlaps) >= len(overlaps) * 0.9
    assert "==== Covenant ====" in "".join(c["text"] for c in chunks)


def test_a_heading_starts_its_own_chunk_and_a_long_line_is_cut():
    text = "# One\n\npara one\n\n" + "x" * 5000 + "\n\n# Two\n\npara two\n"
    chunks = v_chunks.chunk_text(text, tokens=100, overlap=0)
    assert chunks[0]["text"] == "# One\n\npara one"                 # the long line starts fresh
    assert all(len(c["text"]) <= 400 for c in chunks)
    assert chunks[-1]["text"].endswith("# Two\n\npara two")          # paragraphs stay whole
    starts = [c["start"] for c in chunks]
    assert starts[:2] == [0, 17] and len(starts) == 14                  # 100 tokens = 400 chars a cut
    assert all(b - a == 400 for a, b in zip(starts[1:], starts[2:]))
    assert v_chunks.chunk_text("") == []


def test_raw_chunks_are_cached_per_sha_and_reused(fallout, monkeypatch):
    entry = manifest.manifest_find(manifest.manifest_entries()[0]["sha256"])
    assert cache_files() == [f"{entry['sha256']}.json"]         # built eagerly by the filing
    first = v_chunks.raw_chunks(fallout["raw"])
    assert first["sha"] == entry["sha256"] and first["raw"] == fallout["raw"]
    assert first["source"] == fallout["path"] and first["title"] == "Fallout 4 guide"
    assert {"n", "start", "text"} <= set(first["chunks"][0])

    def no_read(rel):
        raise AssertionError(f"raw bytes read again: {rel}")

    monkeypatch.setattr(v_chunks, "_read_bytes", no_read)
    assert v_chunks.raw_chunks(fallout["raw"]) == first
    assert v_chunks.raw_chunks(fallout["path"]) == first             # the page path works too
    assert v_chunks.search_chunks("covenant compound")               # the corpus reads the cache
    assert f"{store.RAW_INDEX_DIR}/" in store._read(".gitignore").splitlines()


def test_an_existing_vault_gets_the_cache_dir_ignored(wiki_sandbox):
    store.ensure_vault()
    store._write(".gitignore", ".obsidian/\n")
    store.ensure_vault()
    assert store._read(".gitignore") == ".obsidian/\nraw/.index/\n"     # `.index/` ignores itself (PRD 53)


def test_images_and_calendars_have_no_chunks(seeded_vault, monkeypatch):
    monkeypatch.setattr(pipeline, "_downscale", lambda full: None)
    monkeypatch.setattr(pipeline, "_image_part", lambda full: {"type": "image_url"})
    client = FakeClient(summary_json("A screenshot"), "[]")
    monkeypatch.setattr(core_llm, "_client", client)
    filed = pipeline.ingest_upload(b"\x89PNG", "shot.png")
    assert v_chunks.raw_chunks(filed["raw"])["chunks"] == []
    assert v_chunks.search_chunks("screenshot") == []


# --------------------------------------------------------------- 2. search


def test_search_chunks_finds_the_covenant_section(fallout):
    hits = v_chunks.search_chunks("covenant compound turrets")
    assert hits and "Covenant" in hits[0]["text"]
    assert hits[0]["source"] == fallout["path"] and hits[0]["title"] == "Fallout 4 guide"
    assert hits[0]["n"] >= 1 and hits[0]["total"] > 100 and hits[0]["score"] > 0
    assert v_chunks.search_chunks("submarine periscope") == []


def test_page_search_is_unchanged(fallout):
    pages = search.search("anna")
    assert pages[0]["path"] == "people/anna"
    assert set(pages[0]) == {"path", "snippet", "score"}          # no chunks on page hits
    assert search.search("covenant") == [] or search.search("covenant")[0]["path"].startswith("sources/")


# --------------------------------------------------------------- 3. the tools


def test_search_wiki_lists_pages_then_snippets(fallout):
    text = run_tool("search_wiki", {"query": "anna covenant"})
    lines = text.splitlines()
    assert lines[0] == '<document source="wiki pages">' and lines[1].startswith("people/anna:")
    snippets = [line for line in lines if "· part " in line]
    assert 1 <= len(snippets) <= 3
    assert snippets[0].startswith("Fallout 4 guide · part ")
    assert f"({fallout['path']})" in snippets[0]
    assert all(len(line) <= v_chunks.SNIPPET_CHARS + 80 for line in snippets)
    assert lines.index(snippets[0]) > lines.index("</document>")  # pages first, snippets last (PRD 33 wraps each)
    assert run_tool("search_wiki", {"query": "submarine"}) == "no matching wiki pages"


def test_read_source_returns_the_part_with_its_neighbours(fallout):
    hit = v_chunks.search_chunks("covenant compound")[0]
    text = run_tool("read_source", {"path": fallout["path"], "part": hit["n"]})
    assert text.startswith(f'<document source="Fallout 4 guide">\nFallout 4 guide · part {hit["n"]}/{hit["total"]}\n')
    assert len(text) <= v_chunks.MAX_SOURCE_READ + 120
    record = v_chunks.raw_chunks(fallout["raw"])
    before, after = record["chunks"][hit["n"] - 2]["text"], record["chunks"][hit["n"]]["text"]
    assert before[-100:] in text and after[:100] in text
    assert "Fallout 4 guide · part 1/" in run_tool("read_source", {"path": fallout["raw"], "part": 1})
    assert run_tool("read_source", {"path": "sources/nope", "part": 1}) == "no such source: sources/nope"
    assert run_tool("read_source", {"path": fallout["path"], "part": 0}).startswith("no part 0 in")
    assert run_tool("read_source", {"path": fallout["path"], "part": "x"}).startswith("no part x in")
    assert v_chunks.read_source(fallout["path"], 10 ** 6).startswith("no part 1000000 in")


def test_the_tool_is_registered_and_advertised():
    assert "read_source" in focus.RECALL_TOOLS
    assert "read_source" in {t["function"]["name"] for t in toolset.TOOLS}
    assert "read_source" in identity.WIKI_NOTE


# --------------------------------------------------------------- 4. recall_for


def test_recall_for_quotes_the_covenant_chunk_and_nothing_for_chit_chat(fallout):
    block = recall.recall_for("I'm at covenant and the turrets keep shooting me")
    assert block.startswith("\n\n<recall>\n") and block.endswith("</recall>")
    quotes = block.split('From "')[1:]
    assert 1 <= len(quotes) <= recall.RECALL_MAX_CHUNKS
    assert all(q.startswith("Fallout 4 guide\" (part ") and "Covenant" in q for q in quotes)
    assert 'name "Fallout 4 guide"' in block and prompt.DEEPER_OFFERS[0] in block
    assert util.tokens_est(block) <= recall.RECALL_MAX_TOKENS + 40
    for text in ("hi", "what time is it", "what should we cook tonight", "tell me a joke"):
        assert recall.recall_for(text) == "", text


def test_recall_for_can_quote_two_chunks_within_the_budget(fallout, file_sandbox):
    settings.save_setting("recall_min_score", 0.0)
    block = recall.recall_for("how do I pass the SAFE test at Covenant with Honest Dan")
    assert block.count("From \"Fallout 4 guide\"") == recall.RECALL_MAX_CHUNKS
    assert util.tokens_est(block) <= recall.RECALL_MAX_TOKENS + 40


# --------------------------------------------------------------- 5. calibration


def questions(sections: list) -> list:
    return [f"I'm at {title} and {names[0]} keeps attacking me, what now" for title, names in sections[:10]]


OFF_TOPIC = [
    "what should we cook tonight for dinner",
    "how is the weather looking in Budapest today",
    "did I get any email from Anna today",
    "what did we decide about the Friday deploy",
    "remind me what Anna said in the meeting",
    "I want to plan a trip to Italy next summer",
    "can you tell me a joke about programmers",
    "what time is the dentist appointment tomorrow",
    "write a short letter to my landlord about the rent",
    "how long should I boil an egg for breakfast",
]


def test_calibration_in_guide_questions_hit_their_section(both):
    assert v_chunks.recall_min_score() == v_chunks.RECALL_MIN_SCORE
    for key, sections, title in (("fallout", FALLOUT_SECTIONS, "Fallout 4 guide"),
                                 ("skyrim", SKYRIM_SECTIONS, "Skyrim guide")):
        hits = 0
        for (section, _), q in zip(sections, questions(sections)):
            block = recall.recall_for(q)
            first = block.split('From "')[1].split("\nIf this answers")[0] if block else ""
            if first.startswith(title) and section in first:
                hits += 1
            assert f'From "{title}"' in block or not block, (q, block[:200])
        assert hits >= 8, (key, hits)


def test_calibration_off_topic_questions_inject_no_chunk(both):
    """No guide chunk for any of them; the Anna page joining a question about Anna is PRD 42 §3."""
    injected = [q for q in OFF_TOPIC if 'From "' in recall.recall_for(q)]
    assert injected == []
    assert 'From page "Anna" (people/anna)' in recall.recall_for("remind me what Anna said in the meeting")


def test_calibration_keeps_the_guides_apart(both):
    skyrim = recall.recall_for("I'm at Whiterun and the Companions won't talk to me")
    assert 'From "Skyrim guide"' in skyrim and 'From "Fallout 4 guide"' not in skyrim
    fallout = recall.recall_for("I'm at Covenant and they shoot me on sight")
    assert 'From "Fallout 4 guide"' in fallout and 'From "Skyrim guide"' not in fallout
    hits = v_chunks.search_chunks("how do I lockpick a master lock", k=3)
    assert {h["title"] for h in hits} <= {"Fallout 4 guide", "Skyrim guide"}
    assert all("Lockpick" in h["text"] for h in hits)


# --------------------------------------------------------------- 6. per-turn injection


def test_the_recall_block_rides_on_the_user_message_only(fallout):
    conversation.reset()
    system_before = conversation._conversation[0]["content"]
    question = "I'm at Covenant and Honest Dan keeps attacking me, what now"
    fake = FakeClient("[thinking] The guide says talk to Jacob Orden.")
    with patch.object(core_llm, "_client", fake):
        reply = turn.ask(question)
    assert reply.startswith("[thinking]")
    sent = fake.calls[0]["messages"]
    assert sent[0]["content"] == system_before and "<recall>" not in sent[0]["content"]
    user = sent[1]
    assert user["role"] == "user" and user["content"].startswith(question + "\n\n<recall>")
    assert conversation._session_turns[0] == {"role": "user", "content": question}
    assert conversation._conversation[1] == {"role": "user", "content": question}   # bare once the turn is done (PRD 39)
    assert context.estimate_tokens([user]) <= util.tokens_est(question) + recall.RECALL_MAX_TOKENS + 60

    # the next turn carries its own (empty) recall; nothing from the last one leaks
    fake = FakeClient("[happy] Sure.")
    with patch.object(core_llm, "_client", fake):
        turn.ask("thanks a lot for that")
    assert "<recall>" not in conversation._conversation[3]["content"]
    assert conversation._conversation[0]["content"] == system_before


def test_a_typed_turn_gets_the_same_recall_as_a_spoken_one(fallout):
    conversation.reset()
    question = "I'm at Covenant and Honest Dan keeps attacking me, what now"
    fake = FakeClient("[neutral] ok", "[neutral] ok")
    with patch.object(core_llm, "_client", fake):
        list(turn.ask_events(question, origin="typed"))
        list(turn.ask_events(question, origin="user"))
    typed, spoken = fake.calls[0]["messages"][1], fake.calls[1]["messages"][3]
    # the same passages; a typed content question only reads the full-answer wording (PRD 43)
    assert "<recall>" in typed["content"] and typed["content"] == written.written_note(spoken["content"])
    assert conversation._conversation[1]["content"] == conversation._conversation[3]["content"] == question


def test_the_threshold_setting_round_trips(fallout, file_sandbox):
    from fastapi.testclient import TestClient
    from server import app

    assert v_chunks.recall_min_score() == v_chunks.RECALL_MIN_SCORE
    client = TestClient(app.app)
    msg = send_setting("recall_min_score", 50, client)
    assert msg == {"type": "setting", "key": "recall_min_score", "value": 50.0}
    assert v_chunks.recall_min_score() == 50.0
    assert client.get("/api/config").json()["recall_min_score"] == 50.0
    assert recall.recall_for("I'm at Covenant and Honest Dan keeps attacking me, what now") == ""
    assert send_setting("recall_min_score", "high", client)["type"] == "error"


# --------------------------------------------------------------- 7. re-file / forget


def test_refiling_the_same_bytes_builds_nothing_and_forget_drops_the_cache(fallout, monkeypatch):
    before = cache_files()
    stamp = os.path.getmtime(os.path.join(store.WIKI_PATH, store.RAW_INDEX_DIR, before[0]))

    def no_read(rel):
        raise AssertionError(f"raw bytes read again: {rel}")

    monkeypatch.setattr(v_chunks, "_read_bytes", no_read)
    again = file_faq(faq_text("Fallout 4", FALLOUT_SECTIONS, seed=4), "Fallout 4 guide")
    assert again["already"] and again["raw"] == fallout["raw"]
    assert cache_files() == before
    assert os.path.getmtime(os.path.join(store.WIKI_PATH, store.RAW_INDEX_DIR, before[0])) == stamp

    sha = manifest.manifest_entries()[0]["sha256"]
    pipeline.forget(sha)
    assert cache_files() == []
    assert v_chunks.search_chunks("covenant compound") == []
    assert recall.recall_for("I'm at Covenant and Honest Dan keeps attacking me, what now") == ""


# --------------------------------------------------------------- 10. raw named after the title


def test_a_pasted_note_is_renamed_after_the_model_title(seeded_vault, monkeypatch):
    client = FakeClient(summary_json("Falcon launch checklist"), "[]")
    monkeypatch.setattr(core_llm, "_client", client)
    filed = pipeline.ingest_note("Fuel, then countdown, then launch.")
    assert filed["raw"] == f"raw/inbox/{util.today_iso()}-falcon-launch-checklist.md"
    assert os.path.isfile(store._path(filed["raw"]))
    assert not os.path.exists(store._path(f"raw/inbox/{util.today_iso()}-note.md"))
    entry = manifest.manifest_entries()[0]
    assert entry["path"] == filed["raw"]
    assert f"raw: {filed['raw']}" in store._read(filed["path"] + ".md")
    assert pipeline.list_sources()[0]["raw"] == filed["raw"]              # the "Open raw" link
    assert v_chunks.raw_chunks(filed["raw"])["raw"] == filed["raw"]
    assert store._read(f"{store.RAW_INDEX_DIR}/{entry['sha256']}.json")     # cache built under the new name

    monkeypatch.setattr(core_llm, "_client", FakeClient(summary_json("Other"), "[]"))
    again = pipeline.ingest_note("Fuel, then countdown, then launch.")
    assert again["already"] and again["raw"] == filed["raw"]
    assert len(manifest.manifest_entries()) == 1

    pipeline.forget(entry["sha256"])
    assert not os.path.exists(store._path(filed["raw"]))
    assert manifest.manifest_entries() == [] and cache_files() == []


def test_a_second_paste_with_the_same_title_is_deduped(seeded_vault, monkeypatch):
    monkeypatch.setattr(core_llm, "_client", FakeClient(summary_json("Falcon"), "[]"))
    first = pipeline.ingest_note("Fuel first.")
    monkeypatch.setattr(core_llm, "_client", FakeClient(summary_json("Falcon"), "[]"))
    second = pipeline.ingest_note("Countdown second.")
    assert first["raw"].endswith("-falcon.md") and second["raw"].endswith("-falcon-2.md")


def test_a_real_file_name_is_not_renamed(seeded_vault, monkeypatch):
    client = FakeClient(summary_json("Falcon launch checklist"), "[]")
    monkeypatch.setattr(core_llm, "_client", client)
    filed = pipeline.ingest_upload(b"Fuel, then countdown.", "falcon-plan.txt")
    assert filed["raw"].endswith("-falcon-plan.txt")
    assert manifest.manifest_entries()[0]["path"] == filed["raw"]


# --------------------------------------------------------------- 8. live


REAL_GUIDES = (
    ("2026-08-30-note.md", "Fallout 4 FAQ and Walkthrough", "Fallout"),
    ("2026-08-30-note-2.md", "Elder Scrolls V FAQ and Walkthrough", "Elder Scrolls"),
)
LIVE_QUESTIONS = [
    ("I'm at Covenant and they shoot me on sight, what am I supposed to do", "Fallout"),
    ("how do I pass the SAFE test in Covenant and what is Honest Dan looking for", "Fallout"),
    ("where do I find the Silver Shroud costume at Hubris Comics", "Fallout"),
    ("how do I join the Companions in Whiterun and what do they ask first", "Elder Scrolls"),
    ("where do I learn the Unrelenting Force shout from the Greybeards", "Elder Scrolls"),
    ("how do I become Thane of Whiterun after the dragon at the watchtower", "Elder Scrolls"),
    ("how do I lockpick?", None),
]


@pytest.mark.live
def test_live_answers_cite_the_right_guide(seeded_vault, capsys, monkeypatch):
    """The two real guides, filed with a mocked summary (the point is recall, not filing), then
    seven questions through `llm.ask` against oMLX. Asserts the source mention; prints the pairs."""
    dev = os.path.expanduser("~/llm-wiki-dev/raw/inbox")
    for name, title, _ in REAL_GUIDES:
        path = os.path.join(dev, name)
        if not os.path.isfile(path):
            pytest.skip(f"{path} not available")
        with open(path, "rb") as f:
            data = f.read()
        monkeypatch.setattr(core_llm, "_client", FakeClient(summary_json(title), "[]"))
        pipeline.file_bytes(data, f"raw/inbox/{name}", "md")
    core_llm._client = None
    conversation.reset()
    pairs = []
    for question, expect in LIVE_QUESTIONS:
        block = recall.recall_for(question)
        assert block, f"no recall for {question!r}"
        if expect == "Elder Scrolls":
            assert 'From "Fallout' not in block, question
        conversation.reset()
        answer = turn.ask(question)
        cited = expect or ("Fallout" if 'From "Fallout' in block else "Elder Scrolls")
        wrong = "Elder Scrolls" if cited == "Fallout" else "Fallout 4"
        names = (cited, "Skyrim") if cited == "Elder Scrolls" else (cited,)
        generic = any(w in answer.lower() for w in ("guide", "walkthrough", "faq"))
        ok = wrong not in answer and (any(n in answer for n in names) or generic)
        pairs.append((question, answer, block, ok))
    with capsys.disabled():
        print()
        for q, a, block, ok in pairs:
            quoted = [line.split("):")[0] for line in block.splitlines() if line.startswith("From ")]
            print(f"Q: {q}\nA{'' if ok else ' (no source named)'}: {a}\n   recall: {quoted}\n")
    assert all(ok for *_, ok in pairs), [q for q, _, _, ok in pairs if not ok]
