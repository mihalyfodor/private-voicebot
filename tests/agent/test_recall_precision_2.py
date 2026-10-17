"""PRD 42 — recall precision 2: title words never score a chunk, no neighbouring chunks, the
server-side lookup fallback when the model claims the documents do not cover the question,
the wiki page in the recall block, short proper questions, proper bigrams, "summarise this"
under focus, the routing wording, the "open X please" command and the dated raw name.
Scenario numbers are the PRD's (11 is the existing suites, 12 the live battery).

Fixtures: `fallout` / `both` (generated FAQs) from tests/vault/test_source_recall.py, the
streamed fakes and the seeded vault from tests/conftest.py.
"""

import re
from unittest.mock import patch

import pytest

from core import settings
from tools import commands
from agent import conversation
from agent import offers
from agent import prompt as m_prompt
from agent import recall
from agent import turn as a_turn
from agent import turn as a_turn
from core import util
from ingest import pipeline
from vault import chunks
from vault import search
from tests.conftest import (FakeClient, ask_turn, deltas, last_user, seeded_vault,  # noqa: F401
                            stream_chunks, tool_call_chunk, write_page)
from tests.conftest import both, fallout, summary_json  # noqa: F401  (fixtures)
from core import llm as core_llm

pytestmark = pytest.mark.usefixtures("fresh_llm")

TITLE_TOKENS = {"fallout", "4", "guide", "skyrim"}
COVENANT_QUESTION = "I'm at Covenant and Honest Dan keeps attacking me, what now"
NOT_COVERED_REPLY = "[sad] The provided documents do not mention Covenant turrets."
OFFER_REPLY = "[happy] Talk to Jacob Orden, says the Fallout 4 guide. Want me to dig deeper?"


def _parts(block: str) -> list:
    """The `(part n/N)` numbers quoted in a recall / lookup block, in order."""
    return [int(n) for n in re.findall(r"\(part (\d+)/\d+\)", block)]


# --------------------------------------------------------------- 1. title stripping


def test_title_words_pick_the_source_and_never_score(both):
    terms = chunks.content_terms("in the Fallout 4 guide, what about Covenant turrets")
    topic, named = chunks.split_title_terms(terms)
    assert not topic & TITLE_TOKENS
    assert "covenant" in topic
    assert named == {both["fallout"]["path"]}

    hits = chunks.search_chunks("in the Fallout 4 guide, what about Covenant turrets")
    assert hits
    assert {h["source"] for h in hits} == {both["fallout"]["path"]}
    assert all(not set(h["matched"]) & TITLE_TOKENS for h in hits)
    assert "Covenant" in hits[0]["text"]


def test_a_question_naming_the_guide_quotes_a_covenant_chunk_never_the_contents_page(both):
    block = recall.recall_for("in the Fallout 4 guide, what about Covenant turrets")
    assert 'From "Fallout 4 guide"' in block and "Covenant" in block
    assert "(part 1/" not in block


def test_a_title_only_question_still_searches(both):
    assert chunks.search_chunks("what is the Fallout 4 guide about")


# --------------------------------------------------------------- 2. neighbour dedupe


def test_recall_never_quotes_two_neighbouring_chunks(fallout, file_sandbox):
    settings.save_setting("recall_min_score", 0.0)
    parts = _parts(recall.recall_for("how do I pass the SAFE test at Covenant with Honest Dan"))
    assert 1 <= len(parts) <= recall.RECALL_MAX_CHUNKS
    for a in parts:
        for b in parts:
            assert a == b or abs(a - b) >= 2, parts


def test_recall_hits_skip_a_neighbour_of_a_kept_chunk(fallout):
    question = "how do I pass the SAFE test at Covenant with Honest Dan"
    proper = chunks.proper_terms(chunks.content_terms(question))
    hits = recall._recall_hits(question, proper, 0.0, recall.RECALL_CANDIDATES)
    assert 1 <= len(hits) <= recall.RECALL_MAX_CHUNKS
    assert all(abs(a["n"] - b["n"]) >= 2 for a in hits for b in hits if a is not b)
    # every kept hit passes the precision rule
    pairs = chunks.proper_phrases(question)
    assert all(len(h["matched"]) >= 2 or chunks.proper_match(h, proper, pairs) for h in hits)
    # the neighbour rule is what dropped a candidate, if the raw ranking had one
    ranked = chunks.search_chunks(question, k=recall.RECALL_CANDIDATES)
    assert all((h["source"], h["n"]) in {(r["source"], r["n"]) for r in ranked} for h in hits)


# --------------------------------------------------------------- 3. the lookup fallback


def test_a_not_covered_reply_is_dropped_and_the_turn_re_asked_with_a_lookup(fallout, file_sandbox):
    settings.save_setting("recall_min_score", 999)
    assert recall.recall_for(COVENANT_QUESTION) == ""           # recall injects nothing
    events, fake = ask_turn(COVENANT_QUESTION, NOT_COVERED_REPLY, OFFER_REPLY)

    assert not any(kind == "delta" and "do not mention" in payload for kind, payload in events)
    assert ("tool_calls", ["search_wiki"]) in events
    assert len(fake.calls) == 2
    user = last_user(fake)
    assert user.startswith(COVENANT_QUESTION)
    assert "<lookup>" in user and 'From "Fallout 4 guide"' in user and "Covenant" in user
    assert recall.FALLBACK_NOTE.format(titles='"Fallout 4 guide"') in user
    assert deltas(events) == OFFER_REPLY
    assert conversation._conversation[-1]["content"] == OFFER_REPLY
    assert conversation._conversation[-2]["content"] == COVENANT_QUESTION   # the block is stripped after the turn
    assert conversation._deeper_offer["source"] == fallout["path"]


def test_a_grounded_reply_streams_normally_in_one_call(fallout, file_sandbox):
    settings.save_setting("recall_min_score", 999)
    events, fake = ask_turn(COVENANT_QUESTION, "[happy] Talk to Jacob.")
    assert len(fake.calls) == 1
    assert deltas(events) == "[happy] Talk to Jacob."
    assert ("tool_calls", ["search_wiki"]) not in events


def test_not_covered_about_a_topic_no_source_has_streams_as_is(fallout, file_sandbox):
    reply = "[sad] The provided documents do not mention a submarine periscope."
    events, fake = ask_turn("how does a submarine periscope work in the documents", reply)
    assert len(fake.calls) == 1
    assert deltas(events) == reply
    assert "<lookup>" not in last_user(fake)


def test_a_tool_call_turn_never_retries(fallout, file_sandbox):
    settings.save_setting("recall_min_score", 999)
    fake = FakeClient(*[
        [tool_call_chunk(id="call_1", name="get_weather", arguments="{}", finish_reason="tool_calls")],
        stream_chunks(NOT_COVERED_REPLY),
    ])
    with patch.object(core_llm, "_client", fake), patch.object(a_turn, "run_tool", return_value="21C sunny"):
        events = list(a_turn.ask_events(COVENANT_QUESTION))
    assert len(fake.calls) == 2
    assert events[0] == ("tool_calls", ["get_weather"])
    assert ("tool_calls", ["search_wiki"]) not in events
    assert deltas(events) == NOT_COVERED_REPLY


@pytest.mark.parametrize("text", [
    "the provided documents do not mention who trains Unarmored",
    "the snippet only contains the cover and introduction",
    "I don't have any information about Vivec in my files",
    "I don't have any recorded decisions regarding the Falcon project",
])
def test_not_covered_matches_the_stress_test_phrasings(text):
    assert recall.NOT_COVERED.search(text), text


@pytest.mark.parametrize("text", ["Talk to Jacob Orden.", "Ada Lovelace was an English mathematician."])
def test_not_covered_leaves_answers_alone(text):
    assert not recall.NOT_COVERED.search(text), text


# --------------------------------------------------------------- 4. page recall


FALCON_PAGE = ("# Falcon\n\n## Summary\nThe billing rewrite.\n\n## Decisions\n"
               "- 2026-08-28: we decided to ship Falcon on Postgres, not on Kafka event sourcing\n\n"
               "## Notes\n\n## Related\n\n## History\n")


def test_a_named_page_joins_the_recall_block_with_summary_and_decisions(seeded_vault):
    write_page("projects/falcon", "Falcon", FALCON_PAGE)
    block = recall.recall_for("What did we decide on Falcon?")
    assert 'From page "Falcon" (projects/falcon)' in block
    assert "billing rewrite" in block and "Postgres" in block
    assert util.tokens_est(block) <= recall.RECALL_PAGE_TOKENS + 100


def test_a_tool_command_question_injects_no_page(seeded_vault):
    assert recall.recall_for("did I get any email from Anna today") == ""


def test_a_page_never_joins_on_a_common_body_word_alone(seeded_vault):
    # "deploy" is in Anna's summary ("runs the Friday deploy") but is not her name
    assert search.search("what about the platform team deploy", 2)     # the page does match
    assert recall.recall_for("what about the platform team deploy") == ""


# --------------------------------------------------------------- 5. short proper questions


def test_a_short_question_with_a_proper_term_recalls(fallout):
    block = recall.recall_for("Who is Covenant?")
    assert 'From "Fallout 4 guide"' in block and "Covenant" in block


@pytest.mark.parametrize("text", ["any eggs left?", "hi", "Covenant"])
def test_short_questions_without_a_proper_term_or_a_lone_word_stay_empty(fallout, text):
    assert recall.recall_for(text) == ""


# --------------------------------------------------------------- 6. proper bigrams


def test_proper_phrases_pairs_a_proper_term_with_an_unseen_word(fallout):
    assert chunks.proper_phrases("who was Covenant Lovelace") == [("covenant", "lovelace")]
    assert ("honest", "dan") in chunks.proper_phrases("Honest Dan at Covenant")


def test_a_broken_bigram_demotes_the_proper_term(fallout):
    assert chunks.search_chunks("who was Covenant Lovelace")            # lone Covenant chunks exist
    assert recall.recall_for("who was Covenant Lovelace") == ""


def test_a_bigram_the_chunk_holds_whole_still_recalls(fallout):
    block = recall.recall_for(COVENANT_QUESTION)
    assert 'From "Fallout 4 guide"' in block


# --------------------------------------------------------------- 7. summarise under focus


def test_summarise_this_gets_the_source_head_not_a_chunk(fallout):
    recall.set_focus("sources", fallout["path"])
    block = recall.focus_block("summarise this")
    assert "Fallout 4 guide" in block
    assert "Part " not in block


def test_a_topical_question_under_focus_still_gets_the_chunk(fallout):
    recall.set_focus("sources", fallout["path"])
    assert "Part " in recall.focus_block("what about Covenant turrets")


def test_topic_terms_drop_the_generic_words():
    assert chunks.topic_terms("summarise this") == set()
    assert chunks.topic_terms("what does it say about turrets") == {"turret"}


# --------------------------------------------------------------- 8. routing wording


def test_the_system_prompt_routes_news_and_opinions():
    prompt = m_prompt.build_system_prompt({"name": "Test", "persona": "You are Test.", "verbosity": None})
    assert "today's headlines only" in prompt
    assert "never get the offer either" in prompt
    assert "asked for the news or today's headlines" in prompt


def test_focus_and_deeper_notes_forbid_names_from_memory():
    assert "do not add locations or names from memory" in recall.FOCUS_NOTE
    assert "do not add locations or names from memory" in offers.DEEPER_NOTE


# --------------------------------------------------------------- 9. "open X please"


@pytest.mark.parametrize("text, query", [
    ("open the window please", "window"),
    ("open Anna, please", "Anna"),
    ("please open Anna", "Anna"),
    ("open the window", "window"),
])
def test_page_command_strips_a_trailing_please(text, query):
    assert commands.page_command(text) == ("open", query)


# --------------------------------------------------------------- 10. the dated raw name


def test_an_upload_named_with_a_date_carries_todays_date_once(seeded_vault, monkeypatch):
    client = FakeClient(summary_json("Launch note"), "[]")
    monkeypatch.setattr(core_llm, "_client", client)
    filed = pipeline.ingest_upload(b"Fuel first.", "2026-08-30-note.md")
    today = util.today_iso()
    # "note" is a generic name, so the raw is renamed after the model title (PRD 32) — still
    # `raw/inbox/<today>-…note.md`, dated exactly once
    assert filed["raw"].startswith(f"raw/inbox/{today}-") and filed["raw"].endswith("note.md")
    assert len(re.findall(r"\d{4}-\d{2}-\d{2}", filed["raw"])) == 1


def test_dated_never_doubles_the_date():
    today = util.today_iso()
    assert pipeline._dated("2026-08-30-note") == f"{today}-note"
    assert pipeline._dated("note") == f"{today}-note"
