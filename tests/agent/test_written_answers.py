"""PRD 43 — written answers: length by origin + groundedness, the "write it up" card, Copy /
Save to wiki, the explicit `notes/` save, "save that as a note", new topic + reconnect replay.
Scenario numbers are the PRD's (7 is the live battery in tests/qa/stress.py). The briefing card is
docs/46-briefing.md scenario 7.

Fixtures: `fallout` (the generated FAQ) from tests/vault/test_source_recall.py, the streamed
fakes, the vault and the turn helpers from tests/conftest.py.
"""
import os

import re
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from server.session import Session

from server import app
from server import panels
from server import turns as server_turns
from server import ws as m_ws
from server import turns as server_turns
from server import panels
from server import speech as s_speech
from tools import commands
from agent import conversation
from agent import offers
from agent import prompt
from agent import recall as m_recall
from agent import turn as m_turn
from agent import written as m_written
from core import util
from vault import pages
from vault import store
from tests.conftest import (FakeClient, commits, fake_asr, fake_tts, last_user, read,  # noqa: F401
                            run_turn, stream_chunks, vault)
from tests.conftest import fallout  # noqa: F401  (fixtures)
from core import llm as core_llm

pytestmark = pytest.mark.usefixtures("fresh_llm")

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
QUESTION = "how do I get the Compound Key at Covenant?"
OFFER_REPLY = "[thinking] Talk to Jacob Orden, says the Fallout 4 guide. Want me to dig deeper, or write it up?"
CARD_MD = ("# The Compound Key at Covenant\n\n## Where it is\nJacob Orden holds it.\n\n"
           "## Steps\n1. Pass the SAFE test.\n2. Talk to Jacob.\n")


def card_response(markdown: str = CARD_MD):
    """A non-streamed completion (what `write_up` asks for) holding the card markdown."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=markdown))])


def grounded_turn(origin: str = "user", reply: str = OFFER_REPLY) -> FakeClient:
    """One recall turn on the Fallout FAQ; returns the fake so the call can be inspected."""
    fake = FakeClient(*[stream_chunks(reply)])
    with patch.object(core_llm, "_client", fake):
        list(m_turn.ask_events(QUESTION, origin=origin))
    return fake


def turn(text: str, responses: list, spoken: bool = False) -> list:
    """One typed (or spoken) turn through chatbot with the queued fake responses; the messages sent."""
    fake = FakeClient(*responses)
    patches = [patch.object(core_llm, "_client", fake), patch.object(s_speech, "synthesize", fake_tts)]
    if spoken:
        patches.append(patch.object(server_turns, "asr", fake_asr(text)))
    with patches[0], patches[1]:
        if spoken:
            with patches[2]:
                return run_turn(server_turns.handle_utterance, Session(), 1, np.zeros(10, dtype=np.float32))
        return run_turn(server_turns.handle_text, Session(), 1, text)


def write_up_turn(spoken: bool = False) -> list:
    return turn("write it up", [card_response()], spoken=spoken)


def written(sent: list) -> list:
    """The cards of a turn as they end up: each `card` with its `card_update`s applied (PRD 51)."""
    cards = {m["id"]: dict(m) for m in sent if m["type"] == "card"}
    for m in sent:
        if m["type"] == "card_update":
            cards[m["id"]].update({k: v for k, v in m.items() if k not in ("type", "id")})
    return list(cards.values())


def collect(ws, until: str) -> list:
    got = []
    while True:
        msg = ws.receive_json()
        got.append(msg)
        if msg["type"] in (until, "error"):
            return got


# --------------------------------------------------------------- 1. length by origin


def test_a_typed_grounded_content_question_runs_long_without_the_offer(fallout):
    fake = grounded_turn(origin="typed")
    call = fake.calls[0]
    assert call["max_tokens"] == prompt.VERBOSITY_MAX_TOKENS["long"]
    system, user = call["messages"][0]["content"], last_user(fake)
    assert prompt.WRITTEN_CLAUSE in system and prompt.OFFER_CLAUSE not in system
    assert prompt.VERBOSITY_RULES["long"] in system
    assert "<recall>" in user and m_recall.FULL_PHRASE in user and m_recall.NOTE_NO_OFFER in user
    assert m_recall.NOTE_OFFER not in user and m_recall.SHORT_PHRASE not in user
    # the swap was for that completion only: the stored prompt still carries the short rule
    assert prompt.OFFER_CLAUSE in conversation._conversation[0]["content"]
    assert conversation._conversation[1]["content"] == QUESTION


def test_typed_small_talk_keeps_the_normal_budget(fallout):
    fake = FakeClient("[happy] Hi!")
    with patch.object(core_llm, "_client", fake):
        list(m_turn.ask_events("hi", origin="typed"))
    assert fake.calls[0]["max_tokens"] == prompt._effective_max_tokens() == 400
    assert "<recall>" not in last_user(fake) and prompt.WRITTEN_CLAUSE not in fake.calls[0]["messages"][0]["content"]


def test_the_same_question_spoken_stays_short_and_offers_to_write_it_up(fallout):
    fake = grounded_turn(origin="user")
    call = fake.calls[0]
    assert call["max_tokens"] == 400
    assert prompt.OFFER_CLAUSE in call["messages"][0]["content"]
    assert m_recall.NOTE_OFFER in last_user(fake) and m_recall.SHORT_PHRASE in last_user(fake)
    assert prompt.DEEPER_OFFERS[0] == "Want me to dig deeper, or write it up?"
    assert offers.ends_with_offer(OFFER_REPLY) and conversation._deeper_offer["title"] == "Fallout 4 guide"


def test_a_typed_question_with_no_grounding_is_not_written(fallout):
    fake = FakeClient("[neutral] Boil it for three minutes.")
    with patch.object(core_llm, "_client", fake):
        list(m_turn.ask_events("how long should I boil an egg for breakfast today?", origin="typed"))
    assert fake.calls[0]["max_tokens"] == 400 and "<recall>" not in last_user(fake)


@pytest.mark.parametrize("text, expected", [
    ("how do I get the key?", True), ("explain the SAFE test", True), ("Vivec?", False),
    ("I am at Covenant and Dan keeps attacking me now", True), ("hi", False), ("yes", False),
])
def test_content_question(text, expected):
    assert m_written.content_question(text) is expected


@pytest.mark.parametrize("text, expected", [
    ("write it up", True), ("Write that up, please", True), ("yes, write it up", True), ("write it out", True),
    ("give me the written version", True), ("/write", True), ("write a poem", False), ("yes please", False),
])
def test_wants_writeup(text, expected):
    assert commands.wants_writeup(text) is expected


# --------------------------------------------------------------- 2. the card


def test_write_it_up_after_a_grounded_answer_makes_a_card(fallout):
    grounded_turn()
    sent = write_up_turn(spoken=True)
    speech = [m for m in sent if m["type"] == "speech"]
    assert [m["text"] for m in speech] == ["Written out below."]
    cards = written(sent)
    assert len(cards) == 1
    card = cards[0]
    assert card["id"] == "card1" and card["title"] == "The Compound Key at Covenant"
    assert card["status"] == "done" and card["workflow"] == "write-up"
    assert len(re.findall(r"^## ", card["markdown"], re.M)) >= 2
    sources_line = card["markdown"].splitlines()[-1]
    assert sources_line.startswith("Sources: ") and re.search(r"Fallout 4 guide · part \d+/\d+", sources_line)
    assert card["sources"] and all(s.startswith("Fallout 4 guide · part ") for s in card["sources"])
    # the transcript shows the spoken line; the card goes out running, its body comes as data after the
    # speech (PRD 51), and the turn ends with context + idle
    types = [m["type"] for m in sent]
    assert types.index("card") < types.index("speech") < types.index("speech_end") < types.index("card_update")
    assert types[-2:] == ["context", "state"]
    # bookkeeping: the card is an assistant turn for the model, a one-liner for memory
    assert conversation._session_turns[-2:] == [{"role": "user", "content": "write it up"},
                                       {"role": "assistant", "content": "wrote up: The Compound Key at Covenant"}]
    assert conversation._conversation[-1]["role"] == "assistant" and CARD_MD.strip() in conversation._conversation[-1]["content"]
    assert m_written.card_in(conversation._conversation[-1]["content"]) is conversation._cards[card["id"]] is conversation.last_card


def test_the_card_completion_reuses_the_last_answers_context(fallout):
    fake = grounded_turn()
    recall = last_user(fake)
    fake2 = FakeClient(*[card_response()])
    with patch.object(core_llm, "_client", fake2):
        m_written.write_up("/write")
    call = fake2.calls[0]
    assert call["max_tokens"] == m_written.CARD_MAX_TOKENS and not call.get("stream")
    assert call["messages"][0]["content"] == m_written.CARD_PROMPT
    user = call["messages"][-1]["content"]
    assert user.startswith(f"Question: {QUESTION}") and "Material:" in user
    assert re.search(r"<recall>.*</recall>", recall, re.S).group(0)[:200] in user   # the same block again


def test_typed_slash_write_and_the_offer_reply_take_the_write_up_path(fallout):
    grounded_turn()
    sent = turn("/write", [card_response()])
    assert [m["type"] for m in sent if m["type"] in ("card", "speech", "card_update")] == ["card", "speech", "card_update"]
    grounded_turn()
    sent = turn("yes, write it up", [card_response()])
    assert [c["status"] for c in written(sent)] == ["done"]
    assert conversation._deeper_offer is None


def test_write_it_up_before_any_answer_is_a_turn_error():
    sent = turn("write it up", [])
    assert [m for m in sent if m["type"] == "error"] == [{"type": "error", "scope": "turn", "text": "nothing to write up yet"}]
    assert sent[-1] == {"type": "state", "value": "idle"} and conversation._session_turns == []


def test_an_empty_card_is_refused_and_leaves_no_trace(fallout):
    grounded_turn()
    before = list(conversation._conversation), list(conversation._session_turns)
    with patch.object(core_llm, "_client", FakeClient(*[card_response("")])):
        with pytest.raises(ValueError):
            m_written.write_up("write it up")
    assert (conversation._conversation, conversation._session_turns) == before and conversation.last_card is None


# --------------------------------------------------------------- 3. the client (fixture)


def test_card_renders_collapsed_copies_and_saves(r):
    assert r["card_rendered"]["role"] == "Wanko · written"
    assert r["card_rendered"]["cls"] == "card-body collapsed" and r["card_rendered"]["more"] == "show all"
    assert r["card_rendered"]["text"].startswith("# Bleak Falls Barrow") and "Sources:" in r["card_rendered"]["text"]
    assert r["card_rendered"]["decodes"] == 0                      # nothing about the card reaches the audio queue
    assert r["card_rendered"]["history_role"] == "msg card"
    assert r["card_expanded"] == {"cls": "card-body", "more": "show less"}
    assert r["card_copied"] == [r["card_rendered"]["text"]]
    assert r["card_picker_before"] is True
    assert r["card_picker"] == {"hidden": False, "folder": "notes/", "sent": []}
    assert r["card_save_sent"] == [{"action": "save_card", "id": "card1", "folder": "notes"}]
    assert r["card_saving"] == {"label": "saving…", "disabled": True}
    assert r["card_save_failed"] == {"label": "Save to wiki", "disabled": False, "note": "already saved"}
    assert r["card_save_folder"] == {"action": "save_card", "id": "card1", "folder": "drafts"}
    assert r["card_saved"] == {"label": "saved → notes/2026-08-30-bleak-falls-barrow", "disabled": True,
                               "picker_hidden": True}
    assert r["card_replayed_saved"] == "saved → notes/2026-08-30-two"


# --------------------------------------------------------------- 4. save_card


@pytest.fixture
def card(vault, fallout):
    """A written card in the session (a grounded turn + write-up), ready to save."""
    grounded_turn()
    with patch.object(core_llm, "_client", FakeClient(*[card_response()])):
        return m_written.write_up("write it up")


def test_save_card_files_a_note_with_provenance_in_one_commit(card):
    before = len(commits())
    panels.greeted = True
    with TestClient(app.app).websocket_connect("/ws") as ws:
        collect(ws, "state")
        ws.send_json({"action": "save_card", "id": card["id"], "folder": "notes"})
        got = collect(ws, "sources")
    path = f"notes/{util._today_iso()}-the-compound-key-at-covenant"
    assert [m["type"] for m in got] == ["wiki_open", "card_saved", "wiki", "sources"]
    assert got[0] == {"type": "wiki_open", "path": path}
    assert got[1] == {"type": "card_saved", "id": card["id"], "path": path}
    text = read(path + ".md")
    fm, body = store.parse_page(text)
    assert fm["source"] == "written answer" and fm["type"] == "note" and fm["name"] == card["title"]
    assert fm["sources"] == card["sources"] and fm["sources"][0].startswith("Fallout 4 guide · part ")
    assert body.startswith("# The Compound Key at Covenant") and "## Steps" in body and "Sources: " in body
    assert len(commits()) == before + 1 and "save card" in commits()[0]
    assert any(p["path"] == path for p in got[2]["pages"])
    assert card["saved"] == path and m_recall.focus == {"view": "wiki", "path": path}


def test_saving_twice_is_refused_and_no_card_is_nothing_to_save(card):
    with TestClient(app.app).websocket_connect("/ws") as ws:
        collect(ws, "state")
        ws.send_json({"action": "save_card", "id": card["id"]})
        collect(ws, "sources")
        ws.send_json({"action": "save_card", "id": card["id"]})
        assert collect(ws, "error")[-1] == {"type": "error", "scope": "wiki", "text": "already saved"}
        ws.send_json({"action": "save_card", "id": "card9"})
        assert collect(ws, "error")[-1] == {"type": "error", "scope": "wiki", "text": "nothing to save"}
    conversation.reset()
    with TestClient(app.app).websocket_connect("/ws") as ws:
        collect(ws, "state")
        ws.send_json({"action": "save_card"})
        assert collect(ws, "error")[-1] == {"type": "error", "scope": "wiki", "text": "nothing to save"}


def test_a_folder_override_and_a_bad_folder(card):
    with TestClient(app.app).websocket_connect("/ws") as ws:
        collect(ws, "state")
        ws.send_json({"action": "save_card", "id": card["id"], "folder": "raw"})
        err = collect(ws, "error")[-1]
        assert err["scope"] == "wiki" and "bad path" in err["text"]
        ws.send_json({"action": "save_card", "id": card["id"], "folder": "drafts/"})
        got = collect(ws, "sources")
    assert got[1]["path"].startswith("drafts/") and os.path.isfile(os.path.join(store.WIKI_PATH, got[1]["path"] + ".md"))


def test_wiki_save_card_frontmatter_lists_survive_a_reparse(vault):
    path, reason = pages.save_card("notes/x", "X", "# X\n\n## A\ntext\n\nSources: a · part 1/2; b", ["a · part 1/2", "b"])
    assert reason is None and path == "notes/x"
    fm, _ = store.parse_page(read("notes/x.md"))
    assert fm["sources"] == ["a · part 1/2", "b"] and fm["source"] == "written answer"
    assert pages.save_card("notes/x", "X", "# X", []) == (None, "notes/x already exists")


# --------------------------------------------------------------- 5. "save that as a note"


@pytest.mark.parametrize("text", ["save that as a note", "Save this to the wiki", "save it, please", "save that"])
def test_save_note_command(text):
    assert commands.save_note_command(text)


@pytest.mark.parametrize("text", ["save the file", "save that as a note about Anna", "let's save money"])
def test_not_a_save_note_command(text):
    assert not commands.save_note_command(text)


def test_spoken_save_that_as_a_note_files_the_last_card(card):
    sent = turn("save that as a note", [], spoken=True)
    types = [m["type"] for m in sent]
    path = f"notes/{util._today_iso()}-the-compound-key-at-covenant"
    assert {"type": "wiki_open", "path": path} in sent
    assert {"type": "card_saved", "id": card["id"], "path": path} in sent
    assert [m["text"] for m in sent if m["type"] == "speech"] == [f"Saved as {path}."]
    assert "error" not in types and sent[-1] == {"type": "state", "value": "idle"}
    assert os.path.isfile(os.path.join(store.WIKI_PATH, path + ".md"))


def test_save_that_before_any_card_is_a_wiki_error(vault):
    sent = turn("save that as a note", [], spoken=True)
    assert [m for m in sent if m["type"] == "error"] == [{"type": "error", "scope": "wiki", "text": "nothing to save"}]
    assert not [m for m in sent if m["type"] == "speech"] and sent[-1] == {"type": "state", "value": "idle"}


# --------------------------------------------------------------- 6. new topic + replay


def test_new_topic_clears_the_last_card(card):
    with patch.object(core_llm, "_client", FakeClient(*[card_response("Covenant and its key.")])):
        conversation.new_thread()
    assert conversation.last_card is None and conversation._cards == {}
    sent = turn("save that as a note", [])
    assert [m for m in sent if m["type"] == "error"] == [{"type": "error", "scope": "wiki", "text": "nothing to save"}]


def test_reconnect_replays_the_card_as_a_card(card):
    panels.greeted = True
    with TestClient(app.app).websocket_connect("/ws") as ws:
        got = collect(ws, "hands_free")
    replayed = [m for m in got if m["type"] in ("transcript", "card")]
    assert [m["type"] for m in replayed] == ["transcript", "transcript", "transcript", "card"]
    assert replayed[2] == {"type": "transcript", "role": "user", "text": "write it up"}
    assert replayed[3] == {**server_turns.card_msg(card), "saved": None}
    assert "<card" not in str([m for m in got if m["type"] == "transcript"])


# --------------------------------------------------------------- 8. protocol rows


def test_the_protocol_names_the_card_messages():
    doc = open(os.path.join(ROOT, "docs", "protocol.md")).read()
    assert all(f"`{name}`" in doc for name in ("card", "card_saved", "save_card"))
    assert "save_card" in m_ws.ACTIONS and panels._scope_of("save_card") == "wiki"


# --------------------------------------------------------------- PRD 46: the briefing card

BRIEF_DIGEST = {"since": "2026-10-10T12:00:00",
                "pages": [{"path": "people/anna", "title": "Anna", "change": "new"},
                          {"path": "projects/falcon", "title": "Falcon", "change": "edited"}],
                "chat_pages": ["people/ben"],
                "sources": [{"page": "sources/guide", "path": "raw/inbox/guide.md", "kind": "text",
                             "at": "2026-10-11T08:00:00"}],
                "memory": {"added": 2, "updated": 0, "removed": 1,
                           "examples": [{"op": "ADD", "path": "people[]", "value": {"name": "Wife"}},
                                        {"op": "DELETE", "path": "people[0]", "value": None}]}}


def test_write_it_up_after_a_briefing_is_the_digest_card_without_a_completion(vault):
    fake = FakeClient("[happy] Anna is new and Falcon changed.")
    with patch.object(core_llm, "_client", fake):
        reply, card_md = m_written.briefing(BRIEF_DIGEST, "what's new?")
    assert len(fake.calls) == 1 and reply.endswith(prompt.DEEPER_OFFERS[0])
    sent = turn("write it up", [])                              # no response queued: a completion would fail
    cards = written(sent)
    assert len(cards) == 1 and cards[0]["markdown"] == card_md
    md = cards[0]["markdown"]
    assert md.startswith("# Briefing ") and cards[0]["title"] == md.splitlines()[0][2:]
    for section in ("## Pages", "## Sources", "## Memory"):
        assert section in md
    assert "[[people/anna|Anna]]" in md and "[[projects/falcon|Falcon]]" in md and "[[sources/guide]]" in md
    path = server_turns.save_card(None)[0]["path"]
    fm, body = store.parse_page(read(path + ".md"))
    assert path.startswith("notes/") and fm["source"] == "written answer" and "## Memory" in body
