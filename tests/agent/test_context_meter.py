"""PRD 39 — context meter, recall stripping and "new topic". Scenario numbers are the PRD's
(8 is tests/test_protocol_doc.py; the client half of 2 and 5 reads the node fixture).

Turn helpers, the streaming fake and the seeded seeded_vault come from tests/conftest.py.
"""

import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from tools import registry

from server import app
from server import turns as server_turns
from server import ws as m_ws
from server import turns as server_turns
from server import panels
from server import speech as s_speech
from agent import context as a_context
from agent import conversation
from agent import recall as a_recall
from agent import turn
from agent import conversation
from agent import context as a_context
from agent import prompt
from agent import recall as a_recall
from core import util
from agent import fold as m_fold
from server.session import Session
from tests.conftest import FakeClient, fake_asr, fake_tts, run_turn, typed
from core import llm as core_llm

pytestmark = pytest.mark.usefixtures("fresh_llm")


def context_msgs(sent: list) -> list:
    return [m for m in sent if m.get("type") == "context"]


def spoken_text(sent: list) -> str:
    return " ".join(m["text"] for m in sent if m.get("type") == "speech")


@pytest.fixture
def recorded_memory(monkeypatch):
    """`memory.save` replaced by a recorder: the turns it was handed, applied ops []."""
    saved = []
    monkeypatch.setattr(m_fold, "save", lambda turns, sources=(): saved.append(list(turns)) or [])
    return saved


@pytest.fixture
def small_budget(monkeypatch):
    """A budget the tool schemas nearly fill, so a few short turns cross 90 %; no trimming."""
    monkeypatch.setattr(prompt, "MAX_TOKENS", 1)
    conversation._conversation[0]["content"] = "sys"
    budget = {"value": 1300}
    monkeypatch.setattr(a_context, "context_budget", lambda: budget["value"])
    return budget


# --------------------------------------------------------------- 1. the context message


def test_a_completed_turn_sends_context_with_the_estimated_usage(seeded_vault):
    sent, _ = typed("hello there, how are you today")
    (msg,) = context_msgs(sent)
    used = a_context.estimate_tokens(conversation._conversation) + registry.tokens()
    assert msg == {"type": "context", "used": used, "budget": a_context.context_budget(),
                   "pct": round(100 * used / a_context.context_budget()), "trimmed": 0}
    assert sent.index(msg) > sent.index({"type": "transcript", "role": "assistant", "text": "Sure."})
    assert registry.tokens() == len(json.dumps(registry.schemas())) // 4          # the tool schemas count too


def test_trim_history_returns_the_user_turns_dropped(monkeypatch):
    monkeypatch.setattr(a_context, "CONTEXT_BUDGET", 200)
    monkeypatch.setattr(prompt, "MAX_TOKENS", 50)
    monkeypatch.setattr(a_context, "model_context_len", lambda: None)
    conversation._conversation[0]["content"] = "sys"
    for i in range(6):
        conversation._conversation.append({"role": "user", "content": f"question {i} " * 10})
        conversation._conversation.append({"role": "assistant", "content": f"answer {i} " * 10})
    conversation._conversation.append({"role": "user", "content": "latest"})
    dropped = a_context.trim_history()
    assert dropped > 0 and a_context.context_usage()["trimmed"] == dropped
    assert dropped == 6 - sum(1 for m in conversation._conversation if m["role"] == "user") + 1
    assert a_context.trim_history() == 0 and a_context.context_usage()["trimmed"] == 0


# --------------------------------------------------------------- 2. the client half (node fixture)


def test_status_suffix_rail_dot_and_forgot_turns(r):
    assert r["ctx_62"]["status"].endswith("ctx 62 %") and r["ctx_62"]["dot"] == "rail-dot"
    assert r["ctx_75"]["dot"] == "rail-dot ctx-amber"
    assert r["ctx_90"]["dot"] == "rail-dot ctx-red"
    assert r["ctx_90"]["status"].endswith("ctx 90 % · forgot 3 old turns")
    assert r["ctx_19"]["dot"] == "rail-dot" and "forgot" not in r["ctx_19"]["status"]


# --------------------------------------------------------------- 3. the 90 % crossing


def test_crossing_90_speaks_the_sentence_once_and_rearms_under_75(seeded_vault, small_budget):
    reply = "[neutral] " + "word " * 80                          # ~100 tokens a turn
    sent, _ = typed("first", reply)
    assert a_context.context_usage()["pct"] >= 90
    assert a_context.CONTEXT_WARNING in spoken_text(sent)
    assert a_context.CONTEXT_WARNING not in conversation._conversation[-1]["content"]      # spoken, not remembered
    assert [m["text"] for m in sent if m.get("type") == "transcript"][-1].endswith(a_context.CONTEXT_WARNING)

    sent, _ = typed("second", reply)
    assert a_context.context_usage()["pct"] >= 90 and a_context.CONTEXT_WARNING not in spoken_text(sent)

    small_budget["value"] = 2200                                  # back under 75 %
    sent, _ = typed("third", reply)
    assert a_context.context_usage()["pct"] < 75 and a_context.CONTEXT_WARNING not in spoken_text(sent)

    small_budget["value"] = 1650                                  # over 90 % again
    sent, _ = typed("fourth", reply)
    assert a_context.context_usage()["pct"] >= 90 and a_context.CONTEXT_WARNING in spoken_text(sent)


# --------------------------------------------------------------- 4. recall stripping


def test_recall_blocks_leave_the_history_once_the_turn_is_done(seeded_vault, monkeypatch):
    block = "\n\n<recall>\n" + "lorem ipsum " * 500 + "\n</recall>"             # ~1500 tokens
    monkeypatch.setattr(a_recall, "recall_for", lambda text: block)
    fake = FakeClient(*["[neutral] Per the guide, yes."] * 10)
    questions = [f"what does the guide say about chapter {i} of the story" for i in range(10)]
    with patch.object(core_llm, "_client", fake):
        for q in questions:
            list(turn.ask_events(q, origin="typed"))                             # typed: no recall band
    assert all(q + block in [m for m in call["messages"] if m["role"] == "user"][-1]["content"]
               for q, call in zip(questions, fake.calls))                       # the model saw it
    users = [m["content"] for m in conversation._conversation if m["role"] == "user"]
    assert users == questions                                                    # the history did not
    assert a_context.estimate_tokens(conversation._conversation) < 4000
    assert [t["content"] for t in conversation.get_session_turns() if t["role"] == "user"] == questions


def test_focus_and_deeper_blocks_are_stripped_too(seeded_vault, monkeypatch):
    a_recall.set_focus("wiki", "people/anna")
    _, fake = typed("what does Anna do")
    assert "<document" in [m for m in fake.calls[-1]["messages"] if m["role"] == "user"][-1]["content"]
    assert conversation._conversation[-2] == {"role": "user", "content": "what does Anna do"}


# --------------------------------------------------------------- 5. new topic


def three_turns():
    fake = FakeClient("[neutral] Falcon is amber.", "[neutral] Anna runs the deploy.", "[neutral] Friday then.")
    with patch.object(core_llm, "_client", fake):
        for q in ("how is Falcon doing", "who runs the deploy", "when is it"):
            turn.ask(q)


def test_new_thread_restarts_on_a_short_summary_and_leaves_the_fold_to_the_caller(seeded_vault, recorded_memory):
    three_turns()
    a_recall.set_focus("wiki", "people/anna")
    conversation._deeper_offer = {"source": "s", "title": "t", "parts": [1]}
    conversation._turn_sources = [("s", "t", 1)]
    summary_client = FakeClient("[neutral] We checked Falcon (amber) and agreed Anna runs Friday's deploy.")
    with patch.object(core_llm, "_client", summary_client):
        summary = conversation.new_thread()

    assert summary == "We checked Falcon (amber) and agreed Anna runs Friday's deploy."
    assert recorded_memory == []                                           # PRD 56: no fold inside new_thread
    assert len(summary_client.calls) == 1 and summary_client.calls[0]["max_tokens"] == conversation.SUMMARY_TOKENS
    assert summary_client.calls[0]["messages"][-1]["content"] == conversation.SUMMARY_PROMPT
    assert conversation._conversation[1:] == [{"role": "user", "content": f"<earlier>{summary}</earlier>"},
                                     {"role": "assistant", "content": conversation.NEW_TOPIC_ACK}]
    assert conversation._conversation[0]["role"] == "system"
    assert util.tokens_est(conversation._conversation[1]["content"]) <= 150
    assert conversation._deeper_offer is None and conversation._turn_sources == [] and a_recall.last_page is None
    assert [t["content"] for t in conversation.get_session_turns()][::2] == [
        "how is Falcon doing", "who runs the deploy", "when is it"]       # still queued for the caller's fold


def test_new_topic_action_sends_thread_then_context(seeded_vault, recorded_memory):
    assert "new_topic" in m_ws.ACTIONS
    three_turns()
    with patch.object(core_llm, "_client", FakeClient("[neutral] Falcon and the deploy.")), \
            patch.object(s_speech, "synthesize", fake_tts):
        sent = run_turn(server_turns.new_topic_turn, Session(), 1)
    kinds = [m["type"] for m in sent]
    assert kinds.index("thread") < kinds.index("speech") < kinds.index("context")
    assert {"type": "thread", "text": "Falcon and the deploy."} in sent
    assert sent[-1] == {"type": "state", "value": "idle"}
    assert context_msgs(sent)[0]["used"] == a_context.estimate_tokens(conversation._conversation) + registry.tokens()
    assert [m for m in sent if m.get("type") == "transcript"] == [
        {"type": "transcript", "role": "assistant", "text": conversation.NEW_TOPIC_ACK}]


def new_topic_turn_via(fn, *args) -> tuple:
    """A typed / spoken turn whose model call is the (non-streamed) summary: (sent, fake)."""
    fake = FakeClient("[neutral] Falcon and the deploy.")
    with patch.object(core_llm, "_client", fake), patch.object(s_speech, "synthesize", fake_tts):
        sent = run_turn(fn, Session(), 1, *args)
    return sent, fake


@pytest.mark.parametrize("text", ["new topic", "New topic.", "/new"])
def test_typed_new_topic_and_slash_new_take_the_new_topic_path(seeded_vault, recorded_memory, text):
    three_turns()
    sent, fake = new_topic_turn_via(server_turns.handle_text, text)
    assert {"type": "thread", "text": "Falcon and the deploy."} in sent
    assert len(fake.calls) == 1 and fake.calls[0]["messages"][-1]["content"] == conversation.SUMMARY_PROMPT
    assert conversation._conversation[-1]["content"] == conversation.NEW_TOPIC_ACK
    assert text not in json.dumps(conversation._conversation)              # the command itself is not a turn


def test_spoken_new_topic_takes_the_new_topic_path(seeded_vault, recorded_memory):
    three_turns()
    with patch.object(server_turns, "asr", fake_asr("new topic")):
        sent, fake = new_topic_turn_via(server_turns.handle_utterance, np.zeros(10, dtype=np.float32))
    assert {"type": "transcript", "role": "user", "text": "new topic"} in sent
    assert {"type": "thread", "text": "Falcon and the deploy."} in sent
    assert registry.match("a new topic please") is None


def test_the_palette_sends_new_topic_and_the_divider_shows_the_summary(r):
    assert r["pal_new_topic"] == [{"action": "new_topic"}]
    assert r["thread_divider"] == {"role": "msg divider", "text": "We planned the Falcon sync; open: the budget.",
                                   "typed": False}


# --------------------------------------------------------------- 6. summary failure


def test_summary_failure_falls_back_to_first_sentences(seeded_vault, recorded_memory, capsys):
    three_turns()
    with patch.object(core_llm, "_client", FakeClient(RuntimeError("model down"))):
        summary = conversation.new_thread()
    assert summary == ("how is Falcon doing Falcon is amber. who runs the deploy Anna runs the deploy. "
                       "when is it Friday then.")
    assert conversation._conversation[1:] == [{"role": "user", "content": f"<earlier>{summary}</earlier>"},
                                     {"role": "assistant", "content": conversation.NEW_TOPIC_ACK}]
    out = capsys.readouterr().out
    assert out.count("summary failed") == 1 and "model down" in out
    assert recorded_memory == []


def test_new_thread_on_an_empty_conversation_makes_no_summary_call(seeded_vault, recorded_memory):
    client = FakeClient()
    with patch.object(core_llm, "_client", client):
        assert conversation.new_thread() == conversation.EARLIER_EMPTY
    assert client.calls == [] and recorded_memory == []


# --------------------------------------------------------------- PRD 56: avatar note, new topic answers first


def test_avatar_note_survives_refresh_and_clears_on_new_thread(seeded_vault, recorded_memory):
    three_turns()
    conversation.set_avatar()
    assert "from now on you are" in conversation._conversation[0]["content"]
    conversation.refresh_context()
    assert "from now on you are" in conversation._conversation[0]["content"]
    with patch.object(core_llm, "_client", FakeClient("[neutral] Summary.")):
        conversation.new_thread()
    assert "from now on you are" not in conversation._conversation[0]["content"]
    conversation.reset()
    conversation.set_avatar()                                              # a one-message conversation: no note
    assert "from now on you are" not in conversation._conversation[0]["content"]
    three_turns()
    conversation.set_avatar()
    assert "from now on you are" in conversation._conversation[0]["content"]
    conversation.reset()
    assert "from now on you are" not in conversation._conversation[0]["content"]
    conversation.refresh_context()
    assert "from now on you are" not in conversation._conversation[0]["content"]


@pytest.fixture
def slow_fold(monkeypatch):
    """`fold.save` parked until released; `calls` holds the turns of every call, `order` the milestones."""
    started, release, calls, order = threading.Event(), threading.Event(), [], []

    def save(turns, sources=()):
        order.append("fold")
        calls.append(list(turns))
        started.set()
        release.wait(timeout=5)
        return []

    monkeypatch.setattr(m_fold, "save", save)
    yield SimpleNamespace(started=started, release=release, calls=calls, order=order)
    release.set()
    panels.controller.join_idle(timeout=5)


def test_new_thread_returns_before_the_fold_and_the_handler_speaks_first(seeded_vault, slow_fold):
    three_turns()
    with patch.object(core_llm, "_client", FakeClient("[neutral] Falcon and the deploy.")):
        conversation.new_thread()
    assert slow_fold.calls == [] and not panels.controller.hook_running   # new_thread alone never folds

    def speak(text):
        slow_fold.order.append("speech")
        return fake_tts(text)

    with patch.object(core_llm, "_client", FakeClient("[neutral] Falcon and the deploy.")), \
            patch.object(s_speech, "synthesize", speak):
        sent = run_turn(server_turns.new_topic_turn, Session(), 1)
    assert slow_fold.started.wait(timeout=5) and panels.controller.hook_running
    assert slow_fold.order == ["speech", "fold"]                           # the ack is spoken, then the fold starts
    assert sent[-1] == {"type": "state", "value": "idle"}                  # the turn is over while the fold still runs
    assert len(slow_fold.calls) == 1


def test_a_failing_fold_after_new_thread_keeps_the_turns_and_a_second_topic_starts_no_second_fold(
        seeded_vault, monkeypatch, slow_fold):
    three_turns()
    six = list(conversation.get_session_turns())
    with patch.object(core_llm, "_client", FakeClient("[neutral] First.", "[neutral] Second.")), \
            patch.object(s_speech, "synthesize", fake_tts):
        run_turn(server_turns.new_topic_turn, Session(), 1)
        assert slow_fold.started.wait(timeout=5)
        run_turn(server_turns.new_topic_turn, Session(), 2)                # the fold of the first is still running
    assert len(slow_fold.calls) == 1                                       # the overlap guard skipped the second
    assert "<earlier>" in conversation._conversation[1]["content"]
    slow_fold.release.set()
    assert panels.controller.join_idle(timeout=5)

    # a failing fold: the snapshot goes back in front, the conversation stays on the new thread
    def boom(turns, sources=()):
        raise RuntimeError("model down")

    monkeypatch.setattr(m_fold, "save", boom)
    conversation._session_turns[:] = list(six)
    with patch.object(core_llm, "_client", FakeClient("[neutral] Third.")), patch.object(s_speech, "synthesize", fake_tts):
        run_turn(server_turns.new_topic_turn, Session(), 3)
    assert panels.controller.join_idle(timeout=5)
    assert conversation.get_session_turns() == six
    assert conversation._conversation[1]["content"].startswith("<earlier>")
    assert conversation._conversation[-1]["content"] == conversation.NEW_TOPIC_ACK


# --------------------------------------------------------------- 7. reload replay


def test_reconnect_replays_the_stripped_conversation_not_the_session_turns(seeded_vault, recorded_memory, monkeypatch):
    three_turns()
    conversation.save_memory()                                             # the periodic fold empties the session turns
    assert conversation.get_session_turns() == []
    with patch.object(core_llm, "_client", FakeClient("[neutral] Falcon and the deploy.")):
        conversation.new_thread()
    with patch.object(core_llm, "_client", FakeClient("[neutral] Sure thing.")):
        turn.ask("one more question then")
    assert conversation.get_session_turns() == [{"role": "user", "content": "one more question then"},
                                       {"role": "assistant", "content": "[neutral] Sure thing."}]

    monkeypatch.setattr(panels, "greeted", True)
    with TestClient(app.app).websocket_connect("/ws") as ws:
        got = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "hands_free":
                break
            got.append(msg)
    replayed = [m for m in got if m["type"] in ("thread", "transcript")]
    assert replayed == [{"type": "thread", "text": "Falcon and the deploy."},
                        {"type": "transcript", "role": "assistant", "text": conversation.NEW_TOPIC_ACK},
                        {"type": "transcript", "role": "user", "text": "one more question then"},
                        {"type": "transcript", "role": "assistant", "text": "Sure thing."}]
    assert m_ws.replay_messages() == replayed and "<recall>" not in json.dumps(got)
