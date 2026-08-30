"""PRD 39 — context meter, recall stripping and "new topic". Scenario numbers are the PRD's
(8 is tests/test_protocol_doc.py; the client half of 2 and 5 reads the node fixture).

Turn helpers, the streaming fake and the seeded seeded_vault come from tests/conftest.py.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import chatbot
import llm
import memory
from session import Session
from tests.conftest import FakeClient, StreamClient, fake_asr, fake_tts, run_turn, typed

pytestmark = pytest.mark.usefixtures("fresh_llm")


def context_msgs(sent: list) -> list:
    return [m for m in sent if m.get("type") == "context"]


def spoken_text(sent: list) -> str:
    return " ".join(m["text"] for m in sent if m.get("type") == "speech")


@pytest.fixture
def recorded_memory(monkeypatch):
    """`memory.save` replaced by a recorder: the turns it was handed, applied ops []."""
    saved = []
    monkeypatch.setattr(memory, "save", lambda turns, client, model, sources=(): saved.append(list(turns)) or [])
    return saved


@pytest.fixture
def small_budget(monkeypatch):
    """A budget the tool schemas nearly fill, so a few short turns cross 90 %; no trimming."""
    monkeypatch.setattr(llm, "MAX_TOKENS", 1)
    llm._conversation[0]["content"] = "sys"
    budget = {"value": 1300}
    monkeypatch.setattr(llm, "context_budget", lambda: budget["value"])
    return budget


# --------------------------------------------------------------- 1. the context message


def test_a_completed_turn_sends_context_with_the_estimated_usage(seeded_vault):
    sent, _ = typed("hello there, how are you today")
    (msg,) = context_msgs(sent)
    used = llm.estimate_tokens(llm._conversation) + llm.TOOLS_TOKENS
    assert msg == {"type": "context", "used": used, "budget": llm.context_budget(),
                   "pct": round(100 * used / llm.context_budget()), "trimmed": 0}
    assert sent.index(msg) > sent.index({"type": "transcript", "role": "assistant", "text": "Sure."})
    from tools import TOOLS
    assert llm.TOOLS_TOKENS == len(json.dumps(TOOLS)) // 4          # the tool schemas count too


def test_trim_history_returns_the_user_turns_dropped(monkeypatch):
    monkeypatch.setattr(llm, "CONTEXT_BUDGET", 200)
    monkeypatch.setattr(llm, "MAX_TOKENS", 50)
    monkeypatch.setattr(llm, "model_context_len", lambda: None)
    llm._conversation[0]["content"] = "sys"
    for i in range(6):
        llm._conversation.append({"role": "user", "content": f"question {i} " * 10})
        llm._conversation.append({"role": "assistant", "content": f"answer {i} " * 10})
    llm._conversation.append({"role": "user", "content": "latest"})
    dropped = llm.trim_history()
    assert dropped > 0 and llm.context_usage()["trimmed"] == dropped
    assert dropped == 6 - sum(1 for m in llm._conversation if m["role"] == "user") + 1
    assert llm.trim_history() == 0 and llm.context_usage()["trimmed"] == 0


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
    assert llm.context_usage()["pct"] >= 90
    assert llm.CONTEXT_WARNING in spoken_text(sent)
    assert llm.CONTEXT_WARNING not in llm._conversation[-1]["content"]      # spoken, not remembered
    assert [m["text"] for m in sent if m.get("type") == "transcript"][-1].endswith(llm.CONTEXT_WARNING)

    sent, _ = typed("second", reply)
    assert llm.context_usage()["pct"] >= 90 and llm.CONTEXT_WARNING not in spoken_text(sent)

    small_budget["value"] = 2200                                  # back under 75 %
    sent, _ = typed("third", reply)
    assert llm.context_usage()["pct"] < 75 and llm.CONTEXT_WARNING not in spoken_text(sent)

    small_budget["value"] = 1650                                  # over 90 % again
    sent, _ = typed("fourth", reply)
    assert llm.context_usage()["pct"] >= 90 and llm.CONTEXT_WARNING in spoken_text(sent)


# --------------------------------------------------------------- 4. recall stripping


def test_recall_blocks_leave_the_history_once_the_turn_is_done(seeded_vault, monkeypatch):
    block = "\n\n<recall>\n" + "lorem ipsum " * 500 + "\n</recall>"             # ~1500 tokens
    monkeypatch.setattr(llm, "recall_for", lambda text: block)
    fake = StreamClient(*["[neutral] Per the guide, yes."] * 10)
    questions = [f"what does the guide say about chapter {i} of the story" for i in range(10)]
    with patch.object(llm, "_client", fake):
        for q in questions:
            llm.ask(q)
    assert all(q + block in [m for m in call["messages"] if m["role"] == "user"][-1]["content"]
               for q, call in zip(questions, fake.calls))                       # the model saw it
    users = [m["content"] for m in llm._conversation if m["role"] == "user"]
    assert users == questions                                                    # the history did not
    assert llm.estimate_tokens(llm._conversation) < 4000
    assert [t["content"] for t in llm.get_session_turns() if t["role"] == "user"] == questions


def test_focus_and_deeper_blocks_are_stripped_too(seeded_vault, monkeypatch):
    llm.set_focus("wiki", "people/anna")
    _, fake = typed("what does Anna do")
    assert "<document" in [m for m in fake.calls[-1]["messages"] if m["role"] == "user"][-1]["content"]
    assert llm._conversation[-2] == {"role": "user", "content": "what does Anna do"}


# --------------------------------------------------------------- 5. new topic


def three_turns():
    fake = StreamClient("[neutral] Falcon is amber.", "[neutral] Anna runs the deploy.", "[neutral] Friday then.")
    with patch.object(llm, "_client", fake):
        for q in ("how is Falcon doing", "who runs the deploy", "when is it"):
            llm.ask(q)


def test_new_thread_folds_first_then_restarts_on_a_short_summary(seeded_vault, recorded_memory):
    three_turns()
    llm.set_focus("wiki", "people/anna")
    llm._deeper_offer = {"source": "s", "title": "t", "parts": [1]}
    llm._turn_sources = [("s", "t", 1)]
    summary_client = FakeClient("[neutral] We checked Falcon (amber) and agreed Anna runs Friday's deploy.")
    with patch.object(llm, "_client", summary_client):
        summary = llm.new_thread()

    assert summary == "We checked Falcon (amber) and agreed Anna runs Friday's deploy."
    assert [t["content"] for t in recorded_memory[0]][::2] == ["how is Falcon doing", "who runs the deploy", "when is it"]
    assert len(summary_client.calls) == 1 and summary_client.calls[0]["max_tokens"] == llm.SUMMARY_TOKENS
    assert summary_client.calls[0]["messages"][-1]["content"] == llm.SUMMARY_PROMPT
    assert llm._conversation[1:] == [{"role": "user", "content": f"<earlier>{summary}</earlier>"},
                                     {"role": "assistant", "content": llm.NEW_TOPIC_ACK}]
    assert llm._conversation[0]["role"] == "system"
    assert memory.tokens_est(llm._conversation[1]["content"]) <= 150
    assert llm._deeper_offer is None and llm._turn_sources == [] and llm.last_page is None
    assert llm.get_session_turns() == []


def test_new_topic_action_sends_thread_then_context(seeded_vault, recorded_memory):
    assert "new_topic" in chatbot.ACTIONS
    three_turns()
    with patch.object(llm, "_client", FakeClient("[neutral] Falcon and the deploy.")), \
            patch.object(chatbot, "synthesize", fake_tts):
        sent = run_turn(chatbot.new_topic_turn, Session(), 1)
    kinds = [m["type"] for m in sent]
    assert kinds.index("thread") < kinds.index("speech") < kinds.index("context")
    assert {"type": "thread", "text": "Falcon and the deploy."} in sent
    assert sent[-1] == {"type": "state", "value": "idle"}
    assert context_msgs(sent)[0]["used"] == llm.estimate_tokens(llm._conversation) + llm.TOOLS_TOKENS
    assert [m for m in sent if m.get("type") == "transcript"] == [
        {"type": "transcript", "role": "assistant", "text": llm.NEW_TOPIC_ACK}]


def new_topic_turn_via(fn, *args) -> tuple:
    """A typed / spoken turn whose model call is the (non-streamed) summary: (sent, fake)."""
    fake = FakeClient("[neutral] Falcon and the deploy.")
    with patch.object(llm, "_client", fake), patch.object(chatbot, "synthesize", fake_tts):
        sent = run_turn(fn, Session(), 1, *args)
    return sent, fake


@pytest.mark.parametrize("text", ["new topic", "New topic.", "/new"])
def test_typed_new_topic_and_slash_new_take_the_new_topic_path(seeded_vault, recorded_memory, text):
    three_turns()
    sent, fake = new_topic_turn_via(chatbot.handle_text, text)
    assert {"type": "thread", "text": "Falcon and the deploy."} in sent
    assert len(fake.calls) == 1 and fake.calls[0]["messages"][-1]["content"] == llm.SUMMARY_PROMPT
    assert llm._conversation[-1]["content"] == llm.NEW_TOPIC_ACK
    assert text not in json.dumps(llm._conversation)              # the command itself is not a turn


def test_spoken_new_topic_takes_the_new_topic_path(seeded_vault, recorded_memory):
    three_turns()
    with patch.object(chatbot, "asr", fake_asr("new topic")):
        sent, fake = new_topic_turn_via(chatbot.handle_utterance, np.zeros(10, dtype=np.float32))
    assert {"type": "transcript", "role": "user", "text": "new topic"} in sent
    assert {"type": "thread", "text": "Falcon and the deploy."} in sent
    assert not chatbot.new_topic_command("a new topic please")


def test_the_palette_sends_new_topic_and_the_divider_shows_the_summary(r):
    assert r["pal_new_topic"] == [{"action": "new_topic"}]
    assert r["thread_divider"] == {"role": "msg divider", "text": "We planned the Falcon sync; open: the budget.",
                                   "typed": False}


# --------------------------------------------------------------- 6. summary failure


def test_summary_failure_falls_back_to_first_sentences(seeded_vault, recorded_memory, capsys):
    three_turns()
    with patch.object(llm, "_client", FakeClient(RuntimeError("model down"))):
        summary = llm.new_thread()
    assert summary == ("how is Falcon doing Falcon is amber. who runs the deploy Anna runs the deploy. "
                       "when is it Friday then.")
    assert llm._conversation[1:] == [{"role": "user", "content": f"<earlier>{summary}</earlier>"},
                                     {"role": "assistant", "content": llm.NEW_TOPIC_ACK}]
    out = capsys.readouterr().out
    assert out.count("summary failed") == 1 and "model down" in out
    assert len(recorded_memory) == 1


def test_new_thread_on_an_empty_conversation_makes_no_summary_call(seeded_vault, recorded_memory):
    client = FakeClient()
    with patch.object(llm, "_client", client):
        assert llm.new_thread() == llm.EARLIER_EMPTY
    assert client.calls == [] and recorded_memory == []


# --------------------------------------------------------------- 7. reload replay


def test_reconnect_replays_the_stripped_conversation_not_the_session_turns(seeded_vault, recorded_memory, monkeypatch):
    monkeypatch.setattr(llm, "recall_for", lambda text: "\n\n<recall>\nbig block\n</recall>")
    three_turns()
    llm.save_memory()                                             # the periodic fold empties the session turns
    assert llm.get_session_turns() == []
    with patch.object(llm, "_client", FakeClient("[neutral] Falcon and the deploy.")):
        llm.new_thread()
    with patch.object(llm, "_client", StreamClient("[neutral] Sure thing.")):
        llm.ask("one more question then")
    assert llm.get_session_turns() == [{"role": "user", "content": "one more question then"},
                                       {"role": "assistant", "content": "[neutral] Sure thing."}]

    monkeypatch.setattr(chatbot, "greeted", True)
    with TestClient(chatbot.app).websocket_connect("/ws") as ws:
        got = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "hands_free":
                break
            got.append(msg)
    replayed = [m for m in got if m["type"] in ("thread", "transcript")]
    assert replayed == [{"type": "thread", "text": "Falcon and the deploy."},
                        {"type": "transcript", "role": "assistant", "text": llm.NEW_TOPIC_ACK},
                        {"type": "transcript", "role": "user", "text": "one more question then"},
                        {"type": "transcript", "role": "assistant", "text": "Sure thing."}]
    assert chatbot.replay_messages() == replayed and "<recall>" not in json.dumps(got)
