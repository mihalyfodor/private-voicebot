"""PRD 24 scenarios: typed turns, the retired `ptt` action, and the client-side layout/mute.

The browser half (scenarios 2-4) runs static/app.js in node against a tiny DOM shim —
see tests/app_fixture.mjs; this file only asserts on the JSON it prints.
"""
import os
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import chatbot
import llm
from session import Session
from tests.conftest import StreamClient, drain_initial, fake_asr, fake_tts, run_turn

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


# ---------------------------------------------------------------- scenario 1: typed == spoken


def test_typed_turn_produces_the_same_events_as_a_spoken_one(monkeypatch):
    """Only the ASR half and the echoed user transcript differ; the reply path is identical."""
    origins = []

    def fake_ask_events(user_text, origin="user", note=""):
        origins.append((user_text, origin))
        yield ("delta", "[happy] Answer.")

    monkeypatch.setattr(llm, "ask_events", fake_ask_events)
    monkeypatch.setattr(chatbot, "synthesize", fake_tts)
    monkeypatch.setattr(chatbot, "asr", fake_asr("hello there"))

    spoken = run_turn(chatbot.handle_utterance, Session(), 1, np.zeros(10, dtype=np.float32))
    typed = run_turn(chatbot.handle_text, Session(), 1, "hello there")

    # the browser already rendered the typed line, so the server does not echo it back
    assert {"type": "transcript", "role": "user", "text": "hello there"} in spoken
    assert not any(m.get("role") == "user" for m in typed)
    # everything else — states, speech, speech_end, the assistant transcript — matches
    assert [m for m in spoken if m.get("role") != "user" and m.get("value") != "processing"] == typed
    assert origins == [("hello there", "user"), ("hello there", "typed")]


def test_text_turn_action_runs_a_turn_and_logs_it_as_typed(tmp_path, monkeypatch):
    """Scenario 1 end to end: `text_turn` → llm → session.log `typed:` line → memory turns."""
    llm.reset()
    monkeypatch.setattr(llm, "LOG_PATH", str(tmp_path / "session.log"))
    monkeypatch.setattr(chatbot, "synthesize", fake_tts)
    fake = StreamClient("[happy] Sure thing.")

    client = TestClient(chatbot.app)
    with patch.object(llm, "_client", fake), client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        ws.send_json({"action": "text_turn", "text": "  remind me to call Anna  "})
        seen = []
        while True:
            msg = ws.receive_json()
            seen.append(msg)
            if msg["type"] == "speech_end":
                ws.send_json({"action": "playback_done", "turn": msg["turn"]})
            if msg["type"] == "state" and msg["value"] == "idle":
                break

    assert [m["value"] for m in seen if m["type"] == "state"] == ["thinking", "speaking", "idle"]
    assert {"type": "transcript", "role": "assistant", "text": "Sure thing.", "turn": 1} in seen

    line = open(llm.LOG_PATH).read()
    assert "| typed: 'remind me to call Anna' |" in line          # stripped, tagged as typed
    # memory sees the typed turn exactly like a spoken one
    assert llm.get_session_turns()[0] == {"role": "user", "content": "remind me to call Anna"}


def test_blank_text_turn_starts_nothing():
    llm.reset()
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        ws.send_json({"action": "text_turn", "text": "   "})
        ws.send_json({"action": "set_backdrop", "key": "none"})   # round trip
        assert ws.receive_json()["type"] == "backdrop"            # nothing came before it
    assert chatbot.controller.busy is False


def test_log_origin_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "LOG_PATH", str(tmp_path / "session.log"))
    llm._log("hi", [], "reply", "typed")
    llm._log("hi", [], "reply")
    lines = open(llm.LOG_PATH).read().splitlines()
    assert "| typed: 'hi' | tools: none | reply_len: 5" in lines[0]
    assert "| user: 'hi' | tools: none | reply_len: 5" in lines[1]


# ---------------------------------------------------------------- scenario 4: ptt is gone


def test_ptt_action_and_button_are_gone():
    assert "ptt" not in chatbot.ACTIONS
    assert "text_turn" in chatbot.ACTIONS
    assert not hasattr(chatbot, "_action_ptt")
    assert not hasattr(Session(), "ptt_active")
    html = open(os.path.join(ROOT, "static", "index.html")).read()
    assert 'id="btn"' not in html
    for added in ('id="companion"', 'id="history"', 'id="composer-input"',
                  'id="mic-btn"', 'id="mute-btn"', 'id="backdrop-thumbs"', 'id="left-slot"'):
        assert added in html, added
    app_js = open(os.path.join(ROOT, "static", "app.js")).read()
    assert "'ptt'" not in app_js and '"ptt"' not in app_js


# ---------------------------------------------------------------- scenarios 2-4: the client


def test_setview_reparents_the_one_companion(r):
    """Scenario 3: the companion element moves between the chat slot and the shell slice."""
    assert (r["parent_chat"], r["class_chat"]) == ("chat-slot", "chat")
    assert (r["parent_memory"], r["class_memory"]) == ("left-slot", "side")
    assert (r["parent_back"], r["class_back"]) == ("chat-slot", "chat")


def test_typed_line_is_rendered_locally_with_a_keyboard_glyph(r):
    assert r["typed_sent"] == [{"action": "text_turn", "text": "hello there"}]
    assert r["after_send_value"] == ""
    assert r["history"][0] == {"role": "msg user", "text": "hello there", "typed": True}


def test_muted_client_shows_the_reply_but_plays_nothing(r):
    """Scenario 2: no decode/playback while muted, but playback_done still goes out."""
    assert r["muted_after_space"] is True
    assert r["decodes_while_muted"] == 0
    assert r["playback_done_while_muted"] == [{"action": "playback_done", "turn": 1}]
    assert r["history"][1] == {"role": "msg assistant", "text": "muted reply", "typed": False}
    assert r["muted_after_second_space"] is False
    assert r["decodes_after_unmute"] == 1
    assert r["muted_after_space_while_typing"] is False   # space belongs to the composer


def test_backdrop_thumbs_match_the_settings_chips(r):
    """Scenario 4: same payload, current key highlighted, 'none' is a plain chip."""
    assert r["thumbs"] == [
        {"cls": "bd-thumb none active"},
        {"cls": "bd-thumb", "bg": 'url("/static/backdrops/office.jpg")'},
    ]
    assert r["thumb_payload"] == [{"action": "set_backdrop", "key": "office"}]
