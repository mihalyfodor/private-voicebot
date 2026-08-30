import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import threading
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

import chatbot
import llm
from session import Session
from tests.conftest import Blocker


def fake_tts(text):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(240, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


FAKE_AVATAR = {
    "key": "wanko",
    "name": "Wanko",
    "tagline": "Office assistant, upbeat",
    "voice": "am_puck",
    "speed": 1.05,
    "greeting": "[happy] Morning! Wanko here.",
    "persona": "You are Wanko.",
}


def test_shipped_greeting_is_the_current_card_greeting():
    """Scenario 3: the spoken greeting comes from wanko.yaml."""
    import avatars
    a = avatars.AVATARS["wanko"]
    assert a["greeting"] == "[happy] Morning! Wanko here. What are we working on today?"


def test_say_canned_emits_state_speech_speech_end_and_transcript():
    sent = []
    reply = chatbot.say_canned("[happy] Hi there.", sent.append, tts=fake_tts)

    assert reply == "Hi there."
    types = [m["type"] for m in sent]
    assert types == ["state", "speech", "speech_end", "transcript"]
    assert sent[0]["value"] == "speaking"
    assert sent[1]["emotion"] == "happy"
    assert sent[1]["text"] == "Hi there."
    assert sent[3] == {"type": "transcript", "role": "assistant", "text": "Hi there."}


def test_greet_does_not_call_llm_and_records_greeting(monkeypatch):
    llm.reset()

    class BoomClient:
        def __getattr__(self, name):
            raise AssertionError("LLM should not be called during greet()")

    monkeypatch.setattr(chatbot, "AVATAR", FAKE_AVATAR)
    monkeypatch.setattr(chatbot, "synthesize", fake_tts)
    monkeypatch.setattr(llm, "_client", BoomClient())

    sent = []
    monkeypatch.setattr(chatbot, "make_sender", lambda loop, turn=None: sent.append)

    sess = Session(websocket=None)  # no socket → no playback wait
    chatbot.greet(sess, sess.next_turn(), SimpleNamespace())

    assert chatbot.is_busy() is False
    assert llm._conversation[-1] == {"role": "assistant", "content": "Morning! Wanko here."}
    assert llm._session_turns[-1] == {"role": "assistant", "content": "Morning! Wanko here."}
    assert {"type": "transcript", "role": "assistant", "text": "Morning! Wanko here."} in sent
    assert {"type": "state", "value": "idle"} in sent


def test_greet_messages_carry_the_turn_id(monkeypatch):
    """Every state/speech/speech_end/assistant-transcript message is tagged with its turn."""
    llm.reset()
    monkeypatch.setattr(chatbot, "AVATAR", FAKE_AVATAR)
    monkeypatch.setattr(chatbot, "synthesize", fake_tts)

    sent = []

    async def fake_send(msg):
        sent.append(msg)

    monkeypatch.setattr(chatbot, "send", fake_send)
    import asyncio
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        sess = Session(websocket=None)
        chatbot.greet(sess, sess.next_turn(), loop)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)

    types = [m["type"] for m in sent]
    assert types[:2] == ["state", "state"]  # thinking, then speaking
    assert types[-3:] == ["speech_end", "transcript", "state"]
    assert all(m["turn"] == 1 for m in sent)


def test_reload_characters_ws_action(monkeypatch):
    import avatars
    llm.reset()

    fake_profile = dict(FAKE_AVATAR)

    monkeypatch.setattr(avatars, "reload", lambda: fake_profile)
    monkeypatch.setattr(avatars, "set_current", lambda key: fake_profile)
    monkeypatch.setattr(llm, "set_avatar", lambda: None)
    monkeypatch.setattr(chatbot, "_filler_wavs", {})

    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # hands_free
        ws.receive_json()  # initial idle state
        ws.send_json({"action": "reload_characters"})
        msg = ws.receive_json()
        assert msg == {"type": "characters_reloaded", "avatar": "wanko", "name": "Wanko"}


def test_reconnect_replays_transcript():
    llm.reset()
    llm._conversation += [{"role": "assistant", "content": "[happy] Hi there."},
                          {"role": "user", "content": "hello"}]                 # what the model holds (PRD 39)
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "transcript", "role": "assistant", "text": "Hi there."}
        assert ws.receive_json() == {"type": "transcript", "role": "user", "text": "hello"}
        assert ws.receive_json()["type"] == "hands_free"
        assert ws.receive_json()["type"] == "state"


def test_new_connection_releases_the_previous_playback_wait():
    llm.reset()
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws1:
        ws1.receive_json()
        first = chatbot.session
        first.playback_done.clear()
        with client.websocket_connect("/ws") as ws2:
            ws2.receive_json()
            assert chatbot.session is not first
            assert first.playback_done.is_set()
            assert first.websocket is None
