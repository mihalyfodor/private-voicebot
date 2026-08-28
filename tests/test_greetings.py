import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import threading
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from unittest.mock import patch

import chatbot
import llm


def fake_tts(text):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(240, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


FAKE_AVATAR = {
    "key": "natori",
    "name": "Natori",
    "tagline": "Office assistant, easygoing",
    "voice": "am_michael",
    "speed": 1.0,
    "greeting": "[happy] Morning! Natori here.",
    "switch_greeting": "[happy] Natori taking over now.",
    "persona": "You are Natori.",
}


def test_say_canned_emits_state_speech_speech_end_and_transcript():
    sent = []
    chatbot.playback_done.clear()
    reply = chatbot.say_canned("[happy] Hi boss.", sent.append, tts=fake_tts)

    assert reply == "Hi boss."
    types = [m["type"] for m in sent]
    assert types == ["state", "speech", "speech_end", "transcript"]
    assert sent[0]["value"] == "speaking"
    assert sent[1]["emotion"] == "happy"
    assert sent[1]["text"] == "Hi boss."
    assert sent[3] == {"type": "transcript", "role": "assistant", "text": "Hi boss."}


def test_greet_does_not_call_llm_and_records_greeting(monkeypatch):
    llm.reset()

    class BoomClient:
        def __getattr__(self, name):
            raise AssertionError("LLM should not be called during greet()")

    monkeypatch.setattr(chatbot, "AVATAR", FAKE_AVATAR)
    monkeypatch.setattr(chatbot, "synthesize", fake_tts)
    monkeypatch.setattr(llm, "_client", BoomClient())
    monkeypatch.setattr(chatbot, "_wait_for_playback_then_idle", lambda send_sync: send_sync({"type": "state", "value": "idle"}))
    monkeypatch.setattr(chatbot, "ws_client", None)

    sent = []
    loop = SimpleNamespace()
    monkeypatch.setattr(chatbot, "make_sender", lambda loop: sent.append)

    chatbot.greet(loop)

    assert chatbot.processing is False
    assert llm._conversation[-1] == {"role": "assistant", "content": "Morning! Natori here."}
    assert llm._session_turns[-1] == {"role": "assistant", "content": "Morning! Natori here."}
    assert {"type": "transcript", "role": "assistant", "text": "Morning! Natori here."} in sent


def test_set_avatar_ws_speaks_switch_greeting(monkeypatch):
    from fastapi.testclient import TestClient
    import avatars

    llm.reset()
    chatbot.greeted = True
    chatbot.processing = False

    fake_profile = dict(FAKE_AVATAR)

    monkeypatch.setattr(avatars, "set_current", lambda key: fake_profile)
    monkeypatch.setattr(llm, "set_avatar", lambda: None)
    monkeypatch.setattr(chatbot, "synthesize", fake_tts)
    monkeypatch.setattr(chatbot, "_filler_wavs", {})
    monkeypatch.setattr(chatbot, "_wait_for_playback_then_idle", lambda send_sync: send_sync({"type": "state", "value": "idle"}))

    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # hands_free
        ws.receive_json()  # initial idle state (greeted already True)
        ws.send_json({"action": "set_avatar", "key": "natori"})

        msg = ws.receive_json()
        assert msg == {"type": "avatar", "key": "natori", "name": "Natori"}

        # Drain messages until we see the switch_greeting speech chunk.
        speech_msg = None
        for _ in range(20):
            m = ws.receive_json()
            if m.get("type") == "speech":
                speech_msg = m
                break
        assert speech_msg is not None
        assert speech_msg["text"] == "Natori taking over now."


def test_reload_characters_ws_action(monkeypatch):
    import llm; llm.reset()
    from fastapi.testclient import TestClient
    import avatars

    chatbot.greeted = True
    chatbot.processing = False

    fake_profile = dict(FAKE_AVATAR)
    fake_listing = [{"key": "natori", "name": "Natori", "description": "Office assistant, easygoing"}]

    monkeypatch.setattr(avatars, "reload", lambda: fake_profile)
    monkeypatch.setattr(avatars, "set_current", lambda key: fake_profile)
    monkeypatch.setattr(avatars, "listing", lambda: fake_listing)
    monkeypatch.setattr(llm, "set_avatar", lambda: None)
    monkeypatch.setattr(chatbot, "_filler_wavs", {})

    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # hands_free
        ws.receive_json()  # initial idle state
        ws.send_json({"action": "reload_characters"})
        msg = ws.receive_json()
        assert msg["type"] == "characters_reloaded"
        assert msg["avatar"] == "natori"
        assert msg["name"] == "Natori"
        assert msg["avatars"] == fake_listing


def test_reconnect_replays_transcript(monkeypatch):
    from fastapi.testclient import TestClient
    import chatbot, llm
    chatbot.greeted = True
    llm.reset()
    llm._session_turns[:] = [{"role": "assistant", "content": "[happy] Hi boss."}, {"role": "user", "content": "hello"}]
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "transcript", "role": "assistant", "text": "Hi boss."}
        assert ws.receive_json() == {"type": "transcript", "role": "user", "text": "hello"}
        assert ws.receive_json()["type"] == "hands_free"
        assert ws.receive_json()["type"] == "state"


def test_new_connection_releases_stale_playback_wait():
    from fastapi.testclient import TestClient
    import chatbot, llm
    chatbot.greeted = True
    llm.reset()
    chatbot.playback_done.clear()
    with TestClient(chatbot.app).websocket_connect("/ws") as ws:
        ws.receive_json()
        assert chatbot.playback_done.is_set()
