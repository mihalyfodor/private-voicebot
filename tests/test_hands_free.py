import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import avatars
import chatbot


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(avatars, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(avatars, "_current", None)
    monkeypatch.delenv("AVATAR", raising=False)
    yield


@pytest.fixture(autouse=True)
def reset_chatbot_state(monkeypatch):
    """chatbot's capture state is module-level (single client); reset it around every test."""
    monkeypatch.setattr(chatbot, "greeted", True)  # skip the greeting thread
    monkeypatch.setattr(chatbot, "processing", False)
    monkeypatch.setattr(chatbot, "hands_free", False)
    monkeypatch.setattr(chatbot, "endpointer", None)
    monkeypatch.setattr(chatbot, "recorder", None)
    monkeypatch.setattr(chatbot, "ptt_active", False)
    monkeypatch.setattr(chatbot, "_hearing", False)
    yield


class FakeEndpointer:
    """Feed #1 flips `hearing` on; feed #3 emits one utterance."""

    def __init__(self, **kwargs):
        self.gated = False
        self.calls = 0
        self.hearing = False

    def feed(self, pcm):
        self.calls += 1
        if self.gated:
            return []
        if self.calls == 1:
            self.hearing = True
        if self.calls == 3:
            return [np.zeros(1600, dtype=np.float32)]
        return []

    def reset(self):
        self.calls = 0
        self.hearing = False


def frame_bytes(n=512):
    return np.zeros(n, dtype=np.int16).tobytes()


def drain_initial(ws):
    """Consume the hands_free announce + idle state sent right after connect."""
    msgs = []
    for _ in range(2):
        msgs.append(ws.receive_json())
    return msgs


def test_set_hands_free_true_persists_and_announces_and_reflects_in_config():
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)  # {"type":"hands_free","value":False}, {"type":"state","value":"idle"}

        ws.send_json({"action": "set_hands_free", "value": True})
        assert ws.receive_json() == {"type": "hands_free", "value": True}

    assert avatars.load_settings()["hands_free"] is True
    assert chatbot.hands_free is True
    assert client.get("/api/config").json()["hands_free"] is True


def test_set_hands_free_false_sends_listening_idle(monkeypatch):
    monkeypatch.setattr(chatbot, "hands_free", True)
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        ws.send_json({"action": "set_hands_free", "value": False})
        assert ws.receive_json() == {"type": "hands_free", "value": False}
        assert ws.receive_json() == {"type": "listening", "value": "idle"}


def test_hands_free_binary_frames_drive_listening_and_utterance(monkeypatch):
    monkeypatch.setattr(chatbot, "hands_free", True)
    monkeypatch.setattr(chatbot.vad, "Endpointer", FakeEndpointer)
    monkeypatch.setattr(chatbot, "asr", SimpleNamespace(transcribe=lambda audio: "hello there"))
    respond_calls = []
    monkeypatch.setattr(chatbot, "respond", lambda text, loop: respond_calls.append(text))

    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)

        ws.send_bytes(frame_bytes())  # call 1: hearing flips on
        assert ws.receive_json() == {"type": "listening", "value": "hearing"}

        ws.send_bytes(frame_bytes())  # call 2: nothing

        ws.send_bytes(frame_bytes())  # call 3: utterance emitted -> handle_utterance thread
        msg1 = ws.receive_json()
        msg2 = ws.receive_json()
        assert msg1 == {"type": "state", "value": "processing"}
        assert msg2 == {"type": "transcript", "role": "user", "text": "hello there"}

    assert respond_calls == ["hello there"]
    assert isinstance(chatbot.endpointer, FakeEndpointer)


def test_hands_free_frames_gated_while_processing_do_not_respond(monkeypatch):
    monkeypatch.setattr(chatbot, "hands_free", True)
    monkeypatch.setattr(chatbot, "processing", True)
    fake = FakeEndpointer()
    monkeypatch.setattr(chatbot, "endpointer", fake)
    respond_calls = []
    monkeypatch.setattr(chatbot, "respond", lambda text, loop: respond_calls.append(text))
    monkeypatch.setattr(chatbot, "asr", SimpleNamespace(transcribe=lambda audio: "should not run"))

    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        ws.send_bytes(frame_bytes())
        ws.send_json({"action": "playback_done"})  # round-trip to be sure the frame was processed
        chatbot.playback_done.wait(timeout=1)

    assert fake.gated is True
    assert respond_calls == []


def test_ptt_start_then_stop_transcribes_and_responds(monkeypatch):
    monkeypatch.setattr(chatbot, "asr", SimpleNamespace(transcribe=lambda audio: "ptt hello"))
    respond_calls = []
    monkeypatch.setattr(chatbot, "respond", lambda text, loop: respond_calls.append(text))

    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)

        ws.send_json({"action": "ptt", "value": "start"})
        assert ws.receive_json() == {"type": "state", "value": "recording"}

        # Enough samples to clear the 0.3s-min-length discard.
        ws.send_bytes(np.zeros(5000, dtype=np.int16).tobytes())

        ws.send_json({"action": "ptt", "value": "stop"})
        msg1 = ws.receive_json()
        msg2 = ws.receive_json()
        assert msg1 == {"type": "state", "value": "processing"}
        assert msg2 == {"type": "transcript", "role": "user", "text": "ptt hello"}

    assert respond_calls == ["ptt hello"]


def test_ptt_start_ignored_while_processing(monkeypatch):
    monkeypatch.setattr(chatbot, "processing", True)
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        ws.send_json({"action": "ptt", "value": "start"})
        ws.send_json({"action": "playback_done"})  # forces a round trip; no "recording" should appear first
    assert chatbot.ptt_active is False
