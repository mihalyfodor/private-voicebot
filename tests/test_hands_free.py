import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import avatars
import chatbot
from tests.conftest import Blocker, untagged


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(avatars, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(avatars, "_current", None)
    monkeypatch.delenv("AVATAR", raising=False)
    yield


@pytest.fixture(autouse=True)
def reset_capture_state(monkeypatch):
    """hands_free is a persisted app setting (module-level); the rest lives on the session."""
    monkeypatch.setattr(chatbot, "hands_free", False)
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


def fake_respond(sess, turn, text, send_sync):
    """Stand-in for the LLM half of a turn."""
    return text


def frame_bytes(n=512):
    return np.zeros(n, dtype=np.int16).tobytes()


def drain_initial(ws):
    """Consume the hands_free announce + idle state sent right after connect."""
    return [ws.receive_json() for _ in range(2)]


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
    monkeypatch.setattr(chatbot, "respond",
                        lambda sess, turn, text, send_sync: respond_calls.append((turn, text)))

    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)

        ws.send_bytes(frame_bytes())  # call 1: hearing flips on
        assert ws.receive_json() == {"type": "listening", "value": "hearing"}

        ws.send_bytes(frame_bytes())  # call 2: nothing

        ws.send_bytes(frame_bytes())  # call 3: utterance emitted -> turn submitted
        msg1 = ws.receive_json()
        msg2 = ws.receive_json()
        assert msg1 == {"type": "state", "value": "processing", "turn": 1}
        assert msg2 == {"type": "transcript", "role": "user", "text": "hello there"}
        assert isinstance(chatbot.session.endpointer, FakeEndpointer)

    chatbot.controller.join_idle(timeout=5)
    assert respond_calls == [(1, "hello there")]


def test_hands_free_frames_gated_while_a_turn_runs(monkeypatch):
    """Scenario 2: the endpointer is gated synchronously in the frame handler."""
    monkeypatch.setattr(chatbot, "hands_free", True)
    fake = FakeEndpointer()
    respond_calls = []
    monkeypatch.setattr(chatbot, "respond",
                        lambda sess, turn, text, send_sync: respond_calls.append(text))
    monkeypatch.setattr(chatbot, "asr", SimpleNamespace(transcribe=lambda audio: "should not run"))

    blocker = Blocker()
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        chatbot.session.endpointer = fake
        blocker.hold(chatbot.controller)  # controller.busy is now True

        for _ in range(3):  # feed #3 would emit an utterance if it were not gated
            ws.send_bytes(frame_bytes())
        ws.send_json({"action": "set_backdrop", "key": "none"})  # round-trip: frames are processed
        assert ws.receive_json()["type"] == "backdrop"

        assert fake.gated is True
        assert fake.calls == 3
        blocker.free(chatbot.controller)

    assert respond_calls == []


def test_ptt_start_then_stop_transcribes_and_responds(monkeypatch):
    monkeypatch.setattr(chatbot, "asr", SimpleNamespace(transcribe=lambda audio: "ptt hello"))
    respond_calls = []
    monkeypatch.setattr(chatbot, "respond",
                        lambda sess, turn, text, send_sync: respond_calls.append(text))

    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)

        ws.send_json({"action": "ptt", "value": "start"})
        assert untagged(ws.receive_json()) == {"type": "state", "value": "recording"}
        assert chatbot.session.ptt_active is True

        # Enough samples to clear the 0.3s-min-length discard.
        ws.send_bytes(np.zeros(5000, dtype=np.int16).tobytes())

        ws.send_json({"action": "ptt", "value": "stop"})
        msg1 = ws.receive_json()
        msg2 = ws.receive_json()
        assert msg1 == {"type": "state", "value": "processing", "turn": 1}
        assert msg2 == {"type": "transcript", "role": "user", "text": "ptt hello"}
        assert chatbot.session.ptt_active is False

    chatbot.controller.join_idle(timeout=5)
    assert respond_calls == ["ptt hello"]


def test_ptt_start_ignored_while_a_turn_runs():
    blocker = Blocker()
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        blocker.hold(chatbot.controller)
        ws.send_json({"action": "ptt", "value": "start"})
        ws.send_json({"action": "set_backdrop", "key": "none"})  # round trip
        assert ws.receive_json()["type"] == "backdrop"  # no "recording" came first
        assert chatbot.session.ptt_active is False
        blocker.free(chatbot.controller)


def test_stray_ptt_stop_while_busy_emits_nothing():
    """Scenario 11: a stop with no active recording must not emit `state idle`."""
    blocker = Blocker()
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        blocker.hold(chatbot.controller)
        ws.send_json({"action": "ptt", "value": "stop"})
        ws.send_json({"action": "set_backdrop", "key": "none"})  # round trip
        assert ws.receive_json()["type"] == "backdrop"  # nothing between
        assert chatbot.session.recorder is None
        blocker.free(chatbot.controller)
