
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core import settings
from server import app
from server import panels
from server import turns as server_turns
from server import vad as server_vad
from server import turns
from tests.conftest import Blocker, drain_initial, fake_asr


@pytest.fixture(autouse=True)
def reset_capture_state(monkeypatch):
    """hands_free is a persisted app setting (module-level); the rest lives on the session."""
    monkeypatch.setattr(panels, "hands_free", False)
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


def test_set_hands_free_true_persists_and_announces_and_reflects_in_config():
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)  # {"type":"hands_free","value":False}, {"type":"state","value":"idle"}

        ws.send_json({"action": "set_hands_free", "value": True})
        assert ws.receive_json() == {"type": "hands_free", "value": True}

    assert settings.load_settings()["hands_free"] is True
    assert panels.hands_free is True
    assert client.get("/api/config").json()["hands_free"] is True


def test_set_hands_free_false_sends_listening_idle(monkeypatch):
    monkeypatch.setattr(panels, "hands_free", True)
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        ws.send_json({"action": "set_hands_free", "value": False})
        assert ws.receive_json() == {"type": "hands_free", "value": False}
        assert ws.receive_json() == {"type": "listening", "value": "idle"}


def test_hands_free_binary_frames_drive_listening_and_utterance(monkeypatch):
    monkeypatch.setattr(panels, "hands_free", True)
    monkeypatch.setattr(server_vad, "Endpointer", FakeEndpointer)
    monkeypatch.setattr(server_turns, "asr", fake_asr("hello there"))
    respond_calls = []
    monkeypatch.setattr(turns, "respond",
                        lambda sess, turn, text, send_sync: respond_calls.append((turn, text)))

    client = TestClient(app.app)
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
        assert isinstance(panels.session.endpointer, FakeEndpointer)

    panels.controller.join_idle(timeout=5)
    assert respond_calls == [(1, "hello there")]


def test_hands_free_frames_gated_while_a_turn_runs(monkeypatch):
    """Scenario 2: the endpointer is gated synchronously in the frame handler."""
    monkeypatch.setattr(panels, "hands_free", True)
    fake = FakeEndpointer()
    respond_calls = []
    monkeypatch.setattr(turns, "respond",
                        lambda sess, turn, text, send_sync: respond_calls.append(text))
    monkeypatch.setattr(server_turns, "asr", fake_asr("should not run"))

    blocker = Blocker()
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        panels.session.endpointer = fake
        blocker.hold(panels.controller)  # controller.busy is now True

        for _ in range(3):  # feed #3 would emit an utterance if it were not gated
            ws.send_bytes(frame_bytes())
        ws.send_json({"action": "set_backdrop", "key": "none"})  # round-trip: frames are processed
        assert ws.receive_json()["type"] == "backdrop"

        assert fake.gated is True
        assert fake.calls == 3
        blocker.free(panels.controller)

    assert respond_calls == []


def test_text_turn_runs_while_hands_free_is_on(monkeypatch):
    """PRD 24: typing works whatever the mic is doing — it is just another queued turn."""
    monkeypatch.setattr(panels, "hands_free", True)
    respond_calls = []
    monkeypatch.setattr(turns, "respond",
                        lambda sess, turn, text, send_sync, origin="user": respond_calls.append((text, origin)))

    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        ws.send_json({"action": "text_turn", "text": "typed hello"})
        ws.send_json({"action": "set_backdrop", "key": "none"})   # round trip
        assert ws.receive_json()["type"] == "backdrop"

    panels.controller.join_idle(timeout=5)
    assert respond_calls == [("typed hello", "typed")]
