from types import SimpleNamespace

from fastapi.testclient import TestClient

from server import app
from server import panels
from server import turns
from server import panels
from server import speech as s_speech
from agent import conversation
from agent import conversation
from server.session import Session
from tests.conftest import drain_initial, fake_tts, run_turn
from core import llm as core_llm


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
    from core import avatars
    a = avatars.AVATARS["wanko"]
    assert a["greeting"] == "[happy] Morning! Wanko here. What are we working on today?"


def test_say_canned_emits_state_speech_speech_end_and_transcript():
    sent = []
    reply = turns.say_canned("[happy] Hi there.", sent.append, tts=fake_tts)

    assert reply == "Hi there."
    types = [m["type"] for m in sent]
    assert types == ["state", "speech", "speech_end", "transcript"]
    assert sent[0]["value"] == "speaking"
    assert sent[1]["emotion"] == "happy"
    assert sent[1]["text"] == "Hi there."
    assert sent[3] == {"type": "transcript", "role": "assistant", "text": "Hi there."}


def test_greet_does_not_call_llm_and_records_greeting(monkeypatch):
    conversation.reset()

    class BoomClient:
        def __getattr__(self, name):
            raise AssertionError("LLM should not be called during greet()")

    monkeypatch.setattr(s_speech, "AVATAR", FAKE_AVATAR)
    monkeypatch.setattr(s_speech, "synthesize", fake_tts)
    monkeypatch.setattr(core_llm, "_client", BoomClient())

    sent = []
    monkeypatch.setattr(panels, "make_sender", lambda loop, turn=None: sent.append)

    sess = Session(websocket=None)  # no socket → no playback wait
    turns.greet(sess, sess.next_turn(), SimpleNamespace())

    assert panels.is_busy() is False
    assert conversation._conversation[-1] == {"role": "assistant", "content": "Morning! Wanko here."}
    assert conversation._session_turns[-1] == {"role": "assistant", "content": "Morning! Wanko here."}
    assert {"type": "transcript", "role": "assistant", "text": "Morning! Wanko here."} in sent
    assert {"type": "state", "value": "idle"} in sent


def test_greet_messages_carry_the_turn_id(monkeypatch):
    """Every state/speech/speech_end/assistant-transcript message is tagged with its turn."""
    conversation.reset()
    monkeypatch.setattr(s_speech, "AVATAR", FAKE_AVATAR)
    monkeypatch.setattr(s_speech, "synthesize", fake_tts)

    sess = Session(websocket=None)
    sent = run_turn(turns.greet, sess, sess.next_turn(), via="send")

    types = [m["type"] for m in sent]
    assert types[:2] == ["state", "state"]  # thinking, then speaking
    assert types[-3:] == ["speech_end", "transcript", "state"]
    assert all(m["turn"] == 1 for m in sent)


def test_reload_characters_ws_action(monkeypatch):
    from core import avatars
    conversation.reset()

    fake_profile = dict(FAKE_AVATAR)

    monkeypatch.setattr(avatars, "reload", lambda: fake_profile)
    monkeypatch.setattr(avatars, "set_current", lambda key: fake_profile)
    monkeypatch.setattr(conversation, "set_avatar", lambda: None)
    monkeypatch.setattr(s_speech, "_filler_wavs", {})

    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        ws.send_json({"action": "reload_characters"})
        msg = ws.receive_json()
        assert msg == {"type": "characters_reloaded", "avatar": "wanko", "name": "Wanko"}


def test_reconnect_replays_transcript():
    conversation.reset()
    conversation._conversation += [{"role": "assistant", "content": "[happy] Hi there."},
                          {"role": "user", "content": "hello"}]                 # what the model holds (PRD 39)
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json() == {"type": "transcript", "role": "assistant", "text": "Hi there."}
        assert ws.receive_json() == {"type": "transcript", "role": "user", "text": "hello"}
        assert ws.receive_json()["type"] == "hands_free"
        assert ws.receive_json()["type"] == "state"


def test_new_connection_releases_the_previous_playback_wait():
    conversation.reset()
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws1:
        ws1.receive_json()
        first = panels.session
        first.playback_done.clear()
        with client.websocket_connect("/ws") as ws2:
            ws2.receive_json()
            assert panels.session is not first
            assert first.playback_done.is_set()
            assert first.websocket is None
