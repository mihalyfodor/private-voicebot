import io, base64

import pytest
import soundfile as sf
import numpy as np

from server import app
from server import panels
from server import speech
from server import turns as server_turns
from server import turns as server_turns
from server import turns
from server.session import Session
from tests.conftest import fake_asr, fake_tts, run_turn


def test_speak_stream_emits_speech_per_sentence_then_end():
    sent = []
    reply = speech.speak_stream([("delta", "[happy] Sure thing."), ("delta", " Anything else?")], sent.append, tts=fake_tts)

    assert reply == "Sure thing. Anything else?"
    types = [m["type"] for m in sent]
    assert types == ["state", "speech", "speech", "speech_end"]
    assert sent[0]["value"] == "speaking"
    assert [m["text"] for m in sent[1:3]] == ["Sure thing.", "Anything else?"]
    assert all(m["emotion"] == "happy" for m in sent[1:3])
    for m in sent[1:3]:
        data, sr = sf.read(io.BytesIO(base64.b64decode(m["wav"])))
        assert sr == 24000 and len(data) == 240      # the fake's payload, framed as a WAV


def test_tool_call_emits_cached_filler_before_answer(monkeypatch):
    from server import speech
    calls = []

    def counting_tts(text):
        calls.append(text)
        return fake_tts(text)

    events = [("tool_calls", ["get_weather"]), ("delta", "[neutral] It is sunny.")]
    speech._filler_wavs.clear()
    speech._last.clear()
    monkeypatch.setitem(speech.FILLERS, "get_weather", ["Let me check outside."])  # deterministic

    for _ in range(2):
        sent = []
        reply = speech.speak_stream(events, sent.append, tts=counting_tts)
        assert reply == "It is sunny."
        assert [m["type"] for m in sent] == ["state", "speech", "speech", "speech_end"]
        assert (sent[1]["emotion"], sent[1]["text"]) == ("thinking", "Let me check outside.")
        assert (sent[2]["emotion"], sent[2]["text"]) == ("neutral", "It is sunny.")

    assert calls.count("Let me check outside.") == 1  # cached across runs


def test_speak_stream_sends_speech_end_and_reraises_on_tts_error():
    calls = []

    def flaky_tts(text):
        calls.append(text)
        if len(calls) == 2:
            raise RuntimeError("tts died")
        return fake_tts(text)

    sent = []
    events = [("delta", "[happy] First one."), ("delta", " Second one.")]
    with pytest.raises(RuntimeError, match="tts died"):
        speech.speak_stream(events, sent.append, tts=flaky_tts)

    assert [m["type"] for m in sent] == ["state", "speech", "speech_end"]


def test_apply_avatar_respects_env_override(monkeypatch):
    monkeypatch.setenv("KOKORO_VOICE", "af_bella")
    monkeypatch.delenv("KOKORO_SPEED", raising=False)

    profile = speech.apply_avatar("wanko")

    assert profile["voice"] == "am_puck"  # profile itself is unaffected
    assert speech.KOKORO_VOICE == "af_bella"  # but env wins for runtime setting
    assert speech.KOKORO_SPEED == profile["speed"]  # no env override here


def _run_utterance(sess):
    """One utterance turn on a background loop, captured at `chatbot.send`; the sent messages."""
    return run_turn(server_turns.handle_utterance, sess, sess.next_turn(), np.zeros(10, dtype=np.float32), via="send")


def test_empty_transcript_still_sends_idle_state(monkeypatch):
    monkeypatch.setattr(server_turns, "asr", fake_asr())
    sent = _run_utterance(Session(websocket=None))

    assert {"type": "state", "value": "idle", "turn": 1} in sent
    assert not any(m["type"] == "transcript" for m in sent)


def test_nonempty_transcript_sends_transcript_and_calls_respond(monkeypatch):
    respond_calls = []
    monkeypatch.setattr(server_turns, "asr", fake_asr("hello there"))
    monkeypatch.setattr(turns, "respond",
                        lambda sess, turn, text, send_sync: respond_calls.append((turn, text)))

    sent = _run_utterance(Session(websocket=None))

    assert respond_calls == [(1, "hello there")]
    assert {"type": "transcript", "role": "user", "text": "hello there"} in sent
    assert not any(m.get("value") == "idle" for m in sent)


def test_disconnect_only_clears_the_current_session():
    from fastapi.testclient import TestClient
    client = TestClient(app.app)

    with client.websocket_connect("/ws") as ws1:
        ws1.receive_json()
        first = panels.session
        assert first is not None
    # ws1's disconnect handler has now run (it only clears the module global when the
    # session is still its own); a fresh connection below must not be affected by it.
    assert panels.session is None
    with client.websocket_connect("/ws") as ws2:
        ws2.receive_json()
        second = panels.session
        assert second is not None and second is not first
        assert second.websocket is not None
