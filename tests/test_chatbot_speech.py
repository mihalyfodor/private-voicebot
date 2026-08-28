import sys, os, io, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
import soundfile as sf
import numpy as np

import chatbot


def fake_tts(text):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(2400, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_speak_stream_emits_speech_per_sentence_then_end():
    sent = []
    reply = chatbot.speak_stream([("delta", "[happy] Sure thing."), ("delta", " Anything else?")], sent.append, tts=fake_tts)

    assert reply == "Sure thing. Anything else?"
    types = [m["type"] for m in sent]
    assert types == ["state", "speech", "speech", "speech_end"]
    assert sent[0]["value"] == "speaking"
    assert [m["text"] for m in sent[1:3]] == ["Sure thing.", "Anything else?"]
    assert all(m["emotion"] == "happy" for m in sent[1:3])
    for m in sent[1:3]:
        data, sr = sf.read(io.BytesIO(base64.b64decode(m["wav"])))
        assert sr == 24000 and len(data) == 2400


def test_tool_call_emits_cached_filler_before_answer(monkeypatch):
    import fillers
    calls = []

    def counting_tts(text):
        calls.append(text)
        return fake_tts(text)

    events = [("tool_calls", ["get_weather"]), ("delta", "[neutral] It is sunny.")]
    chatbot._filler_wavs.clear()
    fillers._last.clear()
    monkeypatch.setitem(fillers.FILLERS, "get_weather", ["Let me check outside."])  # deterministic

    for _ in range(2):
        sent = []
        reply = chatbot.speak_stream(events, sent.append, tts=counting_tts)
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
        chatbot.speak_stream(events, sent.append, tts=flaky_tts)

    assert [m["type"] for m in sent] == ["state", "speech", "speech_end"]


def test_apply_avatar_respects_env_override(monkeypatch, tmp_path):
    import avatars
    monkeypatch.setattr(avatars, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(avatars, "_current", None)
    monkeypatch.delenv("AVATAR", raising=False)
    monkeypatch.setenv("KOKORO_VOICE", "af_bella")
    monkeypatch.delenv("KOKORO_SPEED", raising=False)

    profile = chatbot.apply_avatar("natori")

    assert profile["voice"] == "am_michael"  # profile itself is unaffected
    assert chatbot.KOKORO_VOICE == "af_bella"  # but env wins for runtime setting
    assert chatbot.KOKORO_SPEED == profile["speed"]  # no env override here


def test_empty_transcript_still_sends_idle_state(monkeypatch):
    sent = []

    async def fake_send(msg):
        sent.append(msg)

    monkeypatch.setattr(chatbot, "send", fake_send)
    monkeypatch.setattr(chatbot, "transcribe", lambda path: "")
    monkeypatch.setattr(chatbot.sf, "write", lambda *a, **k: None)
    monkeypatch.setattr(chatbot.os, "remove", lambda path: None)
    monkeypatch.setattr(chatbot, "recording", True)
    monkeypatch.setattr(chatbot, "audio_chunks", [np.zeros((10, 1), dtype=np.float32)])
    monkeypatch.setattr(chatbot, "stream", SimpleNamespace(stop=lambda: None, close=lambda: None))

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    try:
        chatbot.handle_toggle(loop)
        for _ in range(100):
            if not chatbot.processing:
                break
            time.sleep(0.02)
        else:
            pytest.fail("processing never finished")
        # give the last run_coroutine_threadsafe call a beat to land
        time.sleep(0.05)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)

    assert {"type": "state", "value": "idle"} in sent


def test_disconnect_only_clears_the_current_ws_client():
    from fastapi.testclient import TestClient
    import chatbot
    chatbot.greeted = True
    client = TestClient(chatbot.app)

    with client.websocket_connect("/ws") as ws1:
        ws1.receive_json()
        first_client = chatbot.ws_client
        assert first_client is not None
    # ws1's disconnect handler has now run (checked `ws_client is websocket`
    # before clearing); a fresh connection below must not be affected by it.
    with client.websocket_connect("/ws") as ws2:
        ws2.receive_json()
        second_client = chatbot.ws_client
        assert second_client is not None and second_client is not first_client
        assert chatbot.ws_client is second_client


def test_mic_open_failure_stays_idle(monkeypatch):
    import asyncio
    sent = []

    class BoomStream:
        def __init__(self, *a, **k):
            raise RuntimeError("PaErrorCode -9986")

    monkeypatch.setattr(chatbot.sd, "InputStream", BoomStream)
    monkeypatch.setattr(chatbot, "recording", False)
    monkeypatch.setattr(chatbot, "processing", False)
    monkeypatch.setattr(chatbot.asyncio, "run_coroutine_threadsafe", lambda coro, loop: (sent.append(coro), coro.close()))
    chatbot.handle_toggle(loop=None)
    assert chatbot.recording is False
    assert len(sent) == 2  # error + idle
