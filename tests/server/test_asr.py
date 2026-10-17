import sys, os

from types import SimpleNamespace

import numpy as np
import pytest

from server import asr
from tests.conftest import fake_asr


# Scenario 5: is_junk cases.
@pytest.mark.parametrize(
    "text,logprob,expected",
    [
        ("", None, True),
        ("   ", None, True),
        ("[BLANK_AUDIO]", None, True),
        ("[blank_audio]", None, True),
        ("(blank)", None, True),
        ("[Music]", None, True),
        ("♪", None, True),
        ("♪♪♪", None, True),
        ("uh", -1.5, True),
        ("uh", -0.2, False),
        ("uh", None, False),
        ("what time is it", -3.0, False),
        ("what time is it", None, False),
    ],
)
def test_is_junk(text, logprob, expected):
    assert asr.is_junk(text, logprob) is expected


def test_transcribe_returns_clean_text(monkeypatch):
    monkeypatch.setattr(
        asr,
        "_mlx_transcribe",
        lambda audio, prompt=None: {
            "text": " Hello there. ",
            "segments": [{"avg_logprob": -0.3}],
        },
    )
    audio = np.zeros(16000, dtype=np.float32)
    assert asr.transcribe(audio) == "Hello there."


def test_transcribe_filters_junk(monkeypatch):
    monkeypatch.setattr(
        asr,
        "_mlx_transcribe",
        lambda audio, prompt=None: {"text": "[BLANK_AUDIO]", "segments": []},
    )
    audio = np.zeros(16000, dtype=np.float32)
    assert asr.transcribe(audio) == ""


def test_transcribe_filters_low_confidence_short_text(monkeypatch):
    monkeypatch.setattr(
        asr,
        "_mlx_transcribe",
        lambda audio, prompt=None: {
            "text": "uh",
            "segments": [{"avg_logprob": -2.0}],
        },
    )
    audio = np.zeros(16000, dtype=np.float32)
    assert asr.transcribe(audio) == ""


# PRD 26 scenario 6: the profile's names are handed to whisper as `initial_prompt`.
def test_transcribe_passes_the_prompt_to_whisper_as_initial_prompt(monkeypatch):
    seen = {}

    def fake(audio, **kwargs):
        seen.update(kwargs)
        return {"text": "Misi called", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=fake))

    audio = np.zeros(16000, dtype=np.float32)
    assert asr.transcribe(audio, prompt="Names: Mihaly Fodor, Misi.") == "Misi called"
    assert seen["initial_prompt"] == "Names: Mihaly Fodor, Misi."
    assert seen["path_or_hf_repo"] == asr.MODEL


def test_transcribe_without_a_prompt_sends_no_initial_prompt(monkeypatch):
    seen = {}

    def fake(audio, **kwargs):
        seen.update(kwargs)
        return {"text": "hello there", "segments": []}

    monkeypatch.setitem(sys.modules, "mlx_whisper", SimpleNamespace(transcribe=fake))

    assert asr.transcribe(np.zeros(16000, dtype=np.float32)) == "hello there"
    assert "initial_prompt" not in seen


def test_the_chat_path_hands_the_profile_names_to_the_asr(monkeypatch, memory_sandbox):
    from server import turns as server_turns
    from server import turns as server_turns
    from server import panels
    from memory import store

    store._write_profile(store._normalise(
        {"identity": {"name": "Mihaly Fodor", "aliases": ["Misi"]}}))
    prompts = []
    monkeypatch.setattr(server_turns, "asr", fake_asr(prompts=prompts))
    monkeypatch.setattr(panels, "make_sender", lambda loop, turn=None: lambda msg: None)
    server_turns.handle_utterance(SimpleNamespace(), 1, np.zeros(16000, dtype=np.float32), None)
    assert prompts == ["Names: Mihaly Fodor, Misi."]


# --------------------------------------------------------------------------- PRD 40 scenario 3: host ASR service


def _asr_server():
    import importlib.util
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    spec = importlib.util.spec_from_file_location("asr_server", os.path.join(root, "scripts", "asr_server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_asr_server_returns_text_and_segments(monkeypatch):
    from fastapi.testclient import TestClient
    calls = []

    def fake_mlx(audio, prompt=None):
        calls.append((len(audio), prompt))
        return {"text": " hello there ", "segments": [{"avg_logprob": -0.2, "id": 0}]}

    monkeypatch.setattr(asr, "_mlx_transcribe", fake_mlx)
    client = TestClient(_asr_server().app)
    wav = asr.wav_bytes(np.zeros(16000, dtype=np.float32))

    r = client.post("/transcribe", content=wav, params={"prompt": "Anna"}, headers={"Content-Type": "audio/wav"})
    assert r.status_code == 200
    assert r.json() == {"text": " hello there ", "segments": [{"avg_logprob": -0.2}]}
    assert calls == [(16000, "Anna")]

    assert client.get("/health").json() == {"ok": True, "model": asr.MODEL}
    assert client.post("/transcribe", content=b"not a wav", headers={"Content-Type": "audio/wav"}).status_code == 400


def test_remote_backend_posts_wav_and_returns_the_same_shape(monkeypatch):
    import requests
    seen = {}

    def fake_post(url, data=None, params=None, headers=None, timeout=None):
        seen.update(url=url, data=data, params=params, headers=headers)
        return SimpleNamespace(raise_for_status=lambda: None,
                               json=lambda: {"text": "hello there", "segments": [{"avg_logprob": -0.2}]})

    monkeypatch.setattr(asr, "BACKEND", "remote")
    monkeypatch.setattr(asr, "URL", "http://asr.test")
    monkeypatch.setattr(requests, "post", fake_post)

    assert asr.transcribe(np.zeros(1600, dtype=np.float32), prompt="Anna") == "hello there"
    assert seen["url"] == "http://asr.test/transcribe"
    assert seen["params"] == {"prompt": "Anna"}
    assert seen["data"].startswith(b"RIFF")


def test_remote_backend_down_is_an_error_string_not_an_exception(monkeypatch, capsys):
    from server import turns as server_turns
    from server import speech
    from server.session import Session
    from tests.conftest import fake_tts, run_turn

    monkeypatch.setattr(asr, "BACKEND", "remote")
    monkeypatch.setattr(asr, "URL", "http://127.0.0.1:1")
    monkeypatch.setattr(asr, "REMOTE_TIMEOUT_S", 2)
    audio = np.zeros(1600, dtype=np.float32)

    assert asr.raw_transcribe(audio)["error"].startswith("ASR service at http://127.0.0.1:1 unavailable")
    assert asr.transcribe(audio) == ""
    assert any(line.startswith("[ASR] ASR service") for line in capsys.readouterr().out.splitlines())

    monkeypatch.setattr(speech, "synthesize", fake_tts)
    sent = run_turn(server_turns.handle_utterance, Session(), 1, np.zeros(10, dtype=np.float32))
    assert [m["value"] for m in sent if m["type"] == "state"] == ["processing", "idle"]
    assert not [m for m in sent if m["type"] in ("transcript", "error")]
