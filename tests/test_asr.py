import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace

import numpy as np
import pytest

import asr


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


def test_model_env_default():
    assert isinstance(asr.MODEL, str) and asr.MODEL


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
    import chatbot
    import memory

    memory._write_profile(memory._normalise(
        {"identity": {"name": "Mihaly Fodor", "aliases": ["Misi"]}}))
    calls = []
    monkeypatch.setattr(chatbot, "asr", SimpleNamespace(
        transcribe=lambda audio, prompt=None: calls.append(prompt) or ""))
    monkeypatch.setattr(chatbot, "make_sender", lambda loop, turn=None: lambda msg: None)
    chatbot.handle_utterance(SimpleNamespace(), 1, np.zeros(16000, dtype=np.float32), None)
    assert calls == ["Names: Mihaly Fodor, Misi."]
