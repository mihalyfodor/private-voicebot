import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
        lambda audio: {
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
        lambda audio: {"text": "[BLANK_AUDIO]", "segments": []},
    )
    audio = np.zeros(16000, dtype=np.float32)
    assert asr.transcribe(audio) == ""


def test_transcribe_filters_low_confidence_short_text(monkeypatch):
    monkeypatch.setattr(
        asr,
        "_mlx_transcribe",
        lambda audio: {
            "text": "uh",
            "segments": [{"avg_logprob": -2.0}],
        },
    )
    audio = np.zeros(16000, dtype=np.float32)
    assert asr.transcribe(audio) == ""


def test_model_env_default():
    assert isinstance(asr.MODEL, str) and asr.MODEL
