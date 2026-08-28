"""In-process speech-to-text via mlx-whisper.

Replaces the whisper-cli subprocess: accepts a float32 16 kHz mono array and
returns transcribed text, filtering out common whisper hallucination
artifacts on near-silent/junk audio.
"""

import os
import re

MODEL = os.getenv("WHISPER_MLX_MODEL", "mlx-community/whisper-small-mlx")

_JUNK_PATTERNS = [
    re.compile(r"^\[blank_audio\]$"),
    re.compile(r"^\(blank\)$"),
    re.compile(r"^\[music\]$"),
    re.compile(r"^♪+$"),
]


def is_junk(text, avg_logprob=None):
    if text is None:
        return True
    stripped = text.strip()
    if not stripped:
        return True

    normalized = stripped.lower().strip("♪ ").strip()
    if not normalized:
        return True
    for pattern in _JUNK_PATTERNS:
        if pattern.match(normalized):
            return True
    # Catch stray artifact tokens even if embedded with other punctuation.
    if "♪" in stripped and len(stripped.strip("♪ \t")) == 0:
        return True

    words = stripped.split()
    if len(words) < 2 and avg_logprob is not None and avg_logprob < -1.0:
        return True

    return False


def _mlx_transcribe(audio):
    import mlx_whisper

    return mlx_whisper.transcribe(
        audio, path_or_hf_repo=MODEL, language="en", fp16=True
    )


def transcribe(audio):
    result = _mlx_transcribe(audio)
    text = (result.get("text") or "").strip()

    segments = result.get("segments") or []
    logprobs = [
        seg["avg_logprob"]
        for seg in segments
        if isinstance(seg, dict) and "avg_logprob" in seg
    ]
    avg_logprob = sum(logprobs) / len(logprobs) if logprobs else None

    if is_junk(text, avg_logprob):
        return ""

    return text
