"""In-process speech-to-text via mlx-whisper.

Accepts a float32 16 kHz mono array and returns transcribed text, filtering
out common whisper hallucination artifacts on near-silent/junk audio.
"""

import os
import re

MODEL = os.getenv("WHISPER_MLX_MODEL", "mlx-community/whisper-small-mlx")

_JUNK_PATTERNS = [
    re.compile(r"^\[blank_audio\]$"),
    re.compile(r"^\(blank\)$"),
    re.compile(r"^\[music\]$"),
]


def is_junk(text, avg_logprob=None):
    """True for empty text, whisper's silence artifacts, and low-confidence one-word blips."""
    stripped = (text or "").strip()
    normalized = stripped.lower().strip("♪ ").strip()
    if not normalized:
        return True
    if any(pattern.match(normalized) for pattern in _JUNK_PATTERNS):
        return True
    if len(stripped.split()) < 2 and avg_logprob is not None and avg_logprob < -1.0:
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
