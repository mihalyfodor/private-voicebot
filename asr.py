"""Speech-to-text: in-process mlx-whisper, or a host ASR service over HTTP (PRD 40).

Accepts a float32 16 kHz mono array and returns transcribed text, filtering
out common whisper hallucination artifacts on near-silent/junk audio.

`ASR_BACKEND=mlx` (default) runs mlx-whisper in this process — Apple Silicon only.
`ASR_BACKEND=remote` posts the utterance as wav bytes to `ASR_URL` (`scripts/asr_server.py`,
which wraps the same mlx path on the host) and expects the same `{text, segments}` JSON back.
A dead service is an error line and an empty transcript, never a crash.
"""

import io
import os
import re

MODEL = os.getenv("WHISPER_MLX_MODEL", "mlx-community/whisper-small-mlx")
BACKEND = os.getenv("ASR_BACKEND", "mlx")
URL = os.getenv("ASR_URL", "http://127.0.0.1:8020").rstrip("/")
REMOTE_TIMEOUT_S = 30
SAMPLERATE = 16000
#: the last remote failure's message, "" after a call that reached the service (PRD 41 re-probe hook)
last_error = ""

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


def _mlx_transcribe(audio, prompt=None):
    import mlx_whisper

    kwargs = {"initial_prompt": prompt} if prompt else {}
    return mlx_whisper.transcribe(
        audio, path_or_hf_repo=MODEL, language="en", fp16=True, **kwargs
    )


def wav_bytes(audio) -> bytes:
    """The utterance as a 16 kHz mono PCM16 wav — the wire format of the ASR service."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLERATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _remote_transcribe(audio, prompt=None):
    """POST the wav to the host ASR service; `{error}` with a clear message on any failure."""
    import requests

    params = {"prompt": prompt} if prompt else {}
    try:
        r = requests.post(f"{URL}/transcribe", data=wav_bytes(audio), params=params,
                          headers={"Content-Type": "audio/wav"}, timeout=REMOTE_TIMEOUT_S)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as exc:
        return {"error": f"ASR service at {URL} unavailable: {exc}"}


def raw_transcribe(audio, prompt=None) -> dict:
    """The configured backend's `{text, segments}` (or `{error}`) — no junk filtering."""
    if BACKEND == "remote":
        return _remote_transcribe(audio, prompt)
    return _mlx_transcribe(audio, prompt)


def transcribe(audio, prompt=None):
    """Transcribe one utterance. `prompt` biases whisper towards names it keeps mangling
    (memory.asr_prompt(); see docs/26-identity-aliases.md) — empty means no biasing.

    Returns "" for junk and for a failed remote call (the error is printed, never raised),
    so the turn takes the same "nothing heard" path either way.
    """
    global last_error
    result = raw_transcribe(audio, prompt)
    last_error = result.get("error") or ""
    if last_error:
        print(f"[ASR] {last_error}")
        return ""
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
