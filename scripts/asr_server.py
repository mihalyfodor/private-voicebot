"""Host ASR service (PRD 40): mlx-whisper stays native, the container only talks to it.

    POST /transcribe   body = 16 kHz mono wav bytes, optional ?prompt=   -> {text, segments}
    GET  /health                                                         -> {ok, model}

Runs from the checkout's venv (`python scripts/asr_server.py`, or via
scripts/launchd/com.voicebot.asr.plist); binds ASR_HOST:ASR_PORT, 127.0.0.1:8020 by default.
The junk filter stays in `asr.transcribe` on the caller's side, so both backends behave the same.
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import soundfile as sf
from fastapi import FastAPI, HTTPException, Request

import asr

app = FastAPI()


def decode_wav(data: bytes):
    """wav bytes -> float32 mono at asr.SAMPLERATE; anything else is a 400."""
    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"not a wav: {exc}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != asr.SAMPLERATE:
        raise HTTPException(status_code=400, detail=f"expected {asr.SAMPLERATE} Hz, got {rate}")
    return audio


@app.get("/health")
async def health():
    return {"ok": True, "model": asr.MODEL}


@app.post("/transcribe")
async def transcribe(request: Request, prompt: str = ""):
    audio = decode_wav(await request.body())
    result = asr._mlx_transcribe(audio, prompt or None)
    segments = [{"avg_logprob": s.get("avg_logprob")} for s in result.get("segments") or []
                if isinstance(s, dict)]
    return {"text": result.get("text") or "", "segments": segments}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("ASR_HOST", "127.0.0.1"),
                port=int(os.getenv("ASR_PORT", "8020")), log_level="warning")
