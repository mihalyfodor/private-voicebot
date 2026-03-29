# PRD: Local Voice Chatbot

## Overview

A fully offline, privacy-first voice chatbot that runs entirely on a MacBook. The user speaks, the system listens, thinks, and speaks back — no internet required, no API keys, no cloud services.

---

## Goals

- 100% local and offline after initial setup
- Total RAM footprint under 8GB
- Start everything from the CLI with minimal commands
- Fast enough for natural conversation on Apple Silicon

---

## Non-Goals

- Voice cloning or custom voice training
- Multi-user support
- GUI or web interface
- Cloud fallback

---

## Stack

| Layer | Tool | Footprint |
|---|---|---|
| Speech-to-Text | Whisper.cpp (`small` model) | ~460MB |
| LLM | Ollama + Llama 3.2 3B | ~2GB |
| Text-to-Speech | Kokoro TTS (`af_heart` voice) | ~400MB RAM |
| **Total** | | **~3GB** |

All three components are open source and run natively on Apple Silicon via Metal (Whisper.cpp) and CPU/Metal (Ollama, Kokoro).

---

## Architecture

```
[user speaks]
     ↓
whisper.cpp          # STT: transcribes mic audio to text
     ↓
ollama (llama3.2)    # LLM: generates a text response
     ↓
kokoro-tts           # TTS: speaks the response aloud
```

---

## Component Details

### Speech-to-Text — Whisper.cpp

- C++ port of OpenAI Whisper, optimized for Apple Silicon via Metal
- Uses the `small` model (~460MB) — good balance of speed and accuracy
- Fully offline after model download
- Install via Homebrew: `brew install whisper-cpp`

### LLM — Ollama + Llama 3.2 3B

- Ollama manages model downloads and serves a local REST API at `http://localhost:11434`
- Llama 3.2 3B is ~2GB and fast enough for real-time conversation
- Install via Homebrew: `brew install ollama`
- Pull model: `ollama pull llama3.2`

### Text-to-Speech — Kokoro TTS

- 82M parameter ONNX model, ~400MB RAM at runtime
- High quality for its size — noticeably better than Piper
- No voice cloning, uses preset voices
- Default voice: `af_heart` (American female)
- Install: `pip install kokoro-tts`
- Requires model files downloaded separately (~300MB total)

---

## Setup

### 1. Install dependencies

```bash
brew install whisper-cpp ollama
pip install kokoro-tts sounddevice soundfile requests
```

### 2. Download Whisper model

```bash
whisper-cpp-download-ggml-model small
```

### 3. Download Kokoro model files

```bash
curl -L -o kokoro-v1.0.onnx https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
```

### 4. Pull LLM

```bash
ollama pull llama3.2
```

---

## Usage

### Start

```bash
# Terminal 1
ollama serve

# Terminal 2
python chatbot.py
```

### Stop

`Ctrl+C` in both terminals.

---

## Core Script — `chatbot.py`

```python
import subprocess
import requests
import sounddevice as sd
import soundfile as sf
import tempfile
import os

WHISPER_CLI = "./whisper.cpp/main"
WHISPER_MODEL = "./models/ggml-small.bin"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

def record_audio(duration=5, samplerate=16000):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, audio, samplerate)
    return tmp

def transcribe(wav_path):
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path, "--no-timestamps", "-nt"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def ask_llm(text):
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": text,
        "stream": False
    })
    return response.json()["response"]

def speak(text):
    subprocess.run(
        f'echo "{text}" | kokoro-tts - --stream --voice af_heart',
        shell=True
    )

while True:
    wav = record_audio(duration=5)
    text = transcribe(wav)
    os.remove(wav)
    if not text:
        continue
    print(f"You: {text}")
    response = ask_llm(text)
    print(f"Bot: {response}")
    speak(response)
```

---

## Known Limitations

- **Fixed recording window** — currently records for a fixed 5 seconds. Push-to-talk or VAD (voice activity detection) would be a better UX improvement for v2.
- **Kokoro cold start** — ONNX runtime takes a few seconds to initialize on first call per session.
- **No conversation memory** — each turn is sent to the LLM independently. Adding a message history buffer would make conversations feel more natural.
- **Single voice only** — no voice cloning or custom voices.

---

## Future Improvements (v2)

- Push-to-talk trigger via keyboard shortcut
- Conversation history / memory passed to LLM
- Swap Whisper.cpp for Voxtral Transcribe Mini for higher STT accuracy
- Swap Llama 3.2 3B for a larger model if RAM allows
- Persistent Kokoro server mode to eliminate cold start latency
- Wake word detection ("hey bot")