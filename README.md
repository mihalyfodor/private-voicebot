# Voicebot

A local voice assistant with memory, real-time tools and an animated Live2D avatar (Haru, an office assistant) that lip-syncs to speech. Runs entirely on your machine — no cloud LLM.

**Stack:** Whisper (STT) → oMLX/Gemma 4 (LLM) → Kokoro (TTS) → Live2D avatar in the browser

**Built-in tools:** current time, weather (open-meteo), BBC world news headlines + article detail

## Requirements

- macOS or Linux (WSL not supported — audio passthrough too unreliable)
- [oMLX](https://github.com/jundot/omlx) serving an OpenAI-compatible API on `http://localhost:8000/v1` with Gemma 4 loaded

## Setup

**macOS**
```bash
brew install whisper-cpp portaudio
```

**Linux**
```bash
apt install portaudio19-dev
# Build whisper-cpp from source: https://github.com/ggerganov/whisper.cpp
```

**Both**
```bash
# Download Whisper model
mkdir -p ~/models/whisper
curl -L -o ~/models/whisper/ggml-small.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin

# Download Kokoro models
mkdir -p ~/models/kokoro
curl -L -o ~/models/kokoro/kokoro-v1.0.onnx \
  https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
curl -L -o ~/models/kokoro/voices-v1.0.bin \
  https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Browser libs + Live2D Haru model (not committed — Live2D license)
scripts/fetch_assets.sh

# Config
cp .env.example .env   # set OMLX_MODEL to the id shown by: curl http://localhost:8000/v1/models
```

## Run

**Voice assistant:**
```bash
source .venv/bin/activate
python3 chatbot.py    # opens http://localhost:8010 automatically
```

Press **Space** or click the button to start/stop recording. Use **Shut down** in the browser to exit cleanly. Audio plays in the browser; Haru's mouth follows the audio and her expression follows the `[emotion]` tag the LLM prefixes each reply with (`neutral`, `happy`, `thinking`, `surprised`, `apologetic`). Tune the mapping in `CONFIG` at the top of `static/app.js`.

**Email triage dashboard:**
```bash
source .venv/bin/activate
python3 workflows/email_classification.py   # opens browser automatically
```

Fetches your Gmail inbox, classifies each email via local LLM (category, urgency, confidence), and streams results to a browser dashboard. Click ▲/▼ to adjust urgency or the category badge to change category — corrections are remembered and applied automatically on future runs. Requires Gmail API credentials (`credentials.json`) in the project root.

## Weather configuration

Weather uses [open-meteo](https://open-meteo.com) (no API key). Set your location via env vars before running:

```bash
export LOCATION_NAME="Seychelles"
export LOCATION_LAT=-4.6796
export LOCATION_LON=55.4920
export LOCATION_TIMEZONE="Indian/Mahe"
```

Defaults to Seychelles if unset. Enjoy the tropical weather reports.

## Memory

Conversations are summarized and saved to `shortmem.txt` on exit. This file is loaded on next startup as background context. It is gitignored — personal to your machine.

## Development process

This project follows the [side-project-sdlc](https://github.com/mihalyfodor/side-project-sdlc) — a lightweight 3-phase process (Exploration → Implementation → Refactoring) for building personal apps with AI assistance.
