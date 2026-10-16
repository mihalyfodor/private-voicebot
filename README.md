# Voicebot

A local voice assistant with memory, real-time tools and an animated Live2D avatar that lip-syncs to speech. Runs entirely on your machine — no cloud LLM.

**Stack:** Whisper (`mlx-whisper`, in-process STT) + Silero VAD → oMLX/Gemma 4 (OpenAI-compatible API on `http://localhost:8000/v1`, key `omlx`) → Kokoro (TTS) → Live2D avatar in the browser, served on port 8011 (the prod container owns 8010).

**Avatar:** **Wanko**, an upbeat dog office assistant, with her own voice, persona and expressions. The menu also lets you pick a **backdrop** behind her.

**Built-in tools:** current time, weather (open-meteo), BBC world news headlines + article detail, Gmail inbox — each plays a short spoken filler ("Let me check outside...") the moment it's called, so the wait feels natural.

## Requirements

- macOS or Linux (WSL not supported — audio passthrough too unreliable)
- [oMLX](https://github.com/jundot/omlx) serving an OpenAI-compatible API on `http://localhost:8000/v1` with Gemma 4 loaded

## Setup

**Apple Silicon Mac required** for speech recognition (`mlx-whisper`; the model `mlx-community/whisper-small-mlx` downloads on first run). **Chrome** is the target browser: microphone capture happens in the page (echo cancellation / noise suppression), so allow the mic permission when prompted.

```bash
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

# Browser libs, Live2D avatar models and backdrop images (not committed — see Licensing)
scripts/fetch_assets.sh

# Config
cp .env.example .env   # set OMLX_MODEL to the id shown by: curl http://localhost:8000/v1/models
```

## Run

**Voice assistant:**
```bash
source .venv/bin/activate
python3 chatbot.py    # opens http://localhost:8011 automatically
#   OPEN_BROWSER=0 (env or .env) starts the server without opening a tab
```

Type in the composer (**Enter** sends, **Shift+Enter** newline) or turn the mic on with the composer's mic button / the **Listening** switch in Settings to just talk: a VAD detects when you stop speaking. The mic is gated while the assistant speaks, so it won't answer itself. **Space** mutes the replies (they still arrive and are shown, they just aren't played). The avatar's mouth follows the audio and its expression follows the `[emotion]` tag the LLM prefixes each reply with (`neutral`, `happy`, `thinking`, `surprised`, `apologetic`). Hover the avatar in Chat for the backdrop thumbnails; the **Settings** view (rail, key 5) has every other control, including a clean shut down.

Wanko is the only avatar; anything else in `AVATAR` or in `settings.json` (gitignored) quietly resolves back to her. Her persona lives in `characters/wanko.yaml`. `KOKORO_VOICE` / `KOKORO_SPEED` in `.env` override the current avatar's default voice/speed if set. Client-side avatar behaviour (model, framing, expression mapping) lives in `PROFILES` at the top of `static/app.js`; emotion→expression tuning is in the same file's `CONFIG`.

**Email triage dashboard:**
```bash
source .venv/bin/activate
python3 scripts/email_classification.py   # opens browser automatically
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

## Licensing

The Live2D sample model (Wanko) and Cubism Core, and the backdrop images, are downloaded by `scripts/fetch_assets.sh` and are gitignored — they are not committed to this repo. Backdrops are Pixabay Content License images; credit is shown in-app.

## Development process

This project follows the [side-project-sdlc](https://github.com/mihalyfodor/side-project-sdlc) — a lightweight 3-phase process (Exploration → Implementation → Refactoring) for building personal apps with AI assistance.
