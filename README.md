# Voicebot

A local voice assistant with memory, a markdown wiki, real-time tools and an animated Live2D avatar that lip-syncs to speech. Runs on your machine, no cloud LLM.

**Stack:** Silero VAD + Whisper speech recognition (`mlx-whisper`) -> oMLX / Gemma 4 (OpenAI-compatible API on `http://localhost:8000/v1`, key `omlx`) -> Kokoro TTS -> Live2D avatar in the browser. Dev serves on port 8011, the prod container on 8010.

Speech recognition has two backends (`ASR_BACKEND` in `.env.example`): `mlx` runs `mlx-whisper` inside the app (the code default, used in dev); `remote` posts audio to a host ASR service, `scripts/asr_server.py` on :8020 (what the prod container uses).

**Avatar:** Wanko, an upbeat dog office assistant, with her own voice, persona and expressions. The menu lets you pick a backdrop behind her.

**Tools:** time, weather (open-meteo), BBC news headlines and article detail, Gmail inbox, Wikipedia, and the wiki (search, read page, read source, file a source). Each plays a short spoken filler the moment it is called. Typed or spoken commands: `/new` (new topic), `/brief` (what changed), `/write` (write the answer up), `/research <question>`, "save that", "find <page>".

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

## Weather configuration

Weather uses [open-meteo](https://open-meteo.com), no API key. Set the location in `.env` (default Seychelles): `LOCATION_NAME`, `LOCATION_LAT`, `LOCATION_LON`, `LOCATION_TIMEZONE`.

## Memory

The assistant keeps a structured user profile, not a transcript. Every `MEMORY_SAVE_EVERY` completed turns (default 5) and on shutdown, the session is folded into the profile: the model proposes operations, Python validates and applies them, and a reflection pass cleans up every few saves. Stale entries decay. The profile lives in `memory.json` with its change log `memory_ops.jsonl` under `MEMORY_DIR` (the repo root in dev, `~/.voicebot/data/memory` in prod). It is injected into the system prompt as a `<user_profile>` block (token budget `MEMORY_BUDGET_TOKENS`) and is editable in the Memory view. `shortmem.txt` is the old format, read once to migrate. All of it is gitignored and personal to your machine.

## Wiki / vault

The wiki is an Obsidian-style markdown vault at `WIKI_PATH` (dev default `~/llm-wiki-dev`; prod sets it explicitly, so dev never touches the real vault). The app:

- edits it in the Wiki view;
- ingests documents into it: drop a PDF, DOCX and similar files in Sources, sync reMarkable notebooks, or file a folder with `scripts/ingest_batch.py`;
- searches it with a SQLite BM25 index at `<vault>/.index/` (rebuild with `python -m vault reindex`, only while the app is stopped);
- uses it for recall in chat when a question calls for it.

## Prod (Docker)

Prod is a pinned image built from `main`, with all state under `~/.voicebot/data` and the UI on `http://localhost:8010`. oMLX and the ASR service stay native on the host (ASR via launchd, `scripts/install_launchd.sh asr`).

```bash
scripts/prod.sh build | up | down | status | logs | backup | restore <tgz> <bundle> <dir> | deploy | turn [text]
scripts/voicebot start | stop | status   # the whole stack: colima, oMLX, model, ASR service, container (--dev for the checkout)
```

Setup, migration and the release checklist are in [docs/deploy.md](docs/deploy.md).

## Development

```bash
scripts/check.sh      # the only check command: node --check, DOM fixture, ruff, pytest (no oMLX needed)
scripts/smoke.py      # headless smoke: e2e, one live turn, screenshots to /tmp/smoke/
git config core.hooksPath scripts/hooks   # once per clone (commit-msg hook)
```

Layers, each with its tests under `tests/<layer>/`: `core/` (config, LLM client, tracing), `vault/` (the wiki and its index), `memory/` (profile), `ingest/` (documents in), `tools/` (the model's tools), `agent/` (turns, recall, fold), `server/` (FastAPI and WebSocket). The browser client is `static/`, helper scripts are in `scripts/`. Logs go to `logs/events.jsonl` in dev (read with `scripts/trace_report.py`).

[docs/index.md](docs/index.md) has the full folder tree, all test commands (live, e2e, docker, perf) and the WebSocket protocol link. [docs/process.md](docs/process.md) is the process.

## Also in the repo

`scripts/email_classification.py`: a standalone Gmail triage dashboard (local LLM classifies, you correct). Needs `credentials.json` in the project root.

## Licensing

The Live2D sample model (Wanko), Cubism Core and the backdrop images are downloaded by `scripts/fetch_assets.sh` and are gitignored, not committed. Backdrops are Pixabay Content License images; credit is shown in-app.

## Development process

This project follows the [side-project-sdlc](https://github.com/mihalyfodor/side-project-sdlc), a lightweight 3-phase process (Exploration, Implementation, Refactoring) for building personal apps with AI assistance. See [docs/process.md](docs/process.md).
