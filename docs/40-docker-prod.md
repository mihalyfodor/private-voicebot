Status: done

# 40 — Docker as the prod environment, one data folder

## Overview
Prod runs as a pinned container built from `main`; dev keeps running from the checkout. All state lives in one
folder, `~/docker-data/voicebot/`, which is the single thing to back up; code stays in git; models stay in
`~/models`. The Apple-only pieces — oMLX and mlx-whisper — stay native on the host as services the container calls.
Runtime: colima (Docker Engine 29.x).

## User stories
- As a user, `scripts/prod.sh up` gives me a stable Wanko on :8010 while I break things on :8011 from the checkout.
- As a user, backing up `~/docker-data/` is the whole job; `prod.sh backup` / `restore` prove it round-trips.
- As a user, voice keeps its native speed: ASR runs on the host, the container only talks to it.
- As a developer, every state path has an env knob and nothing is written next to the code.

## Data folder (`DATA=~/docker-data/voicebot`)
```
llm-wiki/        WIKI_PATH  — the existing vault, moved here (its own git repo; Obsidian vault name unchanged)
memory/          MEMORY_DIR — memory.json, memory_ops.jsonl, shortmem.txt
settings.json    SETTINGS_PATH
.env             env_file (secrets + config)
secrets/         GMAIL_DIR — credentials.json, token.json (first OAuth on the host, token refresh writes back)
logs/            SESSION_LOG — session.log
```
Models stay outside: `~/models/kokoro` mounted read-only at `/models/kokoro`. Not backed up: `llm-wiki/raw/.index/`
(chunk cache, regenerable) — excluded from the tarball but harmless in Time Machine.

## Technical approach
- **Env plumbing** (works for host runs too): new `GMAIL_DIR` (tools/gmail.py), `SESSION_LOG` (llm.py),
  `ALLOWED_ORIGINS` (comma list; `_origin_allowed` accepts them in addition to the loopback:PORT pair).
  `.env.example` documents the data-folder layout.
- **Host ASR service**: `scripts/asr_server.py` — a tiny FastAPI app wrapping `asr.transcribe`: `POST /transcribe`
  (wav bytes) → the same `{text, segments}` JSON; `GET /health`; binds `127.0.0.1:8020`. `scripts/launchd/
  com.voicebot.asr.plist` (+ a note on oMLX being started the same way). `asr.py` gets `ASR_BACKEND=mlx|remote`
  (`ASR_URL`, 30 s timeout, a clear error string on failure so a turn degrades to "didn't catch that", never a crash).
- **Portability**: `sources._downscale` uses Pillow with `sips` as a fallback only when Pillow is missing;
  `requirements-docker.txt` = requirements minus mlx-whisper/playwright/pytest, torch from the CPU index.
- **Image**: `Dockerfile` on `python:3.12-slim` + git + curl; `scripts/fetch_assets.sh` runs at build so
  `static/{vendor,models,backdrops}` are baked in; non-root user `app`; `WORKDIR /app`; `CMD python chatbot.py`.
- **Compose** (`docker-compose.yml`): service `voicebot`, `image: voicebot:${TAG}`, `env_file: ${DATA}/.env` plus
  fixed overrides (`HOST=0.0.0.0`, `PORT=8010`, `OPEN_BROWSER=0`, `WIKI_PATH=/data/llm-wiki`, `MEMORY_DIR=/data/memory`,
  `SETTINGS_PATH=/data/settings.json`, `GMAIL_DIR=/data/secrets`, `SESSION_LOG=/data/logs/session.log`,
  `KOKORO_MODEL=/models/kokoro/kokoro-v1.0.onnx`, `KOKORO_VOICES=/models/kokoro/voices-v1.0.bin`,
  `OMLX_BASE_URL=http://host.docker.internal:8000/v1`, `ASR_BACKEND=remote`, `ASR_URL=http://host.docker.internal:8020`),
  `ALLOWED_ORIGINS` for the published port), `ports: 127.0.0.1:${PROD_PORT:-8010}:8010`, volumes `${DATA}:/data`,
  `~/models/kokoro:/models/kokoro:ro`, `extra_hosts: host.docker.internal:host-gateway`, `read_only: true` +
  `tmpfs: /tmp:exec` (espeak-ng, Kokoro's phonemizer, dlopens a library it extracts there), `user: "1000:1000"`,
  `security_opt: no-new-privileges`, memory 4g / cpus 4, healthcheck GET `/` (start_period 90 s),
  `restart: unless-stopped`. `PROD_PORT` / `PROJECT` / `TAG` let tests and `restore` run throwaway copies.
- **`scripts/prod.sh`**: `build` (tag from `git describe --always`, also `latest`), `up`, `down`, `logs`, `status`,
  `migrate` (one-time: `mv ~/llm-wiki`, memory files, settings.json, secrets, session.log into `$DATA` — `.env` is
  *copied* so the dev checkout keeps its config — then symlink-free; prints what moved and refuses if `$DATA` exists), `backup` (`git bundle --all` of the vault + `tar czf` of `$DATA` excluding
  `raw/.index`), `restore <archive> <bundle> <dir>` (untar, `git bundle verify`, start a throwaway container on 8012
  against it and curl `/api/config`).
- Dev unchanged: `.venv`, `PORT=8011 WIKI_PATH=~/llm-wiki-dev` from the checkout (host MLX ASR in-process).
- Docs: docs/index.md "How to run — prod" section; `docs/deploy.md` (~40 lines: first-time setup, migrate, daily
  commands, backup/restore, what is where).

### Decisions
- ASR stays native via a host service (user decision): no CPU-whisper compromise; the container is pure app.
- Vault keeps its name `llm-wiki` inside the data folder (Obsidian registration stays valid).
- Straight migration script over by-hand moves (user decision); refuses to overwrite.
- Loopback-only publish; no egress proxy; oMLX not containerised.
- Assume colima exposes `host.docker.internal` via `host-gateway` (verified: 192.168.5.2 reaches host loopback);
  if not, fall back to the colima VM gateway IP documented in deploy.md.
- `DATA` must live under `$HOME`: colima maps bind-mount ownership only there, so uid 1000 can write to it.

## Test scenarios
1. With `MEMORY_DIR`, `SETTINGS_PATH`, `SESSION_LOG`, `GMAIL_DIR`, `WIKI_PATH` pointing at a temp data dir, a turn +
   a memory save + a wiki op write only under that dir; nothing appears in the repo root (conftest guard extended).
2. `_origin_allowed`: `ALLOWED_ORIGINS=http://localhost:8011` accepted, `http://evil:8010` rejected, loopback:PORT
   still accepted with the variable unset.
3. `asr_server` (TestClient): a fixture wav → `{text, segments}`; `ASR_BACKEND=remote` in `asr.transcribe` posts the
   bytes and returns the same shape; server down → error string (printed as `[ASR] …`), no exception, the turn path
   takes the "nothing heard" route (processing → idle, no transcript, no error message).
4. `_downscale` shrinks a > threshold PNG with Pillow on any platform (no `sips` needed).
5. `prod.sh migrate` in a temp HOME: moves vault/memory/settings/.env/secrets into `$DATA`, prints the moves,
   refuses to run twice; `bash -n` on the script; `docker compose config` validates the compose file.
6. Docker (marker `docker`, skipped without a daemon): `prod.sh build && up` → healthcheck healthy within 90 s;
   `docker exec id -u` ≠ 0; writing to `/app` fails; `/api/config` shows `vault: /data/llm-wiki`.
7. Docker live: a typed turn answers via host oMLX; a wiki save from the UI commits into `$DATA/llm-wiki` (checked
   with `git log -1` on the host); memory entry survives `docker compose restart`; a spoken turn (wav pushed through
   the websocket audio path) transcribes via the host ASR service.
8. `prod.sh backup` → `restore` into a temp dir: bundle verifies, restored vault `git log` matches, the throwaway
   container on 8012 serves `/`.

## Out of scope
Egress proxy, LAN exposure, oMLX in Docker, Gmail OAuth inside the container, remote backup targets, CI builds,
CPU whisper fallback, multi-arch images.
