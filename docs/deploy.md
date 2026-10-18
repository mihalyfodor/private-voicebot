# Deploy — Docker as prod (PRD 40)

Prod is a pinned image built from `main`, running against one data folder; dev keeps running from the
checkout (`.venv`, `PORT=8011`, vault `~/llm-wiki-dev`, which is also the code default; only the container sets `WIKI_PATH` to the real vault). oMLX and the ASR service stay native on the host.

## What is where

```
~/.voicebot/data/          DATA — the only thing to back up
├── llm-wiki/                  the vault (its own git repo; Obsidian name unchanged)     WIKI_PATH
├── memory/                    memory.json, memory_ops.jsonl, shortmem.txt              MEMORY_DIR
├── settings.json                                                                       SETTINGS_PATH
├── .env                       secrets + config (env_file of the container)
├── secrets/                   credentials.json, token.json (first OAuth on the host)   GMAIL_DIR
└── logs/events.jsonl           the event trace, prod profile: never text (PRD 49)          TRACE_DIR
~/.voicebot/backups/       prod.sh backup output
~/models/kokoro            mounted read-only at /models/kokoro (not backed up; re-downloadable)
```

Inside the container everything is under `/data`; `llm-wiki/raw/.index/` (chunk cache) and `llm-wiki/.index/` (the live search index, PRD 53: `search.sqlite` + `-wal`/`-shm`) are
excluded from backups (both are rebuilt from the vault). The container is read-only, non-root (uid 1000), loopback-only on `:8010`, and reaches the host
through `host.docker.internal` (oMLX `:8000`, ASR `:8020`).

## First-time setup

```bash
# 1. host services (Apple-only pieces): oMLX on :8000 as today, plus the ASR service
scripts/install_launchd.sh asr          # renders scripts/launchd/com.voicebot.asr.plist with this checkout's paths, copies + loads it
scripts/install_launchd.sh backup       # daily 03:30 backup
scripts/install_launchd.sh remarkable   # optional: push cards to the reMarkable (:8021); re-run any of these if the checkout moves
# then add REMARKABLE_PUSH_URL=http://host.docker.internal:8021 to $DATA/.env (unset = the push feature is off)
curl -s localhost:8020/health                                        # {"ok":true,...}

# 2. move the state into the data folder (refuses if anything is already there; .env is copied)
scripts/prod.sh migrate            # moves ~/llm-wiki, memory files, settings.json, secrets, session.log

# 3. build and start
scripts/prod.sh build              # tags voicebot:<git describe> and voicebot:latest; the image bakes the tag in as VOICEBOT_VERSION (the trace's start event)
scripts/prod.sh up                 # http://localhost:8010 (healthcheck: `docker compose ps`)
```

## Before a prod deploy

1. On `main`, `git status` clean, `scripts/check.sh` green in this session (quote the counts); `git push origin main` (the user pushes; the AI never pushes).
2. `scripts/prod.sh status` prints `version=<running commit> busy=false` and no "backup agent not loaded" warning (else `scripts/install_launchd.sh backup`, or accept hand-run backups).
3. `scripts/prod.sh deploy` (the user runs it): backup (`FORCE=1`), build (refuses a `-dirty` tag), refuses while `busy`, `down`, `up --wait`, checks that `/api/config` `version` equals the built tag, then `trace_report.py --since 10m --errors`. Stops at the first failing step and names it.
4. `scripts/prod.sh turn` sends one typed turn ("what time is it" unless given) and exits 0 when it completes. It lands in the real session and memory, which is why the default text is content-free.
5. Update `docs/roadmap.md` with the deployed commit.

Rollback: `TAG=<previous commit> scripts/prod.sh up` (the previous image is still on the machine; `docker images voicebot`).

## Daily

The whole stack in one go (PRD 41) — a checklist, not a supervisor: running things are left alone.

```bash
scripts/voicebot start        # colima → oMLX (~/.omlx/bin/omlx start) → the model (admin API load) → ASR (launchd) → container; then the table
scripts/voicebot status       # the table only: colima / omlx / model / asr / app
scripts/voicebot stop         # container down, ASR service unloaded, the model unloaded from oMLX (oMLX itself keeps running)
scripts/voicebot stop --all   # ... and `omlx stop` + `colima stop`
scripts/voicebot start --dev  # skips colima + the container and looks at the dev app on :8011 (does not start it)
```

`scripts/prod.sh start` / `stop` are the same commands. The model load/unload goes through the oMLX admin API: the
`OMLX_API_KEY` bearer is tried first, then `/admin/api/login` with `OMLX_ADMIN_PASSWORD` when it is set in `.env`; when
neither is accepted the script prints the dashboard URL and carries on (oMLX loads the model on the first request anyway;
`stop` then leaves it loaded and says so). `START_HINT` (default `scripts/voicebot start`) is what the UI banner tells the
user to run when a service is down.

| command | what it does |
|---|---|
| `scripts/prod.sh up` / `down` | start / stop the container; `up` returns once the healthcheck passes (up to 150 s) (`restart: unless-stopped` survives reboots; `down` waits up to 90 s, `stop_grace_period`, so the shutdown memory save can finish) |
| `scripts/prod.sh logs` | follow the container log |
| `scripts/trace_report.py $DATA/logs/events.jsonl --since 24h` | turns, ingests and errors from the event trace (PRD 49; `--errors` exits 1 when any) |
| `scripts/prod.sh status` | `compose ps`, waits up to 60 s for the app, then `version=<commit> busy=<true/false>` + the `/api/config` JSON; warns when the backup launchd agent is not loaded |
| `scripts/prod.sh deploy` | the mechanical part of "Before a prod deploy" in one run (backup, build, busy check, down, up, version check, error trace) |
| `scripts/prod.sh turn [text]` | one headless typed turn over the WebSocket (`scripts/turn.py`); exit 1 on an `error` message or after 180 s |
| `scripts/prod.sh build && scripts/prod.sh up` | roll a new `main` into prod |
| `scripts/prod.sh backup` | `git bundle --all` of the vault + `tar.gz` of `DATA` into `~/.voicebot/backups/`; keeps the newest `KEEP` (default 7) of each; skips when the vault head and the memory + settings contents are unchanged (`.last-state`); the tgz contains `.env` and `secrets/`, so keep backups on this machine; refuses below `MIN_FREE_MB` free or above `MAX_TOTAL_MB` total; one line per run in `DATA/logs/backup.log` |
| `scripts/prod.sh restore <tgz> <bundle> <dir>` | untar into `<dir>`, verify the bundle, serve it from a throwaway container on `:8012`, then tear it down |

Env knobs: `DATA` (default `~/.voicebot/data`), `TAG` (image tag), `PROD_PORT` (host port),
`PROJECT` (compose project), `BACKUP_DIR`, `RESTORE_PORT`, `KEEP` (7), `MIN_FREE_MB` (2048),
`MAX_TOTAL_MB` (5120), `FORCE=1` (back up even if nothing changed).

## When something is down

The app probes oMLX (`/health`, then `/v1/models` for the configured model) and the ASR service (`/health`) every 30 s
and right after a failed call. A down service is an amber banner at the top of the workspace naming it ("LLM offline
(oMLX :8000)", "voice input offline (ASR :8020)") with a copy button for the start command; typed and spoken turns are
refused with the same hint while the LLM is down (the wiki and sources keep working). Settings › Services lists each
service with a dot and its last check time. When the app itself is gone the page shows the red "backend is not running"
banner and keeps retrying the socket.

## Notes

- `DATA` must live under your home directory: colima only maps bind-mount ownership there, so the
  container's uid 1000 can write to it.
- If `host.docker.internal` does not resolve, use the colima VM gateway `192.168.5.2` in `docker-compose.yml`
  (`OMLX_BASE_URL`, `ASR_URL`, and `REMARKABLE_PUSH_URL` in `$DATA/.env`).
- Gmail: run the first OAuth on the host (`credentials.json` in `DATA/secrets`, then any `get_emails` from
  the checkout with `GMAIL_DIR=~/.voicebot/data/secrets`); the container only refreshes the token.
- The compose file pins `PORT=8010` inside the container and publishes it on `127.0.0.1:$PROD_PORT`;
  `ALLOWED_ORIGINS` is set to match so the browser's Origin check passes on any host port.
