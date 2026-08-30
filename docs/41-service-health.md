Status: done

# 41 — Service health in the UI; one start/stop for everything

## Overview
Prod is now three host pieces (oMLX, the ASR service, the container) plus the browser. When one is down the UI
shows nothing useful. This PRD adds a server-side health probe pushed to the UI as a banner naming what is down
and the one command that fixes it (copyable), a `voicebot` entry script whose `start` checks what is not running
and starts only that, and a `stop` that stops the container and the models (unloading Gemma from oMLX rather than
killing the server when the admin API allows it).

## User stories
- As a user, if oMLX or ASR is down I see "LLM offline — run `voicebot start`" with a copy button, not a silent
  failure or a spinning turn; when the backend itself is down the page says so and how to start it.
- As a user, `voicebot start` brings up whatever is missing (colima, oMLX, the model, the ASR service, the
  container) and `voicebot stop` frees the GPU/memory (container down, ASR unloaded, Gemma unloaded).
- As the developer, dev on :8011 gets the same banner logic (probes are the same code).

## UI/UX
- A slim banner at the top of the workspace in every view (above the chat column in Chat): red for the backend
  websocket being down ("Wanko's backend is not running — run: `scripts/voicebot start`" + ⧉ copy), amber for
  a degraded service ("LLM offline (oMLX :8000)" / "voice input offline (ASR :8020)" — the app still works for
  what does not need it: typed turns without LLM are refused with the same hint; wiki/sources editing keeps working).
- Disappears within one probe interval after the service is back. Settings › Services row lists each service with
  a green/red dot, last check time, and the same copy button.

## Technical approach
- `services.py`: `probe()` → `{"llm": {...}, "asr": {...}}` each `{ok, label, detail, checked}` (`label` = "oMLX :8000" / "ASR :8020", the piece and its port the banner names); LLM = `GET /health` on
  `OMLX_BASE_URL`'s host then `GET /v1/models` (with the key) and "model loaded" = the configured `OMLX_MODEL` is in the models list (oMLX lists the models it serves; it lazy-loads on the first request); ASR = `GET
  ASR_URL/health` when `ASR_BACKEND=remote`, else `ok` (in-process MLX). Cached; re-probed every 30 s on a daemon
  thread and immediately after any LLM/ASR call fails; a change pushes `services {llm, asr, hint}` over the
  websocket (`hint` = the command to run, from `START_HINT` = `scripts/voicebot start`, overridable by `START_HINT`
  env so the container can say `scripts/voicebot start` too). `/api/config` carries the last probe.
- Turn path: when `llm.ok` is false, a typed/spoken turn gets an immediate `error {scope: turn}` "LLM offline —
  run …" instead of a timeout; the filler/thinking state is not entered.
- Client: `services` message → banner + Settings rows; websocket `onclose` (not code 4000/4003) → the red backend
  banner with the copy button; reconnect attempts continue as today.
- `scripts/voicebot` (bash, in the repo; `scripts/prod.sh` keeps its subcommands and `start`/`stop` delegate here):
  `start`: colima running? (`colima status`) else `colima start`; oMLX responding on :8000? else `~/.omlx/bin/omlx
  start` and wait for `/health`; model loaded? (`GET /admin/api/models` / `/v1/models`) else `POST
  /admin/api/models/<OMLX_MODEL>/load` (admin auth: try the `OMLX_API_KEY` bearer first, then `/admin/api/login`
  with `OMLX_ADMIN_PASSWORD` if set; if neither works print the dashboard URL and continue — oMLX lazy-loads on the
  first request anyway); ASR service healthy on :8020? else `launchctl load` the plist (or `launchctl kickstart`);
  container healthy? else `docker compose up -d`; then print a one-line status table. Idempotent: running things
  are left alone. `stop`: `docker compose down`; `launchctl unload` the ASR plist; `POST /admin/api/models/<model>/
  unload` (fallback: leave oMLX running and say so — never `omlx stop` unless `--all`); `--all` also `omlx stop`
  and `colima stop`. `status`: the table only. `--dev` variant skips the container and targets :8011.
- Tests: `services.probe` with mocked HTTP (up / down / model not loaded); `services` push on state change and
  not on no-change; turn refused with the hint while LLM is down (Panel fixture); node fixture: banner text + copy
  button, backend-down banner on close code 1006, Settings rows; `bash -n scripts/voicebot`; a shell test with
  stubbed `colima`/`omlx`/`launchctl`/`docker`/`curl` on PATH asserting `start` only starts what is down and `stop`
  unloads in order. Docker marker test: banner appears in the container UI when ASR is stopped, disappears after
  it is started (reuse the PRD 40 harness).

### Decisions
- Server probes, UI displays: one source of truth, works for prod and dev, no browser-side host probing.
- Unload over stop for oMLX: frees memory, keeps the admin app and other models available.
- One entry script: `start` is a checklist, not a supervisor — no daemonising, launchd/compose already restart.
- Copy button over auto-run: the browser cannot start host processes; the hint is the honest UX.

## Test scenarios
1. `probe()` reports llm ok/down/model-not-loaded and asr ok/down from mocked responses; `detail` names the port.
2. A state change pushes `services` once; a repeated identical probe pushes nothing; `/api/config.services` matches.
3. LLM down → typed turn gets `error {scope: turn, text contains "run"}` within 1 s, no `thinking` state; spoken
   turn the same; after the probe flips back to ok the next turn proceeds.
4. Fixture: `services` with asr down → amber banner "voice input offline (ASR :8020)" + copy button whose payload is
   the hint; both ok → banner hidden; websocket close 1006 → red backend banner; Settings rows show dots + times.
5. `scripts/voicebot start` with all stubs reporting "running" starts nothing and prints the table; with oMLX and
   the container down it calls `omlx start` and `docker compose up -d` only; `stop` calls compose down, launchctl
   unload, the unload endpoint (recorded by the curl stub); `--all` adds `omlx stop` + `colima stop`.
6. Docker (marker): stop the ASR service → the prod UI shows the amber banner within 60 s; start it → gone.

## Out of scope
Auto-starting host services from the browser, model auto-selection, GPU/memory stats, notifications outside the app.
