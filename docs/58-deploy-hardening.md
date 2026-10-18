Status: implementation

# 58 — Deploy hardening: a status that tells the truth, backups that include memory, a headless typed turn

## Overview
The 2026-10-18 deploy (ad33ad4 → 6ac666e) worked but exposed six small holes in `scripts/prod.sh` and the checklist in
`docs/deploy.md`: `status` right after `up` fails with "Empty reply from server" because the container is still
starting; the running commit and "is a turn in flight" can only be read from `events.jsonl`; the "one typed turn" step
has no tool, so a throwaway WebSocket client was written and its first, malformed message was dropped silently by the
server; the daily backup skips whenever the vault head is unchanged even if memory.json changed that day; and the backup
launchd agent has been missing since the first setup without anything noticing. Design source: the deploy-path review of
2026-10-18 (Opus reviewer, read-only). The environment keeps its name ("prod") and the user runs the script.

## User stories
- As the person deploying, `prod.sh status` tells me the live commit and whether the app is idle, and never fails just because I ran it too early.
- As the person deploying, one command sends a harmless typed turn and exits non-zero if the turn did not complete.
- As the only user, a day of conversations without a wiki commit still ends with my memory in the nightly backup.
- As the only user, I am told when the nightly backup agent is not loaded instead of discovering it weeks later.
- As the person deploying, the checklist is one command for the mechanical part, and the human steps come first.

## Goals / Non-goals
- Goals: `status` and `up` wait for health; `/api/config` carries `version` and `busy`; `prod.sh turn`; unknown WebSocket
  actions answer with an error; backup skip logic sees memory and settings; a warning when the backup agent is not loaded;
  `prod.sh deploy` chains the mechanical steps; deploy.md and roadmap §8 updated.
- Non-goals: no rename of "prod"; no OCI image label; no machine-checked deploy record; no supervisor or rollback tooling
  (rollback is documented as `TAG=<old> scripts/prod.sh up`); no consistent-snapshot tar.

## Technical approach

### 1. `up` and `status` wait for health (`scripts/prod.sh`)
- `cmd_up`: `compose up -d --wait --wait-timeout 150` (Compose ≥ 2.20 supports it; the healthcheck's `start_period` is 90 s).
  Prints the URL only after the wait; a timeout exits 1 with `compose ps` output.
- `cmd_status`: `compose ps`, then `wait_for "$URL/api/config" 60` before the curl. Prints `version=<tag> busy=<true|false>`
  on one line (parsed with python3 from the JSON) and then the raw JSON as today.

### 2. `version` and `busy` in `/api/config` (`server/app.py`)
- `api_config()` adds `"version": trace._version()` and `"busy": panels.is_busy()`. `_version()` already reads
  `VOICEBOT_VERSION` (the start event uses it); `is_busy()` is the existing turn-in-flight flag.
- The idle check for a deploy is `busy == false`; the shutdown memory save is already covered by `stop_grace_period` 90 s.

### 3. `prod.sh turn "<text>"` (`scripts/prod.sh` → `scripts/turn.py`)
- A ~40-line client lifted from `tests/prod/test_docker.py::Socket`: connect to `ws://127.0.0.1:$PROD_PORT/ws`, drain to
  idle, send `{"action": "text_turn", "text": ...}`, answer every `speech_end` with `playback_done` (without it the
  server never returns to idle, which is why today's throwaway client timed out), stop at the first `state: idle` after
  a `state: thinking`, print the assistant `transcript` text, exit 0. Exit 1 on an `error` message or after `--timeout`
  (default 180 s). Runs with the checkout's `.venv` python (websockets is already a dependency).
- Default text when none is given: `what time is it`. The turn lands in the real session and the disconnect triggers a
  memory save, so the default is deliberately content-free. deploy.md says so.

### 4. Unknown WebSocket actions answer (`server/ws.py`)
- When `ACTIONS.get(msg.get("action"))` is None: `trace.error("ws", "ws", ValueError(f"unknown action {action!r}"), fatal_for="none")`
  and `panels._error("settings", f"unknown action {action!r}", action=action)`, then continue the loop. A frame with no
  `action` key reads `None`. `docs/protocol.md` gets one row for the behaviour; `tests/test_protocol_doc.py` keeps the table honest.

### 5. Backup skip sees memory and settings (`scripts/prod.sh`, `cmd_backup`)
- The marker `.last-head` becomes `.last-state`, holding `<vault head> <sha256 of memory/* + settings.json contents>`
  (`cat "$DATA"/memory/* "$DATA/settings.json" 2>/dev/null | shasum -a 256`). Skip only when the whole line is unchanged.
  `FORCE=1` and the log line (`skip "state unchanged"`) keep working; an old `.last-head` is simply ignored (one extra backup).

### 6. Warn when the backup agent is not loaded (`scripts/prod.sh`)
- `cmd_backup` (when not running under launchd: `[ -z "${XPC_SERVICE_NAME:-}" ]`) and `cmd_status` run
  `launchctl list com.voicebot.backup >/dev/null 2>&1 || echo "warning: backup agent not loaded — scripts/install_launchd.sh backup" >&2`.
  On a non-mac host (no `launchctl`) the check is skipped. A warning, not a failure: backups by hand still run.

### 7. `prod.sh deploy` (`scripts/prod.sh`)
- In order: refuse unless `git status --porcelain` is empty and the branch is `main`; `cmd_backup` with `FORCE=1`;
  `cmd_build`, refusing a `-dirty` tag; `busy` must be `false` (via `/api/config`; a container that is down counts as idle);
  `compose down`; `cmd_up`; `version` from `/api/config` must equal the built tag; `scripts/trace_report.py "$DATA/logs/events.jsonl" --since 10m --errors`.
  The typed turn stays a separate command (`turn`) because it writes to the session. Any failing step stops the run with
  the step name on stderr. `check.sh` is not run by `deploy`: it is step 1 of the checklist and quoted by the human.

### 8. Docs
- `docs/deploy.md` "Before a prod deploy" becomes: 1 check.sh green + push (human), 2 backup agent loaded (`status` warns),
  3 `scripts/prod.sh deploy`, 4 `scripts/prod.sh turn`, 5 roadmap line. The command table gains `turn` and `deploy`,
  says `status` waits up to 60 s, notes the tgz contains `.env` and `secrets/`, and documents rollback as `TAG=<old> scripts/prod.sh up`.
- `docs/roadmap.md` §8: the buried "backup launchd job is not yet loaded" becomes its own `USER TODO:` line; the stale
  "Do this after PRD 49" sentence goes.

## Scenarios
1. `prod.sh up` returns only once the container is healthy; `status` right after prints version and busy, no curl error.
2. `/api/config` carries `version` equal to `VOICEBOT_VERSION` and `busy` false when idle, true during a turn.
3. `prod.sh turn` against a fake server that sends thinking → speech_end → idle exits 0 and prints the transcript; it answers `playback_done`.
4. `prod.sh turn` against a fake server that sends an `error` exits 1; a server that never reaches idle exits 1 at the timeout.
5. A WebSocket frame `{"type": "text_turn"}` gets an `error` message naming the unknown action and an `error` trace event; the socket stays open.
6. Backup: unchanged vault head but a changed memory.json → a new archive; unchanged both → skip; `FORCE=1` → archive.
7. Backup with `launchctl` reporting no `com.voicebot.backup` → the archive is written and a warning is printed; under `XPC_SERVICE_NAME` no warning.
8. `deploy` refuses on a dirty tree, on a `-dirty` tag, and when `busy` is true; otherwise runs the chain and stops at the first failing step.
9. `docs/index.md` lists this PRD and `scripts/turn.py`.

## Tests
- `tests/prod/test_prod_script.py` (existing bash-subprocess style, `PATH` shims for `docker`, `curl`, `launchctl`):
  `test_backup_runs_when_memory_changed_with_same_head` (6), `test_backup_skips_when_nothing_changed` (6),
  `test_backup_warns_when_agent_not_loaded` (7), `test_backup_quiet_under_launchd` (7),
  `test_deploy_refuses_dirty_tree_and_dirty_tag` (8), `test_deploy_refuses_when_busy` (8), `test_prod_script_parses` (existing).
- `tests/prod/test_turn_script.py`: an in-process `websockets` server fixture; `test_turn_completes_and_answers_playback_done` (3),
  `test_turn_exits_1_on_error` (4), `test_turn_exits_1_on_timeout` (4).
- `tests/server/test_service_health.py` (already exercises `/api/config`): `test_api_config_has_version_and_busy` (2).
- `tests/server/test_error_scopes.py` (error scope tests): `test_unknown_action_gets_error_and_trace_event` (5).
- `tests/test_protocol_doc.py`, `tests/test_index.py` unchanged, must stay green (4, 9).
- Gate: `scripts/check.sh`; smoke not required (no LLM path or UI change). Scenario 1 is verified by hand at the next real deploy.

## Out of scope
- Renaming the environment (decided 2026-10-18: it stays "prod"; the user runs the script).
- OCI `org.opencontainers.image.revision` label; `version` in `/api/config` covers it.
- A machine-checkable deploy record; the roadmap sentence is history.
- Consistent-snapshot tar of a live `DATA`; rollback automation; running `check.sh` from `deploy`.
- Changing the trace `profile` value or `PROD_*` env names.

## Decisions
- Chose `compose up --wait` over a hand-rolled poll: the healthcheck already exists, one flag uses it.
- Chose `/api/config` for version and busy over reading `events.jsonl`: one curl, works from any shell, no log parsing.
- Chose a separate `turn` command over folding it into `deploy`: the turn writes to the session and memory; the person decides when.
- Chose a content hash of memory + settings for the backup marker over mtimes: a save that rewrites identical content does not force a backup.
- Chose a warning for the missing agent over a hard failure: a hand-run backup must never be refused for a launchd reason.

## Risks
- `--wait` needs Compose ≥ 2.20; the installed version is 5.6. Mitigation: `tests/prod/test_compose_file_validates` already requires a daemon; the flag is checked there.
- `turn` leaves a "what time is it" turn in the real session. Mitigation: documented; it is also the smallest possible turn.
- `shasum` over `memory/*` costs nothing at this size but would need a cap if memory grew large. Accepted.

## Size
S + S + S–M + S + S + S + S (about 60 lines of bash, 40 of python for `turn.py`, 10 in server code, 12 tests).

## Files
`scripts/prod.sh`, `scripts/turn.py` (new), `server/app.py`, `server/ws.py`, `docs/protocol.md`, `docs/deploy.md`, `docs/roadmap.md`,
`tests/prod/test_prod_script.py`, `tests/prod/test_turn_script.py` (new), `tests/server/test_service_health.py`, `tests/server/test_error_scopes.py`;
`docs/index.md` gains this PRD and `scripts/turn.py` (enforced by `tests/test_index.py`).
