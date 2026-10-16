Status: implementation

# 49 — Event trace: structured logs with a prod and a dev profile

## Overview
Today the app logs by accident: `session.log` stores the user's words verbatim with no turn id (`llm._log`,
llm.py:1286-1295) and never rotates, stdout prints whole turns and tool results (llm.py:374, chatbot.py:401/580/
604/626), and nothing records recall scores, fallbacks, model timings, token use, ingest outcomes or tracebacks.
This PRD replaces that with one JSONL event stream (`events.jsonl`) under two clearly separated profiles: `prod`
writes actionable events and never any text; `dev` writes the same events plus a separate verbose file with
prompts, turn text and tool results, so an agent can see what happens while building.

## User stories
- As the owner, I run `scripts/trace_report.py events.jsonl --since 24h` on the prod logs and get one row per turn
  and ingest plus an error list, with nothing I said or read in it.
- As an agent building a feature, I read `verbose.jsonl` in dev to see the exact prompt, recall block and tool
  result of turn 17, joined to its events by `turn_id`.
- As the owner, I cannot start a dev-profile process against my prod data by mistake: it refuses with one line.
- As an agent asked "what went wrong yesterday?", the `/trace` skill tells me to read the report first and raw lines
  only for flagged turns.

## UI/UX
- No app UI change. Startup prints one line: `[Trace] profile=prod events=/data/logs/events.jsonl` (dev adds
  `verbose=…`). stdout stays human-readable with the same `[Module]` prefixes; in prod it carries no turn text.
- `trace_report.py` output: a turn table (`turn_id · ts · origin · band · tools · llm ms / ttft ms · tokens in/out ·
  reply_len · error?`), an ingest table (`ingest_id · kind · title · ms · ops applied/rejected · degraded`), then
  errors (`ts · scope · where · exc type · first traceback line`). `--turn ID` / `--ingest ID` print that unit's
  events one per line; `--errors` prints only the error list and exits 1 when it is non-empty.
- `/smoke` ends with `trace_report.py <out>/events.jsonl --errors` and quotes it in RESULT.txt.

## Technical approach
- **`trace.py`** (new, stdlib): `event(kind, **fields)` adds `ts` (ISO, ms), `kind`, `turn_id` and `ingest_id` from
  contextvars, `profile`; JSON-encodes with `default=str` and writes through a `RotatingFileHandler` (10 MB × 5).
  The writer never raises (a failed write is one stderr line, rate-limited). `verbose(kind, **fields)` writes
  `verbose.jsonl` next to `events.jsonl` (same rotation) and is a no-op in prod — the file is never opened.
  `turn(turn_id)` / `ingest(kind)` are context managers that set the contextvars; the turn controller
  (`TurnController.submit`, session.py, using `Session.turn_id` from session.py:29-34) and every ingest entry point
  (`sources.ingest_upload/notebook/note/path/email/news`, sources.py:1462-1584, plus `batch_import`) use them.
  `TRACE_DIR` defaults to the repo root in dev; compose sets `TRACE_DIR: /data/logs`.
- **Profiles**: `TRACE_PROFILE` = `prod|dev`, default `dev` from a checkout; docker-compose.yml sets
  `TRACE_PROFILE: prod`. `trace.configure()` at startup refuses (exit 2, one line on stderr) when the profile is
  `dev` and `/.dockerenv` exists or any of `DATA`, `WIKI_PATH`, `MEMORY_DIR` resolves under `~/docker-data`; it
  prints the profile line and writes `service {event:"start", profile, version, pid}` as the first event.
- **Event kinds and fields** (prod; never text — titles, paths, scores, lengths, timings, ids only):
  1. `turn_start` origin (spoken/typed/card/briefing), text_len, words
  2. `recall` band, page, hits `[{source, n, score, named}]` ≤ 5, offer (none/deeper/wikipedia/writeup), from
     `_recall_evidence`/`recall_band` (llm.py:867-899)
  3. `tool_call` name, arg_keys, ms, result_len, ok (llm.py:374 loses its result text)
  4. `llm_call` purpose (reply/tool_round/fallback/summary/card/ingest), model, max_tokens, ms, ttft_ms,
     tokens_in, tokens_out (null when the stream gives no usage), fallback_reason (llm.py:1112)
  5. `turn_end` ms, reply_len, tools, band, ctx_pct, outcome (ok/error/cancelled) — replaces session.log
  6. `ingest_start` kind, raw path, bytes, pages
  7. `ingest_end` path, ms, ops_applied, ops_rejected `[{op, reason}]`, already, degraded (summary or ops
     completion failed: sources.py:862/1059/1119)
  8. `briefing` items, sources, ms (llm.py:814)
  9. `batch_file` rel path, state, seconds (batch_import.py)
  10. `memory_save` facts_added, facts_dropped, ms (memory.save, memory.py:1630)
  11. `error` scope, where, exc_type, message (≤ 200 chars), traceback (last 20 frames, ≤ 4 KB)
  12. `service` / `ws` start, stop, connect, disconnect {code}, refused_origin (chatbot.py:1329/1344-1363)
- **Verbose kinds** (dev only, `verbose.jsonl`): `prompt` {purpose, messages}, `turn_text` {user, reply},
  `tool_result` {name, args, result}, `ingest_text` {summary, ops}.
- **Usage**: the three streaming completions (llm.py:1066/1121/1138) pass `stream_options={"include_usage":
  True}`; the last chunk's `usage` fills tokens, else nulls. TTFT = first content or tool delta.
- **stdout via `logging`**: each module gets `log = logging.getLogger("<Module>")`, a formatter that prints
  `[<Module>] msg`; `print` calls become `log.info/warning`; turn-text prints (`Bot:`, `You:`) become
  `trace.verbose` plus a `log.debug` shown only in dev; tracebacks via `log.exception` + `error` event.
- **Retire session.log**: `turn_end` carries everything it had except the text. Readers to move: tests/qa/stress.py
  :120-145 (scrapes `[Tool] name(args` from stdout and `| tools:` from session.log → read `tool_call`/`turn_end`
  from events.jsonl), tests/llm/test_ask_before_lookup.py:238-246 (band → `turn_end.band`),
  tests/prod/test_data_paths.py:18/27/65-66, tests/prod/test_prod_script.py:27/39, `SESSION_LOG` in
  docker-compose.yml:18 and llm.py:21-22, `prod.sh migrate` (prod.sh:65 keeps moving an old file, nothing new).
- docker-compose.yml gets `logging: {driver: json-file, options: {max-size: "10m", max-file: "5"}}`.
- **`scripts/trace_report.py`** (stdlib): `<events.jsonl> [--since 24h] [--turn ID] [--ingest ID] [--errors]`; reads
  the rotated `.1…5` files too; turn ids are per-session, so a unit key is `(pid, turn_id)` and `--turn` takes
  `ID` or `pid:ID`.
- **`/trace` skill** (`.claude/skills/trace/SKILL.md`): where prod logs live (`$DATA/logs/`), run the report first,
  open raw lines only for flagged turns (`--turn`), never open `verbose.jsonl` outside dev, never copy event
  lines into a commit or PRD beyond ids and numbers.

### Decisions
- Chose two named profiles with a hard startup refusal over a "redact" flag: a missing flag must fail closed, and
  prod's file set (no verbose file at all) is checkable by `ls`.
- Chose retiring session.log over rotating it: every field but the text moves into `turn_end`; three test files
  and stress.py are the only readers.
- Chose stdlib `logging` + JSONL over a logging library: the app is small and the writer must never raise.
- Assume oMLX honours `stream_options.include_usage` (OpenAI-compatible); if it does not, tokens stay null and
  the report shows `–`; revisit with a tokenizer count.
- Risk: a new call site adds a free-text field to a prod event; mitigation: `event()` drops any string field
  longer than 200 chars outside an allow-list (`title`, `path`, `page`, `source`, `message`) and the sentinel test.
- Chose refusing dev also inside the container (`/.dockerenv`), not only for `~/docker-data` paths: the
  container's paths are `/data/…`, so the path check alone would not catch a bad compose override.

## Data model
- `events.jsonl` line: `{"ts","kind","profile","pid","turn_id"?,"ingest_id"?, …kind fields}`; `ingest_id` =
  `<pid>-<n>`; ids join events to `verbose.jsonl` lines, which share the envelope.
- Invariants: in prod `verbose.jsonl` does not exist; no prod event holds user, reply, document, tool-result or
  prompt text; every `turn_start` has one `turn_end` (outcome error/cancelled included); the first event of a
  process is `service start`.

## Test scenarios
1. tests/session/test_trace.py: one typed and one spoken fixture turn containing the sentinel `ZQX-SENTINEL` →
   events `turn_start`, `recall`, `turn_end` share a `turn_id` per turn; `ZQX-SENTINEL` and the reply text are
   absent from events.jsonl.
2. tests/llm/test_trace.py: a turn calling `search_wiki` → one `tool_call` with `result_len` > 0 and `ok`; the
   result text is absent from events.jsonl and from captured stdout.
3. tests/llm/test_trace.py: fake streaming client with a final `usage` chunk → `llm_call` has tokens_in/out set and
   `ttft_ms ≤ ms`; without usage → both null, no exception.
4. tests/sources/test_sources.py: ingest whose ops completion raises → `ingest_end.degraded` true, ops_applied 0,
   and exactly one `error` event with `exc_type` and a non-empty `traceback` sharing the `ingest_id`.
5. tests/session/test_trace.py: `respond` raising mid-stream → `error` event (scope turn) + `turn_end.outcome`
   error, the client gets `error {scope: turn}` and status returns to idle.
6. tests/session/test_trace.py: writing > 10 MB with a small `maxBytes` override → at most 5 backups exist;
   a write to an unwritable dir does not raise.
7. tests/prod/test_trace_profile.py: `TRACE_PROFILE=prod` turn → `verbose.jsonl` never created; first event is
   `service start` with profile prod; `TRACE_PROFILE=dev` with `WIKI_PATH=~/docker-data/…` → `chatbot.py` exits
   non-zero with one stderr line naming the path; compose file sets `TRACE_PROFILE: prod` and `max-size`.
8. tests/session/test_trace.py: dev profile turn → `prompt` and `turn_text` lines in verbose.jsonl only, carrying
   the sentinel and the same `turn_id`; events.jsonl still has no sentinel.
9. tests/prod/test_trace_report.py: fixture events file (2 turns, 1 ingest, 1 error) → `--turn 2` prints only that
   turn's events; `--errors` prints one row and exits 1; a file with no errors exits 0.
10. Existing readers green after the session.log retirement: test_ask_before_lookup band check, test_data_paths,
    test_prod_script; `scripts/check.sh` green.

## Out of scope
- Exception sweep (separate chore, spec here): turn-level handlers that print `{e}` only and must become
  `log.exception` + `error` event — chatbot.py:406, 474, 492, 515, 584, 620, 640; memory handlers chatbot.py:205,
  661, 995, llm.py:1272; WS catch-all chatbot.py:1360; degraded-ingest sites sources.py:862, 1059, 1119,
  1219/1222 (rejected ops), 1238; extractor fallbacks sources.py:242, 253, 284, 407, 445, 471, 566, 588, 687;
  batch_import.py:155; drop handler chatbot.py:747.
- Shipping logs anywhere (no Loki/OTel), dashboards, alerts, metrics endpoints.
- Logging in the browser/static UI; the ASR server's own log.
- Retention beyond rotation; migrating old session.log content into events.
