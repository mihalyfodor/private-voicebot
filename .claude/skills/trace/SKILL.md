---
name: trace
description: Read the app's event trace (events.jsonl) to answer "what went wrong / what happened in turn N / why was that ingest degraded". Use when asked about errors, slow turns, recall bands, token use or ingest outcomes in prod or dev logs.
---

# Event trace

```
scripts/trace_report.py <events.jsonl> [--since 24h] [--turn ID|pid:ID] [--ingest ID] [--errors]
```

## Where the logs are
Written by `core/trace.py` (profile from `TRACE_PROFILE`, folder from `TRACE_DIR`).
- prod (the container): `$DATA/logs/events.jsonl` (`DATA` defaults to `~/docker-data/voicebot`), rotated `.1` … `.5` (10 MB each). No `verbose.jsonl` exists in prod.
- dev (a checkout): `events.jsonl` + `verbose.jsonl` in `TRACE_DIR` (default the repo-local `logs/`).
- a smoke run: every `events.jsonl` under `/tmp/smoke/<stamp>/`; RESULT.txt already quotes `--errors`.

## How to read it
1. Run the report first (`--since 24h` for "yesterday"): one row per turn and ingest, then the errors.
2. Open raw lines only for the flagged units: `--turn 17` (or `pid:17` when the process restarted), `--ingest <id>`. Children are ordered by `start`; `parent_id` is the turn's or ingest's root span; `run_id` groups a run across restarts.
3. In dev only, join `verbose.jsonl` lines on `turn_id` for the exact prompt, turn text and tool results.

## Rules
- Never open `verbose.jsonl` outside dev, and never ask for one from prod: prod writes none by design.
- Never copy event lines into a commit, PRD or chat beyond ids, kinds and numbers.
- The events hold no text; if an answer needs what was said, say so and ask instead of guessing.
