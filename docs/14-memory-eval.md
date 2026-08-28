Status: implementation

# 14 — Memory evaluation harness (text-only)

## Overview

A text-only way to chat with the assistant's LLM + memory pipeline (no audio, no server) across simulated sessions, and to measure what the memory retains, what it loses, what it wrongly keeps, and how the memory file grows over time. Used to evaluate the current append-only `shortmem.txt` design and any replacement.

## User stories

- As the developer, I run one command and get per-category recall rates (single-hop, multi-session, update, temporal, preference, abstention) plus memory growth/duplication stats.
- As the developer, I can chat with the pipeline in a REPL, end a session, and see exactly what was written to memory.
- As the developer, I can replay a scenario N times to see drift, duplication and bloat.

## Technical approach

- `eval/harness.py`: `Sandbox` (isolated shortmem/settings/session.log, tools disabled by default), `Session` (`start/say/end`), `run_script` (YAML scenario → results JSON), CLI with `--rounds`, `--keep-memory`, `--chat` REPL.
- `eval/scripts/basic.yaml`: 4-session scenario adapted from LongMemEval categories; probes score by keyword any-of and `expect_not` for updates/abstention.
- Duplicate detection: normalised lines, difflib ratio > 0.85.
- Results in `eval/results/` (gitignored except `baseline.json`).

### Decisions
- Keyword scoring over LLM-as-judge: deterministic, free, good enough for a handful of probes; a judge can be added per probe later.
- Drive `llm.ask` directly rather than the WebSocket: memory behaviour lives entirely in `llm`/`memory`.
- Tools disabled by default so runs are fast and deterministic; `tools=True` available.

## Test scenarios
1. Scoring: expect / expect_not / category aggregation with stub replies.
2. Sandbox never touches the real `shortmem.txt`.
3. Duplicate detection flags near-duplicate lines.
4. `run_script` wiring with a stubbed `llm.ask` and `save_memory`.

## Out of scope
Changing the memory design itself (PRD 15); LLM-judge scoring.
