---
name: smoke
description: Run the headless smoke pass (e2e, one live turn, one battery section, view screenshots) against the local model and read the results. Use before the Implementation gate of any UI-visible or LLM-path change, or when asked to smoke test.
---

# Smoke pass

```
.venv/bin/python scripts/smoke.py [--out DIR] [--steps 1,2,3,4,5] [--no-start] [--section A] [--answer-timeout 300]
```

Output goes to `/tmp/smoke/<YYYYmmdd-HHMMSS>/` (RESULT.txt, RESULT.json, PNGs, logs). Each step is PASS / FAIL / SKIP.

## What each step proves
1. stack: `scripts/voicebot start --dev`, then oMLX up and the model loaded. If not, steps 3 and 4 SKIP ("model down").
2. e2e: the Playwright suite (`-m e2e tests/e2e`), no model needed. PNGs in `e2e/`.
3. live turn: one typed turn through the real model; the status goes processing / thinking / speaking and a reply arrives. PNGs in `live/`.
4. battery: `tests/qa/stress.py` section A in a throwaway vault. Only proves it ran through and replies arrived; grading stays manual. Files in `qa/`.
5. views: chat, wiki, sources, memory, settings rendered against a throwaway vault, no model needed. PNGs in `views/`.
6. trace (always): `scripts/trace_report.py --errors` over every `events.jsonl` the run's servers wrote (e2e/live in the out folder, `qa/`, `views/`); FAIL when any has an error. Its output ends RESULT.txt.

## How to report
- You MUST open every PNG under the out folder with the Read tool and describe what you see (layout, text, errors, empty panes). Exit codes alone are not a smoke pass.
- Quote RESULT.txt verbatim in the handoff, the trace_report block included.

## Triage
- stack: `stack.log` in the out folder; `scripts/voicebot status --dev`; `/tmp/voicebot-asr.log` for the ASR service.
- e2e / live: `e2e.log`, `live.log`.
- battery: `qa/transcript.md`, `qa/stress.log`, `qa/server.log` (next to the vault under `qa/`).
- views: `views/*.log`, `views/server.log`.
- trace: `scripts/trace_report.py <file> --turn ID` for the flagged unit (the /trace skill).

## Notes
- The first `scripts/voicebot start` can take minutes while the model loads. Use `--no-start` when the stack is known to be up.
- Never run against `~/llm-wiki`: the tool never does, every vault is a throwaway under the out folder.
