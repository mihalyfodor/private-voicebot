Status: implementation

# 11 — Context budget, history trimming, reply length

## Overview

Bound the conversation to a token budget tied to the loaded model, trim old turns safely, cap reply length as a safety ceiling, and make reply verbosity a setting (global + per character). Today nothing is bounded: history grows until restart, `shortmem.txt` is injected whole, and there is no `max_tokens`.

## User stories

- As a user, long sessions don't get slower and slower or eventually fail with a context error.
- As a user, I can choose how chatty the assistant is (`concise` / `normal` / `detailed`), globally or per character.
- As a user, a runaway reply can't talk for minutes.

## Technical approach

- `llm.py`
  - On first use, query `GET /v1/models`, read `max_model_len` for `OMLX_MODEL`; `context_budget = min(max_model_len, CONTEXT_BUDGET)` (env, default 16000 tokens — voice latency beats recall; 256k would cost seconds of prompt processing per turn).
  - Token estimate: `len(text) // 4` (no tokenizer dependency).
  - Before each request: while estimated size of `_conversation` > budget − `MAX_TOKENS`, drop the oldest non-system turn; if it is an assistant turn with `tool_calls`, drop its following `tool` turns with it (never leave an orphaned tool result).
  - `max_tokens=MAX_TOKENS` on every completion (env, default 400).
  - Verbosity (`short` | `normal` | `long`) → prompt rule text:
    - `short`: "at most two sentences" (today's behaviour)
    - `normal` (default): "two to four sentences; a little longer only when actually explaining something"
    - `long`: "as long as the answer needs, but still spoken prose — no lists or headings"
    Resolution order: `settings.json` (slider) > `VERBOSITY` env > card `verbosity` > `normal`.
    `llm.set_verbosity(value)` rebuilds the system prompt in place (like `set_avatar`).
- UI: ☰ menu gets a **Reply length** section with a 3-position slider (short · normal · long). Change → WS `{"action":"set_verbosity","value":...}` → persisted → broadcast `{"type":"verbosity","value":...}`; `/api/config` includes `verbosity`.
- `memory.py`: inject only the last `MEMORY_LINES` (default 60) lines of `shortmem.txt`.
- `characters/*.yaml`: optional `verbosity` field (Wanko: short; Haru, Natori: normal) — used only when no slider value is saved.
- `.env.example`: `CONTEXT_BUDGET`, `MAX_TOKENS`, `VERBOSITY`, `MEMORY_LINES`.
- `MAX_TOKENS` default 400 for short/normal, 800 for long (env overrides both).

### Decisions

- Chose a chars/4 estimate over a real tokenizer: ±20% accuracy is fine for a budget with headroom, and it avoids a tokenizer dependency for a Gemma vocab.
- Chose 16k default budget over the model's 256k because per-turn prompt processing on Apple Silicon scales with context; revisit when barge-in exists and waiting is less painful.
- Verbosity lives in the prompt, `max_tokens` is only a ceiling: the ceiling never shapes the reply, it only stops runaways.
- Risk: trimming removes the turn a user refers back to ("as I said earlier") — accepted; memory file covers long-term facts.

## Data model

`settings.json` gains `verbosity`. `_conversation` invariants: index 0 is always the system message; every `tool` turn is preceded (transitively) by its assistant `tool_calls` turn.

## Test scenarios

1. Trimming: 30 turns with a tiny budget → oldest dropped first, system message kept, total under budget.
2. Trimming never orphans a tool result: assistant(tool_calls) + tool + assistant sequence dropped as a unit.
3. `max_tokens` present in every `create()` call (mocked client) and equals `MAX_TOKENS`.
4. Verbosity resolution: settings > env > card > default; the prompt contains the matching rule text; WS `set_verbosity` persists and broadcasts; invalid value → error message.
5. `memory.load` with a 200-line file injects only the last 60.
6. `/v1/models` unavailable → fallback budget = `CONTEXT_BUDGET`, warning printed, no crash.

## Out of scope

Summarising trimmed history into memory mid-session; token-exact accounting.
