Status: implementation

# 39 — Context meter, recall stripping and "new topic"

## Overview
The conversation is one flat thread trimmed silently, oldest-first, when it exceeds the 16 k budget — and every
recall/deeper block stays in history, so a document session fills the budget in 7–8 exchanges. This PRD makes the
budget visible (meter, 75 %/90 % warnings), stops paying for stale recall blocks, and adds an explicit "new topic"
that folds memory first and restarts with a short, visible summary. Sessions stay throwaway; no branching.

## User stories
- As a user, I see how full Wanko's context is, get a nudge at 90 %, and know when old turns were forgotten.
- As a user, heavy document work no longer burns the budget: only my words stay in history once a turn is done.
- As a user, "new topic" (spoken, typed `/new`, or the palette) starts fresh without losing what was said: memory is
  folded first and a one-paragraph summary shows as a divider I can read.

## Technical approach
- `llm.context_usage()` → `{used, budget, trimmed}`; `used` = `estimate_tokens(_conversation)` + `TOOLS_TOKENS`
  (a measured constant for the tool schemas the estimator ignores). Sent after every completed turn and after
  `new_thread` as `context {used, budget, pct, trimmed}`; `trim_history` returns the count trimmed for that turn.
- UI: status line suffix `ctx 62 %`; rail dot amber ≥ 75 %, red ≥ 90 %; "forgot n old turns" in the status line
  when `trimmed > 0`.
- 90 % crossing: the next reply is followed by one fixed spoken sentence ("I'm close to the end of what I can hold —
  say *new topic* to start fresh."), once per crossing (re-armed when back under 75 %).
- Recall stripping: when a turn completes, the `<recall>`/deeper/Wikipedia note text appended to that turn's user
  message in `_conversation` is removed (the bare user text stays); `_session_turns` were always bare. The reply
  already quotes the source; follow-ups use `read_source`/dig-deeper.
- `llm.new_thread()`: `save_memory()` (fold first) → one summary completion (≤ 150 tokens, "what we talked about,
  decisions, open threads") → `_conversation = [system, {"role": "user", "content": "<earlier>…</earlier>"},
  {"role": "assistant", "content": "Fresh start — I kept the gist above."}]`; `_deeper_offer`, `_turn_sources`,
  `last_page` focus cleared; summary failure → an `<earlier>` line listing the turns' first sentences (never blocks).
  Server action `new_topic`; triggers: palette "New topic", typed `/new`, spoken/typed "new topic" (exact phrase
  regex, typed and voice paths); reply `thread {summary}` + `context`; UI inserts a divider showing the summary.
- Reload replay: `_on_connect` replays `_conversation` minus system/recall tags (what the model actually holds)
  instead of `_session_turns`, so the transcript matches the model's memory after a periodic fold.

### Decisions
- Explicit "new topic" over automatic topic detection (user decision 2026-08-30).
- Strip blocks after the turn over keeping them: 10× cheaper turns; the source stays reachable via tools.
- Summary visible as a divider (user decision): the user can see and correct what carried over.
- No branching, no auto rolling summary, no transcript persistence for now.

## Test scenarios
1. After each completed turn a `context {used, budget, pct, trimmed}` message is sent; `used` equals
   `estimate_tokens(_conversation) + TOOLS_TOKENS`.
2. Fixture: `ctx 62 %` in the status line; rail dot class amber at 75, red at 90; "forgot 3 old turns" when
   `trimmed: 3`.
3. Crossing 90 % appends the fixed sentence once; a later turn at 92 % does not repeat it; dropping to 70 % then
   rising to 91 % repeats it.
4. A 10-turn session where every turn recalled 1500 tokens keeps `_conversation` under 4 k tokens; each stored user
   message equals the bare text after its turn; `_session_turns` unchanged.
5. `new_topic`: `save_memory` is called before the reset (mocked memory records the turns); `_conversation` is
   `[system, <earlier> ≤ 150 tokens, ack]`; deeper offer/focus cleared; `thread {summary}` + `context` sent;
   fixture shows the divider with the summary text; palette entry, `/new` and spoken "new topic" all trigger it.
6. Summary call raises → the reset still happens with the fallback `<earlier>` (first sentences), one error print,
   no user-facing error.
7. Reload after a periodic fold replays exactly the stored (stripped) turns the model holds, not `_session_turns`.
8. `test_protocol_doc.py` green with `new_topic`, `context`, `thread`.

## Out of scope
Named/branching threads, automatic summarising when over budget, transcript persistence, changing `CONTEXT_BUDGET`,
per-thread memory.
