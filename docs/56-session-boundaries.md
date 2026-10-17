Status: exploration (signed off 2026-10-17; implementation after PRD 57)

# 56 — Session boundaries: fresh prompt, idle and disconnect saves, a new thread each day

## Overview
The system prompt is built once and then goes stale; memory is only folded every 5th turn; one implicit thread runs
for days. Design source: `docs/reviews/2026-10-17-session-handling.md` Part B #1, #3, #4. This PRD keeps the prompt
true (no clock in it, a rebuild when the profile changes), saves memory when the user goes quiet or away, starts a
new thread on the first turn of a new day, and makes "new topic" answer before it folds.

## Problem
- At 18:00 the prompt still says "local time 11:00": `location_line()` bakes `%H:%M` in at start (agent/prompt.py:143).
- After "Forget everything" the old `<user_profile>` is still in `_conversation[0]`: `memory_reset`, `memory_op`,
  `memory_ops` and the bridge actions never call `refresh_context` (server/actions/memory.py:78-117).
- `refresh_context` and any rebuild lose the avatar-switch note (`_rebuild_system_prompt(note="")`, conversation.py:70).
- Closing the laptop at night loses up to 4 unsaved turns: the fold runs only every `MEMORY_SAVE_EVERY` = 5 turns
  (server/session.py:18) or at shutdown.
- Next morning the first turn continues yesterday's thread with yesterday's date in the prompt.
- "New topic" waits 10-40 s: `new_thread` folds memory first (conversation.py:190-200), then summarises.

## Goals / Non-goals
Goals: the prompt never states a wrong time and shows the profile as it is; turns are folded within 10 minutes of
silence and on disconnect; the first turn after a night starts a fresh thread; "new topic" returns after one short
summary call. Non-goals: a thread archive, an idle new thread, topic detection, per-turn time, anything of PRD 57.

## Design
### 1. Prompt freshness (S)
- `location_line()` drops `local time HH:MM`: "The user is in Honolulu (timezone Pacific/Honolulu). ". The
  `Today's date is ...` line stays (day granularity). One new `agent/prompt.py:today() -> date` returns the date in
  the user's timezone (`_weather.location()["tz"]`, system zone when it is "auto" or unknown); the prompt and the
  day-boundary check both call it, so tests patch one function.
- `tools/time.py` `run` returns `Saturday, 17 October 2026, 14:05:09 (Pacific/Honolulu)` (same timezone rule);
  the DEFINITION description says "current local date, weekday, time and timezone".
- `save_memory` calls `_rebuild_system_prompt()` when `fold.save` returned a truthy `applied`. It runs on the hook
  thread; the swap is `_conversation[0] = system`, an atomic item assignment.
- `refresh_context()` is called by `_action_memory_op`, `_action_memory_ops`, `_action_memory_reset` and `_bridge`
  after a successful write (a rejected edit does not rebuild), via `asyncio.to_thread` (vault map read).
- Avatar note: module state `_avatar_note`, set by `set_avatar` (only when `len(_conversation) > 1`), cleared by
  `reset()` and `new_thread()`. `_rebuild_system_prompt()` takes no argument and appends `_avatar_note`.

### 2. Idle and disconnect saves (S)
- Constant `IDLE_SAVE_MIN = int(os.getenv("IDLE_SAVE_MIN", "10"))` in server/session.py next to `MEMORY_SAVE_EVERY`.
- One helper `TurnController.save_now()` in session.py: `_start_hook` with the same overlap guard, running
  `_periodic_memory`-style code (`panels._save_memory_hook`, the body of `_periodic_memory` without the count).
  It returns at once when `has_unsaved_turns()` is false.
- **Idle timer, loop side.** `panels.arm_idle_timer()` cancels the pending `loop.call_later(IDLE_SAVE_MIN * 60,
  _idle_fire)` handle and sets a new one; the worker's turn-end path calls it with `loop.call_soon_threadsafe`
  (`panels._loop` is already the app loop). `_idle_fire` runs on the loop: if `controller.busy` it re-arms and returns
  (never folds while a turn is in flight); otherwise it calls `controller.save_now()`, which hands the fold to the hook
  thread, so the loop never blocks. A turn that starts while the timer is pending needs no cancel: `busy` is
  checked at fire time.
- **Disconnect.** The `finally` in server/ws.py (after `approvals.expire`) calls `controller.save_now()` for the
  closing session. The timer is cancelled there too; the next connection arms it with its first completed turn.
- Both paths are the existing fold: failures `restore()` the turns and are reported by `_memory_failed`.

### 3. Day boundary (S)
- `a_conversation.last_turn_at: datetime | None`, stamped (aware, user timezone) when a turn completes, cleared by
  `reset()` and `new_thread()`. Not persisted: a restart already starts a new conversation.
- In `turns.respond` (and `canned_turn`) before `_respond`: `a_conversation.day_rollover(now)` is true when
  `last_turn_at` is set, `now.date() != last_turn_at.date()` (user timezone) and `now - last_turn_at >= 60 min`
  (`DAY_GAP_MIN = 60`). Then the turn runs the same body as `new_topic` first (`new_thread`, `thread` frame, no
  spoken ack), then answers the user's text.
- A date change under 60 minutes (chatting across midnight) calls `refresh_context()` only, so the date line is right.
- **What the user sees:** the "thinking" state for 2-4 s (one summary call), then a quiet `<earlier>` divider in the
  transcript, the same one a reconnect replay shows. No spoken line, no toast.
- Checked at turn start, not on a timer: a timer does not fire while the laptop sleeps.

### 4. "New topic" responds first (S)
- `new_thread()` no longer calls `save_memory()`: it takes `dialogue()`, summarises (one call), swaps
  `_conversation`, clears `_avatar_note` and rebuilds the prompt, then returns the summary.
- The `new_topic` handler sends `thread`, speaks the ack, then calls `controller.save_now()` (hook thread).
  When the fold applies ops, step 1 rebuilds the new thread's prompt with the updated profile.
- **Rollback.** `save_memory` snapshots `_session_turns`, which `new_thread` does not touch; on failure `restore()`
  puts the turns back in front and the next periodic, idle, disconnect or shutdown fold retries them. The thread
  stays swapped, nothing is lost. "New topic" twice quickly: the second `save_now` hits the overlap guard and skips;
  the turns stay queued for the next trigger.

## Scenarios
1. `build_system_prompt` contains no `HH:MM` pattern but keeps the date line and "(timezone <tz>)".
2. `get_time` output contains the weekday, the year, the time and the timezone name.
3. After `save_memory` applies an op, `_conversation[0]` holds the new profile value; with `applied` empty it is the same object.
4. After each of `memory_op`, `memory_ops`, `memory_reset` and a bridge move, `_conversation[0]` reflects the profile (reset: no `<user_profile>`); a rejected edit leaves it alone.
5. After `set_avatar` and then `refresh_context`, the switch note is still in `_conversation[0]`; after `new_thread` or `reset` it is gone.
6. With a fake clock, 10 minutes after the last turn with unsaved turns, `save_memory` runs once on the hook thread; with none unsaved it does not.
7. The idle timer firing while a turn is in flight does not fold; it re-arms and folds after the turn.
8. A turn completing re-arms the timer: two turns 6 minutes apart cause no fold at minute 10 but one at minute 16.
9. A WebSocket disconnect with unsaved turns calls the fold; without them it does not.
10. A fold already running makes the idle and disconnect saves skip, and the turns stay in `_session_turns`.
11. With `today()` patched, last turn yesterday 22:00 and a turn at 08:00: `_conversation[1]` is an `<earlier>` message, a `thread` frame is sent, no ack is spoken, the prompt has today's date, and the turn is then answered.
12. Last turn 23:58 and a turn at 00:03: no new thread, the prompt's date is updated.
13. With a slow fake fold, `new_thread()` returns before it finishes; the `new_topic` handler speaks the ack and then starts the fold.
14. A failing fold after `new_thread` leaves the old turns in `_session_turns` and `_conversation` on the new thread; a second "new topic" during a running fold starts no second `fold.save`.

## Tests
- tests/agent/test_context_meter.py: scenarios 5 (`test_avatar_note_survives_refresh_and_clears_on_new_thread`), 13, 14; update
  `test_new_thread_folds_first_then_restarts_on_a_short_summary` (it asserts the fold-first order) to the new order.
- tests/agent/test_llm_unit.py: 1, 3, 4 (conversation side: `test_prompt_has_no_clock`, `test_save_memory_rebuilds_prompt_when_ops_applied`).
- tests/tools (get_time): 2. tests/server/test_palette.py or test_text_input.py: 4 (the action side).
- tests/server/test_turns.py (beside the fold-cadence tests, ~323-417): 6-10 (`test_idle_save_*`,
  `test_disconnect_saves_unsaved_turns`) and 11-12 (`test_day_rollover_*`, `today()` patched).
- tests/memory/test_memory.py (beside the shutdown-hook tests, ~441-474): 3 if kept at the fold level.

## Out of scope
A thread archive or history browser (PRD 39); a new thread on idle alone; topic detection; the time injected per
turn; a midnight timer; a configurable boundary UI (env vars only); everything in PRD 57 (tool-result stubs, focus).

## Decisions
- Clock out of the prompt over rebuilding every minute or injecting time into each turn: the clock is the one field
  that changes constantly and `get_time` already serves it; a stable prefix keeps oMLX's prefix cache warm, and a
  per-turn tail line adds tokens and makes the model parrot the time.
- Idle saves but never starts a new thread: a mid-afternoon pause must not wipe context; the fold only moves turns to
  memory, the conversation stays.
- Date change needs the 60-minute guard: at 00:03 after a 23:58 turn the user is mid-conversation; only a real gap
  makes a fresh thread natural.
- Turn-start check over a timer: no scheduler, and it works after laptop sleep.
- Loop-side `call_later` over a thread timer: re-arm and cancel are cheap and race-free on the loop; the fold itself
  stays on the hook thread through the existing guard.
- Quiet divider over a spoken ack on rollover: the user asked a question, not for a reset.

## Risks
- A profile change rebuilds the prompt and costs one re-prefill (seconds on the local model) on the next turn;
  accepted, profile edits are rare.
- A fold mid-idle racing a new turn: turns spoken after the snapshot are appended after it, and a failed fold's
  `restore()` and the turn rollback in agent/turn.py (~252) both work by identity, so neither drops the other's turns.
- The rollover summary call lands on the first turn of the day: +2-4 s, once.
- docs/index.md must list this PRD (tests/test_index.py); add it in the implementation commit.

## Size and files
Size: M (four S pieces).
Files: agent/prompt.py, agent/conversation.py, tools/time.py, server/actions/memory.py, server/session.py,
server/panels.py, server/ws.py, server/turns.py, docs/protocol.md (no new frame; unchanged unless `thread` text
changes), docs/index.md, the tests above.
