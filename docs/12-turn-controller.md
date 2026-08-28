Status: done

# 12 — Session, turn controller, protocol hardening

## Overview

Structural fix for the concurrency and connection model, driven by the adversarial code + architecture reviews (2026-08-28). Today `processing` is a flag not a lock, all connection state is module-global, the WS loop dies on one bad frame, the server is reachable and controllable from the LAN, and the protocol has no turn ids. This PRD replaces that with a `Session` per connection, a single `TurnController` that runs turns one at a time, a robust WS loop, and turn-tagged messages — plus the reviews' cheap wins.

## User stories

- As a user, two quick utterances never interleave or answer the wrong question.
- As a user, reloading the page mid-reply never wedges the UI or the server; the new page continues cleanly.
- As a user, nothing on my network (or a random web page) can talk to the app.
- As a user, my settings and session memory survive a crash.

## Technical approach

**Server (`chatbot.py`, new `session.py`)**
- `Session`: holds `websocket`, `endpointer`, `recorder`, `ptt_active`, `hearing`, `playback_done` (Event), `turn_id` counter. Created in `websocket_endpoint`; a second connection **closes the previous** socket (code 4000) and takes over; all per-connection state is reset on disconnect (fixes PTT/recorder/hearing leaks).
- `TurnController`: one worker thread + `queue.Queue`; `submit(kind, payload)` from any thread; `busy` property replaces the `processing` flag; the endpointer is gated from `controller.busy` **synchronously in the frame handler** (no window). Turns: `utterance(audio)`, `text(user_text)`, `canned(text, thinking)`. A stray PTT `stop` when not recording is ignored (no state change).
- WS loop: `try/except Exception` per message (log, continue); `finally` clears the session; `_action_reload_characters` errors become `{"type":"error"}`.
- VAD `feed` runs via `loop.run_in_executor` (torch off the event loop); `send_sync(...).result(timeout=10)`.
- Bind `HOST=127.0.0.1` by default (`HOST` env to override); reject `/ws` unless `Origin` is `http://localhost:PORT` / `http://127.0.0.1:PORT` (close 4003).
- Protocol: `state`, `speech`, `speech_end`, `transcript(assistant)` carry `"turn": n`; client echoes it in `playback_done`; server ignores mismatched/older turns. `{"action":"playback_blocked","turn":n}` lets the client report autoplay blocking → server ends the wait (goes idle) while the client still plays after the gesture.
- Cheap wins: atomic writes for `settings.json` (`.tmp` + `os.replace`, `"version": 1`) and `shortmem.txt`; `save_memory()` also after every 10 turns (idempotent already); `@pytest.mark.live` on live-LLM test files with `addopts = -m "not live"` in `pytest.ini` (docs updated; `-m live` runs them).
- mic-worklet: one-pole/biquad low-pass ~7 kHz before decimation.

**Client (`static/app.js`)**
- Track `currentTurn` from `state`/`speech`; send `playback_done` with it; on `playback_blocked` path send that instead; "listening…" only when `mic.started`; on socket close code 4000 show "opened in another tab" and stop reconnecting; re-assert the audio-unlock hint in `render()`.

### Decisions

- One worker thread over asyncio-only: TTS/ASR/LLM calls are blocking libraries; a single worker keeps them off the loop and serialises turns by construction.
- Close-previous over reject-new for second connections: reload is the common case and must win.
- Origin check + loopback bind over a token: zero UX cost for a single-user localhost app; LAN use is opt-in via `HOST`.
- Turn ids over sequence numbers per message: only turn boundaries matter for playback/idle correctness.
- Streaming the first LLM call (review R2, latency) is deliberately a separate PRD 13 — different risk profile (tool-call delta parsing on oMLX).

## Data model

`settings.json`: `{"version": 1, "avatar", "backdrop", "hands_free", "verbosity"}` (missing version = 0, migrated on load). WS additions: `turn` field; `playback_blocked` action; close codes 4000 (replaced) / 4003 (bad origin).

## Test scenarios

1. Controller: two utterances submitted back-to-back → executed sequentially; `_conversation` has user/assistant pairs in order; second turn's audio starts only after the first turn's `playback_done`.
2. Gating: while a turn is running, frames fed to the session → endpointer gated (no second utterance) — checked synchronously, not after a sleep.
3. Malformed text frame → `error` message, socket stays open, next valid action works.
4. Bad YAML on `reload_characters` → `error` message, socket stays open.
5. Reconnect: connect A, start PTT, connect B → A receives close 4000; B's session has `ptt_active=False`; frames from B reach hands-free/PTT normally.
6. Stale `playback_done` with an old `turn` → ignored; current turn still waits; correct turn id releases it.
7. `playback_blocked` for the current turn → server goes idle without waiting 120 s.
8. Origin `http://evil.example` → close 4003; `http://localhost:8010` → accepted. Default bind host is `127.0.0.1`.
9. `save_setting` is atomic (`os.replace`) and adds `version`; a truncated file loads as `{}` with a warning, not an exception.
10. `save_memory` invoked after 10 turns (mocked).
11. Stray `ptt stop` while a turn is running → no `state idle` emitted.
12. `pytest` default run excludes live tests; `-m live` includes them.

Manual: two fast questions in hands-free; reload mid-reply; open a second tab; ask from a phone on the LAN (refused).

## Out of scope

Streaming first LLM call (PRD 13); barge-in/cancel (protocol now allows it — parked); multi-client support.
