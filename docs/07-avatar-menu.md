Status: implementation

# 07 — Third avatar (Natori) and in-app menu with live switching

## Overview

Add Live2D's male sample **Natori** as a third profile, and a mobile-style menu button (top-left) that opens a drawer with the avatar picker and Shut down. Switching applies immediately, mid-conversation: the server swaps persona/voice, the client reloads the model; history is kept.

## User stories

- As a user, I tap ☰ → pick Wanko / Haru / Natori and the character, name and voice change without restarting or losing the conversation.
- As a user, my choice persists across restarts.
- As a user, Shut down lives in the same menu.

## UI/UX

☰ button top-left (40px, translucent). Tap → drawer slides in from the left (260px, dark, blurred backdrop): section **Avatar** with three rows (name + one-line description, current one highlighted); divider; **Shut down** in red at the bottom. Tap outside or Esc closes. Switching is disabled while the bot is speaking/thinking (button rows greyed) to avoid mid-utterance voice changes.

## Technical approach

- `avatars.py`: add `natori`; runtime state `set_current(key)` / `current()`; persisted to `settings.json` (gitignored). Precedence: settings.json > `AVATAR` env > default.
- `llm.py`: `build_system_prompt(avatar)`; `set_avatar()` replaces `_conversation[0]` (system) in place, keeping history, and appends a hidden note so the model knows it "is now" the new character.
- `chatbot.py`: WS action `{"action":"set_avatar","key":...}` → ignored while `processing`; otherwise updates avatars/llm/voice, clears the filler wav cache, broadcasts `{"type":"avatar", key, name}`. `GET /api/config` returns `{avatar, name, avatars:[{key,name,description}]}`.
- `static/app.js`: `avatar.load(profileKey)` destroys the current model and loads the new one on the same PIXI app; menu drawer; on `avatar` message → reload model + update label.
- `scripts/fetch_assets.sh`: fetch Natori from CubismWebSamples.

### Decisions

- Chose server-authoritative switching (WS action → broadcast) so voice and persona can never diverge from the model shown.
- Chose in-place system-prompt swap over `reset()` so context survives; the risk that the model keeps the old name is mitigated by an explicit note in the prompt.
- Natori voice: Kokoro `am_michael` at 1.0 (clear, mid-range). Not distributable (collab license) — fetched locally only.

## Data model

`settings.json`: `{"avatar": "wanko"}`. WS: client→server `set_avatar`; server→client `avatar`.

## Test scenarios

1. `avatars.set_current("natori")` → `current()["name"] == "Natori"`, `settings.json` written; `set_current("nope")` raises.
2. `llm.set_avatar()` after two turns → `_conversation[0]["content"]` mentions the new name, previous user/assistant turns intact.
3. WS: connect, send `set_avatar natori` → receives `{"type":"avatar","key":"natori",...}`; `/api/config` now reports natori.
4. WS: `set_avatar` while `processing` is True → no `avatar` message.

Manual: switch mid-chat → next reply uses the new name/voice and refers back to earlier context.

## Out of scope

Per-avatar memory files; theme/other settings (menu just leaves room for them).
