Status: implementation

# 06 — Switchable avatars, Wanko default

## Overview

Make the avatar a named profile (model, framing, expressions, voice, persona name) selectable via `AVATAR` in `.env`, and add Live2D's official **Wanko** (dog-mochi mascot) as the default — non-human, office-friendly. Haru stays available.

## User stories

- As a user, I set `AVATAR=wanko` or `AVATAR=haru` and get a matching character, voice and name after restart.
- As a user, Wanko lip-syncs and shows the same four emotions as Haru.

## UI/UX

Unchanged layout. Transcript label uses the avatar's name. Wanko is framed larger/lower (round body, no legs). A small copyright line for the Live2D sample sits bottom-right (license requirement).

## Technical approach

- `avatars.py`: `AVATARS = {key: {name, voice, speed, persona}}`, `current()` reads `AVATAR` env (default `wanko`).
- `llm.py`: system prompt built from `avatars.current()` name/persona.
- `chatbot.py`: Kokoro voice/speed from the profile; `GET /api/config` → `{avatar, name}`.
- `static/app.js`: `PROFILES[key]` = modelUrl, avatarScale/topCrop, idle group, expressions. Expressions support two forms: `{expr: "f04"}` (model expression file) or `{params: {ParamId: value}}` (applied every frame after motion update with a lerped weight → this also gives the intensity cap PRD 04 wanted). Client fetches `/api/config` before loading.
- `scripts/fetch_assets.sh`: generic `fetch_model dir base model3.json` used for Haru and Wanko (CubismWebSamples, develop branch).

### Decisions

- Chose server-side selection (env) over a URL param because the voice is chosen server-side; one source of truth.
- Chose per-frame param overrides for Wanko expressions because the sample ships no `.exp3.json`; same mechanism later works for any model.
- Wanko voice: Kokoro `bm_lewis` at 1.05 — warm British male; user found `am_puck` too deep. Alternatives sampled in `/tmp/voices/`.
- Risk: Wanko's `PARAM_FACE_01` may look smug rather than thoughtful; tune by eye.

## Data model

`AVATARS[key]` server; `PROFILES[key]` client; contract is the key string returned by `/api/config`.

## Test scenarios

1. `avatars.current()` with `AVATAR=haru` → name "Haru", voice `af_sarah`; unset → Wanko profile.
2. `avatars.current()` with unknown `AVATAR` → raises with a clear message listing valid keys.
3. `GET /api/config` (TestClient) → `{"avatar": "wanko", "name": "Wanko"}` by default.
4. `llm.SYSTEM_PROMPT` contains the current avatar's name.

Manual: default start shows Wanko, greeting spoken in the new voice, mouth moves; "say something happy" → ears/blush change; `AVATAR=haru` restart → Haru as before.

## Out of scope

Runtime switching without restart; more avatars (orb/robot); per-avatar filler phrasing.
