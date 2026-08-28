Status: implementation

# 08 — Backdrops

## Overview
Selectable background image behind the avatar, chosen from the menu, persisted in `settings.json`. Five illustrated interiors (offices + anime/lofi rooms) under the Pixabay Content License, fetched by `scripts/fetch_assets.sh` (gitignored).

## Technical approach
`backdrops.py` catalogue (key, name, file, credit, url, license); `GET /api/config` returns `backdrop` + `backdrops`; WS `set_backdrop` → validated, saved, broadcast `{"type":"backdrop"}`. Client: `#backdrop` fixed layer (`background-size: cover`, brightness 0.75 so the avatar and captions stay readable), chips in the drawer, credit line merged with the avatar credit.

### Decisions
- Pixabay illustrations over OpenGameArt: the only anime-ish offices with a clean license; the GPL-licensed office was skipped.
- Images dimmed via CSS rather than pre-processed so any image works.

## Test scenarios
1. `validate` accepts catalogue keys, rejects others; `listing()` maps files to `/static/backdrops/...`.
2. WS `set_backdrop` persists and survives an avatar switch; `/api/config` reflects it.

## Out of scope
Custom user images; per-avatar default backdrops.
