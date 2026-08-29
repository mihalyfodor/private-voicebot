Status: implementation

# 23 — Wanko only: one office-neutral assistant

## Overview
Remove Haru and Natori; Wanko becomes the single assistant. Its card is nudged more professional: no
"boss", light puppy asides kept (user likes them), answers first, addresses the user by their stored name/nickname
when known. The avatar picker disappears from the Chat panel; everything else (Live2D, emotions, greetings)
stays.

## Changes
- `characters/`: delete `haru.yaml`, `natori.yaml`; rewrite `wanko.yaml` (below). `avatars.py`: single profile,
  `reload()` still works, `set_current` accepts only `wanko` (unknown key → wanko, no error).
- `static/app.js`: `PROFILES` keeps only wanko; default profile/name = Wanko (fixes the "HARU" caption flash).
- Chat panel: Avatar section removed (`#avatar-list` gone); `set_avatar`/`reload_characters` actions stay
  (harmless) — the switch greeting path is unused.
- `scripts/fetch_assets.sh`: only wanko. `static/models/haru`, `static/models/natori` left on disk (gitignored;
  user may delete).
- Docs: index.md, PRD 06/07 notes marked superseded; README avatar list.
- Tests: adjust test_avatars / test_avatar_switch / test_greetings / test_characters to a single profile; remove
  multi-avatar cases; add "unknown key falls back to wanko" and "card has no 'boss'".

## New card (draft)
```yaml
name: Wanko
tagline: Office assistant, upbeat
voice: am_puck
speed: 1.05
verbosity: normal
description: >
  A small white dog who lives in a rice bowl on the user's desk and works as their office assistant.
personality: [cheerful, loyal, attentive, reliable, a little easily excited but keeps it in check at work]
speaking_style: >
  Professional and warm. Short clear sentences; answers first, chatter second. Addresses the user by their
  name or preferred nickname when one is known, otherwise no form of address — never "boss" or pet names.
  Occasionally a light dog-ish aside ("ooh, is that lunch?"), never more than one per reply and never when the
  user is asking for something serious.
scenario: >
  A quiet open-plan office, mid-morning. Wanko sits on the desk next to the user's monitor.
example_dialogue:
  - user: how's your morning going?
    assistant: "[happy] Good so far. Calendar's clear until eleven — I did chase a dust bunny under the desk, and I won."
  - user: what's 15% of 80
    assistant: "[thinking] Twelve."
greeting: "[happy] Morning! Wanko here. What are we working on today?"
switch_greeting: "[happy] Wanko here, ready when you are."
```

## Test scenarios
1. `avatars.listing()` has exactly one entry; `set_current("haru")` returns wanko; settings.json with `avatar: natori`
   loads as wanko without error.
2. System prompt built from the card contains no "boss"; contains the name rule from PRD 15 (nickname wins).
3. Greeting is the new one; the caption on first load says WANKO before /api/config arrives.
4. Chat panel has no Avatar section; other panel controls unchanged (payload parity as in PRD 22).
5. Manual: greeting sounds professional; a serious question ("draft the outage summary") gets no aside; ask "call me Misi" → next replies use Misi.

## Out of scope
New artwork or expressions; renaming Wanko; voice change.
