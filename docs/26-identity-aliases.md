Status: implementation

# 26 — Identity aliases

## Overview
`identity.name` is the real name and `identity.nickname` the form of address; add `identity.aliases` — every other
string that refers to the user (other spellings, full name, old nicknames, what the ASR tends to hear) — and use it
everywhere the system decides "is this the user?": prompt render, memory extraction, wiki ingest, and ASR biasing.

## User stories
- As a user, Wanko calls me Michael (easy to say) while knowing my real name is Mihály Fodor and that Misi is me.
- As a user, an email or screenshot mentioning "Misi" or an OCR'd "Mihay" is filed as about me, not as a new person.
- As a user, when Whisper hears "Michael", memory does not overwrite my real name.
- As a user, I can edit the alias list in the Memory view like any other identity value.

## Technical approach
- Schema: `identity.aliases` is a list of strings (first list-valued scalar-section key). `_normalise` accepts a
  comma-separated string (editor input) and normalises to a list; render shows
  `name: Mihaly Fodor (call them: Michael; also: Misi, Mihály)`; `apply_ops` supports `ADD identity.aliases <str>`
  (append if new), `UPDATE identity.aliases <list|str>` (replace), `DELETE identity.aliases` (clear); dedupe case-insensitively.
- `memory.user_names(profile) -> list[str]`: name + nickname + aliases; `memory.is_user(text) -> bool`: exact or
  fuzzy (`_similar`, ratio 0.8) match against `user_names` or any single token of them (for "Mihay").
- `PROPOSE_SYSTEM`: the profile shows aliases; rule: a name matching any alias is the user — never UPDATE
  identity.name from it; ADD to aliases when the user says "people also call me…".
- Ingest prompts (`sources.py`, `wiki.ingest`): list `user_names` as "the user"; `restrict_ops` rejects `new_page
  people/<slug>` and `link` targets whose name `is_user`; entity lists render the user as plain "the user (Mihaly)".
- ASR: `asr.py` passes `initial_prompt` built from `user_names` (+ a few frequent words from the profile: people
  names) to mlx-whisper; cap at ~40 tokens; rebuilt after each memory save.
- Editor: identity row `aliases` shows the comma-joined list; editing sends `UPDATE identity.aliases` with the string.

### Decisions
- A list under identity rather than a new section: it is identity data; the editor's scalar row handles a comma list.
- Fuzzy match only for the user (not for other people): OCR misspellings of the owner are the common failure.
- ASR biasing via initial_prompt over post-correction: fixes the transcript itself, no extra latency.

## Test scenarios
1. `_normalise` turns `"Misi, Mihály"` into a list; render shows the "also:" clause; empty list → no clause.
2. `apply_ops` ADD/UPDATE/DELETE on aliases; duplicates ignored case-insensitively; audit lines written.
3. `is_user`: "Michael", "Misi", "Mihay Fodor" → True with the example profile; "Anna" → False.
4. Extraction (mocked LLM proposing `UPDATE identity.name "Michael"` when "Michael" is an alias) → rejected by a
   Python guard, logged; a proposal to ADD an alias is applied.
5. Ingest: screenshot text naming "Mihay Fodor" and "Elena" → no `people/` page for the user, one for Elena; the
   source page's Entities lists "the user".
6. ASR: `initial_prompt` contains the names; rebuilt after a save adds an alias (unit, mocked transcribe).
7. Editor: aliases row edit sends `UPDATE identity.aliases "Misi, Michael"`; snapshot returns the list.
8. Harness `aliases.yaml` (live): S1 "my name is Mihaly Fodor, call me Michael, friends say Misi"; S2 probe "what's my
   actual name?" → Mihaly; "what do you call me?" → Michael; an update-heavy turn "Michael here again" must not change name.

## Out of scope
Aliases for other people; pronunciation hints for TTS (could be a later `identity.say_as`).

## Results (harness, live, 2026-08-29, gemma-4-26B, `--tools`)
`aliases.yaml`: **8/8** (abstention, multi-session ×2, preference ×2, single-hop, update ×2). Final profile:
`name: Mihaly Fodor (call them: Michael; also: Misi, Mike)`; a later "Michael here again" produced no name change;
one people page for Elena, none for the user; 34 tokens rendered, 0 duplicates.
