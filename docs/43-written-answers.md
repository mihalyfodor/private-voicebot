Status: implementation

# 43 — Answer length by origin; written cards; explicit save to wiki

## Overview
Every answer is 1–3 sentences, which fights content questions. This PRD sets length by how the user asked and
where the answer came from, adds a **written card** — a rendered, copyable long-form answer in the transcript that
is not read aloud — and makes saving to the wiki a separate, explicit action. Memory keeps folding the short beats
automatically; durable notes exist only when the user decides.

## User stories
- As a user typing a content question that Wanko answers from a document/page/Wikipedia, I get the full answer
  inline without asking.
- As a user speaking, I get the short answer and "Want me to dig deeper, or write it up?"; "write it up" puts a
  written card below with headings/lists/tables and source refs; Wanko just says it is written out.
- As a user, I press **Save to wiki** (or say "save that as a note") and the card becomes `notes/<date>-<slug>.md`;
  otherwise nothing lands in the wiki.
- As a user, I can **Copy** a card.

## UI/UX
- Written card: a `.msg.card` in the transcript: rendered markdown (the wiki renderer: marked → DOMParser →
  whitelist), collapsed to ~12 lines with "show all", buttons **Copy** and **Save to wiki** (a small folder picker
  defaulting to `notes/`; after saving the button reads "saved → notes/…" and the page opens via `wiki_open`).
- Spoken turns that produce a card: TTS says one line ("Written out below.") — the card body is never spoken.
- No change for chit-chat.

## Technical approach
- **Length policy** (`llm.py`, per turn, deterministic): `grounded` = the turn carried recall/focus/deeper/Wikipedia
  text or a tool result; `content_question` = starts with/contains what/how/why/explain/compare/list/steps/
  difference/summarise or is ≥ 8 words. Typed ∧ grounded ∧ content → the completion runs at `long` (max_tokens as
  the `long` verbosity), no offer appended. Spoken → current short rule; the offer becomes "Want me to dig deeper,
  or write it up?" (fixed phrase list extended; `is_affirmative` unchanged; new `wants_writeup` regex: write it up /
  write that up / written version / write it out).
- **Written card**: `llm.write_up(question, context)` — one completion at `long` with a card prompt (markdown:
  `##` sections, bullets/steps/table when the material is structured, a final "Sources" line with `title · part n/N`
  or the page/Wikipedia title), using the same context the last answer had (recall/deeper/focus blocks; the last
  turn's sources are kept in `_turn_sources`). Emitted as a new WS message `card {id, markdown, sources, title}`
  after a one-line spoken reply; the card text is appended to `_conversation` as an assistant turn (so follow-ups
  see it) but `_session_turns` get a one-line stand-in ("wrote up: <title>") so memory folding stays cheap.
- Triggers: spoken/typed "write it up…" after any answer; the offer's "write it up" reply; typed `/write`.
- **Save to wiki**: action `save_card {id, folder?}` → `wiki.create_page` + `save_page` with the card markdown under
  `notes/<date>-<slug(title)>` (folder override allowed for user paths), frontmatter `source: written answer`,
  `sources:` list from the card; one commit; reply `wiki_open` + `card_saved {id, path}`; spoken "save that as a
  note" maps to the last card. No card → error {scope: wiki} "nothing to save".
- Protocol rows for `card`, `save_card`, `card_saved`; fixture hooks; focus/deeper/new-topic interplay: `new_topic`
  clears the last card; cards survive reconnect replay (stored as assistant turns with a marker).

### Decisions
- Length by origin + groundedness (deterministic) over letting the model pick: reliable on a 4-bit model.
- Card ≠ note: the transcript is ephemeral (memory folds beats), the wiki is durable and explicit — the user's
  web-chat pain was abandoned chats; this keeps drafts cheap and notes deliberate.
- Cards reuse the wiki renderer and validated writes — no new markdown or write path.

## Test scenarios
1. Typed "how do I get the Compound Key at Covenant?" (grounded) → completion called with the `long` budget, reply
   has no trailing offer; typed "hi" → normal budget; spoken same grounded question → short + "dig deeper, or write
   it up?".
2. Spoken "write it up" after a grounded answer → one short spoken line + a `card` message whose markdown has ≥ 2
   `##` sections and a Sources line naming the source and part; `_session_turns` holds "wrote up: <title>";
   `_conversation` holds the card text as an assistant turn.
3. Fixture: `card` renders collapsed with "show all", Copy puts the markdown on the clipboard stub, Save opens the
   folder picker defaulting to notes/ and sends `save_card {id, folder}`; `card_saved` turns the button into
   "saved → notes/…"; the card body is not passed to TTS (no `speech` message for it).
4. `save_card` → `notes/<date>-<slug>.md` exists with `source: written answer`, `sources:` list, one commit;
   `wiki_open` sent; saving twice is refused ("already saved"); `save_card` with no card → error scope wiki.
5. "save that as a note" spoken after a card → same as 4; before any card → the error.
6. `new_topic` clears the last card (a later "save that" errors); reconnect replay shows the card as a card.
7. Battery: re-run tests/qa/stress.py (extend with 4 write-up/save turns); no regressions in sections A–E and the
   new turns pass.
8. `test_protocol_doc.py` green with the three new rows.

## Out of scope
Auto-saving cards, editing a card in place (use the wiki editor after saving), cards for chit-chat, PDF/export.
