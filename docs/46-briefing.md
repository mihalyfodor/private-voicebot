Status: exploration

# 46 — Briefing ("what's new?")

## Overview
Ingests, page ops and memory folds happen while the user is away, and today the only way to see them is to open the
Wiki, Sources and Memory views one by one. This PRD adds one deterministic turn: "what's new?" gives a spoken 2–3
sentence recap of what changed since the last briefing (24 h on first use), ending with the PRD 43 offer; "write it
up" turns it into a card with Pages / Sources / Memory sections. Tight version only: no windows, no editorial.

## User stories
- As a user, I say "what's new?" (or "brief me", "anything new", "what changed", "what did I miss", typed `/brief`)
  and hear two or three sentences about the pages, sources and memory that changed since I last asked, then
  "Want me to dig deeper, or write it up?".
- As a user, I say "write it up" and get a card listing the changed pages, the new sources and the memory changes
  as wikilinks I can click; I can Copy it or Save to wiki like any PRD 43 card.
- As a user, asking again right away gets "Nothing new since <when>." with no wait for the model.
- As a user, the briefing is not padded with the assistant's own housekeeping (lint runs, `index.md`, `log.md`).

## UI/UX
- No new UI, no new protocol messages: the recap is a normal spoken assistant turn (`state` / `speech` /
  `transcript`), the card is the existing `card` message (`.msg.card`, wiki renderer, so `[[links]]` are clickable).
- Card shape: `## Pages` (new / edited / deleted, then one line "notes from our chats: [[a]], [[b]] (+n)"),
  `## Sources` (`[[sources/…]]` + kind), `## Memory` (added / updated / removed counts + two examples). Each section
  at most 8 items, then "+n more"; an empty section is left out.
- "<when>" reads "14:05" today, "yesterday at 14:05", else "12 Oct".
- LLM offline with something to report → the PRD 41 composer error (`llm_offline_text()`), nothing spoken.

## Technical approach
- **Routing** (`chatbot.py`): `_BRIEF` regex + `briefing_command(text)` next to `_NEW_TOPIC` / `new_topic_command`
  (`chatbot.py:362-367`) and `_SAVE_NOTE` / `save_note_command` (`chatbot.py:377-387`), same whole-turn, whitespace-
  normalised match, case-insensitive: optional "please", one of what's / what is / anything new, brief me,
  what (has) changed, what did I miss, `/brief`, optional ", please", trailing `?.!`. "what's new with you" does not
  match. Checked in `respond` (`chatbot.py:504`) after `save_note_command` (`chatbot.py:517`) and before the
  `services.llm_ok()` check (`chatbot.py:520`), so "nothing new" still answers when oMLX is down. The turn function
  `briefing(sess, turn, send_sync)` is modelled on `write_up` (`chatbot.py:419`): thinking state, `_arm_playback`,
  speak, `context_msg`, `_wait_for_playback_then_idle` (`chatbot.py:333`), errors to idle.
- **Gathering** `wiki.changes_since(ts) -> digest` under `vault_lock.LOCK` (re-entrant), lazily importing `sources`
  like `wiki.py:1036`:
  - Pages: `_git("log", f"--since={ts}", "--name-status", "--format=%x1e%s")` (`wiki._git`, `wiki.py:134`). Every
    writer makes one commit per write (`_commit`, `wiki.py:219`; page ops `wiki.py:649`; user edits `_user_commit`,
    `wiki.py:681`; ingest `sources.py:1265`), so the subject names the writer. `log.md` cannot be used: `_log`
    (`wiki.py:437`) writes the date only. Skip `wiki: lint` commits (`wiki.py:1459`), `index.md`, `log.md`, `raw/**`
    and `sources/**` (those are the Sources section). A page whose every in-range commit is a page-op batch with a
    `(session …)` source (`wiki.py:623`, `:1553`) goes into `chat_pages`. Git missing or not a repo → no pages.
  - Sources: `sources.manifest_entries()` (`sources.py:740`) with `ingested_at` (`sources.py:1259`) > `ts`; old
    entries that keep the date and time apart are joined the way `_when` does (`sources.py:1280`).
  - Memory: `memory.OPS_LOG_PATH` lines (`_log_ops`, `memory.py:1222`) with `ts` > `ts`, ADD / UPDATE / DELETE
    only; records whose `reason` is in `UI_REASONS` (`memory.py:84`, the user's own edits) are skipped.
- **Recap**: empty digest → `say_canned("Nothing new since <when>.")`, zero LLM calls, cutoff unchanged. Otherwise
  `llm.briefing(digest) -> (reply, card_md)`: one completion (short budget) whose user message is the digest text
  wrapped in `untrusted(text, "briefing")` (`untrusted.py:21`) and whose system prompt carries `DOCUMENT_RULE`, asked
  for 2–3 sentences; `DEEPER_OFFERS[0]` (`llm.py:177`) is appended when `ends_with_offer` (`llm.py:577`) is false.
- **Write-up**: `llm.briefing` sets `_last_context` (`llm.py:304`; shape as set at `llm.py:1010`) to
  `{"question": user_text, "extra": <wrapped digest>, "tools": [], "reply", "sources": [], "pages": [titles],
  "card": card_md}` and `_deeper_offer = None` (`llm.py:293`). The existing `wants_writeup` (`llm.py:678`) →
  `chatbot.write_up` → `llm.write_up` (`llm.py:713`) path runs unchanged except one branch: a context with `card`
  returns that markdown as the card without a completion (title "Briefing <date>"). `_conversation` gets the user
  line and the recap; `_session_turns` gets "briefed: <n> pages, <m> sources, <k> memory changes" only.
- **Cutoff** `briefing_at` in `settings.json` via `avatars.load_settings` / `save_setting` (`avatars.py:48`, `:64`).
  `now` is taken with `memory._stamp("%H:%M:%S")` (`memory.py:197`) at gather time — the same clock as `ingested_at`
  and the ops `ts`; unset → `now − 24 h`. Saved as `max(old, now)` only after `speak_stream` returned for the
  recap; nothing new, LLM offline or a failed completion leave it unchanged.
- **PRD 45 coexistence**: the briefing is a deterministic command matched before `llm.ask_events`, so it bypasses
  `recall_for`, `recall_band` and the medium-band offer entirely. It clears any pending `_deeper_offer` (offers live
  one turn anyway) and writes `_last_context` in the PRD 43 shape plus the optional `card` key, which PRD 45 does not
  read. A "yes" / "dig deeper" after the recap is an ordinary turn (band applies), with the recap in history.

### Decisions
- Chose a deterministic card built from the digest over the `CARD_PROMPT` completion because the sections and
  wikilinks must be exact and a second 4-bit completion would reshuffle them; cost: one branch in `llm.write_up`.
- Chose git subjects for pages over `log.md` because `log.md` has dates only and every write is one commit.
- Assume one clock (`memory._stamp`) is enough for tests: they seed `briefing_at`, write explicit `ts` /
  `ingested_at` lines and set `GIT_COMMITTER_DATE` (inherited by `wiki._git`); if a shifted `_today` breaks the git
  side, revisit with a `memory._now()` seam.
- Risk: recap quality on the 4-bit model; mitigation: the digest is structured, capped (8 per section) and the
  model only phrases it; the card never depends on it.
- Risk: "what's new" said as small talk triggers a briefing; accepted — "what's new with you" does not match.

## Data model
- `settings.json`: `briefing_at: "YYYY-MM-DDTHH:MM:SS"` (local, `memory._stamp` form). Invariant: only ever
  replaced by a later value, and only after a spoken recap.
- Digest: `{"since": ts, "pages": [{"path", "title", "change": "new"|"edited"|"deleted"}], "chat_pages": [path],
  "sources": [{"page", "path", "kind", "at"}], "memory": {"added", "updated", "removed", "examples": [{"op",
  "path", "value"}]}}`; empty = no pages, no chat pages, no sources, all counts 0.
- `llm._last_context` gains an optional `card` (markdown); absent everywhere else.

## Test scenarios
1. `tests/wiki/test_wiki.py`: commits before and after the cutoff (`GIT_COMMITTER_DATE`) → `changes_since` lists
   only the later pages with `new` / `edited` / `deleted`; a `wiki: lint` commit, `index.md` and `log.md` never
   appear; a page touched only by a `(session …)` batch is in `chat_pages`, not `pages`.
2. `tests/wiki/test_wiki.py`: manifest entries with `ingested_at` before / after the cutoff (one old split
   date + time entry) → only the later ones in `sources`; ops lines with `reason: "ui"` and `CONFIRM` are not
   counted, two ADDs give `added: 2` and two examples; 10 changed pages → 8 items + "+2 more" in the card.
3. `tests/session/test_turns.py`: "what's new?", "Brief me.", "what did I miss", `/brief` each start a briefing;
   "what's new with you" reaches `llm.ask_events`. With changes: exactly one completion, its user message holds
   the digest inside `<document source="briefing">`, the spoken transcript ends with "Want me to dig deeper, or
   write it up?", and `briefing_at` equals the gather-time stamp.
4. `tests/session/test_turns.py`: nothing changed → transcript "Nothing new since <when>.", 0 completions,
   `briefing_at` unchanged; same with the LLM offline.
5. `tests/session/test_turns.py`: changes + LLM offline → error {scope: turn} with `llm_offline_text()`, 0
   completions, `briefing_at` unchanged; a completion that raises → idle, `briefing_at` unchanged; a seeded
   `briefing_at` later than now is never moved back.
6. `tests/session/test_turns.py`: a pending PRD 45 / PRD 43 `_deeper_offer` before the briefing → `None` after it;
   no recall block in the completion; `_session_turns` holds only the "briefed: …" stand-in.
7. `tests/llm/test_written_answers.py`: "write it up" after a briefing → a `card` with `## Pages`, `## Sources`,
   `## Memory`, the page titles as `[[wikilinks]]`, and no extra completion; Save to wiki files it as a PRD 43 note.
8. Smoke pass (`/smoke`) with one live "what's new?" turn on a vault with a fresh ingest: 2–3 sentences + offer.

## Out of scope
Recent questions (session.log holds only user text), explicit windows ("this week", "since Monday"), a palette
entry, LLM editorial over page contents, PDF / reMarkable output, scheduled auto-briefing, a rail badge, Gmail /
news, a briefing settings toggle.
