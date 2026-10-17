Status: implementation (signed off 2026-10-17)

# 57 — Context hygiene: stub old tool results before dropping turns; page focus only when the question is about the page

## Overview
Two leaks fill the 16k budget with text the model no longer needs. Old tool results (a 1–4k-char `read_page` or
`get_news` body) stay in `_conversation` until `trim_history` drops the whole turn they belong to, taking dialogue
with them. And the open Wiki/Sources page replaces recall on every turn (`agent/turn.py:117`), so "what's the
weather" with a page open gets a 1.2k-token page block and no weather. Design source: `docs/reviews/2026-10-17-session-handling.md`
Part B §2 and §5. Session boundaries and the stale prompt are PRD 56.

## User stories
- As a user in a long session, I lose old tool bodies before I lose old conversation; "what did we say earlier"
  still works long after the bodies would have forced a drop.
- As a user with a page open, "what's the weather" or "who painted the Mona Lisa" gets a normal answer; "summarise
  this" and "what does it say about dragons" still answer from the page.

## Goals / Non-goals
- Goals: the oldest-turn drop becomes the last resort; a page block rides on a turn only when the question points at it.
- Non-goals: no model call, no summary, no new setting, no UI change, no change while under budget.

## Technical approach

### 1. Stub old tool results (`agent/context.py`, `trim_history`)
- **Mechanism.** After `budget` is computed and only if `estimate_tokens(_conversation) > budget`, one pass
  stubs every `role: tool` message that sits before the second-last `role: user` message (so the current turn and
  the one before it are kept: `KEEP_TOOL_TURNS = 2`) and whose content is longer than `STUB_MIN_CHARS = 400`.
  Then the existing `while` loop runs unchanged, so it only drops turns if the stubs were not enough. Under budget
  the function does nothing, as today.
- **Constants.** `KEEP_TOOL_TURNS = 2`, `STUB_MIN_CHARS = 400`, both module-level in `agent/context.py`.
- **Stub text, exact:** `[<tool> result removed to save room — call <tool> again if needed]`. `<tool>` comes from the
  assistant message whose `tool_calls[].id` equals the tool message's `tool_call_id` (`agent/turn.py:38-66` writes
  both); an unmatched id reads `tool`. The assistant `tool_calls` (name and arguments, e.g. the page path) stay, so
  the pairing is never orphaned and the model sees what to re-call. A stub is under 400 chars, so it is never
  stubbed twice.
- **Counters.** A stub is not a dropped turn: `_last_trimmed` and the meter's `trimmed` count stay 0 when only stubs
  were needed; the log line says `stubbed N tool results`.
- **What depends on tool bodies, and why it is safe** (checked in code):
  - Memory / fold and the wiki ingest read `_session_turns` (`agent/turn.py:155,270`), which holds dialogue only.
  - The write-up uses `_last_context["tools"]` = `tool_texts` (`agent/turn.py:160,234,272`), its own copy.
  - Reconnect replay shows cards and dialogue; it never reads `role: tool` messages.
  - Server state such as `tools/news.py` `LAST_RESULTS` (used to resolve "the second headline") is not in `_conversation`.
  - The recall block is already stripped after each turn (`agent/turn.py:267`), so it is not a stub target.
- **Follow-ups.** "More on the second headline" right after the news is inside the kept 2 turns: body intact. Later,
  the model sees the stub and calls `get_news` again; `LAST_RESULTS` is refreshed by that call. After a stubbed
  `read_page`, the kept `read_page(path)` call lets the model re-read; the re-call appends a fresh full body in the
  current turn, which is kept. Small results (time, weather, short lists) are under the floor and stay.
- **Cache.** oMLX's prefix cache breaks at the first stubbed message once; stubbing all eligible results in one
  pass keeps it to one break per pressure event, and later turns stay stable until the next one.

### 2. Focus only when the question points at the page (`agent/recall.py`, `focus_block`)
- **Rule.** `focus_block(question)` returns the block only if the view/page checks pass (unchanged) and one of:
  - (a) **deictic**: the lower-cased word tokens of the question include one of `DEICTIC = {this, that, it, here,
    page, article, summarise, summarize, summary, tldr, recap}` or the phrase `tl;dr`
    (matched before tokenising); or
  - (b) **overlap**: `v_chunks.topic_terms(question)` shares at least one term with `v_chunks.content_terms(title +
    " " + path stem + " " + the page's `#` headings)` minus `GENERIC_TITLE_WORDS`.
  Exception: a question whose content terms are all in `COMMAND_WORDS` ("what time is it", "any news") never
  focuses, deictic or not. The gate runs before `conversation._turn_pages.append`, so a declined turn records no page.
- **Declined.** `focus_block` returns `""`; `turn.py:117` already falls through to the normal recall / band path
  (`elif asked` / `origin == "typed"` / spoken band). The page stays open and `focus` is untouched, so the next
  question about it focuses again. Closing the page or switching to Chat stays the explicit off switch.
- **Examples** (Fallout 4 guide open, headings include "Settlements", "Turrets"):

  | Utterance | Focus? | Why |
  |---|---|---|
  | "summarise this" | yes | deictic |
  | "what does it say about dragons" | yes | deictic ("it") |
  | "what about turrets" | yes | overlap with a heading |
  | "how do I set up a settlement" | yes | overlap ("settlement" stems to the heading term) |
  | "what's the weather" | no | no deictic, no overlap, command words |
  | "who painted the Mona Lisa" | no | no overlap |
  | "what time is it" | no | deictic "it" but command words only |
  | "tell me more about the Moon landing" | no | no overlap |

## Scenarios
1. Over budget with an old 4k `read_page` result: it becomes the exact stub, the user turns are all kept, and the trim count is 0.
2. A tool result after the second-last user message is untouched, even over budget.
3. A 399-char tool result before that point is untouched; a 401-char one is stubbed.
4. After stubbing, the assistant `tool_calls` message and the `tool_call_id` pairing are intact (no orphan, tool name in the stub).
5. Under budget nothing changes, however old and long the results are.
6. Still over budget after stubbing: the oldest-turn drop runs, and the stubbed results count as already small.
7. "Summarise this" with a page open: the turn's user message carries the focus block and `_turn_pages` holds the title.
8. "What about turrets" (shares a heading term, no deictic): focus block.
9. "What's the weather" and "who painted the Mona Lisa" with a page open: no focus block, normal recall runs, `focus["path"]` is unchanged.
10. "What time is it" with a page open: no focus block despite "it".
11. The turn after a declined one, "what does it say about dragons": focus block again.

## Tests
- `tests/agent/test_context.py` (style of `test_trim_history_never_orphans_tool_result`, monkeypatched `CONTEXT_BUDGET`):
  `test_trim_stubs_old_tool_result_before_dropping_turns` (1), `test_trim_keeps_tool_results_in_the_last_two_turns` (2),
  `test_trim_stub_floor_is_400_chars` (3), `test_trim_stub_keeps_tool_call_pairing` (4),
  `test_trim_under_budget_leaves_tool_results` (5), `test_trim_drops_turns_only_if_stubs_not_enough` (6).
- `tests/vault/test_focus_context.py` (existing `seeded_vault`, `a_recall.set_focus`, `typed` / `last_user` helpers):
  `test_deictic_question_gets_the_page` (7), `test_heading_overlap_gets_the_page` (8),
  `test_unrelated_question_skips_focus_and_recalls` (9), `test_command_words_never_focus` (10),
  `test_focus_returns_on_the_next_page_question` (11). Existing focus tests must stay green; any whose question has
  neither deictic nor overlap is fixed to point at the page.
- Gate: `scripts/check.sh`; smoke for the one LLM path (page open + "what's the weather" calls `get_weather`).

## Out of scope
- Summarising tool results, or per-tool stub policies beyond the 400-char floor.
- Eager stubbing after every turn (PRD 39 does that for recall only).
- Embedding similarity for focus; dropping focus after N turns.
- Subtracting the tool schemas from the trim budget (a separate `Fix:`), the stale prompt, idle/day boundaries, "new topic" folding (PRD 56).

## Decisions
- Chose stubbing under pressure over eager stubbing: nothing changes while there is room, the cache stays warm, and an immediate follow-up keeps its body.
- Chose keep 2 turns: the current turn plus one follow-up; "tell me more about the second one" is nearly always the very next turn.
- Chose 400 chars: below it a result costs less than the stub's own ~25 tokens of explanation is worth; time, weather and snippet lists live there. Real bodies are 1–4k.
- Chose a re-call stub over a summary: the call and its arguments stay, a re-call is one tool round, a summary is a model call that buys nothing.
- Chose word overlap plus a short deictic list over a model call: deterministic, testable, free, and reuses `topic_terms` / `content_terms`.
- Chose "decline, keep the page open" over closing the page: the user can see it open; recall may find the same page anyway.

## Risks
- A paraphrase with no shared word and no deictic ("how do I defend my base" on a turrets page) falls to recall: today's behaviour without focus. Mitigation: the user says "this page".
- "that" / "it" inside an unrelated sentence still focuses ("I heard that it will rain"). Mitigation: accepted; the page note says to answer from it only if it applies.
- A stubbed result the model needed forces a re-call (one tool round). Mitigation: only outside the last 2 turns, and only under pressure.
- Title-term overlap on a generic word is filtered by `GENERIC_TITLE_WORDS`; a one-word title page focuses on any question that uses it, which is intended.

## Size
S + S (about 25 lines in `agent/context.py`, about 20 in `agent/recall.py`, 11 tests).

## Files
`agent/context.py`, `agent/recall.py`, `tests/agent/test_context.py`, `tests/vault/test_focus_context.py`; `docs/index.md` gains this PRD (enforced by `tests/test_index.py`).
