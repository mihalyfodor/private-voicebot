# Review 2026-10-17 — session handling: boundaries, compaction, fresh prompt, focus

Scope: the six session-handling problems from today's read-only review, checked against how comparable
systems solve them (Part A), and a design for this app (Part B). No code was changed. Next free PRD
number: 56.

The problems, as confirmed in the code:

1. The system prompt is stale. It is rebuilt only by `new_thread`, wiki/editor actions and location/verbosity
   changes (`agent/conversation.py` `_rebuild_system_prompt` / `refresh_context`, called from
   `server/turns.py:206,421`, `server/actions/wiki.py:20`, `server/actions/settings.py:79`).
   `_periodic_memory` (`server/panels.py:109`), `_action_memory_op`/`_ops`/`_reset`/`_bridge`
   (`server/actions/memory.py`) and the shutdown save never rebuild it. So `Today's date`, `local time HH:MM`
   (`agent/prompt.py:143,164`) and `<user_profile>` stay frozen.
2. Tool results never shrink. `read_page` returns a whole page, `get_news_detail` up to 4,000 chars
   (`tools/news.py:119`) and Wikipedia up to 3,000 per section (`tools/wikipedia.py:24`). They stay in
   `_conversation` until `trim_history` drops the whole turn, while recall blocks are stripped after the
   turn (`agent/turn.py:267`).
3. There is no idle or day boundary. The fold runs every `MEMORY_SAVE_EVERY=5` completed turns
   (`server/session.py:124`), on "new topic", on Memory-tab open/Save, and at shutdown. A WebSocket
   disconnect (`server/ws.py:193`) does nothing to memory.
4. "New topic" blocks. `new_thread` runs `save_memory()` (2–4 model calls) synchronously before
   `_summarise` (`agent/conversation.py:190-211`).
5. Focus is sticky. With a Wiki/Sources page open, `focus_block` replaces recall on every turn
   (`agent/turn.py:111`, `agent/recall.py:86-121`), whatever the question.
6. The trim and the meter disagree. `trim_history`'s budget is `context_budget() - max_tokens`
   (`agent/context.py:84`), but `context_usage` adds `registry.tokens()` (~1.1k) (`agent/context.py:102`).
   After a trim the meter reads ~104 %.

These are fine as they are and get no proposal: the per-turn reset of recall and offers, rollback of failed
turns, replay on reload, and the 90 s shutdown save.

---

## Part A — what comparable systems do

### (a) Session boundaries for an always-on assistant

- **Home Assistant Assist** keeps a chat session in memory with `CONVERSATION_TIMEOUT = timedelta(minutes=5)`.
  Every use refreshes `last_updated`, and a cleanup task drops sessions older than that. An expired
  `conversation_id` silently gets a new session [1][2]. This suits a command assistant ("turn off the
  light", then "and the kitchen"). It is far too short for a chat companion.
- **Pipecat** (voice pipelines) has `user_idle_timeout` on the user aggregator. It fires `on_user_turn_idle`
  and the app decides what to do; the timer is suppressed during tool calls [3]. The design point is that
  idle is an event, and the policy belongs to the app.
- **Mem0** extracts memories on every `add()` call (`infer=True`: one LLM pass per call), so there is no
  session-end step at all [4]. **Letta** runs "sleep-time" (dreaming) agents off the critical path. They are
  triggered "after a set number of completed agent steps or when the context window is compacted" [5].
  **Claude.app** memory is synthesised from chat history on a daily cadence [6]. All three take
  consolidation off the user's critical path and run it on a step count, a compaction, or a schedule. This
  app already does the step count (every 5 turns). What it lacks is the "nobody is talking" trigger.
- **ChatGPT/Claude apps** never auto-split a thread. "New chat" is always explicit, and continuity across
  chats comes from memory, not from the transcript. That fits a typed app with a sidebar of threads. A
  voice app with one implicit thread has no sidebar, so a day boundary is the natural place for a cut.

### (b) Compacting tool results and old turns

- **Anthropic context editing**, `clear_tool_uses_20250919` [7]: `trigger` (default 100k input tokens) sets
  when clearing starts, `keep` (default the last 3 tool uses) sets what survives, and `clear_at_least`
  makes sure one clearing frees enough to be worth the cache break. The oldest results go first. Each one is
  replaced by a placeholder saying it was removed, and the tool *calls* stay visible
  (`clear_tool_inputs=false`). The docs say plainly that clearing invalidates the cached prefix from the
  cleared point on.
- **Anthropic, "Effective context engineering"** [8]: tool-result clearing is "one of the safest, lightest
  touch forms of compaction". Full summarisation comes next, and just-in-time re-retrieval through
  lightweight references beats keeping bodies in context.
- **Claude Code** auto-compacts near the window limit (third-party sources put it at ~83–95 %), and
  `/compact` summarises on demand [9, unverified]. **OpenAI Responses** offers server-side compaction above
  a `compact_threshold` [10]. **LangChain** `ConversationSummaryBufferMemory` keeps the recent messages within
  `max_token_limit` (default 2,000) and folds older ones into a running summary [11]. It is deprecated in
  favour of a summarisation middleware with trigger/keep settings.
- Takeaway for 16k: the industry ladder is (1) clear old tool results, (2) summarise, (3) drop turns. This
  app has (3), and (2) only on demand ("new topic"). Adding (1) before (3) is the cheap win. An automatic
  (2) is not worth it here (see Part B).

### (c) Keeping dynamic prompt fields fresh

- **Prompt-cache ordering** [12]: the prefix is tools → system → messages. Any change invalidates that level
  and everything after it. Anthropic's own example puts `Current time: …` in the *user message after the
  cache breakpoint*, never in the cached system block.
- **oMLX**, the server this app uses, is built around a paged KV cache with a RAM tier and an SSD tier. A
  returning prefix is restored block by block instead of being recomputed. The author reports
  time-to-first-token dropping from 30–90 s to ~1–3 s on long contexts [13][14]. mlx-lm's server has an LRU
  prompt cache with longest-common-prefix reuse [15]. Either way, a changed token in the system prompt
  forces a re-prefill of the entire conversation behind it.
- Consequence: `local time HH:MM` near the top of the system prompt is the worst possible field to keep
  fresh. Refreshing it every turn would throw away the whole cached prefix (~4–8k tokens) on every turn.
  Leaving it unrefreshed is what makes it stale. The date changes once a day and costs one re-prefill per
  day. The profile changes only when a fold applies ops.

### (d) Sticky "focus" / attached-document scope

- **Open WebUI** has two modes per attached file: "Focused Retrieval" (the default; only the chunks relevant
  to each prompt) and "Full Context" (the whole document every turn). It advises keeping Full Context off
  for small local models [16]. LM Studio's analogue sends small files whole and retrieves from large ones.
- No primary source documents an automatic topic-shift drop of a pinned file. ChatGPT, Claude and Cursor
  `@file` pins all persist until the user removes them, and *retrieval per turn* is what keeps them cheap.
  So the established pattern is "pinned scope, retrieved per question", not "pinned scope, always injected".
  This app already retrieves per question (best section/chunk). What it lacks is a "this question isn't
  about the page" exit.

---

## Part B — design for this app

### 1. Stale system prompt → rebuild on profile change; take the clock out of the prompt

- **Mechanism.**
  - Remove `local time HH:MM` from `location_line()`. Keep `The user is in <name> (timezone <tz>)` and
    keep `Today's date` (day granularity).
  - Make `get_time` return the date as well (`Saturday, 17 October 2026, 14:05 (Europe/…)`). Today it
    returns `%H:%M:%S` only, and the prompt rules already make the tool mandatory for time.
  - Rebuild the system prompt (a) after a fold that applied ops, inside `save_memory` when `applied` is
    truthy, and (b) after every Memory-tab write (`memory_op`, `memory_ops`, `memory_reset`, `_bridge`
    actions → `refresh_context`).
  - The date is refreshed by the day-boundary check (#3).
- **Why.** The clock is the only field that changes every minute, and a tool already serves it. Keeping it
  out keeps the prefix stable for oMLX's cache (Part A c). The profile changes rarely, and paying one
  re-prefill when it does is the correct trade.
- **Rejected.**
  - Rebuilding every turn: it would break the cache each turn and take seconds of prefill.
  - Injecting the time into the volatile user tail (Anthropic-style): it adds tokens every turn, and the
    model parrots the time unprompted. It is not needed while `get_time` exists.
- **Bug to fix on the way.** `_rebuild_system_prompt(note="")` drops the avatar-switch note that `set_avatar`
  adds. Any rebuild after a switch loses it, and today `refresh_context` already does. Keep the note in a
  module global that `reset`/`new_thread` clear.
- **Files.** `agent/prompt.py`, `tools/time.py`, `agent/conversation.py`, `server/actions/memory.py`.
- **Size.** S.
- **Tests.**
  - After `save_memory` applies an op, `_conversation[0]` contains the new profile value.
  - After `memory_reset`, `<user_profile>` is gone from `_conversation[0]`.
  - `build_system_prompt` contains no `HH:MM` pattern.
  - `get_time` output contains the weekday and the year.
  - After `set_avatar` + `refresh_context`, the switch note survives.

### 2. Tool results → clear old ones before dropping turns

- **Mechanism.** In `trim_history`, before the turn-dropping loop, run one pass that stubs every `role: tool`
  message that sits before the last `KEEP_TOOL_TURNS = 2` user messages and is longer than
  `STUB_MIN_CHARS = 400`.
  - Stub text: `[<tool name> result removed to save room — call <tool name> again if needed]`. The name
    comes from the matching `tool_call_id`, and the assistant `tool_calls` (including arguments such as the
    page path or URL) stay.
  - All eligible results are stubbed in one pass, which is the `clear_at_least` idea: one cache break, not
    one per turn.
  - Only if the conversation is still over budget does the oldest-turn drop run, as today.
- **Why.**
  - It mirrors Anthropic's `clear_tool_uses`: trigger = our budget, keep = 2 turns, calls stay visible.
  - Nothing is cleared while there is room, so the cache and follow-ups are untouched in short sessions.
  - A pressured session loses 1–4k-char bodies instead of whole dialogue turns.
  - Memory never saw tool results (`_session_turns` holds dialogue only), the write-up keeps its own copy in
    `_last_context["tools"]`, and replay ignores tool messages. Nothing else depends on the bodies.
- **Follow-ups.**
  - "Tell me more about the second headline" is an immediate follow-up, inside the kept 2 turns. Later, the
    model sees the stub and re-calls `get_news`. `LAST_RESULTS` server-side is unaffected.
  - For a wiki page the stub leaves the `read_page(path)` call, so the model can re-read it.
  - Small results (time, weather, a search snippet list) stay under the 400-char floor.
- **Rejected.**
  - Stubbing eagerly after every turn, as PRD 39 does for recall: the cache breaks every turn and immediate
    follow-ups lose the body for nothing.
  - Summarising tool results: it costs model calls that buy nothing over a re-call.
- **Files.** `agent/context.py` only.
- **Size.** S.
- **Tests.**
  - A conversation over budget with an old 4k tool result gets stubbed and keeps all its user turns (no
    trim count).
  - A tool result inside the last 2 user turns is untouched.
  - A 200-char result is untouched.
  - The assistant `tool_calls` message and the `tool_call_id` pairing survive (no orphan).
  - Under budget, nothing changes.

### 3. Idle and day boundary → fold on idle and disconnect; new thread on the first turn of a new day

- **Mechanism.**
  - **Idle fold.** One asyncio timer, re-armed after every completed turn: `IDLE_FOLD_S = 600` (10 min).
    When it fires, if `has_unsaved_turns()` and the controller is not busy, run the same path as
    `_periodic_memory` on the hook thread, with the same overlap guard.
  - **Disconnect fold.** The WebSocket `finally` in `server/ws.py` triggers the same fold when there are
    unsaved turns.
  - **Day boundary.** It is checked lazily at the start of a turn, because a timer cannot fire while the
    laptop sleeps. If the local date of the last completed turn ≠ today *and* the gap is ≥ 60 min, the turn
    first runs `new_thread()` (with the background fold from #4), sends the `thread` summary to the page,
    then answers.
  - A date change with a gap under 60 min (chatting across midnight) only refreshes the system prompt, so
    the date is right.
- **Why.**
  - Folding on idle and disconnect covers the "closed the laptop with 4 turns unsaved" case without waiting
    for turn 5. Pipecat's model applies: idle is an event and the app picks the action. Letta/Claude
    consolidate off the critical path the same way.
  - A *new thread* at idle is wrong for a companion. HA's 5 min suits commands, and a mid-afternoon pause
    shouldn't wipe the context.
  - The day is the natural cut for a voice app with one implicit thread. Yesterday's gist survives as
    `<earlier>` plus the profile.
  - The lazy check needs no scheduler and works after sleep.
- **Cost.** One summary call (≤150 tokens, ~2–4 s) on the first turn of the day. The "thinking" state
  covers it.
- **Do NOT build.**
  - An idle-triggered new thread within the same day.
  - A midnight timer.
  - A thread history or archive browser (PRD 39 put it out of scope; still right).
  - A configurable boundary UI. Env vars are enough.
- **Files.** `server/session.py` or `server/panels.py` (the idle timer and the fold helper), `server/ws.py`
  (disconnect), `server/turns.py` (the day check before the turn), `agent/conversation.py` (`last_turn_at`
  stamped on turn completion and cleared on reset).
- **Size.** M.
- **Tests.**
  - With a fake clock, 10 min after the last turn with unsaved turns, `save_memory` is called once; with
    no unsaved turns, it is not called.
  - On disconnect with unsaved turns, the fold is called.
  - Last turn yesterday 22:00 and a new turn at 08:00 today: `_conversation[1]` is an `<earlier>` message,
    a `thread` frame is sent, and the system prompt has today's date.
  - Last turn 23:58 and a turn at 00:03: no new thread, but the date in the prompt is updated.

### 4. "New topic" blocks → summarise first, fold in the background

- **Mechanism.** `new_thread` no longer calls `save_memory()`.
  - It takes `dialogue()`, summarises (1 call), swaps `_conversation`, rebuilds the prompt and returns.
  - The `new_topic` handler then speaks the ack and starts the fold on the hook thread through the same
    helper as #3, with the same overlap guard.
  - When the fold applies ops, #1 rebuilds the prompt of the new thread with the updated profile.
- **Rollback.** No new mechanism is needed.
  - `save_memory` snapshots `_session_turns`, which is separate from `_conversation`. Replacing the
    conversation does not touch what the fold folds.
  - On failure, `restore()` puts the turns back in front of `_session_turns` and the next periodic, idle or
    shutdown fold retries them.
  - If the hook is already running (a periodic fold in flight), the guard skips. The turns stay in
    `_session_turns` and the next trigger picks them up, so nothing is lost.
  - Turns spoken in the new thread before the fold finishes are appended *after* the snapshot, which is
    what `restore()` already handles (`agent/turn.py:252` deletes by identity for the same reason).
- **Why.** The user waits only for the summary (~2–4 s instead of 10–40 s). Folding after the summary also
  avoids the two calls competing on the local model.
- **Files.** `agent/conversation.py`, `server/turns.py`, `server/panels.py`.
- **Size.** S.
- **Tests.**
  - With a slow fake fold, `new_thread()` returns before the fold finishes.
  - A failing fold leaves the old turns in `_session_turns`, and `_conversation` is still the new thread.
  - A hook already running → no second concurrent `fold.save`.

### 5. Sticky focus → focus only when the question is about the page

- **Mechanism.** `focus_block(question)` returns the page block only when one of these holds:
  - (a) the question is deictic or page-directed: `this, that, it, here, the page, the article, the source,
    the guide, summarise, summary, explain, tl;dr, what does it say`, or
  - (b) it has topic overlap: at least one `v_chunks.topic_terms(question)` term occurs in the page text or
    title.
  Otherwise it returns `""` and the normal recall/band path runs, which may well find the same page anyway.
  The page stays "open" and the next question about it gets focus again.
  Closing the page or switching to Chat stays the explicit off switch (it exists).
- **Why.** This follows the pinned-but-retrieved-per-question pattern (Open WebUI Focused Retrieval, `@file`
  pins). Both checks reuse existing code (`topic_terms`, `best_section` already scores overlap), and the
  rule is deterministic and testable.
- **Rejected.**
  - Dropping focus after N turns: it is arbitrary, and the user can still see the page open.
  - Embedding similarity: heavier, and lexical is enough for a page-scoped check.
- **Risk.** A paraphrase with no shared word and no deictic falls to recall, which is today's
  behaviour without focus. Acceptable.
- **Files.** `agent/recall.py`, maybe `vault/focus.py`.
- **Size.** S.
- **Tests.** With the Skyrim page open:
  - "what's the weather" → no focus block, `get_weather` is called.
  - "summarise this" → focus block.
  - "what does it say about dragons" → focus block.
  - "who painted the Mona Lisa" → no focus block.
  - Eval battery: the focus section keeps its scores.

### 6. Trim and meter disagree → subtract the tool schemas

- **Mechanism.** `budget = context_budget() - prompt._effective_max_tokens() - registry.tokens()`. It is one
  line, and the comment at `agent/context.py:69` already names the gap.
- **Files.** `agent/context.py`.
- **Size.** XS.
- **Test.** After `trim_history()` on an overfull conversation, `context_usage()["pct"] <= 100`.
- This needs no PRD. It qualifies as a `Fix:` commit with a regression test first (process.md, "Small fix").

### What NOT to build

- Automatic summarising compaction (Claude Code / LangChain style). Stub-before-drop plus the explicit "new
  topic" and the daily rollover cover a 16k single-user app. A summary call on the critical path at 90 %
  costs more than it saves.
- A per-turn time line in the prompt (see #1).
- An idle new thread, a midnight timer, or a thread archive (see #3).
- Turn-count focus expiry, or embeddings for focus (see #5).
- Tool-result summarisation or per-tool stub policies beyond the 400-char floor (see #2).

---

## Suggested PRDs

- **Fix now (no PRD):** #6, the trim budget. XS, regression test first.
- **PRD 56 — Session boundaries** (#1, #3, #4). Scope:
  - fresh system prompt after an applied fold and after Memory-tab edits; avatar note kept
  - clock out of the prompt, date into `get_time`
  - idle fold (10 min) and disconnect fold
  - day-boundary new thread (date changed and gap ≥ 60 min)
  - "new topic" folds in the background after the summary

  Out of scope: a thread archive, an idle new thread, auto-summarising compaction. Size: M overall.
- **PRD 57 — Context hygiene** (#2, #5). Scope:
  - old tool results (outside the last 2 user turns, > 400 chars) stubbed in one pass before any turn is
    trimmed
  - focus applies only to deictic or overlapping questions, otherwise normal recall

  Out of scope: summarising tool results, embedding-based focus, turn-count expiry. Size: S+S.
- PRD 57 can go first. It is self-contained and touches only `agent/`, while 56 touches the server
  lifecycle.

## Summary

1. Take the clock out of the system prompt (`get_time` returns the date too), and rebuild the prompt
   whenever the profile changes. This keeps oMLX's prefix cache warm and the profile current.
2. Fold memory after 10 min idle and on disconnect. Start a new thread with a summary on the first turn of
   a new day (gap ≥ 60 min), never on idle alone.
3. "New topic" summarises first and folds in the background. The existing `restore()` already makes a
   failed fold safe to retry.
4. Before trimming any turn, stub old tool results (outside the last 2 turns, > 400 chars) in one pass,
   following Anthropic's `clear_tool_uses`.
5. Focus applies only to page-directed or overlapping questions. The trim budget subtracts the tool
   schemas, as a one-line `Fix:`.

## Sources

1. Home Assistant `helpers/chat_session.py` (`CONVERSATION_TIMEOUT`) — https://github.com/home-assistant/core/blob/dev/homeassistant/helpers/chat_session.py
2. Home Assistant developer docs, conversation agents — https://developers.home-assistant.io/docs/core/conversation/custom_agent
3. Pipecat, detecting user idle — https://docs.pipecat.ai/pipecat/fundamentals/detecting-user-idle
4. Mem0, add memories — https://docs.mem0.ai/core-concepts/memory-operations/add
5. Letta, sleep-time agents — https://docs.letta.com/guides/agents/architectures/sleeptime
6. Claude help center, memory — https://support.claude.com/en/articles/11817273
7. Anthropic, context editing — https://platform.claude.com/docs/en/build-with-claude/context-editing
8. Anthropic engineering, effective context engineering for AI agents — https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
9. Claude Code auto-compact threshold discussion (third-party, unverified) — https://github.com/anthropics/claude-code/issues/11819
10. OpenAI, compaction guide — https://developers.openai.com/api/docs/guides/compaction
11. LangChain `ConversationSummaryBufferMemory` — https://python.langchain.com/api_reference/langchain/memory/langchain.memory.summary_buffer.ConversationSummaryBufferMemory.html
12. Anthropic, prompt caching — https://platform.claude.com/docs/en/build-with-claude/prompt-caching
13. oMLX discussion on ml-explore/mlx — https://github.com/ml-explore/mlx/discussions/3203
14. oMLX — https://omlx.ai
15. mlx-lm persistent prompt cache PR (closed; context on server cache) — https://github.com/ml-explore/mlx-lm/pull/1405
16. Open WebUI RAG troubleshooting (Full Context vs Focused Retrieval) — https://docs.openwebui.com/troubleshooting/rag
