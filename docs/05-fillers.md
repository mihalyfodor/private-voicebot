Status: implementation

# 05 — Tool-intent fillers

## Overview

When a reply requires a tool (weather, news, email, time) the answer is delayed by the tool call plus a second generation. Play a short, cached, tool-specific spoken filler ("Let me check outside.") the moment the tool call is detected, with a `thinking` expression, so Haru reacts instantly and the wait reads as natural. No filler on plain conversational turns — those already stream their first sentence in about a second.

## User stories

- As a user, when I ask about weather/news/email/time, Haru immediately says something relevant while she fetches it.
- As a user, I don't hear the same filler phrase twice in a row.
- As a user, Haru doesn't say "let me check" twice (filler + answer).

## UI/UX

Flow for a tool turn: transcript(user) → state `speaking` → filler chunk (emotion `thinking`) → answer chunks → `speech_end`. Filler text is not added to the transcript. Non-tool turns are unchanged.

## Technical approach

- `fillers.py` (new): `FILLERS: dict[tool_name, list[str]]` with 3–4 short phrases per tool (plus a `default` list for unknown tools). `pick(tool_name)` returns a phrase, never the one used last for that tool.
- `llm.ask_events()` (new): yields `("tool_calls", [names])` before executing tools, then `("delta", text)` chunks. `ask_stream()` becomes a filter over it (deltas only) so `ask()`, memory and tests are untouched.
- `chatbot.speak_stream()` consumes events. On `tool_calls` it emits a `speech` message with emotion `thinking` for `fillers.pick(first_tool)`, using a wav cache (`_filler_wavs`) filled lazily on first use so repeats cost nothing.
- System prompt: add "Never announce that you are checking or looking something up; state the result directly."

### Decisions

- Chose tool-intent detection over an extra LLM "micro-reaction" call because it is context-aware at zero added latency; a parallel LLM call on the 26B would delay the real answer.
- Chose no generic filler for non-tool turns because measured first-sentence latency is already ~1 s.
- Filler wavs are synthesized on first use rather than at startup to keep startup fast; the cache is per-process.
- Risk: filler and answer overlap semantically; mitigated by the prompt rule and by keeping fillers about the *action*, not the result.

## Data model / state shape

`fillers._last: dict[tool, phrase]`; `chatbot._filler_wavs: dict[phrase, bytes]`. WS protocol unchanged (`speech` message reused, emotion `thinking`).

## Test scenarios

1. `fillers.pick("get_weather")` twice → two different phrases; both from `FILLERS["get_weather"]`.
2. `fillers.pick("unknown_tool")` → phrase from `FILLERS["default"]`.
3. `llm.ask_events()` (mocked client, tool round) → first event `("tool_calls", ["get_time"])`, then deltas; `ask_stream()` yields only the delta strings.
4. `chatbot.speak_stream` with events `[("tool_calls",["get_weather"]), ("delta","[neutral] It is sunny.")]` → messages: `state speaking`, `speech`(thinking, filler), `speech`(neutral, "It is sunny."), `speech_end`; returned transcript is `"It is sunny."` (no filler).
5. Same as 4 run twice → `tts` called once for the filler across both runs (cache hit).

Manual: ask "what's the weather?" → Haru says a filler with a thinking face within ~1 s, then the report; ask again → different filler.

## Out of scope

- LLM-generated reactions; fillers on non-tool turns; per-user filler customisation.
