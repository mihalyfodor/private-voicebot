Status: exploration

# 13 — Stream the first LLM call

## Overview

`llm.ask_events` makes a non-streamed first call to detect tool calls; on the common no-tool path the whole reply is generated before the first sentence reaches TTS, so TTS never overlaps generation. Streaming the first call with `tools=` and accumulating `delta.tool_calls` removes most of the dead air (~0.6–1.2 s per turn).

## Technical approach

- `ask_events`: `create(..., tools=TOOLS, stream=True, max_tokens=...)`; for each chunk: text deltas are yielded immediately; `delta.tool_calls` fragments are accumulated by index (`id`, `function.name`, `function.arguments` concatenated). On stream end: if tool calls were accumulated → run `_tool_round`, then the second (streamed) answer call as today, discarding any pre-tool text (Gemma usually emits none); else the streamed text is the reply.
- Guard: if any tool-call fragment arrives after text was already yielded, the yielded text is still spoken (can't unsay it); log it. Measure on oMLX whether this happens; if it does, fall back to buffering the first ~20 tokens before yielding.
- Keep `ask()`/`ask_stream()` wrappers unchanged.

### Decisions
- Accept the tiny risk of speaking a pre-tool preamble over the guaranteed latency win; measure first.
- Assume oMLX streams OpenAI-style `tool_calls` deltas (verify with one probe before implementing; if not supported, keep the two-call design and close this PRD as won't-do).

## Test scenarios
1. Mocked stream with text-only deltas → deltas yielded incrementally in order; one create() call.
2. Mocked stream with tool_call fragments split across chunks → reassembled name/arguments; `run_tool` called once; second call streamed.
3. Rollback on exception mid-stream keeps `_conversation` clean (existing test extended).
4. Live probe (2026-08-28, gemma-4-26B on oMLX): tool call arrives as ONE `tool_calls` delta with `finish_reason="tool_calls"` after 1.14 s, no preceding text; no-tool reply: first text delta at 0.54 s, whole reply 0.71 s (oMLX emits coarse chunks). So the design is viable; expected gain ≈ 0.2 s on short replies, proportionally more on `long`.

## Out of scope
Parallel tool calls; changing the tool set.
