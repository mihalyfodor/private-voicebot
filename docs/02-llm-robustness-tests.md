Status: implementation

# PRD: LLM Tool-Trigger Robustness Tests

## Overview

Current tests check one ideal phrasing per tool. Real users phrase requests differently — indirect, casual, or ambiguous. This PRD covers a parametrized robustness suite that tries 6 natural-language variants per tool and asserts a minimum success rate rather than demanding perfection.

## User stories

- As a developer, I want to know how reliably each tool is triggered across varied phrasings, so I can tune the system prompt when coverage is poor.

## UI/UX

CLI only — `pytest tests/test_llm_robustness.py -v`. Output shows pass/fail per phrasing and a summary success-rate assertion per tool.

## Technical approach

**Stack:** pytest, same `llm.ask()` / `llm.get_last_tool_calls()` pattern as existing tests. Requires Ollama running locally — no mocks.

**Decisions:**
- `pytest.mark.parametrize` for individual variant tests — failures visible per phrasing
- Separate `test_<tool>_success_rate` function per tool asserts ≥ 70% of variants pass
- Individual variant tests marked `xfail(strict=False)` — flaky phrasings don't block the suite
- Threshold at 70% (4/6) to start; tighten as system prompt improves
- `setup_function` calls `llm.reset()` before each test (existing pattern)
- Assume: Ollama + gemma4:e2b running; if not, tests error (not fail) — acceptable

## Data model / state

No persistent state. Each test resets conversation via `llm.reset()`. Tool call results inspected via `llm.get_last_tool_calls()` returning `[{"name": ..., ...}]`.

## Test scenarios

### Weather (`get_weather`) — 6 variants, ≥ 4 must trigger
1. "what's the weather like?" → `get_weather` called
2. "is it going to rain today?" → `get_weather` called
3. "should I bring a jacket?" → `get_weather` called
4. "how's it looking outside?" → `get_weather` called
5. "tell me about the weather" → `get_weather` called
6. "what's the temperature right now?" → `get_weather` called

### Time (`get_time`) — 6 variants, ≥ 4 must trigger
1. "what time is it?" → `get_time` called
2. "what's the time?" → `get_time` called
3. "do you know what time it is?" → `get_time` called
4. "can you tell me the time?" → `get_time` called
5. "what time do we have?" → `get_time` called
6. "give me the current time" → `get_time` called

### News (`get_news`) — 6 variants, ≥ 4 must trigger
1. "what's in the news today?" → `get_news` called
2. "any news?" → `get_news` called
3. "what's happening in the world?" → `get_news` called
4. "catch me up on current events" → `get_news` called
5. "what are the headlines?" → `get_news` called
6. "anything interesting going on?" → `get_news` called

### Email (`get_emails`) — 6 variants, ≥ 4 must trigger
1. "check my emails" → `get_emails` called
2. "do I have any new emails?" → `get_emails` called
3. "what's in my inbox?" → `get_emails` called
4. "any messages for me?" → `get_emails` called
5. "check my inbox" → `get_emails` called
6. "read my email" → `get_emails` called

### No-tool (conversational) — 6 variants, ≥ 4 must trigger NO tool
1. "how are you doing?" → no tool calls
2. "tell me a joke" → no tool calls
3. "what do you think about AI?" → no tool calls
4. "say something interesting" → no tool calls
5. "I'm bored" → no tool calls
6. "thanks" → no tool calls

## Out of scope

- `get_news_detail` — requires chained state (follow-up PRD)
- Mocked / offline LLM testing
- Response quality / content testing
- CI integration (Ollama dependency)
