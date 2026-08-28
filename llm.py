import json
import os
from datetime import datetime
from typing import Iterator

from openai import OpenAI

import memory
import avatars
from splitter import EMOTIONS
from tools import TOOLS, run_tool

OMLX_BASE_URL = os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1")
OMLX_API_KEY = os.getenv("OMLX_API_KEY", "omlx")
OMLX_MODEL = os.getenv("OMLX_MODEL", "gemma-4-26b")
LOG_PATH = os.path.join(os.path.dirname(__file__), "session.log")

CONTEXT_BUDGET = int(os.getenv("CONTEXT_BUDGET", "16000"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "0"))  # 0 = derive from verbosity

VERBOSITY_RULES = {
    "short": "at most two sentences",
    "normal": "two to four sentences; a little longer only when actually explaining something",
    "long": "as long as the answer needs, but still spoken prose — no lists or headings",
}
VERBOSITY_MAX_TOKENS = {"short": 400, "normal": 400, "long": 800}

_verbosity = "normal"
_model_context_len_cache: int | None = None
_model_context_len_warned = False


def resolve_verbosity(avatar: dict) -> str:
    """settings.json > VERBOSITY env > avatar card > 'normal'. Invalid values warn and fall through."""
    sources = (
        ("saved verbosity", avatars.load_settings().get("verbosity")),
        ("VERBOSITY env value", os.getenv("VERBOSITY")),
        ("card verbosity", avatar.get("verbosity")),
    )
    for label, value in sources:
        if not value:
            continue
        if value in VERBOSITY_RULES:
            return value
        print(f"[llm] warning: invalid {label} {value!r}; ignoring")
    return "normal"


def current_verbosity() -> str:
    return _verbosity


def set_verbosity(value: str) -> None:
    global _verbosity
    if value not in VERBOSITY_RULES:
        raise ValueError(f"Unknown verbosity {value!r}; valid: {', '.join(VERBOSITY_RULES)}")
    _verbosity = value
    avatars.save_setting("verbosity", value)
    _rebuild_system_prompt()


def _effective_max_tokens() -> int:
    if MAX_TOKENS:
        return MAX_TOKENS
    return VERBOSITY_MAX_TOKENS.get(_verbosity, 400)


def model_context_len() -> int | None:
    """Lazily query /v1/models for OMLX_MODEL's max_model_len. Returns None on any failure."""
    global _model_context_len_cache, _model_context_len_warned
    if _model_context_len_cache is not None:
        return _model_context_len_cache
    try:
        models = client().models.list()
        for m in models:
            if getattr(m, "id", None) == OMLX_MODEL:
                max_len = getattr(m, "max_model_len", None)
                if max_len is None:
                    dumped = m.model_dump() if hasattr(m, "model_dump") else {}
                    max_len = dumped.get("max_model_len")
                if max_len:
                    _model_context_len_cache = int(max_len)
                    return _model_context_len_cache
        return None
    except Exception as e:
        if not _model_context_len_warned:
            _model_context_len_warned = True
            print(f"[llm] warning: could not query model context length ({e}); using CONTEXT_BUDGET only")
        return None


def context_budget() -> int:
    model_len = model_context_len()
    if model_len:
        return min(CONTEXT_BUDGET, model_len)
    return CONTEXT_BUDGET


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 characters per token, plus ~4 tokens of per-message framing."""
    total = 0
    for m in messages:
        size = len(str(m.get("content") or ""))
        if m.get("tool_calls"):
            size += len(json.dumps(m["tool_calls"]))
        total += size // 4 + 4
    return total


def trim_history() -> None:
    """Drop the oldest turns until the conversation fits the budget.

    Index 0 (the system message) and the final turn (normally the latest user message) are
    never dropped. An assistant turn with tool_calls takes its tool results with it, so no
    tool result is ever left orphaned.
    """
    budget = context_budget() - _effective_max_tokens()
    trimmed = False
    while len(_conversation) > 2 and estimate_tokens(_conversation) > budget:
        turn = _conversation.pop(1)
        trimmed = True
        if turn.get("tool_calls"):
            while len(_conversation) > 2 and _conversation[1].get("role") == "tool":
                del _conversation[1]
    if trimmed:
        print(f"[llm] trimmed conversation history to fit context budget ({budget} tokens)")


def build_system_prompt(avatar: dict) -> str:
    global _verbosity
    verbosity = resolve_verbosity(avatar)
    _verbosity = verbosity
    return (
    f"{avatar['persona']} "
    f"You have memory of past conversations. "
    f"Keep responses conversational: {VERBOSITY_RULES[verbosity]}. "
    f"If the user profile records a preferred name or nickname, address the user by it; that "
    f"overrides any habit from your character. "
    f"Talk like a colleague, not a chatbot. Never use markdown, bullet points, asterisks, or any "
    f"special formatting — plain spoken sentences only. "
    f"Begin every reply with exactly one emotion tag from this list, then a space: "
    f"{' '.join(f'[{e}]' for e in EMOTIONS)}. "
    f"Today's date is {datetime.now().strftime('%A, %B %d %Y')}. "
    f"Use the get_time tool when asked for the current time. "
    f"Use the get_weather tool when asked about the weather — when reporting weather, give a "
    f"summary covering current conditions, feels-like temp, today's min/max, precipitation, and wind. "
    f"Use the get_news tool when asked about the news or current events. "
    f"When the user asks for more details on a headline, use the get_news_detail tool with the URL "
    f"from the previous get_news result — never say you cannot look it up. "
    f"Use the get_emails tool when asked to check email, read emails, or see the inbox. "
    f"Never announce that you are checking or looking something up; state the result directly.\n\n"
    f"Tools are mandatory. You have no clock, no weather data, no news feed and no inbox of your own. "
    f"For anything about the time, the weather (including \"should I bring a jacket\", \"how's it "
    f"looking outside\"), news or current events (\"what's going on in the world\"), or email and "
    f"messages (\"did I get anything today\"), you must call the matching tool BEFORE answering. "
    f"Never invent conditions, times, headlines, senders or message contents — not even in character, "
    f"{avatar['name']} included. Stay in character only in how you phrase the tool's result."
    )


SYSTEM_PROMPT = build_system_prompt(avatars.current())

_client: OpenAI | None = None
_conversation: list = []
_session_turns: list = []
_last_tool_calls: list = []


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY)
    return _client


def reset():
    global _conversation, _session_turns, _last_tool_calls
    _conversation = [{"role": "system", "content": memory.load(SYSTEM_PROMPT)}]
    _session_turns = []
    _last_tool_calls = []


def _tool_round(tool_calls: list[dict], content: str = "") -> None:
    """Execute accumulated tool calls and append the assistant + tool messages to the conversation.

    `tool_calls` is a list of plain dicts: {"id", "name", "arguments"} (arguments is a JSON string).
    """
    _conversation.append({
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"] or "{}"},
            }
            for tc in tool_calls
        ],
    })
    for tc in tool_calls:
        name = tc["name"]
        try:
            args = json.loads(tc["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        _last_tool_calls.append({"name": name, "args": args})
        result = run_tool(name, args)
        print(f"[Tool] {name}() → {result}")
        _conversation.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})


def _rebuild_system_prompt(note: str = "") -> None:
    """Rebuild SYSTEM_PROMPT from the current avatar and swap `_conversation[0]` in place."""
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = build_system_prompt(avatars.current())
    system = {"role": "system", "content": memory.load(SYSTEM_PROMPT) + note}
    if _conversation:
        _conversation[0] = system
    else:
        _conversation.append(system)


def set_avatar():
    """Swap the persona in place, keeping the conversation history."""
    a = avatars.current()
    note = (f"\n\n(Note: from now on you are {a['name']}. Earlier assistant turns in this conversation "
            f"were spoken by a previous character; continue naturally as {a['name']}.)")
    _rebuild_system_prompt(note if len(_conversation) > 1 else "")


def record_assistant(text: str) -> None:
    """Append a canned (non-LLM-generated) assistant turn to the conversation/session history."""
    _conversation.append({"role": "assistant", "content": text})
    _session_turns.append({"role": "assistant", "content": text})


def _accumulate_tool_calls(pending: dict, fragments) -> None:
    """Merge OpenAI-style streamed tool_call fragments into `pending` keyed by index."""
    for frag in fragments:
        index = getattr(frag, "index", None) or 0
        entry = pending.setdefault(index, {"id": None, "name": None, "arguments": ""})
        if getattr(frag, "id", None):
            entry["id"] = entry["id"] or frag.id
        func = getattr(frag, "function", None)
        if func is None:
            continue
        if getattr(func, "name", None):
            entry["name"] = entry["name"] or func.name
        if getattr(func, "arguments", None):
            entry["arguments"] += func.arguments


def ask_events(user_text: str) -> Iterator[tuple[str, object]]:
    """Yield ("tool_calls", [names]) before tools run, then ("delta", text) chunks."""
    global _last_tool_calls

    conv_len = len(_conversation)
    turns_len = len(_session_turns)
    _conversation.append({"role": "user", "content": user_text})
    _session_turns.append({"role": "user", "content": user_text})
    _last_tool_calls = []
    trim_history()

    max_tokens = _effective_max_tokens()

    try:
        # Streamed first pass: text deltas go out immediately, tool-call fragments accumulate.
        first = client().chat.completions.create(
            model=OMLX_MODEL, messages=_conversation, tools=TOOLS, stream=True, max_tokens=max_tokens,
        )
        parts: list[str] = []
        pending: dict = {}
        for chunk in first:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                parts.append(text)
                yield ("delta", text)
            fragments = getattr(delta, "tool_calls", None)
            if fragments:
                _accumulate_tool_calls(pending, fragments)

        if pending:
            tool_calls = [pending[i] for i in sorted(pending)]
            if parts:
                print("[llm] warning: text before tool call was spoken")
                parts = []  # already spoken; don't record it as the reply
            yield ("tool_calls", [tc["name"] for tc in tool_calls])
            _tool_round(tool_calls)
            stream = client().chat.completions.create(
                model=OMLX_MODEL, messages=_conversation, stream=True, max_tokens=max_tokens,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    yield ("delta", delta)
        reply = "".join(parts)
    except Exception:
        del _conversation[conv_len:]
        del _session_turns[turns_len:]
        raise

    _conversation.append({"role": "assistant", "content": reply})
    _session_turns.append({"role": "assistant", "content": reply})
    _log(user_text, _last_tool_calls, reply)


def ask_stream(user_text: str) -> Iterator[str]:
    """Yield only the text deltas of the assistant reply."""
    for kind, payload in ask_events(user_text):
        if kind == "delta":
            yield payload


def ask(user_text: str) -> str:
    return "".join(ask_stream(user_text))


def get_session_turns() -> list:
    return _session_turns


def get_last_tool_calls() -> list:
    return _last_tool_calls


def save_memory():
    """Fold this session into the memory.json profile. Idempotent: turns are cleared once saved."""
    global _session_turns
    if _session_turns:
        turns, _session_turns = _session_turns, []
        memory.save(turns, client(), OMLX_MODEL)


def _log(user_text: str, tool_calls: list, reply: str):
    tools_str = ",".join(tc["name"] for tc in tool_calls) if tool_calls else "none"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | user: {user_text!r} | tools: {tools_str} | reply_len: {len(reply)}\n"
    with open(LOG_PATH, "a") as f:
        f.write(line)


reset()
