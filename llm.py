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
    """settings.json > VERBOSITY env > avatar card > 'normal'."""
    saved = avatars.load_settings().get("verbosity")
    if saved:
        if saved in VERBOSITY_RULES:
            return saved
        print(f"[llm] warning: invalid saved verbosity {saved!r}; ignoring")

    env_val = os.getenv("VERBOSITY")
    if env_val:
        if env_val in VERBOSITY_RULES:
            return env_val
        print(f"[llm] warning: invalid VERBOSITY env value {env_val!r}; ignoring")

    card_val = avatar.get("verbosity")
    if card_val:
        if card_val in VERBOSITY_RULES:
            return card_val
        print(f"[llm] warning: invalid card verbosity {card_val!r}; ignoring")

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
    total = 0
    for m in messages:
        content = m.get("content")
        size = len(str(content or ""))
        tool_calls = m.get("tool_calls")
        if tool_calls:
            size += len(json.dumps(tool_calls))
        total += size // 4 + 4
    return total


def trim_history() -> None:
    """Drop the oldest non-system, non-latest-user turns until under budget."""
    budget = context_budget() - _effective_max_tokens()
    trimmed = False
    while estimate_tokens(_conversation) > budget and len(_conversation) > 2:
        # index 0 is system, last is the latest user turn — never drop those.
        idx = 1
        turn = _conversation[idx]
        del _conversation[idx]
        trimmed = True
        if turn.get("role") == "assistant" and turn.get("tool_calls"):
            while idx < len(_conversation) and _conversation[idx].get("role") == "tool":
                del _conversation[idx]
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
    f"IMPORTANT: You have no clock, no weather data, no news and no inbox of your own. "
    f"Whenever the user asks about the time, weather, news, or email you MUST call the matching tool "
    f"before answering — never guess or make up an answer. "
    f"Never announce that you are checking or looking something up; state the result directly."
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


def _tool_round(msg) -> None:
    """Execute tool calls from an assistant message and append results to the conversation."""
    _conversation.append({
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in msg.tool_calls
        ],
    })
    for tc in msg.tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        _last_tool_calls.append({"name": name, "args": args})
        result = run_tool(name, args)
        print(f"[Tool] {name}() → {result}")
        _conversation.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})


def _rebuild_system_prompt(note: str = "") -> None:
    """Rebuild SYSTEM_PROMPT from the current avatar and swap `_conversation[0]` in place."""
    global SYSTEM_PROMPT
    a = avatars.current()
    SYSTEM_PROMPT = build_system_prompt(a)
    if _conversation:
        _conversation[0] = {"role": "system", "content": memory.load(SYSTEM_PROMPT) + note}
    else:
        _conversation.append({"role": "system", "content": memory.load(SYSTEM_PROMPT) + note})


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
        # Non-streamed first pass so tool calls can be resolved.
        first = client().chat.completions.create(
            model=OMLX_MODEL, messages=_conversation, tools=TOOLS, max_tokens=max_tokens,
        )
        msg = first.choices[0].message

        if msg.tool_calls:
            yield ("tool_calls", [tc.function.name for tc in msg.tool_calls])
            _tool_round(msg)
            stream = client().chat.completions.create(
                model=OMLX_MODEL, messages=_conversation, stream=True, max_tokens=max_tokens,
            )
            parts = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    yield ("delta", delta)
            reply = "".join(parts)
        else:
            reply = msg.content or ""
            yield ("delta", reply)
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
    """Summarise this session into shortmem.txt. Idempotent: turns are cleared once saved."""
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
