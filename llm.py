import os
from datetime import datetime
from typing import Iterator

from openai import OpenAI

import memory
from tools import TOOLS, run_tool

OMLX_BASE_URL = os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1")
OMLX_API_KEY = os.getenv("OMLX_API_KEY", "omlx")
OMLX_MODEL = os.getenv("OMLX_MODEL", "gemma-4-26b")
LOG_PATH = os.path.join(os.path.dirname(__file__), "session.log")

EMOTIONS = ("neutral", "happy", "thinking", "surprised", "apologetic")

SYSTEM_PROMPT = (
    f"You are Haru, a calm and friendly office assistant with memory. "
    f"Keep responses short and conversational: at most two sentences. "
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
    f"Use the get_emails tool when asked to check email, read emails, or see the inbox."
)

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
        import json
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        _last_tool_calls.append({"name": name, "args": args})
        result = run_tool(name, args)
        print(f"[Tool] {name}() → {result}")
        _conversation.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})


def ask_stream(user_text: str) -> Iterator[str]:
    """Yield text deltas of the assistant reply. Tool rounds run non-streamed first."""
    global _last_tool_calls

    _conversation.append({"role": "user", "content": user_text})
    _session_turns.append({"role": "user", "content": user_text})
    _last_tool_calls = []

    # Non-streamed first pass so tool calls can be resolved.
    first = client().chat.completions.create(
        model=OMLX_MODEL, messages=_conversation, tools=TOOLS,
    )
    msg = first.choices[0].message

    if msg.tool_calls:
        _tool_round(msg)
        stream = client().chat.completions.create(
            model=OMLX_MODEL, messages=_conversation, stream=True,
        )
        parts = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                parts.append(delta)
                yield delta
        reply = "".join(parts)
    else:
        reply = msg.content or ""
        yield reply

    _conversation.append({"role": "assistant", "content": reply})
    _session_turns.append({"role": "assistant", "content": reply})
    _log(user_text, _last_tool_calls, reply)


def ask(user_text: str) -> str:
    return "".join(ask_stream(user_text))


def get_session_turns() -> list:
    return _session_turns


def get_last_tool_calls() -> list:
    return _last_tool_calls


def save_memory():
    if _session_turns:
        memory.save(_session_turns, client(), OMLX_MODEL)


def _log(user_text: str, tool_calls: list, reply: str):
    tools_str = ",".join(tc["name"] for tc in tool_calls) if tool_calls else "none"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | user: {user_text!r} | tools: {tools_str} | reply_len: {len(reply)}\n"
    with open(LOG_PATH, "a") as f:
        f.write(line)


reset()
