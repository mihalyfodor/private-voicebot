import os
import requests
from datetime import datetime

import memory
from tools import TOOLS, run_tool

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e2b"
LOG_PATH = os.path.join(os.path.dirname(__file__), "session.log")

SYSTEM_PROMPT = (
    f"You are a voice assistant with memory. Keep responses short and conversational. "
    f"Talk like a person, not a chatbot. Never use markdown, bullet points, asterisks, or any "
    f"special formatting — plain spoken sentences only. "
    f"Today's date is {datetime.now().strftime('%A, %B %d %Y')}. "
    f"Use the get_time tool when asked for the current time. "
    f"Use the get_weather tool when asked about the weather — when reporting weather, give a full "
    f"summary covering current conditions, feels-like temp, today's min/max, precipitation, and wind. "
    f"Use the get_news tool when asked about the news or current events. "
    f"When the user asks for more details on a headline, use the get_news_detail tool with the URL "
    f"from the previous get_news result — never say you cannot look it up. "
    f"Use the get_emails tool when asked to check email, read emails, or see the inbox."
)

_conversation: list = []
_session_turns: list = []
_last_tool_calls: list = []


def reset():
    global _conversation, _session_turns, _last_tool_calls
    _conversation = [{"role": "system", "content": memory.load(SYSTEM_PROMPT)}]
    _session_turns = []
    _last_tool_calls = []


def ask(user_text: str) -> str:
    global _last_tool_calls

    _conversation.append({"role": "user", "content": user_text})
    _session_turns.append({"role": "user", "content": user_text})
    _last_tool_calls = []

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": _conversation,
        "tools": TOOLS,
        "stream": False,
        "options": {"num_ctx": 32768},
    })
    data = response.json()
    if "message" not in data:
        print(f"[Ollama error] {data}")
        raise KeyError(f"No 'message' in response: {data}")
    msg = data["message"]

    if msg.get("tool_calls"):
        _conversation.append(msg)
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = tc["function"].get("arguments", {})
            _last_tool_calls.append({"name": name, "args": args})
            result = run_tool(name, args)
            print(f"[Tool] {name}() → {result}")
            _conversation.append({"role": "tool", "content": result})

        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": _conversation,
            "stream": False,
            "options": {"num_ctx": 32768},
        })
        msg = response.json()["message"]

    reply = msg["content"]
    _conversation.append({"role": "assistant", "content": reply})
    _session_turns.append({"role": "assistant", "content": reply})

    _log(user_text, _last_tool_calls, reply)
    return reply


def get_session_turns() -> list:
    return _session_turns


def get_last_tool_calls() -> list:
    return _last_tool_calls


def save_memory():
    if _session_turns:
        memory.save(_session_turns, OLLAMA_URL, OLLAMA_MODEL)


def _log(user_text: str, tool_calls: list, reply: str):
    tools_str = ",".join(tc["name"] for tc in tool_calls) if tool_calls else "none"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | user: {user_text!r} | tools: {tools_str} | reply_len: {len(reply)}\n"
    with open(LOG_PATH, "a") as f:
        f.write(line)


reset()
