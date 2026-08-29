"""Short spoken fillers played while a tool call is in flight."""
import random

FILLERS = {
    "get_time": ["One moment.", "Let me check the clock.", "Just a second."],
    "get_weather": ["Let me check outside.", "One sec, checking the weather.", "Let me have a look at the forecast."],
    "get_news": ["Pulling up the headlines.", "Let me see what's going on.", "One moment, checking the news."],
    "get_news_detail": ["Let me read that one.", "One sec, opening it up."],
    "get_emails": ["Let me check your inbox.", "One sec, looking at your mail.", "Checking your emails now."],
    "search_wiki": ["Let me look that up in my notes.", "One sec, checking what we wrote down.", "Digging through the wiki."],
    "read_page": ["Let me open that page.", "One sec, reading it up.", "Pulling up the notes."],
    "file_source": ["Filing that away.", "One sec, writing it down.", "Let me put that in the wiki."],
    "default": ["One moment.", "Let me check.", "Just a second."],
}

_last: dict[str, str] = {}


def pick(tool_name: str) -> str:
    options = FILLERS.get(tool_name) or FILLERS["default"]
    key = tool_name if tool_name in FILLERS else "default"
    choices = [p for p in options if p != _last.get(key)] or options
    phrase = random.choice(choices)
    _last[key] = phrase
    return phrase
