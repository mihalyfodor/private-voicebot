"""The system prompt: verbosity, grounding and routing rules, the location line, per-avatar prompt text (PRD 50 step 9, from llm.py)."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from core import avatars
from core import settings
from core import trace
from vault import focus as v_focus
from core.emotions import EMOTIONS
from tools import registry
from tools import weather as _weather
from core.untrusted import DOCUMENT_RULE
from core import toolset

log = trace.logger("llm")


toolset.TOOLS = registry.schemas()
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "0"))  # 0 = derive from verbosity

VERBOSITY_RULES = {
    "short": "at most two sentences",
    "normal": "two to four sentences; a little longer only when actually explaining something",
    "long": "as long as the answer needs, but still spoken prose — no lists or headings",
}
DEEPER_MAX_TOKENS = 2500   # owned here so written/offers do not read each other at import time
VERBOSITY_MAX_TOKENS = {"short": 400, "normal": 400, "long": 800}

_verbosity = "normal"


def resolve_verbosity(avatar: dict) -> str:
    """settings.json > VERBOSITY env > avatar card > 'normal'. Invalid values warn and fall through."""
    sources = (
        ("saved verbosity", settings.load_settings().get("verbosity")),
        ("VERBOSITY env value", os.getenv("VERBOSITY")),
        ("card verbosity", avatar.get("verbosity")),
    )
    for label, value in sources:
        if not value:
            continue
        if value in VERBOSITY_RULES:
            return value
        log.warning(f"warning: invalid {label} {value!r}; ignoring")
    return "normal"


def current_verbosity() -> str:
    return _verbosity


def set_verbosity(value: str) -> None:
    global _verbosity
    if value not in VERBOSITY_RULES:
        raise ValueError(f"Unknown verbosity {value!r}; valid: {', '.join(VERBOSITY_RULES)}")
    _verbosity = value
    settings.save_setting("verbosity", value)
    from agent import conversation      # function-local: conversation imports this module at load, then resets

    conversation._rebuild_system_prompt()


def _effective_max_tokens() -> int:
    if MAX_TOKENS:
        return MAX_TOKENS
    return VERBOSITY_MAX_TOKENS.get(_verbosity, 400)


def _budget(long: bool) -> int:
    """The completion's max_tokens: the `long` budget on a deeper / lookup / written turn."""
    return VERBOSITY_MAX_TOKENS["long"] if long else _effective_max_tokens()


#: the fixed offer phrases a grounded answer may end with (docs/34-grounded-answers.md); the
#: model is asked for exactly one of them so `_deeper_offer` can be detected on its output
DEEPER_OFFERS = ("Want me to dig deeper, or write it up?", "Want me to dig deeper?", "Shall I dig deeper?",
                 "Want me to read on?")
#: the short-answer clause of the grounding rule; a typed content question swaps it for
#: WRITTEN_CLAUSE for that one completion (docs/43-written-answers.md)
OFFER_CLAUSE = ("keep it to one to three sentences and make the very last sentence of the reply exactly: "
                f"{DEEPER_OFFERS[0]} — the document always has more around that passage. ")
WRITTEN_CLAUSE = "give the full answer from it, as long as it needs, and do not offer to dig deeper. "
GROUNDING_RULES = (
    "\n\nLook it up first. When the wiki index or a recall block in the message names a source or "
    "page on the topic, use search_wiki or read_source (or the recall text) before answering from your own "
    "knowledge, and name the source once, by title, in a short natural clause woven into a "
    "sentence (\"per the Skyrim guide\", \"the Fallout 4 walkthrough says\") — vary the wording, "
    "never a separate closing sentence like \"This information comes from\". Whenever the "
    "answer comes from a document passage (a recall block, a search_wiki passage or read_source), "
    + OFFER_CLAUSE +
    "If the recall text is off-topic, ignore it silently. Live facts — the weather, the time or "
    "date, the news, emails — always come from the matching tool (get_weather, get_time, get_news, "
    "get_emails): never state a temperature, a time or a headline from memory, and if the tool "
    "fails, say that it failed."
)


#: the fixed Wikipedia offer (docs/37-wikipedia-routing.md); a short "yes" runs get_wikipedia
WIKIPEDIA_OFFER = "Want me to check Wikipedia?"
ROUTING_RULES = (
    "\n\nWhere an answer comes from, in this order. First, if the wiki index or a recall block names a "
    "source or page on the topic, that is the answer: local first, never Wikipedia. Second, general "
    "knowledge — who someone was, what a thing is or does, how or why something works, everyday facts "
    "like capitals and authors — you answer directly from your own knowledge and stop: no source "
    "named, no offer, and a recall block about something else is ignored, not mentioned. Third, offer "
    "Wikipedia only when the question itself asks for an exact figure (a specific year or date, a "
    "count, a measurement, the precise definition of a named term — \"what exactly is X\"), or is about "
    "the last few years, or when you truly do not know: give your best short answer in one or two "
    "sentences and make the very "
    f"last sentence of the reply exactly: {WIKIPEDIA_OFFER} A confident general answer never gets that "
    "offer, and advice or opinion questions — should I, is X a good fit, what would you do — never get "
    "the offer either: answer them. The news tool is for today's headlines only: a past event of the "
    "last few years gets the Wikipedia offer, not get_news, unless the question asks for the news. "
    "Never call get_wikipedia on your own; call it only after that yes, and then "
    "say once, in passing, that the answer is per Wikipedia. Fourth, never offer or call Wikipedia "
    "for anything personal — the user's own plans, people, meetings, decisions, files, or anything in "
    "the user profile or the wiki; if you have nothing on it, say so."
)


def grounding_rules() -> str:
    """The look-up-first block plus the routing rules, or "" when the recall tools are not
    registered (eval harness); the routing rules only while get_wikipedia is registered too."""
    if not v_focus.tools_registered():
        return ""
    return GROUNDING_RULES + (ROUTING_RULES if wikipedia_registered() else "")


def wikipedia_registered() -> bool:
    return any((t.get("function") or {}).get("name") == "get_wikipedia" for t in toolset.TOOLS if isinstance(t, dict))


def location_line() -> str:
    """`The user is in <name> (timezone <tz>, local time HH:MM). ` — empty when no place name is known
    (the built-in default is a weather fallback, not where the user is)."""
    where = _weather.location()
    if not where.get("name") or where["source"] == "default":
        return ""
    try:
        local = datetime.now(ZoneInfo(where["tz"])).strftime("%H:%M")
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        local = datetime.now().strftime("%H:%M")
    return f"The user is in {where['name']} (timezone {where['tz']}, local time {local}). "


def build_system_prompt(avatar: dict) -> str:
    global _verbosity
    verbosity = resolve_verbosity(avatar)
    _verbosity = verbosity
    return _prompt_text(avatar, verbosity) + grounding_rules()


def _prompt_text(avatar: dict, verbosity: str) -> str:
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
    f"{location_line()}"
    f"Use the get_time tool when asked for the current time. "
    f"Use the get_weather tool when asked about the weather — when reporting weather, give a "
    f"summary covering current conditions, feels-like temp, today's min/max, precipitation, and wind. "
    f"Use the get_news tool when asked for the news or today's headlines. "
    f"When the user asks for more details on a headline, use the get_news_detail tool with the URL "
    f"from the previous get_news result — never say you cannot look it up. "
    f"Use the get_emails tool when asked to check email, read emails, or see the inbox. "
    f"Never announce that you are checking or looking something up; state the result directly.\n\n"
    f"Tools are mandatory. You have no clock, no weather data, no news feed and no inbox of your own. "
    f"For anything about the time, the weather (including \"should I bring a jacket\", \"how's it "
    f"looking outside\"), news or current events (\"what's going on in the world\"), or email and "
    f"messages (\"did I get anything today\"), you must call the matching tool BEFORE answering. "
    f"Never invent conditions, times, headlines, senders or message contents — not even in character, "
    f"{avatar['name']} included. Stay in character only in how you phrase the tool's result.\n\n"
    f"{DOCUMENT_RULE}"
    )


SYSTEM_PROMPT = build_system_prompt(avatars.current())
