import json
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Iterator

import memory
import avatars
from core import settings
from core import trace
from core import llm as core_llm
import wiki
from splitter import EMOTIONS, strip_tag
from tools import registry
from tools import weather as _weather
from core.untrusted import DOCUMENT_RULE, documents_in, untrusted

run_tool = registry.run_tool
TOOLS = registry.schemas()
trace.vocab("name", [t.name for t in registry.REGISTRY if t.schema] + ["dig_deeper"])
log = trace.logger("llm")
tool_log = trace.logger("Tool")

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
    _rebuild_system_prompt()


def _effective_max_tokens() -> int:
    if MAX_TOKENS:
        return MAX_TOKENS
    return VERBOSITY_MAX_TOKENS.get(_verbosity, 400)


def _budget(long: bool) -> int:
    """The completion's max_tokens: the `long` budget on a deeper / lookup / written turn."""
    return VERBOSITY_MAX_TOKENS["long"] if long else _effective_max_tokens()


def model_context_len() -> int | None:
    """Lazily query /v1/models for OMLX_MODEL's max_model_len. Returns None on any failure."""
    global _model_context_len_cache, _model_context_len_warned
    if _model_context_len_cache is not None:
        return _model_context_len_cache
    try:
        models = core_llm.client().models.list()
        for m in models:
            if getattr(m, "id", None) == core_llm.OMLX_MODEL:
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
            log.warning(f"warning: could not query model context length ({e}); using CONTEXT_BUDGET only")
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


#: `estimate_tokens` counts the messages only; the tool schemas ride on every request too.
#: the 90 % crossing sentence (docs/39-context-meter.md), spoken once per crossing
CONTEXT_WARNING = "I'm close to the end of what I can hold — say new topic to start fresh."
CONTEXT_WARN_PCT, CONTEXT_REARM_PCT = 90, 75
#: user turns the last `trim_history` dropped, and whether the 90 % sentence was said this crossing
_last_trimmed = 0
_context_warned = False


def trim_history() -> int:
    """Drop the oldest turns until the conversation fits the budget; returns the user turns dropped.

    Index 0 (the system message) and the final turn (normally the latest user message) are
    never dropped. An assistant turn with tool_calls takes its tool results with it, so no
    tool result is ever left orphaned.
    """
    global _last_trimmed
    budget = context_budget() - _effective_max_tokens()
    dropped = []
    while len(_conversation) > 2 and estimate_tokens(_conversation) > budget:
        turn = _conversation.pop(1)
        dropped.append(turn)
        if turn.get("tool_calls"):
            while len(_conversation) > 2 and _conversation[1].get("role") == "tool":
                del _conversation[1]
    users = sum(1 for t in dropped if t.get("role") == "user")
    _last_trimmed = users or int(bool(dropped))                  # a lone assistant turn still counts
    if dropped:
        log.info(f"trimmed conversation history to fit context budget ({budget} tokens)")
    return _last_trimmed


def context_usage() -> dict:
    """`{used, budget, pct, trimmed}`: what the model holds, the budget, and the user turns the
    last trim forgot (docs/39-context-meter.md)."""
    used = estimate_tokens(_conversation) + registry.tokens()
    budget = context_budget()
    return {"used": used, "budget": budget, "pct": round(100 * used / budget), "trimmed": _last_trimmed}


def context_notice() -> str:
    """CONTEXT_WARNING the first time usage reaches 90 %, "" otherwise; re-armed under 75 %."""
    global _context_warned
    pct = context_usage()["pct"]
    if pct >= CONTEXT_WARN_PCT and not _context_warned:
        _context_warned = True
        return CONTEXT_WARNING
    if pct < CONTEXT_REARM_PCT:
        _context_warned = False
    return ""


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
    if not wiki.tools_registered():
        return ""
    return GROUNDING_RULES + (ROUTING_RULES if wikipedia_registered() else "")


def wikipedia_registered() -> bool:
    return any((t.get("function") or {}).get("name") == "get_wikipedia" for t in TOOLS if isinstance(t, dict))


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

_conversation: list = []
_session_turns: list = []
_last_tool_calls: list = []
#: {"source", "title", "parts"} while the last reply ended with a DEEPER_OFFERS phrase,
#: {"wikipedia": question} while it ended with WIKIPEDIA_OFFER, else None
_deeper_offer: dict | None = None
#: (source, title, part) of every passage this turn used: recall hits, search_wiki snippets, read_source
_turn_sources: list = []
#: the `source` names of every document block this session has read (recall, deeper, tool
#: results) since the last successful memory save — the memory proposal is asked to attribute
#: its ops against this list (docs/33-untrusted-text.md)
_sources_seen: list = []
#: the (page, hits) the last recall search found, and the spoken turn's band — the `recall` event (PRD 49)
_evidence: tuple = (None, [])
last_band: str | None = None
#: the wiki / Wikipedia page titles this turn drew on (page recall, the focused page, get_wikipedia)
_turn_pages: list = []
#: what the last answer was built from, for `write_up` (docs/43-written-answers.md):
#: {"question", "extra" (the recall/deeper/focus block), "tools" (tool result texts), "sources", "pages"},
#: plus "card" (ready markdown) after a briefing (docs/46-briefing.md)
_last_context: dict | None = None
#: every card this conversation wrote, by id; `last_card` is the one "save that" means
_cards: dict = {}
last_card: dict | None = None
#: run cards issued but not (yet) a turn of the conversation — a write-up running or failed (PRD 51):
#: id -> how many `dialogue()` lines preceded it, so a reconnect replays it in place
_loose: dict = {}
_card_seq = 0              # every card issued takes the next id; only `reset` starts over


def _offer_kind(recall_offer: bool) -> str:
    """What the turn ended offering: none / deeper / wikipedia / writeup, or recall (the medium band's check)."""
    if recall_offer:
        return "recall"
    if not _deeper_offer:
        return "none"
    if _deeper_offer.get("wikipedia"):
        return "wikipedia"
    return "writeup" if (_conversation[-1].get("content") or "").rstrip().endswith(DEEPER_OFFERS[0]) else "deeper"


def reset():
    global _conversation, _session_turns, _last_tool_calls, _deeper_offer, _turn_sources, _sources_seen
    global _last_trimmed, _context_warned, _turn_pages, _last_context, _card_seq
    _conversation = [{"role": "system", "content": memory.load(SYSTEM_PROMPT)}]
    _session_turns = []
    _last_tool_calls = []
    _deeper_offer = None
    _turn_sources = []
    _turn_pages = []
    _sources_seen = []
    _last_context = None
    _last_trimmed = 0
    _context_warned = False
    _forget_cards()
    _card_seq = 0


def _forget_cards() -> None:
    global last_card
    _cards.clear()
    _loose.clear()
    last_card = None


def _note_documents(text: str) -> None:
    """Record the document blocks in a sent message or tool result for memory attribution."""
    for name in documents_in(text):
        if name not in _sources_seen:
            _sources_seen.append(name)


def _tool_round(tool_calls: list[dict], content: str = "") -> list:
    """Execute accumulated tool calls and append the assistant + tool messages to the conversation.

    `tool_calls` is a list of plain dicts: {"id", "name", "arguments"} (arguments is a JSON string).
    Returns the tool result texts (the write-up context, PRD 43).
    """
    results = []
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
        keys = sorted(args) if isinstance(args, dict) else []
        with trace.span("tool_call", name=name, arg_keys=keys, ok=False) as span:
            result = run_tool(name, args)
            span.update(result_len=len(str(result)), ok=True)
        tool_log.info(f"{name}({', '.join(keys)})")          # never the result (PRD 49)
        trace.verbose("tool_result", name=name, args=args, result=str(result))
        _note_tool_sources(name, args, str(result))
        _note_documents(str(result))
        results.append(str(result))
        _conversation.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
    return results


def _rebuild_system_prompt(note: str = "") -> None:
    """Rebuild SYSTEM_PROMPT from the current avatar and swap `_conversation[0]` in place."""
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = build_system_prompt(avatars.current())
    system = {"role": "system", "content": memory.load(SYSTEM_PROMPT) + note}
    if _conversation:
        _conversation[0] = system
    else:
        _conversation.append(system)


def refresh_context() -> None:
    """Re-read the profile and wiki index into the system prompt, keeping the conversation.

    Called after anything that changes the vault mid-session (a drop, an edit, an ingest), so
    a page filed a minute ago is visible to the very next turn instead of the next restart.
    """
    if _conversation:
        _rebuild_system_prompt()


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


RECALL_MIN_WORDS = 4
RECALL_MAX_CHUNKS = 2
RECALL_MAX_TOKENS = 1500
#: a question whose content words are all of these is a tool command ("what time is it"),
#: never a question for a filed document
COMMAND_WORDS = frozenset("""time clock date day today weather temperature forecast rain outside
news headlines headline email emails mail inbox messages message""".split())
#: the two phrases of a recall note a typed content question rewrites (`written_note`, PRD 43)
SHORT_PHRASE, FULL_PHRASE = "in one to three sentences", "fully, as long as it needs"
NOTE_OFFER, NOTE_NO_OFFER = "finish with exactly this sentence: " + DEEPER_OFFERS[0], "do not offer to dig deeper."
RECALL_NOTE = ("If this answers the question, answer from it " + SHORT_PHRASE + ", name {titles} "
               "once in a short natural clause inside a sentence (never as a separate closing sentence), "
               "and " + NOTE_OFFER + " Otherwise ignore it "
               "completely: answer as you normally would and never mention the document or what it lacks.")
# --- ask before lookup (docs/45-ask-before-lookup.md): the recall band on spoken turns
RECALL_HIGH_SCORE = wiki.RECALL_HIGH_SCORE
#: the one sentence a medium spoken turn ends with, and the note that keeps the model from asking too
RECALL_OFFER = "Want me to check your {title}?"
RECALL_ASK_NOTE = "Answer briefly as you normally would, do not end with a question."
RECALL_YES_NOTE = "The user said yes to checking. Answer their earlier question from it: «{question}»"
# --- recall precision 2 (docs/42-recall-precision-2.md) --------------------------------
#: a question shorter than RECALL_MIN_WORDS still recalls with this many words and a proper term
RECALL_MIN_PROPER_WORDS = 2
#: chunk candidates fetched per turn; RECALL_MAX_CHUNKS of them are kept, never two neighbours
RECALL_CANDIDATES = 4
#: the wiki page that joins the recall block (head + Summary + best section)
RECALL_PAGE_TOKENS = 400
#: the reply text held back before the fallback decision; the lookup block's budget
FALLBACK_HOLD_CHARS = 240
FALLBACK_MAX_TOKENS = 1500
#: a reply that claims the filed documents do not cover the question (fallback trigger)
NOT_COVERED = re.compile(
    r"(?:do(?:es)? not|don'?t|doesn'?t|did not|didn'?t) (?:mention|cover|contain|include|say|discuss|address|"
    r"appear|have any|have information|have details|have record|have notes?|have anything)"
    r"|not covered|not (?:mentioned|included|found|in) (?:in |among )?(?:my |the |your |these |those )?"
    r"(?:files|documents|notes|sources|guides?|snippets?|passages?|recall|text|provided)"
    r"|no (?:information|mention|details?|record|notes?|entry|data) (?:on|about|of|regarding|for|in)"
    r"|nothing (?:on|about|filed|in my|in the)|(?:can'?t|cannot|couldn'?t) find"
    r"|only contains? the (?:cover|introduction|contents)", re.I)
FALLBACK_NOTE = ("A search of the filed documents found these passages. If they cover the question, answer "
                 "from them " + SHORT_PHRASE + ", name {titles} once in a short natural clause inside "
                 "a sentence, and " + NOTE_OFFER + " If they do "
                 "not cover it, ignore them completely: answer the question as you normally would (your own "
                 "knowledge, the usual rules) and never mention documents, files or what you could not find.")

DEEPER_MAX_CHUNKS = 3
DEEPER_MAX_TOKENS = 2500
WIKIPEDIA_NOTE = ("The user said yes to checking Wikipedia. Call get_wikipedia now with the topic of the "
                  "question as the query (the article name, like \"Ada Lovelace\", not the whole question), "
                  "then answer the question fully from the result, in several sentences, and say once, in "
                  "passing, that it is per Wikipedia. Do not offer to check Wikipedia again.")
DEEPER_NOTE = ("The user said yes to digging deeper. Give the full answer now from this text (the "
               "passage you cited and the parts around it): walk through what it says step by step in "
               "several sentences, at least twice as long as your previous reply, including the extra "
               "details, names and places the surrounding parts add. Name {title} once in passing. "
               "Only what the passage says; do not add locations or names from memory. "
               "Do not offer to dig deeper again.")
#: a short "yes" to the deeper offer: one of these words first ...
_AFFIRMATIVE_LEADS = frozenset("""yes yeah yep yup sure ok okay please absolutely definitely certainly
go dig deeper dive read tell more why do that i lets let check look""".split())
#: ... and nothing but these after it, at most AFFIRMATIVE_MAX_WORDS in all
_AFFIRMATIVE_WORDS = _AFFIRMATIVE_LEADS | frozenset("""on ahead it in me not thanks thank you would be
great good d like s so up wikipedia""".split())
AFFIRMATIVE_MAX_WORDS = 6
_SNIPPET_PART = re.compile(r"^(.+?) · part (\d+)/\d+ \(([^)]+)\)", re.M)


# --- focus context (docs/38-focus-context.md) ----------------------------------------
FOCUS_HEAD_TOKENS = 500
FOCUS_SECTION_TOKENS = 700
FOCUS_MAX_TOKENS = 1200
FOCUS_NOTE = ("The user has «{title}» open; 'this / it / the page' means that page. Answer from it; "
              "if the answer is not in it, say so. Only what the passage says; do not add locations or "
              "names from memory.")
MEMORY_FOCUS_NOTE = "The user is in the Memory editor."
OPENED_NOTE = ("You just opened «{title}» for the user; add one short remark about it (one sentence), "
               "then stop.")
NO_MATCH_NOTE = "No page matched «{query}»; say so briefly."
#: what the UI is looking at (`set_focus`) and the page it last had open; never persisted
focus = {"view": "chat", "path": None}
last_page: str | None = None


def set_focus(view: str, path=None) -> None:
    """Record the client's view and open page; a page in the Wiki / Sources view becomes `last_page`."""
    global last_page
    path = str(path or "").strip() or None
    focus["view"], focus["path"] = str(view or "chat"), path
    if view in ("wiki", "sources") and path:
        last_page = path


def focus_block(question: str) -> str:
    """The `<document>` block for the page the user is looking at, or "".

    Wiki / Sources view with a page: its head (≤ FOCUS_HEAD_TOKENS) and the `##` section — for a
    source, the raw chunk — that best matches `question` (≤ FOCUS_SECTION_TOKENS), together within
    FOCUS_MAX_TOKENS, wrapped as untrusted text plus FOCUS_NOTE. Any other view: "".
    """
    view, path = focus["view"], focus["path"]
    if view not in ("wiki", "sources") or not path:
        return ""
    text = wiki.page_text(path)
    if text is None:
        return ""
    title = wiki.page_title(path)
    _turn_pages.append(title)
    head, sections = wiki.split_sections(wiki.parse_page(text)[1])
    # The schema keeps a page's gist in `## Summary`; that is the head, not a candidate section.
    summary = next((s for s in sections if s.startswith("## Summary")), None)
    if summary:
        head, sections = head + "\n\n" + summary, [s for s in sections if s is not summary]
    parts = _fit([head], FOCUS_HEAD_TOKENS, "")
    # "summarise this" names no topic: the head + Summary is the answer, not a random chunk (PRD 42)
    topical = bool(wiki.topic_terms(question))
    chunk = wiki.best_chunk(path, question) if view == "sources" and topical else None
    if chunk:
        _turn_sources.append((path, chunk["title"], chunk["n"]))
        section = f'Part {chunk["n"]}/{chunk["total"]}: {chunk["text"]}'
    else:
        section = wiki.best_section(sections, question) if topical else None
    if section:
        parts += _fit([section], FOCUS_SECTION_TOKENS, "")
    note = FOCUS_NOTE.format(title=title)
    kept = _fit(["\n\n".join(p for p in parts if p)], FOCUS_MAX_TOKENS, note)
    if not kept or not kept[0].strip():
        return "\n\n" + note
    return "\n\n" + untrusted(kept[0], title) + "\n" + note


def is_affirmative(text: str) -> bool:
    """"yes", "yes please", "go on", "dig deeper", "sure, do it" — ≤ 6 words, nothing else in it."""
    words = re.findall(r"[a-z]+", text.lower())
    return (0 < len(words) <= AFFIRMATIVE_MAX_WORDS and words[0] in _AFFIRMATIVE_LEADS
            and all(w in _AFFIRMATIVE_WORDS for w in words))


def _fit(lines: list, budget: int, note: str) -> list:
    """`lines` cut to fit `budget` tokens together with `note` and the tags: whole lines while
    they fit, the first one that does not is truncated, the rest dropped."""
    used = memory.tokens_est(note) + 40
    kept = []
    for line in lines:
        room = (budget - used) * 4
        if room <= 0:
            break
        kept.append(line[:room])
        used += memory.tokens_est(kept[-1])
    return kept


def ends_with_offer(reply: str, phrases: tuple = DEEPER_OFFERS) -> bool:
    tail = reply.strip().rstrip("\"'”’ ").lower()
    return any(tail.endswith(o.lower()) for o in phrases)


def ends_with_wikipedia_offer(reply: str) -> bool:
    return ends_with_offer(reply, (WIKIPEDIA_OFFER,))


def strip_offer(reply: str) -> str:
    """`reply` without a trailing DEEPER_OFFERS / WIKIPEDIA_OFFER sentence (the medium turn's own
    offer replaces it, PRD 45)."""
    for phrase in DEEPER_OFFERS + (WIKIPEDIA_OFFER,):
        if ends_with_offer(reply, (phrase,)):
            return reply[:reply.lower().rfind(phrase.lower())].rstrip()
    return reply


def _note_tool_sources(name: str, args: dict, result: str) -> None:
    """Remember which source parts a tool call surfaced, for the deeper offer."""
    if name == "read_source":
        try:
            part = int(args.get("part"))
        except (TypeError, ValueError):
            return
        title = next((line.split(" · part ")[0] for line in result.splitlines() if " · part " in line), "")
        _turn_sources.append((str(args.get("path") or ""), title, part))
    elif name == "search_wiki":
        for m in _SNIPPET_PART.finditer(result):
            _turn_sources.append((m.group(3), m.group(1), int(m.group(2))))
    elif name == "get_wikipedia" and args.get("query"):
        _turn_pages.append(f"Wikipedia: {args['query']}")


def _offer_from_turn(reply: str, question: str = "") -> dict | None:
    """The `_deeper_offer` this reply leaves behind: the question when the reply ended with the
    Wikipedia offer, else the first source used, all its parts."""
    if ends_with_wikipedia_offer(reply):
        return {"wikipedia": question}
    if not _turn_sources or not ends_with_offer(reply):
        return None
    source, title, _ = _turn_sources[0]
    parts = sorted({p for s, _, p in _turn_sources if s == source})
    return {"source": source, "title": title, "parts": parts}


def deeper_parts(parts: list, total: int) -> list:
    """The offered parts ± 1 neighbour, at most DEEPER_MAX_CHUNKS, offered ones first."""
    wanted = [p for p in parts if 1 <= p <= total]
    for p in list(wanted):
        for q in (p - 1, p + 1):
            if 1 <= q <= total and q not in wanted:
                wanted.append(q)
    return sorted(wanted[:DEEPER_MAX_CHUNKS])


def deeper_block(offer: dict) -> str:
    """The `<source>` block for a "yes" turn: the offered chunks and their neighbours, capped."""
    try:
        record = wiki.raw_chunks(offer["source"])
    except KeyError:
        return ""
    chunks, title = record["chunks"], record["title"]
    note = DEEPER_NOTE.format(title=f'"{title}"')
    lines = [f"Part {n}/{len(chunks)}: {chunks[n - 1]['text']}"
             for n in deeper_parts(offer["parts"], len(chunks))]
    quotes = _fit(lines, DEEPER_MAX_TOKENS, note)
    if not quotes:
        return ""
    return (f'\n\n<source title="{title}">\n' + "\n\n".join(untrusted(q, title) for q in quotes)
            + f"\n{note}\n</source>")


def _messages(deeper: bool, written: bool = False) -> list:
    """The conversation as sent: on a deeper turn the system rule reads `long` for this completion;
    on a written turn (typed, grounded, content — PRD 43) the short-answer clause reads full too."""
    if not deeper or not _conversation:
        return _conversation
    system = _conversation[0]["content"].replace(VERBOSITY_RULES[_verbosity], VERBOSITY_RULES["long"], 1)
    if written:
        system = system.replace(OFFER_CLAUSE, WRITTEN_CLAUSE, 1)
    return [{"role": "system", "content": system}] + _conversation[1:]


# --- written answers (docs/43-written-answers.md) --------------------------------------
#: a content question: one of these words, or CONTENT_MIN_WORDS words or more
CONTENT_WORDS = re.compile(r"\b(?:what|how|why|explain|compare|list|steps|difference|summari[sz]e)\b", re.I)
CONTENT_MIN_WORDS = 8
#: a name question ("Who is Vivec?") answers directly when its top hit has a proper match (the recall
#: band only, PRD 45; CONTENT_WORDS and the typed length are untouched)
ASK_WORDS = re.compile(r"^\W*(?:who|where|which|when)\b", re.I)
#: the tools whose result is filed / fetched text (a typed content question answered from one runs long)
GROUNDING_TOOLS = frozenset(("search_wiki", "read_source", "read_page", "get_wikipedia"))
WRITTEN_LINE = "[happy] Written out below."
CARD_PROMPT = ("You write reference notes in Markdown. Answer the question below fully from the material "
               "given, as a document: start with one `# Title` line naming the topic, then at least two `## ` "
               "sections; use bullet lists, numbered steps or a table where the material is structured, "
               "prose where it is not. Only what the material says — no names, places or figures from memory. "
               "No emotion tag, no greeting, no closing question, no offer, and no sources section: the "
               "sources are appended for you.")
CARD_MAX_TOKENS = VERBOSITY_MAX_TOKENS["long"]
CARD_CONTEXT_TOKENS = DEEPER_MAX_TOKENS
CARD_HISTORY_TURNS = 6
_CARD_MARK = re.compile(r"^<card id=\"([a-z0-9]+)\">\n(.*)\n</card>$", re.S)


def content_question(text: str) -> bool:
    """A question that wants an explanation: a CONTENT_WORDS word or CONTENT_MIN_WORDS+ words."""
    return bool(CONTENT_WORDS.search(text)) or len(text.split()) >= CONTENT_MIN_WORDS


def written_note(extra: str) -> str:
    """The turn's recall / lookup block with its short-answer wording turned into the full one."""
    return extra.replace(SHORT_PHRASE, FULL_PHRASE).replace(NOTE_OFFER, NOTE_NO_OFFER)


def source_parts(sources: list) -> list:
    """`{title, part: "3/120"}` per passage (deduped, in order): the card's Sources line and the
    write-up run's `context` step (PRD 51)."""
    out = []
    for source, title, part in sources:
        try:
            row = {"title": title, "part": f"{part}/{len(wiki.raw_chunks(source)['chunks'])}"}
        except (KeyError, TypeError):
            row = {"title": title, "part": str(part)}
        if row not in out:
            out.append(row)
    return out


def card_sources(sources: list, pages: list) -> list:
    """`"Fallout 4 guide · part 3/120"` per passage (deduped, in order) and the page titles."""
    out = [f"{p['title']} · part {p['part']}" for p in source_parts(sources)]
    for title in pages:
        if title and title not in out:
            out.append(title)
    return out


def _card_title(markdown: str, question: str) -> str:
    """The card's `# Title` line, else the question itself (trimmed)."""
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or question
    return " ".join(question.split())[:80] or "Written answer"


def can_write_up() -> bool:
    """Is there an answer to write up (else `write_up` raises NOTHING_TO_WRITE)?"""
    return bool(_last_context)


def next_card_id() -> str:
    """A fresh card id: the counter advances on every card issued, so a failed write-up's id is
    never reused by the retry (PRD 51)."""
    global _card_seq
    _card_seq += 1
    return f"card{_card_seq}"


def issue_card(card: dict) -> dict:
    """Keep a run card that is not a conversation turn yet (a write-up still running, or failed) so a
    reconnect replays it where it was issued; `write_up` fills the same dict in when it succeeds."""
    _cards[card["id"]] = card
    _loose[card["id"]] = len(dialogue())
    return card


def loose_cards() -> list:
    """``(position, card)`` of the issued cards that are not conversation turns, in issue order."""
    return [(pos, _cards[cid]) for cid, pos in _loose.items() if cid in _cards]


def context_parts() -> list:
    """The last answer's passages as `{title, part}` (what a write-up is built from)."""
    return source_parts((_last_context or {}).get("sources") or [])


NOTHING_TO_WRITE = "nothing to write up yet"


def write_up(request: str = "", card_id: str | None = None) -> dict:
    """One `long` completion turning the last answer into a Markdown card (PRD 43).

    The card is built from the same context the last answer had (`_last_context`: the recall /
    deeper / focus block and the tool results) and ends with a `Sources:` line. Returns the card
    `{id, markdown, sources, title}`; the conversation keeps the card text as an assistant turn
    (marked, so a reconnect replays it as a card) while the session turns get "wrote up: <title>".
    Raises ValueError when nothing has been answered yet. After a briefing the context carries
    its ready `card` and no completion is made (PRD 46).
    """
    global last_card, _deeper_offer
    ctx = _last_context
    if not ctx:
        raise ValueError(NOTHING_TO_WRITE)
    _deeper_offer = None                                        # the offer's "write it up" reply takes it
    if ctx.get("card"):                                         # a briefing: its card is already built (PRD 46)
        markdown, sources = ctx["card"].strip(), []
    else:
        markdown, sources = _card_completion(ctx)
    card = _cards.get(card_id) if card_id else None         # the running card issued at turn start (PRD 51)
    card = card if card is not None else {"id": card_id or next_card_id()}
    card.update(markdown=markdown, sources=sources, title=_card_title(markdown, ctx["question"]))
    _cards[card["id"]] = card
    _loose.pop(card["id"], None)
    last_card = card
    if request:
        _conversation.append({"role": "user", "content": request})
        _session_turns.append({"role": "user", "content": request})
    _conversation.append({"role": "assistant", "content": f'<card id="{card["id"]}">\n{markdown}\n</card>'})
    _session_turns.append({"role": "assistant", "content": f"{memory.STAND_INS[0]}{card['title']}"})
    return card


def _card_completion(ctx: dict) -> tuple:
    """``(markdown, sources)``: one `long` completion over the last answer's context + its Sources line."""
    material = "\n\n".join(p for p in [ctx["extra"].strip()] + ctx["tools"] if p)
    material = "".join(_fit([material], CARD_CONTEXT_TOKENS, CARD_PROMPT)) if material else ""
    user = f"Question: {ctx['question']}\n\nThe answer you gave: {strip_tag(ctx['reply'])}"
    if material:
        user += "\n\nMaterial:\n" + material
    history = [m for m in dialogue()[-CARD_HISTORY_TURNS:] if not card_in(m["content"])]
    resp = core_llm.completion(
        "card", model=core_llm.OMLX_MODEL, max_tokens=CARD_MAX_TOKENS,
        messages=[{"role": "system", "content": CARD_PROMPT}] + history + [{"role": "user", "content": user}],
    )
    markdown = strip_tag((resp.choices[0].message.content or "").strip()).strip()
    if not markdown:
        raise ValueError("the model wrote nothing")
    sources = card_sources(ctx["sources"], ctx["pages"])
    if sources:
        markdown += "\n\nSources: " + "; ".join(sources)
    return markdown, sources


# --- briefing (docs/46-briefing.md) ----------------------------------------------------
BRIEFING_MAX_TOKENS = 200
BRIEFING_PROMPT = ("The document below lists what changed in the user's wiki, sources and memory since {when}. "
                   "Tell the user about it in two or three short spoken sentences: name the most notable items, "
                   "give counts for the rest. Only what the list says; no question and no offer at the end.")


def _digest_text(digest: dict) -> str:
    """The digest as plain lines for the recap prompt, capped like the card."""
    lines = []
    for change in ("new", "edited", "deleted"):
        titles, extra = wiki.capped([p["title"] for p in digest["pages"] if p["change"] == change])
        if titles:
            more = f" (+{extra} more)" if extra else ""
            lines.append(f"Pages {change}: " + ", ".join(titles) + more)
    if digest["chat_pages"]:
        lines.append(f"Pages noted from our chats: {len(digest['chat_pages'])} ("
                     + ", ".join(wiki.page_title(p) or p for p in wiki.capped(digest["chat_pages"])[0]) + ")")
    if digest["sources"]:
        lines.append("New sources: " + ", ".join(
            f"{wiki.page_title(s['page']) or s['page']} ({s['kind'] or 'file'})" for s in wiki.capped(digest["sources"])[0]))
    mem = digest["memory"]
    if wiki.memory_total(digest):
        lines.append(f"Memory: {mem['added']} added, {mem['updated']} updated, {mem['removed']} removed"
                     + "".join(f"; e.g. {e['op'].lower()} {e['path']}: {wiki.memory_value(e['value'])}"
                               for e in mem["examples"]))
    return "\n".join(lines)


def briefing(digest: dict, user_text: str = "what's new?", origin: str = "user") -> tuple:
    """One short completion phrasing the digest (wrapped as `<document source="briefing">`);
    returns ``(reply, card_md)``. The reply ends with DEEPER_OFFERS[0]; `_last_context` holds the
    card so "write it up" needs no completion; the session turns get one "briefed: …" stand-in.
    Raises on a failed or empty completion, leaving no trace."""
    global _deeper_offer, _last_context
    _deeper_offer = None
    block = untrusted(_digest_text(digest), "briefing")
    prompt = BRIEFING_PROMPT.format(when=wiki.briefing_when(digest["since"]))
    items = len(digest["pages"]) + len(digest["chat_pages"])
    with trace.span("briefing", items=items, sources=len(digest["sources"])):
        resp = core_llm.completion(
            "reply", model=core_llm.OMLX_MODEL, max_tokens=BRIEFING_MAX_TOKENS,
            messages=[_conversation[0], {"role": "user", "content": f"{prompt}\n\n{block}"}],
        )
    reply = (resp.choices[0].message.content or "").strip()
    if not strip_tag(reply).strip():
        raise ValueError("the model wrote nothing")
    if not ends_with_offer(reply):
        reply += " " + DEEPER_OFFERS[0]
    card_md = wiki.briefing_card(digest)
    titles = [p["title"] for p in digest["pages"]]
    _conversation.append({"role": "user", "content": user_text})
    _conversation.append({"role": "assistant", "content": reply})
    _session_turns.append({"role": "assistant", "content":
                           f"{memory.STAND_INS[1]}{len(digest['pages']) + len(digest['chat_pages'])} pages, "
                           f"{len(digest['sources'])} sources, {wiki.memory_total(digest)} memory changes"})
    _last_context = {"question": user_text, "extra": block, "tools": [], "reply": reply, "sources": [],
                     "pages": titles, "card": card_md}
    return reply, card_md


def card_in(content: str) -> dict | None:
    """The card an assistant turn holds (`<card id>` marker), or None."""
    m = _CARD_MARK.match(content or "")
    return _cards.get(m.group(1)) if m else None


def recall_for(user_text: str) -> str:
    """The `<recall>` block for this turn, or "" (docs/32-source-recall.md).

    Short messages (< RECALL_MIN_WORDS words) and pure commands (every content word a
    COMMAND_WORDS one, "what time is it") never search; otherwise the best raw chunks above
    `recall_min_score` are quoted, at most RECALL_MAX_CHUNKS of them within RECALL_MAX_TOKENS.
    """
    return _recall_block(*_recall_evidence(user_text))


def _recall_block(page: dict | None, hits: list) -> str:
    """The `<recall>` block quoting `page` and `hits`, or "" when there is nothing to quote."""
    return _quote_block("recall", page, hits, RECALL_MAX_TOKENS, RECALL_NOTE)


def _recall_evidence(user_text: str) -> tuple:
    """``(page, hits)`` recall would quote for this turn; ``(None, [])`` when the turn never searches."""
    terms = wiki.content_terms(user_text)
    if terms <= COMMAND_WORDS:
        return None, []
    proper = wiki.proper_terms(terms)
    words = len(user_text.split())
    if words < RECALL_MIN_WORDS and not (words >= RECALL_MIN_PROPER_WORDS and proper):
        return None, []
    global _evidence
    page = _page_recall(user_text, terms)
    _evidence = (page, _recall_hits(user_text, proper, wiki.recall_min_score(), RECALL_CANDIDATES))
    return _evidence



def recall_band(user_text: str, page: dict | None, hits: list) -> str:
    """"high" (answer grounded), "medium" (answer plainly, offer to check) or "low" (no recall)
    from the evidence recall found: score per topic word plus the kind of match (PRD 45)."""
    top = hits[0] if hits else None
    proper = wiki.proper_terms(wiki.content_terms(user_text))
    matched = wiki.proper_match(top, proper, wiki.proper_phrases(user_text)) if top else set()
    content = content_question(user_text)
    # low: nothing kept, or the gate fails (chat carried by plain words)
    if not (page or hits) or not (content or matched):
        return "low"
    # high: a named page or source, then the rules that need the top hit
    if page or top.get("named"):
        return "high"
    strong_question = content and (matched or top["score"] >= RECALL_HIGH_SCORE)
    name_question = bool(ASK_WORDS.search(user_text)) and bool(matched)
    whole_name = not content and (top.get("phrases") or len(matched) >= 2)
    if strong_question or name_question or whole_name:
        return "high"
    # medium: everything else with a kept hit
    return "medium"


def offer_title(title: str) -> str:
    """A source title up to the first ":" or " (" — what the recall offer names."""
    return re.split(r":| \(", title, maxsplit=1)[0].strip() or title


def _recall_hits(question: str, proper: set, floor: float, k: int, exclude: set = frozenset()) -> list:
    """Up to RECALL_MAX_CHUNKS chunk hits for `question` scoring ≥ `floor`, in rank order, under
    the precision rules: two matched topic words or a proper term whose bigram the chunk holds
    whole (PRD 34 / 42); never a neighbour of a kept chunk, never a (source, part) in `exclude`.
    A source the question names by title is quoted regardless of `floor` (naming it settles
    relevance, which is all the threshold decides)."""
    pairs = wiki.proper_phrases(question)
    kept = []
    for hit in wiki.search_chunks(question, k=k + len(exclude)):
        if len(kept) >= RECALL_MAX_CHUNKS or (hit["score"] < floor and not hit.get("named")):
            break                                               # a named source has no threshold
        if (hit["source"], hit["n"]) in exclude:
            continue
        if any(h["source"] == hit["source"] and abs(h["n"] - hit["n"]) == 1 for h in kept):
            continue                                            # overlaps a kept chunk
        if len(hit["matched"]) >= 2 or wiki.proper_match(hit, proper, pairs):
            kept.append(hit)                                    # one common word is no match
    return kept


def _page_recall(question: str, terms: set) -> dict | None:
    """The top wiki page for `question` as ``{line, path, title}`` when the question names it
    (every token of the page's name among the terms, no tool-command word), else None; the line
    is the head + Summary + best-matching section within RECALL_PAGE_TOKENS (PRD 42 §3)."""
    if terms & COMMAND_WORDS:
        return None
    hits = [h for h in wiki.search(question, 2) if not h["path"].startswith("sources/")]
    if not hits:
        return None
    path = hits[0]["path"]
    text = wiki.page_text(path)
    if text is None:
        return None
    title = wiki.page_title(path)
    name_terms = wiki.content_terms(title) or wiki.content_terms(path.rsplit("/", 1)[-1])
    if not name_terms or not name_terms <= terms:
        return None
    head, sections = wiki.split_sections(wiki.parse_page(text)[1])
    summary = next((s for s in sections if s.startswith("## Summary")), None)
    section = wiki.best_section([s for s in sections if s is not summary], question)
    body = "\n\n".join(p for p in (head, summary, section) if p)
    kept = _fit([f'From page "{title}" ({path}): {body}'], RECALL_PAGE_TOKENS, "")
    if kept:
        _turn_pages.append(title)
    return {"line": kept[0], "path": path, "title": title} if kept else None


def _quote_block(tag: str, page: dict | None, hits: list, budget: int, note_text: str) -> str:
    """`<tag>` block of the page line and the chunk quotes within `budget`, or ""; the chunks
    quoted are recorded in `_turn_sources` for the deeper offer."""
    lines = ([page["line"]] if page else []) + [
        f'From "{h["title"]}" (part {h["n"]}/{h["total"]}): {h["text"]}' for h in hits]
    titles = ([page["title"]] if page else []) + [h["title"] for h in hits]
    quotes = _fit(lines, budget, note_text)
    if not quotes:
        return ""
    titles = titles[:len(quotes)]
    hits = hits[:len(quotes) - (1 if page else 0)]
    _turn_sources.extend((h["source"], h["title"], h["n"]) for h in hits)
    note = note_text.format(titles=" or ".join(f'"{t}"' for t in dict.fromkeys(titles)))
    blocks = [untrusted(q, t) for q, t in zip(quotes, titles)]
    return f"\n\n<{tag}>\n" + "\n\n".join(blocks) + f"\n{note}\n</{tag}>"


def lookup_worthy(question: str) -> bool:
    """A question the filed documents could answer: at least RECALL_MIN_PROPER_WORDS words with a
    content word that is not a tool command ("hi" and "time?" stream untouched)."""
    terms = wiki.content_terms(question)
    return len(question.split()) >= RECALL_MIN_PROPER_WORDS and bool(terms) and not terms <= COMMAND_WORDS


def fallback_block(question: str, reply_start: str) -> str:
    """The `<lookup>` block for a re-ask when `reply_start` claims the documents do not cover
    the question: the search_wiki equivalent run server-side (top page + chunks below the
    threshold, minus the parts this turn already quoted), or "" (PRD 42 §2)."""
    if not NOT_COVERED.search(reply_start):
        return ""
    if not lookup_worthy(question):
        return ""
    terms = wiki.content_terms(question)
    used = {(s, n) for s, _, n in _turn_sources}
    page = _page_recall(question, terms)
    hits = _recall_hits(question, wiki.proper_terms(terms), 0.0, RECALL_CANDIDATES, used)
    return _quote_block("lookup", page, hits, FALLBACK_MAX_TOKENS, FALLBACK_NOTE)


def ask_events(user_text: str, origin: str = "user", note: str = "") -> Iterator[tuple[str, object]]:
    """Yield ("tool_calls", [names]) before tools run, then ("delta", text) chunks.

    The conversation and memory see typed and spoken input identically (PRD 24); `origin`
    tags the session log line, sets the answer length (PRD 43) and, on a spoken turn, the
    recall band (PRD 45). `note` is a one-turn system remark (a page was just opened for the
    user, PRD 38) appended to this turn's user message only.
    """
    global _last_tool_calls, _deeper_offer, _turn_sources, _turn_pages, _last_context, _evidence, last_band

    conv_len = len(_conversation)
    _evidence, last_band = (None, []), None
    turns_len = len(_session_turns)
    # An offer lives exactly one turn: a short "yes" takes it, anything else drops it.
    offer, _deeper_offer = _deeper_offer, None
    yes = bool(offer) and is_affirmative(user_text)
    lookup = offer.get("wikipedia") if yes else None            # the question to look up (PRD 37)
    asked = offer.get("recall") if yes else None                # the earlier question to check (PRD 45)
    deeper = deeper_block(offer) if yes and not (lookup or asked) else ""
    _turn_sources = [(offer["source"], offer["title"], p) for p in offer["parts"]] if deeper else []
    _turn_pages = []
    # The recall / source block rides on this turn's user message only; the session turns (what
    # memory and the wiki ingest see) keep the user's own words.
    # A page open in the UI takes the place of recall (PRD 38); Chat and Memory recall as before.
    focused = "" if deeper or lookup or asked else focus_block(user_text)
    band, offer_text, recall_offer = "", "", None
    if deeper or lookup or focused:
        extra = deeper or ("\n\n" + WIKIPEDIA_NOTE if lookup else focused)
    elif asked:                                                 # "yes" to the recall offer: forced high
        band, extra = "high", recall_for(asked) + "\n\n" + RECALL_YES_NOTE.format(question=asked)
    elif origin == "typed":
        extra = recall_for(user_text)
    else:
        # A spoken turn asks before it looks up (PRD 45): only a high band quotes, weak evidence
        # gets one short offer, never twice in a row.
        page, hits = _recall_evidence(user_text)
        band = recall_band(user_text, page, hits)
        extra = _recall_block(page, hits) if band == "high" else ""
        if band != "high":
            _turn_pages = []
        if band == "medium" and not (offer or {}).get("recall"):
            extra = "\n\n" + RECALL_ASK_NOTE
            recall_offer = {"recall": user_text, "title": offer_title(hits[0]["title"])}
            offer_text = RECALL_OFFER.format(title=recall_offer["title"])
    # A plain Chat turn may be re-asked once with a server-side lookup when the reply starts by
    # claiming the documents do not cover the question (PRD 42): its first characters are held.
    # A medium turn never is: its offer replaces the fallback.
    holding = not (deeper or lookup or focused or asked or band == "medium") and lookup_worthy(user_text)
    if focus["view"] == "memory" and not (deeper or lookup):
        extra += "\n\n" + MEMORY_FOCUS_NOTE
    if note:
        extra += "\n\n" + note
    # Length by origin (PRD 43): a typed content question answered from filed text runs long
    # and drops the offer; a spoken one keeps the short rule and the offer.
    typed_content = origin == "typed" and content_question(user_text)
    written = typed_content and bool(extra.strip())
    if written:
        extra = written_note(extra)
    _note_documents(extra)
    user_msg = {"role": "user", "content": user_text + extra}
    _conversation.append(user_msg)
    _session_turns.append({"role": "user", "content": user_text})
    _last_tool_calls = []
    trim_history()

    long = bool(deeper or lookup or written)
    tool_texts: list = []

    try:
        if deeper or asked:
            yield ("tool_calls", ["dig_deeper"])               # a filler while the grounded answer starts
        # Streamed first pass: text deltas go out immediately, tool-call fragments accumulate.
        # After a "yes" to the Wikipedia offer the text is held back instead: the pass is there
        # for the model to name the topic, and get_wikipedia runs either way.
        first = core_llm.stream("reply", model=core_llm.OMLX_MODEL, messages=_messages(long, written), tools=TOOLS,
                        max_tokens=_budget(long))
        parts: list[str] = []
        pending: dict = {}
        held: list[str] = []                                    # text not yet yielded (fallback hold)
        retry = ""
        for chunk in first:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text and lookup:
                parts.append(text)                              # held back: get_wikipedia runs anyway
            elif text and offer_text:
                parts.append(text)                              # a medium reply: held whole (short)
            elif text and holding:
                parts.append(text)
                held.append(text)
                if len("".join(held)) >= FALLBACK_HOLD_CHARS:
                    holding, retry = False, fallback_block(user_text, "".join(held))
                    if retry:
                        break
                    for t in held:
                        yield ("delta", t)
                    held = []
            elif text:
                parts.append(text)
                yield ("delta", text)
            fragments = getattr(delta, "tool_calls", None)
            if fragments:
                _accumulate_tool_calls(pending, fragments)
        if holding and not pending:                             # a short reply: decide on all of it
            retry = fallback_block(user_text, "".join(held))
            if not retry:
                for t in held:
                    yield ("delta", t)
                held = []

        if lookup and not any(tc["name"] == "get_wikipedia" for tc in pending.values()):
            pending = {0: {"id": "wikipedia_offer", "name": "get_wikipedia",
                           "arguments": json.dumps({"query": lookup})}}
            parts = []                                          # held back, never spoken
        if retry:
            # The held reply is dropped unspoken; the turn is asked once more with the lookup.
            log.info("fallback lookup: the reply claimed the documents do not cover it")
            getattr(first, "close", lambda: None)()             # stop the abandoned generation
            yield ("tool_calls", ["search_wiki"])
            if typed_content:                                   # grounded after all: the policy applies
                written = long = True
                retry = written_note(retry)
            user_msg["content"] += retry
            _note_documents(retry)
            parts = []
            stream = core_llm.stream("fallback", fallback_reason="not_covered", model=core_llm.OMLX_MODEL,
                             messages=_messages(long, written), max_tokens=_budget(long))
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    yield ("delta", delta)
        elif pending:
            tool_calls = [pending[i] for i in sorted(pending)]
            if parts and not lookup and not held and not offer_text:
                log.warning("warning: text before tool call was spoken")
            parts = []  # spoken or held back; never the reply
            yield ("tool_calls", [tc["name"] for tc in tool_calls])
            tool_texts = _tool_round(tool_calls)
            if typed_content and any(tc["name"] in GROUNDING_TOOLS for tc in tool_calls):
                written = long = True
            stream = core_llm.stream("tool_round", model=core_llm.OMLX_MODEL, messages=_messages(long, written),
                             max_tokens=_budget(long))
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    yield ("delta", delta)
        if offer_text and not pending:
            # the held medium reply goes out without any offer the model ended it with, then the
            # recall offer: the turn's only one
            text = strip_offer("".join(parts))
            parts = [text, " " + offer_text]
            yield ("delta", text)
            yield ("delta", " " + offer_text)
        reply = "".join(parts)
    except Exception:
        del _conversation[conv_len:]
        del _session_turns[turns_len:]
        raise

    # The turn is done: the block was for this completion only, the history keeps the bare
    # words (PRD 39) — the reply quotes the source, follow-ups use read_source / dig deeper.
    block = user_msg["content"][len(user_text):]
    user_msg["content"] = user_text
    _conversation.append({"role": "assistant", "content": reply})
    _session_turns.append({"role": "assistant", "content": reply})
    _deeper_offer = recall_offer if offer_text and not pending else _offer_from_turn(reply, user_text)
    _last_context = {"question": asked or user_text, "extra": block, "tools": tool_texts, "reply": reply,
                     "sources": list(_turn_sources), "pages": list(_turn_pages)}
    last_band = band or None
    trace.event("recall", band=last_band, page=(_evidence[0] or {}).get("path"),
                offer=_offer_kind(bool(offer_text and not pending)),
                hits=[{"source": h.get("source"), "n": h.get("n"), "score": round(float(h.get("score") or 0), 3),
                       "named": bool(h.get("named"))} for h in _evidence[1][:5]])


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


def has_unsaved_turns() -> bool:
    """True when this session has turns that no save has folded into memory.json yet."""
    return bool(_session_turns)


def save_memory():
    """Fold this session into the memory.json profile. Idempotent: turns are cleared once saved.

    When the save could not reach the model (memory.save returns None, or raises) the turns go
    back in front of anything said meanwhile, so the next save — or the shutdown one — retries.
    """
    global _session_turns, _sources_seen
    if not _session_turns:
        return
    turns, _session_turns = _session_turns, []
    sources, _sources_seen = _sources_seen, []

    def restore():
        global _session_turns, _sources_seen
        _session_turns = turns + _session_turns
        _sources_seen = sources + [s for s in _sources_seen if s not in sources]

    try:
        applied = memory.save(turns, sources=sources)
    except Exception:
        restore()
        raise
    if applied is None:
        restore()


# --- new topic (docs/39-context-meter.md) ---------------------------------------------
SUMMARY_TOKENS = 150
SUMMARY_PROMPT = ("Summarise the conversation above in one short paragraph of plain prose, at most 100 "
                  "words: what we talked about, decisions made, open threads. No lists, no preamble, "
                  "no emotion tag.")
NEW_TOPIC_ACK = "Fresh start — I kept the gist above."
EARLIER_EMPTY = "Nothing was discussed before this point."


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]


def _summarise(turns: list) -> str:
    """One ≤ SUMMARY_TOKENS completion over the turns; on failure the turns' first sentences
    (one error print, never blocks)."""
    if not turns:
        return EARLIER_EMPTY
    try:
        resp = core_llm.completion(
            "summary", model=core_llm.OMLX_MODEL, messages=turns + [{"role": "user", "content": SUMMARY_PROMPT}],
            max_tokens=SUMMARY_TOKENS,
        )
        summary = strip_tag(resp.choices[0].message.content or "").strip()
        if not summary:
            raise ValueError("empty summary")
    except Exception as e:
        log.exception("summary failed; keeping the turns' first sentences")
        trace.error("memory", "new_topic", e, fatal_for="none")
        summary = " ".join(_first_sentence(strip_tag(t["content"])) for t in turns)
    return summary[:SUMMARY_TOKENS * 4].strip()


def dialogue() -> list:
    """The user and assistant lines the model holds, as `{"role", "content"}` copies: tool
    rounds and empty assistant turns left out."""
    return [{"role": m["role"], "content": m["content"]} for m in _conversation[1:]
            if m.get("role") in ("user", "assistant") and m.get("content")]


def new_thread() -> str:
    """Fold memory, then restart the conversation on a short visible summary; returns it.

    `_conversation` becomes [system, `<earlier>summary</earlier>` (user), NEW_TOPIC_ACK]; the
    deeper offer, the turn's sources, the remembered page and the trim count are cleared.
    """
    global _deeper_offer, _turn_sources, last_page, _last_trimmed, _context_warned, _last_context
    turns = dialogue()
    _last_context = None
    try:
        save_memory()
    except Exception as e:
        log.exception("memory save failed")
        trace.error("memory", "memory_save", e, fatal_for="none")
    summary = _summarise(turns)
    _conversation[1:] = [{"role": "user", "content": f"<earlier>{summary}</earlier>"},
                         {"role": "assistant", "content": NEW_TOPIC_ACK}]
    _deeper_offer, _turn_sources, last_page = None, [], None
    _last_trimmed, _context_warned = 0, False
    _forget_cards()
    return summary


def get_sources_seen() -> list:
    return _sources_seen


reset()
