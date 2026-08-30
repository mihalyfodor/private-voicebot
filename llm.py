import json
import os
import re
from datetime import datetime
from typing import Iterator

from openai import OpenAI

import memory
import avatars
import wiki
from splitter import EMOTIONS
from tools import TOOLS, run_tool
from untrusted import DOCUMENT_RULE, documents_in, untrusted

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


#: the fixed offer phrases a grounded answer may end with (docs/34-grounded-answers.md); the
#: model is asked for exactly one of them so `_deeper_offer` can be detected on its output
DEEPER_OFFERS = ("Want me to dig deeper?", "Shall I dig deeper?", "Want me to read on?")
GROUNDING_RULES = (
    "\n\nLook it up first. When the wiki index or a recall block in the message names a source or "
    "page on the topic, use search_wiki or read_source (or the recall text) before answering from your own "
    "knowledge, and name the source once, by title, in a short natural clause woven into a "
    "sentence (\"per the Skyrim guide\", \"the Fallout 4 walkthrough says\") — vary the wording, "
    "never a separate closing sentence like \"This information comes from\". Whenever the "
    "answer comes from a document passage (a recall block, a search_wiki passage or read_source), "
    "keep it to one to three sentences and make the very last sentence of the reply exactly: "
    f"{DEEPER_OFFERS[0]} — the document always has more around that passage. "
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
    "offer. Never call get_wikipedia on your own; call it only after that yes, and then "
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
    f"{avatar['name']} included. Stay in character only in how you phrase the tool's result.\n\n"
    f"{DOCUMENT_RULE}"
    )


SYSTEM_PROMPT = build_system_prompt(avatars.current())

_client: OpenAI | None = None
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


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY)
    return _client


def reset():
    global _conversation, _session_turns, _last_tool_calls, _deeper_offer, _turn_sources, _sources_seen
    _conversation = [{"role": "system", "content": memory.load(SYSTEM_PROMPT)}]
    _session_turns = []
    _last_tool_calls = []
    _deeper_offer = None
    _turn_sources = []
    _sources_seen = []


def _note_documents(text: str) -> None:
    """Record the document blocks in a sent message or tool result for memory attribution."""
    for name in documents_in(text):
        if name not in _sources_seen:
            _sources_seen.append(name)


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
        _note_tool_sources(name, args, str(result))
        _note_documents(str(result))
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
RECALL_NOTE = ("If this answers the question, answer from it in one to three sentences, name {titles} "
               "once in a short natural clause inside a sentence (never as a separate closing sentence), "
               "and finish with exactly this sentence: " + DEEPER_OFFERS[0] + " Otherwise ignore it.")

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
               "Do not offer to dig deeper again.")
#: a short "yes" to the deeper offer: one of these words first ...
_AFFIRMATIVE_LEADS = frozenset("""yes yeah yep yup sure ok okay please absolutely definitely certainly
go dig deeper dive read tell more why do that i lets let check look""".split())
#: ... and nothing but these after it, at most AFFIRMATIVE_MAX_WORDS in all
_AFFIRMATIVE_WORDS = _AFFIRMATIVE_LEADS | frozenset("""on ahead it in me not thanks thank you would be
great good d like s so up wikipedia""".split())
AFFIRMATIVE_MAX_WORDS = 6
_SNIPPET_PART = re.compile(r"^(.+?) · part (\d+)/\d+ \(([^)]+)\)", re.M)


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


def _messages(deeper: bool) -> list:
    """The conversation as sent: on a deeper turn the system rule reads `long` for this completion."""
    if not deeper or not _conversation:
        return _conversation
    system = _conversation[0]["content"].replace(VERBOSITY_RULES[_verbosity], VERBOSITY_RULES["long"], 1)
    return [{"role": "system", "content": system}] + _conversation[1:]


def recall_for(user_text: str) -> str:
    """The `<recall>` block for this turn, or "" (docs/32-source-recall.md).

    Short messages (< RECALL_MIN_WORDS words) and pure commands (every content word a
    COMMAND_WORDS one, "what time is it") never search; otherwise the best raw chunks above
    `recall_min_score` are quoted, at most RECALL_MAX_CHUNKS of them within RECALL_MAX_TOKENS.
    """
    terms = wiki.content_terms(user_text)
    if len(user_text.split()) < RECALL_MIN_WORDS or terms <= COMMAND_WORDS:
        return ""
    floor = wiki.recall_min_score()
    proper = wiki.proper_terms(terms)
    hits = []
    for hit in wiki.search_chunks(user_text, k=RECALL_MAX_CHUNKS):
        if hit["score"] < floor:
            break
        if len(hit["matched"]) >= 2 or proper.intersection(hit["matched"]):
            hits.append(hit)                                    # one common word is no match
    lines = [f'From "{h["title"]}" (part {h["n"]}/{h["total"]}): {h["text"]}' for h in hits]
    quotes = _fit(lines, RECALL_MAX_TOKENS, RECALL_NOTE)
    if not quotes:
        return ""
    hits = hits[:len(quotes)]
    _turn_sources.extend((h["source"], h["title"], h["n"]) for h in hits)
    note = RECALL_NOTE.format(titles=" or ".join(f'"{t}"' for t in dict.fromkeys(h["title"] for h in hits)))
    blocks = [untrusted(q, h["title"]) for q, h in zip(quotes, hits)]
    return "\n\n<recall>\n" + "\n\n".join(blocks) + f"\n{note}\n</recall>"


def ask_events(user_text: str, origin: str = "user") -> Iterator[tuple[str, object]]:
    """Yield ("tool_calls", [names]) before tools run, then ("delta", text) chunks.

    `origin` only reaches the session log line; the conversation and memory see typed
    and spoken input identically (PRD 24).
    """
    global _last_tool_calls, _deeper_offer, _turn_sources

    conv_len = len(_conversation)
    turns_len = len(_session_turns)
    # An offer lives exactly one turn: a short "yes" takes it, anything else drops it.
    offer, _deeper_offer = _deeper_offer, None
    yes = bool(offer) and is_affirmative(user_text)
    lookup = offer.get("wikipedia") if yes else None            # the question to look up (PRD 37)
    deeper = deeper_block(offer) if yes and not lookup else ""
    _turn_sources = [(offer["source"], offer["title"], p) for p in offer["parts"]] if deeper else []
    # The recall / source block rides on this turn's user message only; the session turns (what
    # memory and the wiki ingest see) keep the user's own words.
    extra = deeper or ("\n\n" + WIKIPEDIA_NOTE if lookup else recall_for(user_text))
    _note_documents(extra)
    _conversation.append({"role": "user", "content": user_text + extra})
    _session_turns.append({"role": "user", "content": user_text})
    _last_tool_calls = []
    trim_history()

    long = bool(deeper or lookup)
    max_tokens = VERBOSITY_MAX_TOKENS["long"] if long else _effective_max_tokens()

    try:
        if deeper:
            yield ("tool_calls", ["dig_deeper"])               # a filler while the long answer starts
        # Streamed first pass: text deltas go out immediately, tool-call fragments accumulate.
        # After a "yes" to the Wikipedia offer the text is held back instead: the pass is there
        # for the model to name the topic, and get_wikipedia runs either way.
        first = client().chat.completions.create(
            model=OMLX_MODEL, messages=_messages(long), tools=TOOLS, stream=True,
            max_tokens=max_tokens,
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
                if not lookup:
                    yield ("delta", text)
            fragments = getattr(delta, "tool_calls", None)
            if fragments:
                _accumulate_tool_calls(pending, fragments)

        if lookup and not any(tc["name"] == "get_wikipedia" for tc in pending.values()):
            pending = {0: {"id": "wikipedia_offer", "name": "get_wikipedia",
                           "arguments": json.dumps({"query": lookup})}}
            parts = []                                          # held back, never spoken
        if pending:
            tool_calls = [pending[i] for i in sorted(pending)]
            if parts and not lookup:
                print("[llm] warning: text before tool call was spoken")
                parts = []  # already spoken; don't record it as the reply
            yield ("tool_calls", [tc["name"] for tc in tool_calls])
            _tool_round(tool_calls)
            stream = client().chat.completions.create(
                model=OMLX_MODEL, messages=_messages(long), stream=True, max_tokens=max_tokens,
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
    _deeper_offer = _offer_from_turn(reply, user_text)
    _log(user_text, _last_tool_calls, reply, origin)


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
        applied = memory.save(turns, client(), OMLX_MODEL, sources=sources)
    except Exception:
        restore()
        raise
    if applied is None:
        restore()


def get_sources_seen() -> list:
    return _sources_seen


def _log(user_text: str, tool_calls: list, reply: str, origin: str = "user"):
    """One session.log line per turn; `origin` is the prefix ("user" spoken, "typed" composer)."""
    tools_str = ",".join(tc["name"] for tc in tool_calls) if tool_calls else "none"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {origin}: {user_text!r} | tools: {tools_str} | reply_len: {len(reply)}\n"
    with open(LOG_PATH, "a") as f:
        f.write(line)


reset()
