import json
import re
from typing import Iterator
from agent import fold
from memory import identity
import avatars
from core import trace
from core import llm as core_llm
from vault import chunks as v_chunks
from vault import digest as v_digest
from vault import focus as v_focus
from splitter import strip_tag
from tools import registry
from core.untrusted import documents_in, untrusted
from agent import context as a_context
from agent import offers
from agent import prompt as a_prompt
from agent import recall as a_recall

log = trace.logger("llm")
tool_log = trace.logger("Tool")


run_tool = registry.run_tool
trace.vocab("name", [t.name for t in registry.REGISTRY if t.schema] + ["dig_deeper"])

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


def reset():
    global _conversation, _session_turns, _last_tool_calls, _deeper_offer, _turn_sources, _sources_seen
    global _turn_pages, _last_context, _card_seq
    _conversation = [{"role": "system", "content": identity.load(a_prompt.SYSTEM_PROMPT)}]
    _session_turns = []
    _last_tool_calls = []
    _deeper_offer = None
    _turn_sources = []
    _turn_pages = []
    _sources_seen = []
    _last_context = None
    a_context._last_trimmed = 0
    a_context._context_warned = False
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
        offers._note_tool_sources(name, args, str(result))
        _note_documents(str(result))
        results.append(str(result))
        _conversation.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
    return results


def _rebuild_system_prompt(note: str = "") -> None:
    """Rebuild SYSTEM_PROMPT from the current avatar and swap `_conversation[0]` in place."""

    a_prompt.SYSTEM_PROMPT = a_prompt.build_system_prompt(avatars.current())
    system = {"role": "system", "content": identity.load(a_prompt.SYSTEM_PROMPT) + note}
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


def record_user(text: str) -> None:
    """Append a user turn that took no completion (an approval's request or answer, PRD 52)."""
    _conversation.append({"role": "user", "content": text})
    _session_turns.append({"role": "user", "content": text})


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


def _messages(deeper: bool, written: bool = False) -> list:
    """The conversation as sent: on a deeper turn the system rule reads `long` for this completion;
    on a written turn (typed, grounded, content — PRD 43) the short-answer clause reads full too."""
    if not deeper or not _conversation:
        return _conversation
    system = _conversation[0]["content"].replace(a_prompt.VERBOSITY_RULES[a_prompt._verbosity], a_prompt.VERBOSITY_RULES["long"], 1)
    if written:
        system = system.replace(a_prompt.OFFER_CLAUSE, a_prompt.WRITTEN_CLAUSE, 1)
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
CARD_MAX_TOKENS = a_prompt.VERBOSITY_MAX_TOKENS["long"]
CARD_CONTEXT_TOKENS = offers.DEEPER_MAX_TOKENS
CARD_HISTORY_TURNS = 6
_CARD_MARK = re.compile(r"^<card id=\"([a-z0-9]+)\">\n(.*)\n</card>$", re.S)


def content_question(text: str) -> bool:
    """A question that wants an explanation: a CONTENT_WORDS word or CONTENT_MIN_WORDS+ words."""
    return bool(CONTENT_WORDS.search(text)) or len(text.split()) >= CONTENT_MIN_WORDS


def written_note(extra: str) -> str:
    """The turn's recall / lookup block with its short-answer wording turned into the full one."""
    return extra.replace(a_recall.SHORT_PHRASE, a_recall.FULL_PHRASE).replace(a_recall.NOTE_OFFER, a_recall.NOTE_NO_OFFER)


def source_parts(sources: list) -> list:
    """`{title, part: "3/120"}` per passage (deduped, in order): the card's Sources line and the
    write-up run's `context` step (PRD 51)."""
    out = []
    for source, title, part in sources:
        try:
            row = {"title": title, "part": f"{part}/{len(v_chunks.raw_chunks(source)['chunks'])}"}
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


def card_title(markdown: str, question: str) -> str:
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


def get_card(card_id) -> dict | None:
    """The issued / filed card with this id, or None — the one reader of the card store."""
    return _cards.get(str(card_id or ""))


def cards_for_run(run_id: str) -> list:
    """The cards that belong to run `run_id` (a run card, or an approval's awaiting card)."""
    return [c for c in _cards.values() if c.get("run_id") == run_id]


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
    card.update(markdown=markdown, sources=sources, title=card_title(markdown, ctx["question"]))
    _cards[card["id"]] = card
    _loose.pop(card["id"], None)
    last_card = card
    if request:
        _conversation.append({"role": "user", "content": request})
        _session_turns.append({"role": "user", "content": request})
    _conversation.append({"role": "assistant", "content": f'<card id="{card["id"]}">\n{markdown}\n</card>'})
    _session_turns.append({"role": "assistant", "content": f"{fold.STAND_INS[0]}{card['title']}"})
    return card


def file_research(card: dict, request: str) -> dict:
    """Make a finished research card (docs/54-vault-map-research.md) a turn of the conversation: the
    request and the card marker in `_conversation`, "researched: <title>" in the session turns, `last_card`
    (what "save that" means) and `_last_context` holding its markdown, so "write it up" / `/write` after it
    returns the same card without a completion (the briefing pattern)."""
    global last_card, _deeper_offer, _last_context
    _deeper_offer = None
    _cards[card["id"]] = card
    _loose.pop(card["id"], None)
    last_card = card
    _conversation.append({"role": "user", "content": request})
    _session_turns.append({"role": "user", "content": request})
    _conversation.append({"role": "assistant", "content": f'<card id="{card["id"]}">\n{card["markdown"]}\n</card>'})
    _session_turns.append({"role": "assistant", "content": f"{fold.STAND_INS[2]}{card['title']}"})
    _last_context = {"question": request, "extra": "", "tools": [], "reply": "", "sources": [], "pages": [],
                     "card": card["markdown"], "research_card": card}
    return card


def research_card() -> dict | None:
    """The research card the last answer was, or None: "write it up" / `/write` re-shows it (PRD 54)."""
    return (_last_context or {}).get("research_card")


def reuse_card(card: dict, request: str) -> None:
    """A write-up request answered by an existing card: it becomes `last_card` again; the request is a turn."""
    global last_card, _deeper_offer
    _deeper_offer = None
    last_card = card
    record_user(request)


def _card_completion(ctx: dict) -> tuple:
    """``(markdown, sources)``: one `long` completion over the last answer's context + its Sources line."""
    material = "\n\n".join(p for p in [ctx["extra"].strip()] + ctx["tools"] if p)
    material = "".join(a_recall.fit([material], CARD_CONTEXT_TOKENS, CARD_PROMPT)) if material else ""
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
        titles, extra = v_digest.capped([p["title"] for p in digest["pages"] if p["change"] == change])
        if titles:
            more = f" (+{extra} more)" if extra else ""
            lines.append(f"Pages {change}: " + ", ".join(titles) + more)
    if digest["chat_pages"]:
        lines.append(f"Pages noted from our chats: {len(digest['chat_pages'])} ("
                     + ", ".join(v_focus.page_title(p) or p for p in v_digest.capped(digest["chat_pages"])[0]) + ")")
    if digest["sources"]:
        lines.append("New sources: " + ", ".join(
            f"{v_focus.page_title(s['page']) or s['page']} ({s['kind'] or 'file'})" for s in v_digest.capped(digest["sources"])[0]))
    mem = digest["memory"]
    if v_digest.memory_total(digest):
        lines.append(f"Memory: {mem['added']} added, {mem['updated']} updated, {mem['removed']} removed"
                     + "".join(f"; e.g. {e['op'].lower()} {e['path']}: {v_digest.memory_value(e['value'])}"
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
    prompt = BRIEFING_PROMPT.format(when=v_digest.briefing_when(digest["since"]))
    items = len(digest["pages"]) + len(digest["chat_pages"])
    with trace.span("briefing", items=items, sources=len(digest["sources"])):
        resp = core_llm.completion(
            "reply", model=core_llm.OMLX_MODEL, max_tokens=BRIEFING_MAX_TOKENS,
            messages=[_conversation[0], {"role": "user", "content": f"{prompt}\n\n{block}"}],
        )
    reply = (resp.choices[0].message.content or "").strip()
    if not strip_tag(reply).strip():
        raise ValueError("the model wrote nothing")
    if not offers.ends_with_offer(reply):
        reply += " " + a_prompt.DEEPER_OFFERS[0]
    card_md = v_digest.briefing_card(digest)
    titles = [p["title"] for p in digest["pages"]]
    _conversation.append({"role": "user", "content": user_text})
    _conversation.append({"role": "assistant", "content": reply})
    _session_turns.append({"role": "assistant", "content":
                           f"{fold.STAND_INS[1]}{len(digest['pages']) + len(digest['chat_pages'])} pages, "
                           f"{len(digest['sources'])} sources, {v_digest.memory_total(digest)} memory changes"})
    _last_context = {"question": user_text, "extra": block, "tools": [], "reply": reply, "sources": [],
                     "pages": titles, "card": card_md}
    return reply, card_md


def card_in(content: str) -> dict | None:
    """The card an assistant turn holds (`<card id>` marker), or None."""
    m = _CARD_MARK.match(content or "")
    return _cards.get(m.group(1)) if m else None


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
    yes = bool(offer) and offers.is_affirmative(user_text)
    lookup = offer.get("wikipedia") if yes else None            # the question to look up (PRD 37)
    asked = offer.get("recall") if yes else None                # the earlier question to check (PRD 45)
    deeper = offers.deeper_block(offer) if yes and not (lookup or asked) else ""
    _turn_sources = [(offer["source"], offer["title"], p) for p in offer["parts"]] if deeper else []
    _turn_pages = []
    # The recall / source block rides on this turn's user message only; the session turns (what
    # memory and the wiki ingest see) keep the user's own words.
    # A page open in the UI takes the place of recall (PRD 38); Chat and Memory recall as before.
    focused = "" if deeper or lookup or asked else a_recall.focus_block(user_text)
    band, offer_text, recall_offer = "", "", None
    if deeper or lookup or focused:
        extra = deeper or ("\n\n" + offers.WIKIPEDIA_NOTE if lookup else focused)
    elif asked:                                                 # "yes" to the recall offer: forced high
        band, extra = "high", a_recall.recall_for(asked) + "\n\n" + a_recall.RECALL_YES_NOTE.format(question=asked)
    elif origin == "typed":
        extra = a_recall.recall_for(user_text)
    else:
        # A spoken turn asks before it looks up (PRD 45): only a high band quotes, weak evidence
        # gets one short offer, never twice in a row.
        page, hits = a_recall._recall_evidence(user_text)
        band = a_recall.recall_band(user_text, page, hits)
        extra = a_recall._recall_block(page, hits) if band == "high" else ""
        if band != "high":
            _turn_pages = []
        if band == "medium" and not (offer or {}).get("recall"):
            extra = "\n\n" + a_recall.RECALL_ASK_NOTE
            recall_offer = {"recall": user_text, "title": a_recall.offer_title(hits[0]["title"])}
            offer_text = a_recall.RECALL_OFFER.format(title=recall_offer["title"])
    # A plain Chat turn may be re-asked once with a server-side lookup when the reply starts by
    # claiming the documents do not cover the question (PRD 42): its first characters are held.
    # A medium turn never is: its offer replaces the fallback.
    holding = not (deeper or lookup or focused or asked or band == "medium") and a_recall.lookup_worthy(user_text)
    if a_recall.focus["view"] == "memory" and not (deeper or lookup):
        extra += "\n\n" + a_recall.MEMORY_FOCUS_NOTE
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
    a_context.trim_history()

    long = bool(deeper or lookup or written)
    tool_texts: list = []

    try:
        if deeper or asked:
            yield ("tool_calls", ["dig_deeper"])               # a filler while the grounded answer starts
        # Streamed first pass: text deltas go out immediately, tool-call fragments accumulate.
        # After a "yes" to the Wikipedia offer the text is held back instead: the pass is there
        # for the model to name the topic, and get_wikipedia runs either way.
        first = core_llm.stream("reply", model=core_llm.OMLX_MODEL, messages=_messages(long, written), tools=a_prompt.TOOLS,
                        max_tokens=a_prompt._budget(long))
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
                if len("".join(held)) >= a_recall.FALLBACK_HOLD_CHARS:
                    holding, retry = False, a_recall.fallback_block(user_text, "".join(held))
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
            retry = a_recall.fallback_block(user_text, "".join(held))
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
                             messages=_messages(long, written), max_tokens=a_prompt._budget(long))
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
                             max_tokens=a_prompt._budget(long))
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    yield ("delta", delta)
        if offer_text and not pending:
            # the held medium reply goes out without any offer the model ended it with, then the
            # recall offer: the turn's only one
            text = offers.strip_offer("".join(parts))
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
    _deeper_offer = recall_offer if offer_text and not pending else offers._offer_from_turn(reply, user_text)
    _last_context = {"question": asked or user_text, "extra": block, "tools": tool_texts, "reply": reply,
                     "sources": list(_turn_sources), "pages": list(_turn_pages)}
    last_band = band or None
    trace.event("recall", band=last_band, page=(_evidence[0] or {}).get("path"),
                offer=offers._offer_kind(bool(offer_text and not pending)),
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
        applied = fold.save(turns, sources=sources)
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
    global _deeper_offer, _turn_sources, _last_context
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
    _deeper_offer, _turn_sources, a_recall.last_page = None, [], None
    a_context._last_trimmed, a_context._context_warned = 0, False
    _forget_cards()
    _rebuild_system_prompt()            # the vault map is computed here and at start, not per turn (PRD 54)
    return summary


def get_sources_seen() -> list:
    return _sources_seen


reset()


# --- step 9 forwarding (PRD 50; deleted with the shim at step 12) ---------------------------------------------
# Names that moved to agent/ stay readable here; assigning one would patch a copy nothing reads, so it raises.
import sys as _sys
import types as _types

_MOVED = (a_prompt, a_context, a_recall, offers)


def _build():
    owners = {}
    for module in _MOVED:
        for name, obj in vars(module).items():
            if not isinstance(obj, _types.ModuleType) and (not callable(obj) or getattr(obj, "__module__", module.__name__) == module.__name__):
                owners.setdefault(name, module)
    return owners


_OWNERS = _build()      # fixed at import, so a patched value still forwards


def __getattr__(name):
    module = _OWNERS.get(name)
    if module is None:
        raise AttributeError(f"module 'llm' has no attribute {name!r}")
    return getattr(module, name)


class _Forwarding(_types.ModuleType):
    def __setattr__(self, name, value):
        if name not in vars(self) and name in _OWNERS:
            raise AttributeError(f"llm.{name} moved to agent/: patch the module that defines it")
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Forwarding
