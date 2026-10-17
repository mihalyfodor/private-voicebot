"""Written answers and briefings: cards (issue, get, reuse, file research), the write-up completion and the briefing recap (docs/43, 46, 51; PRD 50 step 10, from llm.py)."""
import re
from agent import fold
from core import trace
from core import llm as core_llm
from vault import chunks as v_chunks
from agent import digest as v_digest
from vault import focus as v_focus
from core.emotions import strip_tag
from core.untrusted import untrusted
from agent import offers
from agent import prompt as a_prompt
from agent import recall as a_recall
from agent import conversation as a_conversation



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
CARD_CONTEXT_TOKENS = a_prompt.DEEPER_MAX_TOKENS
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
    return bool(a_conversation._last_context)


def next_card_id() -> str:
    """A fresh card id: the counter advances on every card issued, so a failed write-up's id is
    never reused by the retry (PRD 51)."""

    a_conversation._card_seq += 1
    return f"card{a_conversation._card_seq}"


def issue_card(card: dict) -> dict:
    """Keep a run card that is not a conversation turn yet (a write-up still running, or failed) so a
    reconnect replays it where it was issued; `write_up` fills the same dict in when it succeeds."""
    a_conversation._cards[card["id"]] = card
    a_conversation._loose[card["id"]] = len(a_conversation.dialogue())
    return card


def get_card(card_id) -> dict | None:
    """The issued / filed card with this id, or None — the one reader of the card store."""
    return a_conversation._cards.get(str(card_id or ""))


def cards_for_run(run_id: str) -> list:
    """The cards that belong to run `run_id` (a run card, or an approval's awaiting card)."""
    return [c for c in a_conversation._cards.values() if c.get("run_id") == run_id]


def loose_cards() -> list:
    """``(position, card)`` of the issued cards that are not conversation turns, in issue order."""
    return [(pos, a_conversation._cards[cid]) for cid, pos in a_conversation._loose.items() if cid in a_conversation._cards]


def context_parts() -> list:
    """The last answer's passages as `{title, part}` (what a write-up is built from)."""
    return source_parts((a_conversation._last_context or {}).get("sources") or [])


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

    ctx = a_conversation._last_context
    if not ctx:
        raise ValueError(NOTHING_TO_WRITE)
    a_conversation._deeper_offer = None                                        # the offer's "write it up" reply takes it
    if ctx.get("card"):                                         # a briefing: its card is already built (PRD 46)
        markdown, sources = ctx["card"].strip(), []
    else:
        markdown, sources = _card_completion(ctx)
    card = a_conversation._cards.get(card_id) if card_id else None         # the running card issued at turn start (PRD 51)
    card = card if card is not None else {"id": card_id or next_card_id()}
    card.update(markdown=markdown, sources=sources, title=card_title(markdown, ctx["question"]))
    a_conversation._cards[card["id"]] = card
    a_conversation._loose.pop(card["id"], None)
    a_conversation.last_card = card
    if request:
        a_conversation._conversation.append({"role": "user", "content": request})
        a_conversation._session_turns.append({"role": "user", "content": request})
    a_conversation._conversation.append({"role": "assistant", "content": f'<card id="{card["id"]}">\n{markdown}\n</card>'})
    a_conversation._session_turns.append({"role": "assistant", "content": f"{fold.STAND_INS[0]}{card['title']}"})
    return card


def file_research(card: dict, request: str) -> dict:
    """Make a finished research card (docs/54-vault-map-research.md) a turn of the conversation: the
    request and the card marker in `_conversation`, "researched: <title>" in the session turns, `last_card`
    (what "save that" means) and `_last_context` holding its markdown, so "write it up" / `/write` after it
    returns the same card without a completion (the briefing pattern)."""

    a_conversation._deeper_offer = None
    a_conversation._cards[card["id"]] = card
    a_conversation._loose.pop(card["id"], None)
    a_conversation.last_card = card
    a_conversation._conversation.append({"role": "user", "content": request})
    a_conversation._session_turns.append({"role": "user", "content": request})
    a_conversation._conversation.append({"role": "assistant", "content": f'<card id="{card["id"]}">\n{card["markdown"]}\n</card>'})
    a_conversation._session_turns.append({"role": "assistant", "content": f"{fold.STAND_INS[2]}{card['title']}"})
    a_conversation._last_context = {"question": request, "extra": "", "tools": [], "reply": "", "sources": [], "pages": [],
                     "card": card["markdown"], "research_card": card}
    return card


def research_card() -> dict | None:
    """The research card the last answer was, or None: "write it up" / `/write` re-shows it (PRD 54)."""
    return (a_conversation._last_context or {}).get("research_card")


def reuse_card(card: dict, request: str) -> None:
    """A write-up request answered by an existing card: it becomes `last_card` again; the request is a turn."""

    a_conversation._deeper_offer = None
    a_conversation.last_card = card
    a_conversation.record_user(request)


def _card_completion(ctx: dict) -> tuple:
    """``(markdown, sources)``: one `long` completion over the last answer's context + its Sources line."""
    material = "\n\n".join(p for p in [ctx["extra"].strip()] + ctx["tools"] if p)
    material = "".join(a_recall.fit([material], CARD_CONTEXT_TOKENS, CARD_PROMPT)) if material else ""
    user = f"Question: {ctx['question']}\n\nThe answer you gave: {strip_tag(ctx['reply'])}"
    if material:
        user += "\n\nMaterial:\n" + material
    history = [m for m in a_conversation.dialogue()[-CARD_HISTORY_TURNS:] if not card_in(m["content"])]
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

    a_conversation._deeper_offer = None
    block = untrusted(_digest_text(digest), "briefing")
    prompt = BRIEFING_PROMPT.format(when=v_digest.briefing_when(digest["since"]))
    items = len(digest["pages"]) + len(digest["chat_pages"])
    with trace.span("briefing", items=items, sources=len(digest["sources"])):
        resp = core_llm.completion(
            "reply", model=core_llm.OMLX_MODEL, max_tokens=BRIEFING_MAX_TOKENS,
            messages=[a_conversation._conversation[0], {"role": "user", "content": f"{prompt}\n\n{block}"}],
        )
    reply = (resp.choices[0].message.content or "").strip()
    if not strip_tag(reply).strip():
        raise ValueError("the model wrote nothing")
    if not offers.ends_with_offer(reply):
        reply += " " + a_prompt.DEEPER_OFFERS[0]
    card_md = v_digest.briefing_card(digest)
    titles = [p["title"] for p in digest["pages"]]
    a_conversation._conversation.append({"role": "user", "content": user_text})
    a_conversation._conversation.append({"role": "assistant", "content": reply})
    a_conversation._session_turns.append({"role": "assistant", "content":
                           f"{fold.STAND_INS[1]}{len(digest['pages']) + len(digest['chat_pages'])} pages, "
                           f"{len(digest['sources'])} sources, {v_digest.memory_total(digest)} memory changes"})
    a_conversation._last_context = {"question": user_text, "extra": block, "tools": [], "reply": reply, "sources": [],
                     "pages": titles, "card": card_md}
    return reply, card_md


def card_in(content: str) -> dict | None:
    """The card an assistant turn holds (`<card id>` marker), or None."""
    m = _CARD_MARK.match(content or "")
    return a_conversation._cards.get(m.group(1)) if m else None
