"""The one-turn offers: dig deeper, check Wikipedia, check your page; affirmative detection, offer stripping and the deeper block (docs/34, 37; PRD 50 step 9, from llm.py)."""
import re
from vault import chunks as v_chunks
from core.untrusted import untrusted
from agent import conversation
from agent import prompt
from agent import recall as a_recall



def _offer_kind(recall_offer: bool) -> str:
    """What the turn ended offering: none / deeper / wikipedia / writeup, or recall (the medium band's check)."""
    if recall_offer:
        return "recall"
    if not conversation._deeper_offer:
        return "none"
    if conversation._deeper_offer.get("wikipedia"):
        return "wikipedia"
    return "writeup" if (conversation._conversation[-1].get("content") or "").rstrip().endswith(prompt.DEEPER_OFFERS[0]) else "deeper"


def drop_offer() -> None:
    """Forget the pending deeper / write-up offer: the turn answered something else (PRD 52)."""

    conversation._deeper_offer = None

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


def is_affirmative(text: str) -> bool:
    """"yes", "yes please", "go on", "dig deeper", "sure, do it" — ≤ 6 words, nothing else in it."""
    words = re.findall(r"[a-z]+", text.lower())
    return (0 < len(words) <= AFFIRMATIVE_MAX_WORDS and words[0] in _AFFIRMATIVE_LEADS
            and all(w in _AFFIRMATIVE_WORDS for w in words))


def ends_with_offer(reply: str, phrases: tuple = prompt.DEEPER_OFFERS) -> bool:
    tail = reply.strip().rstrip("\"'”’ ").lower()
    return any(tail.endswith(o.lower()) for o in phrases)


def ends_with_wikipedia_offer(reply: str) -> bool:
    return ends_with_offer(reply, (prompt.WIKIPEDIA_OFFER,))


def strip_offer(reply: str) -> str:
    """`reply` without a trailing DEEPER_OFFERS / WIKIPEDIA_OFFER sentence (the medium turn's own
    offer replaces it, PRD 45)."""
    for phrase in prompt.DEEPER_OFFERS + (prompt.WIKIPEDIA_OFFER,):
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
        conversation._turn_sources.append((str(args.get("path") or ""), title, part))
    elif name == "search_wiki":
        for m in _SNIPPET_PART.finditer(result):
            conversation._turn_sources.append((m.group(3), m.group(1), int(m.group(2))))
    elif name == "get_wikipedia" and args.get("query"):
        conversation._turn_pages.append(f"Wikipedia: {args['query']}")


def _offer_from_turn(reply: str, question: str = "") -> dict | None:
    """The `_deeper_offer` this reply leaves behind: the question when the reply ended with the
    Wikipedia offer, else the first source used, all its parts."""
    if ends_with_wikipedia_offer(reply):
        return {"wikipedia": question}
    if not conversation._turn_sources or not ends_with_offer(reply):
        return None
    source, title, _ = conversation._turn_sources[0]
    parts = sorted({p for s, _, p in conversation._turn_sources if s == source})
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
        record = v_chunks.raw_chunks(offer["source"])
    except KeyError:
        return ""
    chunks, title = record["chunks"], record["title"]
    note = DEEPER_NOTE.format(title=f'"{title}"')
    lines = [f"Part {n}/{len(chunks)}: {chunks[n - 1]['text']}"
             for n in deeper_parts(offer["parts"], len(chunks))]
    quotes = a_recall.fit(lines, DEEPER_MAX_TOKENS, note)
    if not quotes:
        return ""
    return (f'\n\n<source title="{title}">\n' + "\n\n".join(untrusted(q, title) for q in quotes)
            + f"\n{note}\n</source>")
