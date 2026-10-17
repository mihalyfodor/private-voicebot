"""Per-turn recall: source / page recall blocks, the band on spoken turns, the precision rule, fallback and focus context (docs/32, 38, 42, 45; PRD 50 step 9, from llm.py)."""
import re
from memory import render
from vault import chunks as v_chunks
from vault import focus as v_focus
from vault import search as v_search
from vault import store as v_store
from core.untrusted import untrusted
from agent import conversation
from agent import written
from agent import prompt



RECALL_MIN_WORDS = 4
RECALL_MAX_CHUNKS = 2
RECALL_MAX_TOKENS = 1500
#: a question whose content words are all of these is a tool command ("what time is it"),
#: never a question for a filed document
COMMAND_WORDS = frozenset("""time clock date day today weather temperature forecast rain outside
news headlines headline email emails mail inbox messages message""".split())
#: the two phrases of a recall note a typed content question rewrites (`written_note`, PRD 43)
SHORT_PHRASE, FULL_PHRASE = "in one to three sentences", "fully, as long as it needs"
NOTE_OFFER, NOTE_NO_OFFER = "finish with exactly this sentence: " + prompt.DEEPER_OFFERS[0], "do not offer to dig deeper."
RECALL_NOTE = ("If this answers the question, answer from it " + SHORT_PHRASE + ", name {titles} "
               "once in a short natural clause inside a sentence (never as a separate closing sentence), "
               "and " + NOTE_OFFER + " Otherwise ignore it "
               "completely: answer as you normally would and never mention the document or what it lacks.")
# --- ask before lookup (docs/45-ask-before-lookup.md): the recall band on spoken turns
RECALL_HIGH_SCORE = v_chunks.RECALL_HIGH_SCORE
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
    text = v_chunks.page_text(path)
    if text is None:
        return ""
    title = v_focus.page_title(path)
    conversation._turn_pages.append(title)
    head, sections = v_focus.split_sections(v_store.parse_page(text)[1])
    # The schema keeps a page's gist in `## Summary`; that is the head, not a candidate section.
    summary = next((s for s in sections if s.startswith("## Summary")), None)
    if summary:
        head, sections = head + "\n\n" + summary, [s for s in sections if s is not summary]
    parts = fit([head], FOCUS_HEAD_TOKENS, "")
    # "summarise this" names no topic: the head + Summary is the answer, not a random chunk (PRD 42)
    topical = bool(v_chunks.topic_terms(question))
    chunk = v_focus.best_chunk(path, question) if view == "sources" and topical else None
    if chunk:
        conversation._turn_sources.append((path, chunk["title"], chunk["n"]))
        section = f'Part {chunk["n"]}/{chunk["total"]}: {chunk["text"]}'
    else:
        section = v_focus.best_section(sections, question) if topical else None
    if section:
        parts += fit([section], FOCUS_SECTION_TOKENS, "")
    note = FOCUS_NOTE.format(title=title)
    kept = fit(["\n\n".join(p for p in parts if p)], FOCUS_MAX_TOKENS, note)
    if not kept or not kept[0].strip():
        return "\n\n" + note
    return "\n\n" + untrusted(kept[0], title) + "\n" + note


def fit(lines: list, budget: int, note: str) -> list:
    """`lines` cut to fit `budget` tokens together with `note` and the tags: whole lines while
    they fit, the first one that does not is truncated, the rest dropped."""
    used = render.tokens_est(note) + 40
    kept = []
    for line in lines:
        room = (budget - used) * 4
        if room <= 0:
            break
        kept.append(line[:room])
        used += render.tokens_est(kept[-1])
    return kept


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
    terms = v_chunks.content_terms(user_text)
    if terms <= COMMAND_WORDS:
        return None, []
    proper = v_chunks.proper_terms(terms)
    words = len(user_text.split())
    if words < RECALL_MIN_WORDS and not (words >= RECALL_MIN_PROPER_WORDS and proper):
        return None, []

    page = _page_recall(user_text, terms)
    conversation._evidence = (page, _recall_hits(user_text, proper, v_chunks.recall_min_score(), RECALL_CANDIDATES))
    return conversation._evidence



def recall_band(user_text: str, page: dict | None, hits: list) -> str:
    """"high" (answer grounded), "medium" (answer plainly, offer to check) or "low" (no recall)
    from the evidence recall found: score per topic word plus the kind of match (PRD 45)."""
    top = hits[0] if hits else None
    proper = v_chunks.proper_terms(v_chunks.content_terms(user_text))
    matched = v_chunks.proper_match(top, proper, v_chunks.proper_phrases(user_text)) if top else set()
    content = written.content_question(user_text)
    # low: nothing kept, or the gate fails (chat carried by plain words)
    if not (page or hits) or not (content or matched):
        return "low"
    # high: a named page or source, then the rules that need the top hit
    if page or top.get("named"):
        return "high"
    strong_question = content and (matched or top["score"] >= RECALL_HIGH_SCORE)
    name_question = bool(written.ASK_WORDS.search(user_text)) and bool(matched)
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
    pairs = v_chunks.proper_phrases(question)
    kept = []
    for hit in v_chunks.search_chunks(question, k=k + len(exclude)):
        if len(kept) >= RECALL_MAX_CHUNKS or (hit["score"] < floor and not hit.get("named")):
            break                                               # a named source has no threshold
        if (hit["source"], hit["n"]) in exclude:
            continue
        if any(h["source"] == hit["source"] and abs(h["n"] - hit["n"]) == 1 for h in kept):
            continue                                            # overlaps a kept chunk
        if len(hit["matched"]) >= 2 or v_chunks.proper_match(hit, proper, pairs):
            kept.append(hit)                                    # one common word is no match
    return kept


def _page_recall(question: str, terms: set) -> dict | None:
    """The top wiki page for `question` as ``{line, path, title}`` when the question names it
    (every token of the page's name among the terms, no tool-command word), else None; the line
    is the head + Summary + best-matching section within RECALL_PAGE_TOKENS (PRD 42 §3)."""
    if terms & COMMAND_WORDS:
        return None
    hits = [h for h in v_search.search(question, 2) if not h["path"].startswith("sources/")]
    if not hits:
        return None
    path = hits[0]["path"]
    text = v_chunks.page_text(path)
    if text is None:
        return None
    title = v_focus.page_title(path)
    name_terms = v_chunks.content_terms(title) or v_chunks.content_terms(path.rsplit("/", 1)[-1])
    if not name_terms or not name_terms <= terms:
        return None
    head, sections = v_focus.split_sections(v_store.parse_page(text)[1])
    summary = next((s for s in sections if s.startswith("## Summary")), None)
    section = v_focus.best_section([s for s in sections if s is not summary], question)
    body = "\n\n".join(p for p in (head, summary, section) if p)
    kept = fit([f'From page "{title}" ({path}): {body}'], RECALL_PAGE_TOKENS, "")
    if kept:
        conversation._turn_pages.append(title)
    return {"line": kept[0], "path": path, "title": title} if kept else None


def _quote_block(tag: str, page: dict | None, hits: list, budget: int, note_text: str) -> str:
    """`<tag>` block of the page line and the chunk quotes within `budget`, or ""; the chunks
    quoted are recorded in `_turn_sources` for the deeper offer."""
    lines = ([page["line"]] if page else []) + [
        f'From "{h["title"]}" (part {h["n"]}/{h["total"]}): {h["text"]}' for h in hits]
    titles = ([page["title"]] if page else []) + [h["title"] for h in hits]
    quotes = fit(lines, budget, note_text)
    if not quotes:
        return ""
    titles = titles[:len(quotes)]
    hits = hits[:len(quotes) - (1 if page else 0)]
    conversation._turn_sources.extend((h["source"], h["title"], h["n"]) for h in hits)
    note = note_text.format(titles=" or ".join(f'"{t}"' for t in dict.fromkeys(titles)))
    blocks = [untrusted(q, t) for q, t in zip(quotes, titles)]
    return f"\n\n<{tag}>\n" + "\n\n".join(blocks) + f"\n{note}\n</{tag}>"


def lookup_worthy(question: str) -> bool:
    """A question the filed documents could answer: at least RECALL_MIN_PROPER_WORDS words with a
    content word that is not a tool command ("hi" and "time?" stream untouched)."""
    terms = v_chunks.content_terms(question)
    return len(question.split()) >= RECALL_MIN_PROPER_WORDS and bool(terms) and not terms <= COMMAND_WORDS


def fallback_block(question: str, reply_start: str) -> str:
    """The `<lookup>` block for a re-ask when `reply_start` claims the documents do not cover
    the question: the search_wiki equivalent run server-side (top page + chunks below the
    threshold, minus the parts this turn already quoted), or "" (PRD 42 §2)."""
    if not NOT_COVERED.search(reply_start):
        return ""
    if not lookup_worthy(question):
        return ""
    terms = v_chunks.content_terms(question)
    used = {(s, n) for s, _, n in conversation._turn_sources}
    page = _page_recall(question, terms)
    hits = _recall_hits(question, v_chunks.proper_terms(terms), 0.0, RECALL_CANDIDATES, used)
    return _quote_block("lookup", page, hits, FALLBACK_MAX_TOKENS, FALLBACK_NOTE)
