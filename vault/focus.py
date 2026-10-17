"""Focus context: the open page's head plus the best-matching section or chunk (docs/38-focus-context.md; PRD 50 step 7, from wiki.py)."""
import os
import re
from core import trace
from vault import search as v_search
from vault import store
from vault import chunks as v_chunks

log = trace.logger("Wiki")


# --- focus context (docs/38-focus-context.md) ----------------------------------------
# The open page rides on the turn as its head plus the one `##` section (or, for a source,
# the one raw chunk) that best matches the question — BM25 with the same tokeniser as search.

_H2 = re.compile(r"^## .*$", re.M)


def page_title(path: str) -> str:
    """The `name:` of a page (its file stem when it has none); "" for an unknown path."""
    text = v_chunks.page_text(path)
    if text is None:
        return ""
    fm, _ = store.parse_page(text)
    return str(fm.get("name") or os.path.basename(store.normalise_path(path, user=True) or path))


def split_sections(body: str) -> tuple:
    """``(head, [section, ...])``: the text before the first `## ` heading and each `##`
    section with its heading line (the page's frontmatter is not part of `body`)."""
    starts = [m.start() for m in _H2.finditer(body)]
    if not starts:
        return body.strip(), []
    head = body[:starts[0]].strip()
    ends = starts[1:] + [len(body)]
    return head, [body[a:b].strip() for a, b in zip(starts, ends)]


def _best_match(texts: list, question: str) -> int | None:
    """Index of the text that scores best for the question's content words (BM25, earliest on
    a tie); None when there is nothing to score or no word occurs in any of them."""
    terms = v_chunks.content_terms(question)
    if not texts or not terms:
        return None
    scores = v_search._bm25([v_search._tokenise(t) for t in texts], terms)
    best = max(range(len(texts)), key=lambda i: (scores[i], -i))
    return best if scores[best] > 0 else None


def best_section(sections: list, question: str) -> str | None:
    """The section that scores best for `question`; None with fewer than two sections or
    when no content word of the question occurs in any of them."""
    at = _best_match(sections, question) if len(sections) > 1 else None
    return None if at is None else sections[at]


def best_chunk(path: str, question: str) -> dict | None:
    """The raw chunk of source page `path` that scores best for `question` as
    ``{title, n, total, text}``; None for a page without raw text or when nothing matches."""
    try:
        record = v_chunks.raw_chunks(path)
    except KeyError:
        return None
    chunks = record["chunks"]
    at = _best_match([c["text"] for c in chunks], question)
    if at is None:
        return None
    return {"title": record["title"], "n": chunks[at]["n"], "total": len(chunks), "text": chunks[at]["text"]}


_LINK = re.compile(r"\[\[([^\]|#]+)")


def links_in(text: str) -> list:
    """The wikilink targets in `text`, in order, de-duplicated."""
    seen = []
    for raw in _LINK.findall(text or ""):
        target = raw.strip()
        if target and target not in seen:
            seen.append(target)
    return seen


#: the recall tools tools/wiki.py registers — the prompt may only advertise them when they
#: are actually offered to the model this run (see :func:`tools_registered`).
RECALL_TOOLS = ("search_wiki", "read_page", "read_source")


def tools_registered() -> bool:
    """True when `search_wiki`/`read_page` are on the tool list the model is being given.

    `llm.TOOLS` is the live list: the eval harness empties it to run tool-free, and a prompt
    that tells a tool-less model to "use search_wiki" only teaches it to fake a tool call
    (docs/19-wiki-layer.md, Results).
    """
    try:
        import llm

        names = {(t.get("function") or {}).get("name") for t in llm.TOOLS if isinstance(t, dict)}
    except Exception:  # noqa: BLE001 — no llm (tests, scripts) means no tools
        return False
    return any(name in names for name in RECALL_TOOLS)
