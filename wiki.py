"""Wiki layer — an Obsidian-compatible markdown vault the assistant maintains.

The vault (``WIKI_PATH``, default ``~/llm-wiki-dev``; prod sets its own) is the *durable* half of long-term memory:
the profile in memory.py is "who the user is" (small, always in the prompt, decays), the vault
is "what we are working on" (unbounded, on demand, human co-edited in Obsidian).

Only ``index.md`` is loaded into the prompt; pages are reached through two tools
(``search_wiki``, ``read_page``), so recall costs nothing until it is needed.

The model never writes a file. At memory-save time :func:`ingest` asks it for a JSON array of
**page ops**; :func:`apply_page_ops` validates and applies them, regenerates ``index.md`` from
the pages' frontmatter, appends to ``log.md`` and makes one ``git commit``. Same lesson as
PRD 15: ops + validation, never a whole-file rewrite. ``git log`` inside the vault is the undo.

See docs/19-wiki-layer.md.
"""
import json
import os
import re
from collections import Counter
from vault import index_db
import memory
from core import settings, trace, vault_lock
from core import llm as core_llm
from core.untrusted import DOCUMENT_RULE
from vault import index as v_index
from vault import pages as v_pages
from vault import search as v_search
from vault import store

log = trace.logger("Wiki")


# --- search inside sources (docs/32-source-recall.md) -------------------------------
# Text-kind raw files are split into ~800-token chunks, cached per sha in raw/.index/, and
# searched with the same BM25 as the pages. Pages and chunks are ranked separately.

CHUNK_TOKENS = 800
CHUNK_OVERLAP = 80
SNIPPET_CHARS = 600
MAX_SOURCE_READ = 3000
#: `recall_min_score` default — BM25 per content word, calibrated in tests/test_source_recall.py
RECALL_MIN_SCORE = 4.5
#: a content question matched on plain words only answers grounded from this score (the recall band,
#: docs/45-ask-before-lookup.md); below it a spoken turn asks first
RECALL_HIGH_SCORE = 8.0
#: query words that carry no topic; dropped before scoring so "the" cannot dilute a match
STOPWORDS = frozenset("""a an the and or but if so of to in on at by for from with about into over
after before as is are am was were be been being do does did have has had i me my mine we us our you
your he she it its they them their this that these those what which who whom how when where why
there here can could should would will shall may might must not no yes im ive dont cant lets s m t
re ll d ve keep keeps get got just really very some any all one doe up off per accord exactly thi thes""".split())   # thi/thes: this/these stemmed
#: a query word the sources write Title-case ("Covenant", "Whiterun") names a place, a person
#: or a quest; it counts up to this much more than a plain word ("cook", "weather"), which is
#: what separates "I'm at Covenant" from "what should we cook" when each matches one rare word
PROPER_BOOST = 2.0
#: a query word written Title-case at least this often in the sources is a proper noun for the
#: recall precision rule (PRD 34): one such match may carry a chunk, one plain word may not
PROPER_RATIO_MIN = 0.5
_WORD = re.compile(r"[A-Za-z0-9]+")


def _read_bytes(rel: str) -> bytes:
    with open(store._path(rel), "rb") as f:
        return f.read()


_PARA = re.compile(r"(?:[^\n]*\S[^\n]*(?:\n|\Z))+")


def _blocks(text: str) -> list:
    """``[(start, block)]`` — runs of non-blank lines (paragraphs, headings), offsets into `text`."""
    return [(m.start(), m.group()) for m in _PARA.finditer(text)]


def _split_long(start: int, block: str, limit: int) -> list:
    """A block over `limit` chars is cut at line boundaries (or hard, for one endless line)."""
    parts, buf, buf_at, at = [], "", start, start
    for line in block.splitlines(keepends=True):
        if buf and len(buf) + len(line) > limit:
            parts.append((buf_at, buf))
            buf = ""
        while len(line) > limit:
            parts.append((at, line[:limit]))
            line, at = line[limit:], at + limit
        if not buf:
            buf_at = at
        buf += line
        at += len(line)
    if buf:
        parts.append((buf_at, buf))
    return parts


def _carry_from(buf_at: int, buf: str, tail: int) -> int:
    """Where the overlap carried into the next chunk starts: the last `tail` chars of the
    chunk, moved forward to a line start (or a word start when the tail is one long line)."""
    body = buf.rstrip()
    if not tail or not body:
        return buf_at + len(buf)
    at = max(0, len(body) - tail)
    cut = body.find("\n", at)
    if 0 <= cut < len(body) - 1:
        return buf_at + cut + 1
    space = body.find(" ", at)
    return buf_at + (space + 1 if 0 <= space < len(body) - 1 else at)


def chunk_text(text: str, tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list:
    """``[{n, start, text}]`` — ~`tokens`-token chunks on paragraph boundaries with `overlap`
    tokens of tail carried into the next chunk. Deterministic: same text, same chunks."""
    limit, tail = tokens * 4, overlap * 4
    pieces = [p for start, block in _blocks(text) for p in _split_long(start, block, limit)]
    chunks, buf, buf_at = [], "", 0
    for start, piece in pieces:
        if buf and start + len(piece) - buf_at > limit:
            chunks.append({"n": len(chunks) + 1, "start": buf_at, "text": buf.rstrip()})
            buf_at = _carry_from(buf_at, buf, tail)
            buf = text[buf_at:start]
        if not buf.strip():
            buf_at = start
        buf = text[buf_at:start + len(piece)]      # a slice of `text`, so offsets stay honest
    if buf.strip():
        chunks.append({"n": len(chunks) + 1, "start": buf_at, "text": buf.rstrip()})
    return chunks


def _chunkable(entry: dict) -> bool:
    ext = os.path.splitext(str(entry.get("path") or ""))[1].lower()
    return bool(ext) and ext not in (".png", ".jpg", ".jpeg", ".ics")


def _source_title(source: str, raw_rel: str) -> str:
    fm = store.parse_page(store._read(f"{source}.md"))[0] if source else {}
    return str(fm.get("name") or os.path.splitext(os.path.basename(raw_rel))[0])


def _cache_rel(sha: str) -> str:
    return f"{store.RAW_INDEX_DIR}/{sha}.json"


def raw_chunks(raw_rel: str) -> dict:
    """The chunk record of one raw file: ``{sha, raw, source, title, chunks}``.

    Cached in `raw/.index/<sha>.json`; a cache hit reads no raw bytes. `raw_rel` may be the
    raw path or the source page path; a file that is not in the manifest raises KeyError.
    """
    from vault import manifest

    entry = next((e for e in manifest.manifest_entries()
                  if raw_rel in (e.get("path"), e.get("page"))), None)
    if entry is None:
        raise KeyError(f"no such source: {raw_rel}")
    return _raw_chunks_for(entry)


def _raw_chunks_for(entry: dict) -> dict:
    """:func:`raw_chunks` for a manifest entry already in hand (a rebuild parses the manifest once)."""
    from ingest import pipeline

    sha, raw_rel, source = entry["sha256"], entry["path"], entry.get("page", "")
    cached = store._read(_cache_rel(sha))
    if cached:
        try:
            record = json.loads(cached)
            if record.get("sha") == sha and record.get("raw") == raw_rel:
                return record
        except ValueError:
            pass
    text = pipeline.raw_text(raw_rel, _read_bytes(raw_rel)) if _chunkable(entry) else ""
    record = {"sha": sha, "raw": raw_rel, "source": source,
              "title": _source_title(source, raw_rel),
              "chunks": chunk_text(text, CHUNK_TOKENS, CHUNK_OVERLAP)}
    store._write(_cache_rel(sha), json.dumps(record, ensure_ascii=False))
    return record


def drop_raw_chunks(sha: str) -> None:
    """Remove a forgotten source's cache file (nothing to do when it was never built)."""
    full = store._path(_cache_rel(sha))
    if os.path.exists(full):
        os.remove(full)


def _record(sha: str) -> dict | None:
    """A source's chunk record from its `raw/.index/<sha>.json`, or None when there is none.
    The last few are kept parsed: a hit reads its chunk text from here."""
    key = (store.WIKI_PATH, sha)
    if key in _records:
        return _records[key]
    try:
        record = json.loads(store._read(_cache_rel(sha)) or "null")
    except ValueError:
        record = None
    if record is not None and record.get("sha") != sha:
        record = None
    if record is not None:
        _records[key] = record
        while len(_records) > 16:
            _records.pop(next(iter(_records)))
    return record


_records: dict = {}


def index_source(record: dict) -> None:
    """Write hook: index one filed source's chunks (`record` is :func:`raw_chunks`' result)."""
    if (idx := v_search._hook_db()) is not None:
        _put_source(idx, record)


def _put_source(idx: index_db.Index, record: dict) -> None:
    """Write one source's chunk rows (or drop it when it has no chunks)."""
    seen, titled = _proper_counts(c["text"] for c in record["chunks"])
    chunks = [(c["n"], c["start"], v_search._tokenise(c["text"])) for c in record["chunks"]]
    if chunks:
        idx.put_source(record["sha"], record["raw"], record["source"], record["title"], chunks, seen, titled)
    else:
        idx.drop_source(record["sha"], seen, titled)


def unindex_source(sha: str) -> None:
    """Write hook: a forgotten source's chunks leave the index. Call it before
    :func:`drop_raw_chunks` — its proper-noun counts come from the chunk cache."""
    if (idx := v_search._hook_db()) is not None:
        _drop_source(idx, sha)


def _drop_source(idx: index_db.Index, sha: str) -> None:
    """Remove one source's chunk rows, keeping the proper-noun counts right."""
    record = _record(sha)
    if record is not None:
        idx.drop_source(sha, *_proper_counts(c["text"] for c in record["chunks"]))
    elif idx.drop_source(sha, None, None):           # no cache left to subtract: recount the rest
        idx.replace_proper(*_proper_counts(c["text"] for other in idx.source_shas()
                                           for c in (_record(other) or {"chunks": []})["chunks"]))


def _sync_sources(idx: index_db.Index) -> int:
    """Bring the chunk index in line with the manifest; returns the sources changed. Skipped when the
    manifest bytes hash to what the last sync saw."""
    from vault import manifest

    digest = v_search._sha(store._read(manifest.MANIFEST))
    if idx.get_meta("manifest") == digest:
        return 0
    entries = {e["sha256"]: e for e in manifest.manifest_entries() if e.get("sha256") and _chunkable(e)}
    have = idx.source_shas()
    changed, skipped = 0, False
    for sha, raw in have.items():
        if sha not in entries or entries[sha]["path"] != raw:
            _drop_source(idx, sha)
            changed += 1
    for sha, entry in entries.items():
        if have.get(sha) == entry["path"]:
            continue
        try:
            record = _raw_chunks_for(entry)
        except Exception as exc:    # noqa: BLE001 — one unreadable source must not take the index down
            log.exception(f"search index: skipped {entry.get('path')} (cannot be chunked)")
            trace.error("index", "sync_sources", exc, fatal_for="none")
            skipped = True
            continue
        if record["chunks"]:
            _put_source(idx, record)
            changed += 1
    if not skipped:                 # a skipped source keeps its chance: the next sync retries it
        idx.set_meta("manifest", digest)
    return changed


def _chunk_index() -> index_db.Index | None:
    """The index, with its sources in line with the manifest: one stat per call, a re-read and a
    sync only when the manifest file changed (a batch import in another process, a hand edit)."""
    from vault import manifest

    if not store.exists():
        return None
    idx = v_search._db()
    try:
        st = os.stat(store._path(manifest.MANIFEST))
        mark = (st.st_mtime_ns, st.st_size)
    except OSError:
        mark = None
    if getattr(idx, "manifest_mark", None) != mark:
        _sync_sources(idx)
        idx.manifest_mark = mark
    return idx


def _proper_noun(text: str, m) -> bool:
    """Title-case and mid-sentence ("of Covenant", not "Today you") — a name, not a sentence start."""
    word = m.group()
    if not (word[:1].isupper() and word[1:].islower()):
        return False
    prev = text[:m.start()].rstrip()[-1:]
    return prev.isalnum() or prev in ",;:"


def _proper_counts(texts) -> tuple:
    """``(seen, titled)``: stem -> occurrences, and how many of them are proper nouns, over `texts`.
    The index sums them per source; a stem's ratio is ``titled / seen`` over every chunk, and every
    stem the sources contain has a row, so the table doubles as the vocabulary (:func:`proper_phrases`)."""
    seen, titled = Counter(), Counter()
    for text in texts:
        for m in _WORD.finditer(text):
            stem = v_search._stem(m.group().lower())
            seen[stem] += 1
            if _proper_noun(text, m):
                titled[stem] += 1
    return dict(seen), dict(titled)


def _proper_ratios(terms) -> dict:
    """stem -> share of its occurrences that are proper nouns, for the `terms` the sources hold."""
    idx = _chunk_index()
    return idx.proper_ratios(terms) if idx is not None else {}


def _term_weights(terms: set) -> dict:
    """1 + PROPER_BOOST × (share of the word's occurrences written Title-case)."""
    proper = _proper_ratios(terms)
    return {t: 1.0 + PROPER_BOOST * proper.get(t, 0.0) for t in terms}


def content_terms(query: str) -> set:
    """The query's tokens minus stopwords — what a match is scored and normalised on."""
    return {t for t in v_search._tokenise(query) if t not in STOPWORDS}


def proper_terms(terms: set) -> set:
    """The query terms the sources write Title-case at least `PROPER_RATIO_MIN` of the time."""
    proper = _proper_ratios(terms)
    return {t for t in terms if proper.get(t, 0.0) >= PROPER_RATIO_MIN}


#: words that make a question but name no topic ("summarise this", "what does it say"): a
#: question with nothing else in it gets a page's head, not a chunk (docs/42-recall-precision-2.md)
GENERIC_WORDS = frozenset("""summarise summarize summary sum recap explain describe overview gist
tl dr tell page document source say said says talk about mean thing things content contents""".split())
#: title words that name no source on their own ("the guide", "this book")
GENERIC_TITLE_WORDS = frozenset("""guide guides book books note notes walkthrough faq manual
edition volume part game year new complete official""".split())


def topic_terms(query: str) -> set:
    """Content terms minus GENERIC_WORDS: what the question is actually about, or nothing."""
    return content_terms(query) - GENERIC_WORDS


def _title_terms() -> dict:
    """source page -> the content tokens of its title, over the indexed chunks."""
    idx = _chunk_index()
    if idx is None:
        return {}
    return idx.cached("title_terms", lambda: {source: content_terms(title)
                                              for source, _, title, *_ in idx.chunk_meta().values()})


def split_title_terms(terms: set) -> tuple:
    """``(topic, named)`` for the query's content terms: the terms left after dropping every
    filed source's title tokens (all of them when nothing would be left), and the sources the
    question names — two of a title's tokens, or one that no other title has and is not a
    GENERIC_TITLE_WORDS one (docs/42-recall-precision-2.md §1)."""
    titles = _title_terms()
    title_tokens = set().union(*titles.values()) if titles else set()
    topic = terms - title_tokens or terms
    named = set()
    for source, tokens in titles.items():
        matched = terms & tokens
        unique = {t for t in matched if t not in GENERIC_TITLE_WORDS
                  and not any(t in other for s, other in titles.items() if s != source)}
        if len(matched) >= 2 or unique:
            named.add(source)
    return topic, named


def proper_phrases(query: str) -> list:
    """The proper-noun bigrams in `query` as ``[(a, b)]`` — consecutive tokens where one is a
    proper term and the other is a proper term, a word the sources never use ("Lovelace"),
    or both are written Title-case ("Ada Lovelace"). A chunk must contain the pair whole for
    the proper term to count (docs/42-recall-precision-2.md §5)."""
    words = re.findall(r"[A-Za-z0-9]+", str(query))
    vocab = _proper_ratios({v_search._stem(w.lower()) for w in words})
    pairs = []
    for first, second in zip(words, words[1:]):
        a, b = v_search._stem(first.lower()), v_search._stem(second.lower())
        if a in STOPWORDS or b in STOPWORDS:
            continue
        proper = {t for t in (a, b) if vocab.get(t, 0.0) >= PROPER_RATIO_MIN}
        if not proper:
            continue
        titled = first[:1].isupper() and second[:1].isupper()
        unseen = a not in vocab or b not in vocab
        if len(proper) == 2 or unseen or titled:
            pairs.append((a, b))
    return pairs


def _chunk_subset(meta: dict, named: set, filters: dict | None):
    """The chunk ids scored: the named sources' (all when none is named or they hold none),
    narrowed by `filters` ``{source, kind}``; None means every chunk."""
    from vault import manifest

    subset = {i for i, m in meta.items() if m[0] in named} or None if named else None
    if not filters:
        return subset
    keep = set(meta) if subset is None else subset
    want = filters.get("source")
    if want:
        want = {want} if isinstance(want, str) else set(want)
        keep = {i for i in keep if meta[i][0] in want or meta[i][1] in want}
    kind = filters.get("kind")
    if kind:
        kinds = {kind} if isinstance(kind, str) else set(kind)
        shas = {e.get("sha256") for e in manifest.manifest_entries() if e.get("kind") in kinds}
        keep = {i for i in keep if meta[i][5] in shas}
    return keep


def _phrases_at(idx: index_db.Index, doc: int, pairs: list) -> list:
    """The `pairs` that occur adjacent in chunk `doc`, from the stored token positions."""
    found = []
    for a, b in pairs:
        after = {p - 1 for p in idx.positions(index_db.CHUNK, b, doc)}
        if after and after.intersection(idx.positions(index_db.CHUNK, a, doc)):
            found.append((a, b))
    return found


def search_chunks(query: str, k: int = 3, filters: dict | None = None) -> list:
    """Top-`k` raw chunks for `query`: ``[{source, raw, title, n, total, text, score, matched, phrases}]``.

    The query's title words never score a chunk (:func:`split_title_terms`): they only pick the
    sources searched. `score` is BM25 divided by the number of topic words, so a threshold means
    the same for a three-word and a twelve-word question (see :func:`recall_min_score`); `matched`
    lists the distinct topic words found in the chunk (the PRD 34 precision rule), `phrases`
    the query's proper bigrams the chunk holds whole, `named` whether the question named the
    chunk's source by title. `filters` (``{source, kind}``) narrows the chunks scored, with the
    narrowed set's BM25 statistics, as naming a source does.
    """
    idx = _chunk_index()
    terms = content_terms(query)
    if idx is None or not terms:
        return []
    meta = idx.chunk_meta()
    if not meta:
        return []
    terms, named = split_title_terms(terms)
    terms = terms - GENERIC_WORDS or terms                  # "what does it say about X": X scores
    subset = _chunk_subset(meta, named, filters)
    pairs = proper_phrases(query)
    scores = idx.bm25(index_db.CHUNK, terms, _term_weights(terms), subset)
    ranked = sorted(((round(score / len(terms), 3), doc, matched) for doc, (score, matched) in scores.items()
                     if score > 0), key=lambda r: (-r[0], meta[r[1]][1], meta[r[1]][3]))[:k]
    hits = []
    for score, doc, matched in ranked:
        source, raw, title, n, total, sha = meta[doc]
        record = _record(sha) or _raw_chunks_for({"sha256": sha, "path": raw, "page": source})
        hits.append({"source": source, "raw": raw, "title": title, "n": n, "total": total,
                     "text": record["chunks"][n - 1]["text"], "score": score, "matched": sorted(matched),
                     "phrases": _phrases_at(idx, doc, pairs), "named": source in named})
    return hits


def proper_match(hit: dict, proper: set, pairs: list) -> set:
    """The hit's matched proper terms that may carry it alone: a term inside a proper bigram the
    chunk does not hold whole is demoted to a common word (docs/42-recall-precision-2.md §5)."""
    partial = {t for a, b in pairs if (a, b) not in hit.get("phrases", []) for t in (a, b)}
    return (proper & set(hit["matched"])) - partial


def read_source(path: str, part) -> str:
    """Chunk `part` of a source with a neighbour either side, ≤ `MAX_SOURCE_READ` chars.

    `path` is the source page (`sources/guide`) or the raw path; a bad path or part number
    comes back as a plain error string the model can read.
    """
    rel = store.normalise_path(path, user=True)
    try:
        record = raw_chunks(store.wikilink(rel) if rel else str(path or ""))
    except KeyError:
        return f"no such source: {path}"
    chunks, total = record["chunks"], len(record["chunks"])
    try:
        n = int(part)
    except (TypeError, ValueError):
        n = 0
    if not 1 <= n <= total:
        return f"no part {part} in {path} (parts 1-{total})"
    middle = chunks[n - 1]["text"][:MAX_SOURCE_READ]
    room = (MAX_SOURCE_READ - len(middle)) // 2
    before = chunks[n - 2]["text"][-room:] if n > 1 and room else ""
    after = chunks[n]["text"][:room] if n < total and room else ""
    text = "\n".join(p for p in (before, middle, after) if p)[:MAX_SOURCE_READ]
    return f'{record["title"]} · part {n}/{total}\n{text}'


def recall_min_score() -> float:
    """`recall_min_score` from settings.json (default `RECALL_MIN_SCORE`)."""

    value = settings.load_settings().get("recall_min_score")
    return float(value) if isinstance(value, (int, float)) and value >= 0 else RECALL_MIN_SCORE


def read_page(path: str) -> str:
    """A page's text, capped at 4 kB with a marker. Unknown path → "no such page"."""
    rel = store.normalise_path(path)
    if rel is None or not store.page_exists(rel):
        return f"no such page: {path}"
    text = store._read(rel)
    if len(text.encode()) > store.MAX_READ_BYTES:
        return text.encode()[:store.MAX_READ_BYTES].decode(errors="ignore") + store.TRUNCATED
    return text


def page_text(path: str) -> str | None:
    """The whole page for the editor (no 4 kB cap); None for an unknown path."""
    rel = store.normalise_path(path, user=True)
    if rel is None or not store.page_exists(rel):
        return None
    return store._read(rel)


# --- focus context (docs/38-focus-context.md) ----------------------------------------
# The open page rides on the turn as its head plus the one `##` section (or, for a source,
# the one raw chunk) that best matches the question — BM25 with the same tokeniser as search.

_H2 = re.compile(r"^## .*$", re.M)


def page_title(path: str) -> str:
    """The `name:` of a page (its file stem when it has none); "" for an unknown path."""
    text = page_text(path)
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
    terms = content_terms(question)
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
        record = raw_chunks(path)
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


# --- briefing (docs/46-briefing.md) -------------------------------------------------
# "what's new?": what the vault, the sources and memory gained since a cutoff, from git
# subjects (every writer makes one commit per write), the manifest and the memory ops log.

_NOISE_COMMITS = ("wiki: lint", "wiki: initial vault", "wiki: vault layout")
_USER_COMMIT = re.compile(r"^wiki: (?:edit|create|save card|rename|delete) ")
_CHAT_COMMIT = re.compile(r"^wiki: \d+ ops? \(session [^)]*\)$")
BRIEFING_CAP = 8                 # items per card section, then "+n more"
_CHANGE = {"A": "new", "M": "edited", "D": "deleted"}
_MEMORY_OPS = {"ADD": "added", "UPDATE": "updated", "DELETE": "removed"}


def _news_commit(subject: str) -> bool:
    """False for the layer's own housekeeping (lint, vault creation) and the user's own writes
    (`_user_commit`: editor save / create / rename / delete, PRD 43 save card): never news."""
    return subject not in _NOISE_COMMITS and not _USER_COMMIT.match(subject)


def _news_path(path: str) -> bool:
    return path.endswith(".md") and path not in store.RESERVED and not path.startswith((f"{store.RAW_DIR}/", "sources/"))


def _housekeeping(subject: str, paths: list) -> bool:
    """The layer's own upkeep: a `_NOISE_COMMITS` subject, or a commit touching only index / log /
    schema (docs/51-run-cards.md counts these as the briefing's skipped commits)."""
    return subject in _NOISE_COMMITS or (bool(paths) and all(p in store.RESERVED for p in paths))


def _page_changes(ts: str) -> tuple:
    """``(pages, chat_pages, skipped)`` from `git log --since=ts`; nothing when git or the repo is
    missing. A page carries the short sha of its newest commit; `skipped` counts housekeeping commits."""
    result = store._git("log", f"--since={ts}", "--name-status", "--format=%x1e%h %s", quiet=True)
    if result is None or result.returncode != 0:
        return [], [], 0
    seen: dict = {}                                      # path -> {"status": [..], "chat": bool}, newest first
    skipped = 0
    for record in result.stdout.split("\x1e")[1:]:
        lines = [line for line in record.splitlines() if line.strip()]
        sha, _, subject = (lines[0] if lines else "").partition(" ")
        if _housekeeping(subject, [line.split("\t")[-1] for line in lines[1:]]):
            skipped += 1
        if not _news_commit(subject):
            continue
        for line in lines[1:]:
            fields = line.split("\t")
            status = fields[0][:1]
            touched = [(fields[1], "D"), (fields[2], "A")] if status == "R" and len(fields) == 3 else \
                [(fields[-1], status)]
            for path, st in touched:
                if not _news_path(path):
                    continue
                entry = seen.setdefault(path, {"status": [], "chat": True, "commit": sha})
                entry["status"].append(st)
                entry["chat"] = entry["chat"] and bool(_CHAT_COMMIT.match(subject))
    pages, chat = [], []
    for path, entry in seen.items():
        rel = store.wikilink(path)
        added, here = "A" in entry["status"], os.path.isfile(store._path(path))
        if not here and added:
            continue                                     # born and gone inside the window
        change = "new" if added and here else "deleted" if not here else "edited"
        if entry["chat"] and here:
            chat.append(rel)
        else:
            pages.append({"path": rel, "title": page_title(rel) or os.path.basename(rel), "change": change,
                          "commit": entry["commit"]})
    order = list(_CHANGE.values())
    pages.sort(key=lambda p: (order.index(p["change"]), p["path"]))
    return pages, sorted(chat), skipped


def _source_changes(ts: str) -> list:
    from vault import manifest
    from ingest import pipeline
    out = []
    for entry in manifest.manifest_entries():
        at = str(entry.get("ingested_at") or "")
        if "T" not in at:                                # an old line: the date lives on the page
            at = pipeline._source_meta(entry)["when"].replace(" ", "T")
        if len(at) < 19 or at <= ts:
            continue
        out.append({"page": entry.get("page", ""), "path": entry.get("path", ""),
                    "kind": str(entry.get("kind") or ""), "at": at})
    return sorted(out, key=lambda s: s["at"])


def _memory_changes(ts: str) -> dict:
    counts = {"added": 0, "updated": 0, "removed": 0, "examples": []}
    try:
        with open(memory.OPS_LOG_PATH) as f:
            lines = f.readlines()
    except OSError:
        return counts
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = _MEMORY_OPS.get(str(rec.get("op") or ""))
        if not kind or str(rec.get("ts") or "") <= ts or rec.get("reason") in memory.UI_REASONS:
            continue
        counts[kind] += 1
        if len(counts["examples"]) < 2:
            counts["examples"].append({"op": rec["op"], "path": rec.get("path", ""), "value": rec.get("value")})
    return counts


def changes_since(ts: str) -> dict:
    """The briefing digest: pages, chat pages, sources and memory changes after `ts` (PRD 46), plus
    `skipped`, the housekeeping commits left out (PRD 51)."""
    with vault_lock.LOCK:
        pages, chat, skipped = _page_changes(ts) if store.exists() else ([], [], 0)
        return {"since": ts, "pages": pages, "chat_pages": chat, "sources": _source_changes(ts),
                "memory": _memory_changes(ts), "skipped": skipped}


def head_sha() -> str:
    """The vault's HEAD as a short sha ("" without a repo or a commit)."""
    result = store._git("rev-parse", "--short", "HEAD", quiet=True)
    return result.stdout.strip() if result is not None and result.returncode == 0 else ""


def memory_total(digest: dict) -> int:
    """How many memory ops the digest counts (added + updated + removed)."""
    mem = digest["memory"]
    return mem["added"] + mem["updated"] + mem["removed"]


def digest_empty(digest: dict) -> bool:
    return not (digest["pages"] or digest["chat_pages"] or digest["sources"] or memory_total(digest))


def _day(d) -> str:
    return f"{d.day} {d.strftime('%b')}"


def briefing_when(ts: str) -> str:
    """"14:05" today, "yesterday at 14:05", else "12 Oct"."""
    from datetime import date, timedelta
    try:
        day = date.fromisoformat(ts[:10])
    except ValueError:
        return ts
    today = memory._today()
    if day == today:
        return ts[11:16]
    if day == today - timedelta(days=1):
        return f"yesterday at {ts[11:16]}"
    return _day(day)


def memory_value(value) -> str:
    """An op value as one short line: a dict's values joined by " · "."""
    if isinstance(value, dict):
        value = " · ".join(str(v) for v in value.values() if v not in (None, ""))
    return " ".join(str(value if value is not None else "").split())[:80]


def capped(items: list) -> tuple:
    """``(the first BRIEFING_CAP items, how many more)``."""
    return items[:BRIEFING_CAP], max(0, len(items) - BRIEFING_CAP)


def _capped(items: list) -> list:
    head, extra = capped(items)
    return head + ([f"+{extra} more"] if extra else [])


def briefing_card(digest: dict) -> str:
    """The briefing as a Markdown card: `## Pages` / `## Sources` / `## Memory`, empty ones left out."""
    out = [f"# Briefing {_day(memory._today())}"]
    pages = _capped([f"{p['change']}: " + (p["title"] if p["change"] == "deleted" else f"[[{p['path']}|{p['title']}]]")
                     for p in digest["pages"]])
    chat = digest["chat_pages"]
    if chat:
        more = f" (+{len(chat) - 2})" if len(chat) > 2 else ""
        pages.append("notes from our chats: " + ", ".join(f"[[{c}]]" for c in chat[:2]) + more)
    sections = [("Pages", pages),
                ("Sources", _capped([f"[[{s['page']}]] · {s['kind'] or 'file'}" for s in digest["sources"]]))]
    mem = digest["memory"]
    if memory_total(digest):
        lines = [f"{mem['added']} added, {mem['updated']} updated, {mem['removed']} removed"]
        lines += [f"{_MEMORY_OPS[e['op']]} {e['path']}: {memory_value(e['value'])}".rstrip(": ")
                  for e in mem["examples"]]
        sections.append(("Memory", lines))
    for heading, items in sections:
        if items:
            out += ["", f"## {heading}"] + [f"- {item}" for item in items]
    return "\n".join(out)


# --- listing / lint ----------------------------------------------------------------


def pages() -> list:
    """``[{path, title, type, updated}]`` for the Wiki tab, sorted by path.

    Paths are suffix-less (`people/anna`), the same form wikilinks and the tools use; the
    normalising on the way back in (:func:`normalise_path`) still accepts `.md`.
    """
    listing = []
    for rel in store.page_paths():
        text = store._read(rel)
        fm, _ = store.parse_page(text)
        listing.append({
            "path": store.wikilink(rel),
            "title": str(fm.get("name") or os.path.basename(rel)[:-3]),
            "type": str(fm.get("type") or ""),
            "updated": str(fm.get("updated") or ""),
            "tags": v_search.tags(text),
        })
    return listing


REINDEX_HINT = " — run `python -m wiki reindex`"


def lint() -> list:
    """Findings, worst first: per page broken links, missing from the search index, bad frontmatter,
    size, a `source: raw/…` whose file is gone; then index rows whose page is gone (both mean
    `python -m wiki reindex`), raw files nothing cites, then duplicate names. `index.md` is a capped
    human view (200 lines), so it is not what a page is checked against (docs/53-scalable-index.md).

    A finding names its page the way the rest of the layer does, `people/anna`, or its raw file.
    """
    from vault import manifest

    if not store.exists():
        return []
    findings = []
    cited, labels = set(), []
    paths = store.page_paths()
    known = {store.wikilink(rel) for rel in paths} | {store.wikilink(r) for r in store.RESERVED}
    indexed = set(v_search._db().page_paths())
    for rel in paths:
        text = store._read(rel)
        fm, _ = store.parse_page(text)
        name = store.wikilink(rel)
        for target in links_in(text):
            if target not in known:
                findings.append(f"{name}: broken link [[{target}]]")
        if rel not in indexed:
            findings.append(f"{name}: missing from the search index{REINDEX_HINT}")
        missing = [key for key in store.FIELDS if key not in fm]
        if missing:
            findings.append(f"{name}: frontmatter missing {', '.join(missing)}")
        if len(text.encode()) > store.MAX_PAGE_BYTES:
            findings.append(f"{name}: larger than {store.MAX_PAGE_BYTES // 1024} kB")
        source = str(fm.get("source") or "")
        if source.startswith(store.RAW_DIR + "/") and not os.path.isfile(store._path(source)):
            findings.append(f"{name}: source {source} no longer exists")
        cited |= {str(fm.get(key) or "") for key in ("source", "raw")}
        labels.append((name, " ".join(str(fm.get("name") or "").split())))
    findings += [f"{store.wikilink(rel)}: in the search index but not on disk{REINDEX_HINT}"
                 for rel in sorted(indexed - set(paths))]
    cited |= {e.get("path") for e in manifest.manifest_entries()}
    cited |= {rel for e in manifest.manifest_entries() for rel in e.get("pages") or []}   # PRD 47 PNGs
    for root, dirs, files in os.walk(store._path(store.RAW_DIR)):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fname in sorted(files):
            rel = os.path.relpath(os.path.join(root, fname), store.WIKI_PATH).replace(os.sep, "/")
            if not fname.startswith(".") and rel != manifest.MANIFEST and rel not in cited:
                findings.append(f"{rel}: not in the manifest and cited by no page")
    first = {}
    for name, label in sorted(labels):
        if not label or name.startswith("meetings/"):   # recurring invites share a name
            continue
        group = (name.startswith("sources/"), label.casefold())
        if group in first:
            findings.append(f'{name}: duplicate name "{label}" (also {first[group]})')
        first.setdefault(group, name)
    return findings


def lint_and_log() -> list:
    """`lint()` from the reflection cadence: only a run with findings earns a `log.md` line."""
    findings = lint()
    if findings:
        store.ensure_vault()
        summary = f"lint: {len(findings)} finding{'s' if len(findings) != 1 else ''} ({findings[0]})"
        v_index._log([summary])
        store._commit("wiki: lint")
    return findings


# --- ingest ------------------------------------------------------------------------


INGEST_SYSTEM = """You keep a markdown wiki of the USER's working life: people, projects,
meetings and decisions. You are given the wiki's schema, its index, and a transcript of the
conversation that just happened.

Reply with a JSON array of page operations and nothing else. Each operation is one of:
{"op":"new_page","path":"people/anna","type":"person","name":"Anna","summary":"one line"}
{"op":"append_note","path":"projects/atlas","text":"one sentence"}
{"op":"set_field","path":"projects/atlas","key":"status","value":"paused"}
{"op":"add_decision","title":"deploy on Fridays","text":"one or two sentences"}
{"op":"link","path":"people/anna","target":"projects/atlas"}

Paths are `people/<slug>`, `projects/<slug>`, `meetings/YYYY-MM-DD` or `decisions`.
Frontmatter keys are exactly: type, name, tags, source, updated, seen.

Rules:
- Only file what the USER actually said. Never the assistant's words, never a guess.
- %s
- File people, projects, meetings and decisions — things the user would expect you to remember
  in a month.
- Never file everyday trivia: meals, weather, moods, small talk, chores, commutes.
- Append to an existing page from the index instead of creating a near-duplicate.
- A note is one sentence, at most 600 characters.
- """ + DOCUMENT_RULE + """
- Nothing worth filing? Reply with exactly [].
Today is %s."""


def user_page_reason(op: dict, names) -> str | None:
    """Why `op` would file the user as a person — or None when it is fine.

    Only `new_page people/<slug>` and a `link` whose target is a people page can do that; the
    slug and the op's `name` are both checked, because the model spells either one.
    """
    kind = str(op.get("op") or "").strip().lower()
    if kind == "new_page":
        checked = [op.get("path"), op.get("name")]
    elif kind == "link":
        checked = [op.get("target")]
    else:
        return None
    rel = store.normalise_path(op.get("path") if kind == "new_page" else op.get("target"))
    if rel is None or not rel.startswith("people/"):
        return None
    checked.append(os.path.basename(rel)[:-3].replace("-", " "))
    for value in checked:
        text = " ".join(str(value or "").split())
        if "/" in text:
            continue
        if text and memory.is_user(text, names):
            return f"{text} is the user, not another person"
    return None


def reject_user_pages(ops, names) -> tuple:
    """``(kept, rejected)`` — the user never gets a people page of their own."""
    kept, rejected = [], []
    for raw in ops if isinstance(ops, list) else []:
        reason = user_page_reason(raw if isinstance(raw, dict) else {}, names)
        (rejected if reason else kept).append({"op": raw, "reason": reason} if reason else raw)
    return kept, rejected


def ingest(turns: list) -> list:
    """Ask the model for page ops for this session and apply them; returns the applied ops.

    Never raises: a wiki failure must not cost the user their profile save (see memory.save).
    """
    transcript = memory._transcript(turns)
    if not transcript.strip():
        return []
    names = memory.user_names()
    system = INGEST_SYSTEM % (memory.user_note(names), memory._today_iso())
    related = v_index.related_context(transcript)
    user = (f"Schema:\n{store._read(store.SCHEMA_FILE) or store.SCHEMA_TEXT}\n\n"
            f"Index:\n{v_index.vault_map(store.WIKI_INDEX_TOKENS) or '(empty wiki)'}\n\n"
            + (f"{related}\n\n" if related else "")
            + f"Transcript:\n{transcript}")
    try:
        raw = core_llm.complete(system, user, 800)
    except Exception as exc:  # noqa: BLE001
        log.info(f"ingest failed: {exc}")
        return []
    ops = memory.extract_json(raw)
    if not isinstance(ops, list):
        log.info("could not parse page ops; nothing filed")
        return []
    ops, about_user = reject_user_pages(ops, names)
    applied, rejected = v_pages.apply_page_ops(ops, f"session {memory._today_iso()}")
    v_pages.log_rejected(about_user + rejected, log)
    if applied:
        log.info(f"{len(applied)} page op{'s' if len(applied) != 1 else ''} applied")
    return applied


# --- bridge helpers (used by memory.archive / memory.pull) --------------------------


def page_for(section: str, entry: dict) -> tuple:
    """Where a profile entry of `section` archives to: ``(path, type, name)``."""
    if section == "people":
        name = str(entry.get("name") or "").strip()
        return f"people/{store.slugify(name)}.md", "person", name
    if section == "projects":
        name = str(entry.get("name") or "").strip()
        return f"projects/{store.slugify(name)}.md", "project", name
    return store.RECURRING_FILE, "recurring", "Recurring"


def history_line(section: str, entry: dict) -> str:
    """The `## History` bullet an archived entry leaves behind."""
    fields = {"people": ("name", "rel", "note"),
              "projects": ("name", "status", "note"),
              "recurring": ("what", "when", "note")}.get(section, ("name", "note"))
    parts = [str(entry[f]).strip() for f in fields if str(entry.get(f, "")).strip()]
    seen = str(entry.get(memory.SEEN_FIELD) or "").strip()
    text = "; ".join(parts) or section
    return f"{text} (last seen {seen})" if seen else text


def archive_entry(section: str, entry: dict) -> tuple:
    """Create or extend the wiki page for `entry`; ``(page, None)`` or ``(None, reason)``."""
    store.ensure_vault()
    rel, page_type, name = page_for(section, entry)
    source = f"archived from {section} {memory._today_iso()}"
    ops = []
    if not store.page_exists(rel):
        ops.append({"op": "new_page", "path": rel, "type": page_type,
                    "name": name or rel[:-3], "summary": history_line(section, entry),
                    "source": source})
    ops.append({"op": "append_note", "path": rel, "text": history_line(section, entry),
                "source": source, "section": "## History"})
    applied, rejected = v_pages.apply_page_ops(ops, source)
    if not applied:
        return None, (rejected[0]["reason"] if rejected else "nothing to archive")
    return store.wikilink(rel), None


def entry_from_page(path: str, section: str) -> tuple:
    """A live profile entry built from a page's frontmatter; ``(entry, None)`` or ``(None, reason)``."""
    rel = store.normalise_path(path)
    if rel is None or not store.page_exists(rel):
        return None, f"no such page {path}"
    fm, body = store.parse_page(store._read(rel))
    name = str(fm.get("name") or os.path.basename(rel)[:-3]).strip()
    summary = v_index._summary_line(body)
    entry = {memory.SEEN_FIELD: memory._today_iso()}
    if section == "recurring":
        entry["what"] = name
        if summary:
            entry["when"] = summary[:120]
    else:
        entry["name"] = name
        status = str(fm.get("status") or "").strip()
        if section == "projects":
            entry["status"] = status or "active"
        if summary:
            entry["note"] = summary[:200]
    return entry, None


def cli_vault(env_file: str) -> str:
    """The vault `python -m wiki` works on: WIKI_PATH from the environment, else from `env_file`
    (the checkout's `.env`, so a dev checkout targets its dev vault), else the default."""
    from dotenv import dotenv_values

    value = os.environ.get("WIKI_PATH") or dotenv_values(env_file).get("WIKI_PATH") or "~/llm-wiki-dev"
    return os.path.expanduser(value)


def main(argv) -> int:
    """`python -m wiki reindex`: rebuild the search index; refused (exit 2) while the app runs,
    because swapping the file under a running app would orphan its open handle."""
    import sys

    from ingest import batch

    if list(argv) != ["reindex"]:
        print("usage: python -m wiki reindex", file=sys.stderr)
        return 2
    running = batch.running_app()
    if running:
        print(f"refusing to reindex: {running}", file=sys.stderr)
        return 2
    print(f"reindexed {store.WIKI_PATH} -> {v_search.reindex()}")
    return 0


# --- step 6 forwarding (PRD 50; deleted with the shim at step 12) ---------------------------------------------
# Names that moved to vault/ stay readable here; assigning one would patch a copy nothing reads, so it raises.
import sys as _sys
import types as _types


def _owner(name):
    for module in (store, v_index, v_pages, v_search):
        obj = vars(module).get(name)
        if obj is not None and not isinstance(obj, _types.ModuleType) and \
                getattr(obj, "__module__", module.__name__) == module.__name__:
            return module
    return None


def __getattr__(name):
    module = _owner(name)
    if module is None:
        raise AttributeError(f"module 'wiki' has no attribute {name!r}")
    return getattr(module, name)


class _Forwarding(_types.ModuleType):
    def __setattr__(self, name, value):
        if name not in vars(self) and _owner(name) is not None:
            raise AttributeError(f"wiki.{name} moved to vault/: patch the module that defines it")
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Forwarding


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    import wiki as _wiki      # the imported module, not this __main__ copy: the index handle lives there

    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env)         # the app's ports for the running-app guard
    store.WIKI_PATH = _wiki.cli_vault(_env)
    sys.exit(_wiki.main(sys.argv[1:]))
