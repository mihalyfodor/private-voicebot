"""Search inside sources: raw-file chunking, the `raw/.index/` cache and chunk search (docs/32-source-recall.md; PRD 50 step 7, from wiki.py)."""
import json
import os
import re
from collections import Counter
from vault import index_db
from core import settings, trace
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
