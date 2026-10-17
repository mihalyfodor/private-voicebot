"""Page search: BM25 over the index_db tables, tags, backlinks, reindex (PRD 53; PRD 50 step 6, from wiki.py)."""
import contextlib
import hashlib
import json
import os
import re
from collections import Counter
from vault import index_db
import memory
from core import trace
from vault import index as v_index
from vault import store
from vault import chunks
from vault import focus

log = trace.logger("Wiki")


# --- search ------------------------------------------------------------------------
# BM25 over page bodies + frontmatter. Personal scale (≤ 200 pages), no embeddings, no new
# dependency, and deterministic — which is what makes it testable.

def _stem(word: str) -> str:
    """A light stem so "lockpick" finds "lockpicking" and "turret" "turrets" — no dependency."""
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def _tokenise(text: str) -> list:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())]


def _db() -> index_db.Index:
    """The vault's search index (docs/53-scalable-index.md). The first open in this process runs
    the consistency check, so a vault without an index file (or with a stale one) gets one."""
    idx = index_db.open_index(store.WIKI_PATH)
    if not idx.checked:
        idx.checked = True
        try:
            _check(idx)
        except BaseException:
            idx.checked = False
            raise
    return idx


def _hook_db() -> index_db.Index | None:
    """The index for a write hook; None when the vault does not exist yet (nothing to index)."""
    return _db() if store.exists() else None


def _is_page(rel: str) -> bool:
    """What :func:`page_paths` would list, by path alone (an empty decisions.md is decided on its text)."""
    parts = rel.split("/")
    return (rel.endswith(".md") and rel not in store.RESERVED and parts[0] != store.RAW_DIR
            and not any(p.startswith(".") for p in parts[:-1]))


def _sha(text: str) -> str:
    """Hex SHA-256 of a text."""
    return hashlib.sha256(text.encode()).hexdigest()


def _put_page(idx: index_db.Index, rel: str, text: str, mtime: int) -> None:
    """Write one page's row and postings from its text."""
    fm, body = store.parse_page(text)
    fields = {"type": str(fm.get("type") or ""), "name": str(fm.get("name") or ""),
              "tags": json.dumps(tags(text)), "updated": str(fm.get("updated") or ""),
              "one_liner": v_index._one_liner(fm, body), "links": json.dumps(focus.links_in(body))}
    idx.put_page(rel, mtime, _sha(text), fields, _tokenise(text))


def index_page(rel: str, text: str | None = None) -> None:
    """Write hook: (re)index one page after its file was written (`_write` calls it)."""
    if not _is_page(rel) or (idx := _hook_db()) is None:
        return
    text = store._read(rel) if text is None else text
    if not os.path.isfile(store._path(rel)) or (rel == store.DECISIONS_FILE and not store._has_decisions(text)):
        idx.drop_page(rel)
        return
    _put_page(idx, rel, text, os.stat(store._path(rel)).st_mtime_ns)


def unindex_page(rel: str) -> None:
    """Write hook: a page file was removed (rename, delete, forget); its rows go."""
    if _is_page(rel) and (idx := _hook_db()) is not None:
        idx.drop_page(rel)


def _check(idx: index_db.Index, *, batch: bool | None = None) -> int:
    """Compare the index to the vault and repair the difference; returns the documents changed.

    Pages key on (mtime, sha): an unchanged mtime costs one stat, a changed one a read and a hash.
    Sources key on the manifest: shas gone are dropped, shas new are indexed from their chunk cache.
    """
    with idx.write() if (idx.fresh if batch is None else batch) else contextlib.nullcontext():
        stored = idx.page_stamps()
        changed = 0
        for rel in store.page_paths():
            mtime = os.stat(store._path(rel)).st_mtime_ns
            known = stored.pop(rel, None)
            if known and known[0] == mtime:
                continue
            text = store._read(rel)
            if known and known[1] == _sha(text):
                idx.touch_page(rel, mtime)
                continue
            _put_page(idx, rel, text, mtime)
            changed += 1
        for rel in stored:
            idx.drop_page(rel)
            changed += 1
        changed += chunks._sync_sources(idx)
    idx.fresh = False
    return changed


def startup_check() -> int:
    """The consistency check, run once at app (and batch-import) start; returns the documents
    repaired. A vault that does not exist yet is left alone (it is created lazily)."""
    if not store.exists():
        return 0
    idx = index_db.open_index(store.WIKI_PATH)
    changed = _check(idx)
    idx.checked = True                  # only after it ran: a failed check is retried on the next open
    idx.set_meta("checked_at", memory._stamp("%H:%M:%S"))
    return changed


def reindex(path: str | None = None) -> str:
    """Rebuild the search index from scratch (pages on disk + the manifest); returns the file.

    `path` builds into another file and leaves the live one alone (the rebuild-equals-incremental
    test); otherwise the new file replaces the live index. `python -m wiki reindex` runs this.
    """
    if not store.exists():
        raise FileNotFoundError(f"no vault at {store.WIKI_PATH}")
    live = index_db.db_path(store.WIKI_PATH)
    target = path or live + ".new"
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(target + suffix):
            os.remove(target + suffix)
    idx = index_db.Index(target)
    try:
        _check(idx, batch=True)
    finally:
        idx.close()
    if path:
        return path
    index_db.replace(store.WIKI_PATH, target)
    _db()
    return live


def backlinks(path: str) -> list:
    """The pages (suffix-less) whose wikilinks point at `path`; alias/heading forms count.

    Only the pages the index says hold every word of the target are read."""
    rel = store.normalise_path(path, user=True)
    if rel is None or not store.exists():
        return []
    target = store.wikilink(rel)
    idx, words = _db(), set(_tokenise(target))
    candidates = idx.docs_with_all(index_db.PAGE, words) if words else idx.page_paths()
    found = []
    for other in sorted(candidates):
        if other != rel and target in focus.links_in(store._read(other)):
            found.append(store.wikilink(other))
    return found


_TAG = re.compile(r"(?<![\w/#&])#([A-Za-z][\w/-]*)")


def tags(text: str) -> list:
    """Frontmatter `tags:` merged with the body's `#tag`s, sorted, de-duplicated."""
    fm, body = store.parse_page(text)
    listed = fm.get("tags") or []
    found = {str(t).strip().lstrip("#") for t in (listed if isinstance(listed, list) else [listed])}
    found |= set(_TAG.findall(body))
    return sorted(t for t in found if t)


def _snippet(text: str, terms: set, width: int = 180) -> str:
    body = " ".join(store.parse_page(text)[1].split())
    lower = body.lower()
    at = min((lower.find(t) for t in terms if lower.find(t) >= 0), default=0)
    start = max(0, at - 40)
    return ("…" if start else "") + body[start:start + width] + ("…" if len(body) > start + width else "")


def _bm25(docs: list, terms: set, weights: dict | None = None) -> list:
    """BM25 score of `terms` against each token list in `docs` (0.0 where nothing matched);
    `weights` scales a term's contribution (default 1.0). In memory, for a handful of sections
    (:func:`best_section`); the vault-wide searches score from the index (`index_db.Index.bm25`)."""
    n = len(docs)
    weights = weights or {}
    avgdl = sum(len(t) for t in docs) / n
    df = Counter(term for tokens in docs for term in terms if term in tokens)
    scores = []
    for tokens in docs:
        counts, dl, score = Counter(tokens), len(tokens) or 1, 0.0
        for term in terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            score += index_db.bm25_term(weights.get(term, 1.0), freq, dl, avgdl, df[term], n)
        scores.append(score)
    return scores


def _page_subset(idx: index_db.Index, filters: dict | None):
    """The pages `filters` keeps — ``{type, tags, updated_from, updated_to}`` — or None (all)."""
    if not filters:
        return None
    want_type = filters.get("type")
    types = {want_type} if isinstance(want_type, str) else set(want_type or ())
    want_tags = filters.get("tags")
    tagset = {want_tags} if isinstance(want_tags, str) else set(want_tags or ())
    since, until = str(filters.get("updated_from") or ""), str(filters.get("updated_to") or "")
    keep = set()
    for rel, (kind, tag_json, updated) in idx.page_fields().items():
        if types and kind not in types:
            continue
        if tagset and not tagset <= set(json.loads(tag_json or "[]")):
            continue
        if (since and updated < since) or (until and updated > until):
            continue
        keep.add(rel)
    return keep


def search(query: str, k: int = 5, filters: dict | None = None) -> list:
    """Top-`k` pages for `query` as ``[{path, snippet, score}]`` (BM25, best first).

    `path` is the suffix-less form the model sees everywhere else (`people/anna`), so a hit can
    be handed straight to :func:`read_page`. Raw sources are a separate index: :func:`search_chunks`.
    `filters` (``{type, tags, updated_from, updated_to}``) narrows the pages scored; the BM25
    statistics are then the narrowed set's. Reads the index plus the `k` pages it returns.
    """
    terms = set(_tokenise(query))
    if not store.exists() or not terms:
        return []
    idx = _db()
    if not idx.page_lengths():
        return []
    scores = idx.bm25(index_db.PAGE, terms, None, _page_subset(idx, filters))
    ranked = sorted(((round(score, 3), store.wikilink(rel), rel) for rel, (score, _) in scores.items() if score > 0),
                    key=lambda r: (-r[0], r[1]))[:k]
    return [{"path": path, "snippet": _snippet(store._read(rel), terms), "score": score}
            for score, path, rel in ranked]
