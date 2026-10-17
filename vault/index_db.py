"""The persisted search index (docs/53-scalable-index.md): one SQLite file per vault.

`<vault>/.index/search.sqlite` holds postings for the pages and the source chunks, so a search
touches only the documents that hold a query word instead of re-tokenising the vault. This module
is storage + BM25 only: the caller (wiki.py) tokenises, parses and decides what is a page; nothing
here reads the vault. Chunk text stays in `raw/.index/<sha>.json`.

Every change runs in one transaction (`Index.write`); `meta.generation` moves with each, so the
in-memory read caches (`Index.cached`) notice a write from this or another process.
"""
import array
import contextlib
import math
import os
import sqlite3
import threading
import uuid
from collections import defaultdict

INDEX_DIR = ".index"
DB_NAME = "search.sqlite"
SCHEMA_VERSION = "2"          # 2: `pages.links` (the vault map's inbound counts, PRD 54)
K1, B = 1.5, 0.75
PAGE, CHUNK = "page", "chunk"
#: meta keys that describe this file, not the vault: left out of `dump`
VOLATILE_META = ("epoch", "generation", "checked_at", "manifest")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages(path TEXT PRIMARY KEY, mtime INTEGER, sha TEXT, type TEXT, name TEXT,
                                 tags TEXT, updated TEXT, one_liner TEXT, length INTEGER,
                                 links TEXT);
CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, source TEXT, raw TEXT, sha TEXT, title TEXT,
                                  n INTEGER, total INTEGER, text_offset INTEGER, length INTEGER);
CREATE INDEX IF NOT EXISTS chunks_sha ON chunks(sha);
CREATE TABLE IF NOT EXISTS terms(corpus TEXT, term TEXT, df INTEGER, PRIMARY KEY(corpus, term)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS postings(corpus TEXT, term TEXT, doc, tf INTEGER, positions BLOB,
                                    PRIMARY KEY(corpus, term, doc)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS postings_doc ON postings(corpus, doc);
CREATE TABLE IF NOT EXISTS proper(term TEXT PRIMARY KEY, seen INTEGER, titled INTEGER) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""
TABLES = ("pages", "chunks", "terms", "postings", "proper", "meta")


def bm25_term(weight: float, freq: int, dl: int, avgdl: float, df: int, n: int) -> float:
    """One term's BM25 contribution to one document (the one formula; the in-memory and the indexed scorer share it)."""
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    return weight * idf * freq * (K1 + 1) / (freq + K1 * (1 - B + B * dl / avgdl))


def db_path(vault: str) -> str:
    """Where a vault keeps its index file."""
    return os.path.join(vault, INDEX_DIR, DB_NAME)


def _file_id(path: str):
    """(device, inode) of the file, or None when it is gone: a swapped-in file has a new one."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return st.st_dev, st.st_ino


def _positions(blob) -> list:
    return list(array.array("I", blob)) if blob else []


class Index:
    """One open index file. Thread-safe through one re-entrant lock (reads and writes are short)."""

    def __init__(self, path: str):
        self.path = path
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        ignore = os.path.join(folder, ".gitignore")
        if not os.path.exists(ignore):            # git ignores the folder even before the vault's .gitignore says so
            with open(ignore, "w") as f:
                f.write("*\n")
        self.lock = threading.RLock()
        self.checked = False                       # the owner's consistency check ran against this handle
        self._depth = 0
        self._cache = {}
        try:
            self.conn = self._connect()
        except sqlite3.DatabaseError:              # a corrupt cache file is rebuilt, never fatal
            for suffix in ("", "-wal", "-shm"):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path + suffix)
            self.conn = self._connect()
        self.file_id = _file_id(path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-65536")
            conn.executescript(SCHEMA)
            row = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
            if row and row[0] != SCHEMA_VERSION:
                conn.executescript("".join(f"DROP TABLE IF EXISTS {t};" for t in TABLES) + SCHEMA)
                row = None
            self.fresh = row is None
            if self.fresh:
                conn.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                                 [("schema", SCHEMA_VERSION), ("epoch", uuid.uuid4().hex), ("generation", "0")])
        except sqlite3.DatabaseError:
            conn.close()
            raise
        return conn

    def close(self) -> None:
        """Close and forget the open handle (tests, and before a file swap)."""
        with self.lock:
            self.conn.close()

    # --- transactions + caches ------------------------------------------------------

    @contextlib.contextmanager
    def write(self):
        """One transaction; nested calls join the outer one (a rebuild is one transaction)."""
        with self.lock:
            if self._depth:
                self._depth += 1
                try:
                    yield self.conn
                finally:
                    self._depth -= 1
                return
            self.conn.execute("BEGIN IMMEDIATE")
            self._depth = 1
            try:
                yield self.conn
                self.conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='generation'")
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            finally:
                self._depth = 0

    def stamp(self) -> tuple:
        """The index's epoch/generation stamp, to tell whether another process changed it."""
        with self.lock:
            rows = dict(self.conn.execute("SELECT key, value FROM meta WHERE key IN ('epoch', 'generation')"))
        return rows.get("epoch"), rows.get("generation")

    def cached(self, name: str, build):
        """`build()` memoised until the next write (in any process)."""
        stamp = self.stamp()
        hit = self._cache.get(name)
        if hit and hit[0] == stamp:
            return hit[1]
        value = build()
        self._cache[name] = (stamp, value)
        return value

    def get_meta(self, key: str, default=None):
        """One meta value, or None."""
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        """Store one meta value."""
        with self.write() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, str(value)))

    # --- postings -------------------------------------------------------------------

    def _add_doc(self, db, corpus: str, doc, tokens: list, positions: bool) -> None:
        at = defaultdict(list)
        for i, term in enumerate(tokens):
            at[term].append(i)
        db.executemany("INSERT INTO postings VALUES (?, ?, ?, ?, ?)",
                       [(corpus, term, doc, len(p), array.array("I", p).tobytes() if positions else None)
                        for term, p in at.items()])
        db.executemany("INSERT INTO terms VALUES (?, ?, 1) ON CONFLICT(corpus, term) DO UPDATE SET df = df + 1",
                       [(corpus, term) for term in at])

    def _drop_doc(self, db, corpus: str, doc) -> None:
        terms = [(corpus, t) for (t,) in db.execute(
            "SELECT term FROM postings WHERE corpus=? AND doc=?", (corpus, doc))]
        db.executemany("UPDATE terms SET df = df - 1 WHERE corpus=? AND term=?", terms)
        db.executemany("DELETE FROM terms WHERE corpus=? AND term=? AND df <= 0", terms)
        db.execute("DELETE FROM postings WHERE corpus=? AND doc=?", (corpus, doc))

    # --- pages ----------------------------------------------------------------------

    def put_page(self, path: str, mtime: int, sha: str, fields: dict, tokens: list) -> None:
        """Replace one page's row and postings. `fields`: type, name, tags (json), updated, one_liner,
        links (json: the page's wikilink targets)."""
        with self.write() as db:
            self._drop_doc(db, PAGE, path)
            db.execute("INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (path, mtime, sha, fields.get("type", ""), fields.get("name", ""),
                        fields.get("tags", "[]"), fields.get("updated", ""), fields.get("one_liner", ""),
                        len(tokens), fields.get("links", "[]")))
            self._add_doc(db, PAGE, path, tokens, positions=False)

    def drop_page(self, path: str) -> None:
        """Remove one page's row and postings."""
        with self.write() as db:
            self._drop_doc(db, PAGE, path)
            db.execute("DELETE FROM pages WHERE path=?", (path,))

    def touch_page(self, path: str, mtime: int) -> None:
        """Same bytes, new mtime (a `git checkout`, a no-op save): no postings change."""
        with self.write() as db:
            db.execute("UPDATE pages SET mtime=? WHERE path=?", (mtime, path))

    def page_stamps(self) -> dict:
        """path -> (mtime, sha) for the startup check."""
        with self.lock:
            return {p: (m, s) for p, m, s in self.conn.execute("SELECT path, mtime, sha FROM pages")}

    def page_paths(self) -> list:
        """Every indexed page path."""
        with self.lock:
            return [p for (p,) in self.conn.execute("SELECT path FROM pages ORDER BY path")]

    def index_rows(self, limit: int) -> list:
        """(path, one_liner) in path order — what `index.md` lists, without reading a page."""
        with self.lock:
            return self.conn.execute("SELECT path, one_liner FROM pages ORDER BY path LIMIT ?",
                                     (limit,)).fetchall()

    def page_fields(self) -> dict:
        """path -> (type, tags json, updated) for the search filters."""
        with self.lock:
            return {p: (t, g, u) for p, t, g, u in self.conn.execute("SELECT path, type, tags, updated FROM pages")}

    def map_rows(self) -> list:
        """(path, type, name, tags json, updated, one_liner, links json) of every page, in path order —
        what the vault map folds (PRD 54), memoised until the next write."""
        def build():
            with self.lock:
                return self.conn.execute("SELECT path, type, name, tags, updated, one_liner, links FROM pages "
                                         "ORDER BY path").fetchall()
        return self.cached("map_rows", build)

    def page_lengths(self) -> dict:
        """path -> token count of every indexed page."""
        def build():
            with self.lock:
                return dict(self.conn.execute("SELECT path, length FROM pages"))
        return self.cached("page_lengths", build)

    # --- sources --------------------------------------------------------------------

    def put_source(self, sha: str, raw: str, source: str, title: str, chunks: list,
                   seen: dict, titled: dict) -> None:
        """Replace one source's chunks: `chunks` is ``[(n, start, tokens)]``; `seen` / `titled`
        are this source's proper-noun counts (stem -> occurrences / Title-case occurrences)."""
        with self.write() as db:
            self.drop_source(sha, seen, titled)
            total = len(chunks)
            for n, start, tokens in chunks:
                cur = db.execute("INSERT INTO chunks(source, raw, sha, title, n, total, text_offset, length) "
                                 "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                 (source, raw, sha, title, n, total, start, len(tokens)))
                self._add_doc(db, CHUNK, cur.lastrowid, tokens, positions=True)
            db.executemany("INSERT INTO proper VALUES (?, ?, ?) ON CONFLICT(term) DO UPDATE "
                           "SET seen = seen + excluded.seen, titled = titled + excluded.titled",
                           [(t, c, titled.get(t, 0)) for t, c in seen.items()])

    def drop_source(self, sha: str, seen: dict | None, titled: dict | None) -> bool:
        """Remove one source's chunks; True when it had any. Without its proper counts the caller
        must `replace_proper` (the counts are subtracted, so they have to be the same ones)."""
        with self.write() as db:
            ids = [i for (i,) in db.execute("SELECT id FROM chunks WHERE sha=?", (sha,))]
            if not ids:
                return False
            for doc in ids:
                self._drop_doc(db, CHUNK, doc)
            db.execute("DELETE FROM chunks WHERE sha=?", (sha,))
            if seen is not None:
                rows = [(c, (titled or {}).get(t, 0), t) for t, c in seen.items()]
                db.executemany("UPDATE proper SET seen = seen - ?, titled = titled - ? WHERE term=?", rows)
                db.executemany("DELETE FROM proper WHERE term=? AND seen <= 0", [(t,) for t in seen])
            return True

    def replace_proper(self, seen: dict, titled: dict) -> None:
        """Overwrite the proper-noun counts (a recount after a source went without its cache)."""
        with self.write() as db:
            db.execute("DELETE FROM proper")
            db.executemany("INSERT INTO proper VALUES (?, ?, ?)", [(t, c, titled.get(t, 0)) for t, c in seen.items()])

    def source_shas(self) -> dict:
        """sha -> raw path of every source with chunks."""
        with self.lock:
            return dict(self.conn.execute("SELECT DISTINCT sha, raw FROM chunks"))

    def chunk_meta(self) -> dict:
        """id -> (source, raw, title, n, total, sha), memoised until the next write."""
        def build():
            with self.lock:
                return {i: (s, r, t, n, tot, h) for i, s, r, t, n, tot, h in self.conn.execute(
                    "SELECT id, source, raw, title, n, total, sha FROM chunks")}
        return self.cached("chunk_meta", build)

    def chunk_lengths(self) -> dict:
        """chunk id -> token count of every indexed chunk."""
        def build():
            with self.lock:
                return dict(self.conn.execute("SELECT id, length FROM chunks"))
        return self.cached("chunk_lengths", build)

    def proper_ratios(self, terms) -> dict:
        """term -> titled / seen for the `terms` the sources contain (absent: never seen)."""
        terms = list(terms)
        if not terms:
            return {}
        with self.lock:
            rows = self.conn.execute(f"SELECT term, seen, titled FROM proper WHERE term IN "
                                     f"({','.join('?' * len(terms))})", terms).fetchall()
        return {t: titled / seen for t, seen, titled in rows if seen}

    # --- scoring --------------------------------------------------------------------

    def postings(self, corpus: str, term: str) -> list:
        """(doc, term frequency) rows of one term in a corpus."""
        with self.lock:
            return self.conn.execute("SELECT doc, tf FROM postings WHERE corpus=? AND term=?",
                                     (corpus, term)).fetchall()

    def df(self, corpus: str, term: str) -> int:
        """Documents holding `term`, from `terms` (kept equal to the postings count per change)."""
        with self.lock:
            row = self.conn.execute("SELECT df FROM terms WHERE corpus=? AND term=?", (corpus, term)).fetchone()
        return row[0] if row else 0

    def positions(self, corpus: str, term: str, doc) -> list:
        """Token positions of a term in one chunk."""
        with self.lock:
            row = self.conn.execute("SELECT positions FROM postings WHERE corpus=? AND term=? AND doc=?",
                                    (corpus, term, doc)).fetchone()
        return _positions(row[0]) if row else []

    def docs_with_all(self, corpus: str, terms) -> set:
        """The documents holding every one of `terms`."""
        found = None
        for term in terms:
            docs = {d for d, _ in self.postings(corpus, term)}
            found = docs if found is None else found & docs
            if not found:
                return set()
        return found or set()

    def bm25(self, corpus: str, terms, weights: dict | None = None, subset=None) -> dict:
        """doc -> (score, matched terms) for the documents holding ≥ 1 of `terms`.

        The formula of the in-memory scorer it replaced (PRD 32), term for term and in the same
        order, so scores agree to the last bit: N and avgdl from the document lengths, DF from the
        postings — over `subset` when given (the named sources score as their own corpus).
        """
        weights = weights or {}
        lengths = self.page_lengths() if corpus == PAGE else self.chunk_lengths()
        if subset is not None:
            subset = set(subset)
            n = len(subset)
            total = sum(lengths[d] for d in subset if d in lengths)
        else:
            n = len(lengths)
            total = sum(lengths.values())
        if not n:
            return {}
        avgdl = total / n
        freqs, df = {}, {}
        for term in terms:
            rows = self.postings(corpus, term)
            if subset is not None:
                rows = [(d, tf) for d, tf in rows if d in subset]
                df[term] = len(rows)
            else:
                df[term] = self.df(corpus, term)
            freqs[term] = dict(rows)
        out = {}
        for doc in set().union(*freqs.values()) if freqs else ():
            dl, score, matched = lengths.get(doc, 0) or 1, 0.0, set()
            for term in terms:
                freq = freqs[term].get(doc, 0)
                if not freq:
                    continue
                score += bm25_term(weights.get(term, 1.0), freq, dl, avgdl, df[term], n)
                matched.add(term)
            out[doc] = (score, matched)
        return out

    # --- inspection -----------------------------------------------------------------

    def integrity(self) -> str:
        """SQLite's integrity check result ("ok" when sound)."""
        with self.lock:
            return self.conn.execute("PRAGMA integrity_check").fetchone()[0]

    def dump(self) -> dict:
        """Every row, canonical: chunk ids resolved to (sha, n), volatile meta left out — two
        files that index the same vault dump equal however they got there."""
        with self.lock:
            ids = {i: (sha, n) for i, sha, n in self.conn.execute("SELECT id, sha, n FROM chunks")}
            out = {
                "pages": sorted(self.conn.execute("SELECT * FROM pages")),
                "chunks": sorted(self.conn.execute(
                    "SELECT source, raw, sha, title, n, total, text_offset, length FROM chunks")),
                "terms": sorted(self.conn.execute("SELECT * FROM terms")),
                "postings": sorted((c, t, ids[d] if c == CHUNK else d, tf, tuple(_positions(p)))
                                   for c, t, d, tf, p in self.conn.execute("SELECT * FROM postings")),
                "proper": sorted(self.conn.execute("SELECT * FROM proper")),
                "meta": sorted((k, v) for k, v in self.conn.execute("SELECT * FROM meta")
                               if k not in VOLATILE_META),
            }
        return out


# --- the open handle ----------------------------------------------------------------
# One vault per process (tests swap WIKI_PATH per test), so one cached handle: opening another
# vault closes the previous one, and a file deleted under us is reopened (and rebuilt).

_open: dict = {"index": None}
_open_lock = threading.Lock()


def open_index(vault: str) -> Index:
    """The process-wide handle on the vault's index (reopened when the file was swapped)."""
    path = db_path(vault)
    with _open_lock:
        current = _open["index"]
        if current is not None and current.path == path and current.file_id == _file_id(path):
            return current                         # same file: not deleted, not swapped by a reindex
        if current is not None:
            current.close()
        _open["index"] = Index(path)
        return _open["index"]


def close() -> None:
    """Close and forget the open handle (tests, and before a file swap)."""
    """Close the cached handle (the next `open_index` reopens; tests call this between vaults)."""
    with _open_lock:
        if _open["index"] is not None:
            _open["index"].close()
            _open["index"] = None


def replace(vault: str, built: str) -> None:
    """Swap a freshly built file in for the vault's index (`reindex`)."""
    path = db_path(vault)
    with _open_lock:
        if _open["index"] is not None and _open["index"].path == path:
            _open["index"].close()
            _open["index"] = None
        for suffix in ("-wal", "-shm"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(path + suffix)
        os.replace(built, path)
