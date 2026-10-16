Status: exploration

# 53 — Scalable index: thousands of pages and sources without losing voice latency

## Overview
Search today rebuilds its world on every call: `wiki.search` walks and stats the whole vault, `search_chunks`
re-reads the manifest several times and re-scores every chunk with a fresh `Counter`, and every prompt build reads
every page for `index_text`. That is fine at 50 sources (the PRD 32 assumption) and breaks a spoken turn at 5k pages
/ 1k sources. This PRD persists an inverted index in one SQLite file inside the vault, updates it on every write
path, and keeps scores, recall rules and the prompt index byte-for-byte what they are today.

## User stories
- As a user with thousands of pages and ~1k filed books, a spoken question is answered as fast as with ten.
- As a user, filing one more source or saving a page makes it searchable on the next turn, without a rebuild pause.
- As a user, an edit I make in Obsidian is still picked up (as today, `_corpus` docstring, wiki.py:850).
- As a user, `lint` no longer tells me 50 pages are "missing from index.md" just because there are 250 of them.
- As a developer, `python -m wiki reindex` (or `wiki.reindex()`) recovers a broken or deleted index in one call.

## UI/UX
None. No new screen, message or setting; `index.md` stays a human-readable view.

## Technical approach
**Today's costs (verified, wiki.py unless noted):**
- `_signature()` (844-846) runs `page_paths()` (342-362: `os.walk` + a read of `decisions.md` per call) and one
  `getmtime` per page, on every `search` (917) and every `backlinks` (867) call.
- `_bm25` (890-908): per query, `Counter(...)` over all docs with `term in tokens` (a list scan, 896) for DF, then a
  fresh `Counter(tokens)` per doc (899). O(corpus tokens) per query.
- `_chunk_corpus` (1072-1090): the signature is every manifest sha; any change re-tokenises every chunk of every
  source (1086) and recomputes `_proper_ratios` (1102-1113) over all chunk text (1088). It is called three times per
  `search_chunks` (1212, via `_title_terms` 1150 and `proper_phrases` 1176) plus `proper_terms` (1129) from
  `recall_for`; each call re-reads and parses `raw/manifest.jsonl` (`sources.manifest_entries`, sources.py:745-756).
- `raw_chunks` (1036-1062) calls `manifest_entries()` again per entry (1045): O(sources²) manifest parsing on a
  rebuild; `_source_title` (1027-1029) reads the source page for each.
- `split_title_terms` (1153-1168) walks all chunks per query (`_title_terms`) and compares each title to every other.
- `index_lines` (387-393) reads and parses every page per call, alphabetical by path, cut at `INDEX_MAX_LINES = 200`
  (69); `index_text` (404-418) then pops lines to `WIKI_INDEX_TOKENS` (600, line 67). Called per prompt build
  (memory.py:616) and per ingest (wiki.py:1719, sources.py:1101).
- **Bug:** `lint` (1578) builds `indexed` from `links_in(index.md)` (1592) and flags every page not in it (1600-1601);
  `index.md` holds at most 200 lines, so page 201+ is always "missing from index.md".
- Call sites: `search` — `llm._page_recall` (llm.py:991), `related_context` (wiki.py:427), chatbot.py:546,
  tools/wiki.py:85; `search_chunks` — `llm._recall_hits` (llm.py:973), tools/wiki.py:86.
- Chunk cache: `raw/.index/<sha>.json` (`RAW_INDEX_DIR`, 49; `_cache_rel` 1032), gitignored by `ensure_vault`
  (209-211), dropped by `drop_raw_chunks` (1065) from `forget` (sources.py:1436).

**Design:**
- `index_db.py` (stdlib `sqlite3`): one file `<vault>/.index/search.sqlite`, WAL mode, every update in one
  transaction. `page_paths` already skips dot-dirs (348), so the file is never a page; `ensure_vault` adds `.index/`
  to `.gitignore` next to the `raw/.index/` line.
- Tokens come from the existing `_tokenise`/`_stem`; the DB stores postings, not a second tokeniser.
- **Write hooks**, one call each, after the file write: `_write` of a page (158, covers `_edit_page`, `save_page`,
  `create_page`, `save_card`, ingest ops), `rename_page` (769), `delete_page` (807), `sources.forget`
  (sources.py:1425-1437, with `drop_raw_chunks`), filing a source (sources.py:1299-1304) and each batch-import item
  (batch_import.py `_one`, 121). Pages key on (mtime, sha); sources on manifest sha.
- **Obsidian edits:** replace the per-search `os.walk` with a cheap staleness check: the mtimes of the vault's
  folders (an add, delete or rename changes them) on every search, plus a full file-mtime sweep at most every
  `INDEX_SWEEP_S = 5` s; a stale page is re-indexed alone. Parsed frontmatter + one-liner live in the `pages` table.
- **Scoring:** `_bm25` keeps its formula and `_K1, _B = 1.5, 0.75` (825), fed from postings: N and avgdl from
  `doc_len`, DF from `terms` (or from postings restricted to the named sources, because `search_chunks` scores the
  filtered subset, 1218-1219, so its N/avgdl/DF are the subset's). Only docs holding ≥ 1 query term are touched.
- **Rules unchanged** (search_chunks 1202-1226, llm.py): title words never score (`split_title_terms`); `GENERIC_WORDS`
  dropped; score = BM25 / topic words; `PROPER_BOOST = 2.0` weights from stored proper ratios; `matched`,
  `phrases` (adjacency from stored token positions or the chunk text), `named`; the floor `recall_min_score()` (4.5);
  a hit needs ≥ 2 matched words or `proper_match` (llm.py:980, PRD 34/42); `RECALL_MAX_CHUNKS = 2` (llm.py:490);
  proper bigrams must match whole (PRD 42 §5).
- `index_text`/`index_lines` read `pages` (path, one-liner) in path order: same lines, no page reads. The map (PRD 54)
  replaces this at the prompt; 53 only makes it cheap. `index.md`
  stays capped at 200 lines; `lint` checks the `pages` table, not `index.md`.
- `search(query, k, filters=None)`, `search_chunks(query, k, filters=None)`: optional `{type, tags, updated_from,
  updated_to}` (pages) / `{source, kind}` (chunks). No caller passes them in this PRD.

### Decisions
- Chose SQLite over an in-memory index rebuilt at start because 15k chunks re-tokenise in seconds at every app start
  and batch import runs alongside the app; WAL gives one consistent file for both.
- Chose our scorer over FTS5 `bm25()` because FTS5's idf (no `1 +`), fixed k1 = 1.2 and column-only weights cannot
  reproduce per-term proper weights, subset statistics or `_stem`; parity with PRDs 32/34/42 matters more than code.
- Chose `lint` against the table over an uncapped `index.md` because `index.md` is a human view and 5k lines help no one.
- Assume one indexed term fetch per query term stays < 50 ms p95 at 15k chunks; if false, cache hot postings in memory.
- Risk: a write path misses its hook and search goes stale; mitigation: the 5 s mtime sweep, a sha check on the
  manifest, and `reindex` from lint when the table and disk disagree.
- Keep `raw/.index/<sha>.json` as the chunk text store (chunking stays per sha); the DB indexes it.

## Data model
```
pages(path PK, mtime, sha, type, name, tags, updated, one_liner, length)
chunks(id PK, source, raw, sha, title, n, total, text_offset, length)        -- text stays in raw/.index/<sha>.json
terms(corpus, term, df, PRIMARY KEY(corpus, term))                           -- corpus: 'page' | 'chunk'
postings(corpus, term, doc, tf, positions)  INDEX(corpus, term)              -- doc = pages.path | chunks.id
proper(term PK, seen, titled)                                                -- ratio = titled / seen
meta(key PK, value)                                                          -- schema version, manifest sha, last sweep
```
Invariants: one transaction per document change; `terms.df` = count of postings rows for (corpus, term);
`doc_len` sums give avgdl; a document absent from disk / manifest has no rows; `reindex` from scratch equals the
incrementally reached state (row-for-row).

## Test scenarios
1. **Golden scores** (`tests/wiki/test_index_db.py`): on the conftest fixture vault, `search` and `search_chunks`
   for 20 battery queries return the same paths/chunks, order and `score` (to 3 dp) as today's `_bm25` path
   (snapshot recorded before the change).
2. **Incremental ingest** (same file): filing one source changes postings only for that sha's chunks (row diff
   before/after); the new chunk is the top hit for its rare word on the next call.
3. **Remove paths**: after `rename_page`, `delete_page` and `sources.forget`, the old path / sha has zero rows in
   `pages`, `chunks`, `postings`; the renamed page is found under its new path; `terms.df` dropped by one.
4. **Rebuild equals incremental**: ten mixed writes, then `reindex()` into a second file; both DBs dump identical.
5. **Lint regression** (`tests/wiki/test_wiki.py`): a 250-page vault yields no "missing from index.md" finding;
   `index.md` still has 200 lines; a page on disk the table lacks is reported.
6. **`index_text` golden** (`tests/test_index.py`): for the fixture vault the string equals today's, byte for byte;
   it performs no page reads (patched `_read` counts 0).
7. **Obsidian edit**: an external write (bumped mtime, no hook) is reflected within `INDEX_SWEEP_S`; a new file in
   `people/` is found on the next search (folder mtime changed).
8. **Filters**: `search("anna", filters={"type": "person"})` returns only person pages; `updated_from` excludes older.
9. **Cache invalidation + crash** (`tests/wiki/test_index_db.py`): a changed sha re-indexes that source alone; a
   killed update (exception raised mid-transaction) leaves the prior state intact and `PRAGMA integrity_check` ok.
10. **Perf gates** (`tests/perf/test_index_perf.py`, marker `perf`, deselected in `pytest.ini` addopts): a generated
    vault (this is the shared synthetic-vault generator; PRD 54 and PRD 55 reuse it; `tests/perf/make_vault.py`, seeded: 5k pages, 1k sources ≈ 15k chunks) gives p95 `search_chunks` < 50 ms,
    p95 `search` < 20 ms over 200 queries, one-source incremental update < 200 ms, on the M5 Max. Plus the battery
    (`tests/qa/stress.py`) correct/partial/wrong totals equal the PRD 43 run.

## Out of scope
Embeddings (PRD 55), the vault map (PRD 54), FTS5, any app.js / UI change, new ranking rules or score recalibration,
chunking changes, a file watcher (fsevents) daemon.

## Open design questions
- **FTS5 or our scorer?** Recommend our scorer over plain tables (golden-score parity, proper weights, subset
  stats); revisit FTS5 only if postings fetch misses the 50 ms gate.
- **Where does the file live?** Recommend `<vault>/.index/search.sqlite` (pages and chunks together) over
  `raw/.index/`, which is source-scoped; one more `.gitignore` line.
- **Staleness detection for Obsidian edits:** recommend folder mtimes per search + a 5 s throttled sweep; fsevents
  only if the sweep shows up in p95.
- **Chunk text in DB or JSON?** Recommend keeping `raw/.index/<sha>.json` for text (no migration), DB for postings.
- **Prompt index at scale:** `index_text` is alphabetical and 600 tokens covers ~15 pages of 5k; recommend keeping it
  identical here and letting PRD 54 (the map replaces `index_text` in the prompt; 53 only makes it cheap) decide what the prompt sees.
