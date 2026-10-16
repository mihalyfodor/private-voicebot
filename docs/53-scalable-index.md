Status: refactoring

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
- As a user, an edit I make in Wanko's editor is searchable on the next turn, and after a crash or an edit made while
  the app was down the index repairs itself at startup.
- As a user, `lint` no longer tells me 50 pages are "missing from index.md" just because there are 250 of them.
- As a developer, `python -m wiki reindex` (or `wiki.reindex()`, the `rebuild` command) recovers a broken or deleted index in one call.

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
- **Consistency:** the app's write paths are the only writers (all edits go through Wanko's editor and ingest paths;
  external edits are out of scope), so there is no per-search mtime sweep. A startup consistency check compares the
  stored shas/mtimes to the files on disk and the manifest and repairs what differs; `rebuild` (`reindex`) is the
  safety net. Parsed frontmatter + one-liner live in the `pages` table.
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
- Risk: a write path misses its hook and search goes stale; mitigation: the startup consistency check, a sha check on
  the manifest, and `reindex` from lint when the table and disk disagree.
- Chose `<vault>/.index/search.sqlite` (pages and chunks together) over `raw/.index/`, which is source-scoped; one more
  `.gitignore` line.
- Chose our scorer over plain tables with FTS5 only if the 50 ms gate fails (see above): golden-score parity first.
- Chose to keep chunk text in `raw/.index/<sha>.json` over moving it into the DB: no migration, the DB holds postings.
- Chose to leave the prompt `index_text` identical here over changing what the prompt sees: PRD 54's map decides that;
  53 only makes it cheap.
- Chose no Obsidian support (the user does not use it) over staleness sweeps: no mtime sweeps, no fsevents.
- Chose `index_db.py` as storage + scorer only (stdlib `sqlite3`, no vault reads) over moving `_tokenise`/`_stem` into it:
  wiki.py tokenises and hands in tokens, so there is no import cycle and the tokeniser stays where it was.
- Chose to put scenario 6 in `tests/wiki/test_index_db.py` over `tests/test_index.py`, which is the docs/index.md test.
- Chose a lazy build over a mandatory startup step: the first index open per process runs the consistency check (a
  vault without a DB answers its first search by building one); the app's startup event and batch import also run it.
- Chose to stat `raw/manifest.jsonl` on every chunk-index read (`_chunk_index`: about 4–5 stats per `search_chunks` —
  weights, proper terms, title terms, phrases, the search — and one per `recall_for` proper-term lookup; re-hash and
  re-sync sources only when the stamp changed) as the "sha check on the manifest": a batch import in another process is
  searchable without a restart. Pages get no check.
- Chose to cover batch-import items with the filing hook in `sources.file_bytes` (which `_one` goes through) over a
  second hook in `batch_import._one`.
- Chose subset statistics for `filters` (as naming a source does); `tags` keeps pages holding every listed tag, `kind`
  resolves through the manifest because `chunks` has no kind column.
- Chose to keep the data model as written: `backlinks` reads only the pages whose postings hold every word of the target
  (no links column); chunk ids are integer rowids, and "rebuild equals incremental" compares `Index.dump()`, which
  resolves chunk ids to (sha, n) and leaves out per-file meta (epoch, generation, checked_at, manifest hash).
- Chose a `*` `.gitignore` inside `.index/` over a `.index/` line in the vault's tracked `.gitignore`: nothing would
  commit that edit, so every existing vault would go dirty (and batch import refuse it); the inner file keeps git clean
  from the first index open on (review round 1 dropped the outer line).
- Chose fail-fast for a failing index write (the hook raises after the page is on disk; the transaction rolls back) over
  swallowing it; the startup check repairs the row. Writes that move content write the new file before any hook runs
  (`rename_page`), so a failing hook never loses a page.
- Chose to point `WIKI_PATH` at a missing tmp path in tests/conftest.py before the app import: importing the app builds
  the prompt, which reads the index, before any fixture sandboxes the vault (a read now opens the DB).
- Accept that the first app start after the upgrade builds the index synchronously at import (the prompt build reads
  `index_text`); every later start only runs the stat check. Whatever imports the app against a vault it does not mean
  to index guards it: tests/conftest.py and eval/harness.py point `WIKI_PATH` at a missing tmp path first, and
  `python -m wiki reindex` resolves the vault from the environment, else the checkout's `.env` (dev → ~/llm-wiki-dev).
- Chose to refuse `python -m wiki reindex` while the app answers on its ports (exit 2, the batch-import guard
  `batch_import.running_app`) over swapping the file under it; `open_index` also reopens when the file's (device,
  inode) changed, so a swapped file is picked up instead of writing to an orphan.
- Chose to skip a source that cannot be chunked (raw file gone, extraction fails, no chunk cache) during a sync, logged
  and traced (`error`, scope `index`, where `sync_sources`, not fatal), over failing the index: one bad manifest line
  must not stop search, the prompt index, page writes or app start (whose check is also log-and-continue).
- Chose to read DF from `terms` for the whole corpus (the PRD data model; kept equal to the postings count per change,
  asserted by a test) and to count postings only for a subset (named sources, filters).
- Chose to name the fix in the finding ("… — run python -m wiki reindex") over repairing from `lint`: lint reports,
  the user decides; the startup check already repairs what a missed hook leaves.

## Data model
```
pages(path PK, mtime, sha, type, name, tags, updated, one_liner, length)
chunks(id PK, source, raw, sha, title, n, total, text_offset, length)        -- text stays in raw/.index/<sha>.json
terms(corpus, term, df, PRIMARY KEY(corpus, term))                           -- corpus: 'page' | 'chunk'
postings(corpus, term, doc, tf, positions)  INDEX(corpus, term)              -- doc = pages.path | chunks.id
proper(term PK, seen, titled)                                                -- ratio = titled / seen
meta(key PK, value)                                                          -- schema version, manifest sha, last startup check
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
7. **Startup check**: one page edited while the app was down (file changed, no hook ran) is repaired by the startup
   consistency check and found on the next search.
8. **Filters**: `search("anna", filters={"type": "person"})` returns only person pages; `updated_from` excludes older.
9. **Cache invalidation + crash** (`tests/wiki/test_index_db.py`): a changed sha re-indexes that source alone; a
   killed update (exception raised mid-transaction) leaves the prior state intact and `PRAGMA integrity_check` ok.
10. **Perf gates** (`tests/perf/test_index_perf.py`, marker `perf`, deselected in `pytest.ini` addopts): a generated
    vault (this is the shared synthetic-vault generator; PRD 54 and PRD 55 reuse it; `tests/perf/make_vault.py`, seeded: 5k pages, 1k sources ≈ 15k chunks) gives p95 `search_chunks` < 50 ms,
    p95 `search` < 20 ms over 200 queries, one-source incremental update < 200 ms, on the M5 Max. Plus the battery
    (`tests/qa/stress.py`) correct/partial/wrong totals equal the PRD 43 run.

## Out of scope
Embeddings (PRD 55), the vault map (PRD 54), FTS5, external (non-app) edits, per-search staleness sweeps, any app.js / UI change, new ranking rules or score recalibration,
chunking changes, a file watcher (fsevents) daemon.

## Open design questions
None open.
