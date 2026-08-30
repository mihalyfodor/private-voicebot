Status: implementation

# 32 — Search inside sources (recall from big documents)

## Overview
A filed document is stored whole and summarised, but chat can only reach the summary page: `search_wiki` indexes
pages, `read_page` returns the first 4 kB, and the local model rarely calls either for a natural question. A 400 k-token
game guide is therefore useless as knowledge. This PRD indexes the raw text of text-kind sources in chunks, returns the
best chunks from search, and injects the top hits into a turn automatically when the user's message matches well —
so recall does not depend on the model choosing to call a tool. FAQs are the calibration corpus.

## User stories
- As a user, I ask "I'm at Covenant and they shoot me" after filing the Fallout 4 guide and get the guide's own
  walkthrough of that quest, quoted from the right section, not the model's guess.
- As a user, "what do I have on Fallout" still leads with the summary page, and "where's the Covenant section" lets
  me read the neighbouring text.
- As the developer, dropping more FAQs makes recall better, not slower: indexing is cached per raw sha and search stays
  BM25 (no embeddings, no new service).

## Technical approach
- `wiki.raw_chunks(raw_rel)`: split a text-kind raw file (md/txt/docx-extracted/pptx/xlsx/vtt — use the existing
  `sources.extract` text) into ~800-token chunks on paragraph/heading boundaries with 80-token overlap; cached in
  `raw/.index/<sha>.json` (gitignored inside the vault) keyed by sha, invalidated by sha; built lazily on first search
  after an ingest, and eagerly right after `file_bytes` filing succeeds.
- `wiki.search` keeps its page results and adds `chunks: [{source, raw, n, text, score}]` from a second BM25 index
  over all chunks (same tokeniser; corpus cached with the manifest's shas as signature). Pages and chunks are ranked
  separately; the tool text shows page hits first, then up to 3 chunk snippets (≤ 600 chars each) titled
  `<source title> · part n/N`.
- New tool `read_source(path, part)`: the chunk text plus one neighbour either side (≤ 3 k chars), so the model can
  keep reading a walkthrough.
- **Auto-recall**: `llm.recall_for(message)` runs before each turn (typed or spoken; skipped for messages < 4 words
  and for pure commands like "what time is it"): BM25 over chunks + pages with the user text; hits above a fixed
  score threshold (calibrated in tests against the fixture FAQ) yield a `<recall>` block appended to the *user* message
  for that turn only: `From "<source title>" (part n/N): <chunk text>` ×≤2 (≤ 1500 tokens total), with the instruction
  "If this answers the question, use it and say which source it came from; otherwise ignore it." The block is never
  persisted in the conversation history beyond that turn's user message (kept as-is for context, capped).
- Threshold: BM25 score normalised by query length; below it nothing is injected (chit-chat gets no recall). Setting
  `recall_min_score` in settings.json (default from the calibration), shown in Settings next to the calendar horizon.
- Filler while retrieving is not needed (BM25 over ~500 chunks is milliseconds).

### Decisions
- BM25 chunks over embeddings: no new model/service, deterministic, testable; FAQs are keyword-dense.
- Auto-recall in the user message over a system-prompt block: it is per-turn, and the model treats it as evidence.
- Chunk cache inside the vault (`raw/.index/`, gitignored) over a repo-side cache: it lives with the data it derives from.
- Assume ≤ 50 sources / ≤ 20 k chunks; if false, move the chunk index to sqlite FTS.
- Risk: recall injects an irrelevant chunk into a work question; mitigation: threshold + "ignore if not relevant" line,
  and the calibration test with off-topic queries.

## Data model
- `raw/.index/<sha>.json`: `{"sha", "raw", "source", "title", "chunks": [{"n", "start", "text"}]}`.
- `wiki.search()` return gains `chunks`; `search_wiki` tool text gains snippets; new `read_source` tool.
- settings.json: `recall_min_score` (float).

## Test scenarios (fixture: a generated ~150 k-token FAQ-shaped text with named sections incl. "Covenant", plus a
   few unrelated wiki pages)
1. `raw_chunks` splits on headings/paragraphs, ~800 tokens, overlap present, count deterministic; cache file written
   once and reused (second call reads no raw bytes — assert via a patched `_read`).
2. `wiki.search("covenant compound turrets")` → top chunk is from the Covenant section; page hits unchanged for
   "anna" (people page first, no chunks).
3. `search_wiki` tool text lists page hits then ≤ 3 snippets with `part n/N`; `read_source(path, n)` returns the
   chunk with neighbours, ≤ 3 k chars; bad path/part → clear error string.
4. `llm.recall_for("I'm at covenant and the turrets keep shooting me")` → one `<recall>` block from the Covenant
   chunk; `recall_for("hi")`, `recall_for("what time is it")`, `recall_for("what should we cook tonight")` → "".
5. Calibration: 10 in-guide questions (generated from section titles) hit their own section ≥ 8/10; 10 off-topic
   questions inject nothing — with the default `recall_min_score`.
6. The recall block is attached to that turn's user message only; the system prompt is unchanged; the history entry
   is capped so a long recall does not bloat the context (assert token count).
7. Re-filing the same sha builds no new cache; `forget` removes the cache file.
8. Live (marker `live`): with the real Fallout 4 guide in a temp vault copy, ask 5 Covenant/quest questions through
   `llm.chat`; each answer names the guide as its source and contains a term from the matching chunk. Printed for
   the developer to judge; asserts only the source mention.
9. Manual: ask Wanko about Covenant in the app; add a second FAQ and repeat.

## Out of scope
Embeddings/vector search, cross-source synthesis, re-ranking with the LLM, indexing images/PDF beyond the text pypdf
already extracts, indexing the profile.
