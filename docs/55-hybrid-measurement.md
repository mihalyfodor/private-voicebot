Status: done

# 55 — Measured hybrid retrieval (BM25 + embeddings), decide with numbers

## Overview
Five PRDs turned embeddings down on reasons that were true when written: "no embeddings at personal scale" (PRD 17,
`docs/17-long-term-memory.md:22`), "BM25 over embeddings: personal scale, no new model, deterministic tests" (PRD 19,
`docs/19-wiki-layer.md:85`), "BM25 chunks over embeddings: no new model/service, deterministic, testable; FAQs are
keyword-dense" (PRD 32, `docs/32-source-recall.md:47`), the PRD 34 precision rule (≥ 2 matched words or one proper
term, `docs/34-grounded-answers.md:35`), and "Embeddings or a re-ranker" out of scope (PRD 42 `:160`, PRD 45 `:109`).
Two changed: the vault now holds EPUBs (6 454 chunks in the dev vault) whose prose is not keyword-dense, and an embed
server already runs on the host. This PRD measures first and adopts only if the numbers clear a bar set in advance.

## User stories
- As the developer, I run one script and get a table: BM25 vs embeddings vs hybrid on hit@1/hit@2, precision
  violations and added latency, on the dev vault and a large synthetic vault.
- As the developer, if hybrid does not clear the bar, the PRD closes with the table in `docs/reviews/` and nobody
  re-litigates embeddings without new evidence.
- As a user (phase 2 only), "what does the microservices book say about splitting the database?" finds the
  database-per-service passage (A10 today) without me using the book's wording; chit-chat stays chit-chat.

## UI/UX
None. Phase 1 is an offline script; phase 2 changes no screen, message or setting.

## Technical approach
- **Embed server (probed 2026-10-16):** `~/llm.sh embed` → `llama-server -m ~/models/qwen3-embedding-0.6b/
  Qwen3-Embedding-0.6B-Q8_0.gguf --embedding --pooling last --host 127.0.0.1 --port 8082`; `/health` ok;
  `/v1/models` reports `n_embd 1024`, `n_ctx 32768`. Contract (OpenAI-compatible): `POST /v1/embeddings
  {input: str | [str]}` → `{data: [{embedding: [1024 floats]}], usage}`. One short query: 5–6 ms warm, 29–33 ms
  cold; a 992-token chunk: 141 ms. Queries get the Qwen3 instruction prefix (`Instruct: Given a question, retrieve
  passages that answer it\nQuery: …`), chunks none; vectors L2-normalised client-side.
- **Phase 1 (S) — `scripts/retrieval_bench.py`**, offline, never imported by the app:
  - Corpora: a copy of the dev vault (`~/llm-wiki-dev`, read-only copy as `tests/qa/stress.py:62` `seed` does) and a
    synthetic vault from PRD 53's `tests/perf/make_vault.py` (marker `perf`, run with its new `--prose` option:
    invented-name paragraphs, no FAQ keywords) at ≥ 10× the dev chunk count.
  - Embeds every chunk of `wiki._chunk_corpus()` (`wiki.py:1072`) once, batched 32, into `raw/.index/emb-<model>.f16.npy`
    + `emb-<model>.json` (row → `{sha, n}`), keyed by raw sha like the chunk cache (`wiki.py:1036`, gitignored per
    `wiki.py:211`); a re-run embeds only rows whose sha is new. Since PRD 53's SQLite index is the base (55 depends on 53), the same rows
    become a `chunk_vec(sha, n, model, vec BLOB)` table beside its chunk table; the matrix stays the query form.
  - Runs the question set three ways: **BM25** = today's `wiki.search_chunks` (`wiki.py:1202`) + `_recall_hits`
    rules (`llm.py:965`); **embeddings** = cosine top-k (`q @ M`, numpy); **hybrid** = reciprocal rank fusion
    `score(d) = Σ 1/(60 + rank_i(d))` over the BM25 top-20 and embedding top-20, then the same precision gate.
  - Reports per corpus: hit@1/hit@2 per question group, precision violations (an emitted hit that `_recall_hits`'
    rules — two matched topic words or `wiki.proper_match` (`wiki.py:1229`) — would reject), hard negatives with any
    kept hit, and added query latency p50/p95 (one query embed call + the dot product) measured serial (worst case) and parallel
    (embed alongside BM25) with the chat model loaded, as a markdown table on stdout. Hybrid is reported gated and
    relaxed (cosine ≥ τ counts as one matched word; information only).
  - Question set `eval/retrieval_questions.json`: the battery's A1–A14 and C1–C4 (`tests/qa/stress.py:283`, `:320`)
    with hand-labelled expected `(raw, part)` sets (from the battery's expectation strings, checked against the raw files once, committed as the JSON); 30 paraphrases of them that avoid the source's words; 20 hard negatives
    (chit-chat carrying corpus names: the PRD 45 band cases "Are you listening, Uncle?", "Vivec is annoying", "How
    long should I steep green tea?" (`docs/45-ask-before-lookup.md:87`), "what should we cook", Ada vs Ada Lovelace).
  - **Adoption bar** (judged on the gated hybrid only; all three, dev vault and synthetic): hybrid paraphrase hit@2 ≥ BM25 + 15 points; zero new
    precision violations on the hard negatives; added p95 ≤ 30 ms. Otherwise `docs/reviews/<date>-hybrid-retrieval.md`
    records the table and this PRD is set `done` as "measured, not adopted".
- **Phase 2 (M, only if adopted):** hybrid inside `search_chunks` with the precision gate unchanged — embeddings may
  re-rank BM25 candidates or add candidates, and every candidate still passes the PRD 34/42 rules and the PRD 45
  band; embed at ingest incrementally (new sha → embed its chunks in the ingest worker, drop on `drop_raw_chunks`);
  `:8082` down or slow (> 200 ms) → BM25 only, one `trace` event, no user-visible error; the spoken path makes at
  most one query embedding per turn (cached for the held-reply retry). Never a vector DB service.

### Decisions
- Measured, not adopted (2026-10-17): gated hybrid paraphrase hit@2 +7 points on the dev vault (bar +15), +0 on the synthetic one; two new negatives keep a hit on the dev vault; added p95 14 ms serial / 6 ms parallel. Table and verdict: `docs/reviews/2026-10-17-hybrid-retrieval.md`. Phase 2 is not built.
- Built as `eval/retrieval_bench.py` (run `python -m eval.retrieval_bench`) with tests in `tests/eval/`, beside the labelled set, rather than `scripts/` and `tests/sources/`; it caches the embedding matrix under a work dir in /tmp (text-hash keyed), not `raw/.index/`, which is phase 2 only.
- Measure before building over adopting on taste: five PRDs said no; only a number set in advance can say yes.
- RRF (k = 60) over weighted score blending: BM25 scores and cosines are not on one scale, RRF needs no tuning.
- numpy float16 matrix over a vector DB: 6 454 × 1024 × 2 B ≈ 13 MB; a brute-force dot is a few ms on this box.
- Precision gate stays the arbiter: embeddings only re-rank or propose; PRD 45's band policy is untouched.
- Assume Qwen3-Embedding-0.6B (Q8) is good enough to decide; if hybrid misses the bar by < 5 points, try one
  larger embedding model before closing, never more.
- Chose a `--prose` option on PRD 53's `make_vault.py` over a second generator: one generator, realistic prose.
- Chose reporting a relaxed hybrid for information and adopting on the gated one: the precision rule cannot be
  bypassed.
- Chose hand-labelled expectations committed as JSON over generated labels.
- Chose measuring latency serial and parallel over one number: serial is the worst case, parallel the real one.
- Chose measuring with the chat model loaded (embed server on :8082, chat on :8080/:8000) over an idle box: that is
  the real condition.
- Risk: the 4-bit chat model, not retrieval, causes the A10/A11 misses; mitigation: hit@k is measured on retrieval
  alone, so the answer model cannot blur the result.

## Data model
- `raw/.index/emb-<model>.f16.npy`: `float16[N, 1024]`, rows L2-normalised; `emb-<model>.json`:
  `{model, dim, rows: [{sha, n}]}`. Invariant: every row's sha is in the manifest; a sha leaving the manifest
  drops its rows on the next build; the model name is part of the key (a new model never mixes vectors).
- `eval/retrieval_questions.json`: `[{id, group: battery|paraphrase|negative, text, expect: [[raw, part], …]}]`;
  negatives have `expect: []`.
- Bench result row: `{corpus, method, group, hit1, hit2, violations, negatives_hit, p50_ms, p95_ms}`.

## Test scenarios
New `tests/sources/test_retrieval_bench.py`; a `fake_embed` fixture serves `/v1/embeddings` on a free port with
deterministic vectors (hash-seeded, with planted near-duplicates for the paraphrase cases).
1. Bench on the `fallout`/`both` fixture vault with the fake server → exit 0, stdout table has one row per corpus ×
   method × group, with hit@1/hit@2/violations/p95 columns filled.
2. RRF golden: BM25 order `[a, b, c]`, embedding order `[c, a, d]` → fused order `[a, c, b, d]`, `a` scoring
   `1/61 + 1/62`.
3. Precision gate: a fake embedding ranks a chunk sharing one common word with "what should we cook" first → the
   ungated embeddings row counts 1 violation; the hybrid row keeps no hit for it.
4. Embed server down (port closed) → the bench prints BM25 rows and "embeddings unavailable", exit 0; (phase 2)
   `search_chunks` returns exactly today's BM25 hits and one `trace` event is written.
5. Cache by sha: second run makes zero `/v1/embeddings` chunk calls; adding one raw file embeds only its chunks;
   removing it drops its rows.
6. Latency: the fake server sleeps 10 ms → reported p50 ≥ 10 ms and p95 ≥ p50; the dot product is timed separately.
7. (phase 2) Battery: `tests/qa/stress.py` sections A–F at ≥ 37 correct of 45 (the PRD 43 result,
   `docs/43-written-answers.md:96`), no new wrong answers, and the PRD 45 band table unchanged.

## Out of scope
Rerankers (cross-encoder or LLM), fine-tuning, web search, changing the band policy, `RECALL_MIN_SCORE` or
`RECALL_HIGH_SCORE`, embedding wiki pages (pages join recall by name, PRD 42), a vector DB or new service.

## Open design questions
None open.
