# 2026-10-17 — Hybrid retrieval measurement (PRD 55 phase 1)

Verdict: **measured, not adopted.** The gated hybrid (BM25 + Qwen3-Embedding-0.6B Q8, RRF k = 60) misses the
adoption bar on the dev vault on two of three counts, and ties BM25 on the synthetic vault.

## How it was measured
`.venv/bin/python -m eval.retrieval_bench --scale-dot` (`eval/retrieval_bench.py`), embed server on :8082, chat model
loaded on :8000. Questions: `eval/retrieval_questions.json` minus the follow-ups A2/A6/C4: 13 battery, 30 paraphrase,
22 negative (the 20 hard negatives plus A13/A14, which expect nothing). A hit counts when any expected `(raw, part)`
is in the top-k of what the method emits (duplicates counted once). BM25 = `wiki.search_chunks` + the `_recall_hits`
gate (4 candidates, BM25 floor 4.5, two matched topic words or a proper term, no neighbours, at most 2 kept).
Rows: `embeddings` is the raw cosine top-2 with no gate (violations = questions where an emitted hit fails the
precision rule); `embeddings-gated`, `hybrid` (RRF of the BM25 and cosine top-20, then the same gate, BM25 floor kept
for chunks that have a BM25 score) and `hybrid-nofloor` / `hybrid-relaxed` (information only: floor dropped; cosine
>= 0.55 counts as one matched word). The adoption bar is judged on `hybrid`.

Corpora: the dev vault (a read-only copy of `~/llm-wiki-dev`, 6 454 chunks) and a synthetic prose vault from
`python -m tests.perf.make_vault --prose` (400 sources, 200 pages, 6 739 chunks, 12 planted facts each asked verbatim
and reworded, 6 chit-chat negatives; labels come from the generator, not from hand).

## Results
| corpus | method | group | n | hit@1 | hit@2 | violations | negatives hit | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| dev | bm25 | battery | 13 | 62% | 69% | 0 | - | 0.0 | 0.0 |
| dev | bm25 | paraphrase | 30 | 43% | 47% | 0 | - | 0.0 | 0.0 |
| dev | bm25 | negative | 22 | - | - | 0 | 15 | 0.0 | 0.0 |
| dev | embeddings | battery | 13 | 15% | 31% | 11 | - | 13.7 | 14.0 |
| dev | embeddings | paraphrase | 30 | 20% | 20% | 21 | - | 13.7 | 14.0 |
| dev | embeddings | negative | 22 | - | - | 19 | 22 | 13.7 | 14.0 |
| dev | embeddings-gated | battery | 13 | 23% | 38% | 0 | - | 13.7 | 14.0 |
| dev | embeddings-gated | paraphrase | 30 | 17% | 20% | 0 | - | 13.7 | 14.0 |
| dev | embeddings-gated | negative | 22 | - | - | 0 | 10 | 13.7 | 14.0 |
| dev | hybrid | battery | 13 | 38% | 54% | 0 | - | 13.7 | 14.0 |
| dev | hybrid | paraphrase | 30 | 43% | 53% | 0 | - | 13.7 | 14.0 |
| dev | hybrid | negative | 22 | - | - | 0 | 17 | 13.7 | 14.0 |
| dev | hybrid-nofloor | battery | 13 | 38% | 54% | 0 | - | 13.7 | 14.0 |
| dev | hybrid-nofloor | paraphrase | 30 | 47% | 57% | 0 | - | 13.7 | 14.0 |
| dev | hybrid-nofloor | negative | 22 | - | - | 0 | 19 | 13.7 | 14.0 |
| dev | hybrid-relaxed | battery | 13 | 38% | 54% | 1 | - | 13.7 | 14.0 |
| dev | hybrid-relaxed | paraphrase | 30 | 50% | 60% | 6 | - | 13.7 | 14.0 |
| dev | hybrid-relaxed | negative | 22 | - | - | 2 | 17 | 13.7 | 14.0 |
| synthetic | bm25 | battery | 12 | 92% | 92% | 0 | - | 0.0 | 0.0 |
| synthetic | bm25 | paraphrase | 12 | 83% | 83% | 0 | - | 0.0 | 0.0 |
| synthetic | bm25 | negative | 6 | - | - | 0 | 0 | 0.0 | 0.0 |
| synthetic | embeddings | battery | 12 | 25% | 33% | 10 | - | 13.3 | 13.9 |
| synthetic | embeddings | paraphrase | 12 | 8% | 33% | 11 | - | 13.3 | 13.9 |
| synthetic | embeddings | negative | 6 | - | - | 6 | 6 | 13.3 | 13.9 |
| synthetic | embeddings-gated | battery | 12 | 33% | 33% | 0 | - | 13.3 | 13.9 |
| synthetic | embeddings-gated | paraphrase | 12 | 42% | 42% | 0 | - | 13.3 | 13.9 |
| synthetic | embeddings-gated | negative | 6 | - | - | 0 | 0 | 13.3 | 13.9 |
| synthetic | hybrid | battery | 12 | 83% | 83% | 0 | - | 13.3 | 13.9 |
| synthetic | hybrid | paraphrase | 12 | 83% | 83% | 0 | - | 13.3 | 13.9 |
| synthetic | hybrid | negative | 6 | - | - | 0 | 0 | 13.3 | 13.9 |
| synthetic | hybrid-nofloor | battery | 12 | 83% | 83% | 0 | - | 13.3 | 13.9 |
| synthetic | hybrid-nofloor | paraphrase | 12 | 83% | 83% | 0 | - | 13.3 | 13.9 |
| synthetic | hybrid-nofloor | negative | 6 | - | - | 0 | 0 | 13.3 | 13.9 |
| synthetic | hybrid-relaxed | battery | 12 | 83% | 83% | 0 | - | 13.3 | 13.9 |
| synthetic | hybrid-relaxed | paraphrase | 12 | 83% | 83% | 0 | - | 13.3 | 13.9 |
| synthetic | hybrid-relaxed | negative | 6 | - | - | 0 | 0 | 13.3 | 13.9 |

Added latency over BM25 (ms): serial = embed then dot + fuse, parallel = embed alongside BM25.

| corpus | chunks | bm25 p50 | bm25 p95 | serial p50 | serial p95 | parallel p50 | parallel p95 | dot+fuse p50 |
|---|---|---|---|---|---|---|---|---|
| dev | 6454 | 1.1 | 5.4 | 13.7 | 14.0 | 5.2 | 6.0 | 0.48 |
| synthetic | 6739 | 0.2 | 0.2 | 13.3 | 13.9 | 6.1 | 6.5 | 0.50 |

tau (relaxed hybrid) = 0.55; candidates into the gate = 4; RRF k = 60, depth 20.
dot + top-k at 65 000 x 1024 (float32): 1.3 ms median

Adoption bar (gated hybrid):
- dev: paraphrase hit@2 gain +7 pts (bar >= +15: MISS); new negatives with a kept hit 2 ['N15', 'N16'] (bar 0: MISS); added p95 serial 14.0 / parallel 6.0 ms (bar <= 30: ok)
- synthetic: paraphrase hit@2 gain +0 pts (bar >= +15: MISS); new negatives with a kept hit 0 [] (bar 0: ok); added p95 serial 13.9 / parallel 6.5 ms (bar <= 30: ok)

## Verdict against the bar
Paraphrase hit@2 (hybrid vs BM25): dev 53% vs 47% (+7 points, bar +15, missed); synthetic 83% vs 83% (+0, missed).
New precision violations on the hard negatives: none by the rule itself (the gate admits nothing it would reject:
`violations` is 0 on every gated row), but on the dev vault the hybrid keeps a hit for two negatives BM25 does not
(N15 "picking the wrong code", N16 "That saga on TV never ends"), so the "zero new" bar is missed there; synthetic has
none. Added p95 latency: serial 14.0 ms dev / 13.9 ms synthetic, parallel 6.0 / 6.5 ms, both under 30 ms (a
65 000 x 1 024 float32 dot + top-k is 1.3 ms, so the 10x size is not a latency risk). The embedding model alone is
weak here: raw cosine finds the right part in the top-2 for only 20% of the dev paraphrases, and the precision gate
(correctly) removes the lexically unsupported rest, which leaves the hybrid a re-ranker over chunks BM25 already
found; the gain it gives (two paraphrases out of 30) is far from +15 points. The BM25 floor is not what blocks it:
`hybrid-nofloor` reaches 57% (+10), still short. Per the PRD the gap is more than 5 points, so no larger model is
tried; the PRD closes as measured, not adopted.

## Caveats
- The BM25 baseline itself keeps a hit for 15 of the 22 negatives. In the app most of these turns never reach recall
  (word-count and COMMAND_WORDS checks, the PRD 45 band), which this offline script does not model; the comparison
  that matters is hybrid vs BM25 on the same set.
- The synthetic vault is ~1x the dev vault, not the PRD's 10x (embedding 65 000 chunks would take hours at ~20
  chunks/s); the dot was timed separately at 65 000 rows. Its paraphrases keep the invented names (a question cannot
  avoid them), which is why BM25 already scores 83% there; the 12-fact set is tiny (one question = 8 points).
- A12 is a weak label (README-retrieval.md); the dev battery has 13 questions, so one question = 8 points.
- hybrid candidates: the gate is fed the top 4 of the fused list (what `_recall_hits` is given today).
