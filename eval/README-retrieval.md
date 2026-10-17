# retrieval_questions.json
Labelled set for scripts/retrieval_bench.py (docs/55-hybrid-measurement.md, phase 1).
- `battery`: A1-A14, C1-C4 from tests/qa/stress.py; `paraphrase` (30): reworded, same `expected` as `of`; `negative` (20): chit-chat with corpus names, `expected: []`.
- `expected` = `{raw, part}`: `raw` is the manifest `path` in ~/llm-wiki-dev; `part` is the 1-based chunk number `n` of `wiki.search_chunks` / `raw/.index/<sha>.json` (`chunks[part-1]`).
- Derived once by grepping each battery expectation string over the cached chunks of the dev vault; duplicates (repeated guide sections) and neighbour chunks are all listed.
- A13/A14 ("NOT covered") and the negatives have `expected: []`; A2/A6/C4 are follow-ups ("yes") that need conversation context.
- A12 is a weak label: the book only mentions CQRS in passing (part 36). Re-check if the vault is re-ingested (parts shift).
