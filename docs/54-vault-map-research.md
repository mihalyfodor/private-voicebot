Status: implementation

# 54 — Vault map and research mode

## Overview
The prompt and both ingest calls see the vault through `wiki.index_text` — every page in alphabetical order, cut
at 600 tokens (`WIKI_INDEX_TOKENS`, wiki.py:67, loop at wiki.py:404–418), so past ~60 pages the tail of the alphabet is
invisible and PRD 44's `related_context` is the only thing naming it. This PRD replaces that list with a generated
**vault map** (folders → their most-used pages, same 600-token cap, but representative), and adds **research mode**:
an explicit typed command that runs a fixed, budgeted chain (plan → search → judge → read → write → verify) over the
vault and its sources and ends in a card with verified quotes. Spoken recall (PRD 45) and the band are untouched.

## User stories
- As a user with 300+ pages, I ask about Zephyr and Wanko knows a `projects/zephyr` page exists, though "z" sorts last.
- As the ingest, filing a Zephyr source, I update `projects/zephyr` rather than create a duplicate.
- As a user, I type "research how the Falcon and Zephyr budgets differ" and get a card with `## Findings`,
  `## Sources`, and quotes I can trust, because every quote was checked against the chunk it cites.
- As a user, I see the sub-questions and what was kept or dropped (and why) as the card's steps, and the full chain
  on the run page.
- As a user speaking "research X", Wanko says one short line ("Researching that — written out below.") and the card
  follows; nothing long is read aloud.
- As a user, when a research run hits its budget I still get the findings it had, marked partial, with the reason.

## UI/UX
- No visible change for the map: `index.md` stays the human, alphabetical file `rebuild_index` writes (wiki.py:396).
- Research card: the PRD 43 `.msg.card` (Copy, Save to wiki) with the PRD 51 step list above the body: "Plan: 3
  sub-questions", one row per sub-question ("searched · 5 hits · kept 2"), "Read 4 parts", "Wrote", "Verified 5/6
  quotes". Status chip: `done` / `partial` / `nothing`. A `### Dropped` list under `## Sources` names quotes the
  verifier removed. The card links to its run page (PRD 51); the run page shows every step with timings and reasons.
- Trigger: typed "research X" / "/research X"; spoken "research X" gets the PRD 43 pattern (one short spoken line,
  card below). While it runs the state is `thinking` and each finished step updates the card's step list.

## Technical approach
**A. Vault map** (`wiki.vault_map(max_tokens=WIKI_INDEX_TOKENS)`, replaces `index_text` at its three prompt sites:
memory.py:616 (`<wiki_index>` in the system prompt), sources.py:1101 (`propose_page_ops`), wiki.py:1719 (session
`ingest`)). Shape, two levels:
```
## people (42) — tags: family, work, climbing
- [[people/anna]] — Anna, person — sister, lives in Lyon
- … top pages …  +37 more
```
- Folder order: by page count, then name. Per folder: rank pages by `inbound links × 2` (`backlinks`, wiki.py:861; `MAP_LINK_WEIGHT = 2`) +
  recency (`updated`, half-life 30 days; `MAP_RECENCY_HALF_LIFE_D = 30`), tie-break path; the folder header lists its top 3 tags.
- Fitting: round-robin — every folder gets its header and best page first, then the next page of each folder, until
  the cap; each folder ends with "+n more". A folder never vanishes while the cap allows its header.
- Incremental: the per-page row (one-liner, links, updated, tags) is cached keyed by path+mtime — on the PRD 53 index
  tables (`index_text` stays there as the cheap fallback; this map replaces it in the prompt); the map is a cheap fold over the rows.
- `related_context` (PRD 44) stays next to the map in both ingest calls; together they must name the page to update.

**B. Research mode** (`research.py`, one module; `chatbot.research_command` regex next to `briefing_command`,
chatbot.py:387, dispatched in `_respond` before `wants_writeup`, chatbot.py:590–600). Deterministic steps, each a
PRD 51 card step and a trace event (PRD 49):
1. **Plan** — one completion → JSON list of ≤ 4 sub-questions (extra dropped; unparseable → the question itself).
2. **Search** — per sub-question `wiki.search(q, 5)` (wiki.py:911) + `wiki.search_chunks(q, 3)` (wiki.py:1202), with
   PRD 53 folder/type filters when the plan names them; no model call.
3. **Judge** — one batched completion over all sub-questions' top-k (path/title + 300-char snippet each) → `keep|drop`
   with a one-line reason per hit; if the batch JSON fails to parse, fall back to one call per sub-question; unjudged
   hits are dropped ("not judged").
4. **Read** — the kept chunks via `wiki.read_source(source, n)` (wiki.py:1236), pages via `read_page`, best score
   first, ≤ `RESEARCH_MAX_READS` = 6.
5. **Write** — one `long` completion with the card prompt variant: `## Findings` (bullets, each with a quoted span
   `"…" — Title · part n/N`) and `## Sources` (the `card_sources` lines, llm.py:768).
6. **Verify** — deterministic: each quoted span, whitespace/quote/ellipsis-normalised, must be a substring of the cited
   chunk/page text that was read; otherwise the bullet's quote is removed and listed under `### Dropped`. Zero
   verified quotes → status `nothing`.
- **Budget governor**: sub-questions ≤ 4, reads ≤ 6, model calls ≤ 8 (plan 1 + judge 1 + write 1 + ≤ 5 retries/
  repairs), wall clock 90 s (`time.monotonic`). Every check is a run step; on exhaustion the run skips to Write with
  what it has (or ends `nothing` if nothing was kept), status `partial`, reason "budget: reads" etc.
- Output: `llm.write_up`'s card shape `{id, markdown, sources, title}` stored in `_cards` / `last_card`, so **Save
  to wiki** and "save that as a note" go through `wiki.save_card` (wiki.py:737), keeping `## Findings` / `## Sources`
  only (no step list) with the run id in the frontmatter; `_last_context` gets the
  card (the briefing pattern) so `/write` after research does not re-run it.
- Prompt budget (PRD 11): research completions are side calls, not `_conversation` turns; material per call is
  `_fit` to `CARD_CONTEXT_TOKENS`; the conversation gets the card and the session turns "researched: <title>".
- Escalation hook (later, not built): a typed grounded answer that says "not covered" may append "Research it?".

### Decisions
- Chose a folder-ranked map over the alphabetical cut because the cut hides a deterministic, arbitrary slice;
  ranking by links + recency keeps the pages the user actually uses.
- Chose a fixed step chain over a tool-calling agent loop: a 4-bit local model loops and forgets budgets; fixed
  steps make every call countable and testable with `FakeClient`.
- Verify quotes by substring, not by a model: cheap, exact, and the one guarantee the card can honestly make.
- Research is explicit only (typed or "research X"): it costs up to 90 s, so it is never auto-triggered.
- Chose the map in the system prompt and in ingest (one function, three sites) over ingest only: spoken recall gains a
  representative overview at no extra tokens.
- Chose links × 2 + a 30-day recency half-life, kept as two constants, over tuned or per-folder weights.
- Chose a spoken "research X" that runs without confirmation (short line + card) over asking first: the 90 s cap
  bounds the cost.
- Chose one batched judge call (budget of 8) with a per-sub-question fallback over a call per sub-question: fewer
  calls, and a parse failure still gets judged.
- Chose Save to wiki keeping only Findings/Sources, with the run id in frontmatter, over saving the step list.
- Chose exact substring after whitespace/quote/ellipsis normalisation over edit distance: no fuzzy matches.
- Depends on PRD 51 (run pages, card steps) and PRD 53 (index tables, filters; synthetic vault via
  `tests/perf/make_vault.py`, marker `perf`).
- Risk: the map shifts with recency and makes the system prompt less cache-friendly; mitigation: the map is
  computed at conversation start / `new_topic`, not per turn.

## Data model
- Page row: `{path, folder, one_liner, inbound: int, updated: date, tags: [..]}`; invariant: the map is a pure
  function of the rows; `index.md` unchanged.
- Research run: `{run_id, question, status: done|partial|nothing, reason, steps: [{kind, ms, detail}],
  subquestions: [..], hits: [{q, ref, keep, reason}], reads: [{source, part}], calls: int, card_id}`.
- Card: the PRD 43 card plus `steps` and `dropped: [{quote, ref, why}]`; invariant: every quote left in `##
  Findings` is a substring of text read in this run.

## Test scenarios
1. `tests/wiki/test_vault_map.py`: a 12-page vault in 3 folders → `vault_map()` equals a golden string (folder order,
   per-folder ranking by inbound then updated, tag line, "+n more").
2. `tests/wiki/test_vault_map.py`: 320 generated pages (`tests/perf/make_vault.py`, small size) with `projects/zephyr` (linked from 5 pages) → the map ≤ 600
   tokens names `projects/zephyr` and every folder; `index_text` at the same cap does not (the cut is real).
3. `tests/wiki/test_wiki.py`: session `ingest` over 320 pages about Zephyr → the ingest prompt (FakeClient call)
   contains `projects/zephyr` from the map and `related_context`; the proposed op updates it, no new duplicate page.
4. `tests/wiki/test_vault_map.py`: writing one page re-reads only that page's row (counted reads == 1) and the map
   reflects the new one-liner; deleting a page drops it and the folder count.
5. `tests/llm/test_research.py`: plan reply with 6 sub-questions → 4 kept; the card's `steps` lists exactly those 4.
6. `tests/llm/test_research.py`: judge drops an off-topic hit → `hits` has `keep: false` with the model's reason,
   and that hit is never passed to `read_source`.
7. `tests/llm/test_research.py`: the write reply quotes one real span and one fabricated span → the fabricated
   bullet's quote is removed, `### Dropped` lists it with "not in Title · part n/N"; the real one stays.
8. `tests/llm/test_research.py`: a monotonic clock patched past 90 s after Search (and separately reads capped at 6)
   → card status `partial`, reason `budget: time` / `budget: reads`, findings from what was read; no hits → `nothing`.
9. `tests/llm/test_research.py`: a full run counts `len(FakeClient.calls) <= 8`; `read_source` ≤ 6 calls.
10. `tests/session/test_research_turn.py`: typed "/research Zephyr budget" → `card` message, no `speech` of the
    body; spoken "research Zephyr budget" → one spoken line + the card; "save that as a note" → `wiki.save_card`
    writes `notes/<date>-<slug>.md`; `test_protocol_doc.py` green.
11. Run page (PRD 51 test file): the run shows plan → search ×n → judge → read → write → verify with timings.
12. Battery: tests/qa/stress.py sections A–F unchanged (spoken recall, band, write-up); new section G (3 research
    turns on the seeded guides) records quotes verified / dropped and wall time.

## Out of scope
Web search and Wikipedia as research sources, embeddings / vector search, multi-turn research dialogue
("dig into point 2"), scheduled or background research, auto-triggered research, changing spoken recall or the band.

## Open design questions
None open.
