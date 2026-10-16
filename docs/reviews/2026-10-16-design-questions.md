# Open design questions: PRDs 47b, 50-55

Decided 2026-10-16: all as recommended, plus: no Obsidian (PRD 53 drops staleness sweeps).

## Cross-cutting
- ✓ Voice "yes" for writes vs offers: PRD 52 wants deletes and pushes approved by a spoken "yes", PRD 45/46 offers use `is_affirmative`; PRD 54 spoken research runs unconfirmed. → recommendation: a separate `is_approval` for PRD 52 only (PRD 52), offers unchanged, spoken research needs no confirmation.
- ✓ Run pages in prod: PRD 51 serves `/runs/<id>` with titles, paths and shas, PRD 52 and 54 add proposals and steps to them. → recommendation: expose them with the same origin check as `/ingest`, no extra auth, no text content (PRD 49 sentinel test extended); revisit if the port is ever exposed.
- ✓ FTS5 vs own scorer: PRD 53 keeps BM25 on plain tables, PRD 55 builds `chunk_vec` beside them and PRD 54 caches rows there. → recommendation: own scorer over plain tables (golden-score parity), FTS5 only if the 50 ms gate fails.
- ✓ Does the map go in the system prompt: PRD 54 replaces `index_text` at three sites, PRD 53 keeps `index_text` as the cheap fallback. → recommendation: yes, map computed at conversation start / `new_topic`; `index_text` stays only as fallback.
- ✓ Does `card_update` carry `markdown`: PRD 51 asks; PRD 52 and 47b do not need it. → recommendation: add optional `markdown` in 51 (keeps the running state for write-ups); 52 and 47b stay independent.
- ✓ Shared synthetic vault: PRD 53 defines `tests/perf/make_vault.py` (marker `perf`), PRDs 54 and 55 reuse it. → recommendation: one generator with size and `--prose` options, no per-PRD copies.

## PRD 47b
- ✓ Target ambiguity (card vs open page): recommend newest-wins, named in the proposal title so Cancel is cheap.
- ✓ Page count in the proposal: recommend a local estimate with "~", exact count only in the run card.
- ✓ Daily pull plist: recommend shipped but unloaded, since a tablet that is usually unplugged makes it mostly noise.

## PRD 50
- ✓ **Sequencing against PRDs 51-55:** the split moves lines those PRDs edit. Recommend 50 starts after 49 merges and each later PRD rebases onto the current layout at its own gate; no PRD waits for 50.
- ✓ **Where do PRD 52's approval registry and PRD 51's run pages live?** Recommend `agent/approvals.py` and `server/runs.py` (the `writes` flag on tools is the hook), decided when 52 and 51 are signed off.
- ✓ **Delete the shims at step 12 or keep one release?** Recommend delete: the layer test then fails any old import.

## PRD 51
- ✓ **Should `card_update` carry `markdown`?** The write-up's body arrives with the final status, so `{id, status, steps?, actions?}` cannot fill it. Recommendation: add optional `markdown`; the alternative is emitting the write-up card only at the end (no running state).
- ✓ **Briefing card body.** Today the briefing has no card until "write it up" (llm.py:803). Recommendation: the run card body is empty (header + steps + Open run); "write it up" still produces the PRD 46 content card.
- ✓ **Commit sha source.** Recommendation: add `commit` to `ingest_end` (one `rev-parse` after the commit) over matching `git log` by time window, which mis-assigns when two writers commit in the same second.
- ✓ **Host-run end.** reMarkable runs have no end event. Recommendation: derive status on the page; the alternative is a `POST /runs/{id}/end` from the script — one more write path.
- ✓ **Run page exposure in prod.** The page has titles, paths and shas but no text. Recommendation: same origin check as `/ingest` and no extra auth (localhost-bound app); revisit if the port is ever exposed.
- ✓ **Content labels.** Recommendation: origin from the commit subject (`wiki: edit` → you, ingest → source, page ops → Wanko) via the existing `_USER_COMMIT`/`_CHAT_COMMIT` regexes (wiki.py:1388-1389).

## PRD 52
- ✓ **Affirmative words for writes**: add "delete/forget/send/approve/confirm" to `_AFFIRMATIVE_WORDS` (llm.py:550-551)? Recommend yes, scoped to approvals only (a separate `is_approval` reusing the leads), so offers stay unchanged.
- ✓ **`save_card` behind approval**: recommend no — it is an explicit user action that only adds a page; approvals are for deletes and writes leaving the vault.
- ✓ **Spoken "no" acknowledgement**: silent drop (as offers) or "Left it."? Recommend silent for any other turn, one line only for an explicit short "no" / Reject.
- ✓ **Typed "yes"**: resolves exactly like spoken (via=voice, or a third `via: typed`)? Recommend `via: typed`, same path.
- ✓ **`card_update` and `markdown`**: 52 needs none (status, actions, commit only); PRD 51 asks whether `card_update` may carry `markdown`. Recommend keeping 52 independent of the answer; decide once, in 51.
- ✓ **Button-only Forget from the Sources view**: proposal card in the transcript even when on the Sources view? Recommend yes, plus a toast linking to it, so the one model holds.

## PRD 53
- ✓ **FTS5 or our scorer?** Recommend our scorer over plain tables (golden-score parity, proper weights, subset stats); revisit FTS5 only if postings fetch misses the 50 ms gate.
- ✓ **Where does the file live?** Recommend `<vault>/.index/search.sqlite` (pages and chunks together) over `raw/.index/`, which is source-scoped; one more `.gitignore` line.
- ✓ **Staleness detection for Obsidian edits:** recommend folder mtimes per search + a 5 s throttled sweep; fsevents only if the sweep shows up in p95.
- ✓ **Chunk text in DB or JSON?** Recommend keeping `raw/.index/<sha>.json` for text (no migration), DB for postings.
- ✓ **Prompt index at scale:** `index_text` is alphabetical and 600 tokens covers ~15 pages of 5k; recommend keeping it identical here and letting PRD 54 (the map replaces `index_text` in the prompt; 53 only makes it cheap) decide what the prompt sees.

## PRD 54
- ✓ **Map in the system prompt or only in ingest?** Recommend both (one function, three sites); spoken recall gains a representative overview at no extra tokens.
- ✓ **Ranking weights (links vs recency)?** Recommend inbound links × 2 + recency (half-life 30 days); tune on the prod vault, keep as two constants.
- ✓ **Spoken "research X" — confirm first?** Recommend no confirmation for typed; spoken says the short line and runs, since the 90 s cap bounds the cost.
- ✓ **Judge per sub-question or one batched call?** Recommend one batched call (budget of 8); fall back to per sub-question only if the batch JSON fails to parse.
- ✓ **Should Save to wiki keep the step list?** Recommend no: save `## Findings` / `## Sources`, link the run id in frontmatter.
- ✓ **Verify fuzziness?** Recommend exact substring after normalising whitespace, quotes and ellipses; no edit distance.

## PRD 55
- ✓ **Synthetic vault realism:** PRD 53's `tests/perf/make_vault.py` makes keyword-dense template text; recommend the bench passes a `--prose` option to it (invented-name paragraphs, no FAQ keywords) rather than a second generator.
- ✓ **Gated vs relaxed hybrid:** paraphrases fail today mostly at the precision rule, which a gated hybrid cannot bypass; recommend also reporting a relaxed variant (cosine ≥ τ counts as one matched word) for information only, with adoption judged on the gated one.
- ✓ **Labelling expected parts:** recommend labelling by hand from the battery's expectation strings, checked against the raw files once, and committing the JSON.
- ✓ **Latency budget source:** the 30 ms p95 assumes the embed call runs alongside BM25 at end-of-speech; recommend measuring it serial (worst case) and parallel.
- ✓ **Chat model co-residency:** the embed server is separate from the chat model on :8080/:8000; recommend recording memory and latency with the chat model loaded, since that is the real condition.
