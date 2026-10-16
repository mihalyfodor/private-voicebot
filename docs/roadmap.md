# Roadmap

Updated 2026-10-16. Ordered; each item is one PRD unless marked chore. Sources: the ERNI workflow review (process
lessons, merged into `feature/process-lessons`) and the llm-wiki survey (ideas 1–3 below). Dropped ideas are listed at
the end so they are not re-proposed.

## 0. Chore: daily vault backup with rotation

Done 2026-10-16

Not a PRD; one `Fix:`-style commit. Do first — it protects the vault before anything below touches it.

- `scripts/prod.sh backup`: add `KEEP` (default 7), delete all but the newest N `.tgz` and `.bundle` (`scripts/prod.sh:67`).
- Runaway guards, all in `cmd_backup` so manual and scheduled runs share them:
  - rotate *before* writing, then refuse to run if free space in `BACKUP_DIR` is below `MIN_FREE_MB` (default 2048) or the
    new archive would push the backup folder past `MAX_TOTAL_MB` (default 5120); exit non-zero with a one-line reason;
  - write to a temp name and `mv` into place, so a killed run never leaves a half archive counted as a backup;
  - a lock (`mkdir`-based or `flock`) so an overlapping run exits immediately instead of stacking;
  - skip the run when the vault has no new commit since the last backup (compare `git rev-parse HEAD` with a marker file);
  - log one line per run to `$DATA/logs/backup.log`, itself truncated to the last 200 lines.
- `scripts/launchd/com.voicebot.backup.plist` with `StartCalendarInterval`, explicit `PATH` (launchd has none for git/tar), logs under `$DATA/logs`.
- `docs/deploy.md`: one install line next to the ASR plist.
- Tests: `tests/prod/test_prod_script.py` — temp `DATA`/`BACKUP_DIR` seeded with 9 old files, 7 of each kind remain;
  `MIN_FREE_MB` set absurdly high → non-zero exit and no new file; a held lock → immediate exit; unchanged HEAD → no new
  file; `plutil -lint` on the plist.
- Effort S. No app behaviour change.

## 1. PRD 44 — Vault health

Wiki quality at write time. Two small pieces, neither changes runtime replies.

**Lint, three new finding types** in `wiki.lint()` (`wiki.py:1380`):
- dead `source:` frontmatter refs left behind by `sources.forget()` (`sources.py:1364`) — the one real gap today;
- raw files with no manifest entry and no page citing them;
- duplicate `name:` across pages.
Not orphans (index links everything), not staleness (`reingest` covers it). Keep the list short: every finding type makes
`lint_and_log` (`wiki.py:1408`) commit on more reflections and adds rows to the Wiki tab.

**Related pages at ingest:** `wiki.related_context(text, k=5)` on top of `wiki.search` (`wiki.py:888`), skipping `sources/`
pages. Adds "Existing related pages (update these rather than duplicating)" to both ingest prompts
(`sources.propose_page_ops` `sources.py:1083`, session `wiki.ingest` `wiki.py:1495`). Fixes near-duplicates once the vault
outgrows the 600-token index cap (`WIKI_INDEX_TOKENS`).

- Tests: `tests/wiki/test_wiki.py` (`wiki_sandbox`): forget → finding appears; seeded duplicate name; uncited raw file.
  `tests/sources/test_sources.py` + `tests/wiki/test_wiki.py`: capture the `FakeClient` user message, assert the related
  path is present even when the index is cut.
- Effort S. Risk low (prompt content and lint output only).

## 2. PRD 45 — Ask before lookup (recall calibration)

The open follow-up from 2026-08-31: a mis-transcribed four-word turn pulled a 6k-char guide chunk into a chat reply. The
design point is ask-before-lookup, not a tighter threshold. This PRD also absorbs the "confidence hedge" idea from
llm-wiki — shipping a hedge on its own would speak weak hits *more* often, the opposite of what we want.

- One score ladder on `recall_min_score()` (`wiki.py:1236`, `RECALL_MIN_SCORE = 4.5`) applied in `_recall_hits` (`llm.py:784`):
  high → answer grounded as today; medium → ask first ("Want me to check your Morrowind guide?") reusing the deeper-offer
  and `is_affirmative` machinery (`llm.py:552`); low → no recall.
- Gate recall on `content_question()` (`llm.py:673`), which exists but is never called by `recall_for`.
- Re-tune bands against `eval/` and `tests/llm/test_recall_precision_2.py`; battery must not regress (37/43 today).
- Tests: `tests/llm/` with `FakeClient`, one scenario per band asserting block and note; the four-word mis-transcription
  case as the hostile row.
- Effort M. Risk high: changes wording and length on every grounded turn. Do after PRD 44 (independent, but 44 is cheap
  and safe and should not wait behind a calibration round).

## 3. PRD 46 — Briefing ("what's new?")

Spoken 2–3 sentence recap of what changed since the last briefing, ending with the PRD 43 offer; "write it up" yields a
card with Pages / Sources / Memory sections. Tight version only.

- Routing: deterministic whole-turn regex in `chatbot.respond` next to `new_topic_command` / `save_note_command`
  (`chatbot.py:514-526`), modelled on `write_up` (`chatbot.py:419`). Typed `/brief` works via the same path.
- Gathering `wiki.changes_since(ts)` under `vault_lock.LOCK`: pages from `git log --since --name-only` via `wiki._git`
  (`wiki.py:133`; `log.md` has dates only, so it cannot cut at a time of day); sources from `sources.manifest_entries()`
  on `ingested_at`; memory as counts + two examples from `memory_ops.jsonl`. Skip `wiki: lint` commits, `index.md`,
  `log.md`; group pages touched only by `session …` sources as "notes from our chats".
- Recap: one completion over the digest wrapped in `untrusted()`; set `llm._last_context` so the existing
  `wants_writeup` → `llm.write_up` card path needs no new code. Nothing changed → "Nothing new since …", no LLM call.
- Cutoff `briefing_at` (ISO) in `settings.json` via `avatars` load/save; moves forward only after the recap was spoken;
  default 24 h when unset; never moves backwards.
- Cut: recent questions (session.log holds only user text), explicit windows ("this week"), palette entry, LLM editorial
  over page contents. Out of scope: PDF/reMarkable, scheduled auto-briefing, rail badge, Gmail/news.
- Tests: `tests/wiki/test_wiki.py` (changes_since with commits before/after cutoff; manifest filter),
  `tests/session/test_turns.py` (one completion + cutoff moves; nothing-new → 0 calls; LLM offline → cutoff unchanged;
  digest inside `<document>`), `tests/llm/test_written_answers.py` (card contains the page titles).
- Effort M (≈150 lines + prompt tuning). Risks: recap quality on the 4-bit model (keep the digest structured and capped);
  self-noise from memory-driven page ops; one clock seam for tests (`memory._stamp` mixes `_today` with `datetime.now()`,
  `memory.py:197-199`).

## Backlog (unordered, from the 2026-08-30 handoff)

- Remaining battery misses (A11 bounded-context chunk, stray offers)
- `static/app.js` ES-module split; split candidates among `memory.py` / `wiki.py` / `sources.py` / `chatbot.py` (>1000 lines)
- README refresh
- Tests import fixtures from other test files despite `docs/process.md` forbidding it (ruff F811 silenced for `tests/**`)
- More calibration corpora (spreadsheet, real Teams transcript); Kiwix offline Wikipedia; native Kokoro via host service
- Notes-via-chat editing; `hands_free` / verbosity state triplication
- Vector search (Qwen3 embeddings on `:8082`, RRF) — only if remaining battery misses prove to be paraphrase failures

## Dropped

- Recency rule ("prefer newer, say the date"): recall quotes raw chunks without `updated`, at most two of them; dates in
  speech add the padding PRD 45 is removing.
- Standalone confidence hedge: folded into PRD 45 as the medium band.
- URL ingestion as a model tool (PRD 20/33), `.trash` snapshots, sha duplicate guard, ask history, save-as-note: covered
  by git, `manifest.jsonl`, the transcript and PRD 43.
- ERNI ceremony: ratchet baselines, STATE/DECISIONS ledgers, model tiers, decision rounds, cost audits.
