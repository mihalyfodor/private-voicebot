# Roadmap

Updated 2026-10-17. Ordered; each item is one PRD unless marked chore. Sources: the ERNI workflow review (process
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

Done 2026-10-16 (3daa501).

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

- Tests: `tests/vault/test_wiki.py` (`wiki_sandbox`): forget → finding appears; seeded duplicate name; uncited raw file.
  `tests/ingest/test_sources.py` + `tests/vault/test_wiki.py`: capture the `FakeClient` user message, assert the related
  path is present even when the index is cut.
- Effort S. Risk low (prompt content and lint output only).

## 2. PRD 45 — Ask before lookup (recall calibration)

Done 2026-10-16 (c867f90). Name questions (who/where/which/when + proper match) answer directly; medium turns are held, not streamed.

The open follow-up from 2026-08-31: a mis-transcribed four-word turn pulled a 6k-char guide chunk into a chat reply. The
design point is ask-before-lookup, not a tighter threshold. This PRD also absorbs the "confidence hedge" idea from
llm-wiki — shipping a hedge on its own would speak weak hits *more* often, the opposite of what we want.

- One score ladder on `recall_min_score()` (`wiki.py:1236`, `RECALL_MIN_SCORE = 4.5`) applied in `_recall_hits` (`llm.py:784`):
  high → answer grounded as today; medium → ask first ("Want me to check your Morrowind guide?") reusing the deeper-offer
  and `is_affirmative` machinery (`llm.py:552`); low → no recall.
- Gate recall on `content_question()` (`llm.py:673`), which exists but is never called by `recall_for`.
- Re-tune bands against `eval/` and `tests/agent/test_recall_precision_2.py`; battery must not regress (37/43 today).
- Tests: `tests/llm/` with `FakeClient`, one scenario per band asserting block and note; the four-word mis-transcription
  case as the hostile row.
- Effort M. Risk high: changes wording and length on every grounded turn. Do after PRD 44 (independent, but 44 is cheap
  and safe and should not wait behind a calibration round).

## 3. PRD 46 — Briefing ("what's new?")

Done 2026-10-16 (9f098e3). Card is built from the digest, no model call; own edits and briefing sessions are never news.

Spoken 2–3 sentence recap of what changed since the last briefing, ending with the PRD 43 offer; "write it up" yields a
card with Pages / Sources / Memory sections. Tight version only.

- Routing: deterministic whole-turn regex in `chatbot.respond` as the other registry triggers
  (`tools.registry.match`; at the time `new_topic_command` / `save_note_command`, `chatbot.py:514-526`), modelled on `write_up` (`chatbot.py:419`). Typed `/brief` works via the same path.
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
- Tests: `tests/vault/test_wiki.py` (changes_since with commits before/after cutoff; manifest filter),
  `tests/server/test_turns.py` (one completion + cutoff moves; nothing-new → 0 calls; LLM offline → cutoff unchanged;
  digest inside `<document>`), `tests/agent/test_written_answers.py` (card contains the page titles).
- Effort M (≈150 lines + prompt tuning). Risks: recap quality on the 4-bit model (keep the digest structured and capped);
  self-noise from memory-driven page ops; one clock seam for tests (`core.util.stamp` mixes `today` with `datetime.now()`;
  at the time `memory._stamp`, `memory.py:197-199`).

## 4. PRD 47 — reMarkable 2 over USB: pull handwriting in, push pages out

Pull done 2026-10-16 (ae4106f, `notebooks.py`, `scripts/remarkable_sync.py`, fake device in tests; the real-device run is the user's scenario 8). Push, a host `serve` mode, PDF rendering and launchd → **PRD 47b** (not started).

Decided 2026-10-16 after an integration review (Calendar.app not in use; Graph/Teams/OneNote APIs need tenant consent, so
anything via Microsoft 365 is parked until IT answers). Host-side only: the prod container cannot reach USB
(`host.docker.internal` only, `docs/deploy.md`), so this is a host service or script like the ASR service.

- **Pull:** the device's USB web interface (`http://10.11.99.1`: `GET /documents/`, `GET /download/<guid>/pdf`, image-only
  PDFs). Rasterise pages (`pdftoppm`/`sips`), send each page through the existing image ingest path
  (`sources.py` image → multimodal model "transcribe any text visible") with Apple Vision OCR (`VNRecognizeTextRequest`
  via pyobjc) as a cross-check; file the text as a raw source per notebook page range, normal ingest after that.
  Track seen notebooks by guid + modified time so a re-sync only pulls new pages. Design reference:
  `~/projects/erni/llm-wiki/remarkable.md` (unbuilt). Tesseract is installed but poor at handwriting; don't use it.
- **Push:** `POST /upload` to the device: "send this to my reMarkable" for a written card, a wiki page or the PRD 46
  briefing, rendered to PDF (regex command + yes/no confirm, PRD 43 pattern).
- Risk: transcription quality on the 4-bit model; mitigation: keep the page PNG as the raw file so re-ingest with a
  better model is one command. Risk: device offline; mitigation: the command answers "reMarkable not connected".
- Effort M. Probe first: connect the device, enable the USB web interface, `curl http://10.11.99.1/documents/`.

## 5. PRD 48 — OneNote history import (one-time)

Done 2026-10-16 (0241a53, `batch_import.py`, `scripts/ingest_batch.py`; strict mode: a failed extraction or summary is `failed`, never filed as unreadable). Waits for the user's OneNote exports.

No Graph access without IT, so: OneNote for Mac → export each section as PDF (text layer preserved) into a folder,
then a batch ingest script (`scripts/ingest_batch.py`, PRD 36 style progress) files them through the normal PDF path
with head/middle/tail sampling. Hundreds of pages ≈ 1–3 h on the local model; run overnight. PRD 44 related-pages
keeps entity duplication down. If IT ever grants `Notes.Read`, swap the export step for Graph HTML; nothing else changes.
Effort S–M once the exports exist. Probe first: export one section and check `pdftotext` returns real text, not images.

## 6. PRD 49 — Event trace

Done 2026-10-16.

`events.jsonl` via `trace.py`, two profiles: **prod** = actionable events only, never text (titles, paths, scores,
lengths, timings, ids); **dev** = the same plus a separate `verbose.jsonl` with prompts, turn text and tool results.
The container forces prod; dev is refused on a `~/.voicebot` path or inside a container. Rotation 10 MB × 5 on both
files and a compose log cap, so prod cannot fill the disk. `scripts/trace_report.py` prints one row per turn/ingest and
the error list; the `/trace` skill tells an agent to read the report first and open raw lines only for flagged turns.
session.log retires in favour of `turn_end`.

## 7. Chore — exception sweep (after PRD 49)

Done (branch merged).

The silent turn-level handlers in chatbot.py (briefing, write-up, new topic, respond, ASR, canned turn, save note) emit
an `error` event and a `scope: turn` status line instead of going quietly to idle; degraded ingests say "filed raw
only". File:line list is in PRD 49 §Out of scope. Effort S–M.

## 8. First prod setup (the user's)

Done 2026-10-17 (image voicebot:d7b9c73, redeployed as voicebot:ad33ad4 the same day); redeployed 2026-10-18 as voicebot:6ac666e (PRDs 56 + 57), then as voicebot:d05194a (PRD 58, the first run of `prod.sh deploy`). Before: `~/.voicebot` did not exist and everything ran as the dev process on :8011. Data folder renamed to ~/.voicebot/data 2026-10-17. Steps in docs/deploy.md.

USER TODO: the backup launchd agent is not loaded (`scripts/install_launchd.sh backup`); until then backups are hand-run (`scripts/prod.sh backup`, also part of `deploy`). Decided 2026-10-18: not now.

## 9. PRD 50 — Modular layout (done 2026-10-17)

Seven layers importing only downward (core, vault, memory, ingest, tools, agent, server), enforced by a ratchet layer
test; one tool registry with two entry modes (trigger regex = deterministic workflow, schema = model-pickable); twelve
pure-move refactor commits with shims and per-step test retargeting. See docs/50-modular-layout.md. Can interleave with
51–55; no PRD waited for it. Done: twelve `Refactor:` commits on `feature/modular-layout-2`, `scripts/check.sh` green at every step, `ALLOWED` empty.

## 10. PRD 51 — Run cards and run pages · PRD 52 — Approvals · PRD 47b — reMarkable push

Trust surface: a card with steps, actions and a run link; a server-rendered `/runs/<id>` page built from the PRD 49
events; one pending slot where a spoken "yes" and a card button resolve a write identically; push to the tablet as the
second approved write after forget (PRD 47b done 2026-10-17, card-only). docs/51-run-cards.md, docs/52-approvals.md, docs/47b-remarkable-push.md.

## 11. PRD 53 — Scalable index · PRD 54 — Vault map + research mode · PRD 55 — Hybrid measurement

Scale to thousands of pages: SQLite postings with incremental updates and golden BM25 parity (and the 200-line index
lint bug fixed); a representative two-level map in place of the alphabetical prompt index plus a typed "research X"
workflow with a judge, verified citations and a budget; embeddings via the local :8082 server measured against BM25 on
a synthetic vault and adopted only on numbers. docs/53-, 54-, 55-*.md.

Open design questions for all of the above, with recommendations: docs/reviews/2026-10-16-design-questions.md.

## 12. PRD 57 — Context hygiene · PRD 56 — Session boundaries (both done 2026-10-17)

From docs/reviews/2026-10-17-session-handling.md (Opus review + research, 2026-10-17). Both signed off 2026-10-17.
PRD 57 (S+S, agent/ only) first: stub old tool results under budget pressure before dropping turns; the open page
focuses a turn only when the question points at it. Then PRD 56 (M): the HH:MM clock leaves the system prompt
(`get_time` returns the date too), the prompt is rebuilt after a fold that applied ops and after Memory edits, memory
is saved after 10 min idle and on disconnect, the first turn of a new day starts a new thread quietly, "new topic"
summarises and answers before the fold runs in the background. Fix landed first: the trim budget subtracts the tool
schema tokens so the meter never reads over 100 % after a trim.

## Parked (needs IT)

Microsoft Graph read-only mail/calendar (device-code flow, mirrors `tools/gmail.py`), Teams transcripts (admin-only
scopes; drop the .vtt in by hand instead), OneNote via Graph. Ask whether low-privilege delegated scopes
(`Mail.Read`, `Calendars.Read`, `Notes.Read`) are user-consentable in the tenant.

## Backlog (unordered, from the 2026-08-30 handoff)

- Remaining battery misses (A11 bounded-context chunk, stray offers)
- `static/app.js` ES-module split; split candidates: `ingest/pipeline.py` (~950 lines) is the largest module left
- More calibration corpora (spreadsheet, real Teams transcript); Kiwix offline Wikipedia; native Kokoro via host service
- Notes-via-chat editing; `hands_free` / verbosity state triplication
- Vector search (Qwen3 embeddings on `:8082`, RRF) — only if remaining battery misses prove to be paraphrase failures

Housekeeping backlog (audit 2026-10-17):

- Pin docker requirements: `requirements-docker.lock` from a known-good build (dev is py3.14, the image py3.12)
- Host launchd jobs (asr, backup, remarkable) run from the dev checkout (templates render its path via scripts/install_launchd.sh): keep the checkout on `main`, or install from a separate clone
- Done 2026-11-02 (housekeeping sweep): fixture imports moved into per-layer `conftest.py` files and the F811 ignore dropped;
  `.env.example` complete and guarded by `tests/test_env_example.py`; status per finding in `docs/reviews/2026-08-29-architecture.md`
  (#7 #8 #10 partly, #9 resolved: lifespan + `asyncio.to_thread` shutdown save, #11 by design, #12 vault git failures still invisible). The backup marker already
  covered memory/settings since PRD 58. FastAPI lifespan migration done (`server/app.py`, shutdown save via `asyncio.to_thread`). Review-backlog pass the same day: layer test sees relative + dynamic
  imports; push service caps request bodies (`REMARKABLE_MAX_BODY_MB`, 413); Settings reMarkable-row filter tested; v_ aliases renamed m_/a_. README rewritten (memory, vault, prod, development).

## Dropped

- Recency rule ("prefer newer, say the date"): recall quotes raw chunks without `updated`, at most two of them; dates in
  speech add the padding PRD 45 is removing.
- Standalone confidence hedge: folded into PRD 45 as the medium band.
- URL ingestion as a model tool (PRD 20/33), `.trash` snapshots, sha duplicate guard, ask history, save-as-note: covered
  by git, `manifest.jsonl`, the transcript and PRD 43.
- ERNI ceremony: ratchet baselines, STATE/DECISIONS ledgers, model tiers, decision rounds, cost audits.
