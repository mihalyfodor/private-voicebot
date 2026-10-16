Status: refactoring

# 48 — Batch import (OneNote history)

## Overview
Years of notes sit in OneNote; the only way into the wiki today is dropping files into the intake popup one batch at
a time, with the app open and a 2 MB PDF cap. This PRD adds a host script that walks an export folder and files every
PDF/DOCX through the same ingest a drop uses, one file at a time, idempotent and resumable, so an overnight run can
import the history. There is no OneNote export yet: everything is built and tested against generated PDFs/DOCX, and
the first real run is a manual scenario for the user.

## User stories
- As a user, I run `scripts/ingest_batch.py ~/Exports/OneNote --dry-run` and see, per file, whether it would be filed,
  is already filed, needs OCR or is unsupported, with nothing written and no model called.
- As a user, I start it overnight with `--sleep 5` and in the morning read `batch.log` and the summary line.
- As a user, after a crash or Ctrl-C I re-run the same command (or with `--resume`) and only the unfinished files run.
- As a user, an ink-only OneNote page shows up as `needs-ocr` in the log instead of an empty "unreadable" source page.

## UI/UX
- No app UI change. CLI only:
  `scripts/ingest_batch.py <folder> [--vault PATH] [--kinds pdf,docx] [--resume] [--dry-run] [--limit N] [--sleep S]`.
- One progress line per file: `[12/340] filed  Work/Falcon/Kickoff.pdf -> Falcon kickoff notes (3 pages, 41 s)`;
  states `filed` · `already` · `needs-ocr` · `unsupported` · `too-large` · `failed — <reason>`.
- Final line: `filed 290 · skipped 38 (already 30, needs-ocr 6, unsupported 2) · failed 12 · 3h 52m`; exit 0 when
  nothing failed, 1 otherwise, 2 for a refused start (bad args, dirty vault, app running).
- The filed sources appear in the Sources view and in the PRD 46 briefing like any drop (capped at 8 + "+n more").

## Technical approach
- **Script** `scripts/ingest_batch.py` (argparse, `ROOT` on `sys.path`, `.env` via `load_dotenv` like
  `scripts/smoke.py`); the importable core is `sources.ingest_batch(folder, *, kinds, resume, dry_run, limit, sleep,
  client=None, model="", out=print) -> dict` so tests drive it in-process. `--vault PATH` sets `WIKI_PATH` before
  `wiki`/`sources` are imported (default: the env, i.e. `~/llm-wiki-dev` for dev, `~/docker-data/voicebot/llm-wiki` for
  prod per docs/deploy.md); `MEMORY_DIR` / `SETTINGS_PATH` come from the same env as the app.
- **Walk**: `folder` recursively, sorted by relative path, extensions from `--kinds` (default `pdf,docx`); other files
  (`.one`, `.mht`, images) are counted `unsupported`. Hidden files and `batch.log` are ignored.
- **Per file**: read bytes → `sources.sha256` → `manifest_find` hit = `already` (no model call). Otherwise the text
  layer check (PDF only): `pdf_text` returns "" or fewer than `MIN_PDF_CHARS` characters per page on average →
  `needs-ocr`, skipped, nothing stored. Otherwise `ingest_upload(data, name, hint=<relative folder, e.g.
  "Work/Falcon">, on_stage=...)` — the drop path: `check_upload`, store under `raw/inbox/`, head/middle/tail sampling
  (PRD 31), summary, page ops, PRD 32 chunk cache and rename-after-title, one commit per file. "Pages touched" is
  `len(pages_touched)` of the new manifest line. `IngestError` / any exception → `failed — <reason>`, next file.
- **Source line**: `file_bytes` gains `source=` (default unchanged `user drop <date>`); the batch passes
  `batch import <date>` so imported pages are distinguishable from drops.
- **Size**: `.pdf` moves into `LARGE_TEXT_EXTS` (50 MB, text is sampled anyway); OneNote section PDFs with embedded
  images routinely exceed 2 MB. Applies to drops too.
- **Guards at start** (exit 2, nothing written): the vault's `git status --porcelain` is not clean (a half-filed
  crash leftover would be swept into the next `git add -A` commit); the app answers on `PORT` / `PROD_PORT`
  (`vault_lock.LOCK` is per process, so two writers would race on the manifest and the git index). The message says
  `scripts/voicebot stop` and, for a dirty vault, lists the paths.
- **batch.log** in `<folder>`: appended, one JSON line per file `{rel, sha256, state, title, page, pages_touched,
  seconds, reason, at}` plus a header line per run. `--resume` additionally skips files whose last log state is
  `needs-ocr`, `unsupported`, `too-large` or `failed`; without it those are retried. `--limit N` stops after N files
  that reached the model; `--sleep S` waits between such files. Ctrl-C finishes nothing mid-file: the
  in-flight file has no manifest line, the next run's dirty check tells the user to clean it.

- **Strict mode (batch only):** `ingest_upload(..., strict=True)` raises on a failed extraction or a failed summary
  instead of filing the source as "unreadable" with a fallback summary; an unparseable summary (both attempts) and
  a model `unreadable: true` are failures too. The batch always passes `strict=True`; a drop keeps today's forgiving
  path because the user sees the result in the inbox at once. On a strict failure nothing is committed and the only
  thing written so far — the stored raw file — is removed under `vault_lock.LOCK`; no git reset or clean ever runs on
  the vault (an overnight run must not touch unrelated edits). The file is logged `failed — <reason>`.
- **App-running guard:** probes both `DEV_PORT` (8011) and `PROD_PORT` (8010) regardless of `.env`, so a prod
  container on :8010 blocks a run against the prod data dir.

### Decisions
- Chose a batch-only strict mode over changing the drop path because a silent "unreadable" across hundreds of files
  is a lost import, while a single drop is checked by eye.
- Host script over an upload UI: hundreds of files over hours do not fit a browser popup that must stay open; the
  script runs with the app stopped, against dev or prod data via `WIKI_PATH`, and needs no protocol changes.
- One file at a time through `ingest_upload`, not a bulk path: the 4-bit model is the bottleneck anyway, every file
  gets the exact drop behaviour (sampling, ops, chunks, rename) and one commit, so one bad file never spoils others.
- The manifest is the resume state: its sha guard already makes re-filing a no-op, so a crash needs no checkpoint;
  `batch.log` only remembers skips/failures, and deleting it loses nothing that matters.
- Ink-only pages are skipped as `needs-ocr`, not filed as "unreadable" sources: an empty page per ink note would
  bury the real ones; OCR is a later PRD if the first real run shows many.
- Assume OneNote exports PDF (Mac) or DOCX/PDF (Windows) with page titles as filenames; if the real export differs
  (one PDF per section, `.mht`), revisit in the manual scenario.
- Risk: the per-page text threshold misreads a typed-title-plus-ink page as readable; mitigation: the dry run lists
  chars/page so the user can judge before the overnight run.

## Data model
- No new vault state. Manifest lines are unchanged; source pages get `source: batch import <date>`.
- `<folder>/batch.log` (JSONL, outside the vault): run header `{run, started, vault, args}`, then per-file lines as
  above. Invariant: a file is filed iff its sha is in the manifest; the log is advisory.

## Test scenarios
All in `tests/sources/test_sources.py` (generated with the existing `pdf_bytes` helper and python-docx; `FakeClient`,
no model). Scenario 9 is manual.
1. Folder with 2 text PDFs + 1 DOCX in a subfolder → 3 `filed`, 3 manifest lines, 3 vault commits; the DOCX source
   page has `source: batch import <date>`; the summary/ops calls carry `User hint: <subfolder>`; output has
   `[1/3]`…`[3/3]` lines and `filed 3 · skipped 0 · failed 0`; exit 0.
2. Re-run on the same folder → 3 `already`, zero `FakeClient` calls, no new commit.
3. A no-text-layer PDF (`pdf_bytes(..., with_text=False)`) → `needs-ocr`, no raw file, no source page, no manifest
   line; a PDF averaging below `MIN_PDF_CHARS` per page over 5 pages → `needs-ocr` too.
4. A corrupt `.docx` and a queued `Exception` in `FakeClient` for one file → those two `failed — <reason>`, the rest
   filed; exit 1; `batch.log` has one line per file with the state and reason.
5. `--resume` after scenario 4 (failures fixed by a fresh client) skips the two failed files; without `--resume`
   they are retried and filed.
6. `--dry-run` → per-file state list including `would file`, zero model calls, vault and manifest unchanged, no
   `batch.log`.
7. `--limit 1 --sleep 0` on three new files → exactly one filed, two left; `--sleep 2` with a fake clock → one sleep
   per filed file, none after `already`; `.one` / `.png` files → `unsupported`; a 3 MB PDF is filed (new 50 MB cap).
8. Start guards: an untracked file in the vault → exit 2, path listed, nothing filed; a stub HTTP server on `PORT`
   → exit 2 "app is running — scripts/voicebot stop"; `--vault` points `WIKI_PATH` at the given directory.
9. Manual (user, first real OneNote export): `scripts/voicebot stop`; `--dry-run` on the export and read the
   state counts and chars/page; run with `--limit 10` against `~/llm-wiki-dev`, check 3 source pages and their ops;
   then the full run (`--sleep 5`) and check `batch.log`, the summary, and a re-run reporting all `already`.

## Out of scope
OCR of ink pages, `.one` / `.mht` / OneNote API import, an upload UI or progress over the WebSocket, parallel
ingests, preserving OneNote page dates as page dates, deduplicating near-identical pages, auto-cleaning a dirty vault.
