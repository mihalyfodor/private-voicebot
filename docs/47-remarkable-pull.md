Status: implementation

# 47 — reMarkable 2 over USB: pull handwriting into the vault

## Overview
Handwritten notes on the reMarkable 2 never reach the wiki. With "USB web interface" on, the tablet serves HTTP at
`http://10.11.99.1` (list, PDF export). This PRD adds a host script that pulls changed notebooks in as sources (pages
transcribed by the multimodal model, PNGs kept as the raw files). Pull only. The device is not available while this
is built: everything is written and tested against a fake device; the first real sync is a manual scenario.

## User stories
- As a user, I plug the tablet in and run `scripts/remarkable_sync.py pull`; every notebook in `wiki-inbox` that
  changed since the last pull becomes one source in the Sources view, its pages transcribed.
- As a user, an unchanged notebook is not filed again; an edited one replaces its earlier source.
- As a user with the tablet unplugged, the script says so in one line and exits 0.

## UI/UX
- No new UI. Pull prints one line per notebook (`filed "Standup" (3 pages) -> sources/standup`); the Sources view
  refreshes through the existing `/ingest` path.
- Optional cross-check: when the compiled `scripts/vision_ocr` helper exists, its text per page is attached to the
  transcript as a `> Apple Vision OCR:` block. Absent helper: silently skipped.

## Technical approach
- **Host script** `scripts/remarkable_sync.py` (stdlib only, run with the checkout's venv like `scripts/asr_server.py`;
  the container reaches the host only via `host.docker.internal`, and `10.11.99.1` is a host interface):
  - `pull`: 1 s probe of `GET $REMARKABLE_URL/documents/` (default `http://10.11.99.1`); no answer -> prints
    `reMarkable not connected (10.11.99.1)`, exit 0. Finds the `CollectionType` named `$REMARKABLE_FOLDER`
    (`wiki-inbox`), lists it (`GET /documents/<id>`), keeps `DocumentType` items whose `fileType` is `notebook`
    (when present) and whose `ModifiedClient` differs from the state file. Per notebook: `GET /download/<id>/pdf`
    (image-only), `pdftoppm -r 150 -png` into a temp dir (Homebrew poppler, present), optional OCR cross-check per
    page, then one `POST $VOICEBOT_URL/ingest` (default `http://127.0.0.1:8010`) with `kind=notebook`,
    `confirm=true`, `title=<VissibleName>`, the page PNGs as `file` parts, `ocr` (JSON list, one string per page) and
    `replaces` (the previous sha, if any). State is written per notebook only after the app answered
    `filed`/`already`. App unreachable -> exit 1, state untouched; one notebook failing does not stop the rest
    (exit 1 at the end).
  - Apple Vision cross-check (optional): interface only. `scripts/vision_ocr <png>` prints the recognised text on
    stdout, exit 0. The script looks for it next to itself (or `$REMARKABLE_OCR_BIN`); missing, non-zero exit or
    `REMARKABLE_OCR=0` -> that page's `ocr` entry is an empty string and pull goes on. The Swift source and its build
    are not part of this PRD.
- **Container** (`sources.ingest_notebook`, a new `kind=notebook` in `/ingest` and `check_upload`, built on the
  existing image helpers: PNG only, each ≤ `MAX_IMAGE_BYTES`, ≤ 50 pages): digest = sha256 of the page bytes in
  order, checked against the manifest before any model call; pages stored under
  `raw/remarkable/<date>-<slug>/page-NN.png`; each page `_downscale` + `_image_part` -> one transcription completion
  ("transcribe any text visible on this handwritten page as markdown; `[?]` for illegible words; reply with the text
  only", 600 tokens), done before taking `vault_lock.LOCK`; a page under `MIN_PDF_CHARS` (20) becomes
  `[page N: unreadable]`. Transcript `raw/remarkable/<date>-<slug>.md` = `## Page N` + the model text + (when sent)
  the OCR block. That `.md` goes through `file_bytes` as a text kind (`notebook`): head/middle/tail sampling,
  summary, page ops, chunk cache for recall, one commit. All pages unreadable -> the existing unreadable path (source
  page filed, no ops). `replaces` -> `sources.forget(old)` after the new filing, same request. Progress via
  `on_stage` ("transcribing 2/5").

### Decisions
- Chose the page PNGs over the device PDF as the raw files: they are what the model saw, and a better model can
  re-transcribe them later without the tablet; the transcript `.md` is what recall and re-ingest read.
- Chose a host script over the container: the device is a host USB interface; the script stays a dumb transport plus
  `pdftoppm`, and the model calls stay in `sources.py` (one ingest path, one lock, one commit).
- Gemma vision first, Apple Vision as an optional cross-check block only: Vision is free and local and catches words
  Gemma invents. Assume the 4-bit model transcribes legibly; if not, revisit.
- Chose a file-presence contract for the OCR helper so pull is fully testable (and shippable) without Swift.
- Risk: the real device's listing shape (`fileType`, folder ids) is only known from docs; mitigation: the fake
  device follows the documented shape and the last (manual) scenario checks it before merge.

## Data model / state shape
- `~/.cache/voicebot/remarkable.json`: `{"<doc id>": {"modified": "<ModifiedClient>", "pages": 3, "sha": "<sha256>",
  "name": "Standup 14 Oct"}}`; written atomically (temp + rename); a garbled file is treated as empty.
- Vault: `raw/remarkable/<date>-<slug>/page-01.png …`, `raw/remarkable/<date>-<slug>.md`; manifest line as today plus
  `pages: [...]`, `device_id`; source page `kind: notebook`, `source: remarkable <date>`.
- Invariants: no manifest line without its PNGs; state never runs ahead of a filed notebook; pull never writes to the
  device.

## Test scenarios
Fake device: `fake_remarkable` in `tests/conftest.py` (`ThreadingHTTPServer` on `127.0.0.1:0`: `/documents/[<id>]`
JSON with `ID`/`VissibleName`/`ModifiedClient`/`Type`/`fileType`, `/download/<id>/pdf` a two-page image-only Pillow
PDF). Host-script tests: new `tests/prod/test_remarkable_sync.py` (one file per script, as `test_prod_script.py`);
skipped without `pdftoppm`.
1. Pull, two new notebooks in `wiki-inbox` (a PDF in it and a notebook outside it are ignored) -> the fake `/ingest`
   receives two requests, `kind=notebook`, two PNG parts each; state has both ids with `pages: 2`; stdout one line each.
   (`test_remarkable_sync.py`)
2. Pull again, unchanged -> no `/ingest` request, "nothing new"; bump one `ModifiedClient` -> one request carrying
   `replaces=<old sha>`. (`test_remarkable_sync.py`)
3. Device down (closed port) -> exit 0, one line `reMarkable not connected`, state file unchanged. App down -> exit 1,
   state unchanged. `/ingest` 500 for one notebook -> the other is filed and recorded, exit 1. (`test_remarkable_sync.py`)
4. No `scripts/vision_ocr` (or `REMARKABLE_OCR=0`) -> `ocr` is a list of empty strings, pull succeeds; a stub
   executable that prints fixed text -> that text arrives in `ocr`; a stub exiting 1 -> empty string, pull succeeds.
   (`test_remarkable_sync.py`)
5. `sources.ingest_notebook` with three PNGs and a `FakeClient` -> three transcription calls each carrying one
   `image_url` part, PNGs under `raw/remarkable/<date>-<slug>/`, transcript has `## Page 1..3` and the OCR block,
   source page `kind: notebook`, one commit, manifest `pages` has three paths. (`tests/sources/test_sources.py`)
6. Same pages again -> `already: true`, zero model calls; a page transcribed as "ok" (< 20 chars) -> `[page 2:
   unreadable]`; all unreadable -> source page filed, no page ops; `replaces` -> old source page and raw files gone.
   (`tests/sources/test_sources.py`)
7. 40 pages of long text -> the summary prompt carries the `[… N tokens skipped …]` marker (head/middle/tail);
   51 pages -> 413 before any model call. (`tests/sources/test_sources.py`)
8. Manual, with the device (the user's): enable the USB web interface, create `wiki-inbox`, write one page; `pull` ->
   a readable source; second `pull` -> nothing new; edit -> replaced. Real listing-shape findings go into the fake
   device before the Implementation gate.

## Out of scope
reMarkable cloud / rmapi, `.rm` vector parsing, an on-device app, pulling annotations on PDFs/EPUBs, creating or
deleting device folders, re-transcribing stored PNGs with a newer model, styled (HTML) PDFs.

**Follow-up (PRD 47b):** the push command ("send that to my reMarkable", ask-then-`is_affirmative`), the `serve`
mode on :8021 (`POST /push`, `cupsfilter` PDF, `POST /upload`), markdown→PDF rendering, the launchd plist and any
scheduling, the Swift helper `scripts/vision_ocr.swift` and its `swiftc` build.
