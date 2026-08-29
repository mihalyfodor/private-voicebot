Status: implementation

# 20 — Raw ingestion: the office document stack feeds the wiki

Slice "17c" of [17-long-term-memory.md](17-long-term-memory.md). Depends on PRD 19.

## Overview

Give the wiki real sources beyond conversation: files the user drops (markdown, txt, PDF), emails the user asks
to file, and news items the user flags. Each source lands immutably under `~/llm-wiki/raw/`, gets one summary
page, and the model proposes page ops (PRD 19) against the existing people/projects/decisions pages with
`source:` pointing at the raw file. Nothing is ingested without the user asking.

## User stories

- As a user, I drop a PDF onto the workspace and, a moment later, `raw/inbox/<file>` exists, a `sources/<slug>.md`
  summary page is linked from the index, and the relevant project page has a dated note citing it.
- As a user, I say "file that email from Anna" and the assistant fetches it (existing gmail tool), stores it under
  `raw/email/`, and updates Anna's page.
- As a user, I can always open the raw source from the page that cites it.
- As the developer, the ingest is idempotent: dropping the same file twice does nothing the second time.

## UI/UX
- **Intake popup — nothing is filed automatically.** Any drop (files) or paste (⌘V with an image or text on the
  clipboard) on the workspace opens a small modal: thumbnail/first lines of what arrived, editable title, "file
  under" page picker (pre-filled with the open wiki page if any, else "auto"), an optional one-line hint ("this is
  the Q3 plan"), and **File / Cancel**. Esc = cancel; nothing touches the vault until File. Multiple files → one
  popup listing them. Accidental drags therefore cost one Esc.
- Drop zone: whole workspace shows a `dragover` overlay "drop to file"; the popup opens on drop. Files are uploaded
  via `POST /ingest` (multipart, with `title`/`hint`/`page` fields) — the only HTTP endpoint besides the static app;
  the WS then pushes `wiki` updates.
- **Screenshots from the macOS clipboard** (⌘⇧⌃4 then ⌘V in the app; also drag from the screenshot thumbnail):
  the `paste` event's image blob goes through the same popup; on File it is stored under `raw/screens/<date>-<slug>.png`
  and Gemma (multimodal via oMLX — verified 2026-08-29 with an `image_url` content part) writes the summary page:
  what the screenshot shows, any visible text (transcribed), entities. Pasted text goes to `raw/inbox/<date>-note.md`.
- Wiki tab: a **Sources** folder in the page list; each source page shows the summary, the link to the raw file
  and the pages it touched.
- Voice: "file this / remember this email / save that article" → fillers + tool `file_source(kind, ref)`.

## Technical approach
- `raw/` layout: `raw/inbox/<YYYY-MM-DD>-<slug>.<ext>`, `raw/screens/<date>-<slug>.png`, `raw/email/<id>.md`, `raw/news/<slug>.md`;
  `raw/manifest.jsonl` with sha256 → path, ingested-at, pages touched (idempotency + provenance).
- Text extraction: images (png/jpg, ≤ 4 MB, downscaled to 1568 px long edge before sending) via the multimodal model — the summary prompt gets the image plus the hint; markdown/txt as-is; PDF via `pypdf`; `.eml` via stdlib `email` (headers block + text/plain part), `.msg` via `extract-msg` (Outlook drag-outs; new deps: pypdf, extract-msg); size cap 2 MB / 40 k tokens, the
  model sees the first `RAW_CHUNK_TOKENS`=6 k plus a "truncated" marker (no chunk-and-merge in this PRD).
- `wiki.ingest_source(path, hint_page=None)`: extract → summary prompt (returns `{title, summary, entities}`)
  → `new_page sources/<slug>` → page ops proposal with the same op set as PRD 19, restricted to pages listed in
  the index plus new people/projects pages → validate/apply/commit. Runs on the worker thread under the memory
  lock (same as `save`).
- Tool `file_source(kind: "email"|"news"|"note", ref)`: email = gmail message id from the last `get_emails`
  result; news = url/index from the last `get_news`; note = free text spoken by the user → `raw/inbox/<date>-note.md`.
- Endpoint `POST /ingest` (multipart, `hint` field) — 127.0.0.1 only, origin check like the WS.

### Decisions
- Files copied into the vault, not linked: Obsidian can open them, git tracks them, provenance survives moves.
- Manifest by content hash: idempotent drops, and the source page can list every page it fed.
- Single summary + ops pass, no chunking: personal-scale documents; revisit if a source > 40 k tokens shows up.
- Only user-triggered ingestion; no mailbox polling, no folder watching (out of scope, deliberately).
- Risk: PDFs with garbage extraction. Mitigation: summary prompt reports "unreadable", page is created with the
  raw link only, user sees it in lint.

## Data model
`sources/<slug>.md` frontmatter: `type: source, name, kind: pdf|md|txt|email|news|note, raw: raw/…, sha256,
tags, source: user drop <date>, updated`. Body: `## Summary`, `## Entities` ([[links]]), `## Fed` (pages touched).

## Test scenarios (unit, tmp vault, mocked LLM)
0. `POST /ingest` without an explicit `confirm: true` field → 400 (the popup is the only sender; guards against
   any future auto path). Cancel in the popup → no request at all (frontend unit check by fixture).
1. Drop a .md file → copied under `raw/inbox/`, manifest line, `sources/<slug>.md` with summary from the mock,
   one commit; the same bytes again → no new files, no commit, reply "already filed".
2. PDF extraction via pypdf on a 2-page fixture → text passed to the mock contains a known phrase; unreadable PDF
   (fixture with no text layer) → page created with "unreadable" summary, no ops applied.
3. Page ops proposed for a page not in the index and not people/projects → rejected; others applied; `source:` on
   every applied op equals the raw path.
4. `file_source("note", "…")` → raw note file + source page; `file_source("email", id)` with a mocked gmail
   fetch → `raw/email/<id>.md` with headers block; unknown ref → error string, nothing written.
5. `POST /ingest` via TestClient: multipart upload → 200 + `wiki` push; > 2 MB → 413; wrong origin → 403.
6. Size cap: a 100 k-token text → the prompt contains the marker and ≤ `RAW_CHUNK_TOKENS` of text.
7b. Harness (live) `sources.yaml`: drop a fixture markdown about "project Falcon" via an `ingest:` turn, then probe
   "what's the status of Falcon?" → expect keyword from the file; run log shows `read_page` or `search_wiki`.
8. Image: a PNG fixture with rendered text → FakeClient receives an `image_url` part and the hint; page created with
   `kind: screenshot`, raw file under `raw/screens/`; > 4 MB → popup refuses before upload (frontend) and server 413.
9. Manual: drag a PDF onto the workspace → overlay → popup → Esc files nothing; again → File → source page and
   project note; open the raw file from the page in Obsidian. ⌘⇧⌃4 a region, ⌘V in the app → popup with thumbnail
   → File → page with the transcribed text.

### Work Outlook intake (user question 2026-08-29)
Spike first (≤ 1 h, not a deliverable): device-code login to Microsoft Graph with delegated `Mail.Read`. If the
tenant grants consent → `tools/outlook.py` mirrors `gmail.py` (`get_work_emails`, and `file_source("email", id)`
fetches by id via Graph). If admin consent is required/blocked → fallback is (a) `.eml`/`.msg` drop (always works)
and (b) an AppleScript "file selected message" command for legacy Outlook for Mac (`selected objects`), posting to
`/ingest`. Never auto-forwarding or IMAP workarounds. Check the org's data policy before any automatic path.

## Out of scope
Chunked/long documents, OCR, mailbox polling, folder watching, non-text attachments, web clipping.
