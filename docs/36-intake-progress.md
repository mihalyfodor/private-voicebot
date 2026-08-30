Status: refactoring

# 36 — Bulk intake progress

## Overview
Dropping a dozen EPUBs (1–22 MB each) shows "uploading…" for many minutes: the popup sends the whole batch as one
multipart POST and the server files the items one after another (one 10–30 s model summary each) before it replies.
The user cannot tell whether anything is happening, what is done, or whether re-dropping is safe. This PRD makes the
batch visible: one request per file, per-file state in the popup, and a `summarising` cue pushed from the server.

## User stories
- As a user, I drop 12 books and watch each row go queued → uploading → summarising → filed (with the page title,
  linked); the popup tells me "4 of 12" and how long it has been running.
- As a user, one broken file in the middle shows `error (reason)` and the rest still get filed.
- As a user, I can Cancel mid-batch: the file in flight finishes, the rest stay queued, nothing is half-done.
- As a user, I know from the popup that re-dropping the same file is harmless ("already filed").

## UI/UX
- The intake popup (PRD 20) keeps its shape; each item row gains a state on the right:
  `queued` · `uploading…` · `extracting…` · `summarising…` · `filing…` · `filed → <page title>` (link) ·
  `already filed` · `error — <reason>`. Rows are coloured by state (muted / accent / green / grey / red).
- The status line reads `N of M · m:ss` while the batch runs and `k filed, j already filed, e errors · m:ss` after.
- The popup stays open until the batch ends. The File button disables on submit; the Cancel button reads
  **Cancel** while running (stops after the current file), then **Close**. Esc behaves like the button.
- A single-file batch that files cleanly closes itself as before (toast + the new page opened in the Wiki view).
  A multi-file batch stays open; a filed row's link closes the popup and opens that page.
- A footnote under the list: "re-dropping a file that is already filed is a no-op".
- The Sources rail badge ticks up per file (the server pushes `wiki` + `sources` after every request already).

## Technical approach
- **Client** (`static/app.js`): `submitIntake` becomes a sequential loop over the usable items in list order — one
  `POST /ingest` per file (pasted text is a single item, unchanged). Each response updates that row and the counters;
  a failed request marks the row `error` and the loop continues. `intakeCancel` stops the loop between files.
  `ingest_progress` messages set the current row's stage. An interval re-renders the elapsed time every second.
- **Server** (`chatbot.py`, `sources.py`): `file_bytes` takes an optional `on_stage(stage)` callback, called before
  extraction, before the summary completion and before the page/ops write. `_ingest_all` passes a callback that pushes
  `{"type": "ingest_progress", "filename", "stage", "i", "n"}` through `make_sender(_loop)` — the same loop-safe
  path `_push_saving` uses. `i`/`n` index the items of the request (1-based); with the new client that is `1/1`.
  The `/ingest` contract is otherwise unchanged (it already accepts one or many files per call).
- No change to the on-disk format, the manifest or idempotency (sha match → `already: true`).

### Decisions
- One request per file over a streaming response because the endpoint, its tests and the WS push per request already
  exist; the browser gets per-file results for free and a mid-batch failure loses nothing.
- Stage pushes over WS rather than polling: the ingest thread already has a loop-safe sender.
- The client owns the counters (it knows the list); the server only reports stage within one request.
- Cancel stops between files, never mid-file: an interrupted ingest would leave a raw file without its page/commit.
- Assume a batch is dropped from one popup; a second drop while a batch runs is refused by the existing `intakeBusy`
  guard (the drop still shows the overlay, the popup does not reopen).
- Risk: a progress message for a stale batch; mitigation: the client only applies it to the row currently uploading.

## Data model / state shape
- Client: `intakeItems[i]` gains `state` (`queued|uploading|extracting|summarising|filing|filed|already|error`),
  `result` (`{path, raw}`) and `reason`; module state `intakeCancel`, `intakeStarted` (ms), `intakeTimer`.
- WS message: `ingest_progress {filename: str, stage: "extracting"|"summarising"|"filing", i: int, n: int}`.
- No persisted data changes.

## Test scenarios
1. Unit (`tests/test_sources.py`, `http` fixture, mocked model): one request with `one.md` + `two.md` → the socket
   receives `ingest_progress` in order `extracting, summarising, filing` for `one.md` (i=1, n=2) then the same for
   `two.md` (i=2, n=2), before the `wiki` push; posting `one.md` again → `already: true` and no progress message.
2. Node fixture (`tests/app_fixture.mjs`): three files submitted with a scripted fetch → rows read `uploading…`,
   `queued`, `queued`; an `ingest_progress {stage: summarising}` flips row 1 to `summarising…`; the first response
   makes row 1 `filed → Falcon status` (linked) and row 2 `uploading…`; Cancel then the second response (`already`)
   leaves row 2 `already filed`, row 3 `queued`, exactly two fetch calls, status `2 of 3 …`, button label `Close`.
3. E2E (`tests/e2e/test_sources.py`): drop two generated `.md` files → two rows reach `filed`, the status line says
   `2 filed`, screenshot; Close → popup hidden, both sources listed in the Sources view.
4. Manual: drop 12 EPUBs, watch the rows; re-drop one → `already filed` in seconds.

## Out of scope
Parallel uploads, resumable uploads, a streaming `/ingest` response, cancelling a file mid-ingest, retry buttons,
progress for the voice `file_source` tool or session re-ingest.
