Status: done

# 47b — reMarkable push: send a card or page to the tablet

## Overview
PRD 47 pulls handwritten notebooks into the vault; this adds the other direction. "Send that to my reMarkable"
renders the last written card (PRD 43) to a PDF and uploads it to the tablet over USB. The
container cannot reach USB (docs/deploy.md), so the push runs in a tiny host service on :8021, the same pattern as
ASR on :8020, with one launchd plist. The Swift Vision OCR helper and the daily-pull plist that PRD 47 deferred
stay deferred (cut 2026-10-17: pull already transcribes through the multimodal model; OCR is a cross-check nobody
asked for yet).
Depends on PRD 52 (approvals: a push is a write, using its pending slot, `resolve` action and `decision` event) and
PRD 51 (run cards: `card`/`card_update`, run page); it adds the `push` entry to PRD 52's kind registry.

## User stories
- As a user with a card in front of me, I say or type "send this to my reMarkable"; I see a proposal,
  approve it, and the PDF appears on the tablet.
- As a user with the tablet unplugged, I get "reMarkable not connected" and nothing is left pending.
- As a user, I enable the push service with one `launchctl load` command and one `.env` line.

## UI/UX
- Command (typed or spoken, whole-turn regex like `_SAVE_NOTE`, chatbot.py:429): "send (this|that|it|the page) to
  my remarkable", "push … to the tablet". Target = the last card (`llm.last_card`, chatbot.py:456); none -> "nothing to send" (nothing spoken beyond that, nothing pending).
- Proposal card (PRD 52 pending slot): "Send the card 'Compound Key at Covenant' (~3 pages) to reMarkable?" with Approve /
  Reject; a spoken or typed "yes" resolves it via PRD 52's `is_approval`.
- On approval a run card (PRD 51) shows running -> done ("uploaded 'Title', 3 pages") and Wanko says one line:
  "Sent to your reMarkable." Failure turns the run card to failed with the service's one-line reason.
- Status row "reMarkable push :8021" joins the services table (red = copy button with its own start command, `.venv/bin/python scripts/remarkable_server.py`; the row only exists when `REMARKABLE_PUSH_URL` is set).

## Technical approach
- `scripts/remarkable_server.py`: stdlib `ThreadingHTTPServer` bound to `127.0.0.1:$REMARKABLE_PORT` (8021),
  no auth beyond loopback (Docker reaches it via host.docker.internal like ASR). Endpoints: `GET /health` ->
  `{"ok":true}`; `POST /push` JSON `{title, markdown}` (or `{title, pdf: base64}`) -> `{ok, pages, name}`; errors
  JSON `{ok:false, error}` with 503 when the tablet is unreachable, 400 on bad input.
- Rendering: `title\n\nbody` written as plain text to a temp file, then `cupsfilter -m application/pdf file`
  (probed: present at /usr/sbin/cupsfilter; a `# T / body` file gave a PDF whose `pdftotext` returns both lines).
  Markdown is left as monospaced text: no styling, no images, tables as typed. Page count from `pdftotext`/`pdfinfo`
  form feeds, else `/Type /Page` count. `pdftoppm` (installed) is used only by pull.
- Upload: `POST {device}/upload` multipart (`file` part, `<title>.pdf`, `application/pdf`) reusing `_multipart`
  from scripts/remarkable_sync.py:175 (generalised with a content type). Device reachability is the same pre-check
  as `list_changed` (scripts/remarkable_sync.py:97-104) -> `reMarkable not connected ({host})`.
- Chat side: `chatbot.py` command `push_command(text)` -> builds a PRD 52 proposal (kind `push`, proposal `{card_id, title, pages}`; the markdown is read from the
  card at apply time, and PRD 52's hash binds the proposal to what was shown); on approve, `remarkable.push(...)` (new small module, `requests` to
  `$REMARKABLE_PUSH_URL`, unset = off; `http://127.0.0.1:8021` locally, `http://host.docker.internal:8021` via `$DATA/.env` in prod). The pages
  estimate in the proposal is computed locally (lines / 60), exact count comes back in the run result.
- `services.py`: `probe_remarkable()` modelled on `probe_asr` (services.py:73): GET `/health`, label via
  `_port_label`; included in `probe()`; "not installed" is not an error when `REMARKABLE_PUSH_URL` is unset.
- launchd: `scripts/launchd/com.voicebot.remarkable.plist` (KeepAlive, like the ASR plist, port 8021, log
  /tmp/voicebot-remarkable.log). docs/deploy.md gets one line.

### Decisions
- Host service over a container-side upload: USB is invisible to Colima; ASR already proves the pattern.
- `cupsfilter` text->PDF over HTML->PDF: HTML did not render in the probe; plain monospaced is enough for reading.
- Approval + run card over fire-and-forget: a push puts a file on a device, so it follows PRD 52/51 like any write.
- Card-only target (no open-page), 2026-10-17: simplify; the open wiki page is out of scope.
- URL unset = off, prod opts in via .env (2026-10-17): compose carries no default; add `REMARKABLE_PUSH_URL=http://host.docker.internal:8021` to `$DATA/.env`. Unset = no command, no row, one "not configured" line.
- Chose a local page estimate shown with "~" in the proposal, exact count only in the run card, over rendering twice.
- Cut (2026-10-17) the Vision OCR helper and the daily-pull plist: simplify over completeness; `ocr_bin()` keeps
  returning "" when no binary is installed, which the pull path already handles.
- Risk: the device `/upload` shape is unverified; mitigation: fake device follows the documented shape and a
  manual scenario checks the real tablet before merge.

## Data model
No new persisted state. Proposal `{card_id, title, pages}` lives in the PRD 52 pending slot `sess.pending = {run_id, kind: "push",
proposal, hash, created}` (one at a time, cleared on the next turn, on resolve, or on failure). Service holds nothing between requests; temp files are removed
after each push. Invariants: nothing is uploaded without an approved proposal; one push = one PDF = one upload.

## Test scenarios
`fake_remarkable` (tests/conftest.py:621) gains `POST /upload` recording `{filename, content_type, bytes}`.
1. `POST /push {title:"Plan", markdown:"# Plan\n..."}` to the service with the fake device -> fake receives one
   upload named `Plan.pdf`, `application/pdf`, bytes start `%PDF`; response `{ok:true,pages:1}`.
   (`tests/prod/test_remarkable_server.py`)
2. Rendering: a 150-line markdown -> PDF whose `pdftotext` contains the title and the last line, `pages >= 3`;
   empty body -> 400. (same file; skipped without `cupsfilter`/`pdftotext`)
3. Typed "send this to my reMarkable" after a card -> a proposal with title and page count, pending slot set, zero
   requests to the fake service; no card and no page -> "nothing to send", pending empty.
   (`tests/session/test_remarkable_push.py`)
4. Approve -> exactly one `/push` request carrying the card markdown, run card `done`, one speech line "Sent to
   your reMarkable."; Reject (button `resolve {run_id, decision: reject}`) -> no request, pending cleared, `decision` event. (same file)
5. Device down (closed port) -> `/push` 503, run card `failed`, status line "reMarkable not connected", pending
   empty, no spoken success. (same files)
6. `services.probe()` with the service up -> `remarkable` row ok "answering"; port closed -> not ok with the copy
   command; `REMARKABLE_PUSH_URL` unset -> no row. (`tests/test_services.py`)
7. The plist passes `plutil -lint`, names `scripts/remarkable_server.py`, port 8021, loopback.
   (`tests/prod/test_prod_script.py`, extend the parametrised lint). Manual (the user's): push a card to the real
   tablet and open it.

## Out of scope
Two-way sync, annotations or merging edits back, reMarkable cloud, rendering images/tables/styled HTML, deleting
or listing device documents, choosing a device folder, pushing several items at once, Vision OCR, a scheduled pull.

## Open design questions
None open.
