Status: exploration

# 51 — Run cards and run pages

## Overview
Multi-step work (a briefing, a write-up, a reMarkable pull, a batch import) leaves no trace the user can read: the
card shows only the result, and PRD 49's `events.jsonl` is for the owner's report script. This PRD gives every
multi-step or writing workflow a **run**: its card gains a status, ≤ 6 steps and an "Open run" link, and
`GET /runs/{run_id}` renders one server-side page from the run's events — what it did, what it skipped and why,
what failed. Chit-chat stays cardless; nothing on the card or page claims a confidence it does not have.

Depends on PRD 49 (event trace: `run_id`, spans). Consumed by PRD 52 (approvals), PRD 47b (push) and PRD 54 (research steps).

## User stories
- As a user, I say "what's new?" and, besides the spoken recap, see a small card "Briefing · done" with three steps
  ("gathered since 14:05 — 3 pages, 2 sources, 4 memory changes", "skipped 2 housekeeping commits",
  "phrased, 2.1 s").
- As a user, I click **Open run** and a new tab shows the run: when, how long, outcome in one sentence, a timeline,
  the pages touched (with their commit), the sources used, and what was left out and why.
- As a user, when a write-up fails the card says "Write-up · failed" and the reason sentence, not silence.
- As a user, after a reMarkable pull or an overnight batch import, the run page lists every notebook/file filed,
  the pages each touched, and the skipped ones (`needs-ocr`, unreadable pages) with reasons.
- As a user, an unknown or aged-out run id gets one line, not a stack trace.

## UI/UX
- **Run card** = the PRD 43 `.msg.card` (`addCard`, static/app.js:2796-2842) plus a header line `<workflow> ·
  <status>` (status chip: running/done/nothing/failed/awaiting), a step list (label · status glyph · detail, max 6)
  above the body, and buttons from `actions`: `open` → "Open run" (`<a href=run_url target=_blank rel=noopener>`),
  `copy` → the existing Copy. `approve`/`reject` render nothing until PRD 52 (which also adds the `rejected` status). Copy and Save to wiki stay for cards
  with a markdown body; a run card with an empty body shows only header, steps and actions.
- `card_update {id, status, steps?, actions?}` (`id` = the card id; PRD 52 adds `commit?`) replaces the chip, step list and buttons of that card in place.
- **Run page** `/runs/{run_id}`: plain HTML using static/style.css. Header (workflow, local start time, duration,
  outcome chip + reason sentence); Timeline (child spans by `start`: name · ms · status · count); **Did** (pages
  touched → short sha + `git show --stat` summary line, each labelled *source*, *Wanko* or *you*; sources used as
  `title · part n/N`; memory ops as counts); **Did not** (skipped items + reason); Errors (`exc_type` + first line of
  `message`). Footer: "Runs age out with log rotation (5 × 10 MB)." Unknown id → 404 "No run <id> (it may have aged
  out of the log)."
- Scores, bands and counts appear as facts ("band strong, top score 0.71"), never as "confidence".

## Technical approach
- **Events** (two kinds added to PRD 49's list, both through `trace.py`):
  - `run` point event: `{workflow, phase: start|end, status?, reason?}`; `start` opens, `end` closes with a status
    and a reason code. Emitted by `chatbot.briefing` (chatbot.py:392), `chatbot.write_up` (chatbot.py:471), and
    `batch_import.ingest_batch` (batch_import.py:160). Host-script runs over `/ingest` (chatbot.py:797, run id via
    `trace.run`, chatbot.py:766-773) get `run start` on the first ingest with an unseen `run_id`; their end is
    derived on the page (all `ingest_end` ok → done, any error → failed).
  - `step`: a `trace.span("step", name=…, count=…, detail=…)` (trace.py:325) so it already carries `span_id`,
    `parent_id`, `start`, `duration_ms`, `status`, `run_id`; `detail` is a short fact string (≤ 200 chars, not in
    `NAMED`, so a long one is dropped, trace.py:217-231). New names `workflow`, `phase`, `reason`, `name`, `count`,
    `detail`, `commit`, `skipped` join `FIELDS` (trace.py:55-60).
- **Workflows wired**:
  - Briefing: steps `gather` (count = pages + sources + memory, detail "since <HH:MM>"), `skip` (housekeeping
    commits filtered by `_NOISE_COMMITS`/index/log, wiki.py:1387; `changes_since`, wiki.py:1478, returns their count
    as `digest["skipped"]`), `phrase` (the existing `briefing` span, llm.py:879). Empty digest → status `nothing`,
    reason `nothing-to-do`, no model step.
  - Write-up: steps `context` (sources + parts as `{title, part}` from the last context), `write` (the `card`
    completion, llm.py:827, ms). The card goes out as `running` at turn start, then `card_update` (done/failed).
  - reMarkable pull: `scripts/remarkable_sync.py` sends one `run_id` (`rm-<yyyymmdd-hhmmss>`) on every
    `post_notebook` (scripts/remarkable_sync.py:189); unreadable pages (notebooks.py:88) become a `skip` step per
    notebook; `ingest_end.pages` (sources.py:1236) and a new `ingest_end.commit` (short sha read after `_commit`,
    wiki.py:222) give the Did rows.
  - Batch import: run id `batch-<ts>`, a `step` per file (`needs-ocr`, unsupported, too-large, already → Did not;
    batch_import.py:145, :194). The script runs while the app is down and so writes its own events: it must call
    `trace.configure()` (today only chatbot.py:14 does), and against prod it runs with `TRACE_PROFILE=prod
    TRACE_DIR=~/docker-data/voicebot/logs` so its lines land in the same `events.jsonl` the app writes; dev against
    `~/docker-data` is refused by `trace.refusal` (trace.py:117).
- **Reason codes** `runs.REASONS`: `ok`, `nothing-to-do`, `failed-model`, `failed-offline`, `failed-<exc_type>`,
  `needs-ocr`, `unreadable`, `unsupported`, `too-large`, `already-filed`, `rejected-op` → one sentence each; the
  card's reason line and the page's outcome sentence both come from this table.
- **Run page** `runs.py` (new, stdlib): `load(run_id)` reads `events.jsonl` + rotated `.1…5` (the
  `scripts/trace_report.py` reader, moved into `runs.py` and imported by the script), `render(run) -> str` with
  f-string templates and `html.escape` on every value; `wiki._git("show", "--stat", "--format=%h %s", sha)`
  (wiki.py:137) for the commit lines — never `git show` with a patch, so no page body reaches the HTML.
  `@app.get("/runs/{run_id}")` in chatbot.py (next to `/`, chatbot.py:856) validates the id with `RUN_ID`
  (chatbot.py:792), returns `HTMLResponse`; 404 otherwise.
- **Card contract**: `card_msg` (chatbot.py:442-444) adds the optional fields; `replay_messages`
  (chatbot.py:1313-1325) replays the last status; docs/protocol.md gets the new `card` fields and a `card_update`
  row, and a line naming `GET /runs/{id}` beside `/ingest` (tests/test_protocol_doc.py:23-42 checks both sides).

### Decisions
- Chose two event kinds (`run` point + `step` span) over deriving runs from existing spans only: `turn_start.origin`
  cannot tell a write-up's workflow from a host run, and skipped items are not spans today.
- Chose a server-rendered page over a client view: opens in a new tab from any card, no app state, nothing to keep
  in sync; first such page, kept to one module with escape-on-every-field.
- Chose events + vault git as the only page inputs: the page shows exactly what prod is allowed to log (no text);
  page bodies stay in the wiki where Obsidian shows them.
- Assume the run's events are in the current rotation window; if a run aged out, the 404 says so — no index.
- Risk: a new `detail` string leaks text; mitigation: details are built from counts/titles only and the PRD 49
  sentinel test is extended to the run page HTML.

## Data model
- Card: `{type: card, id, markdown, sources, title, run_id?, workflow?, status?, steps?: [{label, status:
  pending|running|ok|skipped|error, detail?}] ≤ 6, actions?: [{id, label, kind: open|copy|approve|reject}],
  run_url?: "/runs/<run_id>"}`; `card_update {type, id, status, steps?, actions?}`.
- Run (built per request, not stored): `{run_id, workflow, start, end?, status, reason, steps[], did: {pages:
  [{path, commit, stat, origin: source|wanko|user}], sources: [{title, part}], memory: {added, updated, removed}},
  did_not: [{item, reason}], errors: [{exc_type, first_line}]}`.
- Invariants: a cardless chit-chat turn emits no `run` event; every `run start` from the app has one `run end` (the
  same rule as `turn_start`/`turn_end`); one card per run, keyed by `run_id`; steps beyond 6 fold into "+n more".

## Test scenarios
1. tests/trace/test_runs.py: fixture "what's new?" turn → `run start`/`run end` and three `step` events share the
   turn's `run_id`; the `card` has `run_url == "/runs/<run_id>"`, `workflow: briefing`, 3 steps, status done.
2. tests/trace/test_runs.py: empty digest → status `nothing`, reason `nothing-to-do`, no `llm_call` event.
3. tests/trace/test_run_page.py: `GET /runs/<id>` for scenario 1 → 200 HTML listing the touched pages with short
   shas and `--stat` lines, a "Did not" row "2 housekeeping commits"; every value escaped (`<b>` title shows as
   text); no page body text from the fixture vault in the HTML.
4. tests/trace/test_run_page.py: `GET /runs/nope-1` → 404 with the aged-out sentence; `GET /runs/a%20b` → 404.
5. tests/sources/test_sources.py: two notebook `/ingest`s with `run_id=rm-1` (one with an unreadable page) →
   both `ingest_end` have `pages` and `commit`; the run page lists both notebooks, their shas and
   "page 2: unreadable".
6. tests/trace/test_runs.py: write-up whose completion raises → `card` running then `card_update` status failed with
   the reason sentence; the run page shows `exc_type`; the sentinel `ZQX-SENTINEL` from the context is absent from
   events.jsonl and from the page.
7. tests/trace/test_runs.py: `batch_import.ingest_batch` over 1 PDF + 1 image-only PDF → one run, a `needs-ocr`
   Did-not row; with `TRACE_PROFILE=dev` and `WIKI_PATH` under `~/docker-data` the script exits 2.
8. tests/ui/test_run_cards.py (via tests/app_fixture.mjs): a card with steps renders the chip, 3 step rows and an
   "Open run" link with `target=_blank`; `card_update` swaps chip and steps in place (same element); a card without
   run fields renders exactly as PRD 43 (existing tests/llm/test_written_answers.py card assertions green).
9. tests/e2e/test_app.py (live): "what's new?" → the card's "Open run" opens a page whose header names
   "Briefing" → screenshot `/tmp/e2e/run-page.png`.
10. tests/test_protocol_doc.py green with `card_update` and the new `card` fields.

## Out of scope
Approvals and the `approve`/`reject` handlers (PRD 52); streaming each step live while it runs (only start and end
states); a runs list/index page; retention beyond log rotation; React or any client framework; runs for chit-chat.

## Open design questions
- **Should `card_update` carry `markdown`?** The write-up's body arrives with the final status, so `{id, status,
  steps?, actions?}` cannot fill it. Recommendation: add optional `markdown`; the alternative is emitting the write-up
  card only at the end (no running state).
- **Briefing card body.** Today the briefing has no card until "write it up" (llm.py:803). Recommendation: the run
  card body is empty (header + steps + Open run); "write it up" still produces the PRD 46 content card.
- **Commit sha source.** Recommendation: add `commit` to `ingest_end` (one `rev-parse` after the commit) over
  matching `git log` by time window, which mis-assigns when two writers commit in the same second.
- **Host-run end.** reMarkable runs have no end event. Recommendation: derive status on the page; the alternative is
  a `POST /runs/{id}/end` from the script — one more write path.
- **Run page exposure in prod.** The page has titles, paths and shas but no text. Recommendation: same origin check
  as `/ingest` and no extra auth (localhost-bound app); revisit if the port is ever exposed.
- **Content labels.** Recommendation: origin from the commit subject (`wiki: edit` → you, ingest → source,
  page ops → Wanko) via the existing `_USER_COMMIT`/`_CHAT_COMMIT` regexes (wiki.py:1388-1389).
