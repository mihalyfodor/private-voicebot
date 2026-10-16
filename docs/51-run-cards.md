Status: done

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
  page bodies stay in the wiki where Wanko's editor shows them.
- Assume the run's events are in the current rotation window; if a run aged out, the 404 says so — no index.
- Risk: a new `detail` string leaks text; mitigation: details are built from counts/titles only and the PRD 49
  sentinel test is extended to the run page HTML.
- Chose optional `markdown` on `card_update` over emitting the write-up card only at the end: the body arrives with
  the final status and the card keeps its running state.
- Chose an empty run-card body for the briefing (header + steps + Open run) over a body: "write it up" still produces
  the PRD 46 content card.
- Chose `commit` on `ingest_end` (one `rev-parse` after the commit) over matching `git log` by time window, which
  mis-assigns when two writers commit in the same second.
- Chose deriving host-run status on the page over a `POST /runs/{id}/end` from the script: no extra write path.
- Chose serving run pages in prod with the same origin check as `/ingest`, no extra auth and no text, over auth or
  dev-only: the app is localhost-bound; revisit if the port is ever exposed.
- Chose content-label origin from the commit subject (`wiki: edit` -> you, ingest -> source, page ops -> Wanko) via the
  existing `_USER_COMMIT`/`_CHAT_COMMIT` regexes (wiki.py:1388-1389) over a new field.
- Implementation: the trace tests live in `tests/tracing/` (there is no `tests/trace/`): scenarios 1, 2, 6, 7 are
  `tests/tracing/test_runs.py`, 3–4 `tests/tracing/test_run_page.py`.
- Implementation: the card has no reason field, so a failed run's reason sentence is the `detail` of its failed step
  (`{label: write, status: error, detail: "Failed with RuntimeError."}`); the page shows the same sentence from `runs.REASONS`.
- Implementation: besides the eight new names, `FIELDS` gains `part` (a write-up's `{title, part}` sources), `memory`
  with `added`/`updated`/`removed` (the briefing's memory counts) and the step names (`gather`, `skip`, `phrase`,
  `context`, `write`, `file`) in the `name` vocabulary; `reason` is a fixed shape (the `REASONS` codes or
  `failed-<ExcType>`) so a stray string becomes "other"; a `step` body may set its own `status` (`skipped`).
- Implementation: `runs.REASONS` adds `housekeeping` for the briefing's skipped-commits row ("2 housekeeping commits").
- Implementation: workflow codes are `briefing`, `write-up`, `remarkable`, `batch`, `ingest`; a host run's workflow
  is `remarkable` when its first `/ingest` holds a notebook; workflow-less runs are labelled from the run id
  (`rm-`, `batch-`) on the page.
- Implementation: the briefing's run card has id `run-<run_id>` and is sent once with its final status (no running
  card — the whole turn is one gather and one completion); it is not replayed on reconnect (the briefing is not a
  card in the conversation); the write-up card keeps its `cardN` id and replays with its last status. "write it up"
  before any answer stays the plain turn error, no run.
- Implementation: the briefing's `gather` step carries the digest pages as `{path, commit, title}` (titles are
  allowed fields), so the page's Did rows show title, path, short sha and the `--stat` summary of that commit; a
  housekeeping commit is a `_NOISE_COMMITS` subject or one touching only index/log/schema.
- Implementation: the batch import is a run only when it writes (`--dry-run` traces no run and no `step` spans).
- Review fix: the batch run's `run end` is in a `finally`; an exception or Ctrl-C ends it `failed` with reason
  `failed-<ExcType>` (`failed-KeyboardInterrupt`), no separate `aborted` code.
- Review fix: the write-up card is kept in `llm`'s card store from the moment it goes out `running` (and when it
  fails), so a reconnect replays it with its last status where it was issued; card ids come from a counter that
  never reuses one (only `llm.reset` restarts it); a card without a body is "nothing to save"; a `card_update`
  with `markdown` for an unknown id creates the card on the client.
- Review fix: `card_update` also carries `title?` and `sources?` with the write-up's body, so the live card and its
  replay match.
- Review fix: `ingest_end.commit` is recorded only when HEAD moved across the filing commit.

## Data model
- Card: `{type: card, id, markdown, sources, title, run_id?, workflow?, status?, steps?: [{label, status:
  pending|running|ok|skipped|error, detail?}] ≤ 6, actions?: [{id, label, kind: open|copy|approve|reject}],
  run_url?: "/runs/<run_id>"}`; `card_update {type, id, status, steps?, actions?, markdown?, commit?}`.
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
None open.
