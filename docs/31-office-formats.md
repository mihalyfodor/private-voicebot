Status: refactoring

# 31 — Office formats, calendar and transcripts through the same drop

## Overview
The drop gesture (PRD 20) takes md/txt, screenshots, pdf and Outlook `.eml`/`.msg`. Work artefacts are mostly Word,
PowerPoint, Excel, Teams transcripts and calendar exports. This PRD adds those as extractors behind the same
`extract()` seam so nothing else changes: one raw file, one summary page, dated notes on the pages it feeds.
Transcripts and `.ics` additionally land as `meetings/` pages, because that is where commitments live.

## User stories
- As a user, I drop a `.docx`/`.pptx`/`.xlsx` from Finder or Outlook and get a source page with a real summary.
- As a user, I drop a Teams transcript (`.vtt`, or the `.docx` Teams exports) and get a `meetings/<date>-<slug>` page
  with attendees, decisions and action items, linked from the people/project pages.
- As a user, I drop an `.ics` (one event or an exported calendar) and get `meetings/<date>-<slug>` stubs with time,
  attendees and the description — so "Today" already knows the day.
- As the developer, every new format is one extractor function + one test fixture; kinds show up in the Sources chips.

## UI/UX
- No new screens. Sources list kinds gain `docx`, `pptx`, `xlsx`, `transcript`, `calendar` (chips, icons).
- Transcript/calendar source pages carry a **Meeting** link to the `meetings/` page they produced.
- Drop of an unsupported/broken file keeps today's "unreadable" behaviour (page with `unreadable: true`).

## Technical approach
- Extractors in `sources.py` (same `(text, hint) ` shape): `.docx` via `python-docx` (paragraphs + tables, headings as
  `#`), `.pptx` via `python-pptx` (one `## Slide N — title` per slide, notes included), `.xlsx` via `openpyxl`
  (read-only; per sheet `## <name>`, first 200 rows as a markdown table, formulas' cached values), `.vtt` via a
  small parser (merge consecutive cues of the same speaker into `**Speaker:** text` turns; drop timestamps except a
  `duration` line), `.ics` via `icalendar` (VEVENT → date, time, title, organizer, attendees, description; recurring
  events expanded for the next `calendar_horizon_days` (setting in settings.json, default 14, editable in the
  Settings view next to the decay windows — the user plans the week on Monday mornings). Teams' `.docx` transcript is detected by its `Speaker  timestamp` line
  pattern and routed to the transcript path.
- `KINDS` grows accordingly. **Large text:** the byte cap for text kinds (md/txt/docx/pptx/xlsx/vtt) rises to 10 MB
  (images/pdf unchanged); when the extracted text exceeds `RAW_CHUNK_TOKENS` the model is shown head (60 %), a middle
  slice (15 %) and tail (25 %) with `[… N tokens skipped …]` markers, and the source page frontmatter records
  `shown_tokens` / `total_tokens`. No chunk-and-merge (separate PRD if full coverage is ever needed).
- Meeting pages: a second prompt for `transcript` kind returns `{title, date, attendees, decisions[], actions[], summary}`;
  ops = `new_page meetings/<date>-<slug>` + `append_note` on each attendee's people page and the project page
  (existing validated ops, `is_user` rule applies — the user is "the user", never a people page). Calendar VEVENTs
  skip the LLM: the meeting stub is written deterministically (`type: meeting`, `time`, `attendees`, `source`).
- Idempotency stays sha-based; re-dropping the same `.ics` is a no-op, an updated one creates only new stubs
  (existing `meetings/<date>-<slug>` pages get a dated `updated invite` note, never a rewrite).
- The meeting page itself is written by `sources.py` the way the source page is (the `new_page` op cannot carry
  `date`/`time`/`attendees` frontmatter or the meeting sections); the attendee/project notes go through the validated
  ops. A transcript makes two completions (summary, meeting) — the meeting prompt replaces the generic page-ops
  prompt for that kind and may name a `project`.
- Calendar horizon: a one-off VEVENT always gets its stub (past or future — an exported calendar is history too);
  only recurring events are expanded, from today to today + `calendar_horizon_days`.

### Also: pasted text is not titled by its first line
The intake pre-filled the title with the first line of a paste, so the source page was named after a header the
model would have summarised better. Now `paste_text` — and a dropped file with a generic name (`note.txt`,
`untitled`, `pasted text`) — leaves the title field empty with the placeholder "model picks a title"; the summary's
`title` wins, "file under" stays "auto". The summary prompt asks for a short descriptive `title` (not a quote of the
first line) and a `topic` tag (e.g. gaming, work/project, finance) that lands in the source page's `tags:`.

### Decisions
- Deterministic stubs for `.ics`, LLM only for transcripts: invites are structured, transcripts are not.
- Head/middle/tail sampling over chunk-and-merge for big text: one call, and tagging/summary only needs the shape.
- No OneNote: proprietary and not on disk; PDF export covers it (Graph API spike parked).
- No folder watching: drop-to-app stays the only entry (user decision 2026-08-30).
- Assume Teams exports keep the `Speaker  0:00` line shape; if false, the `.docx` falls back to the generic path.
- Risk: xlsx blobs are huge; mitigation: 200 rows/sheet, 20 sheets, cell text ≤ 80 chars, and the cut is stated.

## Data model
- `raw/manifest.jsonl` entries unchanged (+ `kind`); source frontmatter `kind` gets the five new values; transcript /
  calendar source pages add `meeting: meetings/<date>-<slug>`.
- `meetings/<date>-<slug>.md`: `type: meeting`, `name`, `date`, `time` (optional), `attendees: [..]` (people links),
  `source`, `updated`, `seen`; body `# title / ## Summary / ## Decisions / ## Actions / ## Notes`.
- New deps: python-docx, python-pptx, openpyxl, icalendar (all pure Python).

## Test scenarios (fixtures generated in-test with the same libraries; no binaries checked in)
1. `.docx` with headings, paragraphs and a table → extracted markdown keeps heading levels and the table rows; source page kind `docx`.
2. `.pptx` with 3 slides + notes → `## Slide 1 — <title>` sections, notes under each; kind `pptx`.
3. `.xlsx` with 2 sheets, 300 rows → per-sheet table capped at 200 rows with a "200 of 300 rows" line; kind `xlsx`.
4. `.vtt` with interleaved speakers → turns merged per speaker, timestamps gone, `duration:` line present; mocked LLM →
   `meetings/<date>-<slug>` page with attendees/decisions/actions, `append_note` on each attendee's people page,
   the user never gets a people page; source page links the meeting; one commit.
5. Teams `.docx` transcript is detected and takes the transcript path; a normal `.docx` does not.
6. `.ics` with 2 events (one recurring weekly) → deterministic stubs for the next `calendar_horizon_days` only (14 by
   default; 28 when the setting says so; the Settings view round-trips it), no LLM call, one commit; re-dropping the same file → no-op; an updated invite for an existing stub → one `updated invite` note.
7. Broken/empty file of each new kind → `unreadable: true` page, no exception, lint clean.
8. Sources view shows the new kind chips and the **Meeting** link (fixture render); e2e: drop a generated `.docx`
   via the existing synthetic-drop helper, see it in Sources (screenshot).
9. A 3 MB `.txt` (generated) is accepted; the prompt gets head/middle/tail with skip markers; `shown_tokens` <
   `total_tokens` on the page; a `.png` over the unchanged 4 MB image cap (and a 3 MB `.pdf`) is still rejected.
10. Paste a text whose first line is a long FAQ header → source page name = the (mocked) model's title, `tags:` carry
    the topic; the intake fixture shows an empty title field for `paste_text` but keeps the filename-derived title
    for a real file drop.
11. Manual: drop a real Teams transcript and an Outlook `.ics`; check the meeting page and the "Today" note; drop a
    3 MB game FAQ `.txt` and check it is tagged as gaming.

## Out of scope
OneNote, folder watching, Graph/IMAP sync, email thread grouping, `.doc`/`.ppt`/`.xls` legacy binaries, images inside
documents, calendar write-back.
