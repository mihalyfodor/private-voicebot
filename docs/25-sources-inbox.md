Status: implementation

# 25 — Sources view as inbox and provenance

## Overview
The Sources rail item becomes the answer to "what have I fed the assistant and what did it do with it?": every
filed source newest-first, with kind, date, summary line and the pages it touched; per source: open raw file,
re-run ingest, forget. Data already exists in `raw/manifest.jsonl` and the `sources/*.md` pages.

## UI/UX
- List: rows `[kind icon] title · when · N pages fed`, newest first; filter chips by kind (file, screenshot, note,
  email); "file a source…" button stays. Selecting a row shows the source page (existing wiki page renderer) plus
  actions: **Open raw** (`obsidian://` or file link), **Re-ingest** (re-runs summary + ops from the stored raw file;
  the source page is rewritten, touched pages get new dated notes, one commit), **Forget** (confirm; deletes the
  source page and raw file, removes the manifest line, and removes notes whose `source:` is this raw path from the
  pages it fed; one commit — reversible via git).
- Rail badge: count of sources filed since the view was last opened (session-local).

## Technical approach
- `sources.list_sources()` from the manifest joined with page frontmatter; `sources.reingest(sha)`;
  `sources.forget(sha)`; note removal = drop bullets under `## Notes`/`## History` whose `(source: <raw>)` matches,
  via `wiki._edit_page`. Actions `get_sources`, `reingest_source {sha}`, `forget_source {sha}`; `sources` WS message.

## Also (found 2026-08-29 on the first real screenshot)
- Ingest must never create a `people/` page for the user themselves: the summary/ops prompt gets the profile's
  `identity.name`/`nickname` and a rule "the user is the profile owner, not a person page; attribute their side
  as 'the user'"; ops targeting `people/<user-slug>` are rejected in `restrict_ops`. Fuzzy match on the profile
  name (OCR misspellings like "Mihay") counts as the user.
- **Already done by PRD 26** (identity aliases): `memory.is_user` + `user_names`, the "the user" rule in both ingest
  prompts, `restrict_ops` → `wiki.reject_user_pages` refusing `people/<user>` pages and link targets, "the user (…)"
  entity lines, and the fuzzy match. Covered by `tests/test_sources.py::test_a_document_naming_the_user_files_no_page_about_them`
  and `test_restrict_ops_rejects_pages_about_the_user`. Nothing left to build for scenario 0.

### Decisions
- Rows are ordered by manifest order reversed (append-only file = filing order); no manifest schema change.
- The Sources view keeps sharing `#view-wiki`: the left list swaps to source rows + kind chips, the right pane
  is the wiki page renderer with three extra head buttons. No second page renderer.
- `reingest` reuses the ingest core (`_file`): same summary + ops prompts, same source page path, the manifest
  line is replaced in place (sha unchanged, `pages_touched` = the union of old and new).
- `forget` scans only the manifest's `pages_touched` for `(source: <raw>)` bullets under `## Notes`/`## History`;
  decisions blocks and `## Related` links stay (reversible via git anyway).
- `_send_wiki` also pushes `sources` so the badge sees every filing (popup, voice tool, session ingest).

## Test scenarios
0. A screenshot whose text names the user (incl. a misspelling) creates no `people/` page for them; other people still get pages. *(covered by PRD 26, see above)*
1. `list_sources` orders newest first and reports pages fed from the manifest.
2. `reingest` on a stored raw file produces a new summary (mock), rewrites the source page, adds notes, one commit.
3. `forget` removes page, raw, manifest line and only the matching notes from fed pages; other notes untouched; one commit; lint clean.
4. Actions round-trip through the Panel fixture; forget without confirm flag → error.
5. Manual: file a screenshot, see it in Sources, forget it, confirm the project page lost only that note.

## Out of scope
Bulk operations, search within sources, editing summaries by hand (use Obsidian).
