Status: implementation

# 30 — Command palette (⌘K)

## Overview
One keyboard-first entry point to everything the rail, the Wiki side bar and the Settings view already do: ⌘K opens
an overlay, you type a few letters, ↑↓ Enter runs the command. Parked in PRD 22 ("switch avatar", "open wiki
page…"); since PRD 23 there is one avatar, so the avatar group is backdrop / reply length / voice / mute instead.
Client-only: a registry of commands in `app.js`, no server change.

## User stories
- As a user, I press ⌘K (Ctrl+K), type `atl`, press Enter and the Wiki view opens `projects/atlas`.
- As a user, I type `set` and jump to Settings, `office` and switch the backdrop, `voice` and toggle hands-free.
- As a user, I type something that is not a page title, press Enter on the last row and get a wiki full-text search
  for it in the Wiki view.
- As a user, I press Esc (or ⌘K again, or click outside) and the palette is gone; nothing ran.
- As a user, I can run "Shut down" from the palette but it asks first.

## UI/UX
- `#palette`: a dimmed overlay over the workspace (same look as the intake popup), a card near the top with one text
  input and a list of up to 12 rows. Rows show `label` and, right-aligned, the group name (`view`, `wiki`, `avatar`,
  `sources`, `app`). Wiki page rows show the page path as their detail.
- Open: ⌘K / Ctrl+K anywhere (also while typing — it is the one shortcut that always works); toggles. The input is
  focused and empty on every open.
- Close: Esc, ⌘K, click on the dim, or running a command.
- Keys inside the input: ↑ ↓ move the highlight (wrapping), Enter runs the highlighted row, Esc closes. Mouse: hover
  highlights, click runs.
- Empty query: the fixed commands in group order (navigation, wiki tools, avatar, sources, app), then the wiki pages.
- Non-empty query: matching entries ranked (see below); when the query is non-empty a final row `Search wiki for
  "<query>"` is always present, so Enter never does nothing.
- While the palette is open the global shortcuts (Space mute, 1–5 views) are off; the intake popup and the palette
  never overlap (⌘K is ignored while the intake popup is up).

## Technical approach
- One registry: `PALETTE_SOURCES` is a list of functions, each returning `[{id, label, keywords, group, detail?, run}]`.
  Every area appends its own source next to its code (views, wiki, avatar/settings, sources, app). Sources are called
  on every render, so dynamic sets (the `wiki` page list, the backdrop catalogue, the current hands-free / mute state
  in the toggle labels) are always fresh; nothing is cached.
- Matching: `paletteScore(entry, q)` on lower-cased text. Label prefix 4 · label word prefix 3 · label substring 2 ·
  keyword substring 1 · label subsequence (letters in order) 0.5 · else 0 (dropped). Ties: registry order. Result cap
  12, then the search fallback row.
- The palette reuses existing entry points: `setView`, `selectWikiPage`/`createWikiPage`, `wikiTodayBtn.onclick`,
  `wikiNewBtn.onclick`, `setBackdrop`, `set_verbosity`, `setHandsFree`, `toggleMute`, `wikiIngestBtn.onclick`,
  `sourcesInputEl.click()`, `reload_characters`, and the shutdown handler (extracted to `requestShutdown()`).
- Opening the palette with no page list yet sends `get_wiki` once so the pages appear when it answers.

### Decisions
- Registry of plain objects over a per-view "contribute" API: five tables in one file is enough, and each source
  closes over the state it needs.
- Score by hand-rolled prefix/substring/subsequence over a fuzzy library: no build step, ~20 lines, deterministic.
- Fallback is wiki full-text search, not "create page": creating from a typo is worse than a search that finds
  nothing.
- Shutdown keeps a `confirm()` in the palette (the Settings button stays one click) — a mis-typed Enter must not
  stop the server.
- Assume ≤ ~300 wiki pages; scoring everything on each keystroke is fine. If false, pre-index.
- Risk: ⌘K clashes with a browser shortcut (Chrome uses it for nothing by default; Firefox: search bar) — we
  `preventDefault()` so the app wins when focused.

## Data model / state shape
```
entry   = { id, label, keywords: string[], group: 'view'|'wiki'|'avatar'|'sources'|'app', detail?, run() }
palette = { open: bool, query: string, index: int, items: entry[] }   // items = the rendered rows
```
Invariants: `index` is always within `items` when open (wrapping); `items` are recomputed from the sources on every
query change and on every `wiki` message while open; nothing persists.

## Test scenarios
Node fixture (`tests/app_fixture.mjs`, asserted from `tests/test_palette.py`):
1. ⌘K opens: `#palette` visible, input focused and empty, rows = fixed commands first then the three wiki pages;
   ⌘K again closes; Esc closes; the `1` key while open does not switch views.
2. Ranking: query `atl` → first row is the Atlas page; `set` → first row `Go to Settings` (label prefix beats the
   `settings` keyword on other entries); `wk` matches `Wiki` by subsequence but ranks below any substring match;
   `zzz` → only the `Search wiki for "zzz"` row.
3. Keyboard: with `atl`, ↓ then ↑ returns to index 0; Enter sends `read_wiki_page projects/atlas`, the view is
   `wiki`, the palette is closed.
4. Fallback: `zzz` + Enter → view `wiki`, `#wiki-search` value `zzz`, `wiki_search {query:'zzz'}` was sent.
5. Payloads equal the existing controls: backdrop row sends `set_backdrop {key:'office'}`; verbosity row sends
   `set_verbosity {value:'short'}`; voice row sends `set_hands_free`; mute row flips `muted`; re-ingest sends
   `wiki_ingest`; shutdown asks `confirm` and sends `shutdown` only when it agrees.
6. Click: clicking the second row runs it (view changes to that row's target).

Browser e2e (`tests/e2e/test_palette.py`):
7. ⌘K on the loaded app → `#palette` visible; type `atlas`; Enter → Wiki view with `#wiki-text h1` = `Atlas`;
   palette hidden; screenshot of the open palette and of the result.

## Out of scope
Server-side commands or a command API; recent/frequency ordering; memory-entry commands; command arguments beyond
the search fallback; mobile layout tuning beyond "it fits"; theming.
