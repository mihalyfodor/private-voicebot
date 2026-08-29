Status: refactoring

# 28 — Wiki as a light editor (Obsidian-compatible)

## Overview
Make the in-app Wiki a usable lightweight alternative to Obsidian for everyday work: rendered markdown, whole-page
editing with conflict safety, backlinks, search, tags, create-from-missing-link, today's note. On disk nothing
changes shape — Obsidian stays the heavy editor. Research: 2026-08-29 evaluation (marked 40 kB vendored; no
CodeMirror; optimistic save with content-hash etag; rename/graph explicitly not built).

## User stories
- As a user, I read pages rendered (headings, lists, tables, code), with `[[links]]`, `[[page|alias]]`, `[[page#h]]`
  and `![[embeds]]` clickable, and frontmatter shown as a small strip.
- As a user, I press Edit, change the text in a plain editor, ⌘S saves; if the assistant changed the page meanwhile
  I get "changed on disk — reload / overwrite" instead of a silent clobber.
- As a user, I click a red (missing) link and get the page created; I have a "Today" button for `meetings/<date>`.
- As a user, I see backlinks for the open page, can search, and can filter the list by tag.
- As the developer, the user is the only whole-body writer; the assistant keeps using validated ops; every user save
  is one commit tagged `source: user edit`.

## Fixes folded in (found during research)
- Frontend navigates `[[page|alias]]`/`[[page#heading]]` to the full capture → strip `|`/`#` like `wiki.links_in`.
- `FOLDERS` whitelist blocks user-created pages elsewhere → add `notes/` and allow any single-level folder for
  user-sourced ops (assistant ops stay restricted).
- `_action_wiki_note` applies ops without `memory._LOCK` → all wiki writes take the lock.

## UI/UX
- Wiki view: page header (title, path, breadcrumb folder › page), **Edit / Preview** toggle, Save (disabled while an
  ingest is running or nothing changed), "Open in Obsidian", "→ pull into memory". Body: rendered view
  (vendored `marked`, wikilink/embed pre-pass, never raw innerHTML of user text) or a monospace `<textarea>` (tab
  captured, ⌘S save, Esc back to preview with confirm if dirty). Below: **Backlinks** list; frontmatter strip above.
- Left list: search box (BM25 via `search_wiki`), folder groups + a virtual **Tags** group (from `tags:` and `#tag`);
  clicking a tag filters. "Today" button and "New page" (asks for folder/name) at the top. Recent (MRU, client-side).
- Missing links render red; click → confirm → `new_page` with the link's folder/name.
- Page header menu (⋯): **Rename/move** (dialog with folder + name, shows "N pages link here, links will be
  rewritten") and **Delete** (confirm; "links from N pages will break"). Both are single commits; git is the undo.

## Technical approach
- Actions: `wiki_save {path, text, etag}` (lock → re-hash → `parse_page` validate: required keys, size ≤ 20 kB, bump
  `updated` → atomic write → `rebuild_index` → `log.md` line `edited [[path]] (source: user edit)` → commit →
  push `wiki_page` with new etag + `wiki`); on mismatch `{"type":"wiki_conflict", path, text, etag}`. `read_wiki_page`
  returns `etag = sha256(bytes)[:16]` and `backlinks: [paths]`; `wiki_create {path}`; `get_wiki` adds `tags` per page.
- `wiki_rename {path, new_path}`: lock → validate new path (folder rules, not existing) → move file → rewrite links in
  every backlinking page with exact-path regexes for `[[old]]`, `[[old|…]]`, `[[old#…]]`, `![[old]]`, plus the short
  form `[[<basename>…]]` only when that basename is unique in the vault → bump `updated` on touched pages →
  rebuild index → `log.md` line → one commit `wiki: rename old → new (N links)`; reply `wiki` + `wiki_page` of the
  new path. `wiki_delete {path}`: lock → `git rm` → rebuild index → log → commit; lint reports the broken links.
- `wiki.backlinks(path)` from `links_in` over `page_paths()` (cached with the search index); `wiki.tags(page)`.
- Vendor `static/vendor/marked.umd.js` (MIT, 44 kB — marked ≥ 16 ships no minified build; this is the official
  browser file from the npm tarball) with its licence file; render pipeline: strip frontmatter → replace
  wikilinks/embeds with placeholders → `marked.parse` → parse into an inert document → import only whitelisted
  tags/attributes → swap placeholders for link spans. Without `marked` (the node fixture) the body falls back to
  plain text with the same link spans.
- The pane reads the page through `wiki.page_text` (full text, ≤ 20 kB) — `read_page`'s 4 kB tool cap would
  truncate what the editor saves back. Search box → `wiki_search {query}` → `{"type":"wiki_search", hits}`
  (`wiki.search`, the same BM25 as the tool).
- Rename/delete refuse the machinery pages (`index`, `log`, `schema`) and `decisions`. Saving unchanged text is
  a no-op (no commit). Dialogs (new page, rename, delete, missing link, discard) are plain `prompt`/`confirm`.
- Save preserves the body byte-for-byte except the `updated:` line.

### Decisions
- textarea + marked over CodeMirror/EasyMDE: zero build step, tiny, enough for ≤ 20 kB pages.
- Content-hash etag over mtime: git and our own writes touch mtime.
- Rename rewrites links by exact path (plus unique-basename short form) — no fuzzy matching; graph view, live preview, autosave: not built.

## Test scenarios
1. Save round-trip preserves `[[a|b]]`, `[[a#h]]`, `![[x]]`, `#tag` byte-for-byte; only `updated:` changes.
2. Stale etag → `wiki_conflict`, file unchanged, no commit; fresh etag → one commit, one `log.md` line with
   `source: user edit`, index regenerated, new etag returned.
3. Missing required frontmatter key or > 20 kB → rejected, no write.
4. Save while a mocked `memory.save()` holds the lock → waits, then applies against the post-ingest bytes (conflict
   if the ingest changed the page).
5. `wiki_create` from a missing link makes a valid page in the link's folder and clears the lint broken-link finding;
   user-created `notes/foo` accepted, assistant op to `notes/` still rejected.
6. `backlinks(path)` equals the set of pages whose `links_in` contains it; alias/heading forms count.
7. `tags(page)` merges frontmatter and body tags; `get_wiki` exposes them; virtual Tags group filters (fixture render).
8. Frontend fixture: `[[page|alias]]` renders "alias" and navigates to `page`; a red link click sends `wiki_create`.
9. `_action_wiki_note` and `wiki_save` both hold `memory._LOCK` (assert via a patched lock).
10. `wiki_rename` moves the file and rewrites all four link forms in backlinking pages (and the short form only when
    the basename is unique — a second page with the same basename leaves short links untouched); one commit; lint
    clean afterwards; renaming onto an existing path is refused.
11. `wiki_delete` removes the page, index entry; lint lists the now-broken links; one commit; `git log` shows the file.

## Out of scope
Graph view, transclusion, CodeMirror, autosave, section-level edits, fuzzy link rewriting, mobile polish.
