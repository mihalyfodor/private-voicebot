Status: exploration

# 44 — Vault health: three more lint findings; related pages at ingest

## Overview
The vault degrades in two quiet ways. `sources.forget()` deletes a raw file but leaves the pages it created stamped
`source: raw/…`, and nothing reports it; and once the index outgrows `WIKI_INDEX_TOKENS` (600) an ingest no longer sees
the page it should append to, so it files a near-duplicate. This PRD adds three finding types to `wiki.lint()` and
gives both ingest prompts a short list of the existing pages most related to what is being filed. No runtime reply
changes.

## User stories
- As a user who forgot a source, the Wiki tab tells me which pages still claim it as their source, so I can delete or
  re-source them.
- As a user who dropped a file into `raw/` by hand (or after a crashed ingest), lint names the file nobody knows about.
- As a user, two pages called "Anna" show up as a finding instead of splitting what Wanko knows about her.
- As a user with a large vault, dropping a second document about Falcon appends to `projects/falcon` rather than
  creating `projects/falcon-plan`.

## UI/UX
- No new UI: the new findings are rows in the existing lint list of the Wiki tab (`_wiki_msg`, `chatbot.py:142`),
  same plain strings as today.
- Ingest looks the same; only the prompt the model sees changes.

## Technical approach
- **Dead source ref** (in the per-page loop of `lint()`, `wiki.py:1380`): a page whose frontmatter `source` starts with
  `raw/` and whose file is missing → `people/anna: source raw/inbox/2026-08-30-falcon.md no longer exists`. Only the
  `source:` key is checked: ingest stamps a vault-relative raw path there (new pages via `restrict_ops`,
  `sources.py:982`/`991`; meeting pages `sources.py:839`); every other value (`session …`, `user edit`,
  `written answer`, `bootstrap`, `user drop …`) is not a path. The PRD 43 `sources:` list (`save_card`, `wiki.py:714`)
  holds card refs like `title · part n/N`, not paths, and is not checked.
- **Uncited raw file** (after the loop): every file under `raw/` except `raw/manifest.jsonl` and dot-folders
  (`raw/.index/`) that is neither a manifest `path` (`sources.manifest_entries`, `sources.py:740`; function-local
  `import sources`, as `raw_chunks` does at `wiki.py:1014`) nor any page's `source:` or `raw:` frontmatter →
  `raw/inbox/notes.pdf: not in the manifest and cited by no page`.
- **Duplicate name** (after the loop): pages grouped by `name:` (casefolded, whitespace collapsed), `sources/` pages
  grouped apart from the rest (a source titled "Falcon" next to `projects/falcon` is expected); each page after the
  first, in path order → `people/anna-2: duplicate name "Anna" (also people/anna)`.
- Order: existing findings first, then dead refs (with their page), uncited raw files, duplicate names. The
  docstring is updated: a finding names its page, or its raw file.
- `lint_and_log` (`wiki.py:1408`, called from `memory._lint_wiki`, `memory.py:1567`) keeps committing on any
  finding; the new types are not special-cased.
- `forget()` (`sources.py:1364`) is unchanged: it already drops the `(source: raw/…)` note bullets, and the dead-ref
  finding covers the pages it created.
- **Related pages**: `wiki.related_context(text, k=5) -> str` over `wiki.search(text, k + 3)` (`wiki.py:888`), dropping
  `sources/` hits as `llm._page_recall` does (`llm.py:807`), keeping `k`; one line per hit `- people/anna: Anna — <first
  Summary line>` (`_summary_line`, `wiki.py:361`, cut to 120 chars), under the heading `Existing related pages (update
  these rather than duplicating):`, trimmed line by line to `RELATED_TOKENS = 200` (`memory.tokens_est`, as
  `index_text` does at `wiki.py:400`). No hits → "" and the block is left out.
- Query text: in `sources.propose_page_ops` (`sources.py:1076`, index at `:1083`) the title + summary + entity names
  (the model there never sees the sampled source text either); in session `wiki.ingest` (`wiki.py:1484`, index at
  `:1495`) the transcript. The block goes right after `Index:` in both user messages.

### Decisions
- Chose lint over making `forget()` rewrite the `source:` stamp because a page whose provenance is gone needs a human
  call (delete or keep), and a raw file removed in Finder or Obsidian leaves the same dead ref.
- Chose to keep `lint_and_log` committing on every finding over a "changed since last run" rule because the new types
  are rare and the commit touches only `log.md`. Risk: an unfixed finding earns one `wiki: lint` commit per
  reflection; mitigation: same as today's types, and each finding names what to fix.
- Chose BM25 (`wiki.search`) over embeddings because it is already there and the near-duplicates share names.
- Assume the title + summary query finds the page to update; if ingests still duplicate, revisit with the sampled text.
- Risk: 200 more prompt tokens per ingest call; mitigation: the hard cap, and the block is dropped when empty.

## Data model / state shape
- No new files or persisted keys. Findings stay `list[str]`; `related_context` is a pure read of the vault.
- `RELATED_TOKENS = 200` next to `WIKI_INDEX_TOKENS` in `wiki.py` (env override like it).
- Invariants: lint never writes (only `lint_and_log` does); `raw/manifest.jsonl` and `raw/.index/` are never findings;
  the related block never lists a `sources/` page.

## Test scenarios
1. `tests/sources/test_sources.py`: ingest a document whose ops create `projects/falcon`, then `forget` it → `lint()`
   contains `projects/falcon: source raw/… no longer exists`; before the forget it does not.
2. `tests/wiki/test_wiki.py` (`wiki_sandbox`): a page stamped `source: session 2026-08-29` and one stamped
   `source: written answer` with a `sources:` list → no dead-ref finding.
3. `tests/wiki/test_wiki.py`: write `raw/inbox/stray.pdf` with no manifest line → `raw/inbox/stray.pdf: not in the
   manifest and cited by no page`; with a manifest line for it, or a page whose `source:` is that path → no finding;
   `raw/manifest.jsonl` and a file in `raw/.index/` → never a finding.
4. `tests/wiki/test_wiki.py`: `people/anna` and `people/anna-2` both `name: Anna` (one as ` anna `) → exactly one
   finding, `people/anna-2: duplicate name "Anna" (also people/anna)`; `sources/falcon` and `projects/falcon` both
   named "Falcon" → none.
5. `tests/wiki/test_wiki.py`: a vault whose only finding is a duplicate name → `lint_and_log` appends one `log.md`
   line and makes one `wiki: lint` commit; the Wiki message (`get_wiki`) lists the finding.
6. `tests/wiki/test_wiki.py`: `related_context("Falcon launch", k=5)` on a vault with `projects/falcon` and
   `sources/falcon-plan` → contains `- projects/falcon: Falcon — …`, not `sources/`; on an empty vault → "".
   Fifty matching pages → the block stays within `RELATED_TOKENS`.
7. `tests/wiki/test_wiki.py`: session `ingest` with `WIKI_INDEX_TOKENS` patched so `projects/zephyr` is cut from the
   index, a transcript about Zephyr → the `FakeClient` user message has the heading and `projects/zephyr`, and the
   `Index:` part does not.
8. `tests/sources/test_sources.py`: same as 7 through `propose_page_ops` (title "Zephyr rollout") → the captured user
   message lists `projects/zephyr` under the heading.

## Out of scope
Embeddings, orphan pages (the index links everything), staleness (`reingest` covers it), auto-fixing findings,
manifest lines whose raw file is gone, checking `(source: …)` note bullets, UI changes beyond the existing lint list.
