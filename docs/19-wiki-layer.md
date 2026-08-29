Status: done

# 19 — Wiki layer: an Obsidian vault the assistant maintains

Slice "17b" of [17-long-term-memory.md](17-long-term-memory.md). Decisions: vault at `~/llm-wiki`, no plugin,
in-app view stays simple (browse + append + open in Obsidian).

## Overview

A markdown vault at `~/llm-wiki` (`WIKI_PATH`) that the assistant maintains as durable, human-co-edited working
knowledge: people, projects, meetings, decisions. The vault's `index.md` is loaded into the prompt (the only
always-on cost); pages are reached through two tools (`search_wiki`, `read_page`) so recall costs nothing until
needed. At memory-save time the model proposes **page ops** that Python validates and applies; every ingest is a
git commit inside the vault. Stale profile entries (PRD 18) archive into their wiki page instead of vanishing.

## User stories

- As a user, "what did we decide about the deployment last week?" is answered from `decisions.md`, not guessed.
- As a user, I open `~/llm-wiki` in Obsidian and see readable pages with backlinks and a graph; I can edit them and
  the assistant reads my edits next time.
- As a user, when a project drops out of the assistant's working memory, its page still holds the history.
- As a user, I can drag a person/project from the memory view onto the Wiki tab to file it, or drag a wiki page
  onto a profile section to pull it back into working memory.
- As the developer, the assistant can never rewrite a page wholesale, and I can `git log` the vault.

## UI/UX

- Workspace gets a **Wiki** tab: left list of pages grouped by folder (from `index.md`), right the selected page
  rendered as plain text with `[[links]]` clickable; a one-line "add note" box (appends a dated bullet under
  `## Notes`); "Open in Obsidian" link (`obsidian://open?vault=llm-wiki&file=<path>`); a "Re-ingest last session"
  button (runs ingest on demand).
- Drag targets (DnD infra from PRD 18): memory row → Wiki tab button = archive (creates/appends page, marks the
  profile entry stale); wiki page row → a memory section header = pull (creates a profile entry from frontmatter
  `name`/`type`, `seen: today`). Each has a "→" menu fallback.
- Memory view stale rows get an extra "archive" action.

## Technical approach

**Vault layout** (`wiki.py` creates it on first run, `git init`, `.gitignore` with `.obsidian/`):
```
~/llm-wiki/
  index.md            one line per page: - [[people/anna]] — colleague, joined 2026-08 (≤ 200 lines)
  log.md              append-only: - 2026-08-29 filed [[meetings/2026-08-29]] (source: session)
  schema.md           page templates + rules (read by the model at ingest)
  people/<slug>.md    ---\ntype: person\nname: Anna\ntags: []\nsource: session 2026-08-29\nupdated: …\n---
  projects/<slug>.md  type: project, status
  meetings/YYYY-MM-DD.md  type: meeting
  decisions.md        ## YYYY-MM-DD <title>\n<text>\nsource: …
```
Frontmatter keys fixed: `type, name, tags, source, updated, seen` (Bases/Dataview-friendly). Body sections:
`## Summary`, `## Notes` (dated bullets), `## History` (archive entries from PRD 18).

**Page ops** (`wiki.py`, applied in Python, never a whole-file rewrite):
- `new_page {path, type, name, summary}` — rejected if exists;
- `append_note {path, text, source}` — dated bullet under `## Notes`;
- `set_field {path, key, value}` — frontmatter only, key in the fixed set;
- `add_decision {title, text, source}` → `decisions.md`;
- `link {path, target}` — adds `[[target]]` to `## Related` if missing.
Validation: path within the vault, slug rules, page ≤ 20 kB, note ≤ 600 chars, `source` required. Every applied
batch: update `index.md` (regenerated from frontmatter, deterministic), append `log.md`, `git commit -q`.

**Ingest** (`wiki.ingest(turns, client, model)`): called from `memory.save()` after profile ops, same thread
and cadence; the model gets `schema.md`, `index.md` and the transcript, answers a JSON array of page ops;
only things the user *said* get filed; trivia rules from `PROPOSE_SYSTEM` apply. Empty array is normal.

**Prompt**: `memory.load()` also injects `<wiki_index>` (index.md, trimmed to `WIKI_INDEX_TOKENS`=600) with a
note: "pages you can open with read_page; search with search_wiki when the user asks about past work".

**Tools**: `search_wiki(query) -> top 5 {path, snippet}` (a ~30-line BM25 in `wiki.py`, no new dependency; indexed at startup and after each
ingest/edit) and `read_page(path) -> text (≤ 4 kB, truncated with a marker)`. Registered in `tools/__init__.py`;
filler phrases for them added like the other tools.

**Lint** (`wiki.lint()` in `reflect()` cadence and on demand): broken `[[links]]`, pages missing from index,
frontmatter without required keys, pages > 20 kB; result written to `log.md` and returned to the UI.

**Archive** (bridge from PRD 18): `memory.archive(section, index)` → `wiki` `new_page` or `append_note` under
`## History` with the entry's fields + `seen`, then `UPDATE <section>[i].stale true`; `pull(path)` the reverse.

**Actions**: `get_wiki` (index tree + lint summary), `read_wiki_page {path}`, `wiki_note {path, text}`,
`wiki_ingest`, `archive_memory {section, index}`, `pull_wiki {path, section}`.

### Decisions
- Index in prompt, pages by tool: unchanged voice latency; assume ≤ 200 pages.
- Ops + git over free-form LLM page writes: no silent deletions; `git log` is the undo.
- BM25 over embeddings: personal scale, no new model, deterministic tests.
- Vault at `~/llm-wiki` outside the repo; tests use a tmp path via `WIKI_PATH`.
- Risk: the 26B model over-files chatter. Mitigation: `schema.md` rules + `source` required + lint; the harness
  measures "pages created per session" and it must stay ≤ 1 on `basic.yaml`.

## Data model
Pages as above. Invariants: every page has the six frontmatter keys; `index.md` is a pure function of the pages;
every commit touches `log.md`; no page is ever deleted by the assistant (only by the user in Obsidian).

## Test scenarios (unit, tmp vault, mocked LLM unless stated)
1. First run creates the vault, `schema.md`, `index.md`, `.gitignore`, a git repo with one commit.
2. Each page op: valid → file content as specified; invalid (path outside vault, missing source, oversized,
   unknown key, `new_page` on existing) → rejected, no file change, no commit.
3. A batch of ops → one commit, `index.md` regenerated, one `log.md` line per op.
4. `search_wiki("deploy decision")` returns the decisions page first; `read_page` truncates at 4 kB with a marker;
   unknown path → "no such page".
5. `memory.load()` output contains `<wiki_index>` within the token budget; empty vault → no block.
6. `lint()` flags a broken link and a page missing from the index; regenerating the index clears the second.
7. Ingest: FakeClient returns two ops → files written, committed; FakeClient returns garbage → nothing changes.
8. Archive: stale person → `people/anna.md` gets a `## History` bullet with `seen`; entry marked stale; pull
   recreates a live entry from the page.
9. User edit round-trip: modify a page on disk between two ingests → second ingest's context contains the edit
   and the index reflects a changed frontmatter `name`.
10. Harness (live) `wiki.yaml`: S1 states a decision and a meeting; S3 probes "what did we decide about X?" —
    reply must contain the decision and the run log must show a `search_wiki` or `read_page` call; pages created
    on `basic.yaml` ≤ 1 per session.
11. Manual: Wiki tab lists pages, opens one, "add note" appends and commits, Open in Obsidian link works, drag a
    memory row onto the Wiki tab → page appears and row moves to stale.

## Out of scope
Raw-source ingestion (PDF/email inbox) — PRD 20; page deletion/rename from the app; embeddings; sync; any
Obsidian plugin (research in 21).

## Results (harness, live, 2026-08-29, gemma-4-26B)

| run | probes | notes |
|---|---|---|
| `wiki.yaml` (tools disabled — harness default) | 2/3 | model emitted `<|tool_call>call:search_wiki{…}` as text: the `<wiki_index>` note asks for a tool that is not offered |
| `wiki.yaml --tools` | **3/3** | 5 pages created from 3 sessions (people ×2, project, meeting, decision); "what did we decide about deployments?" answered via a real `search_wiki` call |

Takeaway: `wiki.yaml` must run with `--tools`; and `memory.load()` should only add the tool hint when the wiki tools
are actually registered (Phase 3 item). Profile stayed clean: 6 entries, 0 duplicates, 69 tokens.

## Vaults (2026-08-29)
- `~/llm-wiki` — the user's real vault (its own git repo; commits per ingest; `.obsidian/` ignored).
- `~/llm-wiki-dev` — development vault; `.env` sets `WIKI_PATH=~/llm-wiki-dev` so the app started during development,
  sub-agents and screenshots never touch the real vault. Tests and the harness always use a tmp vault regardless.
  For real use set `WIKI_PATH=~/llm-wiki` in `.env` (or unset it — the code default is `~/llm-wiki`).
