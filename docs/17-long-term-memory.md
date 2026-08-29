Status: exploration (research draft, 2026-08-29 — to be turned into PRDs in a follow-up session)

# 17 — Long-term memory: decay, archive and an LLM-wiki layer

## Overview

Two gaps after PRD 15/16: (a) profile entries (people, projects, recurring) never age — they leave only through an
LLM DELETE or the "oldest out" cap, and (b) there is no durable, human-readable layer for real collaboration
(meetings, decisions, documents, project notes) — the profile is deliberately small and prompt-shaped.
This note evaluates the known approaches and proposes a roadmap that unifies both: the profile becomes
**working memory that decays**, and an Obsidian-style **wiki becomes the archive it decays into**.

## Research summary

| approach | mechanism | fit with memory.json |
|---|---|---|
| Generative Agents (Park et al. 2023) | score = recency(0.995^hours since last access) + importance + relevance, used to *rank retrieval* from a stream | Already used for `episodic` (`_episodic_score`). Not applicable to profile sections: the whole profile is in context, nothing is retrieved, so a ranking score changes nothing |
| MemoryBank / Ebbinghaus, Oblivion (2026) | exponential retention, reset on each recall; forgetting = loss of accessibility, not deletion | Right *shape* (fade, don't delete) but needs a "recall" signal. We don't observe which facts the model used; proxy = the user mentions it again |
| Zep / Graphiti bitemporal edges | `valid_at`/`invalid_at` per fact; superseded facts invalidated, never deleted; history queryable | Closest to what exists: `since` + `superseded`. Extending entries with `seen` (last confirmed) and `until` (invalidated) is a small, inspectable step with no tunable constants beyond per-section windows |
| Mem0 eviction | explicit delete, caps, `immutable` flag | This is the status quo; caps are the weak point |
| Claude Code auto-memory | one fact per markdown file, `MEMORY.md` index loaded each session, detail files read on demand | The two-tier "always-loaded index + on-demand pages" is exactly the latency-safe shape for a voice app |
| Karpathy LLM-wiki (Apr 2026) | `raw/` immutable sources → `wiki/` LLM-maintained interlinked markdown (entity/concept pages, index, log) → ops `ingest / query / lint`; no embeddings at personal scale | The durable layer. Raw = session transcripts, emails, dropped documents; wiki = people/projects/meetings/decisions pages; query via tool call, so no per-turn cost |
| basic-memory / Obsidian MCP | assistant writes plain markdown into a vault the human also edits; BM25 search, frontmatter patches | Confirms: markdown + wikilinks + frontmatter is the interoperable format; the user gets Obsidian for free as the "visual editor" of the durable layer |

## Recommendation

### PRD 17a — decay + archive (small)
- Add `seen: YYYY-MM-DD` to every list entry; set on ADD/UPDATE. Add a `CONFIRM` op to `propose_ops`
  (`{"op":"CONFIRM","path":"people[2]"}`) so a fact the user mentions again gets its `seen` refreshed — the
  model already sees the profile, so this costs nothing extra.
- `reflect()` may mark entries `stale: true` when `today - seen` exceeds a per-section window
  (people 180 d, projects 120 d, recurring 90 d; env-tunable). Stale entries are dropped from `render()`,
  kept on disk, shown in a "stale" group in the PRD 16 editor with Restore / Delete.
- Cap eviction by oldest `seen` instead of list position; stale entries evicted first.
- When a stale entry is evicted or deleted, and 17b exists, it is **archived**: appended to the matching
  wiki page (`people/anna.md`: "2026-08 colleague, joined the team; not mentioned since 2026-11") instead
  of vanishing. That turns "forgetting" into "moving to the archive", the same policy as `superseded`.
- Tests: harness `stale.yaml` with a mocked clock (`memory._today`) advanced 200 days between rounds:
  the assistant abstains on "how is Anna settling in?" instead of treating her as new; entry is stale on disk,
  present in editor; `CONFIRM` in round 2 un-stales it. Unit: eviction order, CONFIRM refreshes `seen`,
  reflection can only set `stale`, never delete non-episodic entries.

### PRD 17b — wiki layer (medium; the real long-term investment)
- `wiki/` folder = Obsidian vault: `index.md` (one line per page, ≤ 200 lines, loaded into the prompt like
  the profile — the only always-on cost), `log.md` (append-only, dated), `people/`, `projects/`,
  `meetings/YYYY-MM-DD.md`, `decisions.md`, `schema.md` (page templates + rules). Frontmatter + `[[wikilinks]]`.
- `raw/` = immutable inputs: session transcripts (already `session.log`), emails fetched via the gmail tool
  when the user asks to "file this", documents dropped in `raw/inbox/`.
- Three operations, mirroring Karpathy: **ingest** at memory-save time (same background cadence: the model
  proposes page edits as *ops* — `append_section`, `set_field`, `new_page` — validated and applied in Python,
  never a whole-file rewrite, same lesson as PRD 15); **query** via two tools `search_wiki(query)` (BM25 over
  pages, no embeddings) and `read_page(name)`, so recall only costs a tool round-trip when needed; **lint** in
  `reflect()`: broken links, orphans, index coverage, contradictions between a page and the profile.
- Division of labour, stated in `schema.md`: **profile = who the user is** (small, always in context,
  decays); **wiki = what we are working on** (unbounded, on demand, durable, human co-edited).
- Versioning: `git init` inside `wiki/`; every ingest commits, which gives diff/undo and a history the PRD 16
  editor doesn't have to build. The user edits pages in Obsidian; the assistant reads them fresh each ingest.
- Tests: ingest→query harness (LongMemEval-style multi-session: facts stated in S1 filed to a page, recalled in S4
  via `search_wiki` — `expect` on the reply and on the tool call), lint unit tests (index/page parity, no
  dangling links), op validation (unknown page, oversized append rejected), determinism (ingesting the same
  transcript twice produces no diff).

### PRD 17c — raw ingestion for the "office document stack" (later)
Inbox folder watcher for PDFs/markdown/emails; per-source summary page + entity pages updated; this is where
gmail/news tools stop being read-only and start feeding the wiki. Only after 17b proves the query path works with
the local 26B model.

### Decisions
- Bitemporal fields (`seen`, `stale`, `superseded`) over a numeric decay score: inspectable in the editor, no
  constants to tune except windows; the model never sees or computes a score.
- Archive into the wiki instead of deleting: forgetting from the prompt ≠ losing the fact.
- Wiki edits as validated ops + git, not LLM file rewrites: same reason PRD 15 rejected rewriting memory.json.
- Index in context, pages by tool: keeps per-turn voice latency unchanged; assume ≤ 200 pages; if false, index
  becomes sectioned and `search_wiki` becomes the only entry point.
- Risk: a 26B local model writing durable pages hallucinates or over-files. Mitigation: ingest only from `raw/`
  sources it was given, every page section carries a `source:` line, lint flags unsourced claims, git makes
  reverts cheap.
- Sequence: 16 (editor) → 17a (decay + stale UI) → 17b (wiki) → 17c (ingestion). Each is independently useful.

## Out of scope (for the whole series)
Embeddings/vector DB; multi-user; syncing the vault to a cloud service; automatic email ingestion without the user asking.

Sources: Park et al. 2023 (Generative Agents); MemoryBank (2023); Oblivion (arXiv 2604.00131); Zep/Graphiti
temporal KG docs; Mem0 "memory eviction and forgetting" blog; Claude Code auto-memory docs; Karpathy LLM-wiki
gist (Apr 2026) and write-ups; basic-memory / Obsidian MCP.

## Obsidian plugin question (research, 2026-08-29)

Three routes: **(A)** server writes markdown, user opens `wiki/` as a vault — Graph, Backlinks, Bases (core since 1.9) and Dataview work off files/frontmatter with no plugin; Obsidian's file watcher re-indexes external writes. This is what basic-memory does. **(B)** route writes/reads through the *Local REST API* plugin (coddingtonbear, MIT, v5.x adds a built-in MCP endpoint): gains heading-level patch and full-text search endpoints, costs a dependency on Obsidian being open. **(C)** a custom plugin (TypeScript, `Plugin`/Vault/MetadataCache APIs, `requestUrl` for loopback HTTP; `desktopOnly`) — only needed for in-app UI: side panel, commands ("ingest this note"), knowing the active note. Plugins cannot run headless, so (B)/(C) can never be the only write path for a standalone voice server.

**Decision:** (A) now. Design for (B)/(C) later by: all wiki mutation through the ops layer (swap `pathlib` for REST calls without touching callers); fixed frontmatter keys (`type`, `tags`, `source`, `updated`, `seen`) and page naming now, since Bases/any plugin key off them; `wiki/` as its own git repo with `.obsidian/` ignored; `[[wikilinks]]`, `meetings/YYYY-MM-DD.md` daily-note naming, no Templater syntax in generated pages. A (C) plugin is a candidate PRD 18: sidebar "what the assistant just filed" + "re-ingest this note" command + active-note context for the voicebot.

Sources: obsidian-api README; DeepWiki MetadataCache; coddingtonbear/obsidian-local-rest-api; basicmemory.com Obsidian integration; cyanheads/obsidian-mcp-server; got.md Bases guide.
