Status: implementation

# 50 — Modular layout: seven downward-only layers and one tool registry

## Overview
Four modules carry most of the app (wc -l: wiki.py 1800, memory.py 1688, sources.py 1526, chatbot.py 1388, llm.py
1298) and import each other in cycles, mostly through function-local imports (wiki.py:1039/1074/1261/1372/1441/1581,
memory.py:608/1355/1377/1577/1588, sources.py:173, tools/sources.py:89 `import chatbot`). Adding a tool means
touching tools/__init__.py, a hardcoded token constant and an if-chain in `chatbot.respond`. This umbrella PRD moves
the code into seven layers that import only downward, enforced by a test, and puts every tool and command behind one
ordered registry — as a series of behaviour-preserving `Refactor:` commits, nothing observable changes.

## User stories
- As the developer, I can tell from a file's folder what it may import, and a test fails when I import upward.
- As the developer, I add a tool by writing one module and one `REGISTRY` line; schemas, the token count and command
  routing follow from it.
- As the developer, every model call goes through `core/llm.py`, so one conftest fake serves every test.
- As the developer, I can review each step as one commit with `scripts/check.sh` green and its counts quoted.

## UI/UX
None. No observable change: the WS protocol (every `ACTIONS` key and message literal), routes, `settings.json`,
`memory.json`, `memory_ops.jsonl`, the vault layout and its commit messages, and `static/app.js` are untouched.

## Technical approach
**Layers** (each imports only from layers below it; `scripts/` sits outside and may import anything):
1. `core/` — `config.py` (env paths now spread over wiki.py:31 `WIKI_PATH`, memory.py:38 `_DIR`, llm.py:22
   `LOG_PATH`, avatars.py:11 `SETTINGS_PATH`), `settings.py` (`load_settings/save_setting/save_settings`, avatars.py
   48-80), `trace.py` (PRD 49, 378 lines, untracked today), `untrusted.py`, `vault_lock.py`, `formats.py` (the pure
   extractors, sources.py 189-735: `sample_text` … `extract`), `llm.py` (the single model-call chokepoint: `client()`
   from llm.py:311, `complete()` replacing `sources._client_model`/`_ask` (sources.py:171-187) and `memory._complete`
   (memory.py:742), plus `stream()` for the six `client().chat.completions.create` sites in llm.py).
2. `vault/` — wiki.py split by its own section markers: `store.py` (vault/git 29-224, frontmatter 225-290, paths
   291-358), `index.py` (359-445), `pages.py` (page ops + editor 446-817), `search.py` (818-923), `chunks.py`
   (924-1286), `focus.py` (1287-1379), `digest.py` (1380-1551), `lint.py` (1552-1638), `ingest.py` + `bridge.py`
   (1639-1800); `manifest.py` moves down from sources.py (736-767, 1299-1302) so chunks stop importing sources.
3. `memory/` — `store.py` (files/schema 37-182, migration 1510-1547), `render.py` (183-383), `identity.py`
   (384-637), `ops.py` (638-1405 minus `propose_ops`).
4. `ingest/` — `pipeline.py` (the rest of sources.py), `inbox.py`, `notebooks.py`, `batch.py` (batch_import.py).
5. `tools/` — one module per tool (as today) plus `registry.py`; `tools/__init__.py` becomes empty.
6. `agent/` — llm.py split: `prompt.py` (222-310), `context.py` (PRD 11 budget governor, 39-165), `recall.py`
   (recall/band/focus, 447-567, 851-992), `offers.py` (568-656), `turn.py` (`ask_events` 993-1170 + `_tool_round`
   347-381), `written.py` (card + briefing completions, 668-850), `conversation.py` (state, reset, new topic
   1182-1298), `fold.py` (`propose_ops` 752-781, `reflect` 1406-1509, `save` 1548-1688 moved out of memory).
7. `server/` — chatbot.py split: `app.py` (651-835), `ingest.py` (`/ingest` 714-813), `ws.py` (`ACTIONS` 1230,
   `/ws` 1317), `actions/` (836-1229 by panel), `turns.py` (respond and the turn helpers 254-650), `speech.py`
   (TTS 45-135 + splitter.py + fillers.py), `panels.py` (136-198), `session.py`, `services.py`, `asr.py`, `vad.py`.

`chatbot.py` stays as a ~10-line entry point (Dockerfile:24 `CMD ["python", "chatbot.py"]` unchanged).
`workflows/email_classification.py` (692 lines, offline script) moves to `scripts/` to free the word "workflow";
README.md:56, docs/index.md:133 and .dockerignore:11 follow.

**Tool registry** (`tools/registry.py`): an explicit ordered `REGISTRY` tuple of `Tool` records. `schemas()` returns
the schema tools' definitions in today's `TOOLS` order (tools/__init__.py:15-27); `tokens()` returns
`len(json.dumps(schemas())) // 4`, replacing `TOOLS_TOKENS = 1089` (llm.py:123, asserted in
tests/session/test_context_meter.py:60); `match(text)` returns the first trigger tool whose regex matches,
replacing the if-chain in `chatbot.respond` (547-589) in today's order: `new_topic`, `save_note`, `briefing`, then
the `llm_ok` gate, then `write_up`, then `open_page` (which falls through to the model with a note). `run_tool` keeps
its signature. `Ctx` carries `text, origin, session, turn, send, push_panel`; the server builds it, so a tool never
imports the server (removes tools/sources.py:89). A workflow is simply a trigger-mode tool. `READ_TOOLS` and
`ALLOWED_HOSTS` derive from the `untrusted` and `hosts` fields (tests/sources/test_untrusted_text.py:21 keeps
importing them).

**Shims**: during the move each old module (`llm`, `chatbot`, `sources`, `wiki`, `memory`, `avatars` settings)
re-exports its new home. `memory/` shadows memory.py, so step 8 turns it into `memory/__init__.py` re-exports.
Tests retarget their `monkeypatch.setattr` sites in the same step that moves the patched name — 219 today
(`llm` 65, `chatbot` 42, `sources` 28, `wiki` 15, `memory` 13); the two conftest fakes (`FakeClient`
tests/conftest.py:130, `StreamClient` 279) become one fake patched at `core.llm`. `tests/test_protocol_doc.py:11`
(reads chatbot.py source) retargets to `server/*.py`. The layer test forbids app code importing a shim from the step
that creates it; step 12 deletes them.

**Migration** — one branch `feature/modular-layout`, started after PRD 49 merges (it can interleave with PRDs 51-55, which land on whichever layout is current and are rebased per step); one `Refactor: modular layout —
<step>` commit per step; `scripts/check.sh` green with passed/skipped/failed quoted at every step. Steps are ≤ ~500
moved lines except the pure `git mv`-and-split steps (1, 5, 6-7, 8, 9-10, 11), which carry no edits besides imports
(total ≈ 10,300 lines moved, ≈ 9,600 without the workflows move):
| step | content | ~moved lines |
|---|---|---|
| 1 | `tests/test_layers.py` + allow-list; `core/` (config, settings, trace, untrusted, vault_lock); workflows → scripts | 550 + 692 |
| 2 | `tools/registry.py` for the 11 schema tools; golden schema test; `tokens()` | 100 |
| 3 | commands as trigger tools (chatbot.py 343-546 + `respond` routing) | 200 |
| 4 | `core/llm.py` chokepoint; one conftest fake | 60 |
| 5 | `core/formats.py` + `vault/manifest.py` down from sources; the rest of sources.py, notebooks.py, batch_import.py → `ingest/` | 580 + 1290 |
| 6-7 | wiki.py → `vault/` (29-817, then 818-1800) | 790 + 980 |
| 8 | memory.py → `memory/` package; propose/reflect/save → `agent/fold.py` | 1690 |
| 9-10 | llm.py → `agent/` (context/prompt/recall/offers, then turn/written/conversation) | 1300 |
| 11 | chatbot.py → `server/`; splitter, fillers, session, services, asr, vad move in | 1190 + 790 |
| 12 | shims deleted; allow-list empty; docs/index.md tree + process.md test layout (`tests/core/`, `tests/vault/`, `tests/tools/`, `tests/agent/`, `tests/server/`, …) | 100 |

### Decisions
- Downward-only rule enforced by a ratchet test over a written convention: an allow-list that may only shrink makes
  each step's progress checkable and stops new cycles.
- One registry with two entry modes (schema / trigger) over separate command and tool systems: one place to add,
  order and describe anything the user can invoke.
- Explicit ordered tuple over discovery (entry points, decorators, folder scan): routing order is behaviour.
- Shims plus per-step patch retargeting over a big-bang move: every commit is green and revertable on its own.
- No ABCs, DI container, async rewrite, config framework, event bus or gateway; plain modules and functions.
- `static/app.js` untouched; risk: a shim hides a missed retarget; mitigation: step 12 deletes all shims and the
  full suite must stay at the same count.
- Chose starting after PRD 49 merges (done 2026-10-16) over waiting on 51-55: later PRDs rebase onto the current
  layout at their own gate, and no PRD waits for 50.
- Chose `agent/approvals.py` (PRD 52 registry) and `server/runs.py` (PRD 51 run pages) over other homes: the `writes`
  flag on tools is the hook.
- Chose deleting the shims at step 12 over keeping them one release: the layer test then fails any old import.
- Step 2 keeps each module's `run(args: dict)` signature; `run_tool` passes the dict as today — `**args` unpacking is deferred until the modules are touched again, so step 2 stays a pure registry move.
- Step 3: the registry holds every trigger tool's record (name, regex, order, flags) and `match()`; the body is bound by the server with `@registry.handler(name)` because it drives TTS and turn state, which must stay above the agent layer. Adding a command = one registry line + one decorated function. Risk: a record with no bound handler; mitigation: `run` raises naming the tool, and a test covers it.
- Step 4: `core/llm.py` owns the client and the three call shapes (`stream`, raw `completion`, text `complete`); memory's `save/reflect/propose_ops` no longer take a client and model, so one conftest fake patched at `core.llm._client` serves every test. `StreamClient`/`RawStreamClient` stay as two spellings of the one `FakeClient` until step 12.
- Step 3 side effect: `file_source` now really refreshes the Wiki panel in the running app. The old `_push_panel` did `import chatbot` while the app ran as `__main__`, so it loaded a second module copy whose session was None and the push never fired. Accepted as the one observable improvement of steps 1–4.
- Steps 1–4 landed overnight 2026-10-16/17 (commits d837ca7, d1f799a, 40d245b, 9139454, Impl 0c0b0f6); steps 5–12 wait for the user's review before any further move, per the overnight plan.
- Step 5 landed 2026-10-17: `core/formats.py` (pure extractors), `vault/manifest.py`, `ingest/{pipeline,notebooks,batch}.py`; `raw_text` and `extract` stay in `ingest/pipeline.py` because they read the vault path (`wiki._path`), which `core/` may not import; `ics_events` defaults `today` to `date.today()` and `extract` passes the clock `memory._today_iso()` as before. ALLOWED gained the pairs for modules that still import root `memory`/`wiki`/`remarkable` (each leaves with its target's step).
- Step 6 landed 2026-10-17: wiki.py's vault, frontmatter, paths, index, page ops, editor and search sections became `vault/{store,index,pages,search}.py`, `index_db.py` became `vault/index_db.py` (root `index_db` is a `sys.modules` alias of it). Modules reach each other as `module.name` (never `from m import name`) so one patch site on the defining module reaches every caller despite the import cycles; `store` imports its siblings last because they read its constants at import time. `wiki.py` keeps the step-7 half and forwards reads of moved names; assigning one raises, so a missed patch retarget fails loudly. `WIKI_PATH` lives in `vault.store` (conftest patches it there).
- Step 7 landed 2026-10-17: the rest of wiki.py became `vault/{chunks,focus,digest,lint,ingest,bridge}.py` by its section markers (`vault_map` / `related_context` sit in `index.py`, the `python -m wiki reindex` entry points `cli_vault` / `main` in `bridge.py`); `wiki.py` is now a pure shim (reads forward, assignment raises) and is in `SHIMS`, so app code imports `vault.*` modules directly. ALLOWED pairs whose target was wiki are gone; the new ones are vault -> memory (step 8), `vault/focus.py` -> llm (steps 9-10) and `vault/{chunks,digest,bridge}.py` -> ingest, the PRD's expected vault/ingest cycle.
- Step 8 landed 2026-10-17: memory.py became `memory/{store,render,identity,ops}.py` plus `agent/fold.py` (`propose_ops`, `reflect` and its validators, `save`, and `migrate_if_needed`, which calls `propose_ops`, so it cannot stay in the store). Banners were too coarse (memory.py had five), so nodes were assigned by banner plus a name override list. `extract_json` and `_transcript` live in `memory/ops.py`, not in fold: `ingest/pipeline` and `vault/ingest` use them and may not import `agent`. `memory/__init__.py` forwards reads of every moved name (reaching `agent.fold` by `importlib` on first read, which the layer test cannot see) and raises on assignment; `memory.render` is the submodule, so the old function call is `render.render(profile)`. `ingest` -> `memory` ALLOWED pairs are gone (ingest sits above memory); the vault -> memory pairs stay, since vault is below memory.
- Step 9 landed 2026-10-17: llm.py's prompt, context, recall and offers code became `agent/{prompt,context,recall,offers}.py`; the rest stays in llm.py until step 10, which forwards reads of moved names and raises on assignment. `TOOLS` lives in `agent/prompt.py` (it was computed before `SYSTEM_PROMPT`, which reads it at import; any later home breaks that order), so tests that rebind `llm.TOOLS` patch `agent.prompt.TOOLS`. `vault/focus.py` reads the live list from `agent.prompt` instead of `llm`: still an upward import, allowed until step 12 because the eval harness empties the list at run time. `agent/prompt.py` imports `avatars` and `splitter` (root; `EMOTIONS` must move down before step 11).
- Step 10 landed 2026-10-17: the rest of llm.py became `agent/{turn,written,conversation}.py` (`_forget_cards` joined the conversation state it clears; written answers and the briefing share `written.py`); llm.py is a pure shim. The old module-level `reset()` still runs once at import, now at the bottom of `agent/conversation.py`: the shim imports conversation first so the other modules load before the reset, and `agent/prompt.py` imports conversation inside `set_verbosity` so loading prompt first cannot deadlock the cycle. `research.py` uses `splitter.strip_tag` directly (it only borrowed the name through llm). `agent/context.py` still calls `core.llm.client().models.list()` for the context length (not a completion; the chokepoint stays core/llm.py). ALLOWED: `agent/*` -> llm pairs are gone; `agent/{prompt,conversation,written}.py` -> avatars / splitter stay until step 11.

## Data model
- Layer graph: `core < vault < memory < ingest < tools < agent < server`; `scripts/` and `chatbot.py` on top. A
  module in layer N may import layers < N (stdlib and third-party always allowed). `tests/test_layers.py` holds
  `ALLOWED = {("vault/chunks.py", "ingest"), …}` — (importer, imported layer) pairs; an entry not observed any more
  fails too, so the list can only shrink. Function-local imports count.
- Registry record:
  `Tool(name: str, description: str, trigger: re.Pattern | None, schema: dict | None, run: Callable[[Ctx], str |
  None], takeover: bool, needs_llm: bool, writes: bool, hosts: tuple[str, ...], untrusted: bool)`.
  Invariants: exactly one of `trigger`/`schema` set; names unique; `takeover` = the turn ends in the tool (no model
  completion); `needs_llm` tools sit after the gate; `writes` is metadata only (approval is PRD 52, which reads it later).
  `Ctx(text, origin, session, turn, send, push_panel)`; schema tools receive their args as `**args`.

## Test scenarios
1. `tests/test_layers.py`: an injected `import server` inside a function in `vault/store.py` (tmp copy) fails
   naming file:line; the real tree passes with the allow-list; a stale allow-list entry fails.
2. `tests/tools/test_registry.py`: `json.dumps(schemas())` is byte-identical to the golden string captured from
   today's `tools.TOOLS` before step 2 (`tests/fixtures/tool_schemas.json`).
3. Same file: `tokens() == 1089` (the old constant); test_context_meter.py:60 asserts against `tokens()`.
4. Same file: a phrase table of every existing command phrase (new topic, `/new`, save that as a note, what's new?,
   `/brief`, write it up, `/write`, open/find/go back) routes to the same tool as today; "what's new with you"
   matches nothing and reaches the model.
5. Order: with `services.llm_ok` False, `new_topic`, `save_note` and `briefing` still run; `write_up` and a model
   turn yield the `error {scope: turn}` offline line (tests/session/test_turns.py).
6. A trigger tool receives a `Ctx`; `file_source` calls `ctx.push_panel()` and the panel stub receives the wiki
   refresh, while `tools/sources.py` contains no `chatbot`/`server` import (tests/tools/test_registry.py).
7. `tests/conftest.py` has one fake model client patched at `core.llm`; an ingest, a memory save and a typed turn all
   consume its queued replies (tests/agent/test_fold.py, tests/ingest/test_pipeline.py, tests/agent/test_turn.py).
8. `tests/test_protocol_doc.py` green reading `server/*.py`; `ACTIONS` keys unchanged.
9. Every step: `scripts/check.sh` green, passed count ≥ the previous step's, quoted in the commit body.
10. After step 12: an app module importing `llm` / `wiki` / `sources` / `chatbot` (the deleted shims) fails the
    layer test; `python chatbot.py` serves `/` and `/ws` (tests/prod/test_docker.py scenario 7 unchanged).

## Out of scope
Approval/confirm UI for `writes` tools (PRD 52), run pages for workflows (PRD 51), splitting app.js, any WS, route or persisted-data
change, performance work, a new tool or command.

## Open design questions
None open.
