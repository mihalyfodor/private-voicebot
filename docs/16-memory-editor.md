Status: exploration

# 16 — Memory editor: view, correct and delete what the assistant remembers

## Overview

A "Memory" panel opened from the ☰ drawer that shows the `memory.json` profile as six sectioned lists (identity, preferences, people, projects, recurring, episodic), lets the user edit or delete any entry and add new ones, and shows exactly what block the model sees. Every edit is expressed as an existing `apply_ops` op (ADD / UPDATE / DELETE) so the schema, caps, `superseded` history and the audit log stay intact — the UI never writes the file itself.

Grounded in the 2026-08-29 research pass: ChatGPT and Open WebUI expose memory as grouped flat lists with inline edit/delete; Mem0's dashboard adds per-memory history; Letta's free-text blocks are what PRD 15 deliberately moved away from. The stored profile is at most two levels deep with fixed sections, so a tree view would only show leaves under six headings — a sectioned list matches the data.

## User stories

- As a user, I can open the menu and see everything the assistant remembers about me, grouped the way it is stored.
- As a user, I can fix a wrong value (e.g. the nickname stored as my name) or delete a stale person/project, and the change takes effect on the next reply.
- As a user, I can see the previous value of something that was updated ("was: Budapest") and restore it in one tap.
- As a user, I can add a fact by hand without waiting for the assistant to extract it.
- As a user, I can see the exact `<user_profile>` text the model is given, to understand why it said something.
- As the developer, hand edits go through the same validated path as LLM ops and show up in `memory_ops.jsonl`.

## UI/UX

- Drawer gets a **Memory** section with one item: "Memory (N facts)" → opens a full-height panel over the chat (same open/close pattern as the drawer; Esc / ✕ closes).
- Panel header: title, fact count, "What the model sees" toggle, "Reload" (re-reads disk).
- Six sections in `render()` order. Empty sections show a single muted "nothing yet" row plus "+ add".
  - **identity / preferences** rows: `key` (readonly for existing rows) · editable value · ⌄ caret if `superseded[key]` exists → expands to "was: …" with a **Restore** button.
  - **people** rows: `name` · `rel` · `note`; **projects**: `name` · `status` · `note`; **recurring**: `what` · `when`; **episodic**: `text` · `date` · `importance` (1–3) · `ttl_days`; expired episodics greyed with "expired" tag.
- Each row: click a field to edit inline (input), Enter/blur = save, Esc = cancel; 🗑 deletes immediately (no confirm — undo is `superseded` for scalars; for list rows the last deletion is shown as an "Undo delete" toast for ~8 s, implemented as a re-ADD of the removed entry).
- "+ add" at the bottom of each section opens a one-line form with that section's fields.
- Rejected ops (validation error, cap reached) show inline under the row in red with the server's reason; the row reverts.
- "What the model sees" replaces the list with a `<pre>` of `memory.render(profile)` — read-only.
- Danger zone at the bottom: "Forget everything" (confirm dialog) → resets to `_empty_profile()`; audited as one `DELETE *` line.

## Technical approach

**Backend (`chatbot.py` actions + small `memory.py` additions)**

- `get_memory` → `{"type":"memory","profile":_public(profile),"rendered":render(profile),"count":N}`.
- `memory_op` with `{op, path, value}` → runs `load_profile` → `apply_ops` (single op) → `_write_profile` → `_log_ops` under a **module-level `threading.Lock`** shared with `memory.save()`, because `save()` runs on the worker thread (`_periodic_memory`) while actions run on the asyncio loop. Reply is the fresh `memory` message on success or `{"type":"memory_error","reason":…,"path":…}` on rejection.
- `apply_ops` gains an optional collector so rejection reasons are returned, not just printed (`_reject` currently only prints, memory.py:617). Minimal change: `apply_ops(profile, ops, rejected: list | None = None)`.
- `memory_reset` → writes `_empty_profile()`, logs one audit line, returns `memory`.
- After every `memory.save()` / reflect on the worker thread, push a `memory` message so an open panel refreshes (`send_sync`).
- `count` = scalar keys + list entries (excluding `superseded`).

**Frontend (`static/app.js`, `index.html`, `style.css`)**

- One `renderMemoryPanel(profile)` from a section spec table `{section, fields, keyField}` mirroring `memory.LIST_KEY`; no per-section code.
- Edits build op paths exactly as `_parse_path` accepts: `identity.name`, `people[2].note`, `episodic[]` (ADD), `projects[0]` (DELETE). Index-based paths are computed from the last `memory` message; the server reply re-renders, so a stale index after a concurrent save is refused by validation rather than hitting the wrong row.
- Restore = `UPDATE identity.<key>` with the superseded value (old current value becomes the new superseded — acceptable).

### Decisions
- Sectioned lists over a tree because the profile is depth-2 with six fixed sections; a tree adds chrome without information.
- Every UI write is an `apply_ops` op, never a direct file write: one validated path, audit and caps for free (same reason PRD 15 rejected LLM-rewritten files).
- A process-wide lock around load→apply→write rather than making `save()` async: the worker-thread save and the UI action are the only two writers; a lock is 4 lines.
- Undo = `superseded` for scalars and a short-lived re-ADD toast for list rows; no ops-log replay — the log stays append-only and human-readable.
- Assume the panel is used with < 200 facts; render the whole thing, no virtualisation/search. If false, add a filter box.
- Risk: the user edits while the model is mid-reply (profile already injected). Mitigation: none needed — the next turn re-runs `memory.load`; document it in the panel footer ("applies from the next reply").

## Data model

No schema change. Paths used by the UI are the existing ones (`_parse_path`):

| section | ADD value | UPDATE path | DELETE path |
|---|---|---|---|
| identity, preferences | `UPDATE <section>.<key>` (creates) | `<section>.<key>` | `<section>.<key>` |
| people, projects, recurring, episodic | `ADD <section>[]` + dict | `<section>[i].<field>` | `<section>[i]` |

Invariants: caps still apply (ADD beyond cap is rejected with reason); `superseded` only ever grows through UPDATE on scalars; `memory_ops.jsonl` gets one line per applied op with `reason: "ui"` (or `"ui:restore"`, `"ui:undo"`, `"ui:reset"`) so UI edits are distinguishable from LLM ops in the log.

## Test scenarios

Unit (mocked LLM, tmp memory path — `tests/test_memory_editor.py`):
1. `memory_op UPDATE identity.name "Mihaly"` on a profile where name is "Misi" → reply `memory` has `identity.name == "Mihaly"` and `identity.superseded.name == ["Misi"]`; one audit line with `reason == "ui"`.
2. `memory_op DELETE people[1]` on 3 people → reply has 2 people, the right one gone; `DELETE people[7]` → `memory_error` with reason containing "does not exist", profile unchanged on disk.
3. `memory_op ADD people[]` when people already at cap 30 → `memory_error` mentioning the cap; disk unchanged.
4. Malformed op (`op: "REPLACE"`, path `identity/name`, value `None` for ADD) → `memory_error`, no audit line, no file write (mtime unchanged).
5. `get_memory` → `rendered` equals `memory.render(profile)` and `count` equals number of user-visible facts (superseded excluded).
6. `memory_reset` → file equals `_empty_profile()` (+ version/updated), one audit line `DELETE *`.
7. Concurrency: run `memory.save(turns, mock_client)` in a thread that sleeps inside the mocked LLM call while a `memory_op` runs on the main thread → both changes present in the final file (lock held across load→write). Without the lock this test must fail (verified once during implementation by removing the lock).
8. `apply_ops(..., rejected=[])` returns the same reasons `_reject` prints; existing PRD 15 tests unchanged and green.

Frontend (manual, listed for the manual-testing pass):
9. Open panel with an empty profile → six sections with "nothing yet", count 0; add a preference → appears without reload; log has the line.
10. Edit a person's note, press Esc → reverts; press Enter → persists; refresh the page → still there.
11. Toggle "What the model sees" → text matches `python -c "import memory; print(memory.render(memory.load_profile()))"`.
12. Delete a project → toast "Undo delete" → click → project back at the end of the list.

## Stress / interaction testing with the memory system

Evaluated options, in order of value for cost:

1. **Harness scenario with mid-script UI edits (recommended, do it).** Extend `eval/harness.py` scripts with an `edit:` turn type: `{edit: {op, path, value}}` that calls the same `memory_op` code path between sessions. New script `eval/scripts/editor.yaml`: S1 introduces facts; between S1 and S2 the "user" corrects the job and deletes the dog; S2 probes that the assistant uses the corrected job, abstains on the dog (`expect_not: ["bolt"]`), and that the next LLM save does **not** re-add the deleted dog from session turns (the probe about the dog must not re-create `people[]`). This is the real risk of an editor: the extraction pass fighting the user's correction. Target: 100% on update/abstention probes, 0 resurrected entries after `--rounds 2`.
2. **Reflection vs. hand edits.** Run `editor.yaml` with `MEMORY_REFLECT_EVERY=1` so `reflect()` runs after every save; assert `_validate_reflection` keeps hand-added entries (they are grounded — they exist in `original`) and never restores a hand-deleted one. Cheap: same script, one env var, add to the results table.
3. **Property-style fuzz of `apply_ops` (unit, no LLM).** Hypothesis is not in requirements; a 30-line random-op generator in a test is enough: random sequences of 200 valid/invalid ops over all sections, assert after each step that the profile still passes `_normalise` round-trip, list lengths ≤ caps, `superseded` values are strings, and every applied op has an audit line. This is what protects the editor from index-shift bugs (delete `people[0]` then update `people[0].note`).
4. **Concurrent hammer (unit).** 4 threads each doing 50 `memory_op` calls plus one thread calling `memory.save` with a mocked LLM → final file valid JSON, count of applied ops equals audit lines, no `JSONDecodeError` from a torn write (atomic write + lock).
5. **Cap and budget at scale.** Seed a profile at every cap (30/20/15/40) via ops, open the panel: renders without jank (manual), `render()` stays ≤ `BUDGET_TOKENS` (already covered by PRD 15 test 2 — just reuse with the seeded profile). ADD at cap in the UI shows the cap error (unit test 3 covers the server side).

Not worth it: browser-automation tests of the panel (Playwright not in the stack, single-user app); ops-log replay tests (no replay feature).

## Out of scope

- Search/filter in the panel, virtualised lists.
- Undo via replaying `memory_ops.jsonl`; editing `superseded` history directly.
- Editing the persona/character cards (that's PRD 12 territory).
- Multi-user profiles; importing/exporting memory files.
- Fixing existing nickname-in-`identity.name` profiles automatically (the editor is the manual fix).
