Status: refactoring

# 18 — Memory decay: `seen`, stale entries, archive group, drag-and-drop

Slice "17a" of [17-long-term-memory.md](17-long-term-memory.md).

## Overview

Profile entries (people, projects, recurring) currently never age: they leave only through an LLM DELETE
or the "oldest out" cap. This PRD adds a last-confirmed date (`seen`) to every list entry, a `CONFIRM` op so
re-mentions refresh it, and a reflection rule that marks entries `stale` after a per-section window. Stale
entries leave the prompt but stay on disk, shown in a "stale" group in the memory editor with restore/delete.
The editor gains drag-and-drop for the two moves that make sense inside the profile: episodic → recurring
(promote) and any row → stale (retire).

## User stories

- As a user, a colleague I mentioned once six months ago is not presented as current news; if I mention them
  again, they come back without me doing anything.
- As a user, I can see when a fact was last confirmed, and which facts the assistant has stopped using.
- As a user, I can drag a repeated episodic ("gym again") onto Recurring to make it a habit, or drag anything
  onto Stale to retire it, with a "→" menu doing the same for trackpad-hostile moments.
- As the developer, nothing is deleted automatically: forgetting = leaving the prompt.

## UI/UX (memory view, keeps PRD 16 simple)

- Every list row shows a muted `seen 2026-05-12` (relative: "3 months ago") at the right; rows past half their
  window get slightly dimmed; stale rows move to a **Stale** group at the bottom of their section (collapsed
  when empty) with **restore** (sets `seen: today`, clears `stale`) and 🗑.
- Drag-and-drop (HTML5 DnD, `draggable` rows, `dragover` highlights valid targets only):
  - episodic row → **Recurring** section header: creates `recurring` entry `{what: text, when: ""}` and deletes
    the episodic (one `ADD` + one `DELETE`, sent as a single `memory_ops` batch so it's atomic).
  - any list row → its section's **Stale** group header: `UPDATE <section>[i].stale = true`.
  - stale row → its section header: restore.
  - Invalid targets don't highlight; a drop elsewhere does nothing.
- "→" button on each row opens a 2–3 item menu with the same moves.
- Model view unchanged (stale entries are simply absent).

## Technical approach

- Schema (`memory.json` stays version 2; fields optional, `_normalise` backfills):
  list entries gain `seen: "YYYY-MM-DD"` (set on ADD/UPDATE/CONFIRM; backfilled with `updated` date on first
  load) and optional `stale: true`.
- `PROPOSE_SYSTEM` gains `CONFIRM` op: `{"op":"CONFIRM","path":"people[2]"}` — "emit when the user refers to
  an existing entry again and nothing changed". `apply_ops` handles CONFIRM: sets `seen` = today, clears `stale`;
  logged like other ops (reason "confirm"). Never rejected for stale entries (that's how they come back).
- Windows: `STALE_DAYS = {"people": 180, "projects": 120, "recurring": 90}` (env `MEMORY_STALE_*`);
  `reflect()` runs `_mark_stale(profile)` in Python **before** the LLM call (no model involvement) and
  `_validate_reflection` keeps `stale`/`seen` fields from the original for every traceable entry. Episodic keeps
  TTL, unchanged.
- `render()` skips `stale` entries; `_enforce_caps` evicts stale first, then oldest `seen` (replaces "oldest out").
- `snapshot()` adds `seen`, `stale`, and `stale_after` (the window) per entry so the UI can dim.
- New action `memory_ops` (batch): list of ops applied under one lock hold; all-or-nothing (a rejected op in the
  batch rejects the batch, nothing written). `memory_op` stays for single edits.
- Harness: new turn type `edit: {op, path, value}` and `clock: "+200d"` (monkeypatches `memory._today`) in
  `eval/harness.py`; scripts `eval/scripts/editor.yaml` (PRD 16 stress item 1) and `eval/scripts/stale.yaml`.

### Decisions
- `seen`/`stale` bitemporal fields over a decay score: inspectable, no tunables beyond three windows.
- Staleness computed in Python inside reflection, not by the LLM: deterministic, testable with a mocked clock.
- Restore = user action or CONFIRM; the model never un-stales by itself except through CONFIRM tied to a real
  mention.
- Batch action for DnD promote so the profile never shows both entries or neither.
- Assume windows 180/120/90 d feel right; if false, they are env vars.

## Data model

```json
{"name":"Anna","rel":"colleague","note":"joined in August","seen":"2026-08-29","stale":false}
```
Invariants: `seen` ≤ today; `stale` ⇒ not rendered; CONFIRM ⇒ `stale` false and `seen` = today; caps evict
stale entries before live ones; no automatic DELETE of non-episodic entries anywhere.

## Test scenarios

1. `_normalise` backfills `seen` on an old profile (all entries get the profile `updated` date); file unchanged
   until the next write.
2. `apply_ops` ADD/UPDATE set `seen` = today; CONFIRM on `people[1]` sets `seen` = today and clears `stale`;
   CONFIRM on a missing index is rejected; CONFIRM logged with reason `confirm`.
3. `_mark_stale` with a mocked `_today` 200 days after `seen`: people/projects/recurring entries become `stale`;
   an entry seen 100 days ago is not; episodic untouched.
4. `render()` omits stale entries; `fact_count` still counts them; snapshot carries `stale`, `seen`, `stale_after`.
5. `_enforce_caps` at cap 30 with 2 stale entries: the stale ones go first, then the oldest `seen`.
6. `_validate_reflection` preserves `seen`/`stale` even if the LLM output dropped them.
7. `memory_ops` batch: ADD recurring + DELETE episodic[i] applies atomically; a batch with one bad op writes
   nothing and returns the reason.
8. Harness `stale.yaml` (live, mocked clock): S1 introduces Anna; `clock: +200d`; S2 probe "how is Anna settling
   in?" → `expect_not: ["settling", "new"]` and abstention wording; user says "had lunch with Anna" → next probe
   "who is Anna?" → `expect: ["colleague"]` (CONFIRM brought her back). Unit-level version with a mocked LLM
   emitting the CONFIRM op.
9. Harness `editor.yaml` (PRD 16 stress item 1): hand DELETE of the dog + UPDATE job between sessions; probe
   `expect_not: ["bolt"]`, job probe expects the corrected title; after `--rounds 2` the dog has not been
   re-added.
10. Manual: drag an episodic onto Recurring → appears there, gone from Episodic, log shows ADD+DELETE; drag a
    person onto Stale → dims and moves to the group; restore brings it back with today's date; "→" menu does
    the same.

## Out of scope
Archiving to the wiki (PRD 19 adds the archive target); decay for identity/preferences (they only change by
UPDATE); any numeric scoring; DnD between sections other than the two moves above.

## Results (harness, live, 2026-08-29, gemma-4-26B)

| script | probes | notes |
|---|---|---|
| `stale.yaml` | 2/2 | Anna retired by the +200 d save (`[Memory] stale — people: Anna`), abstention on "how is Anna settling in?", real mention → `CONFIRM`, entry live again with the original note |
| `editor.yaml` ×2 rounds | 5/5, 5/5 | hand `DELETE` of the dog + `UPDATE` job between sessions; dog never re-added across rounds; 0 duplicates; 26 → 37 tokens rendered |

Run with `.env` sourced (`set -a; source .env; set +a`) — the harness does not load it itself.
