Status: implementation

# 15 — Memory v2: structured profile with upsert + reflection

## Overview

Replace the append-only `shortmem.txt` fact log with a two-tier structured profile: a durable **profile** (identity, preferences, people, projects, recurring) plus dated **episodic** entries with TTL. The LLM proposes upsert **ops** (ADD / UPDATE / DELETE / NOOP) against the current profile; Python validates and applies them, keeps an audit log, and runs a periodic **reflection** pass to merge duplicates, expire episodics and promote repeats. The profile is injected after the persona with wording that encourages natural use ("as a friend would") instead of "never bring it up".

Grounded in the 2026-08-28 research pass (Mem0 ops, Letta core memory, Generative Agents decay/reflection, Zep `superseded`) and measured with the PRD 14 harness.

## User stories

- As a user, the assistant remembers my name, my dog, my preferences and uses them naturally without me prompting.
- As a user, when I say my job changed, it stops mentioning the old one (but can say "you used to…").
- As a user, yesterday's weather or lunch never becomes a permanent "fact" about me.
- As a user, when it doesn't know something, it says so instead of inventing.
- As the developer, memory stays small and readable (`memory.json`), with an undoable audit trail.

## Technical approach

**Data (`memory.json`, atomic writes, `version: 2`)**
```json
{"version": 2, "updated": "2026-08-28T21:00",
 "identity": {"name": "…", "location": "…", "occupation": {"v": "…", "since": "…", "superseded": ["…"]}},
 "preferences": {"coffee": "black"},
 "people": [{"name": "Anna", "rel": "colleague", "note": "joining next week"}],
 "projects": [{"name": "half marathon", "status": "training", "note": "race in November"}],
 "recurring": [{"what": "gym", "when": "Tue/Thu"}],
 "episodic": [{"date": "2026-08-28", "text": "…", "ttl_days": 30, "importance": 2}]}
```
Migration: on first load, if `shortmem.txt` exists and `memory.json` doesn't, run one extraction over its lines to seed the profile; keep the old file untouched.

**Ops (`memory.py`)**
- `propose_ops(profile, turns, client, model) -> list[op]`: prompt per research sketch; only facts *the user stated about themselves*; never assistant statements, world facts, or transients unless dated episodic; output JSON array `{"op","path","value","reason"}`; `[]` if nothing.
- `apply_ops(profile, ops)`: schema-validated; UPDATE on a scalar path moves the old value into `superseded`; DELETE removes; unknown paths / malformed values are rejected and logged, never applied; caps per list (people 30, projects 20, recurring 15, episodic 40); returns applied ops.
- Audit: append applied ops to `memory_ops.jsonl`.
- `reflect(profile, client, model)`: every 5 saves (or on demand), merge near-duplicate list entries, drop expired episodics, promote ≥3 similar episodics into `recurring`; result validated the same way (never a blind overwrite).
- `render(profile, budget_tokens=700)`: profile block + top-8 episodics by recency×importance, within budget; the block that goes into the prompt.
- Same triggers as today: every 10 turns and on shutdown (via `llm.save_memory()`).

**Prompt wording (`memory.load`)**
`<user_profile>…</user_profile>` — "facts about the user you already know. Use them naturally, as a friend would (name, preferences, ongoing projects). Don't recite the list or announce that you remembered; don't raise sensitive items unprompted. If a fact conflicts with what the user says now, believe the user."

**Harness**: PRD 14 scripts + a new `eval/scripts/update-heavy.yaml` (repeated job/location changes) and `--rounds 3` bloat check; targets below.

### Decisions
- Ops applied in Python, never an LLM-rewritten file: prevents silent deletions (Mem0/Letta lesson).
- Whole profile in context, no retrieval: <500 facts ≈ 1–2k tokens, well inside the 16k budget; revisit at ~1000 facts (BM25 first).
- Keep the "extract at save time" cadence (background), not per-turn tool calls: no added voice latency.
- `superseded` instead of delete on UPDATE: enables "you used to" and undo.
- Per-list caps + reflection over FIFO: the current last-60-lines window evicts the oldest *stable* facts first — the worst possible policy.

## Test scenarios (unit, mocked LLM)
1. `apply_ops`: ADD/UPDATE/DELETE/NOOP on each path type; UPDATE stores `superseded`; malformed op rejected and logged; caps enforced.
2. `render` stays within budget and orders episodics by recency×importance; expired TTLs excluded.
3. `reflect` merges two near-duplicate people entries and drops an expired episodic (mocked LLM output validated).
4. Migration from `shortmem.txt` seeds `memory.json` once and leaves the text file intact.
5. `memory.load` injects the `<user_profile>` block with the new wording; no "never bring it up".
6. Atomic write + audit line per applied op.

## Baseline (memory v1, 2026-08-28)

| script | round | pass | notes |
|---|---|---|---|
| basic | 1 / 2 | 11/11 / 11/11 | 0 duplicates; stale job+marathon lines both kept on disk |
| long-horizon | 1 | 13/15 | transients leaked ("had eggs for breakfast", "went for a run"); 2 fails = persona "boss" overriding the user's "call me Misi" |
| update-heavy | 1 / 2 | 10/13 / 7/13 | 4 job titles + 3 cities on disk; replay re-appends old values; answers drift to stale ones |

Also found: character `speaking_style` ("calls the user boss") beats a stated user preference → v2 must render preferences so they win (and the persona rule should say "unless the user asked otherwise").

## Evaluation targets (harness, live)
- basic.yaml round 1: single-hop ≥ 90%, update = 100% with `expect_not` clean, abstention ≥ 80%, transient leakage = 0 stored transients.
- 3 rounds: memory size grows sublinearly; duplicates = 0.

## Out of scope
Embedding retrieval; multi-user profiles; UI for editing memory (a later menu item); memory for assistant-side facts.
