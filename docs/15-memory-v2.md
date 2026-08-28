Status: done

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
- `propose_ops(profile, turns, client, model) -> list[op]`: prompt per research sketch; only facts *the user stated about themselves*; never assistant statements, world facts, or transients unless dated episodic; output JSON array `{"op","path","value","reason"}`; `[]` if nothing. A preferred form of address goes to `identity.nickname`, never `identity.name`; everyday trivia (meals, sleep, weather, commute) is skipped entirely rather than stored as a low-importance episodic.
- `apply_ops(profile, ops)`: schema-validated; UPDATE on a scalar path moves the old value into `superseded`; DELETE removes; unknown paths / malformed values are rejected and logged, never applied; caps per list (people 30, projects 20, recurring 15, episodic 40); returns applied ops.
- Audit: append applied ops to `memory_ops.jsonl`.
- `reflect(profile, client, model)`: every 5 saves (or on demand), merge near-duplicate list entries, drop expired episodics, promote ≥3 similar episodics into `recurring`; result validated the same way (never a blind overwrite).
- `render(profile, budget_tokens=700)`: profile block + top-5 episodics of importance ≥ 2 by recency×importance, within budget; the block that goes into the prompt. `identity.nickname` is folded into the name line (`name: Mihaly (call them: Misi)`).
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

## Results (harness, live, 2026-08-28)

| script | round | v1 | v2 iteration 1 | v2 iteration 2 |
|---|---|---|---|---|
| basic | 1 / 2 | 11/11 / 11/11 | 11/11 / 11/11 | 11/11 |
| long-horizon | 1 | 13/15 | 12/15 | **15/15** |
| update-heavy | 1 / 2 | 10/13 / 7/13 | 12/13 / 13/13 | 12/13 / 12/13 |

v1 notes: transients leaked into the fact log ("had eggs for breakfast", "went for a run");
4 job titles + 3 cities on disk at once; a replay re-appends the old values and answers drift
back to them. Both long-horizon fails were the persona's "boss" beating the user's "call me Misi".

v2 iteration 1: no duplicates, updates clean (`superseded` keeps the old value out of the prompt),
but three new problems — the persona still won over the stated preference (3 fails: name, "what
should you call me", proactive greeting), "call me Misi" was stored as `UPDATE identity.name`
(the real name Mihaly was pushed into `superseded`), and 9 one-day episodics ("had a sandwich for
lunch", "slept badly") were stored with 8 of them rendered under Recent.

v2 iteration 2 (this pass): `identity.nickname` for a preferred form of address, rendered as
`name: Mihaly (call them: Misi)`; a fixed prompt rule that a recorded preferred name overrides
any habit from the character card, plus the same escape hatch in each card's `speaking_style`;
the extraction prompt now skips everyday trivia outright and only writes an episodic with future
relevance (importance ≥ 2), and `render` shows at most 5 episodics of importance ≥ 2.
Long-horizon went 12/15 → 15/15 with 1 episodic stored and 1 rendered (was 9 stored / 8 rendered),
and the real name survived alongside the nickname. Update-heavy round 2 lost one probe versus
iteration 1 ("when's the marathon now?" — the model answered "back in June" without the date; the
marathon had been promoted from `projects` to `recurring: marathon training (June 2nd)`), and
round 1 misses "where did I live before Vienna?" by design: `superseded` is deliberately not
rendered, so the assistant abstains instead of naming Budapest.

Migration note: a profile written before this pass may hold the nickname in `identity.name` with
the real name under `identity.superseded["name"]`. Nothing repairs that automatically — guessing
which of the two is the real name is worse than leaving it; the user restating their name fixes it.

## Evaluation targets (harness, live)
- basic.yaml round 1: single-hop ≥ 90%, update = 100% with `expect_not` clean, abstention ≥ 80%, transient leakage = 0 stored transients.
- 3 rounds: memory size grows sublinearly; duplicates = 0.

## Out of scope
Embedding retrieval; multi-user profiles; UI for editing memory (a later menu item); memory for assistant-side facts.
