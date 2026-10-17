# Development Process

Small-scale personal apps. Pragmatic, no over-engineering.

## Phases

### Phase 1: Exploration

**Goal:** Lock down requirements in a detailed, actionable PRD before any production code is written.

**How it works:**
- Discuss the feature/app idea collaboratively
- Align on scope — cut anything that isn't essential for personal use
- Define concrete test scenarios (these become acceptance criteria for Phase 2)
- Optionally build a clickable prototype to explore UX ideas (see Conventions)
- Write the PRD to `docs/<number>-<feature-name>.md`

**New app?** The first PRD should broadly explore the domain and identify the main features, then slice them vertically so each subsequent PRD targets one deliverable slice.

**PRD structure:**

Each PRD starts with a status line at the top:

```
Status: exploration
```

Status meanings:
- `exploration` — requirements being defined, no production code yet
- `implementation` — PRD signed off, code being written
- `refactoring` — implementation complete, tests passing, now cleaning up
- `done` — refactoring finished, merged to main

Followed by:
1. Overview — what and why, in 2-3 sentences
2. User stories — short, concrete
3. UI/UX — screens, flows, key interactions
4. Technical approach — stack, architecture decisions, constraints
   - **Decisions:** 3-6 one-liners capturing key choices
     - "Chose X over Y because Z"
     - "Assume A; if false, revisit"
     - "Risk: B; mitigation: C"
5. Data model / state shape — entities and key invariants (even a rough sketch)
6. Test scenarios — critical paths that must be covered by tests. Each scenario must name its expected observable result.
7. Out of scope — what we're explicitly not doing

**Gate checklist:**
- [ ] PRD has: overview, user stories, UI/UX, technical approach with decisions, data model, test scenarios, out of scope
- [ ] Each test scenario names its expected observable result
- [ ] Out of scope is explicit
- [ ] Status set to `exploration`
- [ ] Feature branch created: `feature/<name>`
- [ ] AI commits on the feature branch with message: `PRD: <feature>`; user reviews

**After gate:** Update PRD status to `implementation`.

**Model:** Opus

### Phase 2: Implementation

**Goal:** Build exactly what the PRD specifies. All test scenarios from the PRD implemented and passing.

**Kickoff (mandatory):**
Before writing any code, AI must:
- Reference the PRD
- Restate scope and out-of-scope
- Copy the test scenarios verbatim
- List assumptions explicitly
- If blocked, ask — don't invent

**How it works:**
- Implement features per the PRD
- Write tests following the test pyramid (see Defaults)
- Tests must be runnable via a single command documented in `docs/index.md`, and passing before handoff
- If a change affects behavior or acceptance criteria, stop and update the PRD (or write a follow-up PRD). No "quick improvements."
- User performs manual testing after: execute every PRD test scenario once end-to-end
- If implementation gets stuck on architecture, escalate to Opus for a single design-skeleton pass, then continue on Sonnet

**When manual testing finds issues:**

Escalate based on symptoms:
- **Small fix** — discuss in chat; write a regression test and see it fail first; fix; re-run tests; commit as `Fix: <what>`
- **Bigger problem** — soft reset, write a focused follow-up PRD. Triggers:
  - Same fix needed twice (regression loop)
  - Can't explain the failure in 1-2 sentences
  - Fix requires touching multiple areas (e.g. schema + UI + tests)
  - New behavior needed that isn't in the PRD
- **Full fail** — drop the branch, restart with better instructions. Triggers:
  - Tests become flaky / nondeterministic
  - Repeated regressions after follow-up PRD
  - Requirements still unclear after revision

**Review (one pass, before the gate):**
Spawn the reviewer agent (`.claude/agents/reviewer.md`) on the branch diff. It never edits; it reports BLOCKER / MAJOR / backlog findings and states what it checked and ruled out. Fix every BLOCKER and MAJOR before the gate; backlog items go into the PRD's Out of scope or a follow-up PRD.

**Gate checklist:**
- [ ] `scripts/check.sh` green in this session — quote its passed/skipped/failed counts in the handoff; never report green from memory
- [ ] Smoke pass for UI-visible or LLM-path changes: `scripts/smoke.py` (the /smoke skill), screenshots read, RESULT.txt quoted; backend-only changes may skip it
- [ ] Reviewer pass done; no open BLOCKER or MAJOR
- [ ] User has executed every PRD test scenario once end-to-end
- [ ] PRD status updated to `refactoring`
- [ ] `docs/index.md` updated
- [ ] AI commits on the feature branch with message: `Impl: <feature>`; user reviews

**Model:** Sonnet

### Phase 3: Refactoring

**Goal:** Consolidate and improve, keeping tests green throughout.

**How it works:**
- Review for consolidation, simplification, deduplication
- Run tests after every change — never break green
- No new features in this phase
- Must not change externally observable behavior; if it does, it's a PRD change. Observable = user-visible UI/behavior, URL/route behavior, persisted data shape, public component props.
- Files over ~1000 lines are split candidates (today: memory.py, wiki.py, sources.py, chatbot.py). Judgement call, not a gate.

**Gate checklist:**
- [ ] `scripts/check.sh` green in this session — quote its passed/skipped/failed counts in the handoff; never report green from memory
- [ ] No change in externally observable behavior (UI, routes, persisted data, public props)
- [ ] PRD status updated to `done`
- [ ] `docs/index.md` updated
- [ ] AI commits on the feature branch with message: `Refactor: <feature>`
- [ ] AI runs the smoke pass (`/smoke`); green → AI fast-forward-merges the branch to `main`; red → it stays on the branch and the user decides

**Model:** Opus

## Git

- `main` stays clean and deployable
- Feature branch created at Phase 1 sign-off: `feature/<name>`
- Each quality gate = one AI commit on the feature branch (`PRD:` / `Impl:` / `Refactor:` messages, no trailers)
- AI merges to `main` after Phase 3 once this session's `scripts/check.sh` and the smoke pass are green (user rule of 2026-10-17); the user reviews on `main`
- AI never pushes
- Bug fixes outside a PRD phase: `Fix: <what>` commits, each with a regression test that failed before the fix
- A commit-msg hook (`scripts/hooks/commit-msg`, enabled via `git config core.hooksPath scripts/hooks`) rejects AI-attribution trailers

## Repo Index

`docs/index.md` — lightweight map of the repo.

- Folder tree + one-liner per key file
- How to run the app
- How to run tests
- Links to current/active PRDs
- Updated at the end of Phase 2 and Phase 3
- Not exhaustive — just enough for fast orientation on a cold start
- AI consults this first when picking up an existing project

## Defaults

Apply to all projects unless a PRD explicitly overrides.

- **Linter:** ruff (`ruff.toml`), run by `scripts/check.sh`. No formatter.
- **Secrets:** Never paste secrets/keys/tokens into prompts or code. Use `.env` + `.gitignore`. Sanitize logs before sharing.
- **Error handling:** Fail fast with clear messages. No silent swallows.
- **Test split:** Pure logic → unit tests. API/service boundaries → integration tests. User-critical flows → E2E. Aim for more unit/integration than E2E.
- **Test layout:** `tests/<module>/test_<feature>.py`, one folder per app module (llm, memory, wiki, sources, session, ui, audio). A new PRD adds tests to the existing file for that module before it adds a file. Fakes, sandboxes and helpers live in `tests/conftest.py` — never copy one into a test file, and never import a fixture from another test file. Assert on the messages a scenario is about (filter by type), not on the exact full frame list, unless wire order is the contract.
- **Storage:** Define where state lives in the PRD (local file, memory, external service, etc.)

## Conventions

- **Tech stack:** Python, discussed per feature
- **PRD files:** `docs/<number>-<feature-name>.md`
- **Prototypes:** Optional Phase 1 output. HTML/JS/CSS only, stored in `docs/prototypes/<feature-name>/`. Kept for reference but never used as a base for production code.
- **Scope discipline:** If it's not in the PRD, it doesn't get built. If we discover something new mid-implementation, stop and update the PRD first. No "quick improvements."
