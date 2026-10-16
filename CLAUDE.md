# Voicebot

## Process

Follow [docs/process.md](docs/process.md). Every feature goes through Exploration -> Implementation -> Refactoring with quality gates between phases.

## Principles

- Pragmatic, no over-engineering
- Small-scale personal app
- No production code until the PRD is signed off
- Tests define "done" — if it's not tested, it's not finished
- Keep scope tight — cut anything non-essential
- Never paste secrets/keys/tokens into prompts or code
- Delegate basic implementation to small sub-agents; the main thread plans, coordinates and reviews
- AI commits on feature branches at each quality gate (no trailers); user reviews and merges to `main`; AI never pushes

## Lessons

One line per real past failure. Add to this list, don't delete from it.

- Eager recall: a mis-transcribed four-word turn pulled a 6k-char wiki chunk into a chat reply. Ask before looking things up; gate recall on a content question, not on word count.
- CLAUDE.md and process.md contradicted each other on who commits. One rule, stated in one place.
- docs/index.md drifted (missing test files, stale PRD statuses) until tests/test_index.py enforced it. Docs nothing checks will rot; add a test.
- process.md promised ruff/black for months with neither installed. Don't promise tooling that scripts/check.sh doesn't run.
- The WebSocket protocol had two sources of truth (server dict + client ladder). One table in docs/protocol.md, enforced by a test.
- Without a single check command, `node --check` was never run. Everything goes through scripts/check.sh.
- "Tests pass" claimed from memory is worthless. Only this session's check.sh output with its counts proves green.

## Quick Reference

- PRDs: `docs/<number>-<feature-name>.md`
- Prototypes: `docs/prototypes/<feature-name>/`
- Repo index: `docs/index.md`
- Branching: `main` (clean) + `feature/<name>` branches
