# Voicebot

## Process

Follow [docs/process.md](docs/process.md). Every feature goes through Exploration -> Implementation -> Refactoring with quality gates between phases.

## Principles

- Pragmatic, no over-engineering
- Small-scale personal app
- No production code until the PRD is signed off
- Tests define "done" — if it's not tested, it's not finished
- Keep scope tight — cut anything non-essential
- AI never commits — user signs off and commits at each quality gate
- Never paste secrets/keys/tokens into prompts or code
- Delegate basic implementation to small sub-agents; the main thread plans, coordinates and reviews
- AI may commit on feature branches (no co-author trailers); never push

## Quick Reference

- PRDs: `docs/<number>-<feature-name>.md`
- Prototypes: `docs/prototypes/<feature-name>/`
- Repo index: `docs/index.md`
- Branching: `main` (clean) + `feature/<name>` branches
