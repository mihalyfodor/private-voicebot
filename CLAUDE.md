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

## Quick Reference

- PRDs: `docs/<number>-<feature-name>.md`
- Prototypes: `docs/prototypes/<feature-name>/`
- Repo index: `docs/index.md`
- Branching: `main` (clean) + `feature/<name>` branches
