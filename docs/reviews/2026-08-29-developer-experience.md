# Developer-experience review — 2026-08-29 (adversarial, read-only)

Baseline: unit suite 468 passed in 36 s, no flakes; node fixture 0.09 s.

Ranked findings:
1. CLAUDE.md:14 vs :17 and process.md Git section contradict on whether AI commits. Fix: one rule — AI commits on feature branches at gates, user merges, never push.
2. .env.example lists 17 vars, code reads 29; the isolation ones (WIKI_PATH, SETTINGS_PATH, MEMORY_DIR, HOST) are undocumented; still mentions haru. Fix: exhaustive, grouped .env.example.
3. e2e prerequisites undocumented (playwright not in requirements, fetch_assets, Kokoro); missing assets fail as opaque console errors. Fix: document + early guard in tests/e2e/conftest.py.
4. No single "check everything" command (node --check not listed anywhere). Fix: scripts/check.sh.
5. process.md promises ruff/black; neither installed nor configured. Fix: add ruff + minimal pyproject, or drop the line (user decision).
6. docs/index.md drift: 8 test files missing, PRD 19/20 shown refactoring but done, docs/reviews + prototypes absent. Fix: sync + tests/test_index.py enforcing it.
7. PRD status format: 03 uses bold/backticks and says implementation (index says done); 17/21 carry free text. Fix: normalise 03; allow parenthetical; "Research" bucket in index.
8. WS protocol has no single source of truth (ACTIONS dict at chatbot.py:803 + 30-line ladder app.js:2236). Fix: docs/protocol.md table.
9. Logs undocumented: prints, session.log, /tmp/voicebot.log only in agent memory, E2E_KEEP. Fix: "Logs" paragraph in index.md (+ scripts/run.sh).
10. No cache-busting for /static (fetch-loaded model json/backdrops can stale). Fix: Cache-Control: no-store middleware for /static.
11. tests/test_turns.py::test_scenario_12 spawns two pytest subprocesses (1.9 s). Fix: assert on pytestconfig.
12. README stale (menu backdrops, shortmem.txt, external process link; editor/inbox/palette unmentioned).

Top 3 now: A = finding 1; B = findings 2+4; C = findings 6+7+3(doc half).
