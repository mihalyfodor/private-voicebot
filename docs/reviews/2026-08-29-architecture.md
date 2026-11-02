Statuses added 2026-11-02.

# Architecture review — 2026-08-29 (adversarial, read-only)

Ranked findings (severity — location — issue — fix):
1. HIGH — llm.py:339-344, memory.py:723-748,1523-1541 — session turns popped before save; propose_ops swallows transport errors → whole session memory lost silently when oMLX is down. Fix: propose_ops returns None on transport failure; llm.save_memory restores turns on failure.
Status: resolved (`propose_ops` returns None on a failed call, agent/fold.py:107; `save_memory` restores the turns, agent/conversation.py:136)
2. HIGH — session.py:84-88, chatbot.py:165-174 — periodic memory save runs on the turn worker under memory._LOCK; every 5th turn the next utterance hangs 10–60 s with no UI signal. Fix: run hook on its own daemon thread (+ overlap guard), push a saving status.
Status: resolved (periodic hook runs on its own daemon thread, server/session.py:94-112)
3. MED — memory._LOCK shared by wiki/sources; memory.py 1564 lines; lazy circular imports (memory→wiki→llm→memory). Fix: vault_lock.py + clock.py.
Status: resolved (core/vault_lock.py; clock in core/util.py; memory.py split into memory/ + agent/fold.py in PRD 50)
4. MED — tests/test_turns.py:134,149,172, test_hands_free.py — unit tests write the real settings.json. Fix: autouse sandbox fixture in conftest.
Status: resolved (autouse `file_sandbox` + `wiki_sandbox` fixtures, tests/conftest.py:52-73)
5. MED — chatbot.py:191-197,217-224 — send() swallows all WS errors; worker blocks 10 s per message on a dead socket. Fix: drop the socket on first failure, set playback_done.
Status: resolved (`send()` forgets a dead socket at once, server/panels.py:178; ws.py sets `playback_done` on disconnect)
6. MED — one generic `error` message routed client-side by currentView (app.js:2268). Fix: `scope` field.
Status: resolved (`scope` field + `ERROR_SCOPES`, server/panels.py:225-234; client routes on it, table in docs/protocol.md)
7. MED — three page-path validators (wiki.normalise_path/_target/_user_target, sources._target_allowed) + client WIKI_FOLDERS copy. Fix: single primitive with a folders option; ship FOLDERS via /api/config.
Status: partly — one primitive exists (`vault/store.normalise_path(user=)`, `FOLDERS`), but `pages._target`/`_user_target` wrappers, `ingest/pipeline._target_allowed` and the client `WIKI_FOLDERS` copy (static/app.js:1454, not shipped via /api/config) remain
8. MED — docker blockers: mlx-whisper/sips/Kokoro paths (macOS), origin allow-list uses server PORT (chatbot.py:850), open_browser, obsidian:// links, ~/llm-wiki default. Cheap now: ALLOWED_ORIGINS env.
Status: partly — `ALLOWED_ORIGINS` env (server/ws.py:133) and dev-by-default vault path (core/config.py:7) done; `open_browser` (server/app.py:136) and obsidian:// links remain
9. LOW-MED — chatbot.py:351-357 _on_shutdown does blocking LLM calls on the loop. Fix: to_thread + lifespan.
Status: resolved (lifespan + asyncio.to_thread, server/app.py, 2026-11-02)
10. LOW-MED — dead avatar-switch actions (set_avatar, switch_greet, listing) still exposed; hands_free/verbosity state triplicated.
Status: partly — the WS `set_avatar`/`switch_greet`/listing actions are gone (PRD 23, not in server/ws.py ACTIONS); `hands_free` state still lives in settings.json + `panels.hands_free` and an internal `conversation.set_avatar` remains; roadmap backlog keeps it open
11. LOW — app.js 2600 lines single scope; split into ES modules when next touched.
Status: open — by design (roadmap backlog): static/app.js still one scope, ES-module split when next touched
12. LOW — vault git failures invisible (wiki.py:127-141,211). Fix: count failures, surface in lint list.
Status: resolved (vault.store.GIT_FAILURES surfaced by vault.lint, 2026-11-02)

Top 3 to act on now: A = finding 4 (test sandbox), B = finding 1 (no lost turns), C = findings 2+3 (save off worker + vault_lock).
