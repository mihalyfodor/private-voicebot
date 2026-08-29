# Architecture review — 2026-08-29 (adversarial, read-only)

Ranked findings (severity — location — issue — fix):
1. HIGH — llm.py:339-344, memory.py:723-748,1523-1541 — session turns popped before save; propose_ops swallows transport errors → whole session memory lost silently when oMLX is down. Fix: propose_ops returns None on transport failure; llm.save_memory restores turns on failure.
2. HIGH — session.py:84-88, chatbot.py:165-174 — periodic memory save runs on the turn worker under memory._LOCK; every 5th turn the next utterance hangs 10–60 s with no UI signal. Fix: run hook on its own daemon thread (+ overlap guard), push a saving status.
3. MED — memory._LOCK shared by wiki/sources; memory.py 1564 lines; lazy circular imports (memory→wiki→llm→memory). Fix: vault_lock.py + clock.py.
4. MED — tests/test_turns.py:134,149,172, test_hands_free.py — unit tests write the real settings.json. Fix: autouse sandbox fixture in conftest.
5. MED — chatbot.py:191-197,217-224 — send() swallows all WS errors; worker blocks 10 s per message on a dead socket. Fix: drop the socket on first failure, set playback_done.
6. MED — one generic `error` message routed client-side by currentView (app.js:2268). Fix: `scope` field.
7. MED — three page-path validators (wiki.normalise_path/_target/_user_target, sources._target_allowed) + client WIKI_FOLDERS copy. Fix: single primitive with a folders option; ship FOLDERS via /api/config.
8. MED — docker blockers: mlx-whisper/sips/Kokoro paths (macOS), origin allow-list uses server PORT (chatbot.py:850), open_browser, obsidian:// links, ~/llm-wiki default. Cheap now: ALLOWED_ORIGINS env.
9. LOW-MED — chatbot.py:351-357 _on_shutdown does blocking LLM calls on the loop. Fix: to_thread + lifespan.
10. LOW-MED — dead avatar-switch actions (set_avatar, switch_greet, listing) still exposed; hands_free/verbosity state triplicated.
11. LOW — app.js 2600 lines single scope; split into ES modules when next touched.
12. LOW — vault git failures invisible (wiki.py:127-141,211). Fix: count failures, surface in lint list.

Top 3 to act on now: A = finding 4 (test sandbox), B = finding 1 (no lost turns), C = findings 2+3 (save off worker + vault_lock).
