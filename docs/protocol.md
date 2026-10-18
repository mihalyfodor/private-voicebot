# WebSocket protocol

One socket, `ws://localhost:8011/ws`, one page at a time (a second connection replaces the first). Text
frames are JSON objects; the browser's binary frames are 16 kHz Int16 mic audio (hands-free only). Boot
state (`avatar`, `backdrop`, `backdrops`, `hands_free`, `verbosity`, settings facts) comes from `GET
/api/config`; file intake goes over `POST /ingest` and only its progress comes back on the socket. A run
card's "Open run" link is `GET /runs/{id}` (PRD 51): a server-rendered page of that run's events, behind the
same origin check as `/ingest`; an unknown or aged-out id is a 404 "No run <id> (it may have aged out of the log)."

The server half is the `ACTIONS` dict in `chatbot.py` (one `_action_<name>` handler each); the client
half is the `ws.onmessage` ladder near the bottom of `static/app.js`. `tests/test_protocol_doc.py`
fails when either side names something this file does not.

## Client → server (`{"action": ...}`)

An unknown action (or a frame with no `action`) answers `error` (settings, `action` = what was sent) and a trace error event; the socket stays open. An exception inside a handler comes back as `error` (scope from the
action name, plus `action`).

| action | fields | replies |
|---|---|---|
| `text_turn` | `text` | a full turn: `transcript` (user), `state` thinking/speaking, `speech`…, `speech_end`, `transcript` (assistant), `state` idle. "write it up" / typed `/write` (PRD 43) answer with a `card` (status running, PRD 51), one spoken line, then `card_update` (done + the markdown, or failed); "what's new?" adds one run `card` (PRD 51); "research X" / typed `/research X` (PRD 54, typed or spoken, never confirmed) speaks one line "Researching that — written out below.", then a research `card` (status running) and one `card_update` per finished step while the state stays thinking, the last with status done/partial/nothing + the markdown, title and sources (the body is never spoken); "write it up" / `/write` right after it sends that card's `card_update` again (same id) and one spoken line, no new card; "save that as a note" with the `save_card` replies then one spoken line; "forget the <title> guide/notebook/file/…" / "remove the <title> source" and "send that to my reMarkable" propose (PRD 52): an awaiting `card` + one spoken question (a title naming none or several sources is one spoken line instead; without `REMARKABLE_PUSH_URL` the push is one spoken line "The reMarkable push service is not configured." and nothing is pending; no card to send → `error` (turn) "nothing to send"); while one is pending the next turn decides it — an explicit yes approves (`resolve`'s replies; "delete it" / "send it" only for their own kind), a short no or anything opening "forget it/that/this/about" or "never mind" rejects ("Left it."), anything else drops it silently and runs as usual |
| `set_hands_free` | `value` (bool) | `hands_free`, then `listening` idle when switched off |
| `set_focus` | `view`, `path?` | none — the server remembers what the user is looking at (PRD 38): a page in Wiki / Sources rides on the next turns, Chat drops it |
| `playback_done` | `turn` | none — releases the worker waiting for playback (stale turn ids ignored) |
| `playback_blocked` | `turn` | same as `playback_done` (autoplay refused; audio plays after the next gesture) |
| `set_backdrop` | `key` | `backdrop`, or `error` (settings) for an unknown key |
| `reload_characters` | — | `characters_reloaded` (the YAML card re-read), or `error` (settings) |
| `set_verbosity` | `value` short/normal/long | `verbosity`, or `error` (settings) |
| `set_setting` | `key` in `SETTABLE`, `value` | `setting`, or `error` (settings) for an unknown key / wrong type. `key: omlx_admin_password` (a string, `""` clears) answers `setting {key, omlx_admin_password_set, omlx_admin_password_source}` — never the value |
| `set_location` | `lat`, `lon`, `tz` | `setting` (`key: location`, the resolved value), or `error` (settings) |
| `model_load` | — | loads `OMLX_MODEL` through the oMLX admin API with the resolved password (PRD 41 addendum): `services` after a re-probe, or `error` (settings) naming the reason (no password, login refused, oMLX down) |
| `model_unload` | — | the same for unload |
| `get_memory` | — | `memory` now; a second `memory` after the save when the session has unsaved turns |
| `save_memory` | — | `memory` (before) + `memory` (after the save), or `error` (memory) while a turn runs |
| `memory_op` | `op`, `path`, `value`, `reason?` | `memory`, or `memory_error` |
| `memory_ops` | `ops: [{op, path, value, reason?}]` | `memory` (all-or-nothing), or `memory_error` naming the first bad op |
| `memory_reset` | — | `memory` (empty profile) |
| `get_wiki` | — | `wiki` + `sources` |
| `read_wiki_page` | `path` | `wiki_page` |
| `wiki_note` | `path`, `text` | `wiki_page` + `wiki` + `sources`, or `error` (wiki) |
| `wiki_ingest` | — | `memory` + `wiki` + `sources` ("Re-ingest last session"), or `error` (intake) |
| `wiki_save` | `path`, `text`, `etag` | `wiki_page` + `wiki` + `sources`, `wiki_conflict` on a stale etag, or `error` (wiki) |
| `wiki_create` | `path` | `wiki_page` + `wiki` + `sources`, or `error` (wiki) |
| `wiki_rename` | `path`, `new_path` | `wiki_page` (moved) + `wiki` + `sources`, or `error` (wiki) |
| `wiki_delete` | `path` | `wiki` + `sources`, or `error` (wiki) |
| `wiki_search` | `query` | `wiki_search` |
| `get_sources` | — | `sources` |
| `reingest_source` | `sha` | `wiki` + `sources`, or `error` (sources) |
| `forget_source` | `sha` | the forget proposal turn (PRD 52): an awaiting `card` (workflow forget, `actions` Approve / Reject / Open run) in the transcript and one spoken question — nothing is deleted until it is approved (`resolve` or a spoken / typed "yes"); `error` (sources) for an unknown sha |
| `resolve` | `run_id`, `decision` approve/reject | the pending approval decided on the turn worker (PRD 52): approve → the write in one commit, `card_update` (status done, `commit`, only Open run left), after a forget `wiki` + `sources`, one spoken line ("Forgotten."); reject → `card_update` (status rejected) + "Left it."; a `run_id` that is not the pending one → `error` (turn) "that request expired", plus `card_update` (status expired, or its own resolved status; only Open run) for that run's card; the proposal changed since it was shown → `error` (turn) "that changed since I asked — ask again" and `card_update` failed; an unknown decision → `error` (turn) |
| `archive_memory` | `section`, `index` | `memory` + `wiki` + `sources`, or `memory_error` |
| `pull_wiki` | `path`, `section` | `memory` + `wiki` + `sources`, or `memory_error` |
| `new_topic` | — | a new-topic turn (PRD 39): `state` thinking, `thread` (the summary), the spoken ack as `speech`…/`speech_end`/`transcript`, `context`, `state` idle; typed `/new` and the exact phrase "new topic" (typed or spoken) take the same path through `text_turn` / the mic |
| `save_card` | `id?` (the last card when omitted), `folder?` (default `notes`) | `wiki_open` + `card_saved` + `wiki` + `sources` — the card filed as `<folder>/<date>-<slug>` in one commit (PRD 43) — or `error` (wiki): "nothing to save", "already saved", a bad folder / an existing page |
| `shutdown` | — | none — the process gets SIGTERM; the page shows its shutdown screen when the socket closes |

## Server → client (`{"type": ...}`)

`state`, `speech` and `speech_end` (and the assistant `transcript`) carry `turn`; the client adopts the
turn id from `state` and answers `playback_done` with it. On connect the server replays what the model
holds: the conversation's `transcript`s (bare user text, PRD 39), any `thread` divider and any written `card` (PRD 43), then sends `hands_free` and either greets (a turn) or sends `state` idle.

| type | fields | consumed by (app.js) |
|---|---|---|
| `state` | `value` idle/thinking/processing/speaking, `turn` | `setState` — composer status, avatar cues, mic gating |
| `transcript` | `role`, `text` | `addMessage` — the history |
| `card` | `id`, `markdown`, `sources`, `title`, `saved?` (the path, on a reconnect replay); a run card (PRD 51) adds `run_id`, `workflow` (briefing/write-up/…), `status` running/done/partial/nothing/failed/awaiting/rejected/expired, `steps` (≤ 6 `{label, status pending/running/ok/skipped/error, detail?}`, the rest folded into "+n more"; a research card's are not folded: Plan, one row per sub-question, Read, Budget?, Wrote, Verified — PRD 54), `actions` (`{id, label, kind open/copy/approve/reject}`), `run_url` (`/runs/<run_id>`) | `addCard` — a written answer (PRD 43): the markdown through the wiki renderer, collapsed with "show all", Copy, Save to wiki with its folder picker; never spoken. A run card adds a `<workflow> · <status>` chip, the step list above the body and "Open run" (new tab); `approve`/`reject` render as Approve / Reject on an `awaiting` card (PRD 52): a click sends `resolve {run_id, decision}` and both stay disabled until the `card_update` or an `error`; an approval card (workflow forget/push) shows its proposal without Copy / Save; a Sources-view Forget toasts a link to its card; an empty body shows header, steps and actions only. The briefing sends one (empty body, final status); a write-up's goes out `running` before its body |
| `card_update` | `id` (the card's), `status` (incl. `rejected` and `expired`, PRD 52), `steps?`, `actions?`, `markdown?` + `title?` + `sources?` (a write-up's body, with its final status), `commit?` (an approved write's short sha, PRD 52) | `updateCard` — re-renders that card in place: chip, steps, buttons (+ the body when `markdown` comes); an update with `markdown` for an id the page never saw creates the card; a reconnect replays every card with its last status, a running or failed write-up (and an approval card) where it was issued — an awaiting card that is not the session's pending approval comes without Approve / Reject |
| `card_saved` | `id`, `path` | `markCardSaved` — the card's Save button reads "saved → path" |
| `speech` | `emotion`, `text`, `wav` (base64), `filler?`, `turn` | `enqueueSpeech` — audio queue, lip-sync, expression |
| `speech_end` | `turn` | ends the reply: `playNext`, then `playback_done` |
| `listening` | `value` hearing/idle | `setListening` — the mic indicator |
| `hands_free` | `value` | mirrors the switch, `renderComposer` |
| `verbosity` | `value` | `renderVerbosity` — the Settings control |
| `backdrop` | `key` | `applyBackdrop` + thumbnails |
| `characters_reloaded` | `avatar`, `name` | caption + "characters reloaded" on the status line |
| `setting` | `key`, `value` (or, for `omlx_admin_password`, `omlx_admin_password_set` + `omlx_admin_password_source` env/settings/null) | Settings rows: calendar horizon, recall threshold, location, the admin password state ("set (from .env)" / "set" / "not set") |
| `memory` | `profile`, `rendered`, `count` | `setMemory` — the Memory view |
| `memory_error` | `path`, `reason` | `showMemoryError` — the row that owns `path` (or the wiki slot for a wiki-side path) |
| `memory_saving` | `value` | composer status while a background save runs |
| `wiki` | `pages`, `lint`, `vault` | `setWiki` — the page list (+ the palette) |
| `wiki_page` | `path`, `text`, `links`, `etag`, `backlinks` | `setWikiPage` — the pane / editor |
| `wiki_open` | `path` | `openFromServer` — a "find / open / go back" turn: selects the page, switches to Wiki (Sources for a source page); the focus is already set, so no `set_focus` goes back |
| `wiki_conflict` | `path`, `text`, `etag` | `setWikiConflict` — the etag conflict dialog |
| `wiki_search` | `query`, `hits` | `setWikiSearch` |
| `sources` | `sources` | `setSources` — the inbox rows, the rail badge, clears `sourcesBusy` |
| `ingest_progress` | `filename`, `stage`, `i`, `n` | `setIntakeStage` — the intake popup rows |
| `context` | `used`, `budget`, `pct`, `trimmed` | `setContext` — after every completed turn and after a new topic: the `ctx N %` status suffix, the Chat rail dot amber ≥ 75 / red ≥ 90, "forgot n old turns" when trimmed |
| `thread` | `text` | `addDivider` — the "New topic" divider showing the summary carried over (also replayed on connect for an earlier new topic) |
| `services` | `llm`, `asr`, optional `remarkable` (only with `REMARKABLE_PUSH_URL`; each `{ok, label, detail, checked}`, the down `remarkable` adds its own start `hint`), `hint` | `setServices` — the PRD 41 banner (amber, naming what is offline, with a copy-the-command button) and the Settings › Services rows; pushed when a probe changes state, the last probe rides on `/api/config.services`. The red "backend is not running" banner is the client's own `ws.onclose` (any code but 4000 / 4003) |
| `error` | `scope` wiki/memory/sources/turn/settings/intake, `text`, `action?` | `showError` — routes on `scope`: wiki/intake → wiki status slot (intake also re-enables the ingest button), memory → memory error line, sources → wiki slot + `sourcesBusy` cleared, turn/settings → the composer status line |
