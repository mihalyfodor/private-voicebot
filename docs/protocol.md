# WebSocket protocol

One socket, `ws://localhost:8010/ws`, one page at a time (a second connection replaces the first). Text
frames are JSON objects; the browser's binary frames are 16 kHz Int16 mic audio (hands-free only). Boot
state (`avatar`, `backdrop`, `backdrops`, `hands_free`, `verbosity`, settings facts) comes from `GET
/api/config`; file intake goes over `POST /ingest` and only its progress comes back on the socket.

The server half is the `ACTIONS` dict in `chatbot.py` (one `_action_<name>` handler each); the client
half is the `ws.onmessage` ladder near the bottom of `static/app.js`. `tests/test_protocol_doc.py`
fails when either side names something this file does not.

## Client → server (`{"action": ...}`)

Unknown actions are ignored. An exception inside a handler comes back as `error` (scope from the
action name, plus `action`).

| action | fields | replies |
|---|---|---|
| `text_turn` | `text` | a full turn: `transcript` (user), `state` thinking/speaking, `speech`…, `speech_end`, `transcript` (assistant), `state` idle |
| `set_hands_free` | `value` (bool) | `hands_free`, then `listening` idle when switched off |
| `playback_done` | `turn` | none — releases the worker waiting for playback (stale turn ids ignored) |
| `playback_blocked` | `turn` | same as `playback_done` (autoplay refused; audio plays after the next gesture) |
| `set_backdrop` | `key` | `backdrop`, or `error` (settings) for an unknown key |
| `reload_characters` | — | `characters_reloaded` (the YAML card re-read), or `error` (settings) |
| `set_verbosity` | `value` short/normal/long | `verbosity`, or `error` (settings) |
| `set_setting` | `key` in `SETTABLE`, `value` | `setting`, or `error` (settings) for an unknown key / wrong type |
| `set_location` | `lat`, `lon`, `tz` | `setting` (`key: location`, the resolved value), or `error` (settings) |
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
| `forget_source` | `sha`, `confirm: true` | `wiki` + `sources`, or `error` (sources) without the confirm / for an unknown sha |
| `archive_memory` | `section`, `index` | `memory` + `wiki` + `sources`, or `memory_error` |
| `pull_wiki` | `path`, `section` | `memory` + `wiki` + `sources`, or `memory_error` |
| `shutdown` | — | none — the process gets SIGTERM; the page shows its shutdown screen when the socket closes |

## Server → client (`{"type": ...}`)

`state`, `speech` and `speech_end` (and the assistant `transcript`) carry `turn`; the client adopts the
turn id from `state` and answers `playback_done` with it. On connect the server replays the session's
`transcript`s, sends `hands_free` and either greets (a turn) or sends `state` idle.

| type | fields | consumed by (app.js) |
|---|---|---|
| `state` | `value` idle/thinking/processing/speaking, `turn` | `setState` — composer status, avatar cues, mic gating |
| `transcript` | `role`, `text` | `addMessage` — the history |
| `speech` | `emotion`, `text`, `wav` (base64), `filler?`, `turn` | `enqueueSpeech` — audio queue, lip-sync, expression |
| `speech_end` | `turn` | ends the reply: `playNext`, then `playback_done` |
| `listening` | `value` hearing/idle | `setListening` — the mic indicator |
| `hands_free` | `value` | mirrors the switch, `renderComposer` |
| `verbosity` | `value` | `renderVerbosity` — the Settings control |
| `backdrop` | `key` | `applyBackdrop` + thumbnails |
| `characters_reloaded` | `avatar`, `name` | caption + "characters reloaded" on the status line |
| `setting` | `key`, `value` | Settings rows: calendar horizon, recall threshold, location |
| `memory` | `profile`, `rendered`, `count` | `setMemory` — the Memory view |
| `memory_error` | `path`, `reason` | `showMemoryError` — the row that owns `path` (or the wiki slot for a wiki-side path) |
| `memory_saving` | `value` | composer status while a background save runs |
| `wiki` | `pages`, `lint`, `vault` | `setWiki` — the page list (+ the palette) |
| `wiki_page` | `path`, `text`, `links`, `etag`, `backlinks` | `setWikiPage` — the pane / editor |
| `wiki_conflict` | `path`, `text`, `etag` | `setWikiConflict` — the etag conflict dialog |
| `wiki_search` | `query`, `hits` | `setWikiSearch` |
| `sources` | `sources` | `setSources` — the inbox rows, the rail badge, clears `sourcesBusy` |
| `ingest_progress` | `filename`, `stage`, `i`, `n` | `setIntakeStage` — the intake popup rows |
| `error` | `scope` wiki/memory/sources/turn/settings/intake, `text`, `action?` | `showError` — routes on `scope`: wiki/intake → wiki status slot (intake also re-enables the ingest button), memory → memory error line, sources → wiki slot + `sourcesBusy` cleared, turn/settings → the composer status line |
