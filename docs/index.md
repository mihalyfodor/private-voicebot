# Repo Index

## How to run

```bash
# oMLX must be running on :8000; run scripts/fetch_assets.sh once
source .venv/bin/activate
python3 chatbot.py    # opens http://localhost:8011 automatically
```

## How to run — prod (docker)

```bash
# one-time: host ASR service (launchd), then move the state into ~/docker-data/voicebot — see docs/deploy.md
cp scripts/launchd/com.voicebot.asr.plist ~/Library/LaunchAgents/ && launchctl load ~/Library/LaunchAgents/com.voicebot.asr.plist
scripts/prod.sh migrate
# every release
scripts/prod.sh build && scripts/prod.sh up     # http://localhost:8010, read-only container, state under /data
scripts/prod.sh status | logs | down | backup | restore <tgz> <bundle> <dir>
# the whole stack (PRD 41): starts only what is missing — colima, oMLX, the model, the ASR service, the container
scripts/voicebot start          # `stop` frees the GPU (container down, ASR unloaded, model unloaded); `--all` also stops oMLX + colima
scripts/voicebot status         # the one-line-per-service table; `--dev` targets :8011 and skips the container
```

## How to run tests

```bash
source .venv/bin/activate
# everything: node --check static/app.js, node DOM fixture, `ruff check .`, unit suite (no oMLX needed)
# fresh clone: also run `git config core.hooksPath scripts/hooks` once (commit-msg hook)
scripts/check.sh
scripts/ingest_batch.py <folder> [--vault PATH] [--dry-run] [--resume] [--limit N] [--sleep S]   # PRD 48: file a folder of PDF/DOCX through the drop ingest; app stopped first
scripts/remarkable_sync.py pull [--device URL] [--app URL] [--state PATH]   # PRD 47: tablet plugged in (USB web interface on), app running; files changed `wiki-inbox` notebooks
scripts/trace_report.py events.jsonl [--since 24h] [--turn ID] [--errors]   # PRD 49: turns, ingests, errors from the event trace
scripts/smoke.py   # headless smoke: e2e + one live turn + battery section A + view screenshots → /tmp/smoke/<ts>/
python -m wiki reindex   # PRD 53: rebuild the vault's search index (<vault>/.index/search.sqlite) from scratch; vault = WIKI_PATH, else .env; refused (exit 2) while the app runs
# or just the unit suite — pytest.ini excludes the `live`, `e2e`, `docker` and `perf` markers by default
python3 -m pytest tests/
# the PRD 53 index perf gates (minutes: generates a 5k-page / 1k-source vault in tmp; prints the measured p95s with -s)
scripts/check.sh -m perf tests/perf -s
# re-record the PRD 53 search / index_text goldens (only on purpose: they pin today's scores)
GOLDEN_UPDATE=1 python3 -m pytest tests/wiki/test_index_db.py -k golden
# the PRD 40 docker scenarios (builds voicebot:test-<sha> once — minutes —, boots it on :8013 against a temp data dir under
# ~/.cache, needs oMLX on :8000 and starts the host ASR service on :8020 if nothing answers there; skips without a daemon)
python3 -m pytest -m docker tests/prod/test_docker.py -s
# the PRD 41 docker scenario alone (needs the harness to start the ASR service itself: nothing may answer on :8020 beforehand)
python3 -m pytest -m docker tests/prod/test_docker.py -k service_banner -s
# live tests (needs oMLX on :8000)
python3 -m pytest -m live tests/
# the PRD 32 live recall check alone (files the two real guides from ~/llm-wiki-dev into a temp vault, prints the Q/A pairs)
python3 -m pytest -m live tests/sources/test_source_recall.py -s
# the PRD 34 live grounding check (three real guides in a temp vault; prints each Q/A pair and its "yes" follow-up)
python3 -m pytest -m live tests/sources/test_grounded_answers.py -s
# the PRD 37 live routing calibration (28 questions, two real guides when ~/llm-wiki-dev has them; prints the confusion table; real Wikipedia calls)
python3 -m pytest -m live tests/llm/test_wikipedia_routing.py -s
# the PRD 38 live focus check in the browser (a real guide from ~/llm-wiki-dev filed as a source; "summarise this" with it open, "go back to that page" from Chat; prints both replies)
python3 -m pytest -m "e2e and live" tests/e2e/test_focus.py -s
# the PRD 33 live injection eval (six canned documents through a recall turn, a save and "did that document try to tell you something?"; prints the table)
python3 -m pytest -m live tests/sources/test_untrusted_text.py -s
# browser e2e (Playwright + Chromium; spawns chatbot.py on a free port with a temp vault) — screenshots in /tmp/e2e/
# prerequisites: `playwright install chromium`, scripts/fetch_assets.sh (pixi + Wanko model), Kokoro model files;
# tests/e2e/conftest.py fails early with "run scripts/fetch_assets.sh" when the assets are missing. E2E_KEEP=1 keeps the temp dir.
python3 -m pytest -m e2e tests/e2e
python3 -m pytest -m "e2e and live" tests/e2e   # one real LLM turn
# the browser-side fixture the client tests assert on (also run by pytest; needs node)
node tests/app_fixture.mjs
# the QA stress battery (PRD 42 / 43): a real chatbot.py on a throwaway port driven by Playwright over a temp vault seeded
# from read-only copies of ~/llm-wiki-dev/raw/inbox; sections A–F, results in /tmp/qa/results.json + transcript.md
python tests/qa/stress.py both /tmp/qa/vault --port 8014
```

## Protocol

[protocol.md](protocol.md) — the WebSocket contract: client actions with their fields and replies, server
message types with their fields and the app.js code that consumes them. Enforced by `tests/test_protocol_doc.py`.

## Logs

The server prints to stdout through `logging` (`[Memory]`, `[WS]`, `[Tool] name(keys)` lines, never a tool result; turn text
only as DEBUG in dev); started as `nohup python3 chatbot.py > /tmp/voicebot.log 2>&1 &` that is where to look; in prod
`scripts/prod.sh logs` (json-file, 10 MB × 5). PRD 49: `events.jsonl` (+ `verbose.jsonl` in dev only) in `TRACE_DIR`
(repo-local `logs/` in dev, gitignored; `/data/logs` in prod), rotated 10 MB × 5; `TRACE_PROFILE` = dev (checkout default) | prod
(forced by compose; dev refuses to start against `~/docker-data` or in a container). Read it with
`scripts/trace_report.py events.jsonl [--since 24h] [--turn ID] [--ingest ID] [--errors]` (the /trace skill).
`session.log` is retired (an old one is still moved by `prod.sh migrate`). The e2e run writes screenshots to
`/tmp/e2e/<test>-<name>.png` and the spawned server's output to `server.log` in its temp dir (kept with `E2E_KEEP=1`).

## Folder tree

```
voicebot/
├── chatbot.py        # main entry point, FastAPI + WS server, browser mic capture (hands-free), typed turns (`text_turn`), sentence-streamed TTS, backdrop switching, character-card reload, `_respond` routing through `registry.match` with the five trigger-tool bodies bound by `@registry.handler` (PRD 50), `set_focus` + deterministic find/open/go-back page commands pushing `wiki_open` (PRD 38), `context` after every turn + the 90 % spoken nudge, the `new_topic` action / "new topic" / `/new` turn with its `thread` divider, reconnect replay from the stripped conversation (PRD 39), the "write it up" / `/write` write-up turn (one spoken line + `card`), "save that as a note" + the `save_card` action filing the last card under `notes/` (PRD 43), the "what's new?" / `/brief` briefing turn ("Nothing new since <when>." without the model, else one spoken recap; `briefing_at` in settings.json) (PRD 46); PRD 51 runs: the briefing (gather / skip / phrase steps, one empty-body run card with its final status) and the write-up (card `running` first, then `card_update` done + markdown / failed) are runs (a failure ends the run through one `_run_failed`), `card_msg` carries the optional run fields (`run_id`, `workflow`, `status`, `steps` ≤ 6, `actions`, `run_url`) and the replay sends a card's last status, a host script's first `/ingest` with an unseen `run_id` writes `run start`, `GET /runs/{run_id}` (next to `/`; `RUN_ID` + the `/ingest` origin check; 404 "No run <id> (it may have aged out of the log).") renders the run page
├── llm.py             # the chat agent (model calls go through core/llm.py), system prompt per avatar, tool dispatch, ask_events/ask_stream/ask, per-turn source recall (`recall_for`, PRD 32) with the precision rule + look-up-first prompt block + one-turn "dig deeper" offer (PRD 34), the routing rules + "Want me to check Wikipedia?" offer whose "yes" runs get_wikipedia at `long` (PRD 37), memory; every recall / deeper block is wrapped as `<document>` and the sources a session read go to the memory save for attribution (PRD 33); the focus state + `focus_block` (open page head + best-matching section/chunk in place of recall, PRD 38); the context meter (`context_usage`, `registry.tokens()`, `context_notice`), recall/focus blocks stripped from the stored user message once the turn completes, `new_thread` (fold memory → ≤ 150-token summary as `<earlier>` → fixed ack) (PRD 39); length by origin (a typed grounded content question runs `long` with the offer clause swapped out), `write_up` (one card completion over the last answer's context + a Sources line, kept as a marked assistant turn) (PRD 43; PRD 51: `next_card_id` never reuses an id, `issue_card` keeps a running / failed write-up card for the replay, `write_up` fills it in place); `recall_band` on spoken turns (high = grounded, medium = plain reply + server-appended "Want me to check your <title>?" whose "yes" answers the earlier question grounded, low = no recall; band in the `recall` event) (PRD 45); `briefing` (one short recap completion over the `<document source="briefing">` digest, its card kept in `_last_context` so "write it up" makes none) (PRD 46)
├── core/               # PRD 50 layer 1: imports only stdlib/third-party and other core modules
│   ├── config.py        # the one place env paths and oMLX settings are read (ROOT, WIKI_PATH, MEMORY_DIR, SETTINGS_PATH, TRACE_DIR, OMLX_*)
│   ├── llm.py           # PRD 50 step 4: the single model-call chokepoint — `client()` (lazy oMLX client in `_client`, what tests patch), `stream()` (streamed, one `llm_call` span), `completion()` (raw response, one span), `complete(system, user, max_tokens, purpose=)` (text; spanned only with a purpose); OMLX_* from config
│   ├── settings.py      # settings.json: `load_settings` / `save_setting` / `save_settings` (atomic tmp + os.replace, version stamp)
│   ├── trace.py         # PRD 49 event trace (moved from tracing.py): `FIELDS` (one table: allow-list, per-field vocabularies, 200-char cap), `event` / `verbose` (dev only) via one writer, `decide` (profile + refusal), `span` / `turn` / `ingest` / `run` contexts (turn_id, ingest_id, run_id, span ids), `error`, `configure`, `llm_span` / `usage_fields` / `ms_since` call-site helpers, 10 MB × 5 rotation, the `[Module]` stdout loggers; PRD 51: `run_event` (the `run` start/end point event), `current_run`, `step` spans whose body may set `status` (skipped), the run/step fields (`workflow`, `phase`, `reason` = `runs.REASONS` codes, `name` step names, `count`, `detail` ≤ 200 or dropped, `commit` a sha, `skipped`, `part`, `memory` counts)
│   ├── untrusted.py     # PRD 33: `untrusted(text, source)` → `<document source="…">` (nested tags neutralised), `documents_in`, the standing DOCUMENT_RULE every prompt carries
│   └── vault_lock.py    # the one RLock over memory.json + the vault, shared by memory/wiki/sources
├── characters.py     # character card loader + persona builder
├── characters/       # editable YAML character cards
├── session.py        # Session (per-connection state) + TurnController (single worker; `submit(..., turn_id=)` opens the turn's trace context)
├── tracing.py        # shim → core/trace.py (PRD 50)
├── runs.py           # PRD 51 runs (stdlib + core; wiki lazily): `REASONS` (one sentence per reason code), `WORKFLOWS` (labels keyed by `trace.FIELDS["workflow"]`), `Run` (an app workflow's `run` start/end + the card's steps, `card()` = the card's run fields, ≤ 6 steps with "+n more"), `read` (events.jsonl + rotated .1…5; also scripts/trace_report.py's reader) / `load(run_id)`, `build(events)` → the run dict (status derived for host-script runs; Did pages with sha + `git show --stat` summary + origin source/Wanko/you, sources, memory counts; Did not; Errors), `render(run)` (f-strings, `html.escape` on every value, never page text), `missing(run_id)` (the 404 line)
├── vad.py            # Silero VAD endpointer (hands-free); Recorder is an unused PTT leftover
├── asr.py            # mlx-whisper transcription + junk filter, optional initial_prompt name biasing; `last_error` after a failed remote call (PRD 41)
├── services.py       # PRD 41: oMLX / ASR health probe (`probe`, `refresh`, 30 s daemon + `recheck` after a failed call), `START_HINT`, the `services` payload; addendum: `admin_password()` (env, else settings.json; `python -m services --admin-password` for the script), `model_op` load/unload via the admin API
├── avatars.py          # avatar profile (Wanko only since PRD 23: name, voice, speed, persona), current/set_current, settings.json persistence
├── backdrops.py        # backdrop catalogue (key, name, file, credit, url, license), validate/listing
├── fillers.py           # short spoken fillers played while a tool call (or a `dig_deeper` turn) is in flight, no immediate repeats
├── splitter.py         # LLM delta stream → (emotion, sentence) pairs, strips the leading [tag]
├── vault_lock.py       # shim → core/vault_lock.py (PRD 50)
├── untrusted.py        # shim → core/untrusted.py (PRD 50)
├── memory.py           # structured profile memory: load/render, LLM-proposed ops (`save`/`reflect`/`propose_ops` call `core.llm.complete` unspanned; no client/model parameters) (attributed to `conversation` or a document title when the session read documents — `source` on entries, `sources` map for scalar keys, `forget_source`; PRD 33), seen/stale decay, reflection, the identity section (aliases, is_user, user_note, ASR name prompt) and the editor entry points (snapshot/edit/edit_many/reset)
├── index_db.py         # PRD 53 search index: one SQLite file per vault (`<vault>/.index/search.sqlite`, WAL; tables pages/chunks/terms/postings/proper/meta), one transaction per document change, the BM25 scorer over postings (`bm25_term` is the one formula, also used by wiki.py's in-memory `best_section` scorer; subset statistics for named sources / filters), `dump` (canonical rows) + `integrity`; storage only — wiki.py tokenises and decides what is a page
├── wiki.py             # wiki layer: ~/llm-wiki vault, page ops + git commits, BM25 search (PRD 53: `search` / `search_chunks` / `index_lines` / `index_text` / `backlinks` read the index_db tables, optional `filters`; write hooks `index_page` from `_write` / `unindex_page` / `index_source` / `unindex_source`; `startup_check` at app + batch start, `reindex` = `python -m wiki reindex`; `lint` checks the `pages` table both ways, not the 200-line index.md), raw chunking + `raw/.index/` cache + chunk search (with matched terms + `proper_terms`) / read_source / recall threshold (PRD 32), ingest (never a people page about the user; the prompt names the pages most related to the transcript, PRD 44), lint (also dead `source: raw/…` refs, uncited raw files, duplicate names, PRD 44), archive/pull bridge, editor writes (save with etag / create / rename / delete under vault_lock.LOCK), backlinks, tags, page sections + best section/chunk for a question (PRD 38), `save_card` — a written answer filed as a note with `source: written answer` + `sources:` in one commit (PRD 43), `changes_since` + `briefing_card` — pages from git subjects (lint / index / log / raw / sources skipped, chat batches grouped), new sources, memory ops; 8 per section (PRD 46); each digest page carries its newest commit's short sha and the digest its `skipped` housekeeping-commit count, `head_sha` (PRD 51)
├── sources.py          # raw ingestion (every completion is `core.llm.complete(..., purpose="ingest")`; no client/model parameters): raw/ storage + manifest, pdf/eml/msg/image extraction, docx/pptx/xlsx/vtt/ics extractors + head/middle/tail sampling (`raw_text` is the unsampled text the chunk index reads), transcript → meetings page, deterministic .ics stubs (PRD 31), raw renamed after the model title + eager chunk cache (PRD 32), summary (one retry when it comes back untitled) + page ops (the prompt lists related pages the index cut, PRD 44; published works — epub/pdf/docx — never open a people page for an author, only for someone in the profile), POST /ingest core; `strict=` for the batch (PRD 48: a failed extraction/summary raises, the stored original is removed; pdf now has the 50 MB text cap); filing indexes the new chunks and forget drops them (PRD 53); inbox: list_sources/reingest/forget (PRD 25; forget also drops the memory entries the source fed, PRD 33; and a notebook's page PNGs); `file_bytes` takes a notebook's own `digest`, its page PNGs (stored beside the raw file, listed in the manifest) and `extra` manifest fields (PRD 47); `ingest_end.commit` = the short sha after the filing commit (PRD 51)
├── batch_import.py     # PRD 48 batch runner over the drop path: folder walk, start guards (dirty vault, app running), needs-ocr check (`pdf_chars`), strict `ingest_upload` per file, batch.log + --resume, --limit/--sleep, progress + summary lines; `ingest_batch` is the core of scripts/ingest_batch.py; runs the PRD 53 index check before filing; `running_app` (the port probe) also guards `python -m wiki reindex`; a real run is one PRD 51 run (`batch-<ts>`, `run` start/end, a `step` per file with its skip reason)
├── notebooks.py        # PRD 47 reMarkable notebooks: `check_request` (the `kind=notebook` gate of POST /ingest: PNG pages ≤ 50, `replaces` same device, `ocr` JSON list), `ingest_notebook` (sha of the pages before any model call, one transcription completion per page outside the lock, `[page N: unreadable]` under 20 chars, optional Apple Vision OCR block, transcript `.md` through `sources.file_bytes`, PNGs under `raw/remarkable/<date>-<slug>/`, `replaces` forgets the earlier sha); unreadable pages are one `skip` step per notebook (PRD 51)
├── ruff.toml          # lint config (E4/E7/E9/F, compact-style ignores); no formatter
├── .claude/agents/reviewer.md  # read-only one-pass branch review before the Implementation gate (BLOCKER / MAJOR / backlog)
├── .claude/skills/smoke/SKILL.md  # /smoke: run scripts/smoke.py, read every PNG, quote RESULT.txt (ends with trace_report --errors)
├── .claude/skills/trace/SKILL.md  # /trace: where the event logs live, report first, raw lines only for flagged turns, never verbose.jsonl outside dev
├── scripts/
│   ├── check.sh         # node --check app.js, node fixture, `ruff check .`, unit suite; extra args go to pytest (`-m e2e tests/e2e`)
│   ├── hooks/commit-msg # rejects AI-attribution trailers in commit messages; enabled via `git config core.hooksPath scripts/hooks`
│   ├── ingest_batch.py  # PRD 48: file an export folder of PDF/DOCX — a thin argparse CLI over `batch_import.ingest_batch` (the drop path), idempotent via the manifest, `--resume`/`--dry-run`/`--limit`/`--sleep`, `batch.log`; exit 2 on a dirty vault or a running app
│   ├── fetch_assets.sh  # downloads pixi.js, pixi-live2d-display, Cubism Core, the Wanko Live2D model (all 12 motions) and backdrop images (gitignored), then generates Wanko.app.model3.json
│   ├── wanko_groups.py  # PRD 27 motion-group table + generator for static/models/wanko/Wanko.app.model3.json
│   ├── smoke.py         # headless smoke pass: stack, e2e, one live turn, one battery section, view screenshots → /tmp/smoke/<ts>/RESULT.txt, then `trace_report.py --errors` over every events.jsonl the run wrote
│   ├── trace_report.py  # PRD 49: imports the repo's `core.trace` + `runs` (`runs.read`, PRD 51), so it runs with the venv (Python 3.10+), not a bare system python: events.jsonl (+ rotated files) → turn table, ingest table, errors; `--since`, `--turn ID|pid:ID`, `--ingest ID`, `--errors` (exit 1 when any)
│   ├── screenshot.py    # dev helper: headless screenshot of the running app (`--view memory`, `--drag`, `--drop-file`)
│   ├── remarkable_sync.py  # PRD 47: `pull` (stdlib, host venv) — tablet USB web interface (`--device`, default http://10.11.99.1) → changed `wiki-inbox` notebooks → `pdftoppm` PNGs (+ optional `scripts/vision_ocr` / `$REMARKABLE_OCR_BIN`, `REMARKABLE_OCR=0` off) → one `kind=notebook` POST /ingest (`--app`, default :8010); state `--state` ~/.cache/voicebot/remarkable.json, written per filed notebook (atomic temp + `os.replace`); one function per step (`list_changed`, `download`, `rasterise`, `post_notebook`, `record`), every printed line in `LINES`; one run id per pull, `rm-<yyyymmdd-hhmmss>` (PRD 51)
│   ├── email_classification.py  # Gmail triage dashboard (category/urgency via local LLM); moved from workflows/ (PRD 50); data files classification_* beside it are gitignored
│   ├── classification_rules.example.py
│   ├── asr_server.py    # PRD 40: host ASR service — POST /transcribe (wav) → {text, segments}, GET /health; 127.0.0.1:8020
│   ├── prod.sh          # PRD 40: build | up | down | logs | status | migrate | backup | restore (env DATA, TAG, PROD_PORT, PROJECT); start | stop delegate to voicebot
│   ├── voicebot         # PRD 41: start [--dev] | stop [--all] [--dev] | status [--dev] — checklist start (only what is down), stop = compose down + ASR unload + model unload via the oMLX admin API (password from .env, else the one saved in Settings)
│   └── launchd/com.voicebot.asr.plist, com.voicebot.backup.plist   # LaunchAgents: asr_server.py, daily prod.sh backup
├── static/
│   ├── index.html      # browser UI: icon rail, companion column (avatar/history/composer), workspace views (Chat / Memory / Wiki editor / Sources / Settings), intake popup, command palette
│   ├── mic-worklet.js # AudioWorklet: mic → 16 kHz Int16 frames
│   ├── app.js           # PROFILES/CONFIG/MOTIONS, Live2D model load, audio queue, lip-sync, expressions, motion groups + click-to-pet + state cues (PRD 27), rail (VIEWS table), setView (companion re-parenting), renderComposer (mic/mute/status), memory view, wiki editor (marked render + safe import, textarea ⌘S/Esc, etag conflict dialog, backlinks, search, Tags group, Today/New/Rename/Delete — PRD 28), settings view (incl. the calendar horizon and recall threshold fields), drop-paste intake, ⌘K command palette (PALETTE_SOURCES registry, ranking — PRD 30), keyboard, websocket, focus context (`set_focus` on view/page change, `wiki_open`, "ask about <title>…" placeholder — PRD 38), context meter (`ctx N %` status suffix, Chat rail dot amber/red, "New topic" palette entry + divider — PRD 39)
│   ├── style.css        # shell layout (--rail-w), rail, #companion chat-grid vs side slice, history/composer, views, settings rows, palette overlay, narrow-screen bottom tab bar
│   ├── vendor/          # gitignored, fetched — except marked.umd.js + LICENSE-marked.md (MIT, committed; PRD 28)
│   ├── models/           # gitignored, fetched (wanko/)
│   └── backdrops/        # gitignored, fetched
├── tools/              # every module declares `HOSTS` (PRD 33 read-only policy: no free URL fetch, get_news_detail only takes a URL from the last get_news); `__init__.py` re-exports from registry
│   ├── registry.py     # the ordered `REGISTRY` of `Tool` records (PRD 50): 11 schema tools, then the trigger tools in routing order (new_topic, save_note, briefing, write_up, open_page); `match(text)`, import-time `_validate` (unique names, one mode, needs_llm triggers last), `Ctx` + `bind`/`current` (the turn's context, used by file_source's panel refresh), `handler(name)` (the server binds trigger bodies; an unbound one raises); `schemas()`, `tokens()`, `run_tool`, and `READ_TOOLS`/`ALLOWED_HOSTS` derived from the records
│   ├── commands.py     # the trigger regexes (OPEN_PAGE/BACK_TO_PAGE/OPEN_PAGE_ANY, NEW_TOPIC, BRIEF, SAVE_NOTE, WANTS_WRITEUP) and their predicates (`page_command`, `new_topic_command`, `briefing_command`, `save_note_command`, `wants_writeup`) (PRD 50)
│   ├── time.py         # current time tool
│   ├── weather.py       # open-meteo weather tool + one Nominatim reverse geocode per new position (PRD 35)
│   ├── news.py          # BBC news headlines + article detail
│   ├── gmail.py         # Gmail read/search via OAuth
│   ├── wiki.py          # search_wiki (pages + document passages) / read_page / read_source — the model's way into the vault
│   ├── sources.py       # file_source(email|news|note) — voice ingestion of the last email/headline/a note
│   └── wikipedia.py     # get_wikipedia (search + summary) / read_wikipedia (one section): en.wikipedia.org only, 5 s timeouts, 1 req/s, per-title cache, error strings never exceptions
├── eval/             # memory eval harness: harness.py, scripts/*.yaml (incl. aliases.yaml), fixtures/, results/
├── tests/                # one folder per module; shared fixtures live in conftest.py (see the DX note in process.md)
│   ├── conftest.py              # sandboxes (wiki/settings/memory paths — WIKI_PATH also set to a missing tmp path before the app import, PRD 53 —, the search index closed per test, chatbot session, `core.llm._client` restored), the one fake model
│   │                            # client `FakeClient` (`StreamClient`/`RawStreamClient` legacy spellings) + `fake_model` (patched at core.llm._client), vault helpers (vault, seeded_vault, commits, read, write_page), the generated FAQ
│   │                            # sources (`fallout`, `both`, faq_text, file_faq, summary_json), turn helpers
│   │                            # (fake_tts, run_turn, typed, spoken), Panel, send_setting, and the node client fixture run once per session
│   ├── app_fixture.mjs          # boots static/app.js in node on a DOM/WS/audio/PIXI shim and prints one JSON of results
│   │                            # (companion re-parent, typed turn, mute, thumbs, avatar cues, wiki editor, palette, sources chips/intake
│   │                            # titles/horizon field, intake batch, context meter, error scopes, written cards); its header documents the hooks
│   ├── tools/test_registry.py   # PRD 50: golden schema fixture byte-compare, tokens() == 1089, READ_TOOLS/ALLOWED_HOSTS derived, record invariants (incl. needs_llm triggers after gate-free ones); trigger phrase table + `match` agrees with the predicates, handler binding (unbound raises), file_source refreshes the panel via the bound Ctx without importing the server
│   ├── fixtures/tool_schemas.json  # PRD 50: golden `json.dumps(registry.schemas())`, byte-compared by tests/tools/test_registry.py
│   ├── fixtures/search_golden.json  # PRD 53: `search` (k=5) / `search_chunks` (k=3) for 20 FAQ queries on the `both` vault, recorded on the pre-index scorer
│   ├── fixtures/index_text_golden.txt  # PRD 53: `wiki.index_text()` on the `both` vault, byte for byte
│   ├── perf/                    # PRD 53 scenario 10, marker `perf` (deselected by default): make_vault.py = the shared seeded synthetic-vault generator (`make_vault(path, pages=5000, sources_n=1000, seed=0)` + `queries(n)`; PRDs 54/55 reuse it), test_index_perf.py = p95 search_chunks < 50 ms, search < 20 ms, one-source update < 200 ms (+ the live battery comparison, skipped)
│   ├── fixtures/routing_questions.json  # PRD 37 calibration set: 8 general / 8 filed / 8 precise / 4 personal questions with the expected route
│   ├── test_index.py            # every tests/**/test_*.py and docs/NN-*.md is listed here with its current status
│   ├── test_protocol_doc.py     # docs/protocol.md names every ACTIONS key, every `"type"` chatbot.py sends and every type app.js handles — and nothing else
│   ├── core/test_llm.py         # PRD 50 scenario 7: `core.llm.complete` (temperature 0, max_tokens, default model; spanned only with a purpose); an ingest, a memory save and a typed turn draw from one fake at core.llm._client, llm_call purposes ingest/ingest/reply
│   ├── test_layers.py           # PRD 50: layer ratchet — app code imports only downward (`ALLOWED` of (importer, imported module) pairs may only shrink), no imports of the root shims, stale allow-list entries fail
│   ├── llm/
│   │   ├── test_llm_unit.py         # llm.py with a mocked OpenAI client: ask/ask_events, PRD 13 streamed first call, tool-call detection mid-stream
│   │   ├── test_context.py          # PRD 11: context budget, history trimming
│   │   ├── test_verbosity.py        # PRD 11: verbosity levels, max_tokens derivation, set_verbosity action
│   │   ├── test_fillers.py          # tool-intent fillers: pick() no immediate repeats
│   │   ├── test_tools.py            # tool schemas/behaviour, needs live oMLX for some cases
│   │   ├── test_llm_robustness.py   # LLM tool-trigger robustness (single phrases + variants), needs live oMLX
│   │   ├── test_written_answers.py  # PRD 43: length by origin (typed grounded content → long, no offer; spoken → short + "dig deeper, or write it up?"), the write-up turn + card bookkeeping, the client card (fixture), save_card → notes/ with provenance in one commit, "save that as a note", new topic clears the card, reconnect replays it, protocol rows; PRD 46: "write it up" after a briefing is the digest card with no completion
│   │   ├── test_ask_before_lookup.py # PRD 45: the recall band (live band table over a dev-vault copy + fixture rules), the medium offer, yes / no / never twice, typed unchanged, the band in the `recall` event
│   │   ├── test_recall_precision_2.py # PRD 42: title words never score a chunk, neighbour dedupe, the held-reply lookup fallback, page recall, short proper questions, proper bigrams, summarise-this focus, routing wording, trailing "please", single-dated raw names
│   │   └── test_wikipedia_routing.py  # PRD 37: canned Wikipedia tool tests, routing rules in the prompt, the Wikipedia offer/yes path, the 28-question calibration (mocked + live confusion table), fixed-host policy
│   ├── memory/
│   │   ├── test_memory.py           # memory v2: ops, caps, render budget, reflection, identity aliases
│   │   ├── test_memory_decay.py     # seen/stale decay, CONFIRM, memory_ops batches, harness edit/clock turns
│   │   ├── test_memory_editor.py    # memory panel actions: get_memory/memory_op/memory_reset, locking, fuzz
│   │   └── test_memory_eval.py      # PRD 14: eval harness scripts, Sandbox isolation, scoring
│   ├── wiki/
│   │   ├── test_wiki.py             # wiki layer: vault, page ops, search, ingest, lint (PRD 44: dead source refs, uncited raw files, duplicate names; PRD 53: a 250-page vault is clean, the search index checked both ways), related_context, archive/pull, actions; PRD 46 briefing digest (git cutoff, self-noise incl. editor saves + saved cards, chat pages, sources, memory ops, 8-item cap, <when>)
│   │   ├── test_wiki_editor.py      # PRD 28: save/etag/conflict, lock wait, create, backlinks, tags, rename, delete, client fixture assertions
│   │   ├── test_index_db.py         # PRD 53: golden scores + golden index_text (no page reads), incremental ingest row diff, rename/delete/forget leave no rows, reindex == incremental, deleted index rebuilt, startup check repair, filters, changed sha re-indexes one source, killed update rolls back, a skipped source is retried by the next startup check
│   │   └── test_focus_context.py    # PRD 38: `set_focus` → the page block on typed and spoken turns, head + matching section cap, Chat drops focus / "go back", find/open + no-match notes, a focused source feeds the offer, injected page text wrapped, client fixture assertions
│   ├── sources/
│   │   ├── test_sources.py          # raw ingestion: raw/ + manifest, pdf/eml/image extraction, restricted ops, file_source, POST /ingest, PRD 44 dead ref after forget + related pages in the ops prompt, PRD 48 batch import, PRD 47 `ingest_notebook` (per-page transcription, OCR block, sha guard, unreadable pages, replaces, sampling, 51 pages → 413) + `kind=notebook` on /ingest; PRD 51 scenario 5: two notebooks under `run_id=rm-1` → `ingest_end.commit`, one `run start`, the run page with both shas and "page 2: unreadable"
│   │   ├── test_sources_inbox.py    # PRD 25: list_sources newest-first, reingest, forget (only matching notes), get_sources/reingest_source/forget_source actions
│   │   ├── test_office_formats.py   # PRD 31: docx/pptx/xlsx/vtt/ics extraction (fixtures generated in-test), Teams detection, meeting pages + attendee notes, calendar stubs + horizon setting, unreadable files, 10 MB text sampling, model-titled pastes, client fixture assertions, epub, book authors are not people pages + the summary retry
│   │   ├── test_source_recall.py    # PRD 32: raw chunking + per-sha cache, chunk search, search_wiki snippets, read_source, recall_for + per-turn injection, calibration (two generated FAQs), threshold setting, raw renamed after the title; `live` scenario against the real guides
│   │   ├── test_grounded_answers.py # PRD 34: recall precision (matched terms + proper nouns), look-up-first prompt block, one-turn "dig deeper" offer (neighbour chunks, long for that completion only), Settings note, the lookup fillers
│   │   ├── test_untrusted_text.py   # PRD 33: `untrusted()` + every read tool / recall block wraps, memory-op attribution and `source` stamps, editor snapshot + forget, the six-document injection eval (mocked + `live`), the read-only tool policy
│   │   └── test_intake_progress.py  # PRD 36: intake batch in the client fixture — one POST per file, row states, progress push, Cancel after the current file, error rows
│   ├── session/
│   │   ├── test_turns.py            # PRD 12: Session/TurnController, turn ids, origin check, periodic memory hook, dead sockets, static caching; PRD 46 briefing turn (phrasings, one completion + offer, nothing new offline, cutoff only forward, a stand-ins-only session is not ingested); PRD 50 scenario 5 (offline: only new topic / save note / briefing run)
│   │   ├── test_context_meter.py    # PRD 39: `context` after a turn, registry.tokens(), trim count, 90 % crossing once per crossing, recall/focus stripping, `new_thread` fold-first + summary + fallback, the new-topic triggers, reconnect replay, client fixture assertions
│   │   ├── test_hands_free.py       # WS hands-free actions, binary audio frames, text_turn while listening
│   │   ├── test_greetings.py        # canned greetings: greet() turn shape, turn ids, no LLM call
│   │   ├── test_chatbot_speech.py   # speak_stream: mocked Kokoro + WS, wav framing
│   │   ├── test_text_input.py       # PRD 24: text_turn == spoken turn, `typed:` log line, no `ptt`, client fixture assertions
│   │   ├── test_service_health.py   # PRD 41: probe() up/down/model-not-loaded (mocked HTTP), one `services` push per change, /api/config.services, turns refused with the hint while the LLM is down + the re-probe hooks, client banner / copy button / Settings rows (fixture); addendum: admin password set/clear round trip (value never echoed), env precedence, model_load/unload against a mocked admin API, the password row + model buttons (fixture)
│   │   ├── test_error_scopes.py     # scoped `error` messages: one server refusal per scope through the Panel fixture (which asserts a scope on every error), the client routes on `scope` not on the open view
│   │   └── test_settings_action.py  # set_setting whitelist, /api/config payload, no panel markup in index.html, OPEN_BROWSER flag, PRD 35 location + reverse geocode (one `live` Nominatim test)
│   ├── ui/
│   │   ├── test_avatars.py          # avatars.current()/AVATAR env handling, set_current()/settings.json persistence, retired keys, backdrops.validate()/listing() + set_backdrop
│   │   ├── test_avatar_cues.py      # PRD 27: generated motion groups, click-to-pet, state cues, new emotion tags
│   │   ├── test_characters.py       # character card loader, persona builder and caps
│   │   ├── test_palette.py          # PRD 30: ⌘K palette — open/close, ranking, keyboard, search fallback, payloads, click (client fixture assertions)
│   │   └── test_run_cards.py        # PRD 51 scenario 8: the run card's chip, steps and "Open run" (target _blank), `card_update` in the same element, a write-up's body on update, a plain card keeps the PRD 43 shape (client fixture assertions)
│   ├── audio/
│   │   ├── test_asr.py              # junk filtering + transcribe() with mlx-whisper mocked, initial_prompt biasing, PRD 40 host ASR service + remote backend
│   │   ├── test_vad.py              # Endpointer state machine (fake prob_fn) + Recorder
│   │   └── test_splitter.py         # SentenceSplitter / split_stream / strip_tag, every emotion tag round-trips
│   ├── tracing/                 # PRD 49: the event trace
│   │   ├── test_trace.py            # scenarios 1–3, 5, 6, 8: a turn's ids + no text in events, tool_call size not result, llm_call usage/ttft/span, a turn raising mid-stream, rotation + unwritable dir, the field allow-list, dev verbose file; PRD 51: run/step events carry no text in prod
│   │   ├── test_trace_profile.py    # scenario 7: prod never creates verbose.jsonl, service start first, dev against ~/docker-data refused (chatbot.py and ingest_batch.py), host scripts log next to their data folder, compose forces prod + caps the container log
│   │   ├── test_trace_report.py     # scenario 9: scripts/trace_report.py `--turn` (ordered by start), `--errors` exit codes, the summary tables over rotated files
│   │   ├── test_runs.py             # PRD 51 scenarios 1, 2, 6, 7: a briefing run (start/end, gather/skip/phrase steps, the run card), nothing new without a model step, chit-chat has no run, "+n more" folding, reason codes allowed, a failing write-up (`card` running → `card_update` failed, page shows exc_type, no sentinel), a working write-up's markdown on the update + replay, a batch import run with a needs-ocr row, the dev refusal against a docker-data-looking path
│   │   └── test_run_page.py         # PRD 51 scenarios 3–4: GET /runs/<id> (pages with sha + --stat line, "2 housekeeping commits", the timeline, footer, every value escaped, no page body), 404s (unknown, malformed, too long), the /ingest origin check
│   ├── prod/                    # PRD 40: Docker prod
│   │   ├── test_data_paths.py       # every state path follows its env knob; a turn + memory save + wiki op write only under the sandbox
│   │   ├── test_remarkable_sync.py  # PRD 47 scenarios 1–4: scripts/remarkable_sync.py pull against `fake_remarkable` + `fake_ingest` (conftest) — folder/notebook filter, state, nothing new, replaces, device/app down, one failure, OCR helper stub paths; skipped without pdftoppm
│   │   ├── test_prod_script.py      # prod.sh migrate moves state into DATA (refuses twice), backup guards/rotation, plist lint, bash -n, docker compose config
│   │   ├── test_voicebot_script.py  # PRD 41 scenario 5: scripts/voicebot against stubbed colima/omlx/launchctl/docker/curl — start touches only what is down, stop order, --all, status --dev, the Settings password used for the admin login when the env is unset
│   │   └── test_docker.py           # scenarios 6–8 for real: build + boot (healthcheck, uid, read-only), typed/spoken turns via host oMLX/ASR, backup/restore round trip (marker `docker`); PRD 41 scenario 6: the amber banner data while the host ASR service is stopped
│   └── e2e/                     # PRD 29: Playwright browser tests (conftest checks the fetched assets, spawns the server; test_app.py, test_sources.py incl. the PRD 31 .docx drop and the PRD 36 two-file batch, test_palette.py, test_focus.py = the PRD 38 live turn; test_app.py also has the PRD 51 live run-page check → /tmp/e2e/run-page.png), marker `e2e`
├── requirements.txt
├── requirements-docker.txt   # PRD 40: requirements minus mlx-whisper/playwright/pytest (torch from the CPU index)
├── Dockerfile / docker-compose.yml / .dockerignore   # PRD 40: python:3.12-slim image with the assets baked in; read-only, non-root, loopback-only service
└── docs/
    ├── index.md               # this file
    ├── process.md              # development process
    ├── roadmap.md              # ordered next steps (PRDs 44–46, backlog, dropped ideas)
    ├── protocol.md             # WS protocol reference: every client action and server message type (tests/test_protocol_doc.py keeps it in sync)
    ├── reviews/                # dated read-only review notes (architecture, developer experience, 2026-10-16 design questions for PRDs 47b/50–55)
    ├── prototypes/             # Phase 1 clickables and sketches: sidebar/, obsidian-host/, wanko-motions.md
    ├── 01-voicebot.md           # PRD: initial voicebot (done)
    ├── 02-llm-robustness-tests.md  # PRD: LLM tool-trigger robustness tests (done)
    ├── 03-email-classification.md  # PRD: Gmail triage workflow (done)
    ├── 04-avatar.md             # PRD: Live2D avatar + oMLX backend (done)
    ├── 05-fillers.md            # PRD: tool-intent spoken fillers (done)
    ├── 06-avatar-switch.md      # PRD: switchable avatars, Wanko default (done, superseded by 23)
    ├── 07-avatar-menu.md        # PRD: Natori + in-app menu with live switching (done, superseded by 23)
    ├── 08-backdrops.md          # PRD: selectable backdrops (done)
    ├── 09-continuous-mode.md    # PRD: hands-free conversation mode (done)
    ├── 10-personalities.md      # PRD: character cards (done)
    ├── 11-context-and-length.md # PRD: context budget, trimming, verbosity (done)
    ├── 12-turn-controller.md    # PRD: session, turn controller, protocol hardening (done)
    ├── 13-streamed-first-call.md # PRD: stream the first LLM call (done)
    ├── 14-memory-eval.md        # PRD: text-only memory evaluation harness (done)
    ├── 15-memory-v2.md          # PRD: structured profile memory (done)
    ├── 16-memory-editor.md      # PRD: memory viewer/editor panel (done)
    ├── 17-long-term-memory.md   # PRD: research — decay, archive, wiki layer (exploration)
    ├── 18-memory-decay.md        # PRD: seen/stale decay, CONFIRM op (done)
    ├── 19-wiki-layer.md          # PRD: ~/llm-wiki vault, page ops, search (done)
    ├── 20-raw-ingestion.md       # PRD: raw ingestion — drops/pastes/emails/news (done)
    ├── 21-obsidian-host.md       # PRD: research only, Obsidian as host (exploration, not building)
    ├── 22-sidebar-navigation.md  # PRD: rail + contextual panel, Settings view (done, panel part superseded by 22b)
    ├── 22b-rail-only.md          # PRD: no contextual panel — rail switches views, settings live in the Settings view (done)
    ├── 23-wanko-only.md          # PRD: Wanko as the only assistant, avatar picker removed (done)
    ├── 24-chat-layout-text.md    # PRD: companion column, typed turns, mute instead of PTT (done)
    ├── 25-sources-inbox.md       # PRD: Sources view as inbox and provenance (done)
    ├── 26-identity-aliases.md    # PRD: identity.aliases, is_user everywhere, ASR name biasing (done)
    ├── 27-wanko-interactivity.md # PRD: motion groups, click-to-pet, state cues, new emotions (done)
    ├── 28-wiki-editor.md         # PRD: wiki as a light Obsidian-compatible editor (done)
    ├── 29-e2e.md                 # PRD: Playwright end-to-end tests (done)
    ├── 30-command-palette.md     # PRD: ⌘K command palette (done)
    ├── 31-office-formats.md      # PRD: docx/pptx/xlsx/vtt/ics drops, meeting pages, calendar stubs, big-text sampling, epub, authors are not people (done)
    ├── 32-source-recall.md       # PRD: search inside sources — chunk index, read_source, auto-recall into the turn (done)
    ├── 33-untrusted-text.md      # PRD: documents are data not instructions, memory-op provenance, read-only tool policy (done)
    ├── 34-grounded-answers.md    # PRD: look it up first, short cited answer + "dig deeper" offer, recall precision (done)
    ├── 35-browser-location.md    # PRD: browser location, stored in settings.json, .env fallback; addendum: reverse-geocoded name in the system prompt (done)
    ├── 37-wikipedia-routing.md   # PRD: get_wikipedia tool + routing between own knowledge / local base / Wikipedia offer (done)
    ├── 38-focus-context.md       # PRD: the open wiki/source page rides on the turn (head + matching section), find/open from chat, go back (done)
    ├── 39-context-meter.md       # PRD: context meter + 75/90 % warnings, recall stripping, explicit "new topic" with a visible summary (done)
    ├── 40-docker-prod.md         # PRD: prod container + ~/docker-data/voicebot single data folder, host ASR service, prod.sh migrate/backup/restore (done)
    ├── 41-service-health.md      # PRD: service health banner + Settings rows, scripts/voicebot start/stop/status (done)
    ├── 42-recall-precision-2.md  # PRD: the stress-test recall defects — title words, held-reply lookup fallback, page recall, proper bigrams (done)
    ├── 43-written-answers.md    # PRD: length by origin+groundedness, written cards (Copy / Save to wiki), explicit notes/ save (done)
    ├── 44-vault-health.md       # PRD: lint for dead source refs / uncited raw files / duplicate names; related pages at ingest (done)
    ├── 45-ask-before-lookup.md  # PRD: spoken turns with weak recall evidence ask before looking up; typed unchanged (done)
    ├── 46-briefing.md           # PRD: "what's new?" — spoken recap of vault/memory changes since the last briefing, card on request (done)
    ├── 47-remarkable-pull.md    # PRD: scripts/remarkable_sync.py pull — notebooks over USB → page PNGs → image ingest, one source per notebook; fake device in tests (done)
    ├── 47b-remarkable-push.md   # PRD: "send this to my reMarkable" — host push service on :8021, markdown→PDF, approval via PRD 52 (exploration)
    ├── 48-batch-import.md       # PRD: scripts/ingest_batch.py — file a folder of PDFs/docx (OneNote exports) through the normal pipeline, resumable (done)
    ├── 49-event-trace.md        # PRD: events.jsonl via core/trace.py — prod profile (actionable, never text) vs dev profile (+ verbose file); rotation; trace_report.py; /trace skill (done)
    ├── 50-modular-layout.md     # PRD: layered packages (core/vault/memory/ingest/tools/agent/server), tool registry with trigger + schema modes, layer ratchet test; 12 pure-move steps (implementation; steps 1–4 done — step 1: core/, workflows → scripts, tests/test_layers.py; step 2: tools/registry.py; step 3: trigger tools in the registry, tools/commands.py, `@registry.handler`; step 4: core/llm.py model-call chokepoint, one conftest fake)
    ├── 51-run-cards.md          # PRD: run cards (steps, actions, run_url) + server-rendered /runs/<id> page from the event log (done)
    ├── 52-approvals.md          # PRD: one pending slot; spoken "yes" and a card button resolve writes identically; stale run_id refused (exploration)
    ├── 53-scalable-index.md     # PRD: SQLite postings index, incremental updates, golden BM25 parity, perf gates; fixes the 200-line index lint bug (done)
    ├── 54-vault-map-research.md # PRD: two-level vault map replaces the alphabetical prompt index; typed "research X" workflow with judge, citations, budget (exploration)
    └── 55-hybrid-measurement.md # PRD: BM25 vs embeddings vs RRF hybrid measured on a synthetic vault; adopt only on numbers (exploration)
```

## Active PRDs

- [04-avatar.md](04-avatar.md) — Live2D avatar, browser playback, oMLX backend (done)
- [05-fillers.md](05-fillers.md) — tool-intent spoken fillers (done)
- [06-avatar-switch.md](06-avatar-switch.md) — switchable avatars, Wanko default (done, superseded by 23)
- [07-avatar-menu.md](07-avatar-menu.md) — third avatar (Natori) and in-app menu with live switching (done, superseded by 23)
- [08-backdrops.md](08-backdrops.md) — selectable backdrops (done)
- [09-continuous-mode.md](09-continuous-mode.md) — hands-free conversation mode (done)
- [10-personalities.md](10-personalities.md) — character cards, canned greetings (done)
- [11-context-and-length.md](11-context-and-length.md) — context budget, history trimming, verbosity (done)
- [12-turn-controller.md](12-turn-controller.md) — session, turn controller, turn ids, origin check, atomic settings (done)
- [13-streamed-first-call.md](13-streamed-first-call.md) — stream the first LLM call (done)
- [14-memory-eval.md](14-memory-eval.md) — memory evaluation harness, `python -m eval.harness` (done)
- [15-memory-v2.md](15-memory-v2.md) — structured profile memory with upsert + reflection (done)
- [16-memory-editor.md](16-memory-editor.md) — memory viewer/editor panel (done)
- [18-memory-decay.md](18-memory-decay.md) — seen/stale decay, CONFIRM op, memory-view drag-and-drop (done)
- [19-wiki-layer.md](19-wiki-layer.md) — ~/llm-wiki Obsidian vault: page ops, search_wiki/read_page, ingest, lint (done)
- [20-raw-ingestion.md](20-raw-ingestion.md) — raw ingestion: drops/pastes/emails/news into `raw/` + `sources/` pages (done)
- [22-sidebar-navigation.md](22-sidebar-navigation.md) — sidebar/navigation review: rail + contextual panel, Settings view (done, panel part superseded by 22b)
- [22b-rail-only.md](22b-rail-only.md) — rail switches views only; all settings live in the Settings view (done)
- [23-wanko-only.md](23-wanko-only.md) — Wanko as the only assistant, avatar picker removed (done)
- [24-chat-layout-text.md](24-chat-layout-text.md) — companion column (avatar/history/composer), text input, mute replaces push-to-talk, backdrop thumbnails (done)
- [25-sources-inbox.md](25-sources-inbox.md) — Sources view as an inbox: newest-first rows, kind chips, Open raw / Re-ingest / Forget, rail badge; `sources` WS message (done)
- [26-identity-aliases.md](26-identity-aliases.md) — `identity.aliases`, `memory.is_user` for extraction/ingest, whisper name biasing (done)
- [27-wanko-interactivity.md](27-wanko-interactivity.md) — Wanko motion groups, click-to-pet, listening/thinking/tool/greeting/shutdown cues, three new emotion tags (done)
- [28-wiki-editor.md](28-wiki-editor.md) — wiki as a light, Obsidian-compatible editor: rendered pages, ⌘S save with etag conflicts, backlinks, search, tags, create/rename/delete (done)
- [29-e2e.md](29-e2e.md) — Playwright e2e layer: spawned server, temp vault, wiki editor flows, one live turn; `SETTINGS_PATH`/`MEMORY_DIR` env overrides (done)
- [30-command-palette.md](30-command-palette.md) — ⌘K command palette: views, wiki pages/Today/New, backdrop/reply length/voice/mute, file a source, re-ingest, shutdown; wiki-search fallback (done)
- [31-office-formats.md](31-office-formats.md) — Word/PowerPoint/Excel/Teams transcript/.ics through the same drop: new kinds and chips, `meetings/` pages from transcripts, deterministic calendar stubs with the `calendar_horizon_days` setting, 10 MB text with head/middle/tail sampling, model-titled pastes with a topic tag (done)
- [32-source-recall.md](32-source-recall.md) — search inside sources: text-kind raw files chunked and cached per sha in `raw/.index/`, chunk BM25 behind `search_wiki` snippets and a `read_source` tool, the best passages injected into the user message per turn above `recall_min_score` (Settings), raw files renamed after the model's title (done)
- [33-untrusted-text.md](33-untrusted-text.md) — hardening against prompt injection in filed/fetched text: `<document>` wrapping + standing rule, `source:` provenance on memory ops, read-only tool policy test, injection eval (done)
- [36-intake-progress.md](36-intake-progress.md) — bulk intake progress: one `/ingest` request per file in list order, per-row queued/uploading/summarising/filed/already/error states with elapsed time, `ingest_progress` pushes from the ingest thread, Cancel after the current file (done)
- [34-grounded-answers.md](34-grounded-answers.md) — grounded answers: look-it-up-first prompt block (only with recall tools registered), 1–3 sentence cited answer ending in "Want me to dig deeper?", a short "yes" reads the neighbouring chunks at `long` for that one completion, recall needs ≥ 2 matched content words or a proper noun, Settings note (done)
- [35-browser-location.md](35-browser-location.md) — the page offers its position once per connect (`set_location`), the weather tool reads settings.json → `.env` → default, Settings shows the resolved location; addendum 2026-08-30: one Nominatim reverse geocode per new position, `The user is in <name>` in the system prompt (done)
- [37-wikipedia-routing.md](37-wikipedia-routing.md) — Wikipedia search/summary tool with fixed hosts and an offer-then-fetch flow reusing the PRD 34 offer state; routing rules and a 28-question calibration set (done)
- [38-focus-context.md](38-focus-context.md) — focus context: `set_focus` from the UI, the open page attached to each turn as a capped document block (head + best-matching section), deterministic find/open with `wiki_open`, last-page memory for "go back" (done)
- [39-context-meter.md](39-context-meter.md) — context meter (`context` message, rail dot, 90 % spoken nudge), recall blocks stripped after their turn, explicit `new_topic` that folds memory first and restarts with a visible `<earlier>` summary divider (done)
- [40-docker-prod.md](40-docker-prod.md) — Docker as prod: one data folder `~/docker-data/voicebot` (vault keeps the name llm-wiki), env knobs for every state path, host ASR service so mlx-whisper stays native, compose with read-only rootfs, `scripts/prod.sh` build/up/migrate/backup/restore; runbook in [deploy.md](deploy.md) (done)
- [41-service-health.md](41-service-health.md) — server-side probes of oMLX / ASR pushed as `services`, UI banner + Settings rows with a copy-the-command button, turns refused with the hint while the LLM is down, `scripts/voicebot start` (starts only what is missing) / `stop` (container down, ASR unloaded, Gemma unloaded via the oMLX admin API) / `status`; addendum 2026-08-30: the admin password stored from Settings (`.env` wins), Load / Unload model buttons through the server (done)
- [42-recall-precision-2.md](42-recall-precision-2.md) — the 2026-08-30 stress-test recall defects: a source's title words pick the source but never score a chunk, four candidates with no neighbouring pair, a held first sentence that turns "the documents do not mention" into one server-side lookup re-ask (`<lookup>`), wiki pages join recall by name (head + Summary + best section), short questions recall on a proper term, proper bigrams must match whole, "summarise this" uses the page head, routing wording for past events / opinions, trailing "please" in open-page commands, no double date in raw names (done)

## Research

Exploration-only documents; nothing is being built from them directly.

- [17-long-term-memory.md](17-long-term-memory.md) — research: decay, archive, LLM-wiki layer (exploration, research draft)
- [21-obsidian-host.md](21-obsidian-host.md) — research only: Obsidian as host vs TS rewrite (exploration, not building)

## Completed PRDs

- [01-voicebot.md](01-voicebot.md) — initial local voice chatbot (done)
- [02-llm-robustness-tests.md](02-llm-robustness-tests.md) — LLM tool-trigger robustness tests (done)
