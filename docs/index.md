# Repo Index

## How to run

```bash
# oMLX must be running on :8000; run scripts/fetch_assets.sh once
source .venv/bin/activate
python3 chatbot.py    # opens http://localhost:8010 automatically
```

## How to run tests

```bash
source .venv/bin/activate
# unit suite (no oMLX needed) — pytest.ini excludes the `live` marker by default
python3 -m pytest tests/
# live tests (needs oMLX on :8000)
python3 -m pytest -m live tests/
```

## Folder tree

```
voicebot/
├── chatbot.py        # main entry point, FastAPI + WS server, browser mic capture (hands-free), typed turns (`text_turn`), sentence-streamed TTS, avatar/backdrop switching
├── llm.py             # LLM client (oMLX via openai SDK), system prompt per avatar, tool dispatch, ask_events/ask_stream/ask, memory + session log
├── characters.py     # character card loader + persona builder
├── characters/       # editable YAML character cards
├── session.py        # Session (per-connection state) + TurnController (single worker)
├── vad.py            # Silero VAD endpointer (hands-free); Recorder is an unused PTT leftover
├── asr.py            # mlx-whisper transcription + junk filter, optional initial_prompt name biasing
├── avatars.py          # avatar profile (Wanko only since PRD 23: name, voice, speed, persona), current/set_current, settings.json persistence
├── backdrops.py        # backdrop catalogue (key, name, file, credit, url, license), validate/listing
├── fillers.py           # short spoken fillers played while a tool call is in flight, no immediate repeats
├── splitter.py         # LLM delta stream → (emotion, sentence) pairs, strips the leading [tag]
├── memory.py           # structured profile memory: load/render, LLM-proposed ops, seen/stale decay, reflection, the identity section (aliases, is_user, user_note, ASR name prompt) and the editor entry points (snapshot/edit/edit_many/reset)
├── wiki.py             # wiki layer: ~/llm-wiki vault, page ops + git commits, BM25 search, ingest (never a people page about the user), lint, archive/pull bridge
├── sources.py          # raw ingestion: raw/ storage + manifest, pdf/eml/msg/image extraction, summary + page ops, POST /ingest core
├── scripts/
│   ├── fetch_assets.sh  # downloads pixi.js, pixi-live2d-display, Cubism Core, the Wanko Live2D model (all 12 motions) and backdrop images (gitignored), then generates Wanko.app.model3.json
│   ├── wanko_groups.py  # PRD 27 motion-group table + generator for static/models/wanko/Wanko.app.model3.json
│   └── screenshot.py    # dev helper: headless screenshot of the running app (`--view memory`, `--drag`, `--drop-file`)
├── static/
│   ├── index.html      # browser UI: icon rail, companion column (avatar/history/composer), workspace views (Chat / Memory / Wiki / Sources / Settings), intake popup
│   ├── mic-worklet.js # AudioWorklet: mic → 16 kHz Int16 frames
│   ├── app.js           # PROFILES/CONFIG/MOTIONS, Live2D model load, audio queue, lip-sync, expressions, motion groups + click-to-pet + state cues (PRD 27), rail (VIEWS table), setView (companion re-parenting), renderComposer (mic/mute/status), memory/wiki/settings views, drop-paste intake, keyboard, websocket
│   ├── style.css        # shell layout (--rail-w), rail, #companion chat-grid vs side slice, history/composer, views, settings rows, narrow-screen bottom tab bar
│   ├── vendor/          # gitignored, fetched
│   ├── models/           # gitignored, fetched (wanko/)
│   └── backdrops/        # gitignored, fetched
├── tools/
│   ├── time.py         # current time tool
│   ├── weather.py       # open-meteo weather tool
│   ├── news.py          # BBC news headlines + article detail
│   ├── gmail.py         # Gmail read/search via OAuth
│   ├── wiki.py          # search_wiki / read_page — the model's way into the vault
│   └── sources.py       # file_source(email|news|note) — voice ingestion of the last email/headline/a note
├── workflows/
│   ├── email_classification.py       # Gmail triage dashboard (category/urgency via local LLM)
│   └── classification_rules.example.py
├── eval/             # memory eval harness: harness.py, scripts/*.yaml (incl. aliases.yaml), fixtures/, results/
├── tests/
│   ├── test_splitter.py         # SentenceSplitter / split_stream / strip_tag
│   ├── test_llm_unit.py          # llm.py with a mocked OpenAI client, no oMLX needed
│   ├── test_llm.py                # llm.py against a live oMLX server
│   ├── test_llm_robustness.py     # LLM tool-trigger robustness, needs live oMLX
│   ├── test_tools.py              # tool schemas/behaviour, needs live oMLX for some cases
│   ├── test_chatbot_speech.py     # speak_stream: mocked Kokoro + WS, wav framing
│   ├── test_avatars.py            # avatars.current()/AVATAR env handling
│   ├── test_avatar_switch.py      # set_current()/settings.json persistence, invalid keys
│   ├── test_backdrops.py          # backdrops.validate()/listing()
│   ├── test_vad.py                # Endpointer state machine (fake prob_fn) + Recorder
│   ├── test_asr.py                # junk filtering + transcribe() with mlx-whisper mocked, initial_prompt biasing
│   ├── test_hands_free.py         # WS hands-free actions, binary audio frames, text_turn while listening
│   ├── test_memory.py             # memory v2: ops, caps, render budget, reflection, identity aliases
│   ├── test_wiki.py               # wiki layer: vault, page ops, search, ingest, lint, archive/pull, actions
│   ├── test_sources.py            # raw ingestion: raw/ + manifest, pdf/eml/image extraction, restricted ops, file_source, POST /ingest
│   ├── test_memory_editor.py      # memory panel actions: get_memory/memory_op/memory_reset, locking, fuzz
│   ├── test_memory_decay.py       # seen/stale decay, CONFIRM, memory_ops batches, harness edit/clock turns
│   ├── test_settings_action.py    # set_setting whitelist, /api/config payload, no panel markup in index.html, OPEN_BROWSER flag
│   ├── test_text_input.py         # PRD 24: text_turn == spoken turn, `typed:` log line, no `ptt`, client fixture assertions
│   ├── test_avatar_cues.py        # PRD 27: generated motion groups, click-to-pet, state cues, new emotion tags
│   └── app_fixture.mjs            # boots static/app.js in node on a DOM/WS/audio/PIXI shim and prints one JSON
│                                   # of results (companion re-parent, typed turn, mute, thumbs, avatar cues); its header
│                                   # documents the hooks and how to add a scenario
├── requirements.txt
└── docs/
    ├── index.md               # this file
    ├── process.md              # development process
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
    ├── 19-wiki-layer.md          # PRD: ~/llm-wiki vault, page ops, search (refactoring)
    ├── 20-raw-ingestion.md       # PRD: raw ingestion — drops/pastes/emails/news (refactoring)
    ├── 21-obsidian-host.md       # PRD: research only, Obsidian as host (not building)
    ├── 22-sidebar-navigation.md  # PRD: rail + contextual panel, Settings view (done, panel part superseded by 22b)
    ├── 22b-rail-only.md          # PRD: no contextual panel — rail switches views, settings live in the Settings view (done)
    ├── 23-wanko-only.md          # PRD: Wanko as the only assistant, avatar picker removed (done)
    ├── 24-chat-layout-text.md    # PRD: companion column, typed turns, mute instead of PTT (done)
    ├── 25-sources-inbox.md       # PRD: Sources view as inbox and provenance (exploration)
    ├── 26-identity-aliases.md    # PRD: identity.aliases, is_user everywhere, ASR name biasing (done)
    ├── 27-wanko-interactivity.md # PRD: motion groups, click-to-pet, state cues, new emotions (implementation)
    └── 28-wiki-editor.md         # PRD: wiki as a light Obsidian-compatible editor (exploration)
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
- [17-long-term-memory.md](17-long-term-memory.md) — research: decay, archive, LLM-wiki layer (exploration)
- [18-memory-decay.md](18-memory-decay.md) — seen/stale decay, CONFIRM op, memory-view drag-and-drop (done)
- [19-wiki-layer.md](19-wiki-layer.md) — ~/llm-wiki Obsidian vault: page ops, search_wiki/read_page, ingest, lint (refactoring)
- [20-raw-ingestion.md](20-raw-ingestion.md) — raw ingestion: drops/pastes/emails/news into `raw/` + `sources/` pages (refactoring)
- [21-obsidian-host.md](21-obsidian-host.md) — research only: Obsidian as host vs TS rewrite (not building)
- [22-sidebar-navigation.md](22-sidebar-navigation.md) — sidebar/navigation review: rail + contextual panel, Settings view (done, panel part superseded by 22b)
- [22b-rail-only.md](22b-rail-only.md) — rail switches views only; all settings live in the Settings view (done)
- [23-wanko-only.md](23-wanko-only.md) — Wanko as the only assistant, avatar picker removed (done)
- [24-chat-layout-text.md](24-chat-layout-text.md) — companion column (avatar/history/composer), text input, mute replaces push-to-talk, backdrop thumbnails (done)
- [25-sources-inbox.md](25-sources-inbox.md) — Sources view as an inbox with provenance (exploration)
- [26-identity-aliases.md](26-identity-aliases.md) — `identity.aliases`, `memory.is_user` for extraction/ingest, whisper name biasing (done)
- [27-wanko-interactivity.md](27-wanko-interactivity.md) — Wanko motion groups, click-to-pet, listening/thinking/tool/greeting/shutdown cues, three new emotion tags (implementation)
- [28-wiki-editor.md](28-wiki-editor.md) — wiki as a light, Obsidian-compatible editor (exploration)

## Completed PRDs

- [01-voicebot.md](01-voicebot.md) — initial local voice chatbot (done)
- [02-llm-robustness-tests.md](02-llm-robustness-tests.md) — LLM tool-trigger robustness tests (done)
