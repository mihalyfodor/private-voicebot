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
├── chatbot.py        # main entry point, FastAPI + WS server, browser mic capture (PTT + hands-free), sentence-streamed TTS, avatar/backdrop switching
├── llm.py             # LLM client (oMLX via openai SDK), system prompt per avatar, tool dispatch, ask_events/ask_stream/ask, memory + session log
├── characters.py     # character card loader + persona builder
├── characters/       # editable YAML character cards
├── session.py        # Session (per-connection state) + TurnController (single worker)
├── vad.py            # Silero VAD endpointer (hands-free) + PTT Recorder
├── asr.py            # mlx-whisper transcription + junk filter
├── avatars.py          # avatar profiles (Wanko/Haru/Natori: name, voice, speed, persona), current/set_current, settings.json persistence
├── backdrops.py        # backdrop catalogue (key, name, file, credit, url, license), validate/listing
├── fillers.py           # short spoken fillers played while a tool call is in flight, no immediate repeats
├── splitter.py         # LLM delta stream → (emotion, sentence) pairs, strips the leading [tag]
├── memory.py           # session summarization, shortmem.txt persistence
├── scripts/
│   └── fetch_assets.sh  # downloads pixi.js, pixi-live2d-display, Cubism Core, the 3 Live2D models and backdrop images (gitignored)
├── static/
│   ├── index.html      # browser UI: avatar canvas + menu + transcript + push-to-talk
│   ├── mic-worklet.js # AudioWorklet: mic → 16 kHz Int16 frames
│   ├── app.js           # PROFILES/CONFIG, Live2D model load, audio queue, lip-sync, expressions, menu drawer, websocket
│   ├── style.css
│   ├── vendor/          # gitignored, fetched
│   ├── models/           # gitignored, fetched (wanko/, haru/, natori/)
│   └── backdrops/        # gitignored, fetched
├── tools/
│   ├── time.py         # current time tool
│   ├── weather.py       # open-meteo weather tool
│   ├── news.py          # BBC news headlines + article detail
│   └── gmail.py         # Gmail read/search via OAuth
├── workflows/
│   ├── email_classification.py       # Gmail triage dashboard (category/urgency via local LLM)
│   └── classification_rules.example.py
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
│   ├── test_vad.py                # Endpointer state machine (fake prob_fn) + PTT Recorder
│   ├── test_asr.py                # junk filtering + transcribe() with mlx-whisper mocked
│   └── test_hands_free.py         # WS hands-free/PTT actions and binary audio frames
├── requirements.txt
└── docs/
    ├── index.md               # this file
    ├── process.md              # development process
    ├── 01-voicebot.md           # PRD: initial voicebot (done)
    ├── 02-llm-robustness-tests.md  # PRD: LLM tool-trigger robustness tests (done)
    ├── 04-avatar.md             # PRD: Live2D avatar + oMLX backend (done)
    ├── 05-fillers.md            # PRD: tool-intent spoken fillers (done)
    ├── 06-avatar-switch.md      # PRD: switchable avatars, Wanko default (done)
    ├── 07-avatar-menu.md        # PRD: Natori + in-app menu with live switching (done)
    ├── 08-backdrops.md          # PRD: selectable backdrops (done)
    ├── 09-continuous-mode.md    # PRD: hands-free conversation mode (done)
    ├── 10-personalities.md      # PRD: character cards (done)
    ├── 11-context-and-length.md # PRD: context budget, trimming, verbosity (done)
    ├── 12-turn-controller.md    # PRD: session, turn controller, protocol hardening (done)
    └── 13-streamed-first-call.md # PRD: stream the first LLM call (exploration)
```

## Active PRDs

- [04-avatar.md](04-avatar.md) — Live2D avatar, browser playback, oMLX backend (done)
- [05-fillers.md](05-fillers.md) — tool-intent spoken fillers (done)
- [06-avatar-switch.md](06-avatar-switch.md) — switchable avatars, Wanko default (done)
- [07-avatar-menu.md](07-avatar-menu.md) — third avatar (Natori) and in-app menu with live switching (done)
- [08-backdrops.md](08-backdrops.md) — selectable backdrops (done)
- [09-continuous-mode.md](09-continuous-mode.md) — hands-free conversation mode (done)
- [10-personalities.md](10-personalities.md) — character cards, canned greetings (done)
- [11-context-and-length.md](11-context-and-length.md) — context budget, history trimming, verbosity (done)
- [12-turn-controller.md](12-turn-controller.md) — session, turn controller, turn ids, origin check, atomic settings (done)
- [13-streamed-first-call.md](13-streamed-first-call.md) — stream the first LLM call (exploration)

## Completed PRDs

- [01-voicebot.md](01-voicebot.md) — initial local voice chatbot (done)
- [02-llm-robustness-tests.md](02-llm-robustness-tests.md) — LLM tool-trigger robustness tests (done)
