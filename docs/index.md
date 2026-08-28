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
python3 -m pytest tests/
# unit-only (no oMLX needed):
python3 -m pytest tests/test_splitter.py tests/test_llm_unit.py tests/test_chatbot_speech.py
```

## Folder tree

```
voicebot/
├── chatbot.py        # main entry point, FastAPI server, voice loop, sentence-streamed TTS over WS
├── llm.py            # LLM client (oMLX via openai SDK), tool dispatch, streaming, history
├── splitter.py       # LLM delta stream → (emotion, sentence) pairs
├── memory.py         # session summarization, shortmem.txt persistence
├── scripts/fetch_assets.sh  # downloads pixi, pixi-live2d-display, Cubism Core, Haru model
├── static/
│   ├── index.html    # browser UI: avatar + transcript + push-to-talk
│   ├── app.js        # Live2D model, audio queue, lip-sync, expressions, websocket
│   ├── style.css
│   ├── vendor/       # gitignored, fetched
│   └── models/haru/  # gitignored, fetched
├── tools/
│   ├── time.py       # current time tool
│   ├── weather.py    # open-meteo weather tool
│   ├── news.py       # BBC news headlines + article detail
│   └── gmail.py      # Gmail read/search via OAuth
├── tests/            # pytest test suite
├── requirements.txt
└── docs/
    ├── index.md      # this file
    ├── process.md    # development process
    ├── 01-voicebot.md  # PRD: initial voicebot (done)
    └── 04-avatar.md    # PRD: Live2D avatar + oMLX (implementation)
```

## Active PRDs

- [04-avatar.md](04-avatar.md) — Live2D avatar, browser playback, oMLX backend (implementation)
- [02-llm-robustness-tests.md](02-llm-robustness-tests.md) — LLM tool-trigger robustness tests (implementation)

## Completed PRDs

- [01-voicebot.md](01-voicebot.md) — initial local voice chatbot (done)
