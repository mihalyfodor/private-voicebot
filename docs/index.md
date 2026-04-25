# Repo Index

## How to run

```bash
ollama serve          # if not already running
source .venv/bin/activate
python3 chatbot.py    # opens browser automatically
```

## How to run tests

```bash
source .venv/bin/activate
python3 -m pytest tests/
```

## Folder tree

```
voicebot/
├── chatbot.py        # main entry point, Flask server, voice loop
├── llm.py            # LLM client (Ollama), tool dispatch, conversation history
├── memory.py         # session summarization, shortmem.txt persistence
├── tools/
│   ├── time.py       # current time tool
│   ├── weather.py    # open-meteo weather tool
│   ├── news.py       # BBC news headlines + article detail
│   └── gmail.py      # Gmail read/search via OAuth
├── tests/            # pytest test suite
├── index.html        # browser push-to-talk UI
├── requirements.txt
└── docs/
    ├── index.md      # this file
    ├── process.md    # development process
    └── 01-voicebot.md  # PRD: initial voicebot (done)
```

## Active PRDs

- [02-llm-robustness-tests.md](02-llm-robustness-tests.md) — LLM tool-trigger robustness tests (implementation)

## Completed PRDs

- [01-voicebot.md](01-voicebot.md) — initial local voice chatbot (done)
