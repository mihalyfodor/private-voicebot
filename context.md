# Voicebot Project Context

## Status
In progress — blocked on macOS microphone permission.

## What's done
- Ollama installed (via site, not brew) and updated, `gemma3:4b` pulled and working
- whisper-cpp installed via brew, binary is `whisper-cli` at `/opt/homebrew/bin/whisper-cli`
- Whisper small model downloaded to `~/whisper-models/ggml-small.bin`
- Kokoro models at `~/kokoro/kokoro-v1.0.onnx` and `~/kokoro/voices-v1.0.bin`
- All Python packages installed globally (no venv): `kokoro-onnx`, `sounddevice`, `soundfile`, `requests`, `onnxruntime`
- portaudio installed via brew
- `chatbot.py` written and ready at `/Users/waycent/claude/voicebot/chatbot.py`

## Current blocker
Microphone amplitude reads 0.0 — terminal app needs mic permission.

**Fix:** System Settings → Privacy & Security → Microphone → enable for your terminal app (Terminal, iTerm2, or VS Code).

## How to run
```bash
# Terminal 1
ollama serve

# Terminal 2
python3 /Users/waycent/claude/voicebot/chatbot.py
```

## Key decisions
- Using `gemma3:4b` instead of `llama3.2`
- Using `/api/chat` endpoint (not `/api/generate`) for conversation history
- Kokoro initialized once at startup to avoid cold start per turn
- TTS plays via sounddevice directly (no shell subprocess)
- No venv — packages installed globally
