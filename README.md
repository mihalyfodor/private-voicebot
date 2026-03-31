# Voicebot

A local voice assistant with memory. Runs entirely on your machine — no cloud, no API keys.

**Stack:** Whisper (STT) → Ollama/Gemma (LLM) → Kokoro (TTS)

## Requirements

- macOS or Linux (WSL not supported — audio passthrough too unreliable)
- [Ollama](https://ollama.com) installed and running

## Setup

**macOS**
```bash
brew install whisper-cpp portaudio
```

**Linux**
```bash
apt install portaudio19-dev
# Build whisper-cpp from source: https://github.com/ggerganov/whisper.cpp
```

**Both**
```bash
# Download Whisper model
mkdir -p ~/whisper-models
curl -L -o ~/whisper-models/ggml-small.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin

# Download Kokoro models
mkdir -p ~/kokoro
curl -L -o ~/kokoro/kokoro-v1.0.onnx \
  https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
curl -L -o ~/kokoro/voices-v1.0.bin \
  https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin

# Pull the LLM
ollama pull gemma3:4b

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
ollama serve          # if not already running
source .venv/bin/activate
python3 chatbot.py    # opens browser automatically
```

Press **Space** or click the button to start/stop recording. Use **Shut down** in the browser to exit cleanly.

## Memory

Conversations are summarized and saved to `shortmem.txt` on exit. This file is loaded on next startup as background context. It is gitignored — personal to your machine.
