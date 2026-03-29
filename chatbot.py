import os
import tempfile
import subprocess
import requests
import sounddevice as sd
import soundfile as sf
from kokoro_onnx import Kokoro

WHISPER_CLI = "/opt/homebrew/bin/whisper-cli"
WHISPER_MODEL = os.path.expanduser("~/whisper-models/ggml-small.bin")
KOKORO_MODEL = os.path.expanduser("~/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.expanduser("~/kokoro/voices-v1.0.bin")
KOKORO_VOICE = "af_heart"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:4b"
SHORTMEM_PATH = os.path.join(os.path.dirname(__file__), "shortmem.txt")

SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses concise and conversational."


def load_shortmem():
    if os.path.exists(SHORTMEM_PATH):
        with open(SHORTMEM_PATH, "r") as f:
            content = f.read().strip()
        if content:
            return f"{SYSTEM_PROMPT}\n\nWhat you know about the user and past conversations:\n{content}"
    return SYSTEM_PROMPT


def save_shortmem(session_turns):
    summary_prompt = (
        "Extract only concrete facts learned about the user in this session: name, preferences, goals, topics discussed. "
        "One fact per line. No filler, no meta-commentary, no mention of what wasn't discussed. "
        "If nothing meaningful was learned, write nothing."
    )
    messages = session_turns + [{"role": "user", "content": summary_prompt}]
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 32768}
    })
    summary = response.json()["message"]["content"].strip()
    with open(SHORTMEM_PATH, "a") as f:
        f.write(f"\n---\n{summary}\n")
    print(f"\n[Memory saved to shortmem.txt]")


# Init Kokoro once to avoid cold start on every turn
print("Loading Kokoro TTS...")
kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
print("Ready.\n")

conversation = [{"role": "system", "content": load_shortmem()}]
session_turns = []


def record_audio(duration=5, samplerate=16000):
    print("Listening...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, audio, samplerate)
    return tmp


def transcribe(wav_path):
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path, "--no-timestamps", "-nt"],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()


def ask_llm(user_text):
    conversation.append({"role": "user", "content": user_text})
    session_turns.append({"role": "user", "content": user_text})
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": conversation,
        "stream": False,
        "options": {"num_ctx": 32768}
    })
    reply = response.json()["message"]["content"]
    conversation.append({"role": "assistant", "content": reply})
    session_turns.append({"role": "assistant", "content": reply})
    return reply


def speak(text):
    samples, sample_rate = kokoro.create(text, voice=KOKORO_VOICE)
    sd.play(samples, sample_rate)
    sd.wait()


if __name__ == "__main__":
    try:
        while True:
            wav = record_audio(duration=5)
            text = transcribe(wav)
            os.remove(wav)

            if not text:
                continue

            print(f"You: {text}")
            response = ask_llm(text)
            print(f"Bot: {response}")
            speak(response)
    except KeyboardInterrupt:
        print("\nExiting...")
        if session_turns:
            save_shortmem(session_turns)
