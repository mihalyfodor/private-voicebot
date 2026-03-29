import os
import sys
from datetime import datetime
import tty
import termios
import tempfile
import subprocess
import numpy as np
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

SYSTEM_PROMPT = f"You are a voice assistant with memory. Keep responses short and conversational. Talk like a person, not a chatbot. Today is {datetime.now().strftime('%A, %B %d %Y %H:%M')}."

SAMPLERATE = 16000


def load_shortmem():
    if os.path.exists(SHORTMEM_PATH):
        with open(SHORTMEM_PATH, "r") as f:
            content = f.read().strip()
        if content:
            return f"{SYSTEM_PROMPT}\n\nBackground context about the user — silent context only. Use it to inform your understanding, never bring it up unless the user does first:\n{content}"
    return SYSTEM_PROMPT


def save_shortmem(session_turns):
    existing = ""
    if os.path.exists(SHORTMEM_PATH):
        with open(SHORTMEM_PATH, "r") as f:
            existing = f.read().strip()

    session_text = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in session_turns)

    messages = [
        {
            "role": "system",
            "content": (
                "You extract new facts about a user from a conversation transcript. "
                "Compare against existing memory and output only facts that are genuinely new. "
                "One fact per line. No duplicates, no filler, no commentary. "
                "If nothing new was learned, reply with exactly: NOTHING"
            )
        },
        {
            "role": "user",
            "content": f"Existing memory:\n{existing}\n\nNew session transcript:\n{session_text}"
        }
    ]

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": 32768}
    })
    summary = response.json()["message"]["content"].strip()

    if not summary or summary.upper() == "NOTHING" or len(summary) < 10:
        print("\n[Nothing new to save]")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(SHORTMEM_PATH, "a") as f:
        f.write(f"\n--- {timestamp} ---\n{summary}\n")
    print("\n[Memory saved to shortmem.txt]")


def get_keypress():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


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


# Init Kokoro once
print("Loading Kokoro TTS...")
kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
print("Ready. Press SPACE to start/stop recording. Ctrl+C to quit.\n")

conversation = [{"role": "system", "content": load_shortmem()}]
session_turns = []

if __name__ == "__main__":
    try:
        while True:
            print("[Idle] Press SPACE to speak...", end="\r", flush=True)
            key = get_keypress()

            if key == "\x03":  # Ctrl+C
                raise KeyboardInterrupt

            if key != " ":
                continue

            # Start recording
            audio_chunks = []
            def callback(indata, frames, time, status):
                audio_chunks.append(indata.copy())

            print("[RECORDING...] Press SPACE to stop.   ", flush=True)
            stream = sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=callback)
            stream.start()

            # Wait for space again
            while True:
                key = get_keypress()
                if key == "\x03":
                    stream.stop()
                    stream.close()
                    raise KeyboardInterrupt
                if key == " ":
                    break

            stream.stop()
            stream.close()

            if not audio_chunks:
                continue

            print("[Processing...]                        ", flush=True)
            audio = np.concatenate(audio_chunks, axis=0)
            tmp = tempfile.mktemp(suffix=".wav")
            sf.write(tmp, audio, SAMPLERATE)

            text = transcribe(tmp)
            os.remove(tmp)

            if not text:
                print("[No speech detected]")
                continue

            print(f"You: {text}")
            print("[Thinking...]")
            response = ask_llm(text)
            print(f"Bot: {response}")
            speak(response)

    except KeyboardInterrupt:
        print("\nExiting...")
        if session_turns:
            save_shortmem(session_turns)
