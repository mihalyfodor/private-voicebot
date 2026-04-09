import os
import asyncio
import tempfile
import subprocess
import threading
import webbrowser
from datetime import datetime

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from kokoro_onnx import Kokoro
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn

WHISPER_CLI = "/opt/homebrew/bin/whisper-cli"
WHISPER_MODEL = os.path.expanduser("~/whisper-models/ggml-small.bin")
KOKORO_MODEL = os.path.expanduser("~/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES = os.path.expanduser("~/kokoro/voices-v1.0.bin")
KOKORO_VOICE = "af_heart"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e2b"
SHORTMEM_PATH = os.path.join(os.path.dirname(__file__), "shortmem.txt")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.html")

SYSTEM_PROMPT = f"You are a voice assistant with memory. Keep responses short and conversational. Talk like a person, not a chatbot. Today's date is {datetime.now().strftime('%A, %B %d %Y')}. Use the get_time tool when asked for the current time."

SAMPLERATE = 16000

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Returns the current local time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]


def run_tool(name, args):
    if name == "get_time":
        return datetime.now().strftime("%H:%M:%S")
    return "unknown tool"


def load_shortmem():
    if os.path.exists(SHORTMEM_PATH):
        with open(SHORTMEM_PATH, "r") as f:
            content = f.read().strip()
        if content:
            return f"{SYSTEM_PROMPT}\n\nThe following is background context about the USER you are speaking with — it describes them, not you. Use it silently to inform your understanding. Never bring it up unless the user does first:\n{content}"
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
        "tools": TOOLS,
        "stream": False,
        "options": {"num_ctx": 32768}
    })
    data = response.json()
    if "message" not in data:
        print(f"[Ollama error] {data}")
        raise KeyError(f"No 'message' in response: {data}")
    msg = data["message"]

    if msg.get("tool_calls"):
        conversation.append(msg)
        for tc in msg["tool_calls"]:
            name = tc["function"]["name"]
            args = tc["function"].get("arguments", {})
            result = run_tool(name, args)
            print(f"[Tool] {name}() → {result}")
            conversation.append({"role": "tool", "content": result})

        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": conversation,
            "stream": False,
            "options": {"num_ctx": 32768}
        })
        msg = response.json()["message"]

    reply = msg["content"]
    conversation.append({"role": "assistant", "content": reply})
    session_turns.append({"role": "assistant", "content": reply})
    return reply


def speak(text):
    samples, sample_rate = kokoro.create(text, voice=KOKORO_VOICE)
    sd.play(samples, sample_rate)
    sd.wait()


# Init
print("Loading Kokoro TTS...")
kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
print("Ready.\n")

conversation = [{"role": "system", "content": load_shortmem()}]
session_turns = []

# State
recording = False
processing = False
audio_chunks = []
stream = None
ws_client = None  # single active WebSocket


async def send(msg: dict):
    if ws_client:
        try:
            await ws_client.send_json(msg)
        except Exception:
            pass


def handle_toggle(loop):
    global recording, processing, audio_chunks, stream

    if processing:
        return

    if not recording:
        # Start recording
        recording = True
        audio_chunks = []

        def callback(indata, frames, time, status):
            if recording:
                audio_chunks.append(indata.copy())

        stream = sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=callback)
        stream.start()
        asyncio.run_coroutine_threadsafe(send({"type": "state", "value": "recording"}), loop)
        print("[RECORDING...]")

    else:
        # Stop recording
        recording = False
        stream.stop()
        stream.close()
        processing = True
        asyncio.run_coroutine_threadsafe(send({"type": "state", "value": "processing"}), loop)

        def process():
            global processing
            try:
                if not audio_chunks:
                    return
                audio = np.concatenate(audio_chunks, axis=0)
                tmp = tempfile.mktemp(suffix=".wav")
                sf.write(tmp, audio, SAMPLERATE)

                text = transcribe(tmp)
                os.remove(tmp)

                if not text:
                    return

                print(f"You: {text}")
                asyncio.run_coroutine_threadsafe(send({"type": "transcript", "role": "user", "text": text}), loop)
                asyncio.run_coroutine_threadsafe(send({"type": "state", "value": "thinking"}), loop)

                reply = ask_llm(text)
                print(f"Bot: {reply}")
                asyncio.run_coroutine_threadsafe(send({"type": "transcript", "role": "assistant", "text": reply}), loop)
                asyncio.run_coroutine_threadsafe(send({"type": "state", "value": "speaking"}), loop)

                speak(reply)
            except Exception as e:
                print(f"[Error] {e}")
            finally:
                processing = False
                asyncio.run_coroutine_threadsafe(send({"type": "state", "value": "idle"}), loop)

        threading.Thread(target=process, daemon=True).start()


def greet(loop):
    reply = ask_llm("(The user just opened the app. Give a short, natural greeting. Do not mention memory or context.)")
    print(f"Bot: {reply}")
    asyncio.run_coroutine_threadsafe(send({"type": "transcript", "role": "assistant", "text": reply}), loop)
    asyncio.run_coroutine_threadsafe(send({"type": "state", "value": "speaking"}), loop)
    speak(reply)
    asyncio.run_coroutine_threadsafe(send({"type": "state", "value": "idle"}), loop)


app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(INDEX_PATH)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global ws_client
    await websocket.accept()
    ws_client = websocket
    loop = asyncio.get_event_loop()
    await send({"type": "state", "value": "thinking"})
    loop = asyncio.get_event_loop()
    threading.Thread(target=greet, args=(loop,), daemon=True).start()
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("action") == "toggle":
                threading.Thread(target=handle_toggle, args=(loop,), daemon=True).start()
            elif msg.get("action") == "shutdown":
                print("\nShutdown requested from UI...")
                if session_turns:
                    save_shortmem(session_turns)
                os._exit(0)
    except WebSocketDisconnect:
        ws_client = None


def open_browser():
    import time
    time.sleep(1)
    webbrowser.open("http://localhost:8000")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nExiting...")
        if session_turns:
            save_shortmem(session_turns)
