import os
import asyncio
from dotenv import load_dotenv
load_dotenv()
import tempfile
import subprocess
import threading
import webbrowser

import llm
import numpy as np
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
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.html")

SAMPLERATE = 16000


def transcribe(wav_path):
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path, "--no-timestamps", "-nt"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def speak(text):
    samples, sample_rate = kokoro.create(text, voice=KOKORO_VOICE)
    sd.play(samples, sample_rate)
    sd.wait()


# Init
print("Loading Kokoro TTS...")
kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
print("Ready.\n")

# State
recording = False
processing = False
greeted = False
audio_chunks = []
stream = None
ws_client = None


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

                reply = llm.ask(text)
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
    reply = llm.ask("(The user just opened the app. Give a short, natural greeting. Do not mention memory or context.)")
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
    global ws_client, greeted
    await websocket.accept()
    ws_client = websocket
    loop = asyncio.get_event_loop()
    if not greeted:
        greeted = True
        await send({"type": "state", "value": "thinking"})
        threading.Thread(target=greet, args=(loop,), daemon=True).start()
    else:
        await send({"type": "state", "value": "idle"})
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("action") == "toggle":
                threading.Thread(target=handle_toggle, args=(loop,), daemon=True).start()
            elif msg.get("action") == "shutdown":
                print("\nShutdown requested from UI...")
                llm.save_memory()
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
        llm.save_memory()
