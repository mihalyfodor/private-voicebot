import os
import io
import base64
import asyncio
from dotenv import load_dotenv
load_dotenv()
import tempfile
import subprocess
import threading
import webbrowser

import llm
import splitter
import numpy as np
import sounddevice as sd
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

WHISPER_CLI = os.getenv("WHISPER_CLI", "/opt/homebrew/bin/whisper-cli")
WHISPER_MODEL = os.path.expanduser(os.getenv("WHISPER_MODEL", "~/whisper-models/ggml-small.bin"))
KOKORO_MODEL = os.path.expanduser(os.getenv("KOKORO_MODEL", "~/kokoro/kokoro-v1.0.onnx"))
KOKORO_VOICES = os.path.expanduser(os.getenv("KOKORO_VOICES", "~/kokoro/voices-v1.0.bin"))
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", "0.95"))
PORT = int(os.getenv("PORT", "8010"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

SAMPLERATE = 16000

kokoro = None  # loaded in main()


def transcribe(wav_path):
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path, "--no-timestamps", "-nt"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def synthesize(text: str) -> bytes:
    """Text → 16-bit PCM wav bytes (Kokoro's native 24 kHz)."""
    samples, sample_rate = kokoro.create(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def speak_stream(deltas, send_sync, tts=None):
    """Consume LLM deltas, TTS each sentence and push it over the socket.

    Returns the full reply text with the emotion tag stripped.
    `send_sync(dict)` must be thread-safe; `tts(text) -> wav bytes` defaults to Kokoro.
    """
    tts = tts or synthesize
    sp = splitter.SentenceSplitter()
    sentences = []
    started = False

    def emit(emotion, sentence):
        nonlocal started
        if not started:
            started = True
            send_sync({"type": "state", "value": "speaking"})
        sentences.append(sentence)
        wav = tts(sentence)
        send_sync({
            "type": "speech",
            "emotion": emotion,
            "text": sentence,
            "wav": base64.b64encode(wav).decode("ascii"),
        })

    for d in deltas:
        for emotion, sentence in sp.feed(d):
            emit(emotion, sentence)
    for emotion, sentence in sp.close():
        emit(emotion, sentence)

    send_sync({"type": "speech_end"})
    return " ".join(sentences)


# State
recording = False
processing = False
greeted = False
audio_chunks = []
stream = None
ws_client = None
playback_done = threading.Event()


async def send(msg: dict):
    if ws_client:
        try:
            await ws_client.send_json(msg)
        except Exception:
            pass


def make_sender(loop):
    def send_sync(msg):
        asyncio.run_coroutine_threadsafe(send(msg), loop).result()
    return send_sync


def respond(user_text: str, loop):
    """LLM → sentence-streamed TTS → wait for browser playback → idle."""
    send_sync = make_sender(loop)
    send_sync({"type": "state", "value": "thinking"})
    playback_done.clear()
    reply = speak_stream(llm.ask_stream(user_text), send_sync)
    print(f"Bot: {reply}")
    send_sync({"type": "transcript", "role": "assistant", "text": reply})
    # Fallback timeout: generous, since the browser normally reports playback_done.
    playback_done.wait(timeout=120)
    send_sync({"type": "state", "value": "idle"})


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
                respond(text, loop)
            except Exception as e:
                print(f"[Error] {e}")
                asyncio.run_coroutine_threadsafe(send({"type": "state", "value": "idle"}), loop)
            finally:
                processing = False

        threading.Thread(target=process, daemon=True).start()


def greet(loop):
    global processing
    processing = True
    try:
        respond("(The user just opened the app. Give a short, natural greeting. Do not mention memory or context.)", loop)
    except Exception as e:
        print(f"[Error] {e}")
        asyncio.run_coroutine_threadsafe(send({"type": "state", "value": "idle"}), loop)
    finally:
        processing = False


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global ws_client, greeted
    await websocket.accept()
    ws_client = websocket
    loop = asyncio.get_event_loop()
    if not greeted:
        greeted = True
        threading.Thread(target=greet, args=(loop,), daemon=True).start()
    else:
        await send({"type": "state", "value": "idle"})
    try:
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")
            if action == "toggle":
                threading.Thread(target=handle_toggle, args=(loop,), daemon=True).start()
            elif action == "playback_done":
                playback_done.set()
            elif action == "shutdown":
                print("\nShutdown requested from UI...")
                llm.save_memory()
                os._exit(0)
    except WebSocketDisconnect:
        ws_client = None
        playback_done.set()


def open_browser():
    import time
    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    from kokoro_onnx import Kokoro
    print("Loading Kokoro TTS...")
    kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    print("Ready.\n")
    threading.Thread(target=open_browser, daemon=True).start()
    try:
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        print("\nExiting...")
        llm.save_memory()
