import os
import io
import json
import signal
import base64
import asyncio
import contextlib
from dotenv import load_dotenv
load_dotenv()
import threading
import webbrowser

import llm
import splitter
import fillers
import avatars
import backdrops
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import vad
import asr

KOKORO_MODEL = os.path.expanduser(os.getenv("KOKORO_MODEL", "~/models/kokoro/kokoro-v1.0.onnx"))
KOKORO_VOICES = os.path.expanduser(os.getenv("KOKORO_VOICES", "~/models/kokoro/voices-v1.0.bin"))
AVATAR = avatars.current()
KOKORO_VOICE = os.getenv("KOKORO_VOICE", AVATAR["voice"])
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", AVATAR["speed"]))


def apply_avatar(key: str) -> dict:
    """Switch avatar at runtime: persona, voice, filler cache. Returns the new profile."""
    global AVATAR, KOKORO_VOICE, KOKORO_SPEED
    AVATAR = avatars.set_current(key)
    KOKORO_VOICE = os.getenv("KOKORO_VOICE", AVATAR["voice"])
    KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", AVATAR["speed"]))
    llm.set_avatar()
    _filler_wavs.clear()
    return AVATAR
PORT = int(os.getenv("PORT", "8010"))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

SAMPLERATE = 16000

kokoro = None  # loaded in main()


def synthesize(text: str) -> bytes:
    """Text → 16-bit PCM wav bytes (Kokoro's native 24 kHz)."""
    samples, sample_rate = kokoro.create(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


_filler_wavs: dict[str, bytes] = {}


def speak_stream(events, send_sync, tts=None):
    """Consume LLM events, TTS each sentence and push it over the socket.

    `events` yields ("tool_calls", [names]) and ("delta", text) — see llm.ask_events.
    A cached filler is spoken as soon as a tool call is detected.
    Returns the full reply text (tag stripped, fillers excluded).
    `send_sync(dict)` must be thread-safe; `tts(text) -> wav bytes` defaults to Kokoro.
    """
    tts = tts or synthesize
    sp = splitter.SentenceSplitter()
    sentences = []
    started = False

    def emit(emotion, text, wav):
        nonlocal started
        if not started:
            started = True
            send_sync({"type": "state", "value": "speaking"})
        send_sync({
            "type": "speech",
            "emotion": emotion,
            "text": text,
            "wav": base64.b64encode(wav).decode("ascii"),
        })

    def say(emotion, sentence):
        sentences.append(sentence)
        emit(emotion, sentence, tts(sentence))

    def filler(tool_names):
        phrase = fillers.pick(tool_names[0] if tool_names else "default")
        if phrase not in _filler_wavs:
            _filler_wavs[phrase] = tts(phrase)
        emit("thinking", phrase, _filler_wavs[phrase])

    try:
        for kind, payload in events:
            if kind == "tool_calls":
                filler(payload)
            elif kind == "delta":
                for emotion, sentence in sp.feed(payload):
                    say(emotion, sentence)
        for emotion, sentence in sp.close():
            say(emotion, sentence)
    finally:
        if started:
            send_sync({"type": "speech_end"})

    return " ".join(sentences)


# State
processing = False
greeted = False
ws_client = None
playback_done = threading.Event()

# Hands-free / PTT capture state (single-client server, module-level is fine)
hands_free = bool(avatars.load_settings().get("hands_free", False))
endpointer = None  # vad.Endpointer, created lazily on first binary frame once hands-free is on
recorder = None  # vad.Recorder, created on "ptt start"
ptt_active = False
_hearing = False  # last "listening" value sent, to only send on change
PTT_MIN_SAMPLES = int(0.3 * SAMPLERATE)


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


def _wait_for_playback_then_idle(send_sync):
    """Common tail: wait for the browser to report playback_done (or time out), then go idle."""
    if ws_client is not None:
        # Fallback timeout: generous, since the browser normally reports playback_done.
        playback_done.wait(timeout=120)
    send_sync({"type": "state", "value": "idle"})


def respond(user_text: str, loop):
    """LLM → sentence-streamed TTS → wait for browser playback → idle."""
    send_sync = make_sender(loop)
    send_sync({"type": "state", "value": "thinking"})
    playback_done.clear()
    reply = speak_stream(llm.ask_events(user_text), send_sync)
    print(f"Bot: {reply}")
    send_sync({"type": "transcript", "role": "assistant", "text": reply})
    _wait_for_playback_then_idle(send_sync)


def say_canned(text: str, send_sync, tts=None):
    """Speak a canned (non-LLM) line: same state/speech/speech_end shape as speak_stream,
    plus a transcript message. Returns the clean (tag-stripped) text.
    """
    playback_done.clear()
    reply = speak_stream([("delta", text)], send_sync, tts=tts)
    send_sync({"type": "transcript", "role": "assistant", "text": reply})
    return reply


@contextlib.contextmanager
def _busy():
    """Hold the `processing` flag for one turn (blocks PTT/avatar switches, gates the VAD)."""
    global processing
    processing = True
    try:
        yield
    finally:
        processing = False


def _spawn(fn, *args):
    threading.Thread(target=fn, args=args, daemon=True).start()


def handle_utterance(audio, loop):
    """Transcribe one captured utterance (hands-free or PTT) and respond.

    Idle is always sent when no reply was produced (empty transcript or an exception).
    """
    with _busy():
        send_sync = make_sender(loop)
        responded = False
        try:
            send_sync({"type": "state", "value": "processing"})
            text = asr.transcribe(audio)
            if not text:
                return
            print(f"You: {text}")
            send_sync({"type": "transcript", "role": "user", "text": text})
            respond(text, loop)
            responded = True
        except Exception as e:
            print(f"[Error] {e}")
        finally:
            if not responded:
                send_sync({"type": "state", "value": "idle"})


def _canned_turn(text, loop, thinking):
    """Speak a canned line as a full turn: optional thinking state, TTS, record, idle."""
    with _busy():
        send_sync = make_sender(loop)
        try:
            if thinking:
                send_sync({"type": "state", "value": "thinking"})
            llm.record_assistant(say_canned(text, send_sync))
            _wait_for_playback_then_idle(send_sync)
        except Exception as e:
            print(f"[Error] {e}")
            send_sync({"type": "state", "value": "idle"})


def _switch_greet(loop):
    """Speak the newly-applied avatar's switch_greeting after a set_avatar action."""
    _canned_turn(AVATAR["switch_greeting"], loop, thinking=False)


def greet(loop):
    """Speak the avatar's greeting on the first page connect."""
    _canned_turn(AVATAR["greeting"], loop, thinking=True)


app = FastAPI()


@app.on_event("shutdown")
async def _on_shutdown():
    """Runs on graceful shutdown (SIGTERM/SIGINT/UI shutdown) — the place session memory is saved."""
    try:
        llm.save_memory()
    except Exception as e:
        print(f"[Memory error] {e}")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def current_backdrop() -> str:
    try:
        return backdrops.validate(avatars.load_settings().get("backdrop") or backdrops.DEFAULT)
    except ValueError:
        return backdrops.DEFAULT


@app.get("/api/config")
async def api_config():
    return {
        "avatar": AVATAR["key"], "name": AVATAR["name"], "avatars": avatars.listing(),
        "backdrop": current_backdrop(), "backdrops": backdrops.listing(),
        "hands_free": hands_free,
        "verbosity": llm.current_verbosity(),
    }


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def _handle_audio(data, loop):
    """One binary PCM frame from the browser: PTT recorder or hands-free endpointer."""
    global endpointer, _hearing
    if ptt_active:
        if recorder is not None:
            recorder.feed(data)
        return
    if not hands_free:
        return
    if endpointer is None:
        endpointer = vad.Endpointer(**vad.DEFAULTS)
    endpointer.gated = processing
    utterances = endpointer.feed(data)
    if endpointer.hearing != _hearing:
        _hearing = endpointer.hearing
        await send({"type": "listening", "value": "hearing" if _hearing else "idle"})
    for utt in utterances:
        _spawn(handle_utterance, utt, loop)


async def _action_ptt(msg, loop):
    global recorder, ptt_active
    value = msg.get("value")
    if value == "start":
        if not processing:
            recorder = vad.Recorder()
            ptt_active = True
            await send({"type": "state", "value": "recording"})
    elif value == "stop":
        ptt_active = False
        audio = recorder.stop() if recorder is not None else np.zeros(0, dtype=np.float32)
        if audio.size < PTT_MIN_SAMPLES:
            await send({"type": "state", "value": "idle"})
        else:
            _spawn(handle_utterance, audio, loop)


async def _action_set_hands_free(msg, loop):
    global hands_free, _hearing
    value = bool(msg.get("value"))
    avatars.save_setting("hands_free", value)
    hands_free = value
    if endpointer is not None:
        endpointer.gated = False
        endpointer.reset()
    _hearing = False
    await send({"type": "hands_free", "value": value})
    if not value:
        await send({"type": "listening", "value": "idle"})


async def _action_playback_done(msg, loop):
    playback_done.set()


async def _action_set_backdrop(msg, loop):
    try:
        key = backdrops.validate(msg.get("key", ""))
        avatars.save_setting("backdrop", key)
        await send({"type": "backdrop", "key": key})
    except ValueError as e:
        await send({"type": "error", "text": str(e)})


async def _action_set_avatar(msg, loop):
    global processing
    if processing:
        await send({"type": "error", "text": "Wait until the reply finishes."})
        return
    try:
        a = apply_avatar(msg.get("key", ""))
    except ValueError as e:
        await send({"type": "error", "text": str(e)})
        return
    await send({"type": "avatar", "key": a["key"], "name": a["name"]})
    processing = True  # claimed here so a second switch is refused deterministically
    _spawn(_switch_greet, loop)


async def _action_reload_characters(msg, loop):
    apply_avatar(avatars.reload()["key"])
    await send({
        "type": "characters_reloaded",
        "avatar": AVATAR["key"], "name": AVATAR["name"],
        "avatars": avatars.listing(),
    })


async def _action_set_verbosity(msg, loop):
    try:
        llm.set_verbosity(msg.get("value"))
        await send({"type": "verbosity", "value": llm.current_verbosity()})
    except ValueError as e:
        await send({"type": "error", "text": str(e)})


async def _action_shutdown(msg, loop):
    print("\nShutdown requested from UI...")
    os.kill(os.getpid(), signal.SIGTERM)  # graceful: triggers the shutdown hook


ACTIONS = {
    "ptt": _action_ptt,
    "set_hands_free": _action_set_hands_free,
    "playback_done": _action_playback_done,
    "set_backdrop": _action_set_backdrop,
    "set_avatar": _action_set_avatar,
    "reload_characters": _action_reload_characters,
    "set_verbosity": _action_set_verbosity,
    "shutdown": _action_shutdown,
}


async def _on_connect(loop):
    """Bring a freshly connected page up to date, and greet on the very first one."""
    global greeted
    playback_done.set()  # a new page cannot finish the previous page's playback
    # Replay what has been said so a reloaded page shows the conversation so far.
    for turn in llm.get_session_turns():
        await send({"type": "transcript", "role": turn["role"], "text": splitter.strip_tag(turn["content"])})
    await send({"type": "hands_free", "value": hands_free})
    if not greeted:
        greeted = True
        _spawn(greet, loop)
    else:
        await send({"type": "state", "value": "idle"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global ws_client
    await websocket.accept()
    ws_client = websocket
    loop = asyncio.get_event_loop()
    await _on_connect(loop)
    try:
        while True:
            frame = await websocket.receive()
            if frame.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(frame.get("code", 1000))
            if frame.get("bytes") is not None:
                await _handle_audio(np.frombuffer(frame["bytes"], dtype=np.int16), loop)
            elif frame.get("text") is not None:
                msg = json.loads(frame["text"])
                handler = ACTIONS.get(msg.get("action"))
                if handler:
                    await handler(msg, loop)
    except WebSocketDisconnect:
        if ws_client is websocket:
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
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
    print("\nExiting...")
