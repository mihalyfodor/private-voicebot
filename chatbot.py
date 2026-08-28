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
from session import Session, TurnController

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
HOST = os.getenv("HOST", "127.0.0.1")  # loopback by default; set HOST=0.0.0.0 to expose on the LAN
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


# ---------------------------------------------------------------- state
# One connection at a time (single-user app): `session` is the live Session,
# `controller` serialises turns on its own worker thread. See docs/12-turn-controller.md.
session: Session | None = None


def _periodic_memory(count: int) -> None:
    """Called by the worker thread every N completed turns: fold the session into the memory profile (memory.json)."""
    try:
        llm.save_memory()
    except Exception as e:
        print(f"[Memory error] {e}")


controller = TurnController(on_turn_complete=_periodic_memory)

greeted = False
hands_free = bool(avatars.load_settings().get("hands_free", False))
PTT_MIN_SAMPLES = int(0.3 * SAMPLERATE)

#: message types that carry the turn id (assistant transcripts only)
_TURN_TAGGED = {"state", "speech", "speech_end"}


def is_busy() -> bool:
    """True while a turn is queued or running."""
    return controller.busy


async def send(msg: dict):
    sess = session
    if sess is not None and sess.websocket is not None:
        try:
            await sess.websocket.send_json(msg)
        except Exception:
            pass


async def send_state(sess: Session, value: str) -> None:
    """Send a state message tagged with the session's latest turn id (event-loop side).

    The worker thread uses `make_sender` instead, which tags with the turn it is running.
    """
    await send({"type": "state", "value": value, "turn": sess.turn_id})


def _tag(msg: dict, turn: int | None) -> dict:
    if turn is None:
        return msg
    kind = msg.get("type")
    if kind in _TURN_TAGGED or (kind == "transcript" and msg.get("role") == "assistant"):
        return {**msg, "turn": turn}
    return msg


def make_sender(loop, turn: int | None = None):
    """Thread-safe sender for the worker thread; tags turn-bearing messages with `turn`."""
    def send_sync(msg):
        try:
            asyncio.run_coroutine_threadsafe(send(_tag(msg, turn)), loop).result(timeout=10)
        except Exception as e:
            print(f"[Send error] {e}")
    return send_sync


def submit_turn(fn, *args) -> int | None:
    """Allocate a turn id on the current session and queue `fn(session, turn, *args)`."""
    sess = session
    if sess is None:
        return None
    turn = sess.next_turn()
    controller.submit(fn, sess, turn, *args)
    return turn


def _turn_matches(sess: Session, turn) -> bool:
    """Is a client echo about the turn we are waiting on?

    True when the ids match, when we are not waiting on any turn, or when the client
    omitted the id altogether (older clients that do not echo it).
    """
    return turn is None or sess.expected_turn is None or turn == sess.expected_turn


# ---------------------------------------------------------------- turns


def _arm_playback(sess: Session, turn: int) -> None:
    """Start waiting for `turn`'s playback echo; call before the first speech is sent."""
    sess.playback_done.clear()
    sess.expected_turn = turn


def _wait_for_playback_then_idle(sess: Session, send_sync):
    """Common tail: wait for the browser to report playback_done (or time out), then go idle."""
    if sess.websocket is not None:
        # Fallback timeout: generous, since the browser normally reports playback_done.
        sess.playback_done.wait(timeout=120)
    send_sync({"type": "state", "value": "idle"})


def respond(sess: Session, turn: int, user_text: str, send_sync):
    """LLM → sentence-streamed TTS → wait for browser playback → idle."""
    send_sync({"type": "state", "value": "thinking"})
    _arm_playback(sess, turn)
    reply = speak_stream(llm.ask_events(user_text), send_sync)
    print(f"Bot: {reply}")
    send_sync({"type": "transcript", "role": "assistant", "text": reply})
    _wait_for_playback_then_idle(sess, send_sync)


def say_canned(text: str, send_sync, tts=None):
    """Speak a canned (non-LLM) line: same state/speech/speech_end shape as speak_stream,
    plus a transcript message. Returns the clean (tag-stripped) text.
    """
    reply = speak_stream([("delta", text)], send_sync, tts=tts)
    send_sync({"type": "transcript", "role": "assistant", "text": reply})
    return reply


def handle_utterance(sess: Session, turn: int, audio, loop):
    """Transcribe one captured utterance (hands-free or PTT) and respond.

    Idle is always sent when no reply was produced (empty transcript or an exception).
    """
    send_sync = make_sender(loop, turn)
    responded = False
    try:
        send_sync({"type": "state", "value": "processing"})
        text = asr.transcribe(audio)
        if not text:
            return
        print(f"You: {text}")
        send_sync({"type": "transcript", "role": "user", "text": text})
        respond(sess, turn, text, send_sync)
        responded = True
    except Exception as e:
        print(f"[Error] {e}")
    finally:
        if not responded:
            send_sync({"type": "state", "value": "idle"})


def canned_turn(sess: Session, turn: int, loop, text: str, thinking: bool):
    """Speak a canned line as a full turn: optional thinking state, TTS, record, idle."""
    send_sync = make_sender(loop, turn)
    try:
        if thinking:
            send_sync({"type": "state", "value": "thinking"})
        _arm_playback(sess, turn)
        llm.record_assistant(say_canned(text, send_sync))
        _wait_for_playback_then_idle(sess, send_sync)
    except Exception as e:
        print(f"[Error] {e}")
        send_sync({"type": "state", "value": "idle"})


def switch_greet(sess: Session, turn: int, loop):
    """Speak the newly-applied avatar's switch_greeting after a set_avatar action."""
    canned_turn(sess, turn, loop, AVATAR["switch_greeting"], thinking=False)


def greet(sess: Session, turn: int, loop):
    """Speak the avatar's greeting on the first page connect."""
    canned_turn(sess, turn, loop, AVATAR["greeting"], thinking=True)


# ---------------------------------------------------------------- app


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


async def _handle_audio(sess: Session, data, loop):
    """One binary PCM frame from the browser: PTT recorder or hands-free endpointer."""
    if sess.ptt_active:
        if sess.recorder is not None:
            sess.recorder.feed(data)
        return
    if not hands_free:
        return
    if sess.endpointer is None:
        sess.endpointer = vad.Endpointer(**vad.DEFAULTS)
    # Gated synchronously, before any await: no window where a turn is running
    # but the endpointer is still open.
    sess.endpointer.gated = controller.busy
    utterances = await loop.run_in_executor(None, sess.endpointer.feed, data)
    if sess.endpointer.hearing != sess.hearing:
        sess.hearing = sess.endpointer.hearing
        await send({"type": "listening", "value": "hearing" if sess.hearing else "idle"})
    for utt in utterances:
        submit_turn(handle_utterance, utt, loop)


async def _action_ptt(sess, msg, loop):
    value = msg.get("value")
    if value == "start":
        if not controller.busy:
            sess.recorder = vad.Recorder()
            sess.ptt_active = True
            await send_state(sess, "recording")
    elif value == "stop":
        if not sess.ptt_active:
            return  # stray stop (e.g. key-up after a refused start): no state change
        sess.ptt_active = False
        recorder, sess.recorder = sess.recorder, None
        audio = recorder.stop() if recorder is not None else np.zeros(0, dtype=np.float32)
        if audio.size < PTT_MIN_SAMPLES:
            await send_state(sess, "idle")
        else:
            submit_turn(handle_utterance, audio, loop)


async def _action_set_hands_free(sess, msg, loop):
    global hands_free
    value = bool(msg.get("value"))
    avatars.save_setting("hands_free", value)
    hands_free = value
    if sess.endpointer is not None:
        sess.endpointer.gated = False
        sess.endpointer.reset()
    sess.hearing = False
    await send({"type": "hands_free", "value": value})
    if not value:
        await send({"type": "listening", "value": "idle"})


async def _action_playback_done(sess, msg, loop):
    """The browser is done with a turn's audio: it finished playing it, or (as
    `playback_blocked`) could not autoplay it and will play it after the next gesture.
    Either way we stop waiting and go idle. Stale (mismatched) turns are ignored.
    """
    if _turn_matches(sess, msg.get("turn")):
        sess.playback_done.set()


#: same handling: both mean "stop waiting for this turn's playback"
_action_playback_blocked = _action_playback_done


async def _action_set_backdrop(sess, msg, loop):
    key = backdrops.validate(msg.get("key", ""))
    avatars.save_setting("backdrop", key)
    await send({"type": "backdrop", "key": key})


async def _action_set_avatar(sess, msg, loop):
    if controller.busy:
        await send({"type": "error", "text": "Wait until the reply finishes."})
        return
    a = apply_avatar(msg.get("key", ""))
    await send({"type": "avatar", "key": a["key"], "name": a["name"]})
    # submit marks the controller busy synchronously, so a second switch is refused.
    submit_turn(switch_greet, loop)


async def _action_reload_characters(sess, msg, loop):
    apply_avatar(avatars.reload()["key"])
    await send({
        "type": "characters_reloaded",
        "avatar": AVATAR["key"], "name": AVATAR["name"],
        "avatars": avatars.listing(),
    })


async def _action_set_verbosity(sess, msg, loop):
    llm.set_verbosity(msg.get("value"))
    await send({"type": "verbosity", "value": llm.current_verbosity()})


async def _action_shutdown(sess, msg, loop):
    print("\nShutdown requested from UI...")
    os.kill(os.getpid(), signal.SIGTERM)  # graceful: triggers the shutdown hook


ACTIONS = {
    "ptt": _action_ptt,
    "set_hands_free": _action_set_hands_free,
    "playback_done": _action_playback_done,
    "playback_blocked": _action_playback_blocked,
    "set_backdrop": _action_set_backdrop,
    "set_avatar": _action_set_avatar,
    "reload_characters": _action_reload_characters,
    "set_verbosity": _action_set_verbosity,
    "shutdown": _action_shutdown,
}


async def _on_connect(sess: Session, loop):
    """Bring a freshly connected page up to date, and greet on the very first one."""
    global greeted
    # Replay what has been said so a reloaded page shows the conversation so far.
    for turn in llm.get_session_turns():
        await send({"type": "transcript", "role": turn["role"], "text": splitter.strip_tag(turn["content"])})
    await send({"type": "hands_free", "value": hands_free})
    if not greeted:
        greeted = True
        submit_turn(greet, loop)
    else:
        await send_state(sess, "idle")


def _origin_allowed(origin: str | None) -> bool:
    """Non-browser clients send no Origin; browsers must be on our own loopback port."""
    if origin is None:
        return True
    return origin in {f"http://localhost:{PORT}", f"http://127.0.0.1:{PORT}"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global session
    if not _origin_allowed(websocket.headers.get("origin")):
        print(f"[WS] refused origin {websocket.headers.get('origin')!r}")
        await websocket.close(code=4003)
        return
    await websocket.accept()
    previous, session = session, Session(websocket)
    sess = session
    if previous is not None:
        # A new page takes over: release anything waiting on the old one, then evict it.
        old_ws, previous.websocket = previous.websocket, None
        previous.playback_done.set()
        if old_ws is not None:
            with contextlib.suppress(Exception):
                await old_ws.close(code=4000)
    loop = asyncio.get_event_loop()
    await _on_connect(sess, loop)
    try:
        while True:
            frame = await websocket.receive()
            if frame.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(frame.get("code", 1000))
            try:
                if frame.get("bytes") is not None:
                    await _handle_audio(sess, np.frombuffer(frame["bytes"], dtype=np.int16), loop)
                elif frame.get("text") is not None:
                    msg = json.loads(frame["text"])
                    handler = ACTIONS.get(msg.get("action"))
                    if handler:
                        await handler(sess, msg, loop)
            except WebSocketDisconnect:
                raise
            except Exception as e:
                print(f"[WS error] {e}")
                await send({"type": "error", "text": str(e)})
    except WebSocketDisconnect:
        pass
    finally:
        sess.websocket = None
        sess.playback_done.set()  # never leave a turn blocked on a dead socket
        if session is sess:
            session = None


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
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    print("\nExiting...")
