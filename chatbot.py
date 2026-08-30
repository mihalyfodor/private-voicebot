import os
import re
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
import memory
import wiki
import splitter
import sources
import fillers
import avatars
from tools import weather as weather_tool
import backdrops
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.datastructures import UploadFile
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
    """(Re)apply the avatar profile: persona, voice, filler cache. Returns the profile."""
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

    def emit(emotion, text, wav, filler=False):
        nonlocal started
        if not started:
            started = True
            send_sync({"type": "state", "value": "speaking"})
        msg = {
            "type": "speech",
            "emotion": emotion,
            "text": text,
            "wav": base64.b64encode(wav).decode("ascii"),
        }
        if filler:
            msg["filler"] = True   # the browser sparkles while a tool runs (PRD 27)
        send_sync(msg)

    def say(emotion, sentence):
        sentences.append(sentence)
        emit(emotion, sentence, tts(sentence))

    def filler(tool_names):
        phrase = fillers.pick(tool_names[0] if tool_names else "default")
        if phrase not in _filler_wavs:
            _filler_wavs[phrase] = tts(phrase)
        emit("thinking", phrase, _filler_wavs[phrase], filler=True)

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
#: the asyncio loop serving the websocket, so worker-thread code can push to the browser
_loop: asyncio.AbstractEventLoop | None = None


def _memory_msg(profile: dict) -> dict:
    """The `memory` message the panel re-renders itself from: profile + rendered block + count."""
    return {"type": "memory", **memory.snapshot(profile)}


def _wiki_msg() -> dict:
    """The `wiki` message the Wiki tab re-renders itself from: page list + lint + vault path."""
    wiki.ensure_vault()   # opening the tab is the first use that may create the vault
    return {"type": "wiki", "pages": wiki.pages(), "lint": wiki.lint(), "vault": wiki.WIKI_PATH}


def _wiki_page_msg(path: str) -> dict:
    """One whole page for the pane/editor, with its etag (PRD 28) and backlinks."""
    text = wiki.page_text(path)
    if text is None:
        return {"type": "wiki_page", "path": path, "text": f"no such page: {path}",
                "links": [], "etag": "", "backlinks": []}
    return {"type": "wiki_page", "path": path, "text": text, "links": wiki.links_in(text),
            "etag": wiki.etag(text), "backlinks": wiki.backlinks(path)}


def _panel_sender():
    """A thread-safe sender to the open panel, or None when there is no socket (or no loop).

    The worker-thread pushes below (memory snapshot, "saving", ingest progress) all go
    through it, so "no panel → silent no-op" is decided in one place.
    """
    sess, loop = session, _loop
    if sess is None or sess.websocket is None or loop is None or not loop.is_running():
        return None
    return make_sender(loop)


def push_memory(with_wiki: bool = False) -> None:
    """Push a fresh memory snapshot (and optionally the wiki) to an open panel."""
    send_sync = _panel_sender()
    if send_sync is None:
        return
    send_sync(_memory_msg(memory.load_profile()))
    if with_wiki:
        send_sync(_wiki_msg())


def _push_saving(value: bool) -> None:
    """Tell the UI a background memory save is running (it shows "saving memory…")."""
    send_sync = _panel_sender()
    if send_sync is not None:
        send_sync({"type": "memory_saving", "value": value})


def _periodic_memory(count: int) -> None:
    """Every N completed turns (on the controller's hook thread, not the turn worker): fold
    the session into the memory profile (memory.json)."""
    filed = wiki.applied_count()
    _push_saving(True)
    try:
        llm.save_memory()
    except Exception as e:
        print(f"[Memory error] {e}")
    finally:
        _push_saving(False)
    # an open panel refreshes itself after a background save; the Wiki tab only when the
    # save actually filed something, so a quiet session costs no extra traffic.
    push_memory(with_wiki=wiki.applied_count() != filed)


controller = TurnController(on_turn_complete=_periodic_memory)

greeted = False
hands_free = bool(avatars.load_settings().get("hands_free", False))

#: message types that carry the turn id (assistant transcripts only)
_TURN_TAGGED = {"state", "speech", "speech_end"}


def is_busy() -> bool:
    """True while a turn is queued or running."""
    return controller.busy


async def send(msg: dict):
    """Send to the current session's socket; the first failure drops the socket.

    A dead socket is forgotten right away (logged once) and the turn waiting on its
    playback is released, so the worker never sits through a timeout per message.
    """
    sess = session
    if sess is None or sess.websocket is None:
        return
    try:
        await sess.websocket.send_json(msg)
    except Exception as e:
        print(f"[WS] send failed, dropping socket: {e}")
        sess.websocket = None
        sess.playback_done.set()


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
        sess = session
        if sess is not None and sess.websocket is None:
            return  # socket already dropped (see `send`): nothing to wait for
        try:
            asyncio.run_coroutine_threadsafe(send(_tag(msg, turn)), loop).result(timeout=10)
        except Exception as e:
            print(f"[Send error] {e}")
    return send_sync


#: Every `error` message names the UI slot it belongs to (the client routes on `scope`, never on
#: which view happens to be open): wiki status slot, memory error line, sources head, the
#: composer status line for `turn` / `settings`, the re-ingest button for `intake`.
ERROR_SCOPES = ("wiki", "memory", "sources", "turn", "settings", "intake")


async def _error(scope: str, text: str, action: str | None = None) -> None:
    """The one shape a refusal comes back in: `{type: error, scope, text[, action]}`."""
    assert scope in ERROR_SCOPES, scope
    msg = {"type": "error", "scope": scope, "text": text}
    if action:
        msg["action"] = action
    await send(msg)


def _scope_of(action) -> str:
    """Which slot an unexpected exception from `action` belongs to (the catch-all in the WS loop)."""
    a = str(action or "")
    if a == "wiki_ingest":
        return "intake"
    if a.startswith("wiki") or a == "pull_wiki":
        return "wiki"
    if "memory" in a:
        return "memory"
    if "source" in a:
        return "sources"
    if a.startswith("set_") or a == "reload_characters":
        return "settings"
    return "turn"


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


# --- find / open pages from chat (docs/38-focus-context.md) ---------------------------
# Deterministic, not NLU: two phrasings, matched on the whole turn, typed or spoken.
_OPEN_PAGE = re.compile(
    r"^(?:please\s+)?(?:find|open|show(?:\s+me)?)\s+(?:the\s+|a\s+)?(?:(?:wiki\s+)?(?:page|note|source)\s+)?"
    r"(?:(?:about|on|for)\s+)?(?P<query>.+?)[\s?.!]*$", re.I)
_BACK_TO_PAGE = re.compile(
    r"^(?:(?:go|take me)\s+)?back\s+to\s+(?:that|the\s+last|the)\s+page[\s?.!]*$"
    r"|^what\s+were\s+we\s+looking\s+at[\s?.!]*$", re.I)


def page_command(text: str):
    """``("open", query)``, ``("back", None)`` or None for a turn that is not a page command."""
    text = " ".join(str(text or "").split())
    if _BACK_TO_PAGE.match(text):
        return ("back", None)
    m = _OPEN_PAGE.match(text)
    if m and m.group("query"):
        return ("open", m.group("query"))
    return None


def open_page_for(text: str, send_sync) -> str:
    """Open the page a find/open/go-back turn asks for and return the one-turn note for the model.

    Pushes `wiki_open {path}` and sets the focus; "" when the turn is not a page command.
    """
    cmd = page_command(text)
    if cmd is None:
        return ""
    kind, query = cmd
    if kind == "back":
        path, query = llm.last_page, "the last page"
    else:
        hits = wiki.search(query, 1)
        path = hits[0]["path"] if hits else None
    if not path:
        return llm.NO_MATCH_NOTE.format(query=query)
    title = wiki.page_title(path) or path
    send_sync({"type": "wiki_open", "path": path})
    llm.set_focus("sources" if path.startswith("sources/") else "wiki", path)
    return llm.OPENED_NOTE.format(title=title)


def respond(sess: Session, turn: int, user_text: str, send_sync, origin: str = "user"):
    """LLM → sentence-streamed TTS → wait for browser playback → idle.

    The one reply path: a typed turn and a transcribed one differ only in `origin`, which
    tags the session log line ("user" spoken, "typed" from the composer); the conversation,
    tools and memory see them identically. Always ends in the idle state, errors included.
    A find/open/go-back turn first opens the page (PRD 38), then reaches the model with a note.
    """
    try:
        send_sync({"type": "state", "value": "thinking"})
        _arm_playback(sess, turn)
        note = open_page_for(user_text, send_sync)
        reply = speak_stream(llm.ask_events(user_text, origin=origin, note=note), send_sync)
        print(f"Bot: {reply}")
        send_sync({"type": "transcript", "role": "assistant", "text": reply})
        _wait_for_playback_then_idle(sess, send_sync)
    except Exception as e:
        print(f"[Error] {e}")
        send_sync({"type": "state", "value": "idle"})


def say_canned(text: str, send_sync, tts=None):
    """Speak a canned (non-LLM) line: same state/speech/speech_end shape as speak_stream,
    plus a transcript message. Returns the clean (tag-stripped) text.
    """
    reply = speak_stream([("delta", text)], send_sync, tts=tts)
    send_sync({"type": "transcript", "role": "assistant", "text": reply})
    return reply


def handle_text(sess: Session, turn: int, text: str, loop):
    """One typed turn (PRD 24): the transcript path without the ASR.

    The browser already rendered the user's line, so no user transcript is echoed back.
    """
    print(f"You (typed): {text}")
    respond(sess, turn, text, make_sender(loop, turn), origin="typed")


def handle_utterance(sess: Session, turn: int, audio, loop):
    """Transcribe one captured utterance (hands-free) and respond.

    Idle is sent here when the transcript is empty or the ASR raised; once there is text,
    `respond` owns the rest of the turn (and its own idle).
    """
    send_sync = make_sender(loop, turn)
    try:
        send_sync({"type": "state", "value": "processing"})
        text = asr.transcribe(audio, prompt=memory.asr_prompt())
    except Exception as e:
        print(f"[Error] {e}")
        text = ""
    if not text:
        send_sync({"type": "state", "value": "idle"})
        return
    print(f"You: {text}")
    send_sync({"type": "transcript", "role": "user", "text": text})
    respond(sess, turn, text, send_sync)


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


@app.middleware("http")
async def _no_static_cache(request: Request, call_next):
    """Static files are served fresh: fetch-loaded model json / backdrops must not go stale."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def current_backdrop() -> str:
    try:
        return backdrops.validate(avatars.load_settings().get("backdrop") or backdrops.DEFAULT)
    except ValueError:
        return backdrops.DEFAULT


#: Shown read-only in the Settings view — the command that takes a screenshot of the app.
SCREENSHOT_CMD = ".venv/bin/python scripts/screenshot.py /tmp/shot.png"


@app.get("/api/config")
async def api_config():
    """Everything the UI needs at boot: what to render, what is switched on, what to display."""
    return {
        # who is on stage (Wanko only since PRD 23) and the backdrop behind her
        "avatar": AVATAR["key"], "name": AVATAR["name"],
        "backdrop": current_backdrop(), "backdrops": backdrops.listing(),
        # session state the UI mirrors
        "hands_free": hands_free,
        "verbosity": llm.current_verbosity(),
        # read-only facts shown in the Settings view
        "stale_days": dict(memory.STALE_DAYS),
        "calendar_horizon_days": sources.calendar_horizon_days(),
        "recall_min_score": wiki.recall_min_score(),
        "location": weather_tool.location(),
        "vault": wiki.WIKI_PATH,
        "screenshot_cmd": SCREENSHOT_CMD,
    }


INGEST_KINDS = ("file", "paste_text", "paste_image")


def _push_ingest_progress(filename: str, stage: str, i: int, n: int) -> None:
    """Tell the intake popup which stage item `i` of `n` is in (PRD 36); no panel → no-op."""
    send_sync = _panel_sender()
    if send_sync is not None:
        send_sync({"type": "ingest_progress", "filename": filename, "stage": stage, "i": i, "n": n})


def _file_one(item: dict, on_stage=None) -> dict:
    """One popup item -> `sources`. Runs on a worker thread: it does IO, git and two LLM calls."""
    if item["kind"] == "paste_text":
        return sources.ingest_note(item["text"], title=item["title"], hint=item["hint"],
                                   page=item["page"], on_stage=on_stage)
    return sources.ingest_upload(item["data"], item["filename"], kind_hint=item["kind"],
                                 title=item["title"], hint=item["hint"], page=item["page"],
                                 on_stage=on_stage)


def _ingest_all(items: list) -> dict:
    """File every item of one popup; a failure on one is reported, the rest still land.

    Each stage of each item is pushed to the popup as `ingest_progress` (PRD 36).
    """
    filed, errors = [], []
    for i, item in enumerate(items, 1):
        name = item.get("filename") or "text"
        try:
            filed.append(_file_one(
                item, lambda stage, name=name, i=i: _push_ingest_progress(name, stage, i, len(items))))
        except sources.IngestError as exc:
            errors.append(exc.message)
        except Exception as exc:  # noqa: BLE001 — one bad drop must not 500 the popup
            print(f"[Ingest] {item.get('filename') or 'text'}: {exc}")
            errors.append(f"{item.get('filename') or 'the text'}: {exc}")
    return {"filed": filed, "errors": errors}


#: pasted text travels as a plain form field; Starlette caps those at 1 MB unless told otherwise
INGEST_MAX_PART = sources.MAX_TEXT_BYTES + 64 * 1024


@app.post("/ingest")
async def ingest(request: Request):
    """The intake popup's File button — the only HTTP write path (docs/20-raw-ingestion.md).

    Nothing is filed without `confirm=true`: the popup is the only sender, and the guard is what
    keeps a future "helpful" auto-path from writing to the vault behind the user's back.
    The form is parsed by hand so a big pasted text is not refused by the framework's 1 MB
    per-part default before `sources` gets to apply its own cap.
    """
    if not _origin_allowed(request.headers.get("origin")):
        raise HTTPException(status_code=403, detail="bad origin")
    form = await request.form(max_part_size=INGEST_MAX_PART)
    confirm = str(form.get("confirm") or "")
    kind = str(form.get("kind") or "file")
    title, hint, page = (str(form.get(k) or "") for k in ("title", "hint", "page"))
    text = str(form.get("text") or "")
    file = [f for f in form.getlist("file") if isinstance(f, UploadFile)]
    if str(confirm).lower() != "true":
        raise HTTPException(status_code=400, detail="ingest needs an explicit confirm=true")
    if kind not in INGEST_KINDS:
        raise HTTPException(status_code=400, detail=f"unknown kind {kind!r}")

    fields = {"kind": kind, "title": title, "hint": hint, "page": page}
    items = []
    try:
        for upload in file or []:
            data = await upload.read()
            # refuse type/size here as well as in sources, so a bad drop costs no worker thread
            sources.check_upload(upload.filename or "", len(data), kind)
            items.append({**fields, "data": data, "filename": upload.filename or "drop",
                          "text": ""})
        if not items and kind == "paste_text":
            items.append({**fields, "data": b"", "filename": "",
                          "text": sources.check_note(text)})
    except sources.IngestError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc
    if not items:
        raise HTTPException(status_code=400, detail="nothing to file")

    before = memory.snapshot(memory.load_profile())
    result = await asyncio.to_thread(_ingest_all, items)
    if any(not f.get("already") for f in result["filed"]):
        await _send_wiki()
    if memory.snapshot(memory.load_profile()) != before:
        await send(_memory_msg(memory.load_profile()))
    return result


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def _handle_audio(sess: Session, data, loop):
    """One binary PCM frame from the browser, fed to the hands-free endpointer."""
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


async def _action_text_turn(sess, msg, loop):
    """Composer input (PRD 24): queued exactly like a finished utterance."""
    text = (msg.get("text") or "").strip()
    if text:
        submit_turn(handle_text, text, loop)


async def _action_set_focus(sess, msg, loop):
    """`{view, path?}` from the client whenever the view or the open page changes (PRD 38)."""
    llm.set_focus(msg.get("view"), msg.get("path"))


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


async def _action_reload_characters(sess, msg, loop):
    """The user edited characters/wanko.yaml: re-read the card, rebuild persona/voice/fillers."""
    apply_avatar(avatars.reload()["key"])
    await send({"type": "characters_reloaded", "avatar": AVATAR["key"], "name": AVATAR["name"]})


async def _action_set_verbosity(sess, msg, loop):
    llm.set_verbosity(msg.get("value"))
    await send({"type": "verbosity", "value": llm.current_verbosity()})


#: Settings the UI may write straight to settings.json via `set_setting`, and the type each
#: one must have. Deliberately tiny: everything with side effects keeps its own action.
# key -> expected type. Empty since PRD 22b removed `panel_pinned`; the action stays so the
# next persisted UI preference is one line here instead of a new action.
#: `calendar_horizon_days` — how far a recurring invite is expanded into meeting stubs (PRD 31)
#: `recall_min_score` — the BM25-per-word floor a source passage must clear to be quoted (PRD 32)
SETTABLE: dict[str, type] = {"calendar_horizon_days": int, "recall_min_score": float}


async def _action_set_location(sess, msg, loop):
    """The browser's position (PRD 35): validated, rounded to ~100 m, saved as `location`."""
    try:
        lat, lon = float(msg.get("lat")), float(msg.get("lon"))
    except (TypeError, ValueError):
        await _error("settings", "location needs numeric lat/lon")
        return
    tz = str(msg.get("tz") or "").strip()
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or not tz:
        await _error("settings", "location out of range or missing tz")
        return
    saved = avatars.load_settings().get("location")
    if weather_tool.needs_geocode(saved, lat, lon):
        name = await asyncio.to_thread(weather_tool.reverse_geocode, lat, lon)   # "" when Nominatim fails
    else:
        name = saved["name"]
    value = {"lat": round(lat, 3), "lon": round(lon, 3), "tz": tz, "source": "browser", "name": name}
    avatars.save_setting("location", value)
    await asyncio.to_thread(llm.refresh_context)   # the next turn knows the place
    await send({"type": "setting", "key": "location", "value": weather_tool.location()})


async def _action_set_setting(sess, msg, loop):
    """Persist one whitelisted UI setting and echo it back; unknown keys and wrong types are refused.

    The whole point is that a stored UI preference does not need an action of its own — anything
    with a side effect (avatar, verbosity, hands-free) still gets a dedicated handler.
    """
    key = msg.get("key")
    kind = SETTABLE.get(key)
    value = msg.get("value")
    if kind is None:
        await _error("settings", f"Unknown setting {key!r}")
        return
    ok = isinstance(value, kind) or (kind is float and isinstance(value, int) and not isinstance(value, bool))
    if not ok:
        await _error("settings", f"{key} needs a {kind.__name__}")
        return
    value = kind(value)
    avatars.save_setting(key, value)
    await send({"type": "setting", "key": key, "value": value})


def _ingest_now() -> None:
    """Re-ingest the session on the worker/executor: the same save path a turn would take."""
    try:
        llm.save_memory()
    except Exception as e:
        print(f"[Memory error] {e}")


async def _save_session_memory() -> None:
    """Fold the session into memory.json off the loop, then push the fresh snapshot.

    The Wiki tab is only refreshed when the save actually filed pages — same rule as the
    periodic save in `_periodic_memory`.
    """
    filed = wiki.applied_count()
    await asyncio.to_thread(_ingest_now)
    await send(_memory_msg(await asyncio.to_thread(memory.load_profile)))
    if wiki.applied_count() != filed:
        await _send_wiki()


async def _action_get_memory(sess, msg, loop):
    """The memory panel asking for the current profile (on open, and after every reconnect).

    Opening the tab is also a save trigger: the current snapshot goes out first so the panel
    never waits on the LLM, then — if this session has turns no save has folded in yet and no
    turn is running — the save runs and a second, fresh snapshot follows. While a turn is in
    flight nothing is saved here; the periodic save catches up.
    """
    await send(_memory_msg(await asyncio.to_thread(memory.load_profile)))
    if llm.has_unsaved_turns() and not controller.busy:
        await _save_session_memory()


async def _action_save_memory(sess, msg, loop):
    """The panel's "Save now": the same save as the tab-open trigger, but unconditional."""
    if controller.busy:
        await _error("memory", "Wait until the reply finishes.")
        return
    await send(_memory_msg(await asyncio.to_thread(memory.load_profile)))
    await _save_session_memory()


async def _send_memory_error(path, reason) -> None:
    """The one shape a rejected memory/wiki edit comes back in; the panel puts it on the row."""
    await send({"type": "memory_error", "path": path, "reason": reason})


def _sources_msg() -> dict:
    """The `sources` message the Sources view (and its rail badge) re-renders from (PRD 25)."""
    return {"type": "sources", "sources": sources.list_sources()}


async def _send_wiki() -> None:
    """Push a fresh `wiki` message (page list + lint); the Wiki tab re-renders from it.

    A `sources` message follows: every path that changes the wiki may have filed a source, and
    the rail badge counts what arrived while the user was elsewhere.
    """
    await asyncio.to_thread(llm.refresh_context)   # the next turn sees the new/changed pages
    await send(await asyncio.to_thread(_wiki_msg))
    await send(await asyncio.to_thread(_sources_msg))


async def _action_get_sources(sess, msg, loop):
    """The Sources view asking for the inbox list (on open, and after every reconnect)."""
    await send(await asyncio.to_thread(_sources_msg))


async def _source_op(fn, *args):
    """One inbox action: run it off the loop, then refresh the list — or report the refusal."""
    try:
        await asyncio.to_thread(fn, *args)
    except sources.IngestError as exc:
        await _error("sources", exc.message)
        return
    await _send_wiki()


async def _action_reingest_source(sess, msg, loop):
    """`{sha}` → the source page rewritten from its raw file; the open page is re-read via `wiki`."""
    await _source_op(sources.reingest, str(msg.get("sha") or ""))


async def _action_forget_source(sess, msg, loop):
    """`{sha, confirm: true}` → page, raw file, manifest line and its notes gone; one commit."""
    if msg.get("confirm") is not True:
        await _error("sources", "forget needs an explicit confirm")
        return
    await _source_op(sources.forget, str(msg.get("sha") or ""))


async def _action_memory_op(sess, msg, loop):
    """One hand edit from the memory panel: {op, path, value, reason?} → memory | memory_error.

    `memory.edit` does file IO under the memory lock (and may wait for a background save), so it
    runs off the event loop.
    """
    op = {"op": msg.get("op"), "path": msg.get("path"),
          "value": msg.get("value"), "reason": msg.get("reason")}
    profile, reason = await asyncio.to_thread(memory.edit, op)
    if reason:
        await _send_memory_error(op["path"], reason)
        return
    await send(_memory_msg(profile))


async def _action_memory_ops(sess, msg, loop):
    """A batch of hand edits from the memory panel: {ops: [{op, path, value, reason?}, ...]}.

    All-or-nothing (the drag-and-drop moves are one ADD + one DELETE): on the first bad op
    nothing is written and the panel gets a `memory_error` naming that op's path.
    """
    ops = msg.get("ops")
    profile, error = await asyncio.to_thread(memory.edit_many, ops)
    if error:
        await _send_memory_error(error.get("path"), error.get("reason"))
        return
    await send(_memory_msg(profile))


async def _action_memory_reset(sess, msg, loop):
    """The panel's "Forget everything": the profile is emptied and the reset is audited."""
    profile = await asyncio.to_thread(memory.reset)
    await send(_memory_msg(profile))


async def _action_get_wiki(sess, msg, loop):
    """The Wiki tab asking for the page list (on open, and after every reconnect)."""
    await _send_wiki()


async def _action_read_wiki_page(sess, msg, loop):
    """One page for the right-hand pane, with its wikilink targets for the clickable links."""
    await send(await asyncio.to_thread(_wiki_page_msg, str(msg.get("path") or "")))


async def _action_wiki_note(sess, msg, loop):
    """The "add note" box: append one dated bullet under `## Notes`, then refresh both panes."""
    path = str(msg.get("path") or "")
    op = {"op": "append_note", "path": path, "text": msg.get("text"),
          "source": f"user note {memory._today_iso()}"}
    applied, rejected = await asyncio.to_thread(wiki.apply_page_ops, [op], op["source"])
    if not applied:
        reason = rejected[0]["reason"] if rejected else "nothing to add"
        await _error("wiki", reason)
        return
    await send(await asyncio.to_thread(_wiki_page_msg, path))
    await _send_wiki()


async def _wiki_user_op(fn, *args, reply_path=None):
    """One editor write (PRD 28): refresh the page (when `reply_path`) and the list, or `error`."""
    result, reason = await asyncio.to_thread(fn, *args)
    if reason:
        await _error("wiki", reason)
        return
    if reply_path is not None:
        await send(await asyncio.to_thread(_wiki_page_msg, reply_path(result)))
    await _send_wiki()


async def _action_wiki_save(sess, msg, loop):
    """⌘S: `{path, text, etag}` → `wiki_page` (new etag) + `wiki`, or `wiki_conflict` on a stale etag."""
    path = str(msg.get("path") or "")
    result, reason = await asyncio.to_thread(wiki.save_page, path, msg.get("text"), msg.get("etag"))
    if reason:
        await _error("wiki", reason)
    elif result.get("conflict"):
        await send({"type": "wiki_conflict", "path": path, "text": result["text"], "etag": result["etag"]})
    else:
        await send(await asyncio.to_thread(_wiki_page_msg, path))
        await _send_wiki()


async def _action_wiki_create(sess, msg, loop):
    """A missing link, "Today" or "New page": `{path}` → the new page + list."""
    await _wiki_user_op(wiki.create_page, str(msg.get("path") or ""), reply_path=lambda p: p)


async def _action_wiki_rename(sess, msg, loop):
    """`{path, new_path}` → the moved page (links rewritten) + list."""
    await _wiki_user_op(wiki.rename_page, str(msg.get("path") or ""), str(msg.get("new_path") or ""),
                        reply_path=lambda r: r["path"])


async def _action_wiki_delete(sess, msg, loop):
    """`{path}` → the list without it (lint names the links that broke)."""
    await _wiki_user_op(wiki.delete_page, str(msg.get("path") or ""))


async def _action_wiki_search(sess, msg, loop):
    """The list's search box: `{query}` → `{type: wiki_search, query, hits}` (BM25, same as the tool)."""
    query = str(msg.get("query") or "")
    hits = await asyncio.to_thread(wiki.search, query, 20) if query.strip() else []
    await send({"type": "wiki_search", "query": query, "hits": hits})


async def _action_wiki_ingest(sess, msg, loop):
    """"Re-ingest last session": run the memory save (which includes wiki ingest) on demand."""
    await asyncio.to_thread(_ingest_now)
    await send(_memory_msg(memory.load_profile()))
    await _send_wiki()


async def _bridge(move, *args):
    """One memory <-> wiki move (PRD 19), then refresh both views — or the error on its row."""
    profile, error = await asyncio.to_thread(move, *args)
    if error:
        await _send_memory_error(error.get("path"), error.get("reason"))
        return
    await send(_memory_msg(profile))
    await _send_wiki()


async def _action_archive_memory(sess, msg, loop):
    """Drag a memory row onto the Wiki tab: file the entry, then mark it stale."""
    await _bridge(memory.archive, str(msg.get("section") or ""), msg.get("index"))


async def _action_pull_wiki(sess, msg, loop):
    """Drag a wiki page onto a memory section: recreate a live entry from the page."""
    await _bridge(memory.pull, str(msg.get("path") or ""), str(msg.get("section") or ""))


async def _action_shutdown(sess, msg, loop):
    print("\nShutdown requested from UI...")
    os.kill(os.getpid(), signal.SIGTERM)  # graceful: triggers the shutdown hook


ACTIONS = {
    "text_turn": _action_text_turn,
    "set_focus": _action_set_focus,
    "set_hands_free": _action_set_hands_free,
    "playback_done": _action_playback_done,
    "playback_blocked": _action_playback_blocked,
    "set_backdrop": _action_set_backdrop,
    "reload_characters": _action_reload_characters,
    "set_verbosity": _action_set_verbosity,
    "set_setting": _action_set_setting,
    "set_location": _action_set_location,
    "get_memory": _action_get_memory,
    "save_memory": _action_save_memory,
    "memory_op": _action_memory_op,
    "memory_ops": _action_memory_ops,
    "memory_reset": _action_memory_reset,
    "get_wiki": _action_get_wiki,
    "read_wiki_page": _action_read_wiki_page,
    "wiki_note": _action_wiki_note,
    "wiki_ingest": _action_wiki_ingest,
    "wiki_save": _action_wiki_save,
    "wiki_create": _action_wiki_create,
    "wiki_rename": _action_wiki_rename,
    "wiki_delete": _action_wiki_delete,
    "wiki_search": _action_wiki_search,
    "get_sources": _action_get_sources,
    "reingest_source": _action_reingest_source,
    "forget_source": _action_forget_source,
    "archive_memory": _action_archive_memory,
    "pull_wiki": _action_pull_wiki,
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
    global session, _loop
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
    loop = _loop = asyncio.get_event_loop()
    await _on_connect(sess, loop)
    try:
        while True:
            frame = await websocket.receive()
            if frame.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(frame.get("code", 1000))
            msg = {}
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
                await _error(_scope_of(msg.get("action")), str(e), action=msg.get("action"))
    except WebSocketDisconnect:
        pass
    finally:
        sess.websocket = None
        sess.playback_done.set()  # never leave a turn blocked on a dead socket
        if session is sess:
            session = None


def env_flag(name: str, default: str = "1") -> bool:
    """Env var as a boolean: `0`, `false`, `no` (any case) are off, everything else is on."""
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no")


# Off with OPEN_BROWSER=0: a restart during development should not steal focus.
OPEN_BROWSER = env_flag("OPEN_BROWSER")


def open_browser():
    import time
    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    from kokoro_onnx import Kokoro
    print("Loading Kokoro TTS...")
    kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    print("Ready.\n")
    if OPEN_BROWSER:
        threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    print("\nExiting...")
