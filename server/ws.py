"""The /ws endpoint, the `ACTIONS` table (every message the browser may send), reconnect replay and the origin check (PRD 50 step 11, from chatbot.py)."""
import os
import re
import json
import asyncio
import contextlib
from core import trace
from agent import approvals
from agent import conversation as a_conversation
from agent import written as a_written
from core import emotions
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from server import vad
from server.session import Session
from server import app as s_app
from server.actions import chat
from server.actions import memory as s_memory
from server import panels
from server.actions import settings as s_settings
from server.actions import sources
from server import turns as s_turns
from server.actions import wiki

ws_log = trace.logger("WS")


async def _handle_audio(sess: Session, data, loop):
    """One binary PCM frame from the browser, fed to the hands-free endpointer."""
    if not panels.hands_free:
        return
    if sess.endpointer is None:
        sess.endpointer = vad.Endpointer(**vad.DEFAULTS)
    # Gated synchronously, before any await: no window where a turn is running
    # but the endpointer is still open.
    sess.endpointer.gated = panels.controller.busy
    utterances = await loop.run_in_executor(None, sess.endpointer.feed, data)
    if sess.endpointer.hearing != sess.hearing:
        sess.hearing = sess.endpointer.hearing
        await panels.send({"type": "listening", "value": "hearing" if sess.hearing else "idle"})
    for utt in utterances:
        panels.submit_turn(s_turns.handle_utterance, utt, loop)


ACTIONS = {
    "text_turn": chat._action_text_turn,
    "set_focus": chat._action_set_focus,
    "set_hands_free": s_settings._action_set_hands_free,
    "playback_done": chat._action_playback_done,
    "playback_blocked": chat._action_playback_blocked,
    "set_backdrop": s_settings._action_set_backdrop,
    "reload_characters": s_settings._action_reload_characters,
    "set_verbosity": s_settings._action_set_verbosity,
    "set_setting": s_settings._action_set_setting,
    "set_location": s_settings._action_set_location,
    "model_load": s_settings._action_model_load,
    "model_unload": s_settings._action_model_unload,
    "get_memory": s_memory._action_get_memory,
    "save_memory": s_memory._action_save_memory,
    "memory_op": s_memory._action_memory_op,
    "memory_ops": s_memory._action_memory_ops,
    "memory_reset": s_memory._action_memory_reset,
    "get_wiki": wiki._action_get_wiki,
    "read_wiki_page": wiki._action_read_wiki_page,
    "wiki_note": wiki._action_wiki_note,
    "wiki_ingest": wiki._action_wiki_ingest,
    "wiki_save": wiki._action_wiki_save,
    "wiki_create": wiki._action_wiki_create,
    "wiki_rename": wiki._action_wiki_rename,
    "wiki_delete": wiki._action_wiki_delete,
    "wiki_search": wiki._action_wiki_search,
    "get_sources": sources._action_get_sources,
    "reingest_source": sources._action_reingest_source,
    "forget_source": sources._action_forget_source,
    "resolve": chat._action_resolve,
    "archive_memory": s_memory._action_archive_memory,
    "pull_wiki": s_memory._action_pull_wiki,
    "new_topic": chat._action_new_topic,
    "save_card": chat._action_save_card,
    "shutdown": s_settings._action_shutdown,
}


_EARLIER = re.compile(r"^<earlier>(.*)</earlier>$", re.S)


def _replayed(card: dict, sess: Session | None) -> dict:
    """A card as a replay sends it: an awaiting approval that is not this session's pending one
    comes without its Approve / Reject (PRD 52) — never live buttons for an expired request."""
    msg = s_turns.card_msg(card)
    pending = getattr(sess, "pending", None)
    if card.get("status") == "awaiting" and (pending is None or pending["run_id"] != card.get("run_id")):
        msg["actions"] = [a for a in msg.get("actions") or [] if a.get("kind") not in approvals.DECISIONS]
    return msg


def replay_messages(sess: Session | None = None) -> list:
    """The `transcript` / `thread` messages that rebuild the history on a (re)connect: the
    user and assistant turns of `_conversation` (bare text), an `<earlier>` summary as `thread`, each
    card with its last status — a write-up still running or failed, an approval card (PRD 52), where
    it was issued (PRD 51)."""
    out, loose = [], a_written.loose_cards()
    for i, m in enumerate(a_conversation.dialogue()):
        while loose and loose[0][0] <= i:                   # a write-up / approval card (PRD 51/52)
            out.append({**_replayed(loose.pop(0)[1], sess), "saved": None})
        earlier = _EARLIER.match(m["content"]) if m["role"] == "user" else None
        card = a_written.card_in(m["content"]) if m["role"] == "assistant" else None
        if earlier:
            out.append({"type": "thread", "text": earlier.group(1)})
        elif card:
            out.append({**s_turns.card_msg(card), "saved": card.get("saved")})
        else:
            out.append({"type": "transcript", "role": m["role"], "text": emotions.strip_tag(m["content"])})
    out += [{**_replayed(card, sess), "saved": None} for _pos, card in loose]
    return out


async def _on_connect(sess: Session, loop):
    """Bring a freshly connected page up to date, and greet on the very first one."""

    # Replay what the model actually holds (the stripped conversation, PRD 39), so a reloaded
    # page matches its memory even after a periodic fold or a trim.
    for msg in replay_messages(sess):
        await panels.send(msg)
    await panels.send({"type": "hands_free", "value": panels.hands_free})
    if not panels.greeted:
        panels.greeted = True
        await panels.send_state(sess, "thinking")  # busy from connect, not only once the greet is picked up
        panels.submit_turn(s_turns.greet, loop)
    else:
        await panels.send_state(sess, "idle")


def _allowed_origins() -> set:
    """Our own loopback:PORT pair plus `ALLOWED_ORIGINS` (comma list) — the Docker prod publishes
    the container's PORT on a host port the app cannot see (docs/40-docker-prod.md)."""
    extra = os.getenv("ALLOWED_ORIGINS", "")
    return {f"http://localhost:{s_app.PORT}", f"http://127.0.0.1:{s_app.PORT}"} | {
        o.strip().rstrip("/") for o in extra.split(",") if o.strip()}


def _origin_allowed(origin: str | None) -> bool:
    """Non-browser clients send no Origin; browsers must be on an allowed origin."""
    if origin is None:
        return True
    return origin.rstrip("/") in _allowed_origins()


@s_app.app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    if not _origin_allowed(websocket.headers.get("origin")):
        ws_log.warning(f"refused origin {websocket.headers.get('origin')!r}")
        trace.event("ws", event="refused_origin", origin=websocket.headers.get("origin"))
        await websocket.close(code=4003)
        return
    await websocket.accept()
    trace.event("ws", event="connect")
    code = None
    previous, panels.session = panels.session, Session(websocket)
    sess = panels.session
    if previous is not None:
        # A new page takes over: release anything waiting on the old one, then evict it.
        old_ws, previous.websocket = previous.websocket, None
        previous.playback_done.set()
        approvals.expire(previous, "reconnect")      # its approval dies with it (PRD 52)
        if old_ws is not None:
            with contextlib.suppress(Exception):
                await old_ws.close(code=4000)
    loop = panels._loop = asyncio.get_running_loop()
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
                    else:
                        action = msg.get("action")
                        trace.error("ws", "ws", ValueError(f"unknown action {action!r}"), fatal_for="none")
                        await panels._error("settings", f"unknown action {action!r}", action=action)
            except WebSocketDisconnect:
                raise
            except Exception as e:
                ws_log.exception("handler failed")
                trace.error("ws", "ws", e, fatal_for="none")
                # a ValueError is a validation message written for the user; anything else is type only
                text = str(e) if isinstance(e, ValueError) else f"Something went wrong ({type(e).__name__})."
                await panels._error(panels._scope_of(msg.get("action")), text, action=msg.get("action"))
    except WebSocketDisconnect as e:
        code = e.code
    finally:
        trace.event("ws", event="disconnect", code=code)
        approvals.expire(sess, "disconnect")
        if code != 4000 and panels.session is sess:     # not on a takeover (4000) or for an evicted socket: the user has not left
            panels.cancel_idle_timer()                  # (and the new tab may already have armed its own timer)
            panels.controller.save_now()                # the user is away: fold what is unsaved (PRD 56)
        sess.websocket = None
        sess.playback_done.set()  # never leave a turn blocked on a dead socket
        if panels.session is sess:
            panels.session = None
