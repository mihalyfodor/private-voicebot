"""Connection state and what goes to the browser: the session, the turn controller, senders, error scopes and the memory / wiki / services panel messages (PRD 50 step 11, from chatbot.py)."""
import asyncio
from core import trace
from agent import conversation as a_conversation
from memory import ops as m_ops
from memory import store as m_store
from vault import chunks
from vault import focus as v_focus
from vault import lint as v_lint
from vault import pages as v_pages
from vault import search as v_search
from vault import store as v_store
from core import settings
from server import services
from server.session import Session, TurnController

log = trace.logger("App")
ws_log = trace.logger("WS")
mem_log = trace.logger("Memory")


# ---------------------------------------------------------------- state
# One connection at a time (single-user app): `session` is the live Session,
# `controller` serialises turns on its own worker thread. See docs/12-turn-controller.md.
session: Session | None = None
#: the asyncio loop serving the websocket, so worker-thread code can push to the browser
_loop: asyncio.AbstractEventLoop | None = None


def _memory_msg(profile: dict) -> dict:
    """The `memory` message the panel re-renders itself from: profile + rendered block + count."""
    return {"type": "memory", **m_ops.snapshot(profile)}


def _wiki_msg() -> dict:
    """The `wiki` message the Wiki tab re-renders itself from: page list + lint + vault path."""
    v_store.ensure_vault()   # opening the tab is the first use that may create the vault
    return {"type": "wiki", "pages": v_lint.pages(), "lint": v_lint.lint(), "vault": v_store.WIKI_PATH}


def _wiki_page_msg(path: str) -> dict:
    """One whole page for the pane/editor, with its etag (PRD 28) and backlinks."""
    text = chunks.page_text(path)
    if text is None:
        return {"type": "wiki_page", "path": path, "text": f"no such page: {path}",
                "links": [], "etag": "", "backlinks": []}
    return {"type": "wiki_page", "path": path, "text": text, "links": v_focus.links_in(text),
            "etag": v_store.etag(text), "backlinks": v_search.backlinks(path)}


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
    send_sync(_memory_msg(m_store.load_profile()))
    if with_wiki:
        send_sync(_wiki_msg())


def _push_saving(value: bool) -> None:
    """Tell the UI a background memory save is running (it shows "saving memory…")."""
    send_sync = _panel_sender()
    if send_sync is not None:
        send_sync({"type": "memory_saving", "value": value})


def _push_services(msg: dict) -> None:
    """A probe changed (PRD 41, on the services daemon thread): the page updates its banner."""
    send_sync = _panel_sender()
    if send_sync is not None:
        send_sync({"type": "services", **msg})


def llm_offline_text() -> str:
    """The refusal a turn gets while the LLM is down: what is down and the command that fixes it."""
    label = services.snapshot().get("llm", {}).get("label", "oMLX")
    return f"LLM offline ({label}) — run {services.START_HINT}"


def _turn_failed(send_sync, where: str, exc: BaseException, text: str, scope: str = "turn",
                 recheck: bool = False) -> None:
    """A turn-level handler failed: log it, trace it (type + code only), tell the client, idle."""
    log.exception(f"{where} failed")
    trace.error("turn", where, exc)         # marks the turn failed (PRD 49)
    if recheck:
        services.recheck()                    # a failed completion: find out now whether oMLX is gone
    send_sync({"type": "error", "scope": scope, "text": text})
    send_sync({"type": "state", "value": "idle"})


def _memory_failed(exc: BaseException) -> None:
    """A background memory save failed: log + trace it; nothing waits on it."""
    mem_log.exception("memory save failed")
    trace.error("memory", "memory_save", exc, fatal_for="none")


def _periodic_memory(count: int) -> None:
    """Every N completed turns (on the controller's hook thread, not the turn worker): fold
    the session into the memory profile (memory.json)."""
    filed = v_pages.applied_count()
    _push_saving(True)
    try:
        a_conversation.save_memory()
    except Exception as e:
        _memory_failed(e)
    finally:
        _push_saving(False)
    # an open panel refreshes itself after a background save; the Wiki tab only when the
    # save actually filed something, so a quiet session costs no extra traffic.
    push_memory(with_wiki=v_pages.applied_count() != filed)


controller = TurnController(on_turn_complete=_periodic_memory)

greeted = False
hands_free = bool(settings.load_settings().get("hands_free", False))

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
        ws_log.warning(f"send failed, dropping socket: {e}")
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
            log.warning(f"send error: {e}")
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
    if a.startswith("wiki") or a in ("pull_wiki", "save_card"):
        return "wiki"
    if "memory" in a:
        return "memory"
    if "source" in a:
        return "sources"
    if a.startswith("set_") or a.startswith("model_") or a == "reload_characters":
        return "settings"
    return "turn"


def submit_turn(fn, *args) -> int | None:
    """Allocate a turn id on the current session and queue `fn(session, turn, *args)`."""
    sess = session
    if sess is None:
        return None
    turn = sess.next_turn()
    controller.submit(fn, sess, turn, *args, turn_id=turn)
    return turn


def _turn_matches(sess: Session, turn) -> bool:
    """Is a client echo about the turn we are waiting on?

    True when the ids match, when we are not waiting on any turn, or when the client
    omitted the id altogether (older clients that do not echo it).
    """
    return turn is None or sess.expected_turn is None or turn == sess.expected_turn
