"""PRD 12 scenarios: turn controller, session lifecycle, protocol hardening.

Covers docs/12-turn-controller.md scenarios 1-8, 10 and 12 (2 and 11 live in
test_hands_free.py, 9 in test_avatars.py).
"""
import io
import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import avatars
import chatbot
import llm
import session as session_mod
from session import Session, TurnController
from tests.conftest import Blocker, fake_asr


def fake_tts(text):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(240, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def loop():
    """A background event loop, as the real server has, for make_sender()."""
    lp = asyncio.new_event_loop()
    t = threading.Thread(target=lp.run_forever, daemon=True)
    t.start()
    yield lp
    lp.call_soon_threadsafe(lp.stop)
    t.join(timeout=2)


# ---------------------------------------------------------------- controller


def test_controller_runs_submitted_turns_in_order():
    order = []
    c = TurnController()
    c.submit(lambda: order.append("a"))
    c.submit(lambda: order.append("b"))
    assert c.join_idle(timeout=5)
    assert order == ["a", "b"]
    assert c.busy is False


def test_controller_is_busy_from_submit_until_the_turn_returns():
    c = TurnController()
    blocker = Blocker()
    assert c.busy is False
    c.submit(blocker)
    assert c.busy is True  # synchronously, before the worker even starts
    blocker.started.wait(timeout=5)
    assert c.busy is True
    blocker.free(c)
    assert c.busy is False


def test_controller_survives_a_failing_turn():
    done = []
    c = TurnController()
    c.submit(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    c.submit(lambda: done.append(1))
    assert c.join_idle(timeout=5)
    assert done == [1]


def test_scenario_1_two_utterances_run_sequentially(monkeypatch, loop):
    """Turn 2 does not start until turn 1's playback_done has landed."""
    llm.reset()
    sent = []
    sess = Session(websocket=object())  # truthy socket → the playback wait is real

    def fake_ask_events(user_text, origin="user", note=""):
        llm._conversation.append({"role": "user", "content": user_text})
        reply = f"[happy] Answer to {user_text}."
        yield ("delta", reply)
        llm._conversation.append({"role": "assistant", "content": reply})

    monkeypatch.setattr(llm, "ask_events", fake_ask_events)
    monkeypatch.setattr(chatbot, "synthesize", fake_tts)
    monkeypatch.setattr(chatbot, "asr", fake_asr(lambda a: f"q{int(a[0])}"))
    monkeypatch.setattr(chatbot, "make_sender",
                        lambda lp, turn=None: lambda m: sent.append((turn, m["type"], m.get("value"))))
    monkeypatch.setattr(chatbot, "session", sess)

    chatbot.submit_turn(chatbot.handle_utterance, np.array([1.0], dtype=np.float32), loop)
    chatbot.submit_turn(chatbot.handle_utterance, np.array([2.0], dtype=np.float32), loop)

    # Turn 1 is parked on playback_done; turn 2 has not produced anything yet.
    for _ in range(200):
        if (1, "speech_end", None) in sent:
            break
        threading.Event().wait(0.01)
    assert sess.expected_turn == 1
    assert not any(t == 2 for t, _, _ in sent)

    sess.playback_done.set()  # browser reports turn 1 finished
    for _ in range(200):
        if sess.expected_turn == 2:
            break
        threading.Event().wait(0.01)
    sess.playback_done.set()  # and turn 2
    assert chatbot.controller.join_idle(timeout=5)

    turns = [t for t, _, _ in sent]
    assert turns == sorted(turns)  # no interleaving
    assert [m["content"] for m in llm._conversation[1:]] == [
        "q1", "[happy] Answer to q1.", "q2", "[happy] Answer to q2.",
    ]
    assert [m["role"] for m in llm._conversation[1:]] == ["user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------- ws robustness


def test_scenario_3_malformed_text_frame_keeps_the_socket_open():
    llm.reset()                                                  # nothing to replay on connect
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json(); ws.receive_json()
        ws.send_text("{not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"action": "set_backdrop", "key": "none"})
        assert ws.receive_json() == {"type": "backdrop", "key": "none"}


def test_scenario_4_bad_character_cards_report_an_error(monkeypatch):
    llm.reset()
    def boom():
        raise ValueError("bad YAML in characters/wanko.yaml")

    monkeypatch.setattr(avatars, "reload", boom)
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json(); ws.receive_json()
        ws.send_json({"action": "reload_characters"})
        msg = ws.receive_json()
        assert msg["type"] == "error" and "bad YAML" in msg["text"]
        ws.send_json({"action": "set_backdrop", "key": "none"})
        assert ws.receive_json()["type"] == "backdrop"


def test_scenario_5_second_connection_closes_the_first(monkeypatch):
    llm.reset()
    monkeypatch.setattr(chatbot, "hands_free", True)
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws1:
        ws1.receive_json(); ws1.receive_json()
        first = chatbot.session
        first.hearing = True   # per-connection state that must not leak into the next socket

        with client.websocket_connect("/ws") as ws2:
            ws2.receive_json(); ws2.receive_json()
            second = chatbot.session
            assert second is not first
            assert second.hearing is False and second.endpointer is None

            with pytest.raises(WebSocketDisconnect) as excinfo:
                ws1.receive_json()
            assert excinfo.value.code == 4000

            # The new connection drives turns normally.
            ws2.send_json({"action": "set_backdrop", "key": "none"})
            assert ws2.receive_json()["type"] == "backdrop"


# ---------------------------------------------------------------- turn ids


def _run(coro):
    asyncio.run(coro)


def test_scenario_6_stale_playback_done_is_ignored():
    sess = Session(websocket=object())
    sess.expected_turn = 5
    sess.playback_done.clear()

    _run(chatbot._action_playback_done(sess, {"turn": 3}, None))
    assert sess.playback_done.is_set() is False  # stale echo: still waiting

    _run(chatbot._action_playback_done(sess, {"turn": 5}, None))
    assert sess.playback_done.is_set() is True


def test_playback_done_without_a_turn_is_treated_as_current():
    """Back-compat: a client that does not echo the id still releases the wait."""
    sess = Session(websocket=object())
    sess.expected_turn = 5
    sess.playback_done.clear()
    _run(chatbot._action_playback_done(sess, {}, None))
    assert sess.playback_done.is_set() is True


def test_scenario_7_playback_blocked_ends_the_wait(monkeypatch, loop):
    """Autoplay was blocked: the server goes idle instead of waiting out the 120 s timeout."""
    llm.reset()
    monkeypatch.setattr(chatbot, "synthesize", fake_tts)
    sent = []
    monkeypatch.setattr(chatbot, "make_sender", lambda lp, turn=None: sent.append)

    sess = Session(websocket=object())
    turn = sess.next_turn()
    t = threading.Thread(target=chatbot.canned_turn,
                         args=(sess, turn, loop, "[happy] Hello there.", False), daemon=True)
    t.start()
    for _ in range(200):  # wait until the turn is parked on playback
        if sess.expected_turn == turn and any(m["type"] == "speech_end" for m in sent):
            break
        threading.Event().wait(0.01)

    _run(chatbot._action_playback_blocked(sess, {"turn": 999}, None))  # wrong turn: ignored
    assert sess.playback_done.is_set() is False
    _run(chatbot._action_playback_blocked(sess, {"turn": turn}, None))
    t.join(timeout=5)
    assert not t.is_alive()
    assert sent[-1] == {"type": "state", "value": "idle"}


# ---------------------------------------------------------------- origin / host


def test_scenario_8_origin_check():
    llm.reset()  # no replayed transcript before the hands_free announce
    client = TestClient(chatbot.app)
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_json()
    assert excinfo.value.code == 4003

    with client.websocket_connect("/ws", headers={"origin": f"http://localhost:{chatbot.PORT}"}) as ws:
        assert ws.receive_json()["type"] == "hands_free"

    with client.websocket_connect("/ws", headers={"origin": f"http://127.0.0.1:{chatbot.PORT}"}) as ws:
        assert ws.receive_json()["type"] == "hands_free"


def test_no_origin_header_is_allowed():
    """Non-browser clients (tests, the websockets lib) send no Origin."""
    assert chatbot._origin_allowed(None) is True
    assert chatbot._origin_allowed("http://evil.example") is False


def test_default_bind_host_is_loopback():
    assert chatbot.HOST == "127.0.0.1"


# ---------------------------------------------------------------- static files


def test_static_files_are_served_without_caching():
    client = TestClient(chatbot.app)
    res = client.get("/static/app.js")
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    assert "cache-control" not in client.get("/api/config").headers


# ---------------------------------------------------------------- dead sockets


class _DeadSocket:
    def __init__(self):
        self.calls = 0

    async def send_json(self, msg):
        self.calls += 1
        raise RuntimeError("socket closed")


def test_send_drops_the_socket_on_the_first_failure(monkeypatch, capsys):
    ws = _DeadSocket()
    sess = Session(websocket=ws)
    monkeypatch.setattr(chatbot, "session", sess)
    sess.playback_done.clear()
    _run(chatbot.send({"type": "state", "value": "idle"}))
    _run(chatbot.send({"type": "state", "value": "idle"}))
    assert ws.calls == 1
    assert sess.websocket is None and sess.playback_done.is_set()
    assert capsys.readouterr().out.count("dropping socket") == 1


def test_make_sender_skips_once_the_socket_is_dropped(monkeypatch):
    monkeypatch.setattr(chatbot, "session", Session(websocket=None))
    scheduled = []
    monkeypatch.setattr(chatbot.asyncio, "run_coroutine_threadsafe",
                        lambda coro, loop: scheduled.append(coro))
    chatbot.make_sender(loop=None)({"type": "state", "value": "idle"})
    assert scheduled == []


# ---------------------------------------------------------------- periodic memory


def test_scenario_10_memory_saved_every_n_turns(monkeypatch):
    """The periodic save fires on every multiple of MEMORY_SAVE_EVERY (default 5), never before."""
    assert session_mod.MEMORY_SAVE_EVERY == 5
    calls = []
    monkeypatch.setattr(llm, "save_memory", lambda: calls.append(1))
    every = session_mod.MEMORY_SAVE_EVERY
    c = TurnController(on_turn_complete=chatbot._periodic_memory)
    for _ in range(every - 1):
        c.submit(lambda: None)
    assert c.join_idle(timeout=5)
    assert calls == []
    c.submit(lambda: None)
    assert c.join_idle(timeout=5)
    assert calls == [1]
    for _ in range(every):
        c.submit(lambda: None)
    assert c.join_idle(timeout=5)
    assert calls == [1, 1]


def test_memory_save_cadence_follows_the_constant(monkeypatch):
    """The hook fires on multiples of `MEMORY_SAVE_EVERY`, whatever the constant says, and is
    handed the turn count."""
    monkeypatch.setattr(session_mod, "MEMORY_SAVE_EVERY", 3)
    counts = []
    c = TurnController(on_turn_complete=counts.append)
    for _ in range(6):
        c.submit(lambda: None)
    assert c.join_idle(timeout=5)
    assert counts == [3, 6]


def test_memory_save_cadence_is_read_from_the_environment(monkeypatch):
    """MEMORY_SAVE_EVERY=<n> overrides the default (session.py reads it at import; loaded
    as a fresh module object so the live `session` is left alone)."""
    import importlib.util
    monkeypatch.setenv("MEMORY_SAVE_EVERY", "7")
    spec = importlib.util.spec_from_file_location("session_under_env", session_mod.__file__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    assert fresh.MEMORY_SAVE_EVERY == 7
    assert session_mod.MEMORY_SAVE_EVERY == 5


def test_next_turn_starts_while_the_memory_save_is_still_running(monkeypatch):
    """The periodic save runs off the turn worker: a slow save must not hold the next turn."""
    save_started, release, order = threading.Event(), threading.Event(), []

    def slow_save():
        save_started.set()
        release.wait(timeout=5)
        order.append("save-done")

    monkeypatch.setattr(llm, "save_memory", slow_save)
    c = TurnController(on_turn_complete=chatbot._periodic_memory)
    for _ in range(session_mod.MEMORY_SAVE_EVERY):
        c.submit(lambda: None)
    assert save_started.wait(timeout=5)
    c.submit(lambda: order.append("next-turn"))
    assert c._idle.wait(timeout=5)          # the turn ran while the hook is still parked
    assert order == ["next-turn"] and c.hook_running
    release.set()
    assert c.join_idle(timeout=5)
    assert order == ["next-turn", "save-done"]


def test_periodic_memory_tells_the_ui_while_it_saves(monkeypatch):
    sent = []
    monkeypatch.setattr(chatbot, "session", Session(websocket=object()))
    monkeypatch.setattr(chatbot, "_loop", SimpleNamespace(is_running=lambda: True))
    monkeypatch.setattr(chatbot, "make_sender", lambda loop, turn=None: sent.append)
    monkeypatch.setattr(llm, "save_memory", lambda: sent.append("saved"))
    chatbot._periodic_memory(5)
    seen = [m for m in sent if m == "saved" or m.get("type") == "memory_saving"]
    assert seen == [{"type": "memory_saving", "value": True}, "saved",
                    {"type": "memory_saving", "value": False}]


def test_overlapping_hook_runs_are_skipped(monkeypatch):
    monkeypatch.setattr(session_mod, "MEMORY_SAVE_EVERY", 1)
    release, counts = threading.Event(), []

    def hook(count):
        counts.append(count)
        release.wait(timeout=5)

    c = TurnController(on_turn_complete=hook)
    c.submit(lambda: None); c.submit(lambda: None)
    assert c._idle.wait(timeout=5)
    release.set()
    assert c.join_idle(timeout=5)
    assert counts == [1]


def test_periodic_memory_error_does_not_kill_the_worker(monkeypatch):
    def boom():
        raise RuntimeError("oMLX down")

    monkeypatch.setattr(llm, "save_memory", boom)
    done = []
    c = TurnController(on_turn_complete=chatbot._periodic_memory)
    for _ in range(session_mod.MEMORY_SAVE_EVERY):
        c.submit(lambda: None)
    c.submit(lambda: done.append(1))
    assert c.join_idle(timeout=5)
    assert done == [1]


# ---------------------------------------------------------------- markers


def test_scenario_12_live_tests_are_excluded_by_default(pytestconfig):
    """pytest.ini deselects `live` and `e2e` unless asked for; this run is the default one."""
    markers = "\n".join(pytestconfig.getini("markers"))
    assert "live:" in markers and "e2e:" in markers
    assert "not live and not e2e" in " ".join(pytestconfig.getini("addopts"))


def test_allowed_origins_env_extends_the_loopback_pair(monkeypatch):
    """PRD 40 scenario 2: ALLOWED_ORIGINS adds to — never replaces — the loopback:PORT pair."""
    extra = f"http://localhost:{chatbot.PORT + 1}"  # anything but the loopback:PORT pair
    monkeypatch.setenv("ALLOWED_ORIGINS", extra)
    assert chatbot._origin_allowed(extra) is True
    assert chatbot._origin_allowed(f"http://evil:{chatbot.PORT}") is False
    assert chatbot._origin_allowed(f"http://localhost:{chatbot.PORT}") is True

    monkeypatch.delenv("ALLOWED_ORIGINS")
    assert chatbot._origin_allowed(extra) is False
    assert chatbot._origin_allowed(f"http://localhost:{chatbot.PORT}") is True
    assert chatbot._origin_allowed(f"http://127.0.0.1:{chatbot.PORT}") is True
    assert chatbot._origin_allowed(None) is True
