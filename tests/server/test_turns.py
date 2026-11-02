"""PRD 12 scenarios: turn controller, session lifecycle, protocol hardening.

Covers docs/12-turn-controller.md scenarios 1-8, 10 and 12 (2 and 11 live in
test_hands_free.py, 9 in test_avatars.py), and the briefing turn (docs/46-briefing.md scenarios 3-6).
"""
import io
import json
import asyncio
import threading
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from core import avatars
from core import settings
from server.actions import chat
from server import app
from server import panels
from server import turns as server_turns
from server import ws as m_ws
from server import speech as s_speech
from tools import registry
from agent import conversation
from agent import prompt as a_prompt
from agent import turn as a_turn
from agent import fold
from server import services
from server import session as session_mod
from vault import store
from server.session import Session, TurnController
from tests.conftest import Blocker, FakeClient, days_ago, fake_asr, log_ops, run_turn, write_page
from core import llm as core_llm
from core import util


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
    conversation.reset()
    sent = []
    sess = Session(websocket=object())  # truthy socket → the playback wait is real

    def fake_ask_events(user_text, origin="user", note=""):
        conversation._conversation.append({"role": "user", "content": user_text})
        reply = f"[happy] Answer to {user_text}."
        yield ("delta", reply)
        conversation._conversation.append({"role": "assistant", "content": reply})

    monkeypatch.setattr(a_turn, "ask_events", fake_ask_events)
    monkeypatch.setattr(s_speech, "synthesize", fake_tts)
    monkeypatch.setattr(server_turns, "asr", fake_asr(lambda a: f"q{int(a[0])}"))
    monkeypatch.setattr(panels, "make_sender",
                        lambda lp, turn=None: lambda m: sent.append((turn, m["type"], m.get("value"))))
    monkeypatch.setattr(panels, "session", sess)

    panels.submit_turn(server_turns.handle_utterance, np.array([1.0], dtype=np.float32), loop)
    panels.submit_turn(server_turns.handle_utterance, np.array([2.0], dtype=np.float32), loop)

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
    assert panels.controller.join_idle(timeout=5)

    turns = [t for t, _, _ in sent]
    assert turns == sorted(turns)  # no interleaving
    assert [m["content"] for m in conversation._conversation[1:]] == [
        "q1", "[happy] Answer to q1.", "q2", "[happy] Answer to q2.",
    ]
    assert [m["role"] for m in conversation._conversation[1:]] == ["user", "assistant", "user", "assistant"]


# ---------------------------------------------------------------- ws robustness


def test_scenario_3_malformed_text_frame_keeps_the_socket_open():
    conversation.reset()                                                  # nothing to replay on connect
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json(); ws.receive_json()
        ws.send_text("{not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"action": "set_backdrop", "key": "none"})
        assert ws.receive_json() == {"type": "backdrop", "key": "none"}


def test_scenario_4_bad_character_cards_report_an_error(monkeypatch):
    conversation.reset()
    def boom():
        raise ValueError("bad YAML in characters/wanko.yaml")

    monkeypatch.setattr(avatars, "reload", boom)
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json(); ws.receive_json()
        ws.send_json({"action": "reload_characters"})
        msg = ws.receive_json()
        assert msg["type"] == "error" and "bad YAML" in msg["text"]
        ws.send_json({"action": "set_backdrop", "key": "none"})
        assert ws.receive_json()["type"] == "backdrop"


def test_scenario_5_second_connection_closes_the_first(monkeypatch):
    conversation.reset()
    monkeypatch.setattr(panels, "hands_free", True)
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws1:
        ws1.receive_json(); ws1.receive_json()
        first = panels.session
        first.hearing = True   # per-connection state that must not leak into the next socket

        with client.websocket_connect("/ws") as ws2:
            ws2.receive_json(); ws2.receive_json()
            second = panels.session
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

    _run(chat._action_playback_done(sess, {"turn": 3}, None))
    assert sess.playback_done.is_set() is False  # stale echo: still waiting

    _run(chat._action_playback_done(sess, {"turn": 5}, None))
    assert sess.playback_done.is_set() is True


def test_playback_done_without_a_turn_is_treated_as_current():
    """Back-compat: a client that does not echo the id still releases the wait."""
    sess = Session(websocket=object())
    sess.expected_turn = 5
    sess.playback_done.clear()
    _run(chat._action_playback_done(sess, {}, None))
    assert sess.playback_done.is_set() is True


def test_scenario_7_playback_blocked_ends_the_wait(monkeypatch, loop):
    """Autoplay was blocked: the server goes idle instead of waiting out the 120 s timeout."""
    conversation.reset()
    monkeypatch.setattr(s_speech, "synthesize", fake_tts)
    sent = []
    monkeypatch.setattr(panels, "make_sender", lambda lp, turn=None: sent.append)

    sess = Session(websocket=object())
    turn = sess.next_turn()
    t = threading.Thread(target=server_turns.canned_turn,
                         args=(sess, turn, loop, "[happy] Hello there.", False), daemon=True)
    t.start()
    for _ in range(200):  # wait until the turn is parked on playback
        if sess.expected_turn == turn and any(m["type"] == "speech_end" for m in sent):
            break
        threading.Event().wait(0.01)

    _run(chat._action_playback_blocked(sess, {"turn": 999}, None))  # wrong turn: ignored
    assert sess.playback_done.is_set() is False
    _run(chat._action_playback_blocked(sess, {"turn": turn}, None))
    t.join(timeout=5)
    assert not t.is_alive()
    assert sent[-1] == {"type": "state", "value": "idle"}


# ---------------------------------------------------------------- origin / host


def test_scenario_8_origin_check():
    conversation.reset()  # no replayed transcript before the hands_free announce
    client = TestClient(app.app)
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_json()
    assert excinfo.value.code == 4003

    with client.websocket_connect("/ws", headers={"origin": f"http://localhost:{app.PORT}"}) as ws:
        assert ws.receive_json()["type"] == "hands_free"

    with client.websocket_connect("/ws", headers={"origin": f"http://127.0.0.1:{app.PORT}"}) as ws:
        assert ws.receive_json()["type"] == "hands_free"


def test_no_origin_header_is_allowed():
    """Non-browser clients (tests, the websockets lib) send no Origin."""
    assert m_ws._origin_allowed(None) is True
    assert m_ws._origin_allowed("http://evil.example") is False


def test_default_bind_host_is_loopback():
    assert app.HOST == "127.0.0.1"


# ---------------------------------------------------------------- static files


def test_static_files_are_served_without_caching():
    client = TestClient(app.app)
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
    monkeypatch.setattr(panels, "session", sess)
    sess.playback_done.clear()
    _run(panels.send({"type": "state", "value": "idle"}))
    _run(panels.send({"type": "state", "value": "idle"}))
    assert ws.calls == 1
    assert sess.websocket is None and sess.playback_done.is_set()
    assert capsys.readouterr().out.count("dropping socket") == 1


def test_make_sender_skips_once_the_socket_is_dropped(monkeypatch):
    monkeypatch.setattr(panels, "session", Session(websocket=None))
    scheduled = []
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe",
                        lambda coro, loop: scheduled.append(coro))
    panels.make_sender(loop=None)({"type": "state", "value": "idle"})
    assert scheduled == []


# ---------------------------------------------------------------- periodic memory


def test_scenario_10_memory_saved_every_n_turns(monkeypatch):
    """The periodic save fires on every multiple of MEMORY_SAVE_EVERY (default 5), never before."""
    assert session_mod.MEMORY_SAVE_EVERY == 5
    calls = []
    monkeypatch.setattr(conversation, "save_memory", lambda: calls.append(1))
    every = session_mod.MEMORY_SAVE_EVERY
    c = TurnController(on_turn_complete=panels._periodic_memory)
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

    monkeypatch.setattr(conversation, "save_memory", slow_save)
    c = TurnController(on_turn_complete=panels._periodic_memory)
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
    monkeypatch.setattr(panels, "session", Session(websocket=object()))
    monkeypatch.setattr(panels, "_loop", SimpleNamespace(is_running=lambda: True))
    monkeypatch.setattr(panels, "make_sender", lambda loop, turn=None: sent.append)
    monkeypatch.setattr(conversation, "save_memory", lambda: sent.append("saved"))
    panels._periodic_memory(5)
    seen = [m for m in sent if m == "saved" or m.get("type") == "memory_saving"]
    assert seen == [{"type": "memory_saving", "value": True}, "saved",
                    {"type": "memory_saving", "value": False}]


# ---------------------------------------------------------------- idle save, disconnect save, day boundary (PRD 56)


class FakeLoop:
    """The app loop's `call_later` on a virtual clock: `advance(seconds)` fires what is due, in order."""

    def __init__(self):
        self.now, self.timers = 0.0, []

    def call_later(self, delay, cb):
        handle = SimpleNamespace(due=self.now + delay, cb=cb, cancelled=False)
        handle.cancel = lambda: setattr(handle, "cancelled", True)
        self.timers.append(handle)
        return handle

    def call_soon_threadsafe(self, cb, *args):
        cb(*args)

    def is_closed(self):
        return False

    def pending(self):
        return [t for t in self.timers if not t.cancelled]

    def advance(self, seconds):
        target = self.now + seconds
        while True:
            due = sorted((t for t in self.pending() if t.due <= target), key=lambda t: t.due)
            if not due:
                break
            self.now = due[0].due
            due[0].cancelled = True
            due[0].cb()
        self.now = target


def wait_for(cond, timeout=5.0):
    import time
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.005)
    assert cond()


@pytest.fixture
def idle(monkeypatch):
    """A fake loop under the panels, a recording `fold.save`, and one unsaved turn."""
    lp = FakeLoop()
    monkeypatch.setattr(panels, "_loop", lp)
    monkeypatch.setattr(panels, "_idle_handle", None)
    folds = []
    monkeypatch.setattr(fold, "save", lambda turns, sources=(): folds.append(list(turns)) or [])
    conversation.reset()
    conversation._session_turns.append({"role": "user", "content": "hello"})
    yield SimpleNamespace(loop=lp, folds=folds)
    panels.cancel_idle_timer()
    panels.controller.join_idle(timeout=5)


def finish_a_turn(lp, count):
    """One turn through the worker; its end re-arms the idle timer (the worker does that after `join_idle` fires)."""
    panels.controller.submit(lambda: None)
    wait_for(lambda: len(lp.timers) >= count)


def test_idle_save_folds_once_ten_minutes_after_the_last_turn(idle):
    assert session_mod.IDLE_SAVE_MIN == 10
    finish_a_turn(idle.loop, 1)
    idle.loop.advance(9 * 60)
    assert idle.folds == []
    idle.loop.advance(60)
    assert panels.controller.join_idle(timeout=5)
    assert idle.folds == [[{"role": "user", "content": "hello"}]]
    assert conversation.get_session_turns() == []


def test_idle_save_does_nothing_without_unsaved_turns(idle):
    conversation._session_turns.clear()
    finish_a_turn(idle.loop, 1)
    idle.loop.advance(11 * 60)
    assert panels.controller.join_idle(timeout=5)
    assert idle.folds == []


def test_idle_save_never_folds_while_a_turn_is_in_flight(idle):
    finish_a_turn(idle.loop, 1)
    blocker = Blocker()
    blocker.hold(panels.controller)
    assert panels.controller.busy
    idle.loop.advance(10 * 60)                              # fires, sees busy, re-arms
    assert idle.folds == [] and len(idle.loop.pending()) == 1
    blocker.free(panels.controller)
    wait_for(lambda: len(idle.loop.pending()) == 1 and idle.loop.pending()[0].due > idle.loop.now)
    idle.loop.advance(10 * 60)
    assert panels.controller.join_idle(timeout=5)
    assert len(idle.folds) == 1


def test_idle_save_timer_restarts_at_every_completed_turn(idle):
    finish_a_turn(idle.loop, 1)
    idle.loop.advance(6 * 60)
    finish_a_turn(idle.loop, 2)                             # six minutes later: the countdown starts over
    idle.loop.advance(4 * 60)                               # minute 10
    assert idle.folds == []
    idle.loop.advance(6 * 60)                               # minute 16
    assert panels.controller.join_idle(timeout=5)
    assert len(idle.folds) == 1


def test_disconnect_saves_unsaved_turns(idle, monkeypatch):
    def connect_and_leave():
        with TestClient(app.app).websocket_connect("/ws"):
            pass
        assert panels.controller.join_idle(timeout=5)

    monkeypatch.setattr(panels, "greeted", True)
    connect_and_leave()
    assert idle.folds == [[{"role": "user", "content": "hello"}]]
    connect_and_leave()                                     # nothing unsaved any more
    assert len(idle.folds) == 1


def test_a_takeover_does_not_fold_but_the_final_close_does(idle, monkeypatch):
    monkeypatch.setattr(panels, "greeted", True)
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws1:
        ws1.receive_json(); ws1.receive_json()
        with client.websocket_connect("/ws") as ws2:
            ws2.receive_json(); ws2.receive_json()
            with pytest.raises(WebSocketDisconnect):
                ws1.receive_json()                          # ws1 was evicted with 4000
            panels.controller.join_idle(timeout=5)
            assert idle.folds == []                         # the new socket continues the same conversation
    assert panels.controller.join_idle(timeout=5)
    assert len(idle.folds) == 1


def test_a_running_fold_makes_the_idle_and_disconnect_saves_skip(idle, monkeypatch):
    started, release, calls = threading.Event(), threading.Event(), []

    def slow(turns, sources=()):
        calls.append(list(turns))
        started.set()
        release.wait(timeout=5)
        return []

    monkeypatch.setattr(fold, "save", slow)
    panels.controller.save_now()
    assert started.wait(timeout=5)
    conversation._session_turns.append({"role": "user", "content": "later"})   # said after the snapshot
    finish_a_turn(idle.loop, 1)
    idle.loop.advance(10 * 60)                              # the idle fire: skipped by the overlap guard
    with TestClient(app.app).websocket_connect("/ws"):
        pass                                                # the disconnect save: skipped too
    assert len(calls) == 1
    assert conversation.get_session_turns() == [{"role": "user", "content": "later"}]
    release.set()
    assert panels.controller.join_idle(timeout=5)


ZONE = ZoneInfo("Pacific/Honolulu")


@pytest.fixture
def rollover(monkeypatch):
    """A conversation with a few turns, a recording fold and a clock the test sets; `answer` runs one typed turn."""
    from tools import time as t_time
    folds = []
    monkeypatch.setattr(fold, "save", lambda turns, sources=(): folds.append(list(turns)) or [])
    conversation.reset()
    conversation._conversation.extend([{"role": "user", "content": "plan the deploy"},
                                       {"role": "assistant", "content": "[neutral] Friday."}])
    conversation._session_turns.extend(conversation.dialogue())

    def at(last, now):
        monkeypatch.setattr(a_prompt, "prompt_date", (last or now - timedelta(days=1)).date())   # the prompt was built then
        conversation.last_turn_at = last
        monkeypatch.setattr(t_time, "now", lambda: now)

    def answer(*replies):
        fake = FakeClient(*replies)
        with patch.object(core_llm, "_client", fake), patch.object(s_speech, "synthesize", fake_tts):
            sent = run_turn(server_turns.handle_text, Session(), 1, "good morning")
        assert panels.controller.join_idle(timeout=5)
        return sent

    return SimpleNamespace(at=at, answer=answer, folds=folds)


def test_day_rollover_starts_a_new_thread_quietly_and_answers(rollover):
    rollover.at(datetime(2026, 10, 16, 22, 0, tzinfo=ZONE), datetime(2026, 10, 17, 8, 0, tzinfo=ZONE))
    sent = rollover.answer("[neutral] We planned Friday's deploy.", "[happy] Good morning.")
    conv = conversation._conversation
    assert conv[1] == {"role": "user", "content": "<earlier>We planned Friday's deploy.</earlier>"}
    assert {"type": "thread", "text": "We planned Friday's deploy."} in sent
    assert conversation.NEW_TOPIC_ACK not in json.dumps(sent)                  # no spoken ack
    assert "Saturday, October 17 2026" in conv[0]["content"]
    assert [m["content"] for m in conv[-2:]] == ["good morning", "[happy] Good morning."]
    assert {"type": "transcript", "role": "assistant", "text": "Good morning."} in [
        {k: v for k, v in m.items() if k != "turn"} for m in sent]
    assert rollover.folds and rollover.folds[0][0]["content"] == "plan the deploy"   # yesterday folded
    assert conversation.last_turn_at == datetime(2026, 10, 17, 8, 0, tzinfo=ZONE)


def test_day_rollover_does_not_split_a_chat_across_midnight(rollover):
    rollover.at(datetime(2026, 10, 16, 23, 58, tzinfo=ZONE), datetime(2026, 10, 17, 0, 3, tzinfo=ZONE))
    sent = rollover.answer("[happy] Still here.")
    assert not [m for m in sent if m.get("type") == "thread"]
    assert conversation._conversation[1]["content"] == "plan the deploy"
    assert "Saturday, October 17 2026" in conversation._conversation[0]["content"]


def test_day_rollover_needs_a_previous_turn_and_a_new_date(rollover):
    rollover.at(None, datetime(2026, 10, 17, 8, 0, tzinfo=ZONE))
    assert conversation.day_rollover(datetime(2026, 10, 17, 8, 0, tzinfo=ZONE)) is False
    rollover.at(datetime(2026, 10, 17, 1, 0, tzinfo=ZONE), datetime(2026, 10, 17, 6, 0, tzinfo=ZONE))
    assert conversation.date_changed() is False                            # same date as the prompt
    assert conversation.day_rollover(datetime(2026, 10, 17, 6, 0, tzinfo=ZONE)) is False


def test_new_topic_at_night_then_a_turn_next_morning_refreshes_the_date_without_a_second_thread(rollover):
    rollover.at(datetime(2026, 10, 16, 22, 55, tzinfo=ZONE), datetime(2026, 10, 16, 23, 0, tzinfo=ZONE))
    with patch.object(core_llm, "_client", FakeClient("[neutral] Deploy planning.")), \
            patch.object(s_speech, "synthesize", fake_tts):
        run_turn(server_turns.new_topic_turn, Session(), 1)
    assert panels.controller.join_idle(timeout=5)
    assert conversation.last_turn_at is None
    assert "October 16 2026" in conversation._conversation[0]["content"]
    rollover.at(None, datetime(2026, 10, 17, 8, 0, tzinfo=ZONE))
    a_prompt.prompt_date = date(2026, 10, 16)
    sent = rollover.answer("[happy] Good morning.")
    assert "Saturday, October 17 2026" in conversation._conversation[0]["content"]
    assert not [m for m in sent if m.get("type") == "thread"]
    assert sum(m["content"].startswith("<earlier>") for m in conversation._conversation[1:]) == 1


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

    monkeypatch.setattr(conversation, "save_memory", boom)
    done = []
    c = TurnController(on_turn_complete=panels._periodic_memory)
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
    extra = f"http://localhost:{app.PORT + 1}"  # anything but the loopback:PORT pair
    monkeypatch.setenv("ALLOWED_ORIGINS", extra)
    assert m_ws._origin_allowed(extra) is True
    assert m_ws._origin_allowed(f"http://evil:{app.PORT}") is False
    assert m_ws._origin_allowed(f"http://localhost:{app.PORT}") is True

    monkeypatch.delenv("ALLOWED_ORIGINS")
    assert m_ws._origin_allowed(extra) is False
    assert m_ws._origin_allowed(f"http://localhost:{app.PORT}") is True
    assert m_ws._origin_allowed(f"http://127.0.0.1:{app.PORT}") is True
    assert m_ws._origin_allowed(None) is True


# --------------------------------------------------------------- PRD 46: the briefing turn

NOW = f"{date.today().isoformat()}T12:00:00"
RECAP = "[happy] You added Anna, and I learned two things about you."


def changes(ts: str = None) -> None:
    """One page committed now and two memory ADDs at `ts` (default: just now)."""
    write_page("people/anna", "Anna", "Runs the Friday deploy.\n")
    store._commit("wiki: created people/anna")
    ts = ts or f"{date.today().isoformat()}T11:00:00"
    log_ops(*({"ts": ts, "op": "ADD", "path": "people[]", "value": {"name": name}, "reason": "said"}
              for name in ("Wife", "Ben")))


def brief(text: str = "what's new?", *replies, llm_up: bool = True) -> tuple:
    """One typed turn through `handle_text` at the gather-time stamp NOW: (sent, fake)."""
    fake = FakeClient(*replies)
    with patch.object(core_llm, "_client", fake), patch.object(s_speech, "synthesize", fake_tts), \
            patch.object(util, "stamp", lambda fmt: NOW), patch.object(services, "llm_ok", lambda: llm_up):
        sent = run_turn(server_turns.handle_text, Session(), 1, text)
    return sent, fake


def spoken_text(sent: list) -> list:
    return [m["text"] for m in sent if m["type"] == "transcript" and m["role"] == "assistant"]


@pytest.mark.parametrize("text, expected", [
    ("what's new?", True), ("Brief me.", True), ("what did I miss", True), ("/brief", True),
    ("anything new?", True), ("please, what has changed?", True), ("What is new, please?", True),
    ("what changed", True), ("what's new with you", False), ("brief me on Falcon", False), ("new topic", False)])
def test_briefing_command(text, expected):
    tool = registry.match(text)
    assert (tool is not None and tool.name == "briefing") is expected


def test_briefing_phrasings_start_a_briefing_and_small_talk_reaches_the_model():
    started, asked = [], []
    with patch.dict(registry._HANDLERS, {"briefing": lambda ctx: started.append(ctx.text)}), \
            patch.object(a_turn, "ask_events", lambda text, **kw: asked.append(text) or iter([("delta", "Fine.")])), \
            patch.object(s_speech, "synthesize", fake_tts):
        for text in ("what's new?", "Brief me.", "what did I miss", "/brief", "what's new with you"):
            server_turns.respond(Session(), 1, text, lambda msg: None)
    assert started == ["what's new?", "Brief me.", "what did I miss", "/brief"]
    assert asked == ["what's new with you"]


def test_with_the_llm_offline_only_the_offline_safe_commands_run(vault, fresh_llm):
    """PRD 50 scenario 5: new topic / save note / briefing run before the llm_ok gate; a write-up,
    a page command and a plain turn get the offline composer line and idle, nothing else."""
    ran = []
    stubs = {name: (lambda ctx, name=name: ran.append(name)) for name in
             ("new_topic", "save_note", "briefing", "write_up", "open_page")}
    offline = [{"type": "error", "scope": "turn", "text": panels.llm_offline_text()},
               {"type": "state", "value": "idle"}]
    with patch.dict(registry._HANDLERS, stubs), patch.object(services, "llm_ok", lambda: False), \
            patch.object(a_turn, "ask_events", lambda *a, **kw: ran.append("model") or iter(())):
        for text in ("new topic", "save that as a note", "what's new?"):
            sent = []
            server_turns.respond(Session(), 1, text, sent.append)
            assert sent == []
        for text in ("write it up", "find fallout", "hello there"):
            sent = []
            server_turns.respond(Session(), 1, text, sent.append)
            assert sent == offline, text
    assert ran == ["new_topic", "save_note", "briefing"]


def test_a_briefing_with_changes_is_one_completion_ending_with_the_offer(vault, fresh_llm):
    settings.save_setting("briefing_at", f"{days_ago(1)}T00:00:00")
    changes()
    sent, fake = brief("what's new?", RECAP)
    assert len(fake.calls) == 1
    user = fake.calls[0]["messages"][-1]["content"]
    assert '<document source="briefing">' in user and "Anna" in user and "2 added" in user
    reply = spoken_text(sent)
    assert len(reply) == 1 and reply[0].endswith("Want me to dig deeper, or write it up?")
    assert settings.load_settings()["briefing_at"] == NOW
    assert sent[-1] == {"type": "state", "value": "idle"} and any(m["type"] == "context" for m in sent)


@pytest.mark.parametrize("llm_up", [True, False])
def test_nothing_new_answers_without_the_model(vault, fresh_llm, llm_up):
    seeded = f"{date.today().isoformat()}T09:30:00"
    settings.save_setting("briefing_at", seeded)
    sent, fake = brief("brief me", llm_up=llm_up)
    assert spoken_text(sent) == ["Nothing new since 09:30."]
    assert fake.calls == [] and settings.load_settings()["briefing_at"] == seeded
    assert not [m for m in sent if m["type"] == "error"]


def test_first_briefing_looks_back_one_day(vault, fresh_llm):
    sent, fake = brief("/brief")
    assert spoken_text(sent) == ["Nothing new since yesterday at 12:00."] and fake.calls == []
    assert "briefing_at" not in settings.load_settings()


def test_changes_with_the_llm_offline_are_the_composer_error(vault, fresh_llm):
    seeded = f"{days_ago(1)}T00:00:00"
    settings.save_setting("briefing_at", seeded)
    changes()
    sent, fake = brief("what's new?", RECAP, llm_up=False)
    assert [m for m in sent if m["type"] == "error"] == [
        {"type": "error", "scope": "turn", "text": panels.llm_offline_text()}]
    assert fake.calls == [] and spoken_text(sent) == [] and not [m for m in sent if m["type"] == "speech"]
    assert settings.load_settings()["briefing_at"] == seeded


def test_a_failed_recap_goes_idle_and_keeps_the_cutoff(vault, fresh_llm):
    seeded = f"{days_ago(1)}T00:00:00"
    settings.save_setting("briefing_at", seeded)
    changes()
    sent, fake = brief("what's new?", RuntimeError("oMLX gone"))
    assert len(fake.calls) == 1 and spoken_text(sent) == []
    assert sent[-1] == {"type": "state", "value": "idle"}
    assert settings.load_settings()["briefing_at"] == seeded and conversation._session_turns == []


def test_a_later_cutoff_is_never_moved_back(vault, fresh_llm):
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    later = f"{tomorrow}T07:00:00"
    settings.save_setting("briefing_at", later)
    changes(f"{tomorrow}T08:00:00")
    sent, fake = brief("what's new?", RECAP)
    assert len(fake.calls) == 1 and spoken_text(sent)
    assert settings.load_settings()["briefing_at"] == later


def test_a_briefing_clears_the_offer_skips_recall_and_files_one_stand_in(vault, fresh_llm):
    settings.save_setting("briefing_at", f"{days_ago(1)}T00:00:00")
    changes()
    conversation._deeper_offer = {"recall": "who is Anna?", "title": "notes"}
    sent, fake = brief("what's new?", RECAP)
    assert conversation._deeper_offer is None
    assert all("<recall>" not in m["content"] for m in fake.calls[0]["messages"])
    assert conversation._session_turns == [{"role": "assistant", "content": "briefed: 1 pages, 0 sources, 2 memory changes"}]
    assert conversation._conversation[-2]["content"] == "what's new?" and conversation._conversation[-1]["role"] == "assistant"


MEETING_OPS = json.dumps([{"op": "new_page", "path": f"meetings/{date.today().isoformat()}", "type": "meeting",
                           "name": "Briefing", "summary": "We went over what changed."}])


@pytest.mark.parametrize("turns, filed", [
    ([{"role": "assistant", "content": "briefed: 1 pages, 0 sources, 2 memory changes"},
      {"role": "user", "content": "write it up"}, {"role": "assistant", "content": "wrote up: Briefing 16 Oct"}],
     False),
    ([{"role": "user", "content": "we ship Falcon on Friday"}, {"role": "assistant", "content": "Noted."},
      {"role": "assistant", "content": "briefed: 1 pages, 0 sources, 0 memory changes"}], True),
    ([{"role": "assistant", "content": "briefed: 1 pages, 0 sources, 0 memory changes"},
      {"role": "user", "content": "yes, write it up please"}, {"role": "assistant", "content": "wrote up: Briefing"}],
     False),
    ([{"role": "assistant", "content": "briefed: 1 pages, 0 sources, 0 memory changes"},
      {"role": "user", "content": "Ben is leaving Falcon on Friday, write it up"},
      {"role": "assistant", "content": "wrote up: Briefing"}], True)])
def test_a_session_of_stand_ins_only_is_not_ingested(vault, turns, filed, monkeypatch):
    fake = FakeClient("[]", MEETING_OPS)
    monkeypatch.setattr(core_llm, "_client", fake)
    fold.save(turns)
    assert store.page_exists(f"meetings/{date.today().isoformat()}.md") is filed
    assert len(fake.calls) == (2 if filed else 1)


def test_an_empty_session_is_not_stand_ins_only():
    assert fold.stand_ins_only([]) is False


def test_a_file_source_tool_round_refreshes_the_wiki_panel(vault, monkeypatch):
    """The turn binds its context around the model call, so file_source's panel push reaches the socket."""
    from tests.conftest import fake_model, stream_chunks, tool_call_chunk
    sent = []
    monkeypatch.setattr(panels, "session", Session(websocket=object()))
    monkeypatch.setattr(panels, "_loop", SimpleNamespace(is_running=lambda: True))
    monkeypatch.setattr(panels, "make_sender", lambda lp, turn=None: sent.append)
    monkeypatch.setattr(s_speech, "synthesize", fake_tts)
    monkeypatch.setattr(services, "llm_ok", lambda: True)
    monkeypatch.setattr("ingest.pipeline.ingest_note", lambda text, hint="": {"path": "sources/x", "raw": "raw/x"})
    args = '{"kind": "note", "ref": "buy milk"}'
    with fake_model([tool_call_chunk(id="c1", name="file_source", arguments=args)],
                    stream_chunks("[neutral] Filed.")):
        server_turns.handle_text(Session(), 1, "file this note: buy milk", object())
    assert any(isinstance(m, dict) and m.get("type") == "wiki" for m in sent)
