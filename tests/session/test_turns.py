"""PRD 12 scenarios: turn controller, session lifecycle, protocol hardening.

Covers docs/12-turn-controller.md scenarios 1-8, 10 and 12 (2 and 11 live in
test_hands_free.py, 9 in test_avatars.py), and the briefing turn (docs/46-briefing.md scenarios 3-6).
"""
import io
import json
import asyncio
import threading
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import avatars
from core import settings
import chatbot
from tools import commands, registry
import llm
from agent import conversation
from agent import turn as a_turn
import memory
from memory import render
import services
import session as session_mod
import wiki
from session import Session, TurnController
from tests.conftest import Blocker, FakeClient, days_ago, fake_asr, log_ops, run_turn, write_page
from core import llm as core_llm


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

    monkeypatch.setattr(a_turn, "ask_events", fake_ask_events)
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
    monkeypatch.setattr(conversation, "save_memory", lambda: calls.append(1))
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

    monkeypatch.setattr(conversation, "save_memory", slow_save)
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
    monkeypatch.setattr(conversation, "save_memory", lambda: sent.append("saved"))
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

    monkeypatch.setattr(conversation, "save_memory", boom)
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


# --------------------------------------------------------------- PRD 46: the briefing turn

NOW = f"{date.today().isoformat()}T12:00:00"
RECAP = "[happy] You added Anna, and I learned two things about you."


def changes(ts: str = None) -> None:
    """One page committed now and two memory ADDs at `ts` (default: just now)."""
    write_page("people/anna", "Anna", "Runs the Friday deploy.\n")
    wiki._commit("wiki: created people/anna")
    ts = ts or f"{date.today().isoformat()}T11:00:00"
    log_ops(*({"ts": ts, "op": "ADD", "path": "people[]", "value": {"name": name}, "reason": "said"}
              for name in ("Wife", "Ben")))


def brief(text: str = "what's new?", *replies, llm_up: bool = True) -> tuple:
    """One typed turn through `handle_text` at the gather-time stamp NOW: (sent, fake)."""
    fake = FakeClient(*replies)
    with patch.object(core_llm, "_client", fake), patch.object(chatbot, "synthesize", fake_tts), \
            patch.object(render, "_stamp", lambda fmt: NOW), patch.object(services, "llm_ok", lambda: llm_up):
        sent = run_turn(chatbot.handle_text, Session(), 1, text)
    return sent, fake


def spoken_text(sent: list) -> list:
    return [m["text"] for m in sent if m["type"] == "transcript" and m["role"] == "assistant"]


@pytest.mark.parametrize("text, expected", [
    ("what's new?", True), ("Brief me.", True), ("what did I miss", True), ("/brief", True),
    ("anything new?", True), ("please, what has changed?", True), ("What is new, please?", True),
    ("what changed", True), ("what's new with you", False), ("brief me on Falcon", False), ("new topic", False)])
def test_briefing_command(text, expected):
    assert commands.briefing_command(text) is expected


def test_briefing_phrasings_start_a_briefing_and_small_talk_reaches_the_model():
    started, asked = [], []
    with patch.dict(registry._HANDLERS, {"briefing": lambda ctx: started.append(ctx.text)}), \
            patch.object(a_turn, "ask_events", lambda text, **kw: asked.append(text) or iter([("delta", "Fine.")])), \
            patch.object(chatbot, "synthesize", fake_tts):
        for text in ("what's new?", "Brief me.", "what did I miss", "/brief", "what's new with you"):
            chatbot.respond(Session(), 1, text, lambda msg: None)
    assert started == ["what's new?", "Brief me.", "what did I miss", "/brief"]
    assert asked == ["what's new with you"]


def test_with_the_llm_offline_only_the_offline_safe_commands_run(vault, fresh_llm):
    """PRD 50 scenario 5: new topic / save note / briefing run before the llm_ok gate; a write-up,
    a page command and a plain turn get the offline composer line and idle, nothing else."""
    ran = []
    stubs = {name: (lambda ctx, name=name: ran.append(name)) for name in
             ("new_topic", "save_note", "briefing", "write_up", "open_page")}
    offline = [{"type": "error", "scope": "turn", "text": chatbot.llm_offline_text()},
               {"type": "state", "value": "idle"}]
    with patch.dict(registry._HANDLERS, stubs), patch.object(services, "llm_ok", lambda: False), \
            patch.object(a_turn, "ask_events", lambda *a, **kw: ran.append("model") or iter(())):
        for text in ("new topic", "save that as a note", "what's new?"):
            sent = []
            chatbot.respond(Session(), 1, text, sent.append)
            assert sent == []
        for text in ("write it up", "find fallout", "hello there"):
            sent = []
            chatbot.respond(Session(), 1, text, sent.append)
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
        {"type": "error", "scope": "turn", "text": chatbot.llm_offline_text()}]
    assert fake.calls == [] and spoken_text(sent) == [] and not [m for m in sent if m["type"] == "speech"]
    assert settings.load_settings()["briefing_at"] == seeded


def test_a_failed_recap_goes_idle_and_keeps_the_cutoff(vault, fresh_llm):
    seeded = f"{days_ago(1)}T00:00:00"
    settings.save_setting("briefing_at", seeded)
    changes()
    sent, fake = brief("what's new?", RuntimeError("oMLX gone"))
    assert len(fake.calls) == 1 and spoken_text(sent) == []
    assert sent[-1] == {"type": "state", "value": "idle"}
    assert settings.load_settings()["briefing_at"] == seeded and llm._session_turns == []


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
    assert llm._deeper_offer is None
    assert all("<recall>" not in m["content"] for m in fake.calls[0]["messages"])
    assert llm._session_turns == [{"role": "assistant", "content": "briefed: 1 pages, 0 sources, 2 memory changes"}]
    assert llm._conversation[-2]["content"] == "what's new?" and llm._conversation[-1]["role"] == "assistant"


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
    memory.save(turns)
    assert wiki.page_exists(f"meetings/{date.today().isoformat()}.md") is filed
    assert len(fake.calls) == (2 if filed else 1)


def test_an_empty_session_is_not_stand_ins_only():
    assert memory.stand_ins_only([]) is False


def test_a_file_source_tool_round_refreshes_the_wiki_panel(vault, monkeypatch):
    """The turn binds its context around the model call, so file_source's panel push reaches the socket."""
    from tests.conftest import fake_model, stream_chunks, tool_call_chunk
    sent = []
    monkeypatch.setattr(chatbot, "session", Session(websocket=object()))
    monkeypatch.setattr(chatbot, "_loop", SimpleNamespace(is_running=lambda: True))
    monkeypatch.setattr(chatbot, "make_sender", lambda lp, turn=None: sent.append)
    monkeypatch.setattr(chatbot, "synthesize", fake_tts)
    monkeypatch.setattr(services, "llm_ok", lambda: True)
    monkeypatch.setattr("ingest.pipeline.ingest_note", lambda text, hint="": {"path": "sources/x", "raw": "raw/x"})
    args = '{"kind": "note", "ref": "buy milk"}'
    with fake_model([tool_call_chunk(id="c1", name="file_source", arguments=args)],
                    stream_chunks("[neutral] Filed.")):
        chatbot.handle_text(Session(), 1, "file this note: buy milk", object())
    assert any(isinstance(m, dict) and m.get("type") == "wiki" for m in sent)
