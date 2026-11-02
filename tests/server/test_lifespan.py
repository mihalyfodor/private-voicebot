"""The app's lifespan: startup runs the search index check, shutdown saves session memory off the event loop."""
import asyncio
import threading

from fastapi.testclient import TestClient

from agent import conversation
from server import app, panels
from vault import search as v_search


def test_lifespan_runs_startup_check_then_save_memory(monkeypatch):
    events = []
    monkeypatch.setattr(v_search, "startup_check", lambda: events.append("startup") or 0)
    monkeypatch.setattr(conversation, "save_memory", lambda: events.append("save"))
    with TestClient(app.app):
        assert events == ["startup"]
    assert events == ["startup", "save"]


def test_failing_save_goes_to_memory_failed_and_does_not_raise(monkeypatch):
    failures = []
    monkeypatch.setattr(v_search, "startup_check", lambda: 0)

    def boom():
        raise RuntimeError("llm down")
    monkeypatch.setattr(conversation, "save_memory", boom)
    monkeypatch.setattr(panels, "_memory_failed", failures.append)
    with TestClient(app.app):
        pass
    assert len(failures) == 1 and str(failures[0]) == "llm down"


def test_save_memory_runs_off_the_event_loop(monkeypatch):
    seen = []
    monkeypatch.setattr(v_search, "startup_check", lambda: 0)

    def save():
        try:
            asyncio.get_running_loop()
            seen.append("on loop")
        except RuntimeError:
            seen.append("off loop")
        seen.append(threading.current_thread() is not threading.main_thread())
    monkeypatch.setattr(conversation, "save_memory", save)
    with TestClient(app.app):
        pass
    assert seen == ["off loop", True]
