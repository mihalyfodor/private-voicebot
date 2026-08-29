import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import chatbot
import memory


@pytest.fixture(autouse=True)
def reset_chatbot_session():
    """Per-connection state now lives on chatbot.session; make sure no test inherits one.

    Also waits for the turn worker to drain so a slow turn cannot bleed into the next test.
    """
    chatbot.session = None
    chatbot.greeted = True  # tests that want the greeting turn set this back to False
    yield
    chatbot.controller.join_idle(timeout=5)
    chatbot.session = None


def untagged(msg: dict) -> dict:
    """A received message without its turn id (for equality asserts that don't care)."""
    return {k: v for k, v in msg.items() if k != "turn"}


class Blocker:
    """A turn that parks the worker until released — makes `controller.busy` deterministic."""

    def __init__(self):
        import threading
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args):
        self.started.set()
        self.release.wait(timeout=10)

    def hold(self, controller):
        controller.submit(self)
        assert self.started.wait(timeout=5), "blocking turn never started"

    def free(self, controller):
        self.release.set()
        controller.join_idle(timeout=5)


# --------------------------------------------------------------------------- memory helpers
# Shared by tests/test_memory.py (PRD 15) and tests/test_memory_editor.py (PRD 16).


class FakeClient:
    """A stand-in OpenAI client: returns the queued completion contents, one per call.

    A queued `Exception` is raised instead of returned. `delay` sleeps inside the call and
    `entered` is set as soon as it starts — that is how the concurrency tests park a save
    inside the memory lock.
    """

    def __init__(self, *contents, delay: float = 0.0, entered=None):
        self._contents = list(contents)
        self.delay = delay
        self.entered = entered
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.entered is not None:
            self.entered.set()
        if self.delay:
            time.sleep(self.delay)
        content = self._contents.pop(0) if self._contents else "[]"
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def memory_sandbox(monkeypatch, tmp_path):
    """Point every memory file at tmp_path, so no test can touch the real memory.json."""
    monkeypatch.setattr(memory, "MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory, "OPS_LOG_PATH", str(tmp_path / "memory_ops.jsonl"))
    monkeypatch.setattr(memory, "SHORTMEM_PATH", str(tmp_path / "shortmem.txt"))
    return tmp_path


def empty_profile(**sections) -> dict:
    """A v2 profile with the given sections filled in (not written to disk)."""
    profile = memory._empty_profile()
    profile.update(sections)
    return profile


def days_ago(n: int) -> str:
    """An ISO date `n` days before today — episodic `date` and entry `seen` values."""
    return (date.today() - timedelta(days=n)).isoformat()


def seed(**sections) -> dict:
    """Write a profile with `sections` filled in — the starting point of an editor scenario."""
    profile = empty_profile(**sections)
    memory._write_profile(profile)
    return profile


def audit_log() -> list:
    """Every line of the sandboxed memory_ops.jsonl, parsed."""
    if not os.path.exists(memory.OPS_LOG_PATH):
        return []
    return [json.loads(line) for line in open(memory.OPS_LOG_PATH) if line.strip()]


def on_disk() -> dict:
    """The sandboxed memory.json as written (no normalisation)."""
    return json.load(open(memory.MEMORY_PATH))


class Panel:
    """A websocket client that only cares about `memory` / `memory_error` messages."""

    def __init__(self, ws):
        self.ws = ws

    def send(self, msg: dict) -> dict:
        self.ws.send_json(msg)
        while True:
            reply = self.ws.receive_json()
            if reply.get("type") in ("memory", "memory_error", "error"):
                return reply


@pytest.fixture
def panel():
    """A connected memory panel — the websocket the editor tests drive."""
    chatbot.greeted = True
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        yield Panel(ws)
