import sys, os
import copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import time
from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import avatars
import chatbot
import llm
import memory
import wiki


@pytest.fixture(autouse=True)
def wiki_sandbox(monkeypatch, tmp_path):
    """No test may touch the real ~/llm-wiki: every one gets its own empty vault path.

    Nothing creates it — `wiki.ensure_vault()` is lazy — so a test that never writes to the
    wiki pays nothing for this.
    """
    vault = tmp_path / "wiki"
    monkeypatch.setattr(wiki, "WIKI_PATH", str(vault))
    monkeypatch.setattr(wiki, "_index_cache", {"sig": None, "docs": []})
    monkeypatch.setattr(wiki, "_chunk_cache", {"sig": None, "docs": [], "proper": {}})
    return vault


@pytest.fixture(autouse=True)
def file_sandbox(monkeypatch, tmp_path):
    """No test may touch the real settings.json, session.log or memory files.

    Every module-level path is pointed at tmp_path; the avatar cache and AVATAR env are
    reset so settings start from a clean slate. `memory_sandbox` below is the same thing
    under the name the memory tests use.
    """
    monkeypatch.setattr(avatars, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(avatars, "_current", None)
    monkeypatch.delenv("AVATAR", raising=False)
    monkeypatch.setattr(llm, "LOG_PATH", str(tmp_path / "session.log"))
    monkeypatch.setattr(memory, "MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory, "OPS_LOG_PATH", str(tmp_path / "memory_ops.jsonl"))
    monkeypatch.setattr(memory, "SHORTMEM_PATH", str(tmp_path / "shortmem.txt"))
    return tmp_path


@pytest.fixture(autouse=True)
def reset_chatbot_session():
    """Per-connection state now lives on chatbot.session; make sure no test inherits one.

    Also waits for the turn worker to drain so a slow turn cannot bleed into the next test.
    """
    chatbot.session = None
    chatbot.greeted = True  # tests that want the greeting turn set this back to False
    # The turn counter drives the periodic memory save; zero it so one test's turns cannot
    # push an unexpected `memory` snapshot into the next test's socket.
    chatbot.controller.completed = 0
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


def fake_asr(reply="", prompts=None) -> SimpleNamespace:
    """A stand-in for `chatbot.asr`: `transcribe(audio, prompt=None)` without mlx-whisper.

    `reply` is the transcript — a string, or a callable over the audio for tests that need one
    per utterance. Pass a list as `prompts` to record the `initial_prompt` each call was given.
    """

    def transcribe(audio, prompt=None):
        if prompts is not None:
            prompts.append(prompt)
        return reply(audio) if callable(reply) else reply

    return SimpleNamespace(transcribe=transcribe)


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
        self.calls.append(copy.deepcopy(kw))   # a snapshot: the live conversation is stripped after the turn (PRD 39)
        if self.entered is not None:
            self.entered.set()
        if self.delay:
            time.sleep(self.delay)
        content = self._contents.pop(0) if self._contents else "[]"
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def memory_sandbox(file_sandbox):
    """The tmp_path every memory file already points at (see `file_sandbox`)."""
    return file_sandbox


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
    """A websocket client that only cares about the memory / wiki replies.

    Shared by tests/test_memory_editor.py, tests/test_memory_decay.py and tests/test_wiki.py:
    `send` waits for one reply, `send_many` for the `n` an action answers with.
    """

    WANTED = ("memory", "memory_error", "wiki", "wiki_page", "wiki_conflict", "wiki_search", "error")

    def __init__(self, ws):
        self.ws = ws

    def send(self, msg: dict) -> dict:
        return self.send_many(msg, 1)[0]

    def send_many(self, msg: dict, count: int) -> list:
        self.ws.send_json(msg)
        got = []
        while len(got) < count:
            reply = self.ws.receive_json()
            if reply.get("type") == "error":
                # every refusal names the slot it belongs to (tests/test_error_scopes.py)
                assert reply.get("scope") in chatbot.ERROR_SCOPES, reply
            if reply.get("type") in self.WANTED:
                got.append(reply)
        return got


@pytest.fixture
def panel():
    """A connected memory panel — the websocket the editor tests drive."""
    chatbot.greeted = True
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        yield Panel(ws)
