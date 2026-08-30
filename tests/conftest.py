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
from tools import gmail as gmail_tool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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
    monkeypatch.setattr(gmail_tool, "GMAIL_CREDENTIALS_PATH", str(tmp_path / "credentials.json"))
    monkeypatch.setattr(gmail_tool, "GMAIL_TOKEN_PATH", str(tmp_path / "token.json"))
    return tmp_path


@pytest.fixture(autouse=True)
def repo_root_guard():
    """Nothing is written next to the code (PRD 40): the repo root listing must not grow."""
    before = set(os.listdir(ROOT))
    yield
    new = set(os.listdir(ROOT)) - before - {".pytest_cache", "__pycache__"}   # pytest's own, not the app's
    assert not new, f"test wrote into the repo root: {sorted(new)}"


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
# Shared by tests/memory/test_memory.py (PRD 15) and tests/memory/test_memory_editor.py (PRD 16).


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

    Shared by tests/memory/test_memory_editor.py, tests/memory/test_memory_decay.py and tests/wiki/test_wiki.py:
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
                # every refusal names the slot it belongs to (tests/session/test_error_scopes.py)
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


# --------------------------------------------------------------------------- git isolation
# Vault commits must never see the developer's global git config (signing, hooks, templates).
os.environ.setdefault("GIT_CONFIG_GLOBAL", os.devnull)
os.environ.setdefault("GIT_CONFIG_NOSYSTEM", "1")


# --------------------------------------------------------------------------- streamed LLM fakes


def stream_chunks(*texts) -> list:
    """Streamed text deltas the way the OpenAI client yields them, one chunk per text."""
    return [SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=t, tool_calls=None), finish_reason=None)]) for t in texts]


def tool_call_chunk(index=0, id=None, name=None, arguments=None, finish_reason=None):
    """One streamed chunk carrying a tool_call fragment."""
    frag = SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=None, tool_calls=[frag]), finish_reason=finish_reason)])


class RawStreamClient:
    """A stand-in OpenAI client that answers each call with the next queued *response*.

    A response is whatever `stream_chunks` / `tool_call_chunk` built (a list or an iterator of
    chunks); a streamed call gets it back as an iterator, so tool-call round trips and error
    paths can be scripted chunk by chunk. A non-streamed call (memory.save's completion) gets
    the queued object as-is. `StreamClient` below is the short form for plain text replies.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(copy.deepcopy(kw))   # a snapshot: the live conversation is stripped after the turn (PRD 39)
        response = self._responses.pop(0)
        return iter(response) if kw.get("stream") else response


class StreamClient(RawStreamClient):
    """One streamed text reply per call; records the messages it was given."""

    def __init__(self, *replies):
        super().__init__([stream_chunks(reply) for reply in replies])


@pytest.fixture
def fresh_llm():
    """A reset conversation on Chat with no remembered page, before and after the test.

    Modules whose every test needs this say `pytestmark = pytest.mark.usefixtures("fresh_llm")`.
    """
    def reset():
        llm.reset()
        llm.set_focus("chat")
        llm.last_page = None
    reset()
    yield
    reset()


# --------------------------------------------------------------------------- vault helpers


@pytest.fixture
def vault(wiki_sandbox, memory_sandbox):
    """A created, committed, empty vault (the autouse fixture only points WIKI_PATH at tmp).

    `memory_sandbox` comes along so an ingest reading the profile (for the user's names, PRD 26)
    can never see the real memory.json.
    """
    wiki.ensure_vault()
    return wiki_sandbox


@pytest.fixture
def seeded_vault(vault):
    """`vault` with the two pages the recall / focus / routing scenarios talk about."""
    wiki.apply_page_ops([
        {"op": "new_page", "path": "people/anna", "type": "person", "name": "Anna",
         "summary": "Colleague on the platform team; runs the Friday deploy."},
        {"op": "new_page", "path": "projects/falcon", "type": "project", "name": "Falcon",
         "summary": "Load-test project, amber."},
    ], "seed")
    return vault


def commits(path: str = None) -> list:
    """`git log --oneline` of the sandboxed vault (optionally for one path), newest first."""
    import subprocess
    args = ["git", "log", "--oneline"] + (["--", path] if path else [])
    out = subprocess.run(args, cwd=wiki.WIKI_PATH, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def read(rel: str) -> str:
    """A vault file's text (`wiki._read`)."""
    return wiki._read(rel)


def write_page(rel: str, name: str, body: str) -> None:
    """A minimal note page dropped straight into the vault, bypassing page ops."""
    text = (f"---\ntype: note\nname: {name}\ntags: []\nsource: seed\n"
            f"updated: 2026-08-30\nseen: 2026-08-30\n---\n{body}")
    full = os.path.join(wiki.WIKI_PATH, rel + ".md")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "w").write(text)


# --------------------------------------------------------------------------- turn helpers


def fake_tts(text, samples: int = 240) -> bytes:
    """A stand-in for `chatbot.synthesize`: a short silent WAV, whatever the text."""
    import io
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(samples, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def run_turn(fn, *args, via: str = "make_sender") -> list:
    """Run one turn function (`fn(*args, loop)`) on a background loop; returns everything it sent.

    `via="make_sender"` captures through `chatbot.make_sender` (typed / spoken turns),
    `via="send"` through the raw `chatbot.send` coroutine (greeting and speech tests).
    """
    import asyncio
    import threading
    from unittest.mock import patch

    sent = []

    async def fake_send(msg):
        sent.append(msg)

    if via == "send":
        capture = patch.object(chatbot, "send", fake_send)
    else:
        capture = patch.object(chatbot, "make_sender", lambda lp, turn=None: sent.append)
    with capture:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True)
        thread.start()
        try:
            fn(*args, loop)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
    return sent


def typed(text: str, reply: str = "[neutral] Sure.") -> tuple:
    """A typed turn through `handle_text` against a mocked model: (sent messages, fake)."""
    from unittest.mock import patch
    from session import Session
    fake = StreamClient(reply)
    with patch.object(llm, "_client", fake), patch.object(chatbot, "synthesize", fake_tts):
        sent = run_turn(chatbot.handle_text, Session(), 1, text)
    return sent, fake


def spoken(text: str, reply: str = "[neutral] Sure.") -> tuple:
    """The same through the hands-free path (`handle_utterance` with a fake ASR)."""
    from unittest.mock import patch
    import numpy as np
    from session import Session
    fake = StreamClient(reply)
    with patch.object(llm, "_client", fake), patch.object(chatbot, "synthesize", fake_tts), \
            patch.object(chatbot, "asr", fake_asr(text)):
        sent = run_turn(chatbot.handle_utterance, Session(), 1, np.zeros(10, dtype=np.float32))
    return sent, fake


def last_user(fake) -> str:
    """The last user message of the fake client's most recent call — what the model saw."""
    return [m for m in fake.calls[-1]["messages"] if m["role"] == "user"][-1]["content"]


# --------------------------------------------------------------------------- websocket helpers


def drain_initial(ws) -> list:
    """Consume the hands_free announce + idle state sent right after connect."""
    return [ws.receive_json() for _ in range(2)]


def send_setting(key, value, client=None) -> dict:
    """Send one set_setting on a (fresh) socket and return the first `setting`/`error` reply."""
    client = client or TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "set_setting", "key": key, "value": value})
        while True:
            msg = ws.receive_json()
            if msg.get("type") in ("setting", "error"):
                return msg


# --------------------------------------------------------------------------- the node client fixture


@pytest.fixture(scope="session")
def client_fixture() -> dict:
    """static/app.js booted once per session in node against tests/app_fixture.mjs; the JSON it prints.

    Every client-side scenario (companion re-parent, typed turn, palette, wiki editor, sources
    chips, intake batch, context meter, error scopes …) runs in that one pass; tests assert on keys.
    """
    import shutil
    import subprocess
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    root = os.path.join(os.path.dirname(__file__), "..")
    out = subprocess.run(["node", "tests/app_fixture.mjs"], cwd=root, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.fixture
def r(client_fixture) -> dict:
    """The short name the client-assertion tests use for `client_fixture`."""
    return client_fixture
