import sys, os
import contextlib
import copy
import functools
import random
import struct
import tempfile
import zlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Importing the app builds the system prompt, which reads the vault's index — before any fixture can
# sandbox WIKI_PATH. Point it at a path that does not exist, so the import never opens the real vault
# (reading it now opens its search index, docs/53-scalable-index.md). Set before `.env` is loaded.
os.environ["WIKI_PATH"] = os.path.join(tempfile.gettempdir(), f"voicebot-tests-{os.getpid()}", "no-vault")

import json
import time
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import avatars
import chatbot
import index_db
import llm
import memory
import sources
from core import llm as core_llm
from core import settings, trace
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
    yield vault
    index_db.close()           # the search index (PRD 53) is per vault: no handle outlives its test


@pytest.fixture(autouse=True)
def file_sandbox(monkeypatch, tmp_path):
    """No test may touch the real settings.json, trace logs or memory files.

    Every module-level path is pointed at tmp_path; the avatar cache and AVATAR env are
    reset so settings start from a clean slate. `memory_sandbox` below is the same thing
    under the name the memory tests use.
    """
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(avatars, "_current", None)
    monkeypatch.delenv("AVATAR", raising=False)
    monkeypatch.setattr(trace, "TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("TRACE_PROFILE", raising=False)
    trace.reset()
    monkeypatch.setattr(memory, "MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory, "OPS_LOG_PATH", str(tmp_path / "memory_ops.jsonl"))
    monkeypatch.setattr(memory, "SHORTMEM_PATH", str(tmp_path / "shortmem.txt"))
    monkeypatch.setattr(gmail_tool, "GMAIL_CREDENTIALS_PATH", str(tmp_path / "credentials.json"))
    monkeypatch.setattr(gmail_tool, "GMAIL_TOKEN_PATH", str(tmp_path / "token.json"))
    yield tmp_path
    trace.reset()


#: text no prod event may carry: a fixture puts it in what the user said or read, a test greps for it
SENTINEL = "ZQX-SENTINEL"


@pytest.fixture
def prod_profile(monkeypatch):
    """This test's process runs the prod trace profile (no verbose.jsonl, no text)."""
    monkeypatch.setenv("TRACE_PROFILE", "prod")
    trace.reset()


def trace_lines(stream: str = "events", kind: str | None = None) -> list:
    """The parsed lines of this test's events.jsonl (or `stream="verbose"`), optionally one kind."""
    path = trace.paths()[stream]
    if not os.path.exists(path):
        return []
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [r for r in rows if kind is None or r["kind"] == kind]


def trace_text(stream: str = "events") -> str:
    """The raw text of this test's events.jsonl — what a sentinel check greps."""
    path = trace.paths()[stream]
    return open(path).read() if os.path.exists(path) else ""


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


@pytest.fixture(autouse=True)
def model_client_restored(monkeypatch):
    """Whatever a test or helper plants at `core.llm._client` is undone after it."""
    monkeypatch.setattr(core_llm, "_client", core_llm._client)


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
    """The one stand-in OpenAI client: answers each call with the next queued reply ("[]" once empty).

    A str (or None) reply comes back as streamed chunks to a `stream=True` call and as a plain completion
    otherwise; anything else (a `stream_chunks` / `tool_call_chunk` list or iterator, a built
    response) is handed back as-is — iterated for a streamed call — so tool-call round trips can be
    scripted chunk by chunk. A queued `Exception` is raised instead. `delay` sleeps inside the call
    and `entered` is set as soon as it starts — that is how the concurrency tests park a save inside
    the memory lock. `.calls` holds a deep-copied snapshot of every call's kwargs. Patch it in at
    `core.llm._client` (`fake_model`).
    """

    def __init__(self, *replies, delay: float = 0.0, entered=None):
        self._replies = list(replies)
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
        if not self._replies and kw.get("stream"):
            raise IndexError("FakeClient: no reply queued for a streamed call")
        reply = self._replies.pop(0) if self._replies else "[]"
        if isinstance(reply, Exception):
            raise reply
        text = reply is None or isinstance(reply, str)      # None: a completion with no content
        if kw.get("stream"):
            return iter(stream_chunks(reply)) if text else iter(reply)
        if text:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
        return reply


#: legacy spellings of the one fake (docs/50-modular-layout.md, step 4)
StreamClient = FakeClient


def RawStreamClient(responses) -> FakeClient:
    """`FakeClient` over a list of scripted responses (chunk lists, built completions)."""
    return FakeClient(*responses)


@contextlib.contextmanager
def fake_model(*replies, **kw):
    """A `FakeClient(*replies, **kw)` patched in at `core.llm._client` for the block; yields it."""
    fake = FakeClient(*replies, **kw)
    with patch.object(core_llm, "_client", fake):
        yield fake


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


def log_ops(*ops: dict) -> None:
    """Append `ops` to the memory ops log, as `memory` writes them (one JSON line each)."""
    with open(memory.OPS_LOG_PATH, "a") as f:
        f.writelines(json.dumps(o) + "\n" for o in ops)


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


def write_page(rel: str, name: str, body: str, source: str = "seed", extra: str = "") -> None:
    """A minimal note page dropped straight into the vault, bypassing page ops.

    `extra` is raw frontmatter lines (newline-terminated) appended after `seen`.
    """
    text = (f"---\ntype: note\nname: {name}\ntags: []\nsource: {source}\n"
            f"updated: 2026-08-30\nseen: 2026-08-30\n{extra}---\n{body}")
    full = os.path.join(wiki.WIKI_PATH, rel + ".md")
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(text)
    wiki.index_page(rel + ".md")       # the app's write paths index what they write (PRD 53); so does this one


# --------------------------------------------------------------------------- FAQ source fixtures
# Generated FAQ-shaped guides (PRD 32) shared by the recall / grounding / routing tests.

FILLER = """the you your and then head north south east west door room stairs take find kill loot
return talk speak quest reward xp item note area path road bridge lake river hill house town
inside outside before after again later there here cook meal weather rain time clock meeting
plan email letter trip summer friday deploy decide joke tell remind said want next""".split()

FALLOUT_SECTIONS = (
    ("Covenant", ("Honest Dan", "Jacob Orden", "Compound", "SAFE test", "Stockton", "Turrets")),
    ("Diamond City", ("Piper", "Nick Valentine", "Dugout Inn", "Takahashi")),
    ("Sanctuary Hills", ("Codsworth", "Preston Garvey", "Red Rocket", "Concord")),
    ("Goodneighbor", ("Hancock", "Memory Den", "Irma", "Third Rail")),
    ("Fort Hagen", ("Kellogg", "Dogmeat", "Command Center", "Synth")),
    ("Vault 81", ("Overseer Gwen", "Austin", "Curie", "Mole Rat")),
    ("Bunker Hill", ("Kessler", "Deb", "Caravan", "Monument")),
    ("Salem", ("Barney Rook", "Museum of Witchcraft", "Deathclaw", "Rook Family")),
    ("Glowing Sea", ("Virgil", "Crater of Atom", "Rocky Cave", "Hazmat Suit")),
    ("The Castle", ("Ronnie Shaw", "Mirelurk Queen", "Minutemen", "Artillery")),
    ("Lockpicking", ("Bobby Pin", "Sweet Spot", "Master Lock", "Locksmith")),
    ("Hacking", ("Terminal", "Robco", "Likeness", "Science Perk")),
    ("Power Armor", ("Fusion Core", "T-45", "Frame", "Repair Bay")),
    ("Trinity Tower", ("Strong", "Rex Goodman", "Fist", "Elevator")),
    ("Hubris Comics", ("Silver Shroud", "Costume", "Kent Connolly", "Feral Ghoul")),
)

SKYRIM_SECTIONS = (
    ("Whiterun", ("Jarl Balgruuf", "Dragonsreach", "Companions", "Jorrvaskr")),
    ("Riverwood", ("Alvor", "Gerdur", "Sleeping Giant", "Golden Claw")),
    ("Bleak Falls Barrow", ("Draugr", "Arvel", "Word Wall", "Dragon Claw")),
    ("High Hrothgar", ("Greybeards", "Arngeir", "Unrelenting Force", "Seven Thousand Steps")),
    ("Winterhold", ("College of Winterhold", "Tolfdir", "Saarthal", "Arch Mage")),
    ("Riften", ("Thieves Guild", "Brynjolf", "Ragged Flagon", "Maven Black-Briar")),
    ("Solitude", ("Blue Palace", "Elisif", "Bards College", "Castle Dour")),
    ("Markarth", ("Cidhna Mine", "Forsworn", "Silver Blood", "Understone Keep")),
    ("Windhelm", ("Ulfric Stormcloak", "Palace of the Kings", "Grey Quarter", "Butcher")),
    ("Sovngarde", ("Alduin", "Tsun", "Hall of Valor", "Shor")),
    ("Lockpicking", ("Lockpick", "Sweet Spot", "Tumbler", "Skeleton Key")),
    ("Smithing", ("Forge", "Iron Ingot", "Leather Strips", "Warmaidens")),
    ("Dragon Shouts", ("Word of Power", "Dragon Soul", "Thu'um", "Whirlwind Sprint")),
    ("Dawnstar", ("Nightcaller Temple", "Erandur", "Vaermina", "Skull of Corruption")),
    ("Falkreath", ("Dark Brotherhood", "Astrid", "Sanctuary", "Black Door")),
)


@functools.lru_cache(maxsize=None)
def faq_text(game: str, sections: tuple, seed: int, paragraphs: int = 60) -> str:
    """A deterministic FAQ (built once per session — it is ~150 k tokens): a contents list, then one `==== Title ====` section per entry whose
    paragraphs repeat the section's Title-case names between lowercase filler."""
    rng = random.Random(seed)
    out = [f"{game} FAQ and Walkthrough", "", "Contents"]
    out += [f"  {i + 1:02d}. {title}" for i, (title, _) in enumerate(sections)]
    out.append("")
    for title, names in sections:
        out += ["", f"==== {title} ====", ""]
        for _ in range(paragraphs):
            words = []
            for _ in range(rng.randint(90, 130)):
                roll = rng.random()
                if roll < 0.06:
                    words.append(title)
                elif roll < 0.14:
                    words.append(rng.choice(names))
                else:
                    words.append(rng.choice(FILLER))
            out.append(" ".join(words) + ".")
            out.append("")
    return "\n".join(out)


def summary_json(title: str) -> str:
    return json.dumps({"title": title, "summary": f"{title}: a walkthrough.",
                       "entities": [], "topic": "gaming", "unreadable": False})


def file_faq(text: str, title: str, name: str = "guide.md") -> dict:
    with fake_model(summary_json(title), "[]"):
        return sources.ingest_upload(text.encode(), name)


@pytest.fixture
def fallout(seeded_vault):
    """The Fallout-shaped FAQ filed as `Fallout 4 guide`; returns the ingest result."""
    return file_faq(faq_text("Fallout 4", FALLOUT_SECTIONS, seed=4), "Fallout 4 guide")


@pytest.fixture
def both(fallout):
    """Fallout + the Skyrim-shaped FAQ (`Skyrim guide`) in the same vault."""
    skyrim = file_faq(faq_text("Skyrim", SKYRIM_SECTIONS, seed=5), "Skyrim guide", "guide2.md")
    return {"fallout": fallout, "skyrim": skyrim}



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
    with patch.object(core_llm, "_client", fake), patch.object(chatbot, "synthesize", fake_tts):
        sent = run_turn(chatbot.handle_text, Session(), 1, text)
    return sent, fake


def spoken(text: str, reply: str = "[neutral] Sure.") -> tuple:
    """The same through the hands-free path (`handle_utterance` with a fake ASR)."""
    from unittest.mock import patch
    import numpy as np
    from session import Session
    fake = StreamClient(reply)
    with patch.object(core_llm, "_client", fake), patch.object(chatbot, "synthesize", fake_tts), \
            patch.object(chatbot, "asr", fake_asr(text)):
        sent = run_turn(chatbot.handle_utterance, Session(), 1, np.zeros(10, dtype=np.float32))
    return sent, fake


def ask_turn(text: str, *replies, origin: str = "user") -> tuple:
    """One turn through `llm.ask_events` against scripted streamed replies: (events, fake)."""
    from unittest.mock import patch
    fake = StreamClient(*replies)
    with patch.object(core_llm, "_client", fake):
        events = list(llm.ask_events(text, origin=origin))
    return events, fake


def deltas(events: list) -> str:
    """The reply text of an `ask_events` turn: its "delta" payloads joined."""
    return "".join(payload for kind, payload in events if kind == "delta")


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


# --------------------------------------------------------------------------- the reMarkable fakes (PRD 47)


def _serve(handler_cls) -> tuple:
    """A `ThreadingHTTPServer` on 127.0.0.1:0 in a daemon thread -> ``(server, base url)``."""
    import threading
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def notebook_pdf(pages: int = 2, shade: int = 0) -> bytes:
    """An image-only PDF (Pillow, no text layer) of `pages` small flat-colour pages; another
    `shade` gives other pixels (an edited notebook)."""
    import io
    from PIL import Image

    images = [Image.new("RGB", (100, 130), (40 * i, 200, shade % 256)) for i in range(pages)]
    out = io.BytesIO()
    images[0].save(out, "PDF", save_all=True, append_images=images[1:])
    return out.getvalue()


@pytest.fixture
def fake_remarkable():
    """The tablet's USB web interface as documented: `/documents/` (root), `/documents/<id>`
    (a folder) and `/download/<id>/pdf` (two pages, pixels derived from `ModifiedClient`).
    `fake.docs` is ``{id: item}`` with a `parent` key ("" = root); edit it (e.g. bump
    `ModifiedClient`) between pulls. `fake.downloads` logs ids. `POST /upload` (PRD 47b) records
    `{filename, content_type, bytes}` into `fake.uploads`."""
    import zlib
    from http.server import BaseHTTPRequestHandler

    fake = SimpleNamespace(downloads=[], uploads=[], url="", docs={
        "f1": {"ID": "f1", "VissibleName": "wiki-inbox", "Type": "CollectionType", "parent": ""},
        "n1": {"ID": "n1", "VissibleName": "Standup", "Type": "DocumentType", "fileType": "notebook",
               "ModifiedClient": "2026-10-14T09:00:00.000Z", "parent": "f1"},
        "n2": {"ID": "n2", "VissibleName": "Ideas", "Type": "DocumentType", "fileType": "notebook",
               "ModifiedClient": "2026-10-14T10:00:00.000Z", "parent": "f1"},
        "p1": {"ID": "p1", "VissibleName": "Paper", "Type": "DocumentType", "fileType": "pdf",
               "ModifiedClient": "2026-10-14T11:00:00.000Z", "parent": "f1"},
        "n9": {"ID": "n9", "VissibleName": "Outside", "Type": "DocumentType", "fileType": "notebook",
               "ModifiedClient": "2026-10-14T12:00:00.000Z", "parent": ""},
    })
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parts = [p for p in self.path.split("/") if p]
            if parts[:1] == ["documents"] and len(parts) <= 2:
                parent = parts[1] if len(parts) == 2 else ""
                items = [{k: v for k, v in d.items() if k != "parent"}
                         for d in fake.docs.values() if d["parent"] == parent]
                return self._send(200, json.dumps(items).encode(), "application/json")
            if len(parts) == 3 and parts[0] == "download" and parts[2] == "pdf" and parts[1] in fake.docs:
                fake.downloads.append(parts[1])
                doc = fake.docs[parts[1]]
                shade = zlib.crc32((doc["ID"] + doc.get("ModifiedClient", "")).encode())
                return self._send(200, notebook_pdf(2, shade), "application/pdf")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            from email import policy
            from email.parser import BytesParser

            body = self.rfile.read(int(self.headers["Content-Length"]))
            head = f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode()
            for part in BytesParser(policy=policy.default).parsebytes(head + body).iter_parts():
                if self.path == "/upload" and part.get_filename():
                    fake.uploads.append({"filename": part.get_filename(), "content_type": part.get_content_type(),
                                         "bytes": part.get_payload(decode=True)})
                    return self._send(200, b"{}", "application/json")
            self._send(400, b"bad upload", "text/plain")

    server, fake.url = _serve(Handler)
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture
def fake_ingest():
    """A stand-in for the app's `POST /ingest`: every request parsed into `fake.requests`
    (``{fields: {name: str}, files: [(filename, bytes)]}``); a title in `fake.fail` gets a 500."""
    import hashlib
    from email import policy
    from email.parser import BytesParser
    from http.server import BaseHTTPRequestHandler

    fake = SimpleNamespace(requests=[], fail=set(), url="")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            head = f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode()
            msg = BytesParser(policy=policy.default).parsebytes(head + body)
            fields, files = {}, []
            for part in msg.iter_parts():
                name = part.get_param("name", header="content-disposition")
                if part.get_filename():
                    files.append((part.get_filename(), part.get_payload(decode=True)))
                else:
                    fields[name] = part.get_payload(decode=True).decode()
            fake.requests.append({"path": self.path, "fields": fields, "files": files})
            if fields.get("title") in fake.fail:
                status, reply = 500, {"detail": "boom"}
            else:
                slug = wiki.slugify(fields.get("title", ""))
                sha = hashlib.sha256(b"".join(data for _name, data in files)).hexdigest()
                status, reply = 200, {"filed": [{"path": f"sources/{slug}", "raw": "", "already": False,
                                                 "sha256": sha}], "errors": []}
            data = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server, fake.url = _serve(Handler)
    yield fake
    server.shutdown()
    server.server_close()


def free_port() -> int:
    """A port nothing listens on (bound, then released)."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- PDFs (PRD 48, PRD 51)


def pdf_bytes(pages: list, with_text: bool = True) -> bytes:
    """A minimal, valid PDF: one page per entry in `pages`, or no text layer at all."""
    contents, objs, kids = [], [], []
    first_page_obj = 3
    for i, lines in enumerate(pages):
        if with_text:
            stream = ("BT /F1 14 Tf 40 700 Td " +
                      " ".join(f"({line}) Tj 0 -20 Td" for line in lines) + " ET\n")
        else:
            stream = "0 0 0 rg 40 40 100 100 re f\n"
        contents.append(stream)
        kids.append(f"{first_page_obj + i * 2} 0 R")
    objs.append("<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>")
    font_obj = first_page_obj + 2 * len(pages)
    for i, stream in enumerate(contents):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
                    f"/Contents {first_page_obj + i * 2 + 1} 0 R >>")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}endstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


# --------------------------------------------------------------------------- run cards (PRD 51)

#: the run fixture's page: an html-ish title (the page must escape it) and a body no run page may show
RUN_TITLE = "<b>Bold</b> plan"
RUN_BODY = f"The {SENTINEL} body of the plan page."
RUN_SINCE = "2030-01-01T00:00:00"


def briefing_run(monkeypatch, reply: str = "[happy] One new page, the bold plan.", commits_after: bool = True) -> dict:
    """A "what's new?" turn against the sandboxed vault (docs/51-run-cards.md scenario 1): with
    `commits_after`, one page and two `wiki: lint` housekeeping commits land after the cutoff
    (dated 2030, the cutoff is `RUN_SINCE`); without, the digest is empty.
    Returns ``{sent, run_id, card, sha}``: the messages, the turn's run id, the run card, the page's short sha."""
    import services
    from session import Session

    wiki.ensure_vault()
    settings.save_setting("briefing_at", RUN_SINCE)
    monkeypatch.setattr(services, "llm_ok", lambda: True)
    sha = ""
    if commits_after:
        monkeypatch.setenv("GIT_COMMITTER_DATE", "2030-06-01T09:00:00")
        monkeypatch.setenv("GIT_AUTHOR_DATE", "2030-06-01T09:00:00")
        write_page("notes/plan", RUN_TITLE, RUN_BODY)
        wiki._commit("wiki: created notes/plan")
        sha = wiki.head_sha()
        for n in (1, 2):
            wiki._log([f"lint: {n} finding"])
            wiki._commit("wiki: lint")
    with patch.object(core_llm, "_client", FakeClient(reply)), patch.object(chatbot, "synthesize", fake_tts):
        sent = run_turn(chatbot.handle_text, Session(), 1, "what's new?")
    (start,) = trace_lines(kind="turn_start")
    cards = [m for m in sent if m["type"] == "card"]
    return {"sent": sent, "run_id": start["run_id"], "card": cards[-1] if cards else None, "sha": sha}


# --------------------------------------------------------------------------- approvals (PRD 52)
MORROWIND = "Morrowind guide"


@pytest.fixture
def morrowind(seeded_vault):
    """A filed source titled `Morrowind guide` whose ops left a note on `projects/falcon`, plus one
    memory entry learned from it (docs/52-approvals.md). Returns ``{sha, page, raw}``."""
    summary = json.dumps({"title": MORROWIND, "summary": "A Morrowind walkthrough.", "entities": [],
                          "unreadable": False})
    ops = json.dumps([{"op": "append_note", "path": "projects/falcon", "text": "Vivec is in the south."}])
    with fake_model(summary, ops):
        filed = sources.ingest_upload(b"# Morrowind\nVivec city lies in the south.", "morrowind.md")
    seed(people=[{"name": "Vivec", "source": MORROWIND}])
    sha = next(e["sha256"] for e in sources.manifest_entries() if e["path"] == filed["raw"])
    return {"sha": sha, "page": filed["path"], "raw": filed["raw"]}


def session_turn(sess, turn: int, text: str, origin: str = "typed", reply: str = "[neutral] Sure.") -> list:
    """One typed (or `origin="spoken"`, through a fake ASR) turn on a session that outlives it — an
    approval's question and its answer are two turns of one session. The model is up and answers `reply`."""
    import numpy as np
    import services
    with patch.object(core_llm, "_client", FakeClient(reply)), patch.object(chatbot, "synthesize", fake_tts), \
            patch.object(services, "llm_ok", lambda: True), patch.object(chatbot, "asr", fake_asr(text)):
        if origin == "typed":
            return run_turn(chatbot.handle_text, sess, turn, text)
        return run_turn(chatbot.handle_utterance, sess, turn, np.zeros(10, dtype=np.float32))


def button_turn(fn, sess, turn: int, *args) -> list:
    """A queued turn function (`resolve_turn`, `propose_turn`) run as the TurnController runs it: inside
    its own `trace.turn`, with the fake TTS."""
    with patch.object(chatbot, "synthesize", fake_tts), trace.turn(turn):
        return run_turn(fn, sess, turn, *args)


def spoken_lines(sent: list) -> list:
    """The assistant `transcript` lines of a turn's messages — what Wanko said (one per reply)."""
    return [m["text"] for m in sent if m["type"] == "transcript" and m.get("role") == "assistant"]


# --------------------------------------------------------------------------- notebook pages (PRD 47)


def png_bytes(width: int = 4, height: int = 4) -> bytes:
    """A valid greyscale PNG, built by hand — Pillow is not a dependency of this app."""
    raw = b"".join(b"\x00" + bytes([(x * 60) % 256 for x in range(width)]) for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def notebook_pages(n: int) -> list:
    return [png_bytes(4 + i, 4) for i in range(n)]       # distinct bytes per page


# --------------------------------------------------------------------------- research (PRD 54)


def research_run(question: str, *replies, notify=None) -> tuple:
    """One `research.run` inside a turn against scripted plan / judge / write replies: (card, fake)."""
    import research
    with fake_model(*replies) as fake, trace.turn(1):
        card = research.run(question, "typed", notify)
    return card, fake


def judge_reply(hits: list, keep, reason: str = "on topic", drop_reason: str = "off topic") -> str:
    """A judge reply over `hits` (numbered from 1): `keep(hit) -> bool` decides each verdict."""
    return json.dumps([{"id": n + 1, "verdict": "keep" if keep(h) else "drop",
                        "reason": reason if keep(h) else drop_reason} for n, h in enumerate(hits)])


def research_turn(text: str, *replies, origin: str = "typed", sess=None, turn: int = 1) -> tuple:
    """A "research X" turn through the server (typed, or `origin="spoken"` through a fake ASR) against
    scripted plan / judge / write replies, the model up: (sent messages, fake)."""
    import numpy as np
    import services
    from session import Session
    sess = sess or Session()
    with fake_model(*replies) as fake, patch.object(chatbot, "synthesize", fake_tts), \
            patch.object(services, "llm_ok", lambda: True), patch.object(chatbot, "asr", fake_asr(text)):
        if origin == "typed":
            return run_turn(chatbot.handle_text, sess, turn, text), fake
        return run_turn(chatbot.handle_utterance, sess, turn, np.zeros(10, dtype=np.float32)), fake
