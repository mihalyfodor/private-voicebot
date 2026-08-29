"""Unit tests for the memory editor — mocked LLM, tmp memory path. Mirrors docs/16-memory-editor.md.

Scenarios 1-8 of the PRD plus the two unit items from its "Stress" section (random-op fuzz and
the concurrent hammer). Everything here goes through the same `apply_ops` path an LLM op takes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import random
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import chatbot
import memory


# --------------------------------------------------------------------------- helpers


class FakeClient:
    """Queued completion contents; `delay` sleeps inside the call (concurrency tests)."""

    def __init__(self, *contents, delay: float = 0.0, entered: threading.Event | None = None):
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
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory, "OPS_LOG_PATH", str(tmp_path / "memory_ops.jsonl"))
    monkeypatch.setattr(memory, "SHORTMEM_PATH", str(tmp_path / "shortmem.txt"))
    return tmp_path


def _profile(**kw):
    p = memory._empty_profile()
    p.update(kw)
    return p


def _seed(**kw) -> dict:
    profile = _profile(**kw)
    memory._write_profile(profile)
    return profile


def _audit() -> list:
    if not os.path.exists(memory.OPS_LOG_PATH):
        return []
    return [json.loads(line) for line in open(memory.OPS_LOG_PATH) if line.strip()]


def _on_disk() -> dict:
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
    chatbot.greeted = True
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        yield Panel(ws)


# --------------------------------------------------------------- 1. update a scalar


def test_update_identity_name_supersedes_and_audits_as_ui(panel):
    _seed(identity={"name": "Misi"})
    reply = panel.send({"action": "memory_op", "op": "UPDATE",
                        "path": "identity.name", "value": "Mihaly"})

    assert reply["type"] == "memory"
    assert reply["profile"]["identity"]["name"] == "Mihaly"
    assert reply["profile"]["identity"]["superseded"]["name"] == ["Misi"]
    assert _on_disk()["identity"]["name"] == "Mihaly"

    logged = _audit()
    assert len(logged) == 1
    assert (logged[0]["op"], logged[0]["path"], logged[0]["reason"]) == ("UPDATE", "identity.name", "ui")


def test_restore_is_tagged_ui_restore(panel):
    _seed(identity={"location": "Vienna", "superseded": {"location": ["Budapest"]}})
    reply = panel.send({"action": "memory_op", "op": "UPDATE", "path": "identity.location",
                        "value": "Budapest", "reason": "ui:restore"})
    assert reply["profile"]["identity"]["location"] == "Budapest"
    assert _audit()[0]["reason"] == "ui:restore"


def test_an_unknown_reason_tag_cannot_be_injected_into_the_log(panel):
    _seed(identity={"name": "Misi"})
    panel.send({"action": "memory_op", "op": "UPDATE", "path": "identity.name",
                "value": "Mihaly", "reason": "the model said so"})
    assert _audit()[0]["reason"] == "ui"


# --------------------------------------------------------------- 2. delete a list entry


def test_delete_person_by_index_removes_the_right_one(panel):
    _seed(people=[{"name": "Anna"}, {"name": "Bolt", "rel": "dog"}, {"name": "Csaba"}])
    reply = panel.send({"action": "memory_op", "op": "DELETE", "path": "people[1]"})

    assert [p["name"] for p in reply["profile"]["people"]] == ["Anna", "Csaba"]
    assert [p["name"] for p in _on_disk()["people"]] == ["Anna", "Csaba"]
    assert _audit()[0]["path"] == "people[1]"


def test_delete_out_of_range_is_refused_and_leaves_the_file_alone(panel):
    _seed(people=[{"name": "Anna"}, {"name": "Bolt"}, {"name": "Csaba"}])
    before = _on_disk()
    reply = panel.send({"action": "memory_op", "op": "DELETE", "path": "people[7]"})

    assert reply["type"] == "memory_error"
    assert reply["path"] == "people[7]"
    assert "does not exist" in reply["reason"]
    assert _on_disk() == before
    assert _audit() == []


# --------------------------------------------------------------- 3. cap


def test_add_beyond_the_cap_is_refused_with_the_cap_in_the_reason(panel):
    _seed(people=[{"name": f"p{i}"} for i in range(memory.CAPS["people"])])
    before = _on_disk()
    reply = panel.send({"action": "memory_op", "op": "ADD", "path": "people[]",
                        "value": {"name": "one too many"}})

    assert reply["type"] == "memory_error"
    assert str(memory.CAPS["people"]) in reply["reason"] and "cap" in reply["reason"]
    assert _on_disk() == before
    assert _audit() == []


# --------------------------------------------------------------- 4. malformed ops


@pytest.mark.parametrize("op", [
    {"op": "REPLACE", "path": "identity.name", "value": "x"},
    {"op": "UPDATE", "path": "identity/name", "value": "x"},
    {"op": "ADD", "path": "identity.name", "value": None},
    {"op": "ADD", "path": "hobbies.chess", "value": "yes"},
    {"op": "ADD", "path": "people[]", "value": {"rel": "friend"}},
])
def test_malformed_op_is_refused_without_writing_or_logging(panel, op):
    _seed(identity={"name": "Misi"})
    before, before_mtime = _on_disk(), os.stat(memory.MEMORY_PATH).st_mtime_ns

    reply = panel.send({"action": "memory_op", **op})

    assert reply["type"] == "memory_error" and reply["reason"]
    assert reply["path"] == op["path"]
    assert _on_disk() == before
    assert os.stat(memory.MEMORY_PATH).st_mtime_ns == before_mtime
    assert not os.path.exists(memory.OPS_LOG_PATH)


# --------------------------------------------------------------- 5. get_memory


def test_get_memory_returns_the_rendered_block_and_the_visible_fact_count(panel):
    profile = _seed(
        identity={"name": "Mihaly", "superseded": {"name": ["Misi"]}},
        preferences={"coffee": "black"},
        people=[{"name": "Anna"}, {"name": "Bolt"}],
        projects=[{"name": "half marathon"}],
        recurring=[{"what": "gym", "when": "Tue/Thu"}],
        episodic=[{"date": "2026-08-01", "text": "ran a 10k", "ttl_days": 90, "importance": 3}],
    )
    reply = panel.send({"action": "get_memory"})

    assert reply["type"] == "memory"
    assert reply["rendered"] == memory.render(memory.load_profile())
    assert reply["count"] == 7            # 2 scalars + 2 people + 1 project + 1 recurring + 1 episodic
    assert reply["profile"]["identity"]["superseded"] == {"name": ["Misi"]}
    assert "meta" not in reply["profile"] and "version" not in reply["profile"]
    assert memory.fact_count(profile) == 7


def test_snapshot_flags_expired_episodics_without_touching_the_file():
    profile = _seed(episodic=[
        {"date": "2020-01-01", "text": "old news", "ttl_days": 5, "importance": 2},
        {"date": memory._today().isoformat(), "text": "today", "ttl_days": 30, "importance": 2},
    ])
    snap = memory.snapshot(profile)
    assert [e["expired"] for e in snap["profile"]["episodic"]] == [True, False]
    assert snap["count"] == 2
    assert all("expired" not in e for e in _on_disk()["episodic"])
    assert all("expired" not in e for e in profile["episodic"])


# --------------------------------------------------------------- 6. reset


def test_memory_reset_empties_the_file_and_logs_one_delete_all(panel):
    _seed(identity={"name": "Mihaly"}, people=[{"name": "Anna"}])
    reply = panel.send({"action": "memory_reset"})

    assert reply["count"] == 0
    assert reply["rendered"] == ""
    on_disk = _on_disk()
    assert {k: v for k, v in on_disk.items() if k != "updated"} == \
           {k: v for k, v in memory._empty_profile().items() if k != "updated"}
    logged = _audit()
    assert len(logged) == 1
    assert (logged[0]["op"], logged[0]["path"], logged[0]["reason"]) == ("DELETE", "*", "ui:reset")


# --------------------------------------------------------------- 7. concurrency


def test_a_hand_edit_during_a_background_save_is_not_overwritten():
    """The lock must span load -> propose -> write in save(); without it the edit is lost."""
    _seed(identity={"name": "Misi"})
    entered = threading.Event()
    client = FakeClient(
        json.dumps([{"op": "ADD", "path": "preferences.coffee", "value": "black"}]),
        delay=0.3, entered=entered,
    )
    saver = threading.Thread(
        target=memory.save,
        args=([{"role": "user", "content": "I drink black coffee"}], client, "m"))
    saver.start()
    assert entered.wait(timeout=5), "the save never reached the LLM call"

    profile, reason = memory.edit({"op": "UPDATE", "path": "identity.name", "value": "Mihaly"})
    saver.join(timeout=10)

    assert reason is None
    final = _on_disk()
    assert final["identity"]["name"] == "Mihaly"        # the hand edit survived
    assert final["preferences"]["coffee"] == "black"    # so did the background save
    assert sorted(op["path"] for op in _audit()) == ["identity.name", "preferences.coffee"]


# --------------------------------------------------------------- 8. rejection collector


def test_apply_ops_collects_the_same_reasons_it_prints(capsys):
    ops = [
        {"op": "FROBNICATE", "path": "identity.name", "value": "x"},
        {"op": "ADD", "path": "hobbies.chess", "value": "yes"},
        {"op": "ADD", "path": "identity.age", "value": 34},
        {"op": "ADD", "path": "people[]", "value": {"rel": "friend"}},
        {"op": "ADD", "path": "identity.name", "value": "Mihaly"},   # this one lands
    ]
    rejected = []
    profile, applied = memory.apply_ops(memory._empty_profile(), ops, rejected)

    assert len(applied) == 1 and profile["identity"]["name"] == "Mihaly"
    assert [r["op"] for r in rejected] == ops[:4]
    printed = capsys.readouterr().out
    assert len(rejected) == printed.count("[Memory] rejected op")
    for entry in rejected:
        assert entry["reason"] and entry["reason"] in printed


def test_apply_ops_without_a_collector_still_only_prints(capsys):
    profile, applied = memory.apply_ops(memory._empty_profile(),
                                        [{"op": "ADD", "path": "nope.k", "value": "v"}])
    assert applied == [] and "[Memory] rejected op" in capsys.readouterr().out


# --------------------------------------------------------------- stress: fuzz


def _random_op(rng: random.Random, profile: dict) -> dict:
    """A random op — valid or not — over every section and path form the editor can produce."""
    kind = rng.choice(["ADD", "ADD", "UPDATE", "UPDATE", "DELETE", "REPLACE", "NOOP"])
    section = rng.choice(list(memory.SCALAR_SECTIONS + memory.LIST_SECTIONS) + ["hobbies"])
    if section in memory.SCALAR_SECTIONS or section == "hobbies":
        key = rng.choice(["name", "nickname", "location", "coffee", "k%d" % rng.randrange(5), ""])
        value = rng.choice(["Mihaly", "Budapest", "black", "Vienna", "  ", 42, None, {"x": 1}])
        return {"op": kind, "path": f"{section}.{key}" if key else section, "value": value}
    key_field = memory.LIST_KEY.get(section, "name")
    index = rng.randrange(0, max(1, len(profile.get(section, [])) + 2))
    path = rng.choice([f"{section}[]", section, f"{section}[{index}]",
                       f"{section}[{index}].note"])
    value = rng.choice([
        {key_field: "e%d" % rng.randrange(8), "note": "n%d" % rng.randrange(3)},
        {key_field: "e%d" % rng.randrange(8), "importance": rng.randrange(1, 4)},
        {key_field: "e%d" % rng.randrange(8), "when": "Tue", "rel": "friend", "status": "wip"},
        {"note": "orphan"}, "not a dict", None, 7,
    ])
    return {"op": kind, "path": path, "value": value}


def test_fuzz_two_hundred_random_ops_keep_the_profile_valid():
    rng = random.Random(1616)
    applied = 0
    for _ in range(200):
        profile, reason = memory.edit(_random_op(rng, memory.load_profile()))
        applied += reason is None

        reloaded = memory.load_profile()
        assert reloaded == memory._normalise(reloaded)          # survives a round-trip
        for section, cap in memory.CAPS.items():
            assert len(reloaded[section]) <= cap
        for section in memory.SCALAR_SECTIONS:
            for key, history in reloaded[section].get("superseded", {}).items():
                assert isinstance(key, str)
                assert all(isinstance(v, str) for v in history)
            assert all(isinstance(v, str) for k, v in reloaded[section].items()
                       if k != "superseded")

    assert applied > 20, "the generator produced almost nothing valid"
    assert len(_audit()) == applied            # one audit line per applied op


# --------------------------------------------------------------- stress: concurrent hammer


def test_four_threads_hammering_edits_during_a_save_keep_the_file_intact():
    _seed(identity={"name": "Misi"})
    applied = []
    lock = threading.Lock()

    def hammer(tid: int):
        local = 0
        for i in range(50):
            _, reason = memory.edit({"op": "ADD", "path": f"preferences.t{tid}_{i}", "value": "v"})
            local += reason is None
        with lock:
            applied.append(local)

    client = FakeClient(json.dumps([{"op": "ADD", "path": "people[]", "value": {"name": "Anna"}}]),
                        delay=0.05)
    threads = [threading.Thread(target=hammer, args=(t,)) for t in range(4)]
    saver = threading.Thread(target=lambda: applied.append(
        len(memory.save([{"role": "user", "content": "Anna is a colleague"}], client, "m"))))
    for t in threads + [saver]:
        t.start()
    for t in threads + [saver]:
        t.join(timeout=30)
        assert not t.is_alive()

    final = _on_disk()                          # valid JSON, no torn write
    assert len([k for k in final["preferences"] if k != "superseded"]) == 200
    assert [p["name"] for p in final["people"]] == ["Anna"]
    assert sum(applied) == 201 == len(_audit())
