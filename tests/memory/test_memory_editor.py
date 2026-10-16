"""Unit tests for the memory editor — mocked LLM, tmp memory path. Mirrors docs/16-memory-editor.md.

Scenarios 1-8 of the PRD plus the two unit items from its "Stress" section (random-op fuzz and
the concurrent hammer). Everything here goes through the same `apply_ops` path an LLM op takes.
"""
import os

import json
import random
import threading

import pytest

import chatbot
import llm
import memory
from core import llm as core_llm
from tests.conftest import Blocker, FakeClient, audit_log, on_disk, seed


# --------------------------------------------------------------------------- helpers


# --------------------------------------------------------------- 1. update a scalar


def test_update_identity_name_supersedes_and_audits_as_ui(panel):
    seed(identity={"name": "Misi"})
    reply = panel.send({"action": "memory_op", "op": "UPDATE",
                        "path": "identity.name", "value": "Mihaly"})

    assert reply["type"] == "memory"
    assert reply["profile"]["identity"]["name"] == "Mihaly"
    assert reply["profile"]["identity"]["superseded"]["name"] == ["Misi"]
    assert on_disk()["identity"]["name"] == "Mihaly"

    logged = audit_log()
    assert len(logged) == 1
    assert (logged[0]["op"], logged[0]["path"], logged[0]["reason"]) == ("UPDATE", "identity.name", "ui")


def test_restore_is_tagged_ui_restore(panel):
    seed(identity={"location": "Vienna", "superseded": {"location": ["Budapest"]}})
    reply = panel.send({"action": "memory_op", "op": "UPDATE", "path": "identity.location",
                        "value": "Budapest", "reason": "ui:restore"})
    assert reply["profile"]["identity"]["location"] == "Budapest"
    assert audit_log()[0]["reason"] == "ui:restore"


def test_an_unknown_reason_tag_cannot_be_injected_into_the_log(panel):
    seed(identity={"name": "Misi"})
    panel.send({"action": "memory_op", "op": "UPDATE", "path": "identity.name",
                "value": "Mihaly", "reason": "the model said so"})
    assert audit_log()[0]["reason"] == "ui"


# --------------------------------------------------------------- 2. delete a list entry


def test_delete_person_by_index_removes_the_right_one(panel):
    seed(people=[{"name": "Anna"}, {"name": "Bolt", "rel": "dog"}, {"name": "Csaba"}])
    reply = panel.send({"action": "memory_op", "op": "DELETE", "path": "people[1]"})

    assert [p["name"] for p in reply["profile"]["people"]] == ["Anna", "Csaba"]
    assert [p["name"] for p in on_disk()["people"]] == ["Anna", "Csaba"]
    assert audit_log()[0]["path"] == "people[1]"


def test_delete_out_of_range_is_refused_and_leaves_the_file_alone(panel):
    seed(people=[{"name": "Anna"}, {"name": "Bolt"}, {"name": "Csaba"}])
    before = on_disk()
    reply = panel.send({"action": "memory_op", "op": "DELETE", "path": "people[7]"})

    assert reply["type"] == "memory_error"
    assert reply["path"] == "people[7]"
    assert "does not exist" in reply["reason"]
    assert on_disk() == before
    assert audit_log() == []


# --------------------------------------------------------------- 3. cap


def test_add_beyond_the_cap_is_refused_with_the_cap_in_the_reason(panel):
    seed(people=[{"name": f"p{i}"} for i in range(memory.CAPS["people"])])
    before = on_disk()
    reply = panel.send({"action": "memory_op", "op": "ADD", "path": "people[]",
                        "value": {"name": "one too many"}})

    assert reply["type"] == "memory_error"
    assert str(memory.CAPS["people"]) in reply["reason"] and "cap" in reply["reason"]
    assert on_disk() == before
    assert audit_log() == []


# --------------------------------------------------------------- 4. malformed ops


@pytest.mark.parametrize("op", [
    {"op": "REPLACE", "path": "identity.name", "value": "x"},
    {"op": "UPDATE", "path": "identity/name", "value": "x"},
    {"op": "ADD", "path": "identity.name", "value": None},
    {"op": "ADD", "path": "hobbies.chess", "value": "yes"},
    {"op": "ADD", "path": "people[]", "value": {"rel": "friend"}},
])
def test_malformed_op_is_refused_without_writing_or_logging(panel, op):
    seed(identity={"name": "Misi"})
    before, before_mtime = on_disk(), os.stat(memory.MEMORY_PATH).st_mtime_ns

    reply = panel.send({"action": "memory_op", **op})

    assert reply["type"] == "memory_error" and reply["reason"]
    assert reply["path"] == op["path"]
    assert on_disk() == before
    assert os.stat(memory.MEMORY_PATH).st_mtime_ns == before_mtime
    assert not os.path.exists(memory.OPS_LOG_PATH)


# --------------------------------------------------------------- 5. get_memory


def test_get_memory_returns_the_rendered_block_and_the_visible_fact_count(panel):
    profile = seed(
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
    profile = seed(episodic=[
        {"date": "2020-01-01", "text": "old news", "ttl_days": 5, "importance": 2},
        {"date": memory._today().isoformat(), "text": "today", "ttl_days": 30, "importance": 2},
    ])
    snap = memory.snapshot(profile)
    assert [e["expired"] for e in snap["profile"]["episodic"]] == [True, False]
    assert snap["count"] == 2
    assert all("expired" not in e for e in on_disk()["episodic"])
    assert all("expired" not in e for e in profile["episodic"])


# --------------------------------------------------------------- 6. reset


def test_memory_reset_empties_the_file_and_logs_one_delete_all(panel):
    seed(identity={"name": "Mihaly"}, people=[{"name": "Anna"}])
    reply = panel.send({"action": "memory_reset"})

    assert reply["count"] == 0
    assert reply["rendered"] == ""
    stored = on_disk()
    assert {k: v for k, v in stored.items() if k != "updated"} == \
           {k: v for k, v in memory._empty_profile().items() if k != "updated"}
    logged = audit_log()
    assert len(logged) == 1
    assert (logged[0]["op"], logged[0]["path"], logged[0]["reason"]) == ("DELETE", "*", "ui:reset")


# --------------------------------------------------------------- 7. concurrency


def test_a_hand_edit_during_a_background_save_is_not_overwritten(monkeypatch):
    """The lock must span load -> propose -> write in save(); without it the edit is lost."""
    seed(identity={"name": "Misi"})
    entered = threading.Event()
    client = FakeClient(
        json.dumps([{"op": "ADD", "path": "preferences.coffee", "value": "black"}]),
        delay=0.15, entered=entered,
    )
    monkeypatch.setattr(core_llm, "_client", client)
    saver = threading.Thread(
        target=memory.save,
        args=([{"role": "user", "content": "I drink black coffee"}],))
    saver.start()
    assert entered.wait(timeout=5), "the save never reached the LLM call"

    profile, reason = memory.edit({"op": "UPDATE", "path": "identity.name", "value": "Mihaly"})
    saver.join(timeout=10)

    assert reason is None
    final = on_disk()
    assert final["identity"]["name"] == "Mihaly"        # the hand edit survived
    assert final["preferences"]["coffee"] == "black"    # so did the background save
    assert sorted(op["path"] for op in audit_log()) == ["identity.name", "preferences.coffee"]


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
    assert len(audit_log()) == applied            # one audit line per applied op


# --------------------------------------------------------------- stress: concurrent hammer


def test_four_threads_hammering_edits_during_a_save_keep_the_file_intact(monkeypatch):
    seed(identity={"name": "Misi"})
    applied = []
    lock = threading.Lock()

    def hammer(tid: int):
        local = 0
        for i in range(25):
            _, reason = memory.edit({"op": "ADD", "path": f"preferences.t{tid}_{i}", "value": "v"})
            local += reason is None
        with lock:
            applied.append(local)

    client = FakeClient(json.dumps([{"op": "ADD", "path": "people[]", "value": {"name": "Anna"}}]),
                        delay=0.05)
    monkeypatch.setattr(core_llm, "_client", client)
    threads = [threading.Thread(target=hammer, args=(t,)) for t in range(4)]
    saver = threading.Thread(target=lambda: applied.append(
        len(memory.save([{"role": "user", "content": "Anna is a colleague"}]))))
    for t in threads + [saver]:
        t.start()
    for t in threads + [saver]:
        t.join(timeout=30)
        assert not t.is_alive()

    final = on_disk()                          # valid JSON, no torn write
    assert len([k for k in final["preferences"] if k != "superseded"]) == 100
    assert [p["name"] for p in final["people"]] == ["Anna"]
    assert sum(applied) == 101 == len(audit_log())


# --------------------------------------------------- follow-up: session-save triggers


def test_has_unsaved_turns_tracks_the_session_buffer(monkeypatch):
    monkeypatch.setattr(llm, "_session_turns", [])
    assert llm.has_unsaved_turns() is False
    monkeypatch.setattr(llm, "_session_turns", [{"role": "user", "content": "hi"}])
    assert llm.has_unsaved_turns() is True


def test_opening_the_memory_tab_saves_the_session_and_sends_two_snapshots(panel, monkeypatch):
    """Pending turns → the current snapshot goes out first, then the save, then a fresh one."""
    seed(identity={"name": "Misi"})
    calls = []

    def fake_save():
        calls.append(1)
        memory.edit({"op": "ADD", "path": "preferences.coffee", "value": "black"})

    monkeypatch.setattr(llm, "has_unsaved_turns", lambda: True)
    monkeypatch.setattr(llm, "save_memory", fake_save)

    first, second = panel.send_many({"action": "get_memory"}, 2)
    assert calls == [1]
    assert first["type"] == second["type"] == "memory"
    assert "coffee" not in first["profile"]["preferences"]
    assert second["profile"]["preferences"]["coffee"] == "black"


def test_opening_the_memory_tab_with_nothing_pending_does_not_save(panel, monkeypatch):
    seed(identity={"name": "Misi"})
    calls = []
    monkeypatch.setattr(llm, "has_unsaved_turns", lambda: False)
    monkeypatch.setattr(llm, "save_memory", lambda: calls.append(1))

    reply = panel.send({"action": "get_memory"})
    assert reply["type"] == "memory"
    assert calls == []


def test_opening_the_memory_tab_during_a_turn_does_not_save(panel, monkeypatch):
    """A turn owns the LLM; the tab still gets its snapshot and the periodic save catches up."""
    seed(identity={"name": "Misi"})
    calls = []
    monkeypatch.setattr(llm, "has_unsaved_turns", lambda: True)
    monkeypatch.setattr(llm, "save_memory", lambda: calls.append(1))

    blocker = Blocker()
    blocker.hold(chatbot.controller)
    try:
        reply = panel.send({"action": "get_memory"})
    finally:
        blocker.free(chatbot.controller)
    assert reply["type"] == "memory"
    assert calls == []


def test_save_now_saves_even_with_nothing_pending(panel, monkeypatch):
    seed(identity={"name": "Misi"})
    calls = []
    monkeypatch.setattr(llm, "has_unsaved_turns", lambda: False)
    monkeypatch.setattr(llm, "save_memory", lambda: calls.append(1))

    first, second = panel.send_many({"action": "save_memory"}, 2)
    assert calls == [1]
    assert first["type"] == second["type"] == "memory"


def test_save_now_refuses_while_a_turn_is_running(panel, monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "save_memory", lambda: calls.append(1))
    blocker = Blocker()
    blocker.hold(chatbot.controller)
    try:
        reply = panel.send({"action": "save_memory"})
    finally:
        blocker.free(chatbot.controller)
    assert reply == {"type": "error", "scope": "memory", "text": "Wait until the reply finishes."}
    assert calls == []


# --------------------------------------------------------------- 9. aliases (PRD 26 scenario 7)


def test_the_aliases_row_is_edited_as_a_comma_separated_string(panel):
    seed(identity={"name": "Mihaly Fodor", "aliases": ["Misi"]})
    reply = panel.send({"action": "memory_op", "op": "UPDATE",
                        "path": "identity.aliases", "value": "Misi, Michael"})

    assert reply["type"] == "memory"
    assert reply["profile"]["identity"]["aliases"] == ["Misi", "Michael"]     # snapshot: a list
    assert on_disk()["identity"]["aliases"] == ["Misi", "Michael"]
    assert "also: Misi, Michael" in reply["rendered"]
    assert audit_log()[0]["path"] == "identity.aliases"


def test_an_alias_row_can_be_added_and_deleted_from_the_panel(panel):
    seed(identity={"name": "Mihaly Fodor"})
    added = panel.send({"action": "memory_op", "op": "UPDATE",
                        "path": "identity.aliases", "value": "Misi"})
    assert added["profile"]["identity"]["aliases"] == ["Misi"]

    removed = panel.send({"action": "memory_op", "op": "DELETE", "path": "identity.aliases"})
    assert "aliases" not in removed["profile"]["identity"]
