"""Unit tests for memory decay — mocked LLM and a mocked clock. Mirrors docs/18-memory-decay.md.

PRD scenarios 1-7, plus unit versions of the two live harness scenarios (8 and 9) and the
harness plumbing they need (`edit:` / `clock:` turns).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from datetime import date, timedelta

import pytest

import llm
import memory
from eval import harness
from tests.conftest import FakeClient, audit_log, days_ago, empty_profile, on_disk, seed

TODAY = date.today().isoformat()


# --------------------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def sandbox(memory_sandbox):
    """Every test in this module writes to a tmp memory.json (fixture lives in conftest)."""
    return memory_sandbox


@pytest.fixture
def clock(monkeypatch):
    """`clock(200)` makes memory believe 200 days have passed."""
    def travel(days: int) -> date:
        when = date.today() + timedelta(days=days)
        monkeypatch.setattr(memory, "_today", lambda: when)
        return when
    return travel


# --------------------------------------------------------------- 1. schema backfill


def test_normalise_backfills_seen_from_the_profile_updated_date():
    raw = {
        "version": 2,
        "updated": "2026-01-15T09:30",
        "identity": {"name": "Mihaly"},
        "people": [{"name": "Anna", "rel": "colleague"}],
        "projects": [{"name": "half marathon"}],
        "recurring": [{"what": "gym", "when": "Tue/Thu"}],
        "episodic": [{"date": "2026-01-10", "text": "ran 10k"}],
    }
    with open(memory.MEMORY_PATH, "w") as f:
        json.dump(raw, f)
    before = open(memory.MEMORY_PATH).read()

    profile = memory.load_profile()
    for section in ("people", "projects", "recurring", "episodic"):
        assert profile[section][0]["seen"] == "2026-01-15"
        assert not profile[section][0].get("stale")   # absent means live
    assert open(memory.MEMORY_PATH).read() == before   # reading never writes


def test_normalise_falls_back_to_today_and_keeps_an_existing_seen():
    profile = memory._normalise({"people": [{"name": "Anna"}, {"name": "Bob", "seen": "2020-02-02"}]})
    assert profile["people"][0]["seen"] == TODAY
    assert profile["people"][1]["seen"] == "2020-02-02"


# --------------------------------------------------------------- 2. seen on ADD/UPDATE, CONFIRM


@pytest.mark.parametrize("path", ["people[]", "people", "people[0]"])
def test_add_sets_seen_to_today_whatever_the_path_form(path):
    profile, _ = memory.apply_ops(
        memory._empty_profile(), [{"op": "ADD", "path": path, "value": {"name": "Anna"}}])
    assert profile["people"][0]["seen"] == TODAY


def test_update_refreshes_seen_on_every_entry_path_form():
    profile = empty_profile(
        people=[{"name": "Anna", "rel": "colleague", "seen": days_ago(100)}],
        projects=[{"name": "half marathon", "note": "October", "seen": days_ago(100)}],
    )
    profile, _ = memory.apply_ops(profile, [
        {"op": "UPDATE", "path": "people[]", "value": {"name": "Anna", "note": "moved teams"}},
        {"op": "UPDATE", "path": "projects[0].note", "value": "November"},
    ])
    assert profile["people"][0]["seen"] == TODAY
    assert profile["projects"][0]["seen"] == TODAY


def test_episodic_add_carries_seen():
    profile, _ = memory.apply_ops(
        memory._empty_profile(), [{"op": "ADD", "path": "episodic[]", "value": {"text": "ran a 10k"}}])
    assert profile["episodic"][0]["seen"] == TODAY


def test_confirm_refreshes_seen_and_clears_stale():
    profile = empty_profile(people=[
        {"name": "Bob", "seen": days_ago(10)},
        {"name": "Anna", "rel": "colleague", "seen": days_ago(200), "stale": True},
    ])
    profile, applied = memory.apply_ops(
        profile, [{"op": "CONFIRM", "path": "people[1]", "reason": "mentioned at lunch"}])
    anna = profile["people"][1]
    assert anna["seen"] == TODAY and "stale" not in anna
    assert anna["rel"] == "colleague"                     # CONFIRM changes nothing else
    assert profile["people"][0]["seen"] == days_ago(10)  # and touches nothing else
    assert applied == [{"op": "CONFIRM", "path": "people[1]",
                        "value": None, "reason": "mentioned at lunch"}]


def test_confirm_by_identifying_key_works_and_defaults_its_reason():
    profile = empty_profile(recurring=[{"what": "gym", "seen": days_ago(120), "stale": True}])
    profile, applied = memory.apply_ops(
        profile, [{"op": "CONFIRM", "path": "recurring[]", "value": {"what": "Gym"}}])   # matched case-insensitively
    assert profile["recurring"][0] == {"what": "gym", "seen": TODAY}
    assert (applied[0]["path"], applied[0]["reason"]) == ("recurring[]", "confirm")


@pytest.mark.parametrize("op, expected", [
    ({"op": "CONFIRM", "path": "people[3]"}, "people[3] does not exist"),
    ({"op": "CONFIRM", "path": "people[]", "value": {"name": "Nobody"}}, "no people entry named"),
    ({"op": "CONFIRM", "path": "identity.name"}, "CONFIRM needs a list entry path"),
])
def test_confirm_on_something_that_is_not_there_is_rejected(op, expected, capsys):
    profile = empty_profile(people=[{"name": "Anna", "seen": days_ago(5)}])
    profile, applied = memory.apply_ops(profile, [op])
    assert applied == [] and expected in capsys.readouterr().out
    assert profile["people"][0]["seen"] == days_ago(5)


def test_confirm_is_logged_with_the_confirm_reason():
    seed(people=[{"name": "Anna", "seen": days_ago(200), "stale": True}])
    memory.save([{"role": "user", "content": "had lunch with Anna"}],
                FakeClient('[{"op":"CONFIRM","path":"people[0]"}]'), "m")
    logged = audit_log()
    assert [(o["op"], o["path"], o["reason"]) for o in logged] == [("CONFIRM", "people[0]", "confirm")]
    assert on_disk()["people"][0] == {"name": "Anna", "seen": TODAY}


# --------------------------------------------------------------- 3. _mark_stale


def test_mark_stale_retires_entries_past_their_window_only(clock):
    profile = empty_profile(
        people=[{"name": "Anna", "seen": days_ago(0)},
                {"name": "Bob", "seen": days_ago(0)}],
        projects=[{"name": "half marathon", "seen": days_ago(0)}],
        recurring=[{"what": "gym", "seen": days_ago(0)}],
        episodic=[{"date": days_ago(0), "text": "ran a 10k", "seen": days_ago(0),
                   "ttl_days": 30, "importance": 2}],
    )
    profile["people"][1]["seen"] = (date.today() + timedelta(days=100)).isoformat()  # seen later

    clock(200)
    marked = memory._mark_stale(profile)

    assert profile["people"][0]["stale"] is True          # 200 d > 180
    assert "stale" not in profile["people"][1]            # 100 d < 180
    assert profile["projects"][0]["stale"] is True        # 200 d > 120
    assert profile["recurring"][0]["stale"] is True       # 200 d > 90
    assert "stale" not in profile["episodic"][0]          # episodics have a TTL instead
    assert len(marked) == 3 and "Anna" in marked[0]


def test_mark_stale_is_idempotent_and_windows_are_per_section(clock):
    profile = empty_profile(people=[{"name": "Anna", "seen": days_ago(0)}],
                            recurring=[{"what": "gym", "seen": days_ago(0)}])
    clock(100)
    assert len(memory._mark_stale(profile)) == 1          # recurring (90) only
    assert memory._mark_stale(profile) == []              # nothing new to say
    assert "stale" not in profile["people"][0]


def test_stale_windows_are_env_overridable():
    assert memory.STALE_DAYS == {"people": 180, "projects": 120, "recurring": 90}


# --------------------------------------------------------------- 4. render / count / snapshot


def test_render_omits_stale_entries_but_they_stay_counted_andon_disk():
    profile = seed(
        identity={"name": "Mihaly"},
        people=[{"name": "Anna", "rel": "colleague", "seen": days_ago(200), "stale": True},
                {"name": "Bob", "rel": "brother", "seen": TODAY}],
        recurring=[{"what": "gym", "seen": days_ago(100), "stale": True}],
        episodic=[{"date": TODAY, "text": "ran a 10k", "importance": 3, "stale": True}],
    )
    rendered = memory.render(profile)
    assert "Anna" not in rendered and "gym" not in rendered and "10k" not in rendered
    assert "Bob" in rendered
    assert memory.fact_count(profile) == 5                # stale facts still count
    assert len(on_disk()["people"]) == 2                 # nothing was deleted


def test_snapshot_carries_seen_stale_and_the_section_window():
    seed(
        people=[{"name": "Anna", "seen": days_ago(200), "stale": True}, {"name": "Bob"}],
        projects=[{"name": "half marathon", "seen": TODAY}],
        recurring=[{"what": "gym", "seen": TODAY}],
        episodic=[{"date": "2020-01-01", "text": "old news", "ttl_days": 5}],
    )
    snap = memory.snapshot(memory.load_profile())
    people = snap["profile"]["people"]
    assert (people[0]["stale"], people[0]["seen"], people[0]["stale_after"]) == \
           (True, days_ago(200), 180)
    assert (people[1]["stale"], people[1]["stale_after"]) == (False, 180)
    assert snap["profile"]["projects"][0]["stale_after"] == 120
    assert snap["profile"]["recurring"][0]["stale_after"] == 90
    episodic = snap["profile"]["episodic"][0]
    assert episodic["expired"] is True and episodic["stale"] is False and episodic["seen"]
    assert "stale_after" not in episodic
    assert all("stale" not in e for e in on_disk()["people"][1:])   # flags are view-only
    assert snap["count"] == 5


# --------------------------------------------------------------- 5. caps


def test_caps_evict_stale_entries_first_then_the_oldest_seen():
    people = [{"name": f"P{i}", "seen": days_ago(i)} for i in range(30)]   # P29 seen longest ago
    people[3]["stale"] = True
    people[4]["stale"] = True
    profile = empty_profile(people=people)
    profile, _ = memory.apply_ops(profile, [
        {"op": "ADD", "path": "people[]", "value": {"name": "New1"}},
        {"op": "ADD", "path": "people[]", "value": {"name": "New2"}},
    ])
    names = [p["name"] for p in profile["people"]]
    assert len(names) == memory.CAPS["people"] == 30
    assert "P3" not in names and "P4" not in names      # the stale ones went first
    assert "P29" in names and "P28" in names            # oldest `seen` only goes after them
    assert names[:3] == ["P0", "P1", "P2"]              # order is otherwise preserved


def test_caps_fall_back_to_oldest_seen_when_nothing_is_stale():
    people = [{"name": f"P{i}", "seen": days_ago(30 - i)} for i in range(31)]  # P0 is the oldest
    profile = empty_profile(people=people)
    memory._enforce_caps(profile)
    assert [p["name"] for p in profile["people"]][0] == "P1"


# --------------------------------------------------------------- 6. reflection


REFLECTED = json.dumps({
    "identity": {"name": "Mihaly"},
    "preferences": {},
    "people": [{"name": "Anna", "rel": "colleague at work"}],   # merged, decay fields dropped
    "projects": [],
    "recurring": [{"what": "gym", "when": "Tue/Thu"}],
    "episodic": [],
})


def test_validate_reflection_keeps_seen_and_stale_even_when_the_model_drops_them():
    original = empty_profile(
        identity={"name": "Mihaly"},
        people=[{"name": "Anna", "rel": "colleague", "seen": days_ago(200), "stale": True},
                {"name": "Anna K", "rel": "colleague", "seen": days_ago(200), "stale": True}],
        recurring=[{"what": "gym", "when": "Tue", "seen": days_ago(3)}],
        episodic=[{"date": days_ago(2), "text": "gym again", "ttl_days": 30, "importance": 2}],
    )
    result = memory._validate_reflection(original, json.loads(REFLECTED))
    assert result["people"][0]["seen"] == days_ago(200)
    assert result["people"][0]["stale"] is True
    assert result["recurring"][0] == {"what": "gym", "when": "Tue/Thu", "seen": days_ago(3)}


def test_reflection_dates_a_freshly_promoted_recurring_entry_today():
    original = empty_profile(episodic=[
        {"date": days_ago(i), "text": "went to the gym", "ttl_days": 30, "importance": 2}
        for i in (1, 4, 8)
    ])
    cleaned = {"recurring": [{"what": "went to the gym", "when": "most days"}]}
    result = memory._validate_reflection(original, cleaned)
    assert result["recurring"][0]["seen"] == TODAY
    assert "stale" not in result["recurring"][0]


def test_a_stale_entry_survives_a_reflection_pass_through_save():
    seed(people=[{"name": "Anna", "rel": "colleague", "seen": days_ago(200), "stale": True}],
          meta={"saves": memory.REFLECT_EVERY - 1})
    client = FakeClient('[{"op":"ADD","path":"identity.name","value":"Mihaly"}]', REFLECTED)
    memory.save([{"role": "user", "content": "I'm Mihaly"}], client, "m")
    anna = on_disk()["people"][0]
    assert anna["stale"] is True and anna["seen"] == days_ago(200)


# --------------------------------------------------------------- 7. memory_ops batch


def test_memory_ops_batch_promotes_an_episodic_atomically(panel):
    seed(recurring=[], episodic=[{"date": TODAY, "text": "gym again", "ttl_days": 30,
                                   "importance": 2}])
    reply = panel.send({"action": "memory_ops", "ops": [
        {"op": "ADD", "path": "recurring[]", "value": {"what": "gym again", "when": ""}},
        {"op": "DELETE", "path": "episodic[0]"},
    ]})
    assert reply["type"] == "memory"
    assert [e["what"] for e in reply["profile"]["recurring"]] == ["gym again"]
    assert reply["profile"]["episodic"] == []
    stored = on_disk()
    assert stored["recurring"][0]["seen"] == TODAY and stored["episodic"] == []
    assert [(o["op"], o["path"], o["reason"]) for o in audit_log()] == [
        ("ADD", "recurring[]", "ui"), ("DELETE", "episodic[0]", "ui")]


def test_one_bad_op_rejects_the_whole_batch_and_writes_nothing(panel):
    seed(people=[{"name": "Anna", "seen": TODAY}], episodic=[])
    before = on_disk()
    reply = panel.send({"action": "memory_ops", "ops": [
        {"op": "ADD", "path": "recurring[]", "value": {"what": "gym"}},
        {"op": "DELETE", "path": "episodic[7]"},
    ]})
    assert reply["type"] == "memory_error"
    assert reply["path"] == "episodic[7]" and "does not exist" in reply["reason"]
    assert on_disk() == before and audit_log() == []


def test_edit_many_reports_the_path_of_the_op_that_failed():
    seed(people=[{"name": "Anna", "seen": TODAY}])
    profile, error = memory.edit_many([
        {"op": "UPDATE", "path": "identity.name", "value": "Mihaly"},
        {"op": "UPDATE", "path": "identity.name", "value": "Mihaly"},   # nothing changed
    ])
    assert error == {"path": "identity.name", "reason": "nothing changed"}
    assert "name" not in on_disk()["identity"]                          # first op rolled back
    assert memory.edit_many([])[1]["reason"] == "no ops"


def test_edit_is_still_the_single_op_wrapper():
    seed(people=[{"name": "Anna", "seen": days_ago(9)}])
    profile, reason = memory.edit({"op": "UPDATE", "path": "people[0]",
                                   "value": {"rel": "colleague"}})
    assert reason is None and profile["people"][0]["rel"] == "colleague"
    assert profile["people"][0]["seen"] == TODAY


# --------------------------------------------------------------- retire / restore from the UI


def test_the_ui_can_retire_and_restore_an_entry_with_a_boolean_stale(panel):
    seed(people=[{"name": "Anna", "rel": "colleague", "seen": days_ago(30)}])

    retired = panel.send({"action": "memory_op", "op": "UPDATE",
                          "path": "people[0].stale", "value": True})
    assert retired["profile"]["people"][0]["stale"] is True
    assert on_disk()["people"][0]["seen"] == days_ago(30)      # retiring is not a mention
    assert "Anna" not in retired["rendered"]

    restored = panel.send({"action": "memory_op", "op": "UPDATE",
                           "path": "people[0].stale", "value": False,
                           "reason": "ui:restore"})
    assert restored["profile"]["people"][0]["stale"] is False
    assert on_disk()["people"][0] == {"name": "Anna", "rel": "colleague", "seen": TODAY}
    assert "Anna" in restored["rendered"]
    assert [o["reason"] for o in audit_log()] == ["ui", "ui:restore"]


def test_a_non_boolean_stale_value_is_refused(panel):
    seed(people=[{"name": "Anna", "seen": TODAY}])
    reply = panel.send({"action": "memory_op", "op": "UPDATE",
                        "path": "people[0].stale", "value": "maybe"})
    assert reply["type"] == "memory_error" and "true or false" in reply["reason"]
    assert "stale" not in on_disk()["people"][0]


def test_booleans_are_not_accepted_as_ordinary_field_values(capsys):
    profile, applied = memory.apply_ops(
        memory._empty_profile(), [{"op": "ADD", "path": "people[]",
                                   "value": {"name": "Anna", "rel": True}}])
    assert profile["people"][0] == {"name": "Anna", "seen": TODAY}      # the bool was dropped
    assert applied


# --------------------------------------------------------------- 8. decay end to end (mocked LLM)


def test_a_person_goes_quiet_then_comes_back_on_a_confirm(clock):
    """PRD scenario 8 at unit level: introduced, forgotten after 200 days, revived by CONFIRM."""
    intro = '[{"op":"ADD","path":"people[]","value":{"name":"Anna","rel":"colleague"}}]'
    memory.save([{"role": "user", "content": "Anna joined my team"}], FakeClient(intro), "m")
    assert "Anna" in memory.render(memory.load_profile())

    clock(200)
    memory.save([{"role": "user", "content": "quiet week, nothing new"}], FakeClient("[]"), "m")

    profile = memory.load_profile()
    assert profile["people"][0]["stale"] is True
    assert "Anna" not in memory.render(profile)          # out of the prompt, still on disk
    assert memory.fact_count(profile) == 1

    memory.save([{"role": "user", "content": "had lunch with Anna today"}],
                FakeClient('[{"op":"CONFIRM","path":"people[]","value":{"name":"Anna"}}]'), "m")
    profile = memory.load_profile()
    assert "stale" not in profile["people"][0]
    assert profile["people"][0]["seen"] == memory._today().isoformat()
    assert "Anna (colleague)" in memory.render(profile)


def test_a_save_that_learns_nothing_still_writes_the_retirement(clock, capsys):
    seed(people=[{"name": "Anna", "seen": days_ago(0)}])
    clock(200)
    applied = memory.save([{"role": "user", "content": "hi"}], FakeClient("[]"), "m")
    assert applied == []
    assert on_disk()["people"][0]["stale"] is True
    assert "[Memory] stale — people: Anna" in capsys.readouterr().out


# --------------------------------------------------------------- 9. hand edits are not undone


def test_a_hand_deleted_fact_is_not_resurrected_by_the_next_save(monkeypatch):
    """PRD scenario 9 at unit level.

    The dog cannot come back "from old turns" because a session only ever proposes from the
    turns since the last save (llm.save_memory clears them), so the next save sees only what
    the user said after the deletion.
    """
    monkeypatch.setattr(llm, "OMLX_MODEL", "m")
    session_one = FakeClient('[{"op":"ADD","path":"people[]","value":{"name":"Bolt","rel":"dog"}},'
                             '{"op":"ADD","path":"identity.occupation","value":"software engineer"}]')
    monkeypatch.setattr(llm, "client", lambda: session_one)
    llm._session_turns[:] = [{"role": "user", "content": "I've got a dog called Bolt"}]
    llm.save_memory()
    assert [p["name"] for p in on_disk()["people"]] == ["Bolt"]

    # the user fixes the profile by hand between the two sessions
    memory.edit_many([
        {"op": "DELETE", "path": "people[]", "value": {"name": "Bolt"}},
        {"op": "UPDATE", "path": "identity.occupation", "value": "engineering manager"},
    ])

    # the next session says nothing about the dog, so nothing proposes it again
    session_two = FakeClient('[{"op":"ADD","path":"episodic[]",'
                             '"value":{"text":"presentation next Thursday","importance":2}}]')
    monkeypatch.setattr(llm, "client", lambda: session_two)
    llm._session_turns[:] = [{"role": "user", "content": "big presentation next Thursday"}]
    llm.save_memory()

    stored = on_disk()
    assert stored["people"] == []
    assert stored["identity"]["occupation"] == "engineering manager"
    transcript = session_two.calls[0]["messages"][1]["content"]
    assert "Bolt" not in transcript          # the old turns are gone, not just ignored
    assert llm._session_turns == []


# --------------------------------------------------------------- harness: edit / clock turns


def test_harness_clock_turn_shifts_memorys_today():
    original = memory._today
    try:
        assert harness.shift_clock("+200d") == (date.today() + timedelta(days=200)).isoformat()
        assert memory._today() == date.today() + timedelta(days=200)
        harness.shift_clock("+1w")           # offsets accumulate for the rest of the run
        assert memory._today() == date.today() + timedelta(days=207)
    finally:
        memory._today = original
        harness._CLOCK["offset"] = timedelta(0)
    with pytest.raises(ValueError):
        harness._parse_offset("soon")


def test_harness_edit_turn_applies_one_op_and_a_list_as_a_batch(capsys):
    seed(people=[{"name": "Bolt", "rel": "dog", "seen": TODAY}],
          episodic=[{"date": TODAY, "text": "gym again", "ttl_days": 30, "importance": 2}])

    single = harness._run_edit({"op": "DELETE", "path": "people[]", "value": {"name": "Bolt"}})
    assert single["error"] is None and on_disk()["people"] == []

    batch = harness._run_edit([
        {"op": "ADD", "path": "recurring[]", "value": {"what": "gym", "when": "Mon/Wed"}},
        {"op": "DELETE", "path": "episodic[0]"},
    ])
    assert batch["error"] is None
    assert on_disk()["recurring"][0]["what"] == "gym" and on_disk()["episodic"] == []

    refused = harness._run_edit({"op": "DELETE", "path": "people[9]"})
    assert refused["error"]["path"] == "people[9]"
    out = capsys.readouterr().out
    assert "  edit: DELETE people[]" in out and "  edit: ADD recurring[]" in out
    assert "refused" in out


@pytest.mark.parametrize("script", ["editor.yaml", "stale.yaml"])
def test_the_new_eval_scripts_parse_and_only_use_known_turn_types(script):
    path = os.path.join(os.path.dirname(__file__), "..", "eval", "scripts", script)
    sessions = harness.load_script(path)
    turns = [turn for session in sessions for turn in session]
    assert turns and all(
        {"user", "probe", "edit", "clock"} & set(turn) for turn in turns)
    for turn in turns:
        if "clock" in turn:
            harness._parse_offset(turn["clock"])          # every offset is well formed
        if "edit" in turn:
            ops = turn["edit"] if isinstance(turn["edit"], list) else [turn["edit"]]
            assert all(op.get("op") in ("ADD", "UPDATE", "DELETE", "CONFIRM") for op in ops)
