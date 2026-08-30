"""Unit tests for memory v2 — mocked LLM, no network. Mirrors docs/15-memory-v2.md."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from datetime import date

import pytest

import memory
from tests.conftest import FakeClient, audit_log, days_ago, empty_profile, on_disk, seed


# --------------------------------------------------------------------------- helpers


@pytest.fixture(autouse=True)
def sandbox(memory_sandbox):
    """Every test in this module writes to a tmp memory.json (fixture lives in conftest)."""
    return memory_sandbox


TURNS = [{"role": "user", "content": "I'm Mihaly and I live in Budapest"}]


def undated(entries: list) -> list:
    """Entries without the decay fields (`seen` / `stale`) — see docs/18-memory-decay.md."""
    return [{k: v for k, v in e.items() if k not in ("seen", "stale")} for e in entries]


# --------------------------------------------------------------- 1. apply_ops


def test_apply_ops_adds_every_path_type():
    ops = [
        {"op": "ADD", "path": "identity.name", "value": "Mihaly"},
        {"op": "ADD", "path": "preferences.coffee", "value": "black"},
        {"op": "ADD", "path": "people[]", "value": {"name": "Anna", "rel": "colleague"}},
        {"op": "ADD", "path": "projects[]", "value": {"name": "half marathon", "status": "training"}},
        {"op": "ADD", "path": "recurring[]", "value": {"what": "gym", "when": "Tue/Thu"}},
        {"op": "ADD", "path": "episodic[]", "value": {"text": "had pasta for lunch", "ttl_days": 3}},
        {"op": "NOOP", "path": "identity.name", "value": "ignored"},
    ]
    profile, applied = memory.apply_ops(memory._empty_profile(), ops)
    assert len(applied) == 6  # NOOP is not an applied op
    assert profile["identity"]["name"] == "Mihaly"
    assert profile["preferences"]["coffee"] == "black"
    assert profile["people"][0]["rel"] == "colleague"
    assert profile["projects"][0]["name"] == "half marathon"
    assert profile["recurring"][0]["when"] == "Tue/Thu"
    episodic = profile["episodic"][0]
    assert episodic["date"] == date.today().isoformat()
    assert (episodic["ttl_days"], episodic["importance"]) == (3, 2)


def test_update_scalar_stores_superseded():
    profile = empty_profile(identity={"occupation": "software engineer"})
    profile, applied = memory.apply_ops(
        profile, [{"op": "UPDATE", "path": "identity.occupation", "value": "engineering manager"}])
    assert profile["identity"]["occupation"] == "engineering manager"
    assert profile["identity"]["superseded"] == {"occupation": ["software engineer"]}
    assert applied[0]["op"] == "UPDATE"
    # the old value is kept for undo but never rendered — it made the model volunteer it
    assert "software engineer" not in memory.render(profile)
    assert memory.render(profile) == "Identity: occupation: engineering manager"


def test_update_list_entry_matches_by_key_and_delete_removes():
    profile = empty_profile(projects=[{"name": "half marathon", "note": "race in October"}],
                       people=[{"name": "Anna", "rel": "colleague"}])
    profile, _ = memory.apply_ops(profile, [
        {"op": "UPDATE", "path": "projects[]", "value": {"name": "half marathon", "note": "race in November"}},
        {"op": "DELETE", "path": "people[]", "value": {"name": "Anna"}},
    ])
    assert undated(profile["projects"]) == [{"name": "half marathon", "note": "race in November"}]
    assert profile["people"] == []


def test_delete_scalar_removes_key():
    profile = empty_profile(preferences={"coffee": "black"})
    profile, applied = memory.apply_ops(
        profile, [{"op": "DELETE", "path": "preferences.coffee"}])
    assert profile["preferences"] == {} and len(applied) == 1


def test_malformed_ops_are_rejected_and_logged(capsys):
    ops = [
        {"op": "ADD", "path": "hobbies.chess", "value": "yes"},        # unknown section
        {"op": "ADD", "path": "identity.age", "value": 34},            # non-string scalar
        {"op": "ADD", "path": "people[]", "value": {"rel": "friend"}},  # missing name
        {"op": "ADD", "path": "recurring[]", "value": "gym"},           # not an object
        {"op": "DELETE", "path": "projects[]", "value": {"name": "nope"}},  # not present
        {"op": "FROBNICATE", "path": "identity.name", "value": "x"},   # unknown op
        {"op": "UPDATE", "path": "episodic[]", "value": {"text": "x"}},  # episodic is ADD-only
    ]
    profile, applied = memory.apply_ops(memory._empty_profile(), ops)
    assert applied == []
    assert profile == memory._empty_profile()
    out = capsys.readouterr().out
    assert out.count("[Memory] rejected op") == 7
    assert "unknown section 'hobbies'" in out and "needs a non-empty string" in out


@pytest.mark.parametrize("path", ["people[]", "people", "people[0]"])
def test_list_paths_tolerate_a_missing_bracket_suffix(path):
    """Models drop the `[]` suffix all the time; the section name alone is enough."""
    profile, applied = memory.apply_ops(
        memory._empty_profile(), [{"op": "ADD", "path": path, "value": {"name": "Bolt", "rel": "dog"}}])
    assert undated(profile["people"]) == [{"name": "Bolt", "rel": "dog"}]
    assert applied[0]["path"] == "people[]"


def test_indexed_entry_field_path_updates_one_field():
    profile = empty_profile(projects=[{"name": "half marathon", "note": "race in October"}])
    profile, applied = memory.apply_ops(
        profile, [{"op": "UPDATE", "path": "projects[0].note", "value": "race in November"}])
    assert undated(profile["projects"]) == [{"name": "half marathon", "note": "race in November"}]
    assert applied[0]["path"] == "projects[0].note"


def test_indexed_entry_field_path_out_of_range_is_rejected(capsys):
    profile, applied = memory.apply_ops(
        memory._empty_profile(), [{"op": "UPDATE", "path": "projects[3].note", "value": "x"}])
    assert applied == [] and "projects[3] does not exist" in capsys.readouterr().out


def test_restating_a_scalar_in_another_casing_is_not_an_update():
    profile = empty_profile(identity={"occupation": "Engineering manager"})
    profile, applied = memory.apply_ops(
        profile, [{"op": "UPDATE", "path": "identity.occupation", "value": "engineering manager!"}])
    assert applied == []
    assert profile["identity"] == {"occupation": "Engineering manager"}  # no bogus superseded


def test_caps_are_enforced_per_list(capsys):
    profile = empty_profile(people=[{"name": f"p{i}"} for i in range(memory.CAPS["people"])])
    profile, _ = memory.apply_ops(profile, [{"op": "ADD", "path": "people[]", "value": {"name": "newest"}}])
    assert len(profile["people"]) == memory.CAPS["people"]
    assert profile["people"][0]["name"] == "p1"          # oldest evicted
    assert profile["people"][-1]["name"] == "newest"
    assert "cap enforced" in capsys.readouterr().out


def test_episodic_cap_drops_least_important_and_oldest_first():
    profile = empty_profile(episodic=[
        {"date": days_ago(i % 5), "text": f"e{i}", "ttl_days": 90, "importance": 1 if i < 5 else 3}
        for i in range(memory.CAPS["episodic"] + 5)
    ])
    profile, _ = memory.apply_ops(profile, [])
    kept = {e["text"] for e in profile["episodic"]}
    assert len(kept) == memory.CAPS["episodic"]
    assert not kept & {f"e{i}" for i in range(5)}  # importance 1 entries went first


# --------------------------------------------------------------- 2. render


def test_render_orders_episodics_by_recency_times_importance_and_drops_expired():
    profile = empty_profile(episodic=[
        {"date": days_ago(1), "text": "medium thing", "ttl_days": 30, "importance": 2},
        {"date": days_ago(2), "text": "big thing", "ttl_days": 30, "importance": 3},
        {"date": days_ago(40), "text": "expired thing", "ttl_days": 7, "importance": 3},
    ])
    block = memory.render(profile)
    assert "expired thing" not in block
    assert block.index("big thing") < block.index("medium thing")


def test_render_skips_importance_one_episodics_but_keeps_them_on_disk():
    """Everyday trivia that slipped through is stored until its TTL, never prompted."""
    profile = empty_profile(episodic=[
        {"date": days_ago(0), "text": "had a sandwich for lunch", "ttl_days": 3, "importance": 1},
        {"date": days_ago(0), "text": "signed the flat contract", "ttl_days": 30, "importance": 3},
    ])
    block = memory.render(profile)
    assert "sandwich" not in block
    assert "flat contract" in block
    assert len(profile["episodic"]) == 2
    assert len(memory.live_episodics(profile)) == 2  # unfiltered view still sees both


def test_render_folds_nickname_into_the_name_line():
    profile = empty_profile(identity={"name": "Mihaly", "nickname": "Misi", "location": "Budapest"})
    assert memory.render(profile) == "Identity: name: Mihaly (call them: Misi); location: Budapest"


def test_render_shows_a_nickname_without_a_name():
    profile = empty_profile(identity={"nickname": "Misi"})
    assert memory.render(profile) == "Identity: call them: Misi"


def test_render_shows_at_most_five_episodics_and_stays_in_budget():
    profile = empty_profile(
        identity={"name": "Mihaly"},
        episodic=[{"date": days_ago(i), "text": f"episode number {i} " + "x" * 200,
                   "ttl_days": 90, "importance": 2} for i in range(20)])
    full = memory.render(profile, budget_tokens=10 ** 6)
    assert full.count("\n- ") == memory.EPISODIC_SHOWN
    tight = memory.render(profile, budget_tokens=40)
    assert memory.tokens_est(tight) <= 40
    assert tight.startswith("Identity: name: Mihaly")
    assert not tight.rstrip().endswith("Recent:")


# --------------------------------------------------------------- 3. reflect


REFLECTED = json.dumps({
    "identity": {"name": "IMPOSTOR"},
    "preferences": {},
    "people": [{"name": "Anna", "rel": "colleague", "note": "joined the team"},
               {"name": "Invented Person", "rel": "friend"}],
    "projects": [],
    "recurring": [{"what": "gym", "when": "Tue/Thu"}],
    "episodic": [],
})


def _dup_profile():
    return empty_profile(
        identity={"name": "Mihaly"},
        preferences={"coffee": "black"},
        people=[{"name": "Anna", "rel": "colleague"}, {"name": "anna", "note": "joined the team"}],
        episodic=[{"date": days_ago(40), "text": "rained all morning", "ttl_days": 3, "importance": 1},
                  {"date": days_ago(9), "text": "Went to the gym after work", "ttl_days": 30, "importance": 2},
                  {"date": days_ago(7), "text": "Gym session, legs day", "ttl_days": 30, "importance": 2},
                  {"date": days_ago(2), "text": "Hit the gym again on Thursday", "ttl_days": 30, "importance": 2}],
    )


def test_reflect_merges_duplicates_drops_expired_and_promotes():
    profile = memory.reflect(_dup_profile(), FakeClient(REFLECTED), "m")
    assert [p["name"] for p in profile["people"]] == ["Anna"]
    assert profile["episodic"] == []
    assert undated(profile["recurring"]) == [{"what": "gym", "when": "Tue/Thu"}]


def test_reflect_never_rewrites_identity_or_invents_entries():
    profile = memory.reflect(_dup_profile(), FakeClient(REFLECTED), "m")
    assert profile["identity"] == {"name": "Mihaly"}          # LLM said "IMPOSTOR"
    assert profile["preferences"] == {"coffee": "black"}      # LLM dropped it
    assert all(p["name"] != "Invented Person" for p in profile["people"])


def test_reflect_keeps_profile_on_unparseable_output(capsys):
    original = _dup_profile()
    assert memory.reflect(original, FakeClient("sorry, I can't do that"), "m") == original
    assert "could not parse reflection" in capsys.readouterr().out


def test_reflect_runs_every_five_saves(monkeypatch):
    calls = []
    monkeypatch.setattr(memory, "reflect", lambda p, c, m: calls.append(p) or p)
    for i in range(memory.REFLECT_EVERY):
        client = FakeClient(json.dumps([{"op": "ADD", "path": "preferences.k%d" % i, "value": "v"}]))
        memory.save(TURNS, client, "m")
    assert memory.load_profile()["meta"]["saves"] == memory.REFLECT_EVERY
    assert len(calls) == 1


# --------------------------------------------------------------- 4. migration


def test_migration_seeds_profile_once_and_keeps_the_text_file(sandbox, capsys):
    short = sandbox / "shortmem.txt"
    short.write_text("\n--- 2026-08-01 12:00 ---\nThe user is named Mihaly.\nThe user lives in Budapest.\n")
    ops = json.dumps([{"op": "ADD", "path": "identity.name", "value": "Mihaly"},
                      {"op": "ADD", "path": "identity.location", "value": "Budapest"}])
    client = FakeClient(ops, "[]")

    profile = memory.migrate_if_needed(client, "m")
    assert profile["identity"] == {"name": "Mihaly", "location": "Budapest"}
    assert short.read_text().startswith("\n--- 2026-08-01")
    assert "migrated 2 facts" in capsys.readouterr().out
    assert "--- 2026-08-01" not in client.calls[0]["messages"][1]["content"]

    memory.migrate_if_needed(client, "m")   # second call is a no-op
    assert len(client.calls) == 1


def test_no_migration_without_shortmem():
    client = FakeClient()
    assert memory.migrate_if_needed(client, "m") == memory._empty_profile()
    assert client.calls == []


def test_save_migrates_before_proposing(sandbox):
    (sandbox / "shortmem.txt").write_text("The user is named Mihaly.\n")
    client = FakeClient(
        json.dumps([{"op": "ADD", "path": "identity.name", "value": "Mihaly"}]),
        json.dumps([{"op": "ADD", "path": "identity.location", "value": "Budapest"}]),
    )
    memory.save(TURNS, client, "m")
    assert memory.load_profile()["identity"] == {"name": "Mihaly", "location": "Budapest"}
    # the second proposal saw the migrated profile
    assert "Mihaly" in client.calls[1]["messages"][1]["content"]


# --------------------------------------------------------------- 5. load / prompt wording


def test_load_injects_profile_block_with_friendly_wording():
    memory._write_profile(empty_profile(identity={"name": "Mihaly"}, preferences={"coffee": "black"}))
    prompt = memory.load("PERSONA.")
    assert prompt.startswith("PERSONA.")
    assert "<user_profile>\nIdentity: name: Mihaly\nPreferences: coffee: black\n</user_profile>" in prompt
    assert "as a friend would" in prompt
    assert "believe the user" in prompt
    assert "never bring it up" not in prompt.lower()


def test_load_returns_prompt_unchanged_when_profile_is_empty():
    assert memory.load("PERSONA.") == "PERSONA."
    memory._write_profile(memory._empty_profile())
    assert memory.load("PERSONA.") == "PERSONA."


def test_load_ignores_a_corrupt_profile(sandbox, capsys):
    (sandbox / "memory.json").write_text("{not json")
    assert memory.load("PERSONA.") == "PERSONA."
    assert "could not read memory.json" in capsys.readouterr().out


# --------------------------------------------------------------- 6. atomic write + audit


def test_save_writes_atomically_and_audits_every_applied_op(sandbox, monkeypatch, capsys):
    replaces = []
    real_replace = os.replace
    monkeypatch.setattr(memory.os, "replace",
                        lambda s, d: (replaces.append((s, d)), real_replace(s, d))[1])
    ops = [{"op": "ADD", "path": "identity.name", "value": "Mihaly", "reason": "said so"},
           {"op": "ADD", "path": "people[]", "value": {"name": "Anna"}, "reason": "new colleague"}]
    applied = memory.save(TURNS, FakeClient(json.dumps(ops)), "m")

    assert replaces == [(memory.MEMORY_PATH + ".tmp", memory.MEMORY_PATH)]
    assert not os.path.exists(memory.MEMORY_PATH + ".tmp")
    assert len(applied) == 2

    logged = [json.loads(line) for line in (sandbox / "memory_ops.jsonl").read_text().splitlines()]
    assert len(logged) == 2
    assert [op["path"] for op in logged] == ["identity.name", "people[]"]
    assert logged[0]["reason"] == "said so" and logged[0]["ts"]
    assert "[Memory] 2 ops applied (2 add, 0 update, 0 delete)" in capsys.readouterr().out


def test_save_with_nothing_new_writes_nothing(capsys):
    assert memory.save(TURNS, FakeClient("[]"), "m") == []
    assert not os.path.exists(memory.MEMORY_PATH)
    assert not os.path.exists(memory.OPS_LOG_PATH)
    assert "[Nothing new to save]" in capsys.readouterr().out


def test_save_counts_updates_and_deletes(capsys):
    memory._write_profile(empty_profile(identity={"occupation": "software engineer"},
                                   people=[{"name": "Anna"}]))
    ops = [{"op": "UPDATE", "path": "identity.occupation", "value": "engineering manager"},
           {"op": "DELETE", "path": "people[]", "value": {"name": "Anna"}}]
    memory.save(TURNS, FakeClient(json.dumps(ops)), "m")
    assert "1 update, 1 delete" in capsys.readouterr().out


# --------------------------------------------------------------- propose_ops robustness


@pytest.mark.parametrize("content", [
    '```json\n[{"op": "ADD", "path": "identity.name", "value": "Mihaly"}]\n```',
    'Sure! Here are the ops:\n[{"op": "ADD", "path": "identity.name", "value": "Mihaly"}]\nHope that helps.',
    '[{"op": "ADD", "path": "identity.name", "value": "Mihaly"}]',
])
def test_propose_ops_tolerates_fences_and_prose(content):
    ops = memory.propose_ops(memory._empty_profile(), TURNS, FakeClient(content), "m")
    assert ops == [{"op": "ADD", "path": "identity.name", "value": "Mihaly"}]


def test_propose_ops_uses_temperature_zero_and_a_token_cap():
    client = FakeClient("[]")
    memory.propose_ops(memory._empty_profile(), TURNS, client, "m")
    assert client.calls[0]["temperature"] == 0
    assert client.calls[0]["max_tokens"] == 600
    assert "User: I'm Mihaly" in client.calls[0]["messages"][1]["content"]


@pytest.mark.parametrize("content", ["NOTHING", "", None, "{}"])
def test_propose_ops_returns_empty_on_unparseable_output(content, capsys):
    assert memory.propose_ops(memory._empty_profile(), TURNS, FakeClient(content), "m") == []
    assert "could not parse ops" in capsys.readouterr().out


def test_propose_ops_returns_none_when_the_call_fails(capsys):
    client = FakeClient(RuntimeError("connection refused"))
    assert memory.propose_ops(memory._empty_profile(), TURNS, client, "m") is None
    assert "op proposal failed" in capsys.readouterr().out


def test_save_leaves_the_file_alone_when_the_call_fails(memory_sandbox):
    seed(identity={"name": "Michael"})
    before = on_disk()
    assert memory.save(TURNS, FakeClient(RuntimeError("connection refused")), "m") is None
    assert on_disk() == before
    assert audit_log() == []


def test_propose_ops_skips_the_call_for_an_empty_transcript():
    client = FakeClient("[]")
    assert memory.propose_ops(memory._empty_profile(), [], client, "m") == []
    assert client.calls == []


# --------------------------------------------------------------- llm wiring


def test_shutdown_hook_saves_memory(monkeypatch):
    from fastapi.testclient import TestClient
    import chatbot, llm
    calls = []
    monkeypatch.setattr(llm, "save_memory", lambda: calls.append(1))
    with TestClient(chatbot.app):
        pass  # exiting the context runs the shutdown event
    assert calls == [1]


def test_save_memory_is_idempotent(monkeypatch):
    import llm
    saved = []
    monkeypatch.setattr(memory, "save", lambda turns, client, model, **kw: (saved.append(list(turns)), [])[1])
    monkeypatch.setattr(llm, "client", lambda: object())
    llm.reset()
    llm._session_turns.append({"role": "user", "content": "hi"})
    llm.save_memory(); llm.save_memory()
    assert saved == [[{"role": "user", "content": "hi"}]]


def test_save_memory_keeps_the_turns_when_the_save_fails(monkeypatch):
    import llm
    monkeypatch.setattr(memory, "save", lambda turns, client, model, **kw: None)
    monkeypatch.setattr(llm, "client", lambda: object())
    llm.reset()
    llm._session_turns.append({"role": "user", "content": "hi"})
    llm.save_memory()
    assert llm.has_unsaved_turns()
    assert llm.get_session_turns() == [{"role": "user", "content": "hi"}]


def test_save_memory_keeps_the_turns_when_the_save_raises(monkeypatch):
    import llm

    def boom(turns, client, model, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(memory, "save", boom)
    monkeypatch.setattr(llm, "client", lambda: object())
    llm.reset()
    llm._session_turns.append({"role": "user", "content": "hi"})
    with pytest.raises(RuntimeError):
        llm.save_memory()
    assert llm.get_session_turns() == [{"role": "user", "content": "hi"}]


def test_reflect_cannot_invent_recurring_entries():
    original = memory._empty_profile()
    original["episodic"] = [{"date": "2026-08-20", "text": "Went to the gym after work", "ttl_days": 30, "importance": 2}]
    cleaned = memory._empty_profile()
    cleaned["recurring"] = [{"what": "gym", "when": "evenings"}, {"what": "skydiving", "when": "Sundays"}]
    out = memory._validate_reflection(original, cleaned)
    assert [r["what"] for r in out["recurring"]] == ["gym"]


# --------------------------------------------------------------- 11. identity aliases (PRD 26)


ALIAS_IDENTITY = {"name": "Mihaly Fodor", "nickname": "Michael", "aliases": ["Misi", "Mihály"]}


# Scenario 1: normalisation and the render clauses.
def test_aliases_accept_a_comma_separated_string_and_render_as_a_clause():
    profile = memory._normalise({"identity": {"name": "Mihaly Fodor", "nickname": "Michael",
                                              "aliases": "Misi, Mihály"}})
    assert profile["identity"]["aliases"] == ["Misi", "Mihály"]
    assert memory.render(profile) == \
        "Identity: name: Mihaly Fodor (call them: Michael; also: Misi, Mihály)"


def test_an_empty_alias_list_adds_no_clause_and_no_fact():
    profile = memory._normalise({"identity": {"name": "Mihaly Fodor", "aliases": "  ,  "}})
    assert "aliases" not in profile["identity"]
    assert memory.render(profile) == "Identity: name: Mihaly Fodor"
    assert memory.fact_count(profile) == 1


def test_aliases_are_rendered_on_their_own_when_there_is_no_name():
    profile = memory._normalise({"identity": {"aliases": ["Misi"]}})
    assert memory.render(profile) == "Identity: also known as: Misi"


# Scenario 2: the ops.
def test_alias_ops_add_update_and_delete():
    profile = memory._normalise({"identity": dict(ALIAS_IDENTITY)})
    profile, applied = memory.apply_ops(profile, [
        {"op": "ADD", "path": "identity.aliases", "value": "Miki"},
        {"op": "ADD", "path": "identity.aliases", "value": "misi"},      # duplicate, ignored
        {"op": "ADD", "path": "identity.aliases", "value": "MIHÁLY"},    # duplicate, ignored
    ])
    assert profile["identity"]["aliases"] == ["Misi", "Mihály", "Miki"]
    assert len(applied) == 1

    profile, applied = memory.apply_ops(
        profile, [{"op": "UPDATE", "path": "identity.aliases", "value": "Misi, Michael"}])
    assert profile["identity"]["aliases"] == ["Misi", "Michael"]
    assert "superseded" not in profile["identity"]        # a replaced list keeps no history

    profile, applied = memory.apply_ops(
        profile, [{"op": "DELETE", "path": "identity.aliases"}])
    assert "aliases" not in profile["identity"]
    assert len(applied) == 1


def test_alias_ops_are_audited(sandbox):
    memory._write_profile(memory._normalise({"identity": dict(ALIAS_IDENTITY)}))
    profile, error = memory.edit({"op": "ADD", "path": "identity.aliases", "value": "Miki"})
    assert error is None
    logged = [json.loads(line) for line in open(memory.OPS_LOG_PATH) if line.strip()]
    assert [(entry["op"], entry["path"], entry["value"]) for entry in logged] == \
        [("ADD", "identity.aliases", "Miki")]


def test_an_empty_alias_value_is_rejected(capsys):
    profile = memory._normalise({"identity": dict(ALIAS_IDENTITY)})
    profile, applied = memory.apply_ops(
        profile, [{"op": "UPDATE", "path": "identity.aliases", "value": "   "}])
    assert applied == [] and profile["identity"]["aliases"] == ["Misi", "Mihály"]
    assert "comma-separated" in capsys.readouterr().out


# Scenario 3: is_user.
@pytest.mark.parametrize("text,expected", [
    ("Michael", True),
    ("Misi", True),
    ("Mihay Fodor", True),      # an OCR / ASR mangling of the full name
    ("mihaly", True),
    ("Mihály Fodor", True),
    ("Fodor", True),
    ("Anna", False),
    ("Elena", False),
    ("", False),
])
def test_is_user_matches_every_form_of_the_owners_name(text, expected):
    names = memory.user_names(memory._normalise({"identity": dict(ALIAS_IDENTITY)}))
    assert names == ["Mihaly Fodor", "Michael", "Misi", "Mihály"]
    assert memory.is_user(text, names) is expected


def test_is_user_is_false_without_a_profile():
    assert memory.is_user("Michael", []) is False


# Scenario 4: the extraction guard.
def test_an_extracted_name_that_is_only_an_alias_never_overwrites_the_real_name(sandbox, capsys):
    memory._write_profile(memory._normalise({"identity": dict(ALIAS_IDENTITY)}))
    client = FakeClient(json.dumps([
        {"op": "UPDATE", "path": "identity.name", "value": "Michael", "reason": "they said so"},
        {"op": "ADD", "path": "identity.aliases", "value": "Miki", "reason": "also called Miki"},
    ]))
    applied = memory.save([{"role": "user", "content": "Michael here again, some call me Miki"}],
                          client, "fake-model")

    assert [op["path"] for op in applied] == ["identity.aliases"]
    profile = memory.load_profile()
    assert profile["identity"]["name"] == "Mihaly Fodor"
    assert profile["identity"]["aliases"] == ["Misi", "Mihály", "Miki"]
    assert "name matches a known alias" in capsys.readouterr().out


def test_the_real_name_can_still_be_corrected_and_set_when_missing():
    profile, _ = memory.apply_ops(memory._empty_profile(), [
        {"op": "ADD", "path": "identity.nickname", "value": "Michael"},
        {"op": "UPDATE", "path": "identity.name", "value": "Michael"},   # no name yet: allowed
    ])
    assert profile["identity"]["name"] == "Michael"

    profile = memory._normalise({"identity": dict(ALIAS_IDENTITY)})
    profile, applied = memory.apply_ops(
        profile, [{"op": "UPDATE", "path": "identity.name", "value": "Mihaly Kovacs"}])
    assert profile["identity"]["name"] == "Mihaly Kovacs" and len(applied) == 1


def test_the_propose_prompt_documents_the_alias_rules():
    assert "identity.aliases" in memory.PROPOSE_SYSTEM
    assert "Never UPDATE identity.name" in memory.PROPOSE_SYSTEM
    assert "speech recognition" in memory.PROPOSE_SYSTEM


# Scenario 6 (the prompt itself; the whisper call is tests/test_asr.py).
def test_the_asr_prompt_lists_the_user_names_and_people_and_is_rebuilt_after_a_save(sandbox):
    memory._write_profile(memory._normalise(
        {"identity": dict(ALIAS_IDENTITY), "people": [{"name": "Anna"}]}))
    first = memory.asr_prompt()
    assert first == "Names: Mihaly Fodor, Michael, Misi, Mihály, Anna."
    assert memory._ASR_CACHE["text"] == first and memory.asr_prompt() == first   # cached

    memory.edit({"op": "ADD", "path": "identity.aliases", "value": "Miki"})
    assert "Miki" in memory.asr_prompt()


def test_the_asr_prompt_stays_within_its_token_budget(sandbox):
    memory._write_profile(empty_profile(
        identity={"name": "Mihaly Fodor"},
        people=[{"name": f"Person Number {i}"} for i in range(20)]))
    prompt = memory.asr_prompt()
    assert memory.tokens_est(prompt) <= memory.ASR_PROMPT_TOKENS
    assert prompt.startswith("Names: Mihaly Fodor,")


def test_the_asr_prompt_is_empty_without_a_profile(sandbox):
    assert memory.asr_prompt() == ""
