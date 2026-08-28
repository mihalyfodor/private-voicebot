"""Unit tests for memory v2 — mocked LLM, no network. Mirrors docs/15-memory-v2.md."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

import memory


# --------------------------------------------------------------------------- helpers


class FakeClient:
    """Same shape as tests/test_llm_unit.py's FakeClient: queued completion contents."""

    def __init__(self, *contents):
        self._contents = list(contents)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        content = self._contents.pop(0) if self._contents else "[]"
        if isinstance(content, Exception):
            raise content
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "MEMORY_PATH", str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory, "OPS_LOG_PATH", str(tmp_path / "memory_ops.jsonl"))
    monkeypatch.setattr(memory, "SHORTMEM_PATH", str(tmp_path / "shortmem.txt"))
    return tmp_path


def _days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).isoformat()


def _profile(**kw):
    p = memory._empty_profile()
    p.update(kw)
    return p


TURNS = [{"role": "user", "content": "I'm Mihaly and I live in Budapest"}]


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
    profile = _profile(identity={"occupation": "software engineer"})
    profile, applied = memory.apply_ops(
        profile, [{"op": "UPDATE", "path": "identity.occupation", "value": "engineering manager"}])
    assert profile["identity"]["occupation"] == "engineering manager"
    assert profile["identity"]["superseded"] == {"occupation": ["software engineer"]}
    assert applied[0]["op"] == "UPDATE"
    # the old value is kept for undo but never rendered — it made the model volunteer it
    assert "software engineer" not in memory.render(profile)
    assert memory.render(profile) == "Identity: occupation: engineering manager"


def test_update_list_entry_matches_by_key_and_delete_removes():
    profile = _profile(projects=[{"name": "half marathon", "note": "race in October"}],
                       people=[{"name": "Anna", "rel": "colleague"}])
    profile, _ = memory.apply_ops(profile, [
        {"op": "UPDATE", "path": "projects[]", "value": {"name": "half marathon", "note": "race in November"}},
        {"op": "DELETE", "path": "people[]", "value": {"name": "Anna"}},
    ])
    assert profile["projects"] == [{"name": "half marathon", "note": "race in November"}]
    assert profile["people"] == []


def test_delete_scalar_removes_key():
    profile = _profile(preferences={"coffee": "black"})
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
    assert profile["people"] == [{"name": "Bolt", "rel": "dog"}]
    assert applied[0]["path"] == "people[]"


def test_indexed_entry_field_path_updates_one_field():
    profile = _profile(projects=[{"name": "half marathon", "note": "race in October"}])
    profile, applied = memory.apply_ops(
        profile, [{"op": "UPDATE", "path": "projects[0].note", "value": "race in November"}])
    assert profile["projects"] == [{"name": "half marathon", "note": "race in November"}]
    assert applied[0]["path"] == "projects[0].note"


def test_indexed_entry_field_path_out_of_range_is_rejected(capsys):
    profile, applied = memory.apply_ops(
        memory._empty_profile(), [{"op": "UPDATE", "path": "projects[3].note", "value": "x"}])
    assert applied == [] and "projects[3] does not exist" in capsys.readouterr().out


def test_restating_a_scalar_in_another_casing_is_not_an_update():
    profile = _profile(identity={"occupation": "Engineering manager"})
    profile, applied = memory.apply_ops(
        profile, [{"op": "UPDATE", "path": "identity.occupation", "value": "engineering manager!"}])
    assert applied == []
    assert profile["identity"] == {"occupation": "Engineering manager"}  # no bogus superseded


def test_caps_are_enforced_per_list(capsys):
    profile = _profile(people=[{"name": f"p{i}"} for i in range(memory.CAPS["people"])])
    profile, _ = memory.apply_ops(profile, [{"op": "ADD", "path": "people[]", "value": {"name": "newest"}}])
    assert len(profile["people"]) == memory.CAPS["people"]
    assert profile["people"][0]["name"] == "p1"          # oldest evicted
    assert profile["people"][-1]["name"] == "newest"
    assert "cap enforced" in capsys.readouterr().out


def test_episodic_cap_drops_least_important_and_oldest_first():
    profile = _profile(episodic=[
        {"date": _days_ago(i % 5), "text": f"e{i}", "ttl_days": 90, "importance": 1 if i < 5 else 3}
        for i in range(memory.CAPS["episodic"] + 5)
    ])
    profile, _ = memory.apply_ops(profile, [])
    kept = {e["text"] for e in profile["episodic"]}
    assert len(kept) == memory.CAPS["episodic"]
    assert not kept & {f"e{i}" for i in range(5)}  # importance 1 entries went first


# --------------------------------------------------------------- 2. render


def test_render_orders_episodics_by_recency_times_importance_and_drops_expired():
    profile = _profile(episodic=[
        {"date": _days_ago(1), "text": "medium thing", "ttl_days": 30, "importance": 2},
        {"date": _days_ago(2), "text": "big thing", "ttl_days": 30, "importance": 3},
        {"date": _days_ago(40), "text": "expired thing", "ttl_days": 7, "importance": 3},
    ])
    block = memory.render(profile)
    assert "expired thing" not in block
    assert block.index("big thing") < block.index("medium thing")


def test_render_skips_importance_one_episodics_but_keeps_them_on_disk():
    """Everyday trivia that slipped through is stored until its TTL, never prompted."""
    profile = _profile(episodic=[
        {"date": _days_ago(0), "text": "had a sandwich for lunch", "ttl_days": 3, "importance": 1},
        {"date": _days_ago(0), "text": "signed the flat contract", "ttl_days": 30, "importance": 3},
    ])
    block = memory.render(profile)
    assert "sandwich" not in block
    assert "flat contract" in block
    assert len(profile["episodic"]) == 2
    assert len(memory.live_episodics(profile)) == 2  # unfiltered view still sees both


def test_render_folds_nickname_into_the_name_line():
    profile = _profile(identity={"name": "Mihaly", "nickname": "Misi", "location": "Budapest"})
    assert memory.render(profile) == "Identity: name: Mihaly (call them: Misi); location: Budapest"


def test_render_shows_a_nickname_without_a_name():
    profile = _profile(identity={"nickname": "Misi"})
    assert memory.render(profile) == "Identity: call them: Misi"


def test_render_shows_at_most_five_episodics_and_stays_in_budget():
    profile = _profile(
        identity={"name": "Mihaly"},
        episodic=[{"date": _days_ago(i), "text": f"episode number {i} " + "x" * 200,
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
    return _profile(
        identity={"name": "Mihaly"},
        preferences={"coffee": "black"},
        people=[{"name": "Anna", "rel": "colleague"}, {"name": "anna", "note": "joined the team"}],
        episodic=[{"date": _days_ago(40), "text": "rained all morning", "ttl_days": 3, "importance": 1},
                  {"date": _days_ago(9), "text": "Went to the gym after work", "ttl_days": 30, "importance": 2},
                  {"date": _days_ago(7), "text": "Gym session, legs day", "ttl_days": 30, "importance": 2},
                  {"date": _days_ago(2), "text": "Hit the gym again on Thursday", "ttl_days": 30, "importance": 2}],
    )


def test_reflect_merges_duplicates_drops_expired_and_promotes():
    profile = memory.reflect(_dup_profile(), FakeClient(REFLECTED), "m")
    assert [p["name"] for p in profile["people"]] == ["Anna"]
    assert profile["episodic"] == []
    assert profile["recurring"] == [{"what": "gym", "when": "Tue/Thu"}]


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
    memory._write_profile(_profile(identity={"name": "Mihaly"}, preferences={"coffee": "black"}))
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
    memory._write_profile(_profile(identity={"occupation": "software engineer"},
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


def test_propose_ops_survives_a_broken_client(capsys):
    client = FakeClient(RuntimeError("connection refused"))
    assert memory.propose_ops(memory._empty_profile(), TURNS, client, "m") == []
    assert "op proposal failed" in capsys.readouterr().out


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
    monkeypatch.setattr(memory, "save", lambda turns, client, model: saved.append(list(turns)))
    monkeypatch.setattr(llm, "client", lambda: object())
    llm.reset()
    llm._session_turns.append({"role": "user", "content": "hi"})
    llm.save_memory(); llm.save_memory()
    assert saved == [[{"role": "user", "content": "hi"}]]


def test_reflect_cannot_invent_recurring_entries():
    original = memory._empty_profile()
    original["episodic"] = [{"date": "2026-08-20", "text": "Went to the gym after work", "ttl_days": 30, "importance": 2}]
    cleaned = memory._empty_profile()
    cleaned["recurring"] = [{"what": "gym", "when": "evenings"}, {"what": "skydiving", "when": "Sundays"}]
    out = memory._validate_reflection(original, cleaned)
    assert [r["what"] for r in out["recurring"]] == ["gym"]
