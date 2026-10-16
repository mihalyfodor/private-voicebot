"""Unit tests for eval/harness.py — no network, no oMLX."""
import os

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core import settings
import llm
import memory
from core import trace
from eval import harness
from tests.conftest import RawStreamClient, stream_chunks as _chunks, seed as _write_profile
from core import llm as core_llm


# --------------------------------------------------------------------------- fakes



def _ops(*ops):
    """One memory.propose_ops completion carrying `ops` as a JSON array."""
    import json
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(list(ops))))])


# --------------------------------------------------------------------------- scoring


def test_expect_flat_list_is_any_of():
    assert harness.score_probe("His name is Bolt.", expect=["bolt", "rex"])["passed"]
    assert not harness.score_probe("No idea.", expect=["bolt", "rex"])["passed"]


def test_expect_groups_all_must_hit():
    r = harness.score_probe("You are an engineering manager in Budapest.",
                            expect=[["manager"], ["budapest"]])
    assert r["passed"] and not r["missing"]
    r = harness.score_probe("You are an engineering manager.", expect=[["manager"], ["budapest"]])
    assert not r["passed"] and r["missing"] == [["budapest"]]


def test_scoring_is_case_and_punctuation_insensitive():
    assert harness.score_probe("MIHALY!", expect=["mihaly"])["passed"]
    assert harness.score_probe("You don't know, sorry", expect=["don't know"])["passed"]


def test_expect_not_blocks_stale_value():
    r = harness.score_probe("It's in October.", expect=["october"], expect_not=["october"])
    assert not r["passed"]
    assert r["forbidden_hits"] == ["october"]
    assert harness.score_probe("It's in November.", expect=["november"],
                               expect_not=["october"])["passed"]


def test_expect_not_only_probe():
    assert harness.score_probe("I don't know that.", expect_not=["her name is"])["passed"]
    assert not harness.score_probe("Her name is Kata.", expect_not=["her name is"])["passed"]


def test_category_rates():
    probes = [
        {"category": "single-hop", "passed": True},
        {"category": "single-hop", "passed": False},
        {"category": "update", "passed": True},
        {"passed": True},
    ]
    rates = harness.category_rates(probes)
    assert rates["single-hop"] == {"passed": 1, "total": 2, "rate": 0.5}
    assert rates["update"]["rate"] == 1.0
    assert rates["uncategorised"]["total"] == 1


# --------------------------------------------------------------------------- memory stats


def test_duplicate_detection_ignores_headers_and_blanks():
    text = (
        "\n--- 2026-01-01 12:00 ---\n"
        "The user's dog is named Bolt.\n"
        "\n--- 2026-01-02 12:00 ---\n"
        "The user's dog is called Bolt.\n"
        "The user lives in Budapest.\n"
    )
    stats = harness.memory_stats(text)
    assert stats["fact_lines"] == 3
    assert stats["duplicates"] == 1
    assert stats["duplicate_pairs"][0]["duplicate_of"] == "The user's dog is named Bolt."
    assert stats["tokens_est"] == len(text) // 4


def test_no_false_duplicates_for_distinct_facts():
    text = "The user lives in Budapest.\nThe user drinks coffee black.\n"
    assert harness.memory_stats(text)["duplicates"] == 0


def test_appended_diff():
    assert harness._appended("a\n", "a\nb\n") == "b\n"
    assert "b" in harness._appended("a\nx\n", "a\nb\n")


# --------------------------------------------------------------------------- sandbox


def test_sandbox_repoints_every_path_and_restores_them(tmp_path):
    """Sandbox must swap MEMORY_PATH / SETTINGS_PATH / TRACE_DIR for its own, and put them back.

    The "current" paths are the test sandbox's; a marker written there first proves that a
    save inside harness.Sandbox never reaches the outer file, and that it is untouched after exit.
    """
    outer_mem = memory.MEMORY_PATH
    outer_settings = settings.SETTINGS_PATH
    outer_trace = trace.TRACE_DIR
    _write_profile(identity={"name": "outer marker"})
    marker = open(outer_mem).read()

    with harness.Sandbox() as box:
        assert memory.MEMORY_PATH != outer_mem
        assert "memeval-" in memory.MEMORY_PATH
        assert "memeval-" in memory.OPS_LOG_PATH
        assert "memeval-" in memory.SHORTMEM_PATH
        assert settings.SETTINGS_PATH != outer_settings
        assert trace.TRACE_DIR != outer_trace
        assert "outer marker" not in box.read_memory()
        _write_profile(identity={"name": "sandboxed fact"})
        settings.save_setting("verbosity", "short")
        trace.event("turn_end", ms=1, outcome="ok")
        assert "sandboxed fact" in box.read_memory()
        assert os.path.exists(os.path.join(box.trace_dir, "events.jsonl"))
        sandbox_mem = box.memory_path

    assert memory.MEMORY_PATH == outer_mem
    assert settings.SETTINGS_PATH == outer_settings
    assert trace.TRACE_DIR == outer_trace
    assert not os.path.exists(sandbox_mem)
    assert open(outer_mem).read() == marker
    assert not os.path.exists(outer_settings)
    assert not os.path.exists(os.path.join(outer_trace, "events.jsonl"))


def test_sandbox_keeps_given_memory_path(tmp_path):
    keep = tmp_path / "mem.json"
    with harness.Sandbox(memory_path=str(keep)) as box:
        _write_profile(identity={"name": "kept fact"})
        assert box.memory_path == str(keep)
        assert box.ops_path == str(tmp_path / "memory_ops.jsonl")
    assert "kept fact" in keep.read_text()


def test_sandbox_reads_v1_shortmem_when_there_is_no_profile():
    with harness.Sandbox() as box:
        with open(memory.SHORTMEM_PATH, "w") as f:
            f.write("The user is named Mihaly.\n")
        assert box.read_memory() == "The user is named Mihaly.\n"
        assert harness.memory_stats(box.read_memory())["version"] == 1
        _write_profile(identity={"name": "Mihaly"})
        assert harness.memory_stats(box.read_memory())["version"] == 2


def test_session_end_falls_back_to_the_v1_shortmem_delta():
    """A v1 save (no ops logged) still reports what landed in shortmem.txt."""
    def fake_save():
        with open(memory.SHORTMEM_PATH, "a") as f:
            f.write("The user is named Mihaly.\n")

    with harness.Sandbox(), patch.object(llm, "save_memory", side_effect=fake_save):
        sess = harness.Session().start()
        assert sess.end() == "The user is named Mihaly.\n"
        assert sess.last_ops == []


def test_sandbox_disables_tools_by_default_and_keeps_them_on_request():
    real_tools = llm.TOOLS
    assert real_tools, "llm.TOOLS is expected to be non-empty outside a sandbox"
    with harness.Sandbox():
        assert llm.TOOLS == []
    assert llm.TOOLS == real_tools
    with harness.Sandbox(tools=True):
        assert llm.TOOLS == real_tools
    assert llm.TOOLS == real_tools


# --------------------------------------------------------------------------- session


def test_session_say_strips_tag_and_end_returns_applied_ops():
    fake = RawStreamClient([_chunks("[happy] Hi ", "Mihaly."),
                       _ops({"op": "ADD", "path": "identity.name", "value": "Mihaly",
                             "reason": "said so"})])
    with harness.Sandbox() as box, patch.object(core_llm, "_client", fake):
        sess = harness.Session(box).start()
        assert sess.say("hi") == "Hi Mihaly."
        delta = sess.end()
        stats = harness.memory_stats(box.read_memory())
        ops = sess.last_ops
    assert delta == "ADD identity.name = Mihaly  (said so)"
    assert [op["path"] for op in ops] == ["identity.name"]
    assert stats["version"] == 2 and stats["sections"]["identity"] == 1
    assert fake.calls[0]["tools"] == []


def test_session_end_with_nothing_new():
    fake = RawStreamClient([_chunks("[neutral] Sure."), _ops()])
    with harness.Sandbox(), patch.object(core_llm, "_client", fake):
        sess = harness.Session().start()
        sess.say("hello")
        assert sess.end() == ""
        assert sess.last_ops == []


# --------------------------------------------------------------------------- run_script


SCRIPT = [
    [{"user": "I'm Mihaly"}, {"user": "my dog is Bolt"}],
    [
        {"probe": "my name?", "expect": ["mihaly"], "category": "single-hop"},
        {"probe": "my dog?", "expect": ["rex"], "category": "single-hop"},
        {"probe": "marathon month?", "expect": ["november"], "expect_not": ["october"],
         "category": "update"},
    ],
]


def test_run_script_wiring_with_stub_ask():
    replies = {
        "I'm Mihaly": "[happy] Nice to meet you.",
        "my dog is Bolt": "[happy] Good name.",
        "my name?": "[neutral] You're Mihaly.",
        "my dog?": "[neutral] Bolt.",
        "marathon month?": "[neutral] October.",
    }
    asked = []

    def fake_ask(text):
        asked.append(text)
        return replies[text]

    with patch.object(llm, "ask", side_effect=fake_ask), \
         patch.object(llm, "save_memory") as save:
        results = harness.run_script(SCRIPT, verbose=False)

    assert asked == list(replies)
    assert save.call_count == 2
    probes = results["rounds"][0]["probes"]
    assert [p["passed"] for p in probes] == [True, False, False]
    assert probes[2]["forbidden_hits"] == ["october"]
    assert results["rounds"][0]["overall"] == {"passed": 1, "total": 3, "rate": 0.333}
    assert results["rounds"][0]["categories"]["single-hop"]["rate"] == 0.5
    assert probes[0]["session"] == 2 and probes[0]["round"] == 1


def test_run_script_rounds_persist_memory(tmp_path):
    calls = []

    def fake_ask(text):
        calls.append(text)
        return "[neutral] You're Mihaly."

    saves = []

    def fake_save():
        """Upsert the same name every time, plus one new person per save."""
        saves.append(1)
        client = RawStreamClient([_ops(
            {"op": "ADD", "path": "identity.name", "value": "Mihaly"},
            {"op": "ADD", "path": "people[]", "value": {"name": "Anna %d" % len(saves)}},
        )])
        core_llm._client = client
        memory.save([{"role": "user", "content": "hi"}])

    with patch.object(llm, "ask", side_effect=fake_ask), \
         patch.object(llm, "save_memory", side_effect=fake_save):
        results = harness.run_script(SCRIPT, rounds=2, verbose=False)

    assert len(results["rounds"]) == 2
    assert calls == ["I'm Mihaly", "my dog is Bolt", "my name?", "my dog?",
                     "marathon month?"] * 2
    # 4 saves: the repeated name is upserted once, each person is added once
    assert len(saves) == 4
    assert results["memory"]["sections"] == {"identity": 1, "preferences": 0, "people": 4,
                                             "projects": 0, "recurring": 0, "episodic": 0}
    assert results["memory"]["duplicates"] == 0
    assert results["rounds"][0]["memory"]["sections"]["people"] == 2
    assert results["rounds"][0]["sessions"][0]["memory_ops"][0]["path"] == "identity.name"


def test_run_script_reads_yaml_file():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "eval", "scripts", "basic.yaml")
    script = harness.load_script(os.path.abspath(path))
    assert len(script) == 4
    cats = {t["category"] for s in script for t in s if "probe" in t}
    assert cats <= set(harness.CATEGORIES)
    assert {"single-hop", "update", "abstention", "temporal", "preference",
            "multi-session"} == cats


def test_format_report_mentions_failures():
    with patch.object(llm, "ask", side_effect=lambda t: "[neutral] October."), \
         patch.object(llm, "save_memory"):
        results = harness.run_script(SCRIPT, verbose=False)
    report = harness.format_report(results)
    assert "round 1" in report and "FAIL" in report and "stale october" in report


def test_cli_writes_out_json(tmp_path):
    out = tmp_path / "r.json"
    script = tmp_path / "s.json"
    script.write_text('[[{"probe": "my name?", "expect": ["mihaly"], "category": "single-hop"}]]')
    with patch.object(llm, "ask", side_effect=lambda t: "[neutral] Mihaly."), \
         patch.object(llm, "save_memory"):
        assert harness.main([str(script), "--out", str(out), "--quiet"]) == 0
    import json
    data = json.loads(out.read_text())
    assert data["rounds"][0]["overall"]["rate"] == 1.0
    assert data["script"] == str(script)


def test_cli_requires_script_without_chat():
    with pytest.raises(SystemExit):
        harness.main(["--quiet"])
