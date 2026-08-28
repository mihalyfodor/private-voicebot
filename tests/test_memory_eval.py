"""Unit tests for eval/harness.py — no network, no oMLX."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import avatars
import llm
import memory
from eval import harness


# --------------------------------------------------------------------------- fakes


class FakeClient:
    """Same shape as tests/test_llm_unit.py's FakeClient."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self._responses.pop(0)


def _chunks(*texts):
    return iter([
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t, tool_calls=None))])
        for t in texts
    ])


def _summary(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


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


def test_sandbox_isolates_real_files(tmp_path):
    real_mem = memory.SHORTMEM_PATH
    real_settings = avatars.SETTINGS_PATH
    real_tools = llm.TOOLS
    real_mem_before = open(real_mem).read() if os.path.exists(real_mem) else None

    with harness.Sandbox() as box:
        assert memory.SHORTMEM_PATH != real_mem
        assert avatars.SETTINGS_PATH != real_settings
        assert llm.TOOLS == []
        memory._append_atomic("sandboxed fact\n")
        avatars.save_setting("verbosity", "short")
        llm._log("hi", [], "there")
        assert "sandboxed fact" in box.read_memory()
        assert os.path.exists(box.log_path)
        sandbox_mem = box.memory_path

    assert memory.SHORTMEM_PATH == real_mem
    assert avatars.SETTINGS_PATH == real_settings
    assert llm.TOOLS == real_tools
    assert not os.path.exists(sandbox_mem)
    real_mem_after = open(real_mem).read() if os.path.exists(real_mem) else None
    assert real_mem_after == real_mem_before
    if real_mem_after is not None:
        assert "sandboxed fact" not in real_mem_after


def test_sandbox_keeps_given_memory_path(tmp_path):
    keep = tmp_path / "mem.txt"
    with harness.Sandbox(memory_path=str(keep)) as box:
        memory._append_atomic("kept fact\n")
        assert box.memory_path == str(keep)
    assert keep.read_text() == "kept fact\n"


def test_sandbox_can_keep_tools_enabled():
    real_tools = llm.TOOLS
    with harness.Sandbox(tools=True):
        assert llm.TOOLS == real_tools


# --------------------------------------------------------------------------- session


def test_session_say_strips_tag_and_end_returns_delta():
    fake = FakeClient([_chunks("[happy] Hi ", "Mihaly."), _summary("The user is named Mihaly.")])
    with harness.Sandbox() as box, patch.object(llm, "_client", fake):
        sess = harness.Session(box).start()
        assert sess.say("hi") == "Hi Mihaly."
        delta = sess.end()
    assert "The user is named Mihaly." in delta
    assert fake.calls[0]["tools"] == []


def test_session_end_with_nothing_new():
    fake = FakeClient([_chunks("[neutral] Sure."), _summary("NOTHING")])
    with harness.Sandbox(), patch.object(llm, "_client", fake):
        sess = harness.Session().start()
        sess.say("hello")
        assert sess.end() == ""


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

    def fake_save():
        memory._append_atomic("The user is named Mihaly.\n")

    with patch.object(llm, "ask", side_effect=fake_ask), \
         patch.object(llm, "save_memory", side_effect=fake_save):
        results = harness.run_script(SCRIPT, rounds=2, verbose=False)

    assert len(results["rounds"]) == 2
    assert calls == ["I'm Mihaly", "my dog is Bolt", "my name?", "my dog?",
                     "marathon month?"] * 2
    # 4 save_memory calls, each appending the same line -> 3 near-duplicates
    assert results["memory"]["fact_lines"] == 4
    assert results["memory"]["duplicates"] == 3
    assert results["rounds"][0]["memory"]["fact_lines"] == 2


def test_run_script_reads_yaml_file():
    path = os.path.join(os.path.dirname(__file__), "..", "eval", "scripts", "basic.yaml")
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
