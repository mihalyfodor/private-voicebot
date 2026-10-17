"""PRD 51 scenarios 1, 2, 6, 7: the run events of a briefing, a write-up and a batch import
(docs/51-run-cards.md). The PRD names tests/trace/; the trace tests live in tests/tracing/.

Scenarios 3-4 are in test_run_page.py, 5 in tests/ingest/test_sources.py, 8 in
tests/server/test_run_cards.py, 9 in tests/e2e/test_app.py, 10 is tests/test_protocol_doc.py.
"""
import os
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ingest import batch as ingest_batch
from server import app
from server import turns
from server import ws
from server import speech
from agent import written
from agent import conversation
from agent import written
from agent import runs
from core import llm as core_llm
from core import trace
from server.session import Session
from tests.conftest import (ROOT, RUN_TITLE, SENTINEL, FakeClient, briefing_run, fake_tts, pdf_bytes, run_turn, trace_lines,
                            trace_text)
from ingest import batch as ingest_batch
from core import util

pytestmark = pytest.mark.usefixtures("fresh_llm")


def _run_rows(run_id: str) -> list:
    return [r for r in trace_lines() if r.get("run_id") == run_id]


# --------------------------------------------------------------- 1. a briefing is a run


def test_a_briefing_is_a_run_with_three_steps_and_a_run_card(vault, monkeypatch):
    got = briefing_run(monkeypatch)
    rows = _run_rows(got["run_id"])
    run_events = [r for r in rows if r["kind"] == "run"]
    assert [(r["phase"], r["workflow"]) for r in run_events] == [("start", "briefing"), ("end", "briefing")]
    assert run_events[1]["status"] == "done" and run_events[1]["reason"] == "ok"
    steps = [r for r in rows if r["kind"] == "step"]
    assert [s["name"] for s in steps] == ["gather", "skip", "phrase"]
    assert all(s["span_id"] and s["parent_id"] and s["start"] and s["duration_ms"] >= 0 for s in steps)
    gather, skip, _phrase = steps
    assert gather["count"] == 1 and gather["detail"] == "since 00:00"
    assert gather["pages"] == [{"path": "notes/plan", "commit": got["sha"], "title": RUN_TITLE}]
    assert (skip["status"], skip["count"], skip["reason"], skip["detail"]) == ("skipped", 2, "housekeeping",
                                                                                "2 housekeeping commits")
    card = got["card"]
    assert card["run_url"] == f"/runs/{got['run_id']}" and card["run_id"] == got["run_id"]
    assert card["workflow"] == "briefing" and card["status"] == "done" and card["markdown"] == ""
    assert [(s["label"], s["status"]) for s in card["steps"]] == [("gathered", "ok"), ("skipped", "skipped"),
                                                                   ("phrased", "ok")]
    assert card["steps"][1]["detail"] == "2 housekeeping commits"
    assert card["steps"][0]["detail"] == "since 1 Jan — 1 page, 0 sources, 0 memory changes"
    assert card["actions"] == [runs.OPEN]


def test_a_chit_chat_turn_emits_no_run_event(vault):
    with patch.object(core_llm, "_client", FakeClient("[happy] Hi!")), patch.object(speech, "synthesize", fake_tts):
        sent = run_turn(turns.handle_text, Session(), 1, "hello there")
    assert trace_lines(kind="run") == [] and trace_lines(kind="step") == []
    assert not [m for m in sent if m["type"] in ("card", "card_update")]


def test_steps_beyond_six_fold_into_n_more():
    steps = [{"label": f"s{i}", "status": "ok"} for i in range(8)]
    steps[7]["status"] = "error"
    folded = runs.card_steps(steps)
    assert len(folded) == 6 and folded[-1] == {"label": "+3 more", "status": "error"}
    assert runs.card_steps(steps[:6]) == steps[:6]


def test_every_reason_code_is_an_allowed_event_value():
    rule = trace.FIELDS["reason"]
    assert all(rule.fullmatch(code) for code in runs.REASONS)
    assert rule.fullmatch("failed-RuntimeError") and not rule.fullmatch(f"failed {SENTINEL}")


# --------------------------------------------------------------- 2. nothing new


def test_an_empty_digest_ends_the_run_nothing_without_a_model_step(vault, monkeypatch):
    got = briefing_run(monkeypatch, commits_after=False)
    (end,) = [r for r in _run_rows(got["run_id"]) if r["kind"] == "run" and r["phase"] == "end"]
    assert (end["status"], end["reason"]) == ("nothing", "nothing-to-do")
    assert trace_lines(kind="llm_call") == []
    assert [s["name"] for s in trace_lines(kind="step")] == ["gather"]
    assert got["card"]["status"] == "nothing" and [s["label"] for s in got["card"]["steps"]] == ["gathered"]


# --------------------------------------------------------------- 6. a write-up that fails


def _grounded_context():
    """The last answer's context, as `ask_events` leaves it: the sentinel sits in the question and reply."""
    conversation._last_context = {"question": f"what did {SENTINEL} say?", "extra": f"<recall>{SENTINEL} said so</recall>",
                         "tools": [], "reply": f"[neutral] {SENTINEL} said so.", "sources": [], "pages": []}


def test_a_write_up_whose_completion_raises_updates_its_card_to_failed(vault, prod_profile):
    _grounded_context()
    with patch.object(core_llm, "_client", FakeClient(RuntimeError(f"model died {SENTINEL}"))), \
            patch.object(speech, "synthesize", fake_tts):
        sent = run_turn(turns.handle_text, Session(), 1, "write it up")
    types = [m["type"] for m in sent]
    assert types.index("card") < types.index("card_update")
    card = next(m for m in sent if m["type"] == "card")
    update = next(m for m in sent if m["type"] == "card_update")
    assert card["status"] == "running" and card["workflow"] == "write-up" and card["markdown"] == ""
    assert update["id"] == card["id"] and update["status"] == "failed"
    assert update["steps"][-1] == {"label": "write", "status": "error", "detail": "Failed with RuntimeError."}
    assert any(m["type"] == "error" and m["scope"] == "turn" for m in sent)
    (end,) = [r for r in trace_lines(kind="run") if r["phase"] == "end"]
    assert (end["status"], end["reason"]) == ("failed", "failed-RuntimeError")
    page = TestClient(app.app).get(card["run_url"])
    assert page.status_code == 200 and "RuntimeError" in page.text and "Write-up" in page.text
    assert SENTINEL not in trace_text() and SENTINEL not in page.text


def test_a_write_up_that_works_sends_its_markdown_on_the_update_and_replays_done(vault):
    _grounded_context()
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="# Key\n\n## A\nx\n\n## B\ny"))])
    with patch.object(core_llm, "_client", FakeClient(reply)), patch.object(speech, "synthesize", fake_tts):
        sent = run_turn(turns.handle_text, Session(), 1, "write it up")
    update = next(m for m in sent if m["type"] == "card_update")
    assert update["status"] == "done" and update["markdown"].startswith("# Key")
    assert [(s["label"], s["status"]) for s in update["steps"]] == [("context", "ok"), ("write", "ok")]
    # live and replay carry the same title and sources (the running card has neither yet)
    assert update["title"] == "Key" and update["sources"] == []
    assert [a["kind"] for a in update["actions"]] == ["open", "copy"]
    assert [s["name"] for s in trace_lines(kind="step")] == ["context", "write"]
    replayed = [m for m in ws.replay_messages() if m["type"] == "card"]
    assert replayed[-1]["status"] == "done" and replayed[-1]["steps"] == update["steps"]


# --------------------------------------------------------------- 7. a batch import


def test_a_batch_import_is_one_run_with_a_needs_ocr_did_not_row(vault, monkeypatch, tmp_path):
    folder = tmp_path / "export"
    folder.mkdir()
    (folder / "notes.pdf").write_bytes(pdf_bytes([["Falcon status note, amber, load test 940 ms p99"]]))
    (folder / "scan.pdf").write_bytes(pdf_bytes([["ignored"]], with_text=False))
    monkeypatch.setattr(util, "running_app", lambda: "")
    summary = '{"title": "Falcon notes", "summary": "Falcon.", "entities": [], "topic": "work", "unreadable": false}'
    with patch.object(core_llm, "_client", FakeClient(summary, "[]")):
        res = ingest_batch.ingest_batch(str(folder), out=lambda line: None)
    assert res["filed"] == 1 and res["needs-ocr"] == 1
    starts = [r for r in trace_lines(kind="run") if r["phase"] == "start"]
    assert len(starts) == 1 and starts[0]["workflow"] == "batch" and starts[0]["run_id"].startswith("batch-")
    run_id = starts[0]["run_id"]
    steps = [r for r in trace_lines(kind="step")]
    assert [(s["rel"], s["status"], s.get("reason")) for s in steps] == [("notes.pdf", "ok", None),
                                                                         ("scan.pdf", "skipped", "needs-ocr")]
    assert {s["run_id"] for s in steps} == {run_id}
    (end,) = trace_lines(kind="ingest_end")
    assert end["run_id"] == run_id and re.fullmatch(r"[0-9a-f]{7,}", end["commit"])
    run = runs.build(runs.load(run_id))
    assert run["workflow"] == "batch" and run["status"] == "done"
    assert run["did_not"] == [{"item": "scan.pdf", "reason": "needs-ocr"}]
    assert [p["commit"] for p in run["did"]["pages"]] == [end["commit"]]


def test_the_batch_script_in_dev_against_docker_data_exits_2(tmp_path):
    home = tmp_path / "home"
    vault = home / "docker-data" / "voicebot" / "llm-wiki"     # looks like prod data by the refusal's own rule
    vault.mkdir(parents=True)
    (tmp_path / "export").mkdir()
    env = dict(os.environ, HOME=str(home), TRACE_PROFILE="dev", WIKI_PATH=str(vault),
               MEMORY_DIR=str(tmp_path / "memory"), TRACE_DIR=str(tmp_path / "logs"), PORT="1")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ingest_batch.py"), str(tmp_path / "export")],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert out.returncode == 2 and "refusing" in out.stderr
    assert not (tmp_path / "logs").exists()


def test_a_batch_that_raises_mid_run_still_ends_its_run_failed(vault, monkeypatch, tmp_path):
    """Review MAJOR 1: every `run start` gets one `run end`, also when the loop raises (Ctrl-C included)."""
    folder = tmp_path / "export"
    folder.mkdir()
    for name in ("a.pdf", "b.pdf"):
        (folder / name).write_bytes(pdf_bytes([["x"]], with_text=False))
    monkeypatch.setattr(util, "running_app", lambda: "")
    for exc in (RuntimeError("disk gone"), KeyboardInterrupt()):
        calls = []

        def boom(path, rel, kinds, dry_run, exc=exc):
            calls.append(rel)
            if len(calls) == 2:
                raise exc
            return {"state": "needs-ocr", "reason": "0 chars/page"}

        monkeypatch.setattr(ingest_batch, "_one", boom)
        before = len(trace_lines(kind="run"))
        with pytest.raises(type(exc)):
            ingest_batch.ingest_batch(str(folder), out=lambda line: None)
        rows = trace_lines(kind="run")[before:]
        assert [r["phase"] for r in rows] == ["start", "end"]
        run = runs.build(runs.load(rows[0]["run_id"]))
        assert (run["status"], run["reason"]) == ("failed", f"failed-{type(exc).__name__}")
        (folder / "batch.log").unlink()


def test_a_dry_run_writes_no_step_spans(vault, monkeypatch, tmp_path):
    folder = tmp_path / "export"
    folder.mkdir()
    (folder / "a.pdf").write_bytes(pdf_bytes([["Falcon status note, amber, load test 940 ms p99"]]))
    monkeypatch.setattr(util, "running_app", lambda: "")
    ingest_batch.ingest_batch(str(folder), dry_run=True, out=lambda line: None)
    assert trace_lines(kind="step") == [] and trace_lines(kind="run") == []


def test_a_briefing_that_fails_while_gathering_labels_the_gather_step(vault, monkeypatch):
    from agent import digest

    def boom(since):
        raise RuntimeError("git gone")

    monkeypatch.setattr(digest, "changes_since", boom)
    with patch.object(core_llm, "_client", FakeClient()), patch.object(speech, "synthesize", fake_tts):
        sent = run_turn(turns.handle_text, Session(), 1, "what's new?")
    (card,) = [m for m in sent if m["type"] == "card"]
    assert card["status"] == "failed"
    assert card["steps"] == [{"label": "gathered", "status": "error", "detail": "Failed with RuntimeError."}]


def _write_up(client) -> list:
    with patch.object(core_llm, "_client", client), patch.object(speech, "synthesize", fake_tts):
        return run_turn(turns.handle_text, Session(), 1, "write it up")


def _card_reply(markdown: str = "# Key\n\n## A\nx\n\n## B\ny"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=markdown))])


def test_a_reconnect_while_the_write_up_runs_replays_the_running_card(vault, monkeypatch):
    """Review MAJOR 2: the running card is in the replay store from turn start; the update then applies."""
    _grounded_context()
    seen = []
    real = written._card_completion

    def completion(ctx):
        seen.extend(ws.replay_messages())          # a reconnect in the middle of the write-up
        return real(ctx)

    monkeypatch.setattr(written, "_card_completion", completion)
    sent = _write_up(FakeClient(_card_reply()))
    card = next(m for m in sent if m["type"] == "card")
    mid = [m for m in seen if m["type"] == "card"]
    assert len(mid) == 1 and mid[0]["id"] == card["id"]
    assert (mid[0]["status"], mid[0]["workflow"], mid[0]["markdown"]) == ("running", "write-up", "")
    after = [m for m in ws.replay_messages() if m["type"] == "card"]
    assert len(after) == 1 and after[0]["id"] == card["id"] and after[0]["status"] == "done"
    assert after[0]["markdown"].startswith("# Key")


def test_a_failed_write_up_survives_a_reconnect_and_a_retry_gets_a_new_id(vault):
    _grounded_context()
    failed = next(m for m in _write_up(FakeClient(RuntimeError("down"))) if m["type"] == "card")
    replay = [m for m in ws.replay_messages() if m["type"] == "card"]
    assert [(m["id"], m["status"]) for m in replay] == [(failed["id"], "failed")]
    retry = next(m for m in _write_up(FakeClient(_card_reply())) if m["type"] == "card")
    assert retry["id"] != failed["id"]
    replay = [m for m in ws.replay_messages() if m["type"] == "card"]
    assert [(m["id"], m["status"]) for m in replay] == [(failed["id"], "failed"), (retry["id"], "done")]
    with pytest.raises(ValueError):
        turns.save_card(failed["id"])                 # a card without a body is nothing to save


def test_the_three_workflow_lists_agree():
    js = open(os.path.join(ROOT, "static", "app.js")).read()
    block = re.search(r"const RUN_WORKFLOWS = \{(.*?)\};", js, re.S).group(1)
    client = set(re.findall(r"'?([a-z-]+)'?\s*:", block))
    assert set(runs.WORKFLOWS) == trace.FIELDS["workflow"] == client
