"""PRD 49 scenario 9: scripts/trace_report.py over a fixture events file (docs/49-event-trace.md)."""
import json
import os
import subprocess
import sys

from tests.conftest import ROOT

REPORT = os.path.join(ROOT, "scripts", "trace_report.py")


def _row(event_kind, ts, **fields):
    return {"ts": f"2026-10-16T10:00:{ts:02d}.000+02:00", "profile": "prod", "pid": 42, **fields, "kind": event_kind}


def _span(event_kind, ts, start, **fields):
    return _row(event_kind, ts, span_id=f"s{start}", parent_id="root", status="ok", duration_ms=5,
                start=f"2026-10-16T10:00:{start:02d}.000+02:00", **fields)


FIXTURE = [
    _row("service", 0, event="start", version="v1"),
    _row("turn_start", 1, turn_id=1, run_id="42-1", origin="typed", text_len=10, words=2),
    _span("llm_call", 3, 2, turn_id=1, run_id="42-1", purpose="reply", ms=900, ttft_ms=200, tokens_in=10,
          tokens_out=5),
    _row("turn_end", 4, turn_id=1, run_id="42-1", ms=1000, reply_len=40, tools=[], band="low", outcome="ok"),
    _row("turn_start", 5, turn_id=2, run_id="42-2", origin="spoken", text_len=20, words=4),
    # written out of order: the llm_call ends after the tool_call started
    _span("llm_call", 9, 8, turn_id=2, run_id="42-2", purpose="tool_round", ms=500, ttft_ms=100),
    _span("tool_call", 7, 6, turn_id=2, run_id="42-2", name="search_wiki", ms=30, result_len=300, ok=True),
    _row("error", 9, turn_id=2, run_id="42-2", scope="turn", where="respond", exc_type="RuntimeError",
         message="boom", traceback='  File "llm.py", line 1, in ask_events\n    x()\n'),
    _row("turn_end", 10, turn_id=2, run_id="42-2", ms=4000, reply_len=0, tools=["search_wiki"], outcome="error"),
    _span("ingest_start", 11, 11, ingest_id="42-1", run_id="42-i1", ingest_kind="note", raw="raw/inbox/a.md", bytes=10),
    _span("ingest_end", 12, 11, ingest_id="42-1", run_id="42-i1", path="sources/a", title="A", ms=800,
          ops_applied=2, ops_rejected=[{"kind": "new_page", "reason_code": "not_indexed"}], pages=["sources/a"], degraded=False),
]


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _report(*args):
    return subprocess.run([sys.executable, REPORT, *args], capture_output=True, text=True, timeout=30)


def test_turn_prints_only_that_turns_events_in_start_order(tmp_path):
    out = _report(_write(tmp_path / "events.jsonl", FIXTURE), "--turn", "2")
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    kinds = [json.loads(line)["kind"] for line in lines]
    assert kinds == ["turn_start", "tool_call", "llm_call", "error", "turn_end"]
    assert all(json.loads(line)["turn_id"] == 2 for line in lines)
    assert _report(str(tmp_path / "events.jsonl"), "--turn", "42:2").stdout == out.stdout


def test_errors_prints_one_row_and_exits_one(tmp_path):
    out = _report(_write(tmp_path / "events.jsonl", FIXTURE), "--errors")
    assert out.returncode == 1
    rows = [line for line in out.stdout.splitlines() if "RuntimeError" in line]
    assert len(rows) == 1 and "turn" in rows[0] and 'llm.py", line 1' in rows[0]


def test_a_clean_file_exits_zero(tmp_path):
    clean = [r for r in FIXTURE if r["kind"] != "error"]
    out = _report(_write(tmp_path / "events.jsonl", clean), "--errors")
    assert out.returncode == 0, out.stdout


def test_the_summary_has_a_turn_table_an_ingest_table_and_reads_rotated_files(tmp_path):
    _write(tmp_path / "events.jsonl.1", FIXTURE[:4])
    path = _write(tmp_path / "events.jsonl", FIXTURE[4:])
    out = _report(path)
    assert out.returncode == 0, out.stderr
    assert "42:1" in out.stdout and "42:2" in out.stdout          # both turns, one from the rotated file
    assert "42-1" in out.stdout and "note" in out.stdout and "2/1" in out.stdout   # the ingest row
    assert "RuntimeError" in out.stdout


def test_a_torn_last_line_and_junk_lines_are_skipped(tmp_path):
    path = tmp_path / "events.jsonl"
    body = "".join(json.dumps(r) + "\n" for r in FIXTURE[:4]).encode()
    body += b'[1, 2]\n"a string"\n\xff\xfe not utf-8\n' + json.dumps(FIXTURE[4]).encode()[:25]
    path.write_bytes(body)
    for args in ([], ["--turn", "1"]):
        out = _report(str(path), *args)
        assert out.returncode == 0, out.stderr
        assert "42:1" in out.stdout or args
    assert _report(str(path), "--errors").returncode == 0
