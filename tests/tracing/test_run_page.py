"""PRD 51 scenarios 3-4: `GET /runs/{run_id}`, the run page (docs/51-run-cards.md). The PRD names
tests/trace/; the trace tests live in tests/tracing/."""
import html

import pytest
from fastapi.testclient import TestClient

import chatbot
import runs
from tests.conftest import RUN_BODY, RUN_TITLE, SENTINEL, briefing_run, trace_text

pytestmark = pytest.mark.usefixtures("fresh_llm")


def _get(path: str, **headers):
    return TestClient(chatbot.app).get(path, headers=headers)


# --------------------------------------------------------------- 3. the page of a briefing run


def test_the_run_page_lists_pages_shas_stat_lines_and_what_was_left_out(vault, monkeypatch, prod_profile):
    got = briefing_run(monkeypatch)
    page = _get(f"/runs/{got['run_id']}")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    text = page.text
    assert "<h1>Briefing</h1>" in text and 'run-chip run-done">done<' in text and runs.REASONS["ok"] in text
    assert f'<code class="run-sha">{got["sha"]}</code>' in text
    assert "1 file changed" in text                                     # the commit's --stat summary line
    assert "notes/plan" in text and '<em class="run-origin">Wanko</em>' in text
    assert "2 housekeeping commits" in text and html.escape(runs.REASONS["housekeeping"]) in text
    for step in ("gather", "skip", "phrase"):
        assert f"<li>{step} · " in text                                 # the timeline, by start
    assert runs.FOOTER in text
    assert f"{html.escape(RUN_TITLE)} (notes/plan)" in text and "<b>" not in text      # every value escaped
    assert SENTINEL not in text and RUN_BODY not in text                 # no page body reaches the HTML
    assert SENTINEL not in trace_text()


def test_every_value_on_the_page_is_escaped():
    events = [
        {"kind": "run", "ts": "2026-10-16T10:00:00+02:00", "run_id": "x-1", "workflow": "ingest", "phase": "start"},
        {"kind": "ingest_end", "ts": "2026-10-16T10:00:02+02:00", "start": "2026-10-16T10:00:01+02:00",
         "run_id": "x-1", "span_id": "s1", "duration_ms": 900, "status": "ok", "title": RUN_TITLE,
         "path": "sources/x", "pages": ["sources/<script>", "sources/<i>x</i>"], "commit": "not-a-sha"},
        {"kind": "step", "ts": "2026-10-16T10:00:03+02:00", "start": "2026-10-16T10:00:03+02:00", "run_id": "x-1",
         "span_id": "s2", "duration_ms": 1, "status": "skipped", "name": "skip", "title": "<u>Note</u>",
         "detail": "page 2: unreadable", "reason": "unreadable"},
        {"kind": "error", "ts": "2026-10-16T10:00:04+02:00", "run_id": "x-1", "exc_type": "<Boom>", "message": "<Boom>"},
    ]
    text = runs.render(runs.build(events))
    for raw in (RUN_TITLE, "<script>", "<i>x</i>", "<u>Note</u>", "<Boom>"):
        assert raw not in text and html.escape(raw) in text
    assert "&lt;u&gt;Note&lt;/u&gt; — page 2: unreadable" in text


# --------------------------------------------------------------- 4. unknown, aged out, malformed


@pytest.mark.parametrize("path", ["/runs/nope-1", "/runs/a%20b", "/runs/" + "x" * 65])
def test_an_unknown_or_malformed_run_id_is_a_404_with_one_line(vault, path):
    page = _get(path)
    assert page.status_code == 404
    assert "(it may have aged out of the log)." in page.text and "Traceback" not in page.text


def test_the_404_names_the_id_escaped():
    assert "No run nope-1 (it may have aged out of the log)." in _get("/runs/nope-1").text
    assert "&lt;b&gt;" in runs.missing("<b>")


def test_the_run_page_has_the_ingest_origin_check(vault, monkeypatch):
    got = briefing_run(monkeypatch, commits_after=False)
    assert _get(f"/runs/{got['run_id']}", origin="http://evil.example").status_code == 403
    assert _get(f"/runs/{got['run_id']}").status_code == 200


@pytest.mark.parametrize("path", ["/runs/a%20b", "/runs/" + "x" * 65, "/runs/a%2Fb"])
def test_a_malformed_run_id_is_refused_before_the_log_is_read(monkeypatch, path):
    def never(*a, **k):
        raise AssertionError("runs.load called for a malformed id")

    monkeypatch.setattr(runs, "load", never)
    assert _get(path).status_code == 404


# --------------------------------------------------------------- PRD 54 scenario 11: a research run


def test_a_research_run_page_shows_the_chain_with_timings_and_no_question(vault, prod_profile):
    import json
    import re

    from tests.conftest import research_turn, trace_lines, write_page

    question = "Zephyr budget SENTINELQUESTION"
    quote = "The Zephyr budget is 40k for the first quarter SENTINELQUOTE"
    write_page("projects/zephyr", "Zephyr", f"\n## Summary\n{quote}.\n")
    write = (f'# Zephyr SENTINELTITLE\n\n## Findings\n- Q1 SENTINELCLAIM: "{quote}" — Zephyr\n'
             '- Made up SENTINELDROPCLAIM: "SENTINELFAKE words" — Zephyr\n')
    judge = json.dumps([{"id": 1, "verdict": "keep", "reason": "the page SENTINELREASON"}])
    research_turn(f"research {question}", json.dumps(["Zephyr budget SENTINELSUB", "Zephyr quarter"]), judge, write)
    (start,) = trace_lines(kind="turn_start")
    text = _get(f"/runs/{start['run_id']}").text
    assert "<h1>Research</h1>" in text and 'run-chip run-done">done<' in text
    chain = [name for name, _ms in re.findall(r"<li>(plan|search|judge|read|write|verify) · (\d+) ms · ", text)]
    assert chain == ["plan", "search", "search", "judge", "read", "write", "verify"]
    for purpose in ("research_plan", "research_judge", "research_write"):
        assert f"model · {purpose}" in text
    assert "Zephyr" in text                                              # the page read, under Did · Sources
    for sentinel in ("SENTINELQUESTION", "SENTINELSUB", "SENTINELQUOTE", "SENTINELTITLE", "SENTINELCLAIM",
                     "SENTINELDROPCLAIM", "SENTINELFAKE", "SENTINELREASON"):
        assert sentinel not in text and sentinel not in trace_text(), sentinel
