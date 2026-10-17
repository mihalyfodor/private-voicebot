"""PRD 36 — bulk intake progress, the client half: one request per file, per-row states, Cancel.

The scenario runs in tests/app_fixture.mjs (node) with a scripted `fetch`; this module asserts
on its JSON (the `r` fixture in tests/conftest.py). The server half (the `ingest_progress` pushes) is in test_sources.py.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def states(rows: list) -> list:
    return [row["state"] for row in rows]


def test_rows_carry_no_state_until_the_batch_starts(r):
    assert r["batch_before"] == ["", "", ""]


def test_the_batch_posts_one_file_at_a_time_and_shows_the_queue(r):
    started = r["batch_started"]
    assert states(started["rows"]) == ["uploading…", "queued", "queued"]
    assert started["posts"] == 1 and started["first_post_files"] == ["one.md"]
    assert started["status"].startswith("1 of 3 · 0:0")


def test_a_progress_push_names_the_stage_of_the_row_in_flight(r):
    assert r["batch_summarising"] == {"cls": "intake-item st-summarising", "state": "summarising…"}


def test_a_filed_row_links_the_page_title_and_the_next_file_starts(r):
    one = r["batch_one_filed"]
    assert states(one["rows"]) == ["filed → Falcon status", "uploading…", "queued"]
    assert one["posts"] == 2 and one["link"] is True
    assert one["status"].startswith("2 of 3 ·")


def test_cancel_stops_after_the_current_file(r):
    assert r["batch_cancel_status"].endswith("stopping after this file")
    after = r["batch_after_cancel"]
    assert states(after["rows"]) == ["filed → Falcon status", "already filed", "queued"]
    assert after["posts"] == 2                       # the third file was never sent
    assert after["status"].startswith("2 of 3 — 1 filed, 1 already filed ·")
    assert after["button"] == "Close" and after["open"] is True
    assert r["batch_closed"] is True


def test_a_failed_request_marks_its_row_and_the_batch_goes_on(r):
    err = r["batch_error"]
    assert states(err["rows"]) == ["error — 413 — the text is larger than 50 MB", "filed → sources/good"]
    assert err["status"].startswith("1 filed, 1 error ·") and err["open"] is True


def test_a_degraded_filing_says_why_and_keeps_the_popup_open(r):
    summary, ops = r["degraded_rows"]
    assert states(summary["rows"]) == ["filed raw only → sources/summary"] and summary["open"] is True
    assert states(ops["rows"]) == ["filed without page updates → sources/ops"] and ops["open"] is True


def test_the_popup_says_that_re_dropping_is_a_no_op():
    html = open(os.path.join(ROOT, "static", "index.html")).read()
    assert 'id="intake-note"' in html and "already filed is a no-op" in html
