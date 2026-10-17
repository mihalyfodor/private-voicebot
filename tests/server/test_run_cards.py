"""PRD 51 scenario 8: the run card in the client (static/app.js via tests/app_fixture.mjs,
docs/51-run-cards.md). A card without run fields keeps PRD 43's shape: its own assertions are
tests/agent/test_written_answers.py::test_card_renders_collapsed_copies_and_saves."""


def test_a_run_card_renders_its_chip_steps_and_open_run_link(r):
    card = r["run_card"]
    assert card["added"] == 1 and card["workflow"] == "Briefing"
    assert card["chip"] == "running" and card["chip_cls"] == "run-chip run-running"
    assert card["steps"] == ["gathered✓since 14:05 — 3 pages, 2 sources, 4 memory changes",
                             "skipped–2 housekeeping commits", "phrase…"]
    assert card["open"] == {"tag": "A", "text": "Open run", "href": "/runs/9-1", "target": "_blank", "rel": "noopener"}
    # an empty body: header, steps and actions only — no body, no Save; `approve` renders nothing (PRD 52)
    assert card["body"] is False and card["save"] is False and card["buttons"] == 1


def test_card_update_swaps_chip_and_steps_in_the_same_element(r):
    updated = r["run_card_updated"]
    assert updated["same"] is True and updated["count"] == 1
    assert updated["chip"] == "done" and updated["chip_cls"] == "run-chip run-done"
    assert updated["steps"] == ["gathered✓since 14:05", "phrased✓2.1 s"]


def test_a_write_up_update_fills_the_body_and_keeps_copy_and_save(r):
    done = r["write_up_updated"]
    assert done["same"] is True and done["chip"] == "done" and done["workflow"] == "Write-up"
    assert done["body"] == "# Key\n\n## Where\nJacob." and done["copy"] and done["save"] and done["open"]
    failed = r["write_up_failed"]
    assert failed["chip"] == "failed" and failed["steps"] == ["write✗Failed with RuntimeError."]


def test_a_card_without_run_fields_renders_as_prd_43(r):
    assert r["plain_card_children"] == ["role", "card-body", "card-foot"]
    assert r["card_rendered"]["role"] == "Wanko · written"


def test_a_card_update_with_a_body_for_an_unknown_card_creates_it(r):
    assert r["update_unknown"] == {"added": 1, "id": "card99", "body": "# Late\n\n## A\nx"}
    assert r["update_unknown_no_body"] == 1                  # without a body there is nothing to show
