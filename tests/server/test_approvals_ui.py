"""PRD 52 scenario 7: the approval card in the client (static/app.js via tests/app_fixture.mjs,
docs/52-approvals.md) — Approve / Reject on an awaiting card, disabled until the answer, a replay
without live buttons, and the Sources view's Forget without `confirm()`."""
import re
from pathlib import Path


def test_an_awaiting_card_shows_the_proposal_with_approve_and_reject(r):
    card = r["approval_card"]
    assert card["chip"] == "awaiting" and card["chip_cls"] == "run-chip run-awaiting" and card["workflow"] == "Forget"
    assert card["body"].startswith("Forget the source") and card["labels"] == ["Approve", "Reject", "Open run"]
    assert card["copy"] is False and card["save"] is False          # a proposal is not a note


def test_approve_disables_both_buttons_until_the_card_update(r):
    assert r["approval_click"] == {"sent": [{"action": "resolve", "run_id": "9-card20", "decision": "approve"}],
                                   "disabled": [True, True]}
    assert r["approval_still_disabled"] == [True, True]
    done = r["approval_done"]
    assert done["chip"] == "done" and done["same"] is True
    assert (done["approve"], done["reject"], done["open"]) == (False, False, True)
    assert done["steps"][-1] == "applied✓commit abc1234"


def test_an_error_gives_the_buttons_back_and_reject_reads_rejected(r):
    err = r["approval_error"]
    assert err["sent"]["decision"] == "reject" and err["disabled"] == [False, False]
    assert err["status"] == "that request expired"
    rejected = r["approval_rejected"]
    assert rejected["chip_cls"] == "run-chip run-rejected" and not rejected["approve"] and not rejected["reject"]


def test_a_reconnect_replay_has_no_live_buttons(r):
    for key, chip in (("replay_resolved", "done"), ("replay_expired", "awaiting")):
        card = r[key]
        assert card["chip"] == chip and (card["approve"], card["reject"], card["open"]) == (False, False, True)


def test_the_sources_forget_button_asks_no_confirm_and_toasts_a_link_to_the_card(r):
    assert r["src_forget"] == {"confirms": 0, "sent": [{"action": "forget_source", "sha": "s2"}]}
    toast = r["src_forget_toast"]
    assert toast["shown"] and "Chat" in toast["text"] and toast["view"] == "sources"
    assert r["src_forget_toast_click"] == "chat"


def test_a_stale_click_turns_its_card_expired_without_buttons(r):
    stale = r["stale_click"]
    assert stale["chip"] == "expired" and stale["chip_cls"] == "run-chip run-expired"
    assert (stale["approve"], stale["reject"], stale["open"]) == (False, False, True)


def test_a_turn_error_disarms_the_forget_toast(r):
    assert r["toast_after_turn_error"] == ""


def test_recall_threshold_is_a_text_input_with_a_decimal_keyboard():
    html = (Path(__file__).resolve().parents[2] / "static" / "index.html").read_text()
    tag = re.search(r"<input[^>]*id=\"recall-min-score\"[^>]*>", html).group(0)
    assert 'type="text"' in tag and 'inputmode="decimal"' in tag
    horizon = re.search(r"<input[^>]*id=\"calendar-horizon\"[^>]*>", html).group(0)
    assert 'type="number"' in horizon
