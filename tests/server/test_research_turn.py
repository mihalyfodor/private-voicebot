"""PRD 54 scenario 10 — the research turn through the server (docs/54-vault-map-research.md): typed and
spoken "research X", then "save that as a note" and `/write` on its card."""
import os
import json

import pytest

from agent import research
from vault import pages
from vault import store
from server.session import Session
from tests.conftest import read, research_turn, session_turn, spoken_lines, write_page

pytestmark = pytest.mark.usefixtures("fresh_llm")

QUOTE = "The Zephyr budget is 40k for the first quarter"
PLAN = json.dumps(["Zephyr budget"])
JUDGE = json.dumps([{"id": 1, "verdict": "keep", "reason": "the Zephyr page"}])
WRITE = f'# Zephyr budget\n\n## Findings\n- Forty thousand for Q1: "{QUOTE}" — Zephyr\n'


@pytest.fixture
def zephyr(seeded_vault):
    write_page("projects/zephyr", "Zephyr", f"\n## Summary\n{QUOTE}, approved in October.\n")


def of(sent: list, kind: str) -> list:
    return [m for m in sent if m["type"] == kind]


def test_a_typed_research_turn_sends_the_card_and_never_speaks_its_body(zephyr):
    sent, fake = research_turn("/research Zephyr budget", PLAN, JUDGE, WRITE)
    (card,) = of(sent, "card")
    assert card["workflow"] == "research" and card["status"] == "running" and card["markdown"] == ""
    updates = of(sent, "card_update")
    assert len(updates) >= 5 and all(u["id"] == card["id"] for u in updates)        # one per finished step
    final = updates[-1]
    assert final["status"] == "done" and final["title"] == "Zephyr budget"
    assert f'"{QUOTE}" — Zephyr' in final["markdown"] and "## Sources\n- Zephyr" in final["markdown"]
    assert [s["label"] for s in final["steps"]] == ["Plan", "Zephyr budget", "Read", "Wrote", "Verified"]
    assert [m["text"] for m in of(sent, "speech")] == [research.LINE]               # the line, never the body
    assert len(fake.calls) == 3
    assert of(sent, "state")[-1]["value"] == "idle"


def test_a_spoken_research_turn_says_one_short_line_then_the_card(zephyr):
    sent, _ = research_turn("Research, the Zephyr budget.", PLAN, JUDGE, WRITE, origin="spoken")
    assert spoken_lines(sent) == [research.LINE]
    assert of(sent, "card") and of(sent, "card_update")[-1]["status"] == "done"
    states = [m["value"] for m in of(sent, "state")]
    assert states.index("speaking") < len(states) - 1 and "thinking" in states[states.index("speaking"):]


def test_save_that_as_a_note_files_findings_and_sources_with_the_run_id(zephyr):
    sess = Session()
    sent, _ = research_turn("research Zephyr budget", PLAN, JUDGE, WRITE, sess=sess)
    card = of(sent, "card")[0]
    saved = session_turn(sess, 2, "save that as a note")
    (done,) = of(saved, "card_saved")
    assert done["id"] == card["id"] and done["path"].startswith("notes/") and done["path"].endswith("-zephyr-budget")
    text = read(done["path"] + ".md")
    fm, body = store.parse_page(text)
    assert fm["run_id"] == card["run_id"] and fm["source"] == pages.CARD_SOURCE
    assert "## Findings" in body and "## Sources" in body and QUOTE in body
    assert "Plan" not in body and "searched" not in body                            # no step list


def test_write_it_up_after_research_reuses_the_research_card(zephyr):
    sess = Session()
    first, _ = research_turn("research Zephyr budget", PLAN, JUDGE, WRITE, sess=sess)
    card = of(first, "card")[0]
    again, fake = research_turn("/write", sess=sess, turn=2)
    assert fake.calls == [] and of(again, "card") == []                # no second card is minted
    (update,) = of(again, "card_update")
    assert update["id"] == card["id"] and update["markdown"] == of(first, "card_update")[-1]["markdown"]
    saved = session_turn(sess, 3, "save that as a note")
    (done,) = of(saved, "card_saved")
    assert done["id"] == card["id"]
    fm, _ = store.parse_page(read(done["path"] + ".md"))
    assert fm["run_id"] == card["run_id"] and fm["sources"] == ["Zephyr"]


def test_a_saved_research_note_leaves_the_dropped_quotes_out_of_the_vault(zephyr):
    sess = Session()
    write = WRITE + '- Cheap: "the budget is made of cheese" — Zephyr\n'
    sent, _ = research_turn("research Zephyr budget", PLAN, JUDGE, write, sess=sess)
    assert "### Dropped" in of(sent, "card_update")[-1]["markdown"]       # the card shows what was dropped
    saved = session_turn(sess, 2, "save that as a note")
    body = read(of(saved, "card_saved")[0]["path"] + ".md")
    assert "Dropped" not in body and "cheese" not in body and QUOTE in body and "## Sources" in body


def test_the_research_phrasing_is_in_the_protocol_doc():
    doc = open(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "protocol.md")).read()
    row = next(line for line in doc.splitlines() if line.startswith("| `text_turn` |"))
    assert "research X" in row and "/research X" in row and research.LINE in row
