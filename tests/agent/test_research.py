"""PRD 54 — research mode (docs/54-vault-map-research.md, scenarios 5–9): the fixed chain against
scripted plan / judge / write replies over the Fallout + Skyrim guides."""
import json

import pytest

from agent import research
from vault import chunks as v_chunks
from vault import chunks as v_chunks
from tests.conftest import judge_reply, research_run

pytestmark = pytest.mark.usefixtures("fresh_llm")

PLAN = json.dumps(["Covenant SAFE test", "Honest Dan caravan", "Bleak Falls Barrow pillars", "Falcon load test"])
SUBS = research.parse_plan(PLAN, "")


def hits() -> list:
    """The hits the scripted plan's search finds (deterministic: the same search the run does)."""
    return research.search(SUBS)


def chunk(h: dict) -> bool:
    return h["kind"] == "chunk"


def span_of(h: dict, words: int = 8) -> str:
    """Eight words copied from the text a read of hit `h` returns: a quote that verifies."""
    text = v_chunks.read_source(h["source"], h["n"]).split("\n", 1)[1]
    return " ".join(text.split()[20:20 + words])


def reads(monkeypatch) -> list:
    """Every `read_source` / `read_page` call the run makes, as its ref."""
    got = []
    real_source, real_page = v_chunks.read_source, v_chunks.read_page
    monkeypatch.setattr(v_chunks, "read_source", lambda s, n: (got.append((s, n)), real_source(s, n))[1])
    monkeypatch.setattr(v_chunks, "read_page", lambda p: (got.append(p), real_page(p))[1])
    return got


# --------------------------------------------------------------- 5. at most four sub-questions


def test_a_plan_of_six_sub_questions_keeps_four_and_the_card_lists_them(both):
    six = ["Covenant SAFE test", "Honest Dan caravan", "Bleak Falls Barrow", "Dragonsreach", "Diamond City", "Vault 81"]
    card, _ = research_run("compare the guides", json.dumps(six))
    assert card["research"]["subquestions"] == six[:4]
    labels = [s["label"] for s in card["steps"]]
    assert labels[0] == "Plan" and card["steps"][0]["detail"] == "4 sub-questions"
    assert labels[1:5] == six[:4]                                 # one row per sub-question, exactly those
    assert all(s["detail"].startswith("searched · ") for s in card["steps"][1:5])


def test_an_unparseable_plan_is_the_question_itself():
    assert research.parse_plan("I would search for things", "Zephyr budget?") == [{"q": "Zephyr budget?", "filters": None}]
    assert research.parse_plan('[{"q": "Zephyr", "type": "project"}, "zephyr", ""]', "x") == [
        {"q": "Zephyr", "filters": {"type": "project"}}]


# --------------------------------------------------------------- 6. the judge drops a hit


def test_a_dropped_hit_keeps_the_judges_reason_and_is_never_read(both, monkeypatch):
    found = hits()
    off = next(h for h in found if h["ref"] == "projects/falcon")        # "test" matched the Falcon load-test page
    read = reads(monkeypatch)
    judged = judge_reply(found, chunk, drop_reason="off topic: a load-test project")
    card, _ = research_run("what is the SAFE test", PLAN, judged, "# SAFE\n\n## Findings\n- nothing quotable")
    rec = {h["ref"]: h for h in card["research"]["hits"]}
    assert rec["projects/falcon"]["keep"] is False
    assert rec["projects/falcon"]["reason"] == "off topic: a load-test project"
    assert off["path"] not in read and "projects/falcon" not in read
    assert all(isinstance(r, tuple) for r in read)                      # only kept chunks were read


def test_an_unparseable_batch_is_judged_per_sub_question_and_the_rest_dropped(both):
    found = hits()
    first = [h for h in found if h["q"] == 0]
    per_q = json.dumps([{"id": 1, "verdict": "keep", "reason": "names Covenant"}])
    card, fake = research_run("q", PLAN, "not json", per_q, "[]", "[]", "# T\n\n## Findings\n- x")
    rec = card["research"]["hits"]
    assert [c["messages"][0]["content"] for c in fake.calls[1:5]] == [research.JUDGE_PROMPT] * 4
    assert rec[0]["ref"] == first[0]["ref"] and rec[0]["keep"] is True and rec[0]["reason"] == "names Covenant"
    assert all(not h["keep"] and h["reason"] == "not judged" for h in rec[1:])


# --------------------------------------------------------------- 7. the verifier


def test_a_fabricated_quote_is_dropped_and_listed_and_the_real_one_stays(both):
    found = hits()
    best = max((h for h in found if chunk(h)), key=lambda h: (h["score"], h["ref"]))
    real = span_of(best)
    write = (f"# The SAFE test\n\n## Findings\n- Covenant tests visitors: “{real}” — {best['ref']}\n"
             f"- The turrets are friendly: \"the turrets are made of cheese\" — {best['ref']}\n")
    card, _ = research_run("what is the SAFE test", PLAN, judge_reply(found, lambda h: h is best), write)
    findings = card["markdown"].split("## Findings")[1].split("## Sources")[0]
    assert real in findings and "made of cheese" not in findings.split("### Dropped")[0]
    assert "- The turrets are friendly" in findings                     # the claim stays, its quote goes
    assert card["dropped"] == [{"quote": "the turrets are made of cheese", "ref": best["ref"],
                                "why": f"not in {best['ref']}"}]
    sources, dropped = card["markdown"].split("## Sources\n")[1].split("### Dropped\n")
    assert f"- {best['ref']}" in sources
    assert dropped.strip() == f'- "the turrets are made of cheese" — not in {best["ref"]}'
    assert card["status"] == "done" and card["steps"][-1] == {"label": "Verified", "status": "ok", "detail": "1/2 quotes"}


def test_the_verifier_normalises_whitespace_and_quotes_and_trims_an_outer_ellipsis_only():
    text = "He said “no” to the   SAFE test,\nthen left for Diamond City at dawn."
    assert research.quoted_in('He said "no" to the SAFE test', text)
    assert research.quoted_in("…the SAFE test, then left…", text)
    assert research.quoted_in("...then left for Diamond City", text)
    assert not research.quoted_in("He said ... left for Diamond City", text)    # an inner ellipsis never verifies
    assert not research.quoted_in("he said no to the SAFE test", text)          # no case or fuzzy leeway
    assert not research.quoted_in("left for Diamond City at dusk", text)


P1, P2, P3 = ("The budget was approved in May.", "Later the project was cancelled by the board.",
              "The board discussed snacks at length.")
PARTS = {1: P1, 2: P2, 3: P3}


def read_part(n: int) -> dict:
    """What reading `Guide` part n gives the verifier: the part with its neighbours' text, as `read_source` returns it."""
    near = {k: PARTS[k] for k in (n - 1, n, n + 1) if k in PARTS}
    return {"label": f"Guide · part {n}/3", "text": "\n".join(near.values()), "kind": "chunk",
            "source": "sources/guide", "title": "Guide", "n": n, "total": 3, "parts": near}


def check(*bullets, read=(1, 2)) -> tuple:
    return research.verify("\n".join(bullets), [read_part(n) for n in read])


@pytest.mark.parametrize("quote", ["The budget … was cancelled by the board", "board ... snacks"])
def test_a_quote_with_an_inner_ellipsis_is_dropped(quote):
    findings, verified, total, dropped = check(f'- It was cancelled: "{quote}" — Guide · part 1/3')
    assert (verified, total) == (0, 1) and quote not in findings and "Guide · part" not in findings
    assert dropped == [{"quote": quote, "ref": "Guide · part 1/3", "why": "contains an ellipsis"}]
    assert findings == "- It was cancelled"


@pytest.mark.parametrize("bullet, quote", [
    ('- The figure: "5" — Guide · part 1/3', "5"),
    ('- The figure: "" — Guide · part 1/3', ""),
    ("- Approved: 'The budget was approved in May.' — Guide · part 1/3", ""),
    ("- Approved in May — Guide · part 1/3", ""),
])
def test_a_cited_bullet_without_one_checkable_quote_loses_its_citation(bullet, quote):
    findings, verified, total, dropped = check(bullet)
    assert (verified, total) == (0, 1)
    assert "Guide · part" not in findings and '"' not in findings
    assert dropped[0]["why"] == "no quote to check" and dropped[0]["ref"] == "Guide · part 1/3"
    assert "The budget was approved" not in findings                   # a single-quoted span goes too


def test_a_blockquote_is_dropped_whole():
    findings, verified, total, dropped = check("> The budget was approved in May. — Guide · part 1/3",
                                               '- Cancelled: "was cancelled by the board" — Guide · part 2/3')
    assert (verified, total) == (1, 2)
    assert findings == '- Cancelled: "was cancelled by the board" — Guide · part 2/3'
    assert dropped[0]["why"] == "no quote to check"


def test_two_quotes_in_one_bullet_are_dropped():
    findings, verified, total, dropped = check(
        '- Both: "The budget was approved" and "cancelled by the board" — Guide · part 1/3')
    assert (verified, total) == (0, 1) and dropped[0]["why"] == "more than one quote"


@pytest.mark.parametrize("bullet, claim", [
    ('- Both: "The budget was approved" and "cancelled by the board" — Guide · part 1/3', "- Both"),
    ('- The budget "was approved" or "was cut" in May — Guide · part 1/3', "- The budget or in May"),
    ('- A "x" and "y" but "z" — Guide · part 1/3', "- A"),
])
def test_a_dropped_bullets_claim_reads_cleanly(bullet, claim):
    findings, _, _, dropped = check(bullet)
    assert findings == claim and dropped[0]["why"] == "more than one quote"


@pytest.mark.parametrize("line", ["> The budget was approved in May.", "> a thought, no quote and no citation"])
def test_any_blockquote_is_a_quote_shape_and_never_renders_as_quoted_text(line):
    findings, verified, total, dropped = check(line, "- Plain claim.")
    assert findings == "- Plain claim." and (verified, total) == (0, 1)
    assert dropped[0]["why"] == "no quote to check" and ">" not in findings


def test_a_quote_re_cited_to_a_neighbour_part_puts_that_part_in_sources():
    material = [read_part(2)]
    findings, verified, _, _ = research.verify('- Snacks: "discussed snacks" — Guide · part 2/3', material)
    assert verified == 1 and findings.endswith("Guide · part 3/3")
    assert research._sources(material, findings) == ["Guide · part 2", "Guide · part 3"]
    assert research._sources(material, "- Plain claim.") == ["Guide · part 2"]


def test_a_quote_citing_a_part_that_was_not_read_is_dropped():
    findings, verified, total, dropped = check('- Snacks: "discussed snacks" — Guide · part 3/3')
    assert (verified, total) == (0, 1)                               # in part 2's neighbour text, but part 3 was cited
    assert dropped == [{"quote": "discussed snacks", "ref": "Guide · part 3/3", "why": "not in Guide · part 3/3"}]


def test_a_quote_found_in_the_cited_parts_neighbour_gets_the_neighbours_citation():
    findings, verified, total, dropped = check('- Snacks: "discussed snacks" — Guide · part 2/3')
    assert (verified, total, dropped) == (1, 1, [])
    assert findings == '- Snacks: "discussed snacks" — Guide · part 3/3'


def test_a_quote_that_is_only_in_another_part_read_is_not_in_the_cited_one():
    findings, verified, total, dropped = check('- May: "approved in May" — Guide · part 2/3', read=(2, 3))
    assert verified == 1 and findings.endswith("Guide · part 1/3")    # part 2's read held part 1's text
    findings, verified, total, dropped = check('- May: "approved in May" — Guide · part 3/3', read=(2, 3))
    assert verified == 0 and dropped[0]["why"] == "not in Guide · part 3/3"


def test_a_quote_without_a_citation_is_dropped_and_an_uncited_claim_stays():
    findings, verified, total, dropped = check('- Approved: "The budget was approved in May."', "- Plain claim.")
    assert (verified, total) == (0, 1) and dropped[0]["why"] == "no source cited"
    assert findings == "- Approved\n- Plain claim."


# --------------------------------------------------------------- 8. the budget governor


def test_the_clock_past_90_s_cuts_the_run_to_partial_with_what_was_read(both, monkeypatch):
    found = hits()
    kept = sorted((h for h in found if chunk(h)), key=lambda h: (-h["score"], h["ref"]))
    quote = span_of(kept[0])
    write = f'# T\n\n## Findings\n- first: "{quote}" — {kept[0]["ref"]}\n'
    read = reads(monkeypatch)
    monkeypatch.setattr(research, "clock", lambda: 100.0 if read else 0.0)      # 90 s gone once reading began
    card, _ = research_run("q", PLAN, judge_reply(found, chunk), write)
    assert card["status"] == "partial" and card["reason"] == "budget: time"
    assert read == [(kept[0]["source"], kept[0]["n"])]
    assert {"label": "Budget", "status": "skipped", "detail": "budget: time"} in card["steps"]
    assert f'"{quote}"' in card["markdown"].split("## Sources")[0]


def test_the_clock_past_90_s_right_after_search_leaves_nothing_judged(both, monkeypatch):
    """Out of time before the judge: no hit is judged, so none is kept or read — `nothing`, with the cut
    on the card (the governor skips to Write only with what was read)."""
    done = []
    real = research.search
    monkeypatch.setattr(research, "search", lambda subs: (done.append(1), real(subs))[1])
    monkeypatch.setattr(research, "clock", lambda: 100.0 if done else 0.0)
    card, fake = research_run("q", PLAN)
    assert len(fake.calls) == 1 and card["status"] == "nothing"
    assert all(h["reason"] == "not judged" for h in card["research"]["hits"])
    assert {"label": "Budget", "status": "skipped", "detail": "budget: time"} in card["steps"]


def test_more_kept_hits_than_reads_cuts_at_six_reads(both, monkeypatch):
    found = hits()
    assert len(found) > research.RESEARCH_MAX_READS
    read = reads(monkeypatch)
    best = max((h for h in found if chunk(h)), key=lambda h: (h["score"], h["ref"]))
    write = f'# T\n\n## Findings\n- "{span_of(best)}" — {best["ref"]}\n'
    read_before = len(read)                                             # span_of read one part itself
    card, _ = research_run("q", PLAN, judge_reply(found, lambda h: True), write)
    assert len(read) - read_before == research.RESEARCH_MAX_READS
    assert card["status"] == "partial" and card["reason"] == "budget: reads"


def test_no_hits_ends_nothing_without_a_write(both):
    card, fake = research_run("xylophone zebra", json.dumps(["xylophone zebra quartz"]))
    assert card["status"] == "nothing" and card["reason"] == "nothing"
    assert len(fake.calls) == 1                                          # the plan only: nothing to judge or write
    assert research.NOTHING_FOUND in card["markdown"]


# --------------------------------------------------------------- 9. call and read caps


def test_a_full_run_stays_within_eight_calls_and_six_reads(both, monkeypatch):
    found = hits()
    read = reads(monkeypatch)
    card, fake = research_run("q", PLAN, "garbage", "garbage", "garbage", "garbage", "garbage",
                              "# T\n\n## Findings\n- nothing")
    assert len(fake.calls) <= research.MAX_CALLS
    assert len([r for r in read if isinstance(r, tuple)]) <= research.RESEARCH_MAX_READS
    assert card["research"]["calls"] == len(fake.calls)
    card, fake = research_run("q", PLAN, judge_reply(found, lambda h: True), "# T\n\n## Findings\n- x")
    assert len(fake.calls) == 3 and len(read) <= 2 * research.RESEARCH_MAX_READS


# --------------------------------------------------------------- fix round 1


def test_a_failing_first_notify_still_ends_the_run(both):
    from tests.conftest import trace_lines

    def boom(card, new=False):
        raise RuntimeError("socket gone")
    with pytest.raises(RuntimeError):
        research_run("q", PLAN, notify=boom)
    phases = [(e["phase"], e.get("status")) for e in trace_lines(kind="run")]
    assert phases == [("start", None), ("end", "failed")]


def test_budget_calls_is_not_a_reason_the_governor_can_give():
    from agent import runs
    from core import trace
    assert "budget-calls" not in runs.REASONS and not trace.FIELDS["reason"].fullmatch("budget-calls")
