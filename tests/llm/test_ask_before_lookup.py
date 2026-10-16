"""PRD 45 — ask before lookup: the recall band (high / medium / low) on spoken turns, the
server-appended "Want me to check your <title>?" offer, "yes" / "no" to it, never twice in a row,
typed turns unchanged, the band in the trace's `turn_end` (PRD 49). Scenario numbers are the PRD's (10 is the
battery in tests/qa/stress.py plus the manual spoken pass).

Fixtures: `fallout` (the generated FAQ, titled "Fallout 4 guide") and the streamed fakes from
tests/conftest.py; scenario 1 (`live`) runs over a copy of ~/llm-wiki-dev.
"""
import os
import shutil

import pytest

from core import settings
import llm
import wiki
from tests.conftest import ask_turn, deltas, fallout, last_user, trace_lines  # noqa: F401  (fixtures)

pytestmark = pytest.mark.usefixtures("fresh_llm")

MISHEARD = "Are you there, Dogmeat?"                  # four words, one proper term, no question
QUESTION = "What about Covenant turrets?"            # four words, a content question, proper hit
OFFER = "Want me to check your Fallout 4 guide?"
CHAT_REPLY = "[happy] Always, I'm right here."
GROUNDED_REPLY = "[thinking] Covenant has turrets at the gate, says the Fallout 4 guide. " + llm.DEEPER_OFFERS[0]


# --------------------------------------------------------------- 1. the band table (live)

DEV_VAULT = os.path.expanduser("~/llm-wiki-dev")
BAND_TABLE = [
    ("Are you listening, Uncle?", "medium"),
    ("Who is Vivec?", "high"),
    ("Vivec is annoying", "medium"),
    ("What does Caius want?", "high"),
    ("Where is Caius Cosades?", "high"),
    ("How long should I steep green tea?", "low"),
]


def _battery_questions() -> list:
    """Every A / C step of tests/qa/stress.py that is a question (not "yes", "/new")."""
    import re
    text = open(os.path.join(os.path.dirname(__file__), "..", "qa", "stress.py")).read()
    steps = re.findall(r'd\.ask\("([AC]\d+)",\s*"((?:[^"\\]|\\.)*)"', text)
    return [(sid, q) for sid, q in steps if q not in ("yes", "/new")]


@pytest.mark.live
def test_the_band_table_over_the_dev_vault(monkeypatch, tmp_path):
    if not os.path.isdir(os.path.join(DEV_VAULT, "raw")):
        pytest.skip("no ~/llm-wiki-dev")
    vault = tmp_path / "dev"
    shutil.copytree(DEV_VAULT, vault, ignore=shutil.ignore_patterns(".git"))
    monkeypatch.setattr(wiki, "WIKI_PATH", str(vault))
    rows, wrong = [], []
    battery = [(q, "high") for sid, q in _battery_questions() if llm.recall_for(q)]
    for question, want in BAND_TABLE + battery:
        page, hits = llm._recall_evidence(question)
        band = llm.recall_band(question, page, hits)
        proper = wiki.proper_terms(wiki.content_terms(question))
        top = hits[0] if hits else {"score": 0.0, "matched": []}
        rows.append(f"{band:6} {top['score']:6.2f} matched={top['matched']} proper={sorted(proper)} "
                    f"| {question}")
        if band != want:
            wrong.append((question, band, want))
    print("\n" + "\n".join(rows))
    assert len(battery) >= 10, battery                         # the battery still recalls
    assert not wrong, wrong


# --------------------------------------------------------------- the band rules (fixture)


def test_band_rules_on_the_fixture(fallout):
    def band(text):
        return llm.recall_band(text, *llm._recall_evidence(text))
    assert band(MISHEARD) == "medium"
    assert band(QUESTION) == "high"
    assert band("Who is Covenant?") == "high"                 # a name question answers directly
    assert band("hi") == "low"
    assert llm.RECALL_HIGH_SCORE == wiki.RECALL_HIGH_SCORE


# --------------------------------------------------------------- 2. spoken mis-transcription


def test_a_spoken_misheard_turn_gets_no_block_and_one_offer(fallout):
    assert llm.recall_for(MISHEARD)                            # today it would quote a chunk
    llm.reset()
    events, fake = ask_turn(MISHEARD, CHAT_REPLY)
    reply = deltas(events)
    assert len(fake.calls) == 1                                # one completion, no fallback retry
    user = last_user(fake)
    assert "<recall>" not in user and "<lookup>" not in user
    assert llm.RECALL_ASK_NOTE in user
    assert reply.endswith(OFFER) and reply.startswith(CHAT_REPLY)
    assert llm._deeper_offer == {"recall": MISHEARD, "title": "Fallout 4 guide"}
    assert llm._conversation[-1]["content"] == reply


def test_the_offer_title_stops_at_a_colon_or_a_bracket():
    assert llm.offer_title("Morrowind Guide: Game of the Year (2002)") == "Morrowind Guide"
    assert llm.offer_title("Fallout 4 guide (complete)") == "Fallout 4 guide"


def test_a_model_ended_offer_on_a_medium_turn_is_replaced_by_the_recall_offer(fallout):
    events, _ = ask_turn(MISHEARD, "[happy] Always. " + llm.DEEPER_OFFERS[1])
    reply = deltas(events)
    assert reply == "[happy] Always. " + OFFER
    assert llm.DEEPER_OFFERS[1] not in deltas(events)   # never spoken
    assert llm._deeper_offer == {"recall": MISHEARD, "title": "Fallout 4 guide"}


# --------------------------------------------------------------- 3. spoken content question


def test_a_content_question_with_a_proper_match_is_high_whatever_the_score(fallout, monkeypatch):
    text = "What does Dogmeat want?"                       # "Caius" in the dev vault
    page, hits = llm._recall_evidence(text)
    assert wiki.proper_match(hits[0], wiki.proper_terms(wiki.content_terms(text)), wiki.proper_phrases(text))
    monkeypatch.setattr(llm, "RECALL_HIGH_SCORE", 999.0)
    assert llm.recall_band(text, page, hits) == "high"


PLAIN_QUESTION = "how do I cook a meal by the lake"     # a content question on filler words only


def test_a_content_question_on_plain_words_below_the_high_score_is_medium(fallout, file_sandbox, monkeypatch):
    settings.save_setting("recall_min_score", 0.0)
    page, hits = llm._recall_evidence(PLAIN_QUESTION)
    assert hits and not page and not hits[0]["named"] and llm.content_question(PLAIN_QUESTION)
    assert not wiki.proper_terms(wiki.content_terms(PLAIN_QUESTION))
    assert hits[0]["score"] < llm.RECALL_HIGH_SCORE
    assert llm.recall_band(PLAIN_QUESTION, page, hits) == "medium"
    events, fake = ask_turn(PLAIN_QUESTION, "[neutral] Fry some fish.")
    reply = deltas(events)
    assert "<recall>" not in last_user(fake) and reply == "[neutral] Fry some fish. " + OFFER
    monkeypatch.setattr(llm, "RECALL_HIGH_SCORE", 0.0)       # at or above the score it answers grounded
    assert llm.recall_band(PLAIN_QUESTION, page, hits) == "high"


def test_a_spoken_content_question_with_strong_evidence_answers_grounded(fallout):
    events, fake = ask_turn(QUESTION, GROUNDED_REPLY)
    reply = deltas(events)
    user = last_user(fake)
    assert "<recall>" in user and llm.RECALL_ASK_NOTE not in user
    assert reply == GROUNDED_REPLY and not reply.endswith(OFFER)
    assert llm._deeper_offer["source"] == fallout["path"] and llm._deeper_offer["parts"]


# --------------------------------------------------------------- 4. typed is unchanged


@pytest.mark.parametrize("text", [MISHEARD, QUESTION])
def test_typed_turns_get_todays_recall_message(fallout, text):
    expected = text + llm.recall_for(text)
    llm.reset()
    events, fake = ask_turn(text, CHAT_REPLY, origin="typed")
    reply = deltas(events)
    user = last_user(fake)
    if llm.content_question(text):
        expected = llm.written_note(expected)                  # PRD 43: typed content runs long
        assert fake.calls[0]["max_tokens"] == llm.VERBOSITY_MAX_TOKENS["long"]
    assert user == expected and "<recall>" in user
    assert reply == CHAT_REPLY and not (llm._deeper_offer or {}).get("recall")


# --------------------------------------------------------------- 5. "yes" to the offer


def test_yes_after_the_offer_answers_the_earlier_question_grounded(fallout):
    ask_turn(MISHEARD, CHAT_REPLY)
    events, fake = ask_turn("yes", GROUNDED_REPLY)
    reply = deltas(events)
    user = last_user(fake)
    assert user.startswith("yes\n\n<recall>") and 'From "Fallout 4 guide"' in user
    assert llm.RECALL_YES_NOTE.format(question=MISHEARD) in user
    assert llm.NOTE_OFFER in user and fake.calls[0]["max_tokens"] == 400   # short rule, deeper offer
    assert ("tool_calls", ["dig_deeper"]) in events
    assert llm.ends_with_offer(reply)
    assert llm._last_context["question"] == MISHEARD
    assert llm._deeper_offer["source"] == fallout["path"]


# --------------------------------------------------------------- 6. "no" / anything else


def test_no_after_the_offer_drops_it_silently(fallout):
    ask_turn(MISHEARD, CHAT_REPLY)
    events, fake = ask_turn("no", "[neutral] Okay.")
    reply = deltas(events)
    assert reply == "[neutral] Okay."
    assert last_user(fake) == "no"
    assert llm._deeper_offer is None


def test_no_and_a_new_question_runs_that_turn_normally(fallout):
    ask_turn(MISHEARD, CHAT_REPLY)
    events, fake = ask_turn("no, what's the weather", "[happy] Sunny.")
    reply = deltas(events)
    assert last_user(fake) == "no, what's the weather"
    assert reply == "[happy] Sunny." and llm._deeper_offer is None


# --------------------------------------------------------------- 7. never twice in a row


def test_two_weak_turns_offer_once(fallout):
    events, _ = ask_turn(MISHEARD, CHAT_REPLY)
    first = deltas(events)
    second_text = "Covenant is so annoying"
    assert llm.recall_band(second_text, *llm._recall_evidence(second_text)) == "medium"
    events, fake = ask_turn(second_text, "[sad] It can be.")
    second = deltas(events)
    assert first.endswith(OFFER)
    user = last_user(fake)
    assert "<recall>" not in user and llm.RECALL_ASK_NOTE not in user
    assert second == "[sad] It can be." and llm._deeper_offer is None


# --------------------------------------------------------------- 8. the gate


def test_a_chat_turn_carried_by_two_plain_words_is_low(fallout, file_sandbox):
    settings.save_setting("recall_min_score", 0.0)
    text = "remind me to cook a meal"
    page, hits = llm._recall_evidence(text)
    assert hits and not llm.content_question(text)            # kept hits, plain words only
    assert not wiki.proper_terms(wiki.content_terms(text))
    assert llm.recall_band(text, page, hits) == "low"
    events, fake = ask_turn(text, "[happy] Will do.")
    reply = deltas(events)
    assert last_user(fake) == text
    assert reply == "[happy] Will do." and llm._deeper_offer is None


# --------------------------------------------------------------- 9. the band in the trace (PRD 49)


def test_spoken_turns_carry_the_band_typed_ones_do_not(fallout):
    ask_turn(MISHEARD, CHAT_REPLY)
    ask_turn("no", "[neutral] Okay.")
    ask_turn(QUESTION, GROUNDED_REPLY)
    ask_turn(QUESTION, GROUNDED_REPLY, origin="typed")
    bands = [r["band"] for r in trace_lines(kind="recall")]
    assert bands == ["medium", "low", "high", None]
