"""PRD 37 — Wikipedia tool and knowledge routing. Scenario numbers are the PRD's.

The FAQ fixtures come from tests/vault/test_source_recall.py, the vault and the fakes from conftest; the 28-question
calibration set lives in tests/fixtures/routing_questions.json.
"""
import os

import json
import pathlib
import re
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from core import avatars
from agent import conversation
from agent import offers
from agent import prompt as a_prompt
from agent import recall as a_recall
from agent import turn as a_turn
from agent import turn as a_turn
from agent import prompt as a_prompt
from agent import recall as a_recall
from tools import registry, run_tool
from tools import wikipedia
from tests.conftest import FakeClient, seeded_vault, stream_chunks, tool_call_chunk  # noqa: F401 — fixture
from tests.conftest import fallout, both  # noqa: F401  (fixtures)
from core import llm as core_llm

ROOT = pathlib.Path(__file__).resolve().parents[2]
QUESTIONS = json.loads((ROOT / "tests" / "fixtures" / "routing_questions.json").read_text())
AVATAR = {"name": "Test", "persona": "You are Test.", "verbosity": None}
PRECISE = "what year exactly did Ada Lovelace publish her notes on the Analytical Engine"
OFFER_REPLY = "[thinking] I believe it was 1843. " + a_prompt.WIKIPEDIA_OFFER

ROUTES = ("direct", "local", "wikipedia-offer", "wikipedia-call", "none")
#: what each expected label accepts as observed route
ACCEPTS = {"direct": {"direct"}, "local": {"local"}, "wikipedia": {"wikipedia-offer", "wikipedia-call"},
           "personal": {"local", "none", "direct"}}
_NOTHING = re.compile(r"don't have|do not have|don't know|do not know|no record|nothing (on|about|filed)|"
                      r"don't recall|haven't|not sure|can't find|cannot find|no notes?", re.I)


@pytest.fixture(autouse=True)
def fresh(fresh_llm):
    """`fresh_llm` plus an empty Wikipedia cache and throttle clock."""
    wikipedia._cache.clear()
    wikipedia._last_request = 0.0
    yield
    wikipedia._cache.clear()


# --------------------------------------------------------------- canned Wikipedia


SEARCH = {"query": {"search": [{"title": "Ada Lovelace"}, {"title": "Analytical Engine"}, {"title": "Charles Babbage"}]}}
SUMMARY = {"title": "Ada Lovelace", "extract": "Augusta Ada King, Countess of Lovelace (1815–1852) was an English "
           "mathematician and writer. " + "x" * 1500,
           "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Ada_Lovelace"}}}
SECTIONS = {"parse": {"sections": [{"line": "Early life", "index": "1"}, {"line": "Notes", "index": "2"}]}}
WIKITEXT = {"parse": {"wikitext": {"*": "In 1843 she published the [[Analytical Engine|notes]] {{cite web}} "
                                        "with '''Note G'''.<ref>ref</ref>"}}}


class Canned:
    """A fake `requests.get`: answers by URL/params, records every call, can fail."""

    def __init__(self, fail: Exception | None = None, limited: int = 0):
        self.calls, self.fail, self.limited = [], fail, limited

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}, "headers": headers, "timeout": timeout})
        if self.fail:
            raise self.fail
        if len(self.calls) <= self.limited:                    # the first N answers are 429s
            return SimpleNamespace(status_code=429, raise_for_status=lambda: (_ for _ in ()).throw(
                requests.HTTPError("429")), json=dict)
        if url.startswith(wikipedia.SUMMARY_URL):
            data = SUMMARY
        elif params.get("list") == "search":
            data = SEARCH
        elif params.get("prop") == "sections":
            data = SECTIONS
        else:
            data = WIKITEXT
        return SimpleNamespace(status_code=200, json=lambda: data, raise_for_status=lambda: None)


@pytest.fixture
def canned(monkeypatch):
    fake = Canned()
    monkeypatch.setattr(wikipedia.requests, "get", fake)
    monkeypatch.setattr(wikipedia, "MIN_INTERVAL", 0.0)
    return fake


# --------------------------------------------------------------- 1. the tool


def test_search_then_summary_from_canned_responses(canned):
    out = run_tool("get_wikipedia", {"query": "Ada Lovelace notes"})
    assert out.startswith('<document source="Wikipedia: Ada Lovelace">\nWikipedia article "Ada Lovelace" '
                          '(https://en.wikipedia.org/wiki/Ada_Lovelace): Augusta')
    assert "Other matches: Analytical Engine, Charles Babbage." in out and "per Wikipedia" in out
    assert len(out.split("): ", 1)[1].split("\n")[0]) <= wikipedia.SUMMARY_CHARS
    assert [c["params"].get("list") for c in canned.calls] == ["search", None]
    assert canned.calls[0]["params"]["srlimit"] == 3
    assert canned.calls[1]["url"] == wikipedia.SUMMARY_URL + "Ada_Lovelace"
    assert all(c["timeout"] == 5 for c in canned.calls)


def test_user_agent_names_the_app_and_the_contact_when_set(canned, monkeypatch):
    monkeypatch.delenv("WIKIPEDIA_CONTACT", raising=False)
    run_tool("get_wikipedia", {"query": "Ada"})
    assert canned.calls[0]["headers"]["User-Agent"] == "voicebot/1.0 (personal assistant)"
    monkeypatch.setenv("WIKIPEDIA_CONTACT", "someone@example.org")
    wikipedia._cache.clear()
    run_tool("get_wikipedia", {"query": "Ada"})
    assert canned.calls[-1]["headers"]["User-Agent"] == "voicebot/1.0 (personal assistant; someone@example.org)"


def test_a_network_error_is_a_clear_string_never_an_exception(monkeypatch):
    monkeypatch.setattr(wikipedia.requests, "get", Canned(requests.ConnectionError("boom")))
    monkeypatch.setattr(wikipedia, "MIN_INTERVAL", 0.0)
    assert run_tool("get_wikipedia", {"query": "Ada"}) == "Wikipedia lookup failed: boom"
    assert run_tool("read_wikipedia", {"title": "Ada Lovelace", "section": "Notes"}) == "Wikipedia lookup failed: boom"
    assert run_tool("get_wikipedia", {}) == "Wikipedia lookup failed: no query given."


def test_no_hit_says_so(monkeypatch):
    fake = Canned()
    monkeypatch.setattr(wikipedia.requests, "get", fake)
    monkeypatch.setattr(wikipedia, "MIN_INTERVAL", 0.0)
    with patch.dict(SEARCH, {"query": {"search": []}}):
        assert run_tool("get_wikipedia", {"query": "zzqx"}) == "No Wikipedia article found for 'zzqx'."


def test_a_cache_hit_makes_no_second_request(canned):
    first = run_tool("get_wikipedia", {"query": "Ada Lovelace"})
    assert len(canned.calls) == 2
    assert run_tool("get_wikipedia", {"query": "ada lovelace"}) == first
    assert len(canned.calls) == 2
    run_tool("read_wikipedia", {"title": "Ada Lovelace", "section": "notes"})
    n = len(canned.calls)
    run_tool("read_wikipedia", {"title": "Ada Lovelace", "section": "Notes"})
    assert len(canned.calls) == n


def test_requests_are_throttled_to_one_per_second(canned, monkeypatch):
    monkeypatch.setattr(wikipedia, "MIN_INTERVAL", 1.0)
    slept = []
    monkeypatch.setattr(wikipedia.time, "sleep", slept.append)
    run_tool("get_wikipedia", {"query": "Ada Lovelace"})
    assert len(slept) == 1 and 0.9 < slept[0] <= 1.0      # the second request waited on the first


def test_a_rate_limit_is_retried_once_then_reported(monkeypatch):
    slept = []
    monkeypatch.setattr(wikipedia.time, "sleep", slept.append)
    monkeypatch.setattr(wikipedia, "MIN_INTERVAL", 0.0)
    fake = Canned(limited=1)
    monkeypatch.setattr(wikipedia.requests, "get", fake)
    assert 'Wikipedia article "Ada Lovelace"' in run_tool("get_wikipedia", {"query": "Ada"})
    assert len(fake.calls) == 3 and slept == [wikipedia.RETRY_AFTER]
    wikipedia._cache.clear()
    monkeypatch.setattr(wikipedia.requests, "get", Canned(limited=2))
    assert run_tool("get_wikipedia", {"query": "Ada"}) == "Wikipedia lookup failed: 429"


def test_read_wikipedia_returns_the_named_section_only(canned):
    out = run_tool("read_wikipedia", {"title": "Ada Lovelace", "section": "Notes"})
    assert out.startswith('<document source="Wikipedia: Ada Lovelace">\nWikipedia article "Ada Lovelace", '
                          'section "Notes": In 1843 she published the notes  with Note G.')
    assert "[[" not in out and "{{" not in out and "<ref" not in out
    assert canned.calls[-1]["params"]["section"] == "2"
    missing = run_tool("read_wikipedia", {"title": "Ada Lovelace", "section": "Legacy"})
    assert missing == 'No section \'Legacy\' in "Ada Lovelace". Sections: Early life, Notes.'


def test_the_section_is_capped(canned):
    with patch.dict(WIKITEXT["parse"]["wikitext"], {"*": "y" * 5000}):
        out = run_tool("read_wikipedia", {"title": "Ada Lovelace", "section": "Notes"})
    assert out.split("\n")[1].count("y") == wikipedia.SECTION_CHARS


# --------------------------------------------------------------- 2. the prompt rules


def test_the_prompt_carries_the_four_routing_rules_with_tools_registered():
    prompt = a_prompt.build_system_prompt(AVATAR)
    for phrase in ("local first, never Wikipedia", "answer directly from your own knowledge",
                   a_prompt.WIKIPEDIA_OFFER, "never offer or call Wikipedia for anything personal"):
        assert phrase in prompt
    assert prompt.count(a_prompt.WIKIPEDIA_OFFER) == 1


# (the tool-free prompt and the fillers are covered in tests/agent/test_grounded_answers.py)


# --------------------------------------------------------------- 3. offer / yes


def _turn(text: str, *replies):
    fake = FakeClient(*replies)
    with patch.object(core_llm, "_client", fake):
        events = list(a_turn.ask_events(text))
    return events, fake


def test_a_reply_ending_with_the_offer_remembers_the_question():
    _turn(PRECISE, stream_chunks(OFFER_REPLY))
    assert conversation._deeper_offer == {"wikipedia": PRECISE}


TOPIC_CALL = [tool_call_chunk(id="c1", name="get_wikipedia", arguments=json.dumps({"query": "Ada Lovelace"}),
                        finish_reason="tool_calls")]


def test_yes_calls_the_tool_with_the_topic_at_long_and_clears_the_offer():
    _turn(PRECISE, stream_chunks(OFFER_REPLY))
    with patch.object(a_turn, "run_tool", return_value="Wikipedia article ...") as rt:
        events, fake = _turn("yes", TOPIC_CALL, stream_chunks("[happy] Per Wikipedia, 1843."))
    rt.assert_called_once_with("get_wikipedia", {"query": "Ada Lovelace"})
    assert events == [("tool_calls", ["get_wikipedia"]), ("delta", "[happy] Per Wikipedia, 1843.")]
    assert [tc["name"] for tc in conversation.get_last_tool_calls()] == ["get_wikipedia"]
    for sent in fake.calls:
        assert sent["max_tokens"] == a_prompt.VERBOSITY_MAX_TOKENS["long"]
        assert a_prompt.VERBOSITY_RULES["long"] in sent["messages"][0]["content"]
    assert a_prompt.VERBOSITY_RULES["long"] not in conversation._conversation[0]["content"]
    assert a_prompt.current_verbosity() == "normal"
    user = [m for m in fake.calls[0]["messages"] if m["role"] == "user"][-1]["content"]
    assert user.startswith("yes\n\n" + offers.WIKIPEDIA_NOTE)
    assert [m["role"] for m in fake.calls[1]["messages"][-3:]] == ["user", "assistant", "tool"]
    assert conversation._session_turns[-2] == {"role": "user", "content": "yes"}
    assert conversation._deeper_offer is None


def test_yes_without_a_tool_call_from_the_model_looks_up_the_question_itself():
    """The model answered in text instead of naming the topic: that text is never spoken, the tool
    runs with the question and the answer comes from the second pass."""
    _turn(PRECISE, stream_chunks(OFFER_REPLY))
    with patch.object(a_turn, "run_tool", return_value="Wikipedia article ...") as rt:
        events, _ = _turn("yes please", stream_chunks("[neutral] It was 1843."), stream_chunks("[happy] Per Wikipedia, 1843."))
    rt.assert_called_once_with("get_wikipedia", {"query": PRECISE})
    assert events == [("tool_calls", ["get_wikipedia"]), ("delta", "[happy] Per Wikipedia, 1843.")]
    assert conversation._conversation[-1]["content"] == "[happy] Per Wikipedia, 1843."
    assert conversation._deeper_offer is None


def test_no_clears_the_offer_without_a_call():
    _turn(PRECISE, stream_chunks(OFFER_REPLY))
    with patch.object(a_turn, "run_tool") as rt:
        _, fake = _turn("no thanks", stream_chunks("[neutral] Alright."))
    rt.assert_not_called()
    assert fake.calls[0]["max_tokens"] == 400 and offers.WIKIPEDIA_NOTE not in str(fake.calls[0]["messages"])
    assert conversation._deeper_offer is None


def test_a_wikipedia_offer_wins_over_a_source_offer_and_needs_no_source(fallout):
    _turn("I'm at Covenant and Honest Dan keeps attacking me, what now", stream_chunks(OFFER_REPLY))
    assert conversation._deeper_offer == {"wikipedia": "I'm at Covenant and Honest Dan keeps attacking me, what now"}


# --------------------------------------------------------------- 4/5. calibration


def observed_route(reply: str, tool_calls: list, recalled: bool) -> str:
    """Which way a turn went, from its observable shape."""
    names = [tc["name"] for tc in tool_calls]
    if "get_wikipedia" in names or "read_wikipedia" in names:
        return "wikipedia-call"
    if offers.ends_with_wikipedia_offer(reply):
        return "wikipedia-offer"
    if any(n in names for n in ("search_wiki", "read_source", "read_page")):
        return "local"
    if recalled and offers.ends_with_offer(reply):                 # the recall was taken, not ignored
        return "local"
    return "none" if _NOTHING.search(reply) else "direct"


def confusion(rows: list) -> str:
    """rows: (expect, observed, ok). A markdown table expected × observed plus the score."""
    counts = Counter((e, o) for e, o, _ in rows)
    lines = ["| expected \\ observed | " + " | ".join(ROUTES) + " |", "|---|" + "---|" * len(ROUTES)]
    for e in ("direct", "local", "wikipedia", "personal"):
        lines.append(f"| {e} | " + " | ".join(str(counts[(e, o)]) for o in ROUTES) + " |")
    lines.append(f"\n{sum(ok for *_, ok in rows)}/{len(rows)} routed as expected")
    return "\n".join(lines)


def test_calibration_mocked(both):
    """Each question with a canned reply of the shape its rule asks for: filed-document questions
    get a recall block (local) and nothing else does; the offer, the tool call and the direct answer
    are detected from the output shape. ≥ 24/28."""
    assert len(QUESTIONS) == 28                       # the thresholds below assume the full set
    rows = []
    for i, q in enumerate(QUESTIONS):
        conversation.reset()
        expect, question = q["expect"], q["question"]
        if expect == "wikipedia" and i % 2:
            replies = [[tool_call_chunk(id="c", name="get_wikipedia", arguments=json.dumps({"query": question}),
                                  finish_reason="tool_calls")], stream_chunks("[neutral] Per Wikipedia, 1843.")]
        elif expect == "wikipedia":
            replies = [stream_chunks(OFFER_REPLY)]
        elif expect == "personal":
            # "I don't have anything" may trigger the PRD 42 lookup re-ask against the FAQ filler
            replies = [stream_chunks("[sad] I don't have anything on that.")] * 2
        elif expect == "local":
            replies = [stream_chunks("[happy] Talk to Jacob, says the guide. " + a_prompt.DEEPER_OFFERS[0])]
        else:
            replies = [stream_chunks("[happy] Here is the answer.")]
        fake = FakeClient(*replies)
        with patch.object(core_llm, "_client", fake), patch.object(a_turn, "run_tool", return_value="Wikipedia article ..."):
            reply = a_turn.ask(question)
        recalled = "<recall>" in fake.calls[0]["messages"][1]["content"]
        observed = observed_route(reply, conversation.get_last_tool_calls(), recalled)
        rows.append((expect, observed, observed in ACCEPTS[expect]))
    table = confusion(rows)
    assert sum(ok for *_, ok in rows) >= 24, table
    assert all(ok for e, _, ok in rows if e == "local"), table


# --------------------------------------------------------------- 6. policy


def test_the_tool_talks_to_fixed_hosts_only_and_takes_no_url():
    assert wikipedia.HOSTS == ("en.wikipedia.org",)
    for url in (wikipedia.API_URL, wikipedia.SUMMARY_URL):
        assert url.startswith("https://en.wikipedia.org/")
    for name in ("get_wikipedia", "read_wikipedia"):
        params = next(t for t in registry.schemas() if t["function"]["name"] == name)["function"]["parameters"]["properties"]
        assert not any("url" in p.lower() for p in params), name
    with pytest.raises(requests.RequestException):
        wikipedia._get("https://example.org/x")


# --------------------------------------------------------------- 5. live


REAL_GUIDES = (
    ("2026-08-30-note.md", "Fallout 4 FAQ and Walkthrough"),
    ("2026-08-30-note-2.md", "Elder Scrolls V FAQ and Walkthrough"),
)


def _file_guides():
    """The two real guides from ~/llm-wiki-dev when present (read-only copy into the temp vault),
    else the generated FAQs."""
    from tests.conftest import FakeClient as SummaryClient
    from tests.conftest import summary_json, file_faq, faq_text, FALLOUT_SECTIONS, SKYRIM_SECTIONS
    from ingest import pipeline

    dev = os.path.expanduser("~/llm-wiki-dev/raw/inbox")
    paths = [os.path.join(dev, name) for name, _ in REAL_GUIDES]
    if all(os.path.isfile(p) for p in paths):
        for path, (name, title) in zip(paths, REAL_GUIDES):
            with open(path, "rb") as f:
                core_llm._client = SummaryClient(summary_json(title), "[]")
                pipeline.file_bytes(f.read(), f"raw/inbox/{name}", "md")
        return "real guides"
    file_faq(faq_text("Fallout 4", FALLOUT_SECTIONS, seed=4), "Fallout 4 guide")
    file_faq(faq_text("Skyrim", SKYRIM_SECTIONS, seed=5), "Skyrim guide", "guide2.md")
    return "generated FAQs"


@pytest.mark.live
def test_live_calibration(seeded_vault, capsys):
    """The 28 questions through `llm.ask` against oMLX, one fresh session each; prints every
    question with its observed route and the confusion table. ≥ 20/28, and no filed-document
    question may go to Wikipedia. The first offered question also gets a real "yes"."""
    guides = _file_guides()
    core_llm._client = None
    a_prompt.SYSTEM_PROMPT = a_prompt.build_system_prompt(avatars.current())
    rows, lines, follow_up = [], [], None
    for q in QUESTIONS:
        conversation.reset()
        expect, question = q["expect"], q["question"]
        recalls = []                                            # the recall blocks quoted (spoken: high band only)
        with patch.object(a_recall, "_quote_block", lambda tag, *a, _q=a_recall._quote_block: recalls.append(_q(tag, *a))
                          or recalls[-1]):
            reply = a_turn.ask(question)
        recalled = any("<recall>" in r for r in recalls)
        observed = observed_route(reply, conversation.get_last_tool_calls(), recalled)
        ok = observed in ACCEPTS[expect]
        rows.append((expect, observed, ok))
        lines.append(f"{'ok ' if ok else 'XX '}{expect:9} → {observed:15} {question}\n    {reply[:220]}")
        if observed == "wikipedia-offer" and follow_up is None:
            answer = a_turn.ask("yes please")
            follow_up = (question, [tc for tc in conversation.get_last_tool_calls()], answer)
    table = confusion(rows)
    with capsys.disabled():
        print(f"\n({guides})")
        print("\n".join(lines))
        print("\n" + table)
        if follow_up:
            print(f"\nyes → {follow_up[1]}\n{follow_up[2][:500]}")
    assert not any(e == "local" and o.startswith("wikipedia") for e, o, _ in rows), table
    assert sum(ok for *_, ok in rows) >= 20, table
    if follow_up:
        assert [tc["name"] for tc in follow_up[1]] == ["get_wikipedia"], follow_up
