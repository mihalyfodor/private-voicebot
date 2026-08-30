"""PRD 34 — grounded answers: recall precision, the look-up-first prompt block, the one-turn
"dig deeper" offer, the Settings note. Scenario numbers are the PRD's.

The FAQ fixtures come from tests/sources/test_source_recall.py, the vault and the streaming fake from conftest.
"""
import sys, os

import re
from unittest.mock import patch

import pytest

import avatars
import fillers
import llm
import memory
import wiki
from tests.conftest import RawStreamClient, StreamClient, last_user, r, seeded_vault, stream_chunks, tool_call_chunk  # noqa: F401
from tests.sources.test_source_recall import fallout, both  # noqa: F401  (fixtures)

pytestmark = pytest.mark.usefixtures("fresh_llm")   # no mocked turn may leak into another test's session replay
QUESTION = "I'm at Covenant and Honest Dan keeps attacking me, what now"
OFFER_REPLY = "[thinking] Talk to Jacob Orden, says the Fallout 4 guide. Want me to dig deeper?"

SMALL_TALK = ["what should we cook tonight?", "what's on today?", "any eggs left?"]
IN_GUIDE = ["I'm at Covenant and the turrets shoot me", "Thieves Guild in Riften"]


def _offer_turn(reply: str = OFFER_REPLY) -> StreamClient:
    """One recall turn whose mocked reply ends with the offer; returns the fake."""
    llm.reset()
    fake = StreamClient(reply)
    with patch.object(llm, "_client", fake):
        llm.ask(QUESTION)
    return fake


def _turn(text: str, reply: str = "[happy] Here is more.") -> tuple[list, StreamClient]:
    fake = StreamClient(reply)
    with patch.object(llm, "_client", fake):
        events = list(llm.ask_events(text))
    return events, fake


# --------------------------------------------------------------- 1. recall precision


def test_small_talk_injects_nothing_and_guide_questions_still_do(both):
    for q in SMALL_TALK:
        assert llm.recall_for(q) == "", q
    for q in IN_GUIDE:
        assert llm.recall_for(q), q


def test_the_precision_rule_holds_even_with_the_threshold_at_zero(both, file_sandbox):
    avatars.save_setting("recall_min_score", 0.0)
    assert wiki.search_chunks("what should we cook tonight?")     # would have matched before the rule
    for q in SMALL_TALK:
        assert llm.recall_for(q) == "", q
    for q in IN_GUIDE:
        assert llm.recall_for(q), q


def test_hits_carry_matched_terms_and_proper_terms_are_title_case_words(both):
    hits = wiki.search_chunks("covenant compound turrets")
    assert hits and all(h["matched"] == sorted(h["matched"]) for h in hits)
    assert "covenant" in hits[0]["matched"]
    assert wiki.proper_terms({"covenant", "cook"}) == {"covenant"}


# --------------------------------------------------------------- 2. the prompt block


AVATAR = {"name": "Test", "persona": "You are Test.", "verbosity": None}


def test_the_system_prompt_asks_to_look_it_up_first_and_offer_more():
    prompt = llm.build_system_prompt(AVATAR)
    assert "Look it up first" in prompt and llm.DEEPER_OFFERS[0] in prompt
    assert prompt.endswith(llm.GROUNDING_RULES + llm.ROUTING_RULES)


def test_a_weather_question_goes_through_the_tool_never_memory():
    """The block names the live-fact rule; a weather turn records the get_weather call."""
    prompt = llm.build_system_prompt({"name": "Test", "persona": "You are Test.", "verbosity": None})
    assert "never state a temperature, a time or a headline from memory" in prompt
    llm.reset()
    fake = RawStreamClient([
        [tool_call_chunk(id="call_1", name="get_weather", arguments="{}", finish_reason="tool_calls")],
        stream_chunks("[neutral] Sunny, 21 degrees."),
    ])
    with patch.object(llm, "_client", fake), patch.object(llm, "run_tool", return_value="21C sunny"):
        events = list(llm.ask_events("how is the weather today?"))
    assert events[0] == ("tool_calls", ["get_weather"])
    assert [tc["name"] for tc in llm.get_last_tool_calls()] == ["get_weather"]
    assert "get_weather" in fake.calls[0]["messages"][0]["content"]
    assert llm._deeper_offer is None


def test_a_tool_free_run_gets_neither_rule(monkeypatch):
    """No tools: no grounding and no routing rules; without get_wikipedia alone only the offer goes (PRD 37)."""
    from tools import TOOLS
    monkeypatch.setattr(llm, "TOOLS", [])
    prompt = llm.build_system_prompt(AVATAR)
    assert "Look it up first" not in prompt and llm.DEEPER_OFFERS[0] not in prompt
    assert llm.WIKIPEDIA_OFFER not in prompt
    assert llm.grounding_rules() == ""
    monkeypatch.setattr(llm, "TOOLS", [t for t in TOOLS if t["function"]["name"] != "get_wikipedia"])
    prompt = llm.build_system_prompt(AVATAR)
    assert "Look it up first" in prompt and llm.WIKIPEDIA_OFFER not in prompt


# --------------------------------------------------------------- 3. the deeper offer


def test_an_offer_after_a_recall_turn_is_remembered_with_its_parts(fallout):
    fake = _offer_turn()
    sent_user = fake.calls[0]["messages"][1]["content"]           # the block is stripped afterwards (PRD 39)
    parts = [int(m) for m in re.findall(r"\(part (\d+)/\d+\)", sent_user)]
    assert parts and set(parts) <= {h["n"] for h in wiki.search_chunks(QUESTION, k=llm.RECALL_MAX_CHUNKS)}
    assert llm._deeper_offer == {"source": fallout["path"], "title": "Fallout 4 guide",
                                 "parts": sorted(parts)}


def test_yes_please_reads_the_neighbours_long_for_that_turn_only(fallout):
    _offer_turn()
    offered = llm._deeper_offer["parts"]
    total = wiki.search_chunks(QUESTION)[0]["total"]
    events, fake = _turn("yes please")
    assert ("tool_calls", ["dig_deeper"]) in events
    sent = fake.calls[-1]
    user = last_user(fake)
    assert user.startswith("yes please\n\n<source title=\"Fallout 4 guide\">")
    parts = [int(m) for m in re.findall(rf"^Part (\d+)/{total}:", user, re.M)]
    assert parts == sorted(parts) and 1 < len(parts) <= llm.DEEPER_MAX_CHUNKS
    assert offered[0] in parts
    assert any(n - 1 in parts or n + 1 in parts for n in offered)
    assert memory.tokens_est(user.split("\n\n", 1)[1]) <= llm.DEEPER_MAX_TOKENS + 40
    assert sent["max_tokens"] == llm.VERBOSITY_MAX_TOKENS["long"]
    assert llm.VERBOSITY_RULES["long"] in sent["messages"][0]["content"]
    assert llm.VERBOSITY_RULES["normal"] in llm._conversation[0]["content"]
    assert llm.VERBOSITY_RULES["long"] not in llm._conversation[0]["content"]
    assert llm.current_verbosity() == "normal"
    assert llm._deeper_offer is None
    assert llm._session_turns[-2] == {"role": "user", "content": "yes please"}


def test_anything_else_clears_the_offer_and_injects_nothing(fallout):
    _offer_turn()
    _, fake = _turn("no, what's the weather")
    user = last_user(fake)
    assert "<source" not in user and user.startswith("no, what's the weather")
    assert fake.calls[-1]["max_tokens"] == 400
    assert llm._deeper_offer is None


def test_an_offer_never_persists_past_one_turn(fallout):
    _offer_turn()
    _turn("tell me a joke about programmers", "[happy] Why do programmers prefer dark mode?")
    assert llm._deeper_offer is None
    _, fake = _turn("yes")
    assert "<source" not in last_user(fake)
    assert fake.calls[-1]["max_tokens"] == 400


def test_a_deeper_reply_ending_with_an_offer_offers_again(fallout):
    _offer_turn()
    _turn("yes please", "[happy] More here. Shall I dig deeper?")
    assert llm._deeper_offer and llm._deeper_offer["source"] == fallout["path"]


@pytest.mark.parametrize("text", ["yes", "Yes please!", "go on", "dig deeper", "sure, do it",
                                  "yeah tell me more", "sure, check wikipedia", "look it up", "go ahead"])
def test_short_affirmatives(text):
    """`is_affirmative` answers both the dig-deeper and the Wikipedia offer (PRD 34 / 37)."""
    assert llm.is_affirmative(text)


@pytest.mark.parametrize("text", ["no", "no, what's the weather", "yes but what's the weather", "",
                                  "yes please go on and read more"])
def test_non_affirmatives(text):
    assert not llm.is_affirmative(text)


def test_ends_with_offer_detects_each_phrase():
    for phrase in llm.DEEPER_OFFERS:
        assert llm.ends_with_offer(f"[neutral] Talk to Jacob. {phrase}")
        assert llm.ends_with_offer(f'[neutral] Talk to Jacob. "{phrase}" ')
    assert not llm.ends_with_offer("[neutral] Talk to Jacob.")
    assert not llm.ends_with_offer("[neutral] Want me to dig deeper? Anyway, talk to Jacob.")
    assert not llm.ends_with_offer("")


def test_deeper_parts_adds_neighbours_offered_first_max_three_sorted():
    assert llm.deeper_parts([5], 100) == [4, 5, 6]
    assert llm.deeper_parts([1], 100) == [1, 2]
    assert llm.deeper_parts([5, 9], 100) == [4, 5, 9]
    assert llm.deeper_parts([1], 1) == [1]


def test_tool_results_register_their_sources():
    llm._turn_sources.clear()
    llm._note_tool_sources("search_wiki", {},
                           "Fallout 4 guide · part 12/300 (sources/fallout-4-guide): text")
    llm._note_tool_sources("read_source", {"path": "sources/x", "part": 3}, "X · part 3/9\ntext")
    assert llm._turn_sources == [("sources/fallout-4-guide", "Fallout 4 guide", 12), ("sources/x", "X", 3)]


# --------------------------------------------------------------- 5. fixture render


def test_the_reply_length_row_notes_that_deeper_answers_are_long(r):
    assert "deeper answers are long regardless" in r["verbosity_note"]


@pytest.mark.parametrize("key", ["dig_deeper", "get_wikipedia", "read_wikipedia"])
def test_the_filler_for_each_lookup_exists(key):
    """The contract with fillers.py: every tool / offer the model can take has a spoken filler."""
    assert fillers.FILLERS[key]
    assert fillers.pick(key) in fillers.FILLERS[key]


# --------------------------------------------------------------- 4. live


REAL_GUIDES = (
    ("2026-08-30-note.md", "Fallout 4 FAQ and Walkthrough"),
    ("2026-08-30-note-2.md", "Elder Scrolls V FAQ and Walkthrough"),
    ("2026-08-30-morrowind-game-of-the-year-edition-guide.md", "Morrowind Game of the Year Edition guide"),
)
WEATHER_QUESTION = "how is the weather today?"
LIVE_QUESTIONS = [
    "I'm at Covenant and they shoot me on sight, what am I supposed to do",
    "how do I pass the SAFE test in Covenant and what is Honest Dan looking for",
    "where do I find the Silver Shroud costume at Hubris Comics",
    "where do I learn the Unrelenting Force shout from the Greybeards",
    "how do I become Thane of Whiterun after the dragon at the watchtower",
    "in Morrowind, who is Caius Cosades in Balmora and what does he want me to do first",
]
CITED = ("guide", "walkthrough", "faq", "fallout", "skyrim", "elder scrolls", "morrowind")


def _neighbour_names(offer: dict) -> set:
    """Title-case words the neighbouring chunks (the ones the short answer never saw) contain."""
    chunks = wiki.raw_chunks(offer["source"])["chunks"]
    names = set()
    for n in llm.deeper_parts(offer["parts"], len(chunks)):
        if n not in offer["parts"]:
            names |= set(re.findall(r"\b[A-Z][a-z]{3,}\b", chunks[n - 1]["text"]))
    return names


@pytest.mark.live
def test_live_answers_cite_offer_and_dig_deeper(seeded_vault, capsys):
    """The three real guides in a temp vault, six questions through `llm.ask` against oMLX:
    ≥ 5 cite a guide, ≥ 4 end with the offer, and ≥ 4 "yes" answers are ≥ 2× as long and mention
    a name from a neighbouring chunk (sampling varies); a weather question calls get_weather. Prints every pair."""
    from tests.conftest import FakeClient
    from tests.sources.test_source_recall import summary_json
    import sources

    dev = os.path.expanduser("~/llm-wiki-dev/raw/inbox")
    for name, title in REAL_GUIDES:
        path = os.path.join(dev, name)
        if not os.path.isfile(path):
            pytest.skip(f"{path} not available")
        with open(path, "rb") as f:
            data = f.read()
        sources.file_bytes(data, f"raw/inbox/{name}", "md", client=FakeClient(summary_json(title), "[]"),
                           model="m")
    llm._client = None
    llm.SYSTEM_PROMPT = llm.build_system_prompt(avatars.current())
    pairs = []
    for question in LIVE_QUESTIONS:
        llm.reset()
        answer = llm.ask(question)
        offer = llm._deeper_offer
        cited = any(w in answer.lower() for w in CITED)
        deeper = neighbour = None
        if offer:
            names = _neighbour_names(offer)
            deeper = llm.ask("yes please")
            neighbour = sorted(n for n in names if n in deeper)
        pairs.append((question, answer, cited, offer, deeper, neighbour))
    llm.reset()
    weather = llm.ask(WEATHER_QUESTION)
    weather_tools = [tc["name"] for tc in llm.get_last_tool_calls()]
    with capsys.disabled():
        print()
        print(f"Q: {WEATHER_QUESTION}\nA (tools {weather_tools}): {weather[:300]}\n")
        for q, a, cited, offer, deeper, neighbour in pairs:
            print(f"Q: {q}\nA{'' if cited else ' (no source named)'}: {a}")
            if offer:
                print(f"   offer: {offer['title']} parts {offer['parts']}\n"
                      f"   yes ({len(deeper)} chars, {len(deeper) / len(a):.1f}x; neighbour names {neighbour}): "
                      f"{deeper[:300]}")
            print()
    assert "get_weather" in weather_tools, weather
    assert sum(cited for _, _, cited, *_ in pairs) >= 5
    assert sum(bool(offer) for *_, offer, _, _ in pairs) >= 4
    deep = [(q, len(a), len(deeper), neighbour) for q, a, _, offer, deeper, neighbour in pairs if offer]
    good = [d for d in deep if d[2] >= 2 * d[1] and d[3]]
    assert len(good) >= 4, [d for d in deep if d not in good]
