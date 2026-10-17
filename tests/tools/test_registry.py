"""PRD 50 steps 2-3 — the tool registry (scenarios 2-4, 6)."""

import importlib
import pathlib
from unittest.mock import patch

import pytest

from tools import commands, registry
from ingest import pipeline

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_schemas_match_the_golden_fixture():
    import json
    assert json.dumps(registry.schemas()) == (ROOT / "tests/fixtures/tool_schemas.json").read_text()


def test_tokens_is_the_old_constant():
    assert registry.tokens() == 1089


def test_read_tools_and_hosts_derive_from_the_records():
    assert set(registry.READ_TOOLS) == {"search_wiki", "read_page", "read_source", "get_emails", "get_news",
                                        "get_news_detail", "get_wikipedia", "read_wikipedia"}
    assert "nominatim.openstreetmap.org" in registry.ALLOWED_HOSTS
    for mod in ("time", "weather", "news", "gmail", "wiki", "sources", "wikipedia"):
        assert set(importlib.import_module(f"tools.{mod}").HOSTS) <= registry.ALLOWED_HOSTS


def test_invariants():
    names = registry.names()
    assert len(set(names)) == len(names) == 19
    assert len(registry.schemas()) == 11
    assert names[11:] == ["new_topic", "save_note", "briefing", "forget", "push", "write_up", "research",
                          "open_page"]   # routing order
    for t in registry.REGISTRY:
        assert (t.trigger is None) != (t.schema is None), t.name
    assert registry.run_tool("nope", {}) == "unknown tool"
    assert isinstance(registry.run_tool("get_time", {}), str)
    assert registry.run_tool("briefing", {}) == "unknown tool"     # a trigger tool is not the model's to call
    by = {t.name: t for t in registry.REGISTRY}
    bad = (by["write_up"], by["briefing"])      # needs_llm before a gate-free trigger would change offline routing
    with pytest.raises(AssertionError, match="needs_llm"):
        registry._validate(bad)


# --------------------------------------------------------------- scenario 4: trigger routing

PREDICATES = {"new_topic": commands.new_topic_command, "save_note": commands.save_note_command,
              "briefing": commands.briefing_command, "write_up": commands.wants_writeup,
              "forget": lambda t: commands.forget_command(t) is not None, "push": commands.push_command,
              "research": lambda t: commands.research_command(t) is not None,
              "open_page": lambda t: commands.page_command(t) is not None}
PHRASES = [
    ("new topic", "new_topic"), ("/new", "new_topic"), ("New topic!", "new_topic"),
    ("save that as a note", "save_note"), ("save this to the wiki", "save_note"), ("save it", "save_note"),
    ("what's new?", "briefing"), ("brief me", "briefing"), ("anything new", "briefing"),
    ("what changed", "briefing"), ("what did I miss", "briefing"), ("/brief", "briefing"),
    ("write it up", "write_up"), ("yes, write that up please", "write_up"),
    ("give me the written version", "write_up"), ("/write", "write_up"),
    ("open the page about covenant", "open_page"), ("find fallout", "open_page"),
    ("show me the note on kyoto", "open_page"), ("go back to that page", "open_page"),
    ("what were we looking at", "open_page"),
    ("forget the Morrowind guide", "forget"), ("please forget Morrowind guide", "forget"),
    ("remove the Morrowind guide source", "forget"),
    ("send that to my reMarkable", "push"), ("push it to the tablet", "push"),
    # PRD 54: research is explicit only
    ("research how the Falcon and Zephyr budgets differ", "research"), ("/research Zephyr budget", "research"),
    ("Research, the Zephyr budget.", "research"), ("research", None), ("I did some research on Zephyr", None),
    ("Research the Falcon and Zephyr budgets", "research"), ("research how the budgets differ", "research"),
    ("research Zephyr", "research"), ("Research shows coffee is healthy", None),
    ("research papers on Zephyr are boring", None), ("Research is fun, isn't it?", None),
    ("research on Zephyr", None),
    ("research about Zephyr", "research"), ("research into the Zephyr budget", "research"),
    ("research for the Zephyr budget", None), ("research about", None), ("research about is fun", None), ("research says otherwise", None), ("/research on Zephyr", "research"),
    ("what's new with you", None), ("hello there", None), ("", None),
    # PRD 52 negatives: none of these is a forget / push
    ("forget it", None), ("forget about it", None), ("don't forget to call Anna", None),
    ("forget what I said", None), ("remove the meeting", None), ("send an email to Anna", None),
    ("forget my address", None), ("Forget Anna", None), ("forget the last thing I told you", None),
    ("forget we talked about that", None), ("remove the Falcon status source", "forget"),
    ("forget the Falcon status notebook", "forget"), ("forget the Morrowind guide source", "forget"),
]


@pytest.mark.parametrize("text, expected", PHRASES)
def test_match_routes_each_phrase_to_its_tool(text, expected):
    tool = registry.match(text)
    assert (tool.name if tool else None) == expected


@pytest.mark.parametrize("text", [p for p, _ in PHRASES] + ["  Find   Fallout  ", "please write it out", "save"])
def test_match_agrees_with_the_predicates(text):
    """The first predicate (in registry order) that holds is the tool `match` picks."""
    hits = [name for name in registry.names()[11:] if PREDICATES[name](text)]
    tool = registry.match(text)
    assert (tool.name if tool else None) == (hits[0] if hits else None)


def test_handler_binds_only_trigger_tools_and_an_unbound_one_raises():
    with pytest.raises(ValueError):
        registry.handler("get_time")
    with pytest.raises(ValueError):
        registry.handler("nope")
    run = registry._dispatch("nope")
    with pytest.raises(RuntimeError, match="no handler bound for nope"):
        run(None)


def test_the_server_binds_every_trigger_tool():
    import chatbot  # noqa: F401  (binding happens at import)
    assert set(registry._HANDLERS) == set(registry.names()[11:])


# --------------------------------------------------------------- scenario 6: file_source without the server

def test_file_source_refreshes_the_panel_through_the_bound_ctx():
    pushed = []
    ctx = registry.Ctx(text="", origin="user", session=None, turn=None, send=lambda m: None,
                       push_panel=lambda: pushed.append(1))
    with patch.object(pipeline, "ingest_note", return_value={"path": "sources/x", "raw": "raw/x"}):
        with registry.bind(ctx):
            assert registry.current() is ctx
            assert registry.run_tool("file_source", {"kind": "note", "ref": "hi"}) == "filed as sources/x (raw: raw/x)."
        assert pushed == [1] and registry.current() is None
        registry.run_tool("file_source", {"kind": "note", "ref": "hi"})      # no turn bound: no panel, no error
    assert pushed == [1]
    text = (ROOT / "tools/sources.py").read_text()
    assert "import chatbot" not in text and "import server" not in text
