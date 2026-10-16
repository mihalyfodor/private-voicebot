"""PRD 50 step 2 — the tool registry (scenarios 2-3)."""

import importlib
import pathlib

from tools import registry

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
    assert len(set(names)) == len(names) == 11
    for t in registry.REGISTRY:
        assert (t.trigger is None) != (t.schema is None), t.name
    assert registry.run_tool("nope", {}) == "unknown tool"
    assert isinstance(registry.run_tool("get_time", {}), str)
