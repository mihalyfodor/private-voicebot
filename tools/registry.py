"""The ordered tool registry: one `Tool` record per tool, in the order the model sees the schemas."""

import json
import re
from dataclasses import dataclass
from typing import Callable

from tools import gmail as _gmail
from tools import news as _news
from tools import sources as _sources
from tools import time as _time
from tools import weather as _weather
from tools import wiki as _wiki
from tools import wikipedia as _wikipedia


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    run: Callable
    trigger: re.Pattern | None = None   # trigger-mode tools arrive in step 3
    schema: dict | None = None          # the OpenAI function definition (schema mode)
    takeover: bool = False              # the turn ends in the tool, no model completion
    needs_llm: bool = False             # sits after the llm_ok gate (step 3)
    writes: bool = False                # metadata only; PRD 52 reads it later
    hosts: tuple[str, ...] = ()
    untrusted: bool = False             # result is text somebody else wrote -> wrapped as <document>


def _schema_tool(definition, run, hosts, untrusted=True, writes=False):
    fn = definition["function"]
    return Tool(name=fn["name"], description=fn["description"], run=run, schema=definition,
                hosts=hosts, untrusted=untrusted, writes=writes)


# routing order is behaviour: this is the order of the schemas sent to the model
REGISTRY: tuple[Tool, ...] = (
    _schema_tool(_time.DEFINITION, _time.run, _time.HOSTS, untrusted=False),
    _schema_tool(_news.DEFINITION_NEWS, _news.run_get_news, _news.HOSTS),
    _schema_tool(_news.DEFINITION_NEWS_DETAIL, _news.run_get_news_detail, _news.HOSTS),
    _schema_tool(_gmail.DEFINITION, _gmail.run, _gmail.HOSTS),
    _schema_tool(_weather.DEFINITION, _weather.run, _weather.HOSTS, untrusted=False),
    _schema_tool(_wiki.DEFINITION_SEARCH, _wiki.run_search, _wiki.HOSTS),
    _schema_tool(_wiki.DEFINITION_READ, _wiki.run_read, _wiki.HOSTS),
    _schema_tool(_wiki.DEFINITION_READ_SOURCE, _wiki.run_read_source, _wiki.HOSTS),
    _schema_tool(_sources.DEFINITION, _sources.run, _sources.HOSTS, untrusted=False, writes=True),
    _schema_tool(_wikipedia.DEFINITION_GET, _wikipedia.run_get_wikipedia, _wikipedia.HOSTS),
    _schema_tool(_wikipedia.DEFINITION_READ, _wikipedia.run_read_wikipedia, _wikipedia.HOSTS),
)


def _validate():
    names = [t.name for t in REGISTRY]
    assert len(set(names)) == len(names), "duplicate tool name"
    for t in REGISTRY:
        assert (t.trigger is None) != (t.schema is None), f"{t.name}: exactly one of trigger/schema"


_validate()
_BY_NAME = {t.name: t for t in REGISTRY}

#: the tools whose result is free text somebody else wrote; each wraps it in document tags
READ_TOOLS: tuple[str, ...] = tuple(t.name for t in REGISTRY if t.untrusted)
#: every host any tool may talk to — the read-only policy test checks each module against it
ALLOWED_HOSTS = frozenset(h for t in REGISTRY for h in t.hosts)


def names() -> list[str]:
    return [t.name for t in REGISTRY]


def schemas() -> list[dict]:
    return [t.schema for t in REGISTRY if t.schema]


def tokens() -> int:
    """What the tool schemas add to every request (chars // 4, like estimate_tokens)."""
    return len(json.dumps(schemas())) // 4


def run_tool(name, args):
    tool = _BY_NAME.get(name)
    return tool.run(args) if tool else "unknown tool"
