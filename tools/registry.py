"""The ordered tool registry: one `Tool` record per tool, in the order the model sees the schemas."""

import contextvars
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

from tools import commands as _commands
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
    run: Callable                       # schema: run(args) -> str; trigger: run(ctx) -> str | None
    trigger: re.Pattern | None = None   # trigger mode: matched on the whole normalised turn
    schema: dict | None = None          # the OpenAI function definition (schema mode)
    takeover: bool = False              # the turn ends in the tool, no model completion
    needs_llm: bool = False             # sits after the llm_ok gate
    writes: bool = False                # metadata: the tool writes (forget/push only propose; PRD 52)
    hosts: tuple[str, ...] = ()
    untrusted: bool = False             # result is text somebody else wrote -> wrapped as <document>


@dataclass(frozen=True)
class Ctx:
    """What a trigger tool's body gets: the turn and the server's ways to talk back."""
    text: str
    origin: str
    session: Any
    turn: int | None
    send: Callable
    push_panel: Callable


_CTX: contextvars.ContextVar = contextvars.ContextVar("tool_ctx", default=None)


@contextmanager
def bind(ctx: Ctx):
    """Make `ctx` the current turn's context for the tools run inside (e.g. file_source's panel refresh)."""
    token = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(token)


def current() -> Ctx | None:
    return _CTX.get()


#: trigger tool name -> its body, bound by the server (`@registry.handler(name)`)
_HANDLERS: dict[str, Callable] = {}


def _dispatch(name: str) -> Callable:
    def run(ctx):
        fn = _HANDLERS.get(name)
        if fn is None:
            raise RuntimeError(f"no handler bound for {name}")
        return fn(ctx)
    return run


def _trigger_tool(name, description, trigger, takeover=False, needs_llm=False, writes=False):
    return Tool(name=name, description=description, run=_dispatch(name), trigger=trigger,
                takeover=takeover, needs_llm=needs_llm, writes=writes, untrusted=False)


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
    # trigger tools: the first whose regex matches the turn routes it (before the gate unless needs_llm)
    _trigger_tool("new_topic", "Fold memory and restart on a summary.", _commands.NEW_TOPIC, takeover=True),
    _trigger_tool("save_note", "File the last card as a wiki note.", _commands.SAVE_NOTE, takeover=True,
                  writes=True),
    _trigger_tool("briefing", "Recap what changed in the wiki since the last briefing.", _commands.BRIEF,
                  takeover=True),
    # approvals (docs/52-approvals.md): each proposes a write and asks; the next turn decides
    _trigger_tool("forget", "Propose forgetting the source named.", _commands.FORGET, takeover=True, writes=True),
    _trigger_tool("push", "Propose sending the last card to the reMarkable.", _commands.PUSH, takeover=True,
                  writes=True),
    _trigger_tool("write_up", "Turn the last answer into a written card.", _commands.WANTS_WRITEUP,
                  takeover=True, needs_llm=True),
    _trigger_tool("open_page", "Open the page asked for, then answer with a note.", _commands.OPEN_PAGE_ANY,
                  needs_llm=True),
)


def _validate(registry=None):
    registry = REGISTRY if registry is None else registry
    names = [t.name for t in registry]
    assert len(set(names)) == len(names), "duplicate tool name"
    for t in registry:
        assert (t.trigger is None) != (t.schema is None), f"{t.name}: exactly one of trigger/schema"
    flags = [t.needs_llm for t in registry if t.trigger is not None]
    assert flags == sorted(flags), "needs_llm triggers must come after every gate-free trigger"


_validate()
_BY_NAME = {t.name: t for t in REGISTRY}

#: the tools whose result is free text somebody else wrote; each wraps it in document tags
READ_TOOLS: tuple[str, ...] = tuple(t.name for t in REGISTRY if t.untrusted)
#: every host any tool may talk to — the read-only policy test checks each module against it
ALLOWED_HOSTS = frozenset(h for t in REGISTRY for h in t.hosts)


def handler(name: str) -> Callable:
    """Bind the body of the trigger tool `name` (the server does this; the body drives TTS and state)."""
    tool = _BY_NAME.get(name)
    if tool is None or tool.trigger is None:
        raise ValueError(f"{name} is not a trigger tool")

    def bind_body(fn):
        _HANDLERS[name] = fn
        return fn
    return bind_body


def match(text) -> Tool | None:
    """The first trigger tool whose regex matches the whole normalised turn, else None."""
    normalised = _commands.normalise(text)
    return next((t for t in REGISTRY if t.trigger is not None and t.trigger.search(normalised)), None)


def names() -> list[str]:
    """Every tool name, schema tools first (in schema order), then the trigger tools in routing order."""
    return [t.name for t in REGISTRY]


def schemas() -> list[dict]:
    return [t.schema for t in REGISTRY if t.schema]


def tokens() -> int:
    """What the tool schemas add to every request (chars // 4, like estimate_tokens)."""
    return len(json.dumps(schemas())) // 4


def run_tool(name, args):
    """Run a schema tool the model called; a trigger tool's name is as unknown to the model as any other."""
    tool = _BY_NAME.get(name)
    return tool.run(args) if tool and tool.schema else "unknown tool"
