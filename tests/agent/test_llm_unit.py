"""Unit tests for llm.py with a mocked OpenAI client (no oMLX needed).

Covers the tool round trip / ask() surface and the streamed first call
(docs/13-streamed-first-call.md); the context budget lives in tests/agent/test_context.py.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import conversation
from agent import turn
from tests.conftest import FakeClient, stream_chunks as _chunks, tool_call_chunk as _tc_chunk
from core import llm as core_llm
from core import toolset


def setup_function():
    conversation.reset()


# --------------------------------------------------------------------------- ask() surface

def test_ask_stream_yields_only_deltas_and_ask_joins_them():
    fake = FakeClient(*[
        [_tc_chunk(id="call_1", name="get_time", arguments="{}", finish_reason="tool_calls")],
        _chunks("[neutral] Noon."),
        _chunks("[happy] Hello ", "there."),
    ])
    with patch.object(core_llm, "_client", fake), patch.object(turn, "run_tool", return_value="12:00:00"):
        assert list(turn.ask_stream("time?")) == ["[neutral] Noon."]
        assert conversation.get_last_tool_calls() == [{"name": "get_time", "args": {}}]
        reply = turn.ask("hi")
    assert reply == "[happy] Hello there."
    assert conversation._conversation[-1] == {"role": "assistant", "content": "[happy] Hello there."}
    assert conversation.get_last_tool_calls() == []


# --------------------------------------------------------------------------- streamed first call
# --- scenario 1: text-only stream ---

def test_text_only_stream_yields_deltas_in_order_with_one_call():
    fake = FakeClient(*[_chunks("[happy] Hi ", "there", "!")])
    with patch.object(core_llm, "_client", fake), patch.object(turn, "run_tool") as rt:
        events = list(turn.ask_events("hi"))

    assert events == [("delta", "[happy] Hi "), ("delta", "there"), ("delta", "!")]
    rt.assert_not_called()
    assert conversation.get_last_tool_calls() == []
    assert len(fake.calls) == 1
    assert fake.calls[0]["stream"] is True
    assert fake.calls[0]["tools"] == toolset.TOOLS
    assert conversation._conversation[-1] == {"role": "assistant", "content": "[happy] Hi there!"}
    assert conversation._session_turns[-1] == {"role": "assistant", "content": "[happy] Hi there!"}


def test_deltas_are_yielded_before_the_stream_is_exhausted():
    """The generator must emit each delta as it arrives, not after the whole stream."""
    seen = []

    def gen():
        for text in ("[happy] one ", "two ", "three"):
            seen.append(text)
            yield _chunks(text)[0]

    fake = FakeClient(*[gen()])
    with patch.object(core_llm, "_client", fake):
        events = turn.ask_events("hi")
        assert next(events) == ("delta", "[happy] one ")
        assert seen == ["[happy] one "]  # producer has not run ahead
        assert next(events) == ("delta", "two ")
        assert seen == ["[happy] one ", "two "]
        list(events)


# --- scenario 2: tool_call fragments split across chunks ---

def test_tool_call_fragments_reassembled_across_chunks():
    first = [
        _tc_chunk(index=0, id="call_1", name="get_time"),
        _tc_chunk(index=0, arguments='{"tz": '),
        _tc_chunk(index=0, arguments='"UTC"}', finish_reason="tool_calls"),
    ]
    fake = FakeClient(*[first, _chunks("[neutral] It is ", "noon.")])

    with patch.object(core_llm, "_client", fake), patch.object(turn, "run_tool", return_value="12:00:00") as rt:
        events = list(turn.ask_events("what time is it?"))

    rt.assert_called_once_with("get_time", {"tz": "UTC"})
    assert events == [
        ("tool_calls", ["get_time"]),
        ("delta", "[neutral] It is "),
        ("delta", "noon."),
    ]
    assert conversation.get_last_tool_calls() == [{"name": "get_time", "args": {"tz": "UTC"}}]

    roles = [m["role"] for m in conversation._conversation]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert conversation._conversation[2]["tool_calls"] == [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_time", "arguments": '{"tz": "UTC"}'}},
    ]
    assert conversation._conversation[3] == {"role": "tool", "tool_call_id": "call_1", "content": "12:00:00"}
    assert conversation._conversation[4] == {"role": "assistant", "content": "[neutral] It is noon."}

    assert len(fake.calls) == 2
    assert fake.calls[0]["stream"] is True and "tools" in fake.calls[0]
    assert fake.calls[1]["stream"] is True and "tools" not in fake.calls[1]


def test_text_before_tool_call_is_warned_and_dropped_from_reply(capsys):
    first = [
        *_chunks("[neutral] Let me check. "),
        _tc_chunk(index=0, id="call_1", name="get_time", arguments="{}", finish_reason="tool_calls"),
    ]
    fake = FakeClient(*[first, _chunks("[neutral] Noon.")])
    with patch.object(core_llm, "_client", fake), patch.object(turn, "run_tool", return_value="12:00:00"):
        events = list(turn.ask_events("time?"))

    assert events[0] == ("delta", "[neutral] Let me check. ")
    assert events[1] == ("tool_calls", ["get_time"])
    assert "text before tool call was spoken" in capsys.readouterr().out
    assert conversation._conversation[-1] == {"role": "assistant", "content": "[neutral] Noon."}


# --- scenario 3: rollback on exception (at create() and mid-stream) ---

def _raising_stream():
    yield _chunks("[happy] partial ")[0]
    raise RuntimeError("boom")


class RaisingClient:
    """Its create() call itself raises — the request never got a stream."""
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        raise RuntimeError("boom")


@pytest.mark.parametrize("make_fake", [RaisingClient, lambda: FakeClient(*[_raising_stream()])],
                         ids=["at_create", "mid_stream"])
def test_exception_rolls_back_conversation(make_fake):
    with patch.object(core_llm, "_client", make_fake()):
        with pytest.raises(RuntimeError, match="boom"):
            list(turn.ask_events("hi"))

    assert [m["role"] for m in conversation._conversation] == ["system"]
    assert conversation._session_turns == []


# --------------------------------------------------------------------------- PRD 56: prompt freshness


def test_prompt_has_no_clock(monkeypatch):
    import re
    from datetime import date
    from core import avatars
    from agent import prompt
    from tools import weather
    monkeypatch.setattr(weather, "location", lambda: {"name": "Honolulu", "tz": "Pacific/Honolulu",
                                                      "lat": 0, "lon": 0, "source": "env"})
    monkeypatch.setattr(prompt, "today", lambda: date(2026, 10, 17))
    text = prompt.build_system_prompt(avatars.current())
    assert not re.search(r"\b\d{1,2}:\d\d\b", text)
    assert "Today's date is Saturday, October 17 2026." in text
    assert "The user is in Honolulu (timezone Pacific/Honolulu). " in text


def test_today_is_the_date_in_the_users_timezone(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from agent import prompt
    from tools import time as t_time
    from tools import weather
    monkeypatch.setattr(weather, "location", lambda: {"tz": "Pacific/Auckland", "source": "env"})
    assert prompt.today() == datetime.now(ZoneInfo("Pacific/Auckland")).date()
    monkeypatch.setattr(weather, "location", lambda: {"tz": "auto", "source": "env"})
    assert t_time.now().tzinfo is not None and prompt.today() == t_time.now().date()


def test_save_memory_rebuilds_prompt_when_ops_applied(monkeypatch):
    from agent import fold
    from tests.conftest import seed
    seed(identity={"name": "Misi"})
    conversation.reset()
    before = conversation._conversation[0]
    assert "Misi" in before["content"]
    conversation._session_turns.append({"role": "user", "content": "call me Mihaly"})

    def applied(turns, sources=()):
        seed(identity={"name": "Mihaly"})
        return [{"op": "UPDATE"}]

    monkeypatch.setattr(fold, "save", applied)
    conversation.save_memory()
    assert "Mihaly" in conversation._conversation[0]["content"]

    same = conversation._conversation[0]
    conversation._session_turns.append({"role": "user", "content": "hello"})
    monkeypatch.setattr(fold, "save", lambda turns, sources=(): [])      # nothing applied: no rebuild
    conversation.save_memory()
    assert conversation._conversation[0] is same
