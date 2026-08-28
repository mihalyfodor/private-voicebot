"""Tests for the streamed first LLM call (docs/13-streamed-first-call.md)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import llm


def _text_chunk(text, finish_reason=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=text, tool_calls=None), finish_reason=finish_reason)])


def _tc_chunk(index=0, id=None, name=None, arguments=None, finish_reason=None):
    frag = SimpleNamespace(index=index, id=id,
                           function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=None, tool_calls=[frag]), finish_reason=finish_reason)])


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return self._responses.pop(0)


def setup_function():
    llm.reset()


# --- scenario 1: text-only stream ---

def test_text_only_stream_yields_deltas_in_order_with_one_call():
    chunks = [_text_chunk("[happy] Hi "), _text_chunk("there"), _text_chunk("!", finish_reason="stop")]
    fake = FakeClient([iter(chunks)])
    with patch.object(llm, "_client", fake), patch.object(llm, "run_tool") as rt:
        events = list(llm.ask_events("hi"))

    assert events == [("delta", "[happy] Hi "), ("delta", "there"), ("delta", "!")]
    rt.assert_not_called()
    assert len(fake.calls) == 1
    assert fake.calls[0]["stream"] is True
    assert fake.calls[0]["tools"] is llm.TOOLS
    assert llm._conversation[-1] == {"role": "assistant", "content": "[happy] Hi there!"}
    assert llm._session_turns[-1] == {"role": "assistant", "content": "[happy] Hi there!"}


def test_deltas_are_yielded_before_the_stream_is_exhausted():
    """The generator must emit each delta as it arrives, not after the whole stream."""
    seen = []

    def gen():
        for text in ("[happy] one ", "two ", "three"):
            seen.append(text)
            yield _text_chunk(text)

    fake = FakeClient([gen()])
    with patch.object(llm, "_client", fake):
        events = llm.ask_events("hi")
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
    second = [_text_chunk("[neutral] It is "), _text_chunk("noon.")]
    fake = FakeClient([iter(first), iter(second)])

    with patch.object(llm, "_client", fake), patch.object(llm, "run_tool", return_value="12:00:00") as rt:
        events = list(llm.ask_events("what time is it?"))

    rt.assert_called_once_with("get_time", {"tz": "UTC"})
    assert events == [
        ("tool_calls", ["get_time"]),
        ("delta", "[neutral] It is "),
        ("delta", "noon."),
    ]
    assert llm.get_last_tool_calls() == [{"name": "get_time", "args": {"tz": "UTC"}}]

    roles = [m["role"] for m in llm._conversation]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert llm._conversation[2]["tool_calls"] == [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_time", "arguments": '{"tz": "UTC"}'}},
    ]
    assert llm._conversation[3] == {"role": "tool", "tool_call_id": "call_1", "content": "12:00:00"}
    assert llm._conversation[4] == {"role": "assistant", "content": "[neutral] It is noon."}

    assert len(fake.calls) == 2
    assert fake.calls[0]["stream"] is True and "tools" in fake.calls[0]
    assert fake.calls[1]["stream"] is True and "tools" not in fake.calls[1]


def test_text_before_tool_call_is_warned_and_dropped_from_reply(capsys):
    first = [
        _text_chunk("[neutral] Let me check. "),
        _tc_chunk(index=0, id="call_1", name="get_time", arguments="{}", finish_reason="tool_calls"),
    ]
    fake = FakeClient([iter(first), iter([_text_chunk("[neutral] Noon.")])])
    with patch.object(llm, "_client", fake), patch.object(llm, "run_tool", return_value="12:00:00"):
        events = list(llm.ask_events("time?"))

    assert events[0] == ("delta", "[neutral] Let me check. ")
    assert events[1] == ("tool_calls", ["get_time"])
    assert "text before tool call was spoken" in capsys.readouterr().out
    assert llm._conversation[-1] == {"role": "assistant", "content": "[neutral] Noon."}


# --- scenario 3: rollback on exception mid-stream ---

def test_exception_mid_stream_rolls_back_conversation():
    def gen():
        yield _text_chunk("[happy] partial ")
        raise RuntimeError("boom")

    fake = FakeClient([gen()])
    with patch.object(llm, "_client", fake):
        with pytest.raises(RuntimeError, match="boom"):
            list(llm.ask_events("hi"))

    assert [m["role"] for m in llm._conversation] == ["system"]
    assert llm._session_turns == []
