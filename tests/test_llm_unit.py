"""Unit tests for llm.py with a mocked OpenAI client (no oMLX needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import llm


def _chunks(*texts):
    return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t, tool_calls=None))]) for t in texts]


def _tc_chunk(index=0, id=None, name=None, arguments=None, finish_reason=None):
    """One streamed chunk carrying a tool_call fragment."""
    frag = SimpleNamespace(
        index=index, id=id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
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


def test_ask_returns_reply_and_records_turn():
    fake = FakeClient([iter(_chunks("[happy] Hello there."))])
    with patch.object(llm, "_client", fake):
        reply = llm.ask("hi")
    assert reply == "[happy] Hello there."
    assert llm._conversation[-1] == {"role": "assistant", "content": "[happy] Hello there."}
    assert llm.get_last_tool_calls() == []


def test_tool_round_then_streamed_answer():
    fake = FakeClient([
        iter([_tc_chunk(id="call_1", name="get_time", arguments="{}", finish_reason="tool_calls")]),
        iter(_chunks("[neutral] It is ", "noon.")),
    ])
    with patch.object(llm, "_client", fake), patch.object(llm, "run_tool", return_value="12:00:00") as rt:
        events = list(llm.ask_events("what time is it?"))
    rt.assert_called_once_with("get_time", {})
    assert events == [("tool_calls", ["get_time"]), ("delta", "[neutral] It is "), ("delta", "noon.")]
    assert llm.get_last_tool_calls() == [{"name": "get_time", "args": {}}]
    roles = [m["role"] for m in llm._conversation]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert llm._conversation[3]["tool_call_id"] == "call_1"
    assert fake.calls[1]["stream"] is True


def test_ask_stream_yields_only_deltas():
    fake = FakeClient([
        iter([_tc_chunk(id="call_1", name="get_time", arguments="{}", finish_reason="tool_calls")]),
        iter(_chunks("[neutral] Noon.")),
    ])
    with patch.object(llm, "_client", fake), patch.object(llm, "run_tool", return_value="12:00:00"):
        assert list(llm.ask_stream("time?")) == ["[neutral] Noon."]


class RaisingClient:
    """Mimics FakeClient but its first create() call raises."""
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        raise RuntimeError("boom")


def test_ask_events_rolls_back_conversation_on_error():
    fake = RaisingClient()
    with patch.object(llm, "_client", fake):
        with pytest.raises(RuntimeError, match="boom"):
            list(llm.ask_events("hi"))
    assert llm._conversation == [llm._conversation[0]]
    assert llm._conversation[0]["role"] == "system"
    assert llm._session_turns == []
