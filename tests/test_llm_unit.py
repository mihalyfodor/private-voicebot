"""Unit tests for llm.py with a mocked OpenAI client (no oMLX needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import SimpleNamespace
from unittest.mock import patch

import llm


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _completion(msg):
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _chunks(*texts):
    return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))]) for t in texts]


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
    fake = FakeClient([_completion(_msg(content="[happy] Hello there."))])
    with patch.object(llm, "_client", fake):
        reply = llm.ask("hi")
    assert reply == "[happy] Hello there."
    assert llm._conversation[-1] == {"role": "assistant", "content": "[happy] Hello there."}
    assert llm.get_last_tool_calls() == []


def test_tool_round_then_streamed_answer():
    tc = SimpleNamespace(id="call_1", function=SimpleNamespace(name="get_time", arguments="{}"))
    fake = FakeClient([
        _completion(_msg(content=None, tool_calls=[tc])),
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
    tc = SimpleNamespace(id="call_1", function=SimpleNamespace(name="get_time", arguments="{}"))
    fake = FakeClient([_completion(_msg(content=None, tool_calls=[tc])), iter(_chunks("[neutral] Noon."))])
    with patch.object(llm, "_client", fake), patch.object(llm, "run_tool", return_value="12:00:00"):
        assert list(llm.ask_stream("time?")) == ["[neutral] Noon."]
