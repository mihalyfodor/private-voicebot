"""Tests for context budget, history trimming, and max_tokens (scenarios 1, 2, 3, 6)."""
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
    llm._model_context_len_cache = None
    llm._model_context_len_warned = False


# --- scenario 1: trimming drops oldest first, keeps system + latest user ---

def test_trim_history_drops_oldest_non_system_turns(monkeypatch):
    monkeypatch.setattr(llm, "CONTEXT_BUDGET", 200)
    monkeypatch.setattr(llm, "MAX_TOKENS", 50)
    monkeypatch.setattr(llm, "model_context_len", lambda: None)

    for i in range(30):
        llm._conversation.append({"role": "user", "content": f"turn {i}" * 5})
        llm._conversation.append({"role": "assistant", "content": f"reply {i}" * 5})

    latest_user = {"role": "user", "content": "the very latest question"}
    llm._conversation.append(latest_user)

    llm.trim_history()

    assert llm._conversation[0]["role"] == "system"
    assert llm._conversation[-1] == latest_user
    # trimming stops once only system + latest user remain, even if still over budget
    assert len(llm._conversation) == 2
    # oldest turns should be gone
    assert not any(t.get("content", "").startswith("turn 0") for t in llm._conversation)


def test_trim_history_never_orphans_tool_result(monkeypatch):
    monkeypatch.setattr(llm, "CONTEXT_BUDGET", 60)
    monkeypatch.setattr(llm, "MAX_TOKENS", 10)
    monkeypatch.setattr(llm, "model_context_len", lambda: None)

    llm._conversation = [{"role": "system", "content": "sys" * 50}]
    llm._conversation.append({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}],
    })
    llm._conversation.append({"role": "tool", "tool_call_id": "1", "content": "12:00"})
    llm._conversation.append({"role": "assistant", "content": "It is noon" * 20})
    llm._conversation.append({"role": "user", "content": "thanks" * 20})

    llm.trim_history()

    roles = [m["role"] for m in llm._conversation]
    # no "tool" turn should exist without an immediately preceding assistant tool_calls turn
    for i, role in enumerate(roles):
        if role == "tool":
            assert roles[i - 1] == "assistant"
    assert roles[0] == "system"
    assert roles[-1] == "user"


def test_trim_history_no_trim_when_under_budget(monkeypatch, capsys):
    monkeypatch.setattr(llm, "CONTEXT_BUDGET", 16000)
    monkeypatch.setattr(llm, "MAX_TOKENS", 400)
    monkeypatch.setattr(llm, "model_context_len", lambda: None)
    llm._conversation.append({"role": "user", "content": "hi"})
    llm.trim_history()
    assert "trimmed" not in capsys.readouterr().out


# --- scenario 3: max_tokens present on every create() call ---

def test_max_tokens_passed_on_every_create_call(monkeypatch):
    monkeypatch.setattr(llm, "MAX_TOKENS", 0)
    monkeypatch.setattr(llm, "_verbosity", "long")
    fake = FakeClient([
        iter([_tc_chunk(id="call_1", name="get_time", arguments="{}", finish_reason="tool_calls")]),
        iter(_chunks("[neutral] Noon.")),
    ])
    with patch.object(llm, "_client", fake), patch.object(llm, "run_tool", return_value="12:00:00"):
        list(llm.ask_events("what time is it?"))
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call["max_tokens"] == 800


def test_max_tokens_env_override(monkeypatch):
    monkeypatch.setattr(llm, "MAX_TOKENS", 123)
    fake = FakeClient([iter(_chunks("[happy] hi"))])
    with patch.object(llm, "_client", fake):
        llm.ask("hi")
    assert fake.calls[0]["max_tokens"] == 123


# --- scenario 6: /v1/models unavailable → fallback, warning, no crash ---

def test_model_context_len_failure_falls_back(monkeypatch, capsys):
    monkeypatch.setattr(llm, "CONTEXT_BUDGET", 16000)

    class BoomClient:
        class models:
            @staticmethod
            def list():
                raise RuntimeError("connection refused")

    with patch.object(llm, "client", lambda: BoomClient()):
        result = llm.model_context_len()
        assert result is None
        assert llm.context_budget() == 16000
    out = capsys.readouterr().out
    assert "warning" in out.lower()


def test_model_context_len_caps_budget(monkeypatch):
    monkeypatch.setattr(llm, "CONTEXT_BUDGET", 16000)
    model = SimpleNamespace(id=llm.OMLX_MODEL, max_model_len=4000)

    class OkClient:
        class models:
            @staticmethod
            def list():
                return [model]

    with patch.object(llm, "client", lambda: OkClient()):
        assert llm.model_context_len() == 4000
        assert llm.context_budget() == 4000


# --- memory: the profile block stays inside its token budget ---

def test_memory_load_caps_the_profile_block(monkeypatch, tmp_path):
    import json
    import memory
    path = tmp_path / "memory.json"
    profile = memory._empty_profile()
    profile["identity"] = {"name": "Mihaly"}
    profile["people"] = [{"name": f"person {i}", "note": "x" * 20} for i in range(50)]
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(memory, "MEMORY_PATH", str(path))
    monkeypatch.setattr(memory, "BUDGET_TOKENS", 100)

    result = memory.load("SYSTEM")
    assert result.startswith("SYSTEM")
    assert "name: Mihaly" in result
    assert "person 0" in result and "person 49" not in result
    block = result.split("<user_profile>\n")[1].split("\n</user_profile>")[0]
    assert memory.tokens_est(block) <= 100
