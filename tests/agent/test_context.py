"""Tests for context budget, history trimming, and max_tokens (scenarios 1, 2, 3, 6)."""
from unittest.mock import patch


from types import SimpleNamespace

from agent import context as a_context
from agent import conversation
from agent import turn as a_turn
from agent import conversation
from agent import turn as a_turn
from agent import context as a_context
from agent import prompt
from tests.conftest import FakeClient, stream_chunks as _chunks, tool_call_chunk as _tc_chunk
from core import llm as core_llm


def setup_function():
    conversation.reset()
    a_context._model_context_len_cache = None
    a_context._model_context_len_warned = False
    a_context._model_context_len_failed_at = None


# --- scenario 1: trimming drops oldest first, keeps system + latest user ---

def test_trim_history_drops_oldest_non_system_turns(monkeypatch):
    monkeypatch.setattr(a_context, "CONTEXT_BUDGET", 200)
    monkeypatch.setattr(prompt, "MAX_TOKENS", 50)
    monkeypatch.setattr(a_context, "model_context_len", lambda: None)

    for i in range(30):
        conversation._conversation.append({"role": "user", "content": f"turn {i}" * 5})
        conversation._conversation.append({"role": "assistant", "content": f"reply {i}" * 5})

    latest_user = {"role": "user", "content": "the very latest question"}
    conversation._conversation.append(latest_user)

    a_context.trim_history()

    assert conversation._conversation[0]["role"] == "system"
    assert conversation._conversation[-1] == latest_user
    # trimming stops once only system + latest user remain, even if still over budget
    assert len(conversation._conversation) == 2
    # oldest turns should be gone
    assert not any(t.get("content", "").startswith("turn 0") for t in conversation._conversation)


def test_trim_history_never_orphans_tool_result(monkeypatch):
    monkeypatch.setattr(a_context, "CONTEXT_BUDGET", 60)
    monkeypatch.setattr(prompt, "MAX_TOKENS", 10)
    monkeypatch.setattr(a_context, "model_context_len", lambda: None)

    conversation._conversation = [{"role": "system", "content": "sys" * 50}]
    conversation._conversation.append({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}],
    })
    conversation._conversation.append({"role": "tool", "tool_call_id": "1", "content": "12:00"})
    conversation._conversation.append({"role": "assistant", "content": "It is noon" * 20})
    conversation._conversation.append({"role": "user", "content": "thanks" * 20})

    a_context.trim_history()

    roles = [m["role"] for m in conversation._conversation]
    # no "tool" turn should exist without an immediately preceding assistant tool_calls turn
    for i, role in enumerate(roles):
        if role == "tool":
            assert roles[i - 1] == "assistant"
    assert roles[0] == "system"
    assert roles[-1] == "user"


def test_trim_history_no_trim_when_under_budget(monkeypatch, capsys):
    monkeypatch.setattr(a_context, "CONTEXT_BUDGET", 16000)
    monkeypatch.setattr(prompt, "MAX_TOKENS", 400)
    monkeypatch.setattr(a_context, "model_context_len", lambda: None)
    conversation._conversation.append({"role": "user", "content": "hi"})
    a_context.trim_history()
    assert "trimmed" not in capsys.readouterr().out


# --- scenario 3: max_tokens present on every create() call ---

def test_max_tokens_passed_on_every_create_call(monkeypatch):
    monkeypatch.setattr(prompt, "MAX_TOKENS", 0)
    monkeypatch.setattr(prompt, "_verbosity", "long")
    fake = FakeClient(*[
        [_tc_chunk(id="call_1", name="get_time", arguments="{}", finish_reason="tool_calls")],
        _chunks("[neutral] Noon."),
    ])
    with patch.object(core_llm, "_client", fake), patch.object(a_turn, "run_tool", return_value="12:00:00"):
        list(a_turn.ask_events("what time is it?"))
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call["max_tokens"] == 800


def test_max_tokens_env_override(monkeypatch):
    monkeypatch.setattr(prompt, "MAX_TOKENS", 123)
    fake = FakeClient(*[_chunks("[happy] hi")])
    with patch.object(core_llm, "_client", fake):
        a_turn.ask("hi")
    assert fake.calls[0]["max_tokens"] == 123


# --- scenario 6: /v1/models unavailable → fallback, warning, no crash ---

def test_model_context_len_failure_falls_back(monkeypatch, capsys):
    monkeypatch.setattr(a_context, "CONTEXT_BUDGET", 16000)

    class BoomClient:
        class models:
            @staticmethod
            def list():
                raise RuntimeError("connection refused")

    with patch.object(core_llm, "_client", BoomClient()):
        result = a_context.model_context_len()
        assert result is None
        assert a_context.context_budget() == 16000
    out = capsys.readouterr().out
    assert "warning" in out.lower()


def test_model_context_len_caps_budget(monkeypatch):
    monkeypatch.setattr(a_context, "CONTEXT_BUDGET", 16000)
    model = SimpleNamespace(id=core_llm.OMLX_MODEL, max_model_len=4000)

    class OkClient:
        class models:
            @staticmethod
            def list():
                return [model]

    with patch.object(core_llm, "_client", OkClient()):
        assert a_context.model_context_len() == 4000
        assert a_context.context_budget() == 4000


# --- memory: the profile block stays inside its token budget ---

def test_memory_load_caps_the_profile_block(monkeypatch, tmp_path):
    import json
    from core import util
    from memory import identity
    from memory import store
    from memory import render
    from memory import store
    path = tmp_path / "memory.json"
    profile = store._empty_profile()
    profile["identity"] = {"name": "Mihaly"}
    profile["people"] = [{"name": f"person {i}", "note": "x" * 20} for i in range(50)]
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(store, "MEMORY_PATH", str(path))
    monkeypatch.setattr(render, "BUDGET_TOKENS", 100)

    result = identity.load("SYSTEM")
    assert result.startswith("SYSTEM")
    assert "name: Mihaly" in result
    assert "person 0" in result and "person 49" not in result
    block = result.split("<user_profile>\n")[1].split("\n</user_profile>")[0]
    assert util.tokens_est(block) <= 100


def test_model_context_len_failure_is_not_retried_within_the_window(monkeypatch):
    calls = []

    class BoomClient:
        class models:
            @staticmethod
            def list():
                calls.append(1)
                raise RuntimeError("connection refused")

    monkeypatch.setattr(a_context, "_model_context_len_failed_at", None)
    with patch.object(core_llm, "_client", BoomClient()):
        assert a_context.model_context_len() is None
        assert a_context.model_context_len() is None
        assert len(calls) == 1
        a_context._model_context_len_failed_at -= a_context._MODEL_CONTEXT_RETRY_S + 1
        assert a_context.model_context_len() is None
        assert len(calls) == 2


def test_trim_then_meter_does_not_exceed_100_percent(monkeypatch):
    """The trim budget subtracts the tool schemas, so the meter never reads over 100 % after a trim."""
    monkeypatch.setattr(a_context, "CONTEXT_BUDGET", 4000)
    monkeypatch.setattr(prompt, "MAX_TOKENS", 50)
    monkeypatch.setattr(a_context, "model_context_len", lambda: None)
    for i in range(60):
        conversation._conversation.append({"role": "user", "content": f"question {i} " * 10})
        conversation._conversation.append({"role": "assistant", "content": f"answer {i} " * 10})
    conversation._conversation.append({"role": "user", "content": "the latest question"})
    assert a_context.context_usage()["pct"] > 100

    a_context.trim_history()

    assert len(conversation._conversation) > 2
    assert a_context.context_usage()["pct"] <= 100


# --- PRD 57: stub old tool results before dropping turns ---

def _tool_turn(call_id: str, name: str, body: str) -> list:
    return [
        {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": body},
        {"role": "assistant", "content": "ok"},
    ]


def _session(monkeypatch, budget: int, bodies: dict) -> None:
    """system, then one user turn per body with its tool call (old to new), then a final user message."""
    monkeypatch.setattr(a_context, "CONTEXT_BUDGET", budget)
    monkeypatch.setattr(prompt, "MAX_TOKENS", 10)
    monkeypatch.setattr(a_context, "model_context_len", lambda: None)
    monkeypatch.setattr(a_context, "message_budget", lambda: budget)
    conv = [{"role": "system", "content": "sys"}]
    for i, (name, body) in enumerate(bodies.items()):
        conv.append({"role": "user", "content": f"question {i}"})
        conv += _tool_turn(str(i), name, body)
    conv.append({"role": "user", "content": "latest"})
    conversation._conversation = conv


def _stub(tool: str) -> str:
    return f"[{tool} result removed to save room \u2014 call {tool} again if needed]"


def _tool_contents() -> list:
    return [m["content"] for m in conversation._conversation if m["role"] == "tool"]


def test_trim_stubs_old_tool_result_before_dropping_turns(monkeypatch):
    _session(monkeypatch, 500, {"read_page": "x" * 4000, "get_time": "12:00", "get_weather": "sunny"})
    assert a_context.trim_history() == 0
    assert _tool_contents()[0] == _stub("read_page")
    assert sum(1 for m in conversation._conversation if m["role"] == "user") == 4
    assert a_context._last_trimmed == 0


def test_trim_keeps_tool_results_in_the_last_two_turns(monkeypatch):
    _session(monkeypatch, 1500, {"read_page": "x" * 4000, "get_news": "y" * 2000, "get_wiki": "z" * 2000})
    a_context.trim_history()
    assert _tool_contents() == [_stub("read_page"), _stub("get_news"), "z" * 2000]


def test_trim_stub_floor_is_400_chars(monkeypatch):
    _session(monkeypatch, 285, {"a": "x" * 399, "b": "y" * 401, "c": "ok", "d": "ok"})
    a_context.trim_history()
    contents = _tool_contents()
    assert "x" * 399 in contents and "y" * 401 not in contents and _stub("b") in contents


def test_trim_stub_keeps_tool_call_pairing(monkeypatch):
    _session(monkeypatch, 500, {"read_page": "x" * 4000, "get_time": "12:00", "get_weather": "sunny"})
    a_context.trim_history()
    conv = conversation._conversation
    i = next(i for i, m in enumerate(conv) if m["role"] == "tool")
    assert conv[i - 1]["tool_calls"][0]["id"] == conv[i]["tool_call_id"] == "0"
    assert conv[i - 1]["tool_calls"][0]["function"]["name"] == "read_page" and "read_page" in conv[i]["content"]


def test_trim_under_budget_leaves_tool_results(monkeypatch):
    _session(monkeypatch, 16000, {"read_page": "x" * 4000, "get_time": "12:00", "get_weather": "sunny"})
    a_context.trim_history()
    assert _tool_contents()[0] == "x" * 4000


def test_trim_drops_turns_only_if_stubs_not_enough(monkeypatch):
    _session(monkeypatch, 120, {"read_page": "x" * 4000, "get_news": "y" * 4000, "get_time": "12:00", "get_weather": "sunny"})
    assert a_context.trim_history() > 0
    assert "x" * 4000 not in _tool_contents() and "y" * 4000 not in _tool_contents()
    assert all(len(c) < 400 for c in _tool_contents())
    assert conversation._conversation[-1]["content"] == "latest"
