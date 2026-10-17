"""ask_events rolls the turn back when it fails or the consumer closes it mid-stream."""
from unittest.mock import patch

import pytest

from agent import context as a_context
from agent import conversation
from agent import turn as a_turn
from core import llm as core_llm
from tests.conftest import stream_chunks

pytestmark = pytest.mark.usefixtures("fresh_llm")


class _Client:
    """A streamed reply that yields its chunks, then raises (or lets the consumer stop)."""

    def __init__(self, chunks, boom=None):
        self.chat = type("C", (), {"completions": type("P", (), {"create": self._create})()})()
        self._chunks, self._boom = chunks, boom

    def _create(self, **kw):
        def gen():
            yield from self._chunks
            if self._boom:
                raise self._boom
        return gen()


def _fill_history():
    for i in range(10):
        conversation._conversation.append({"role": "user", "content": f"old question {i} " * 20})
        conversation._conversation.append({"role": "assistant", "content": f"old reply {i} " * 20})


def test_a_stream_error_after_a_trim_leaves_no_orphaned_user_message(monkeypatch):
    monkeypatch.setattr(a_context, "CONTEXT_BUDGET", 300)
    monkeypatch.setattr(a_context, "model_context_len", lambda: None)
    _fill_history()
    before = len(conversation._conversation)
    client = _Client(stream_chunks("[happy] part one", " part two"), boom=RuntimeError("model died"))
    with patch.object(core_llm, "_client", client):
        with pytest.raises(RuntimeError):
            list(a_turn.ask_events("a brand new question", origin="typed"))
    assert len(conversation._conversation) < before            # the trim really ran
    assert not any("a brand new question" in str(m.get("content")) for m in conversation._conversation)
    assert conversation._session_turns == []


def test_closing_the_generator_mid_stream_rolls_the_turn_back():
    client = _Client(stream_chunks("[happy] part one", " part two", " part three"))
    before = list(conversation._conversation)
    with patch.object(core_llm, "_client", client):
        gen = a_turn.ask_events("a question to abandon", origin="typed")
        assert next(gen)[0] == "delta"
        gen.close()
    assert conversation._conversation == before
    assert conversation._session_turns == []
