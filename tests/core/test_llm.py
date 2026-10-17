"""PRD 50 scenario 7: one model-call chokepoint (`core/llm.py`), one fake patched at `core.llm._client`."""
from unittest.mock import patch

import chatbot
from server import speech
import memory
import sources
from core import llm as core_llm
from session import Session
from tests.conftest import fake_model, fake_tts, run_turn, summary_json, trace_lines, vault  # noqa: F401 — fixture


def test_complete_is_one_plain_call_at_temperature_zero():
    with fake_model("A") as fake:
        assert core_llm.complete("sys", "u", 10) == "A"
    (call,) = fake.calls
    assert call["temperature"] == 0 and call["max_tokens"] == 10 and call["model"] == core_llm.OMLX_MODEL
    assert call["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    assert "stream" not in call
    assert trace_lines(kind="llm_call") == []                       # no purpose: unspanned, like memory's calls


def test_complete_with_a_purpose_runs_under_an_llm_call_span():
    with fake_model("A"):
        assert core_llm.complete("sys", "u", 10, purpose="ingest", model="other") == "A"
    (span,) = trace_lines(kind="llm_call")
    assert span["purpose"] == "ingest" and span["model"] == "other" and span["max_tokens"] == 10


def test_an_ingest_a_memory_save_and_a_typed_turn_share_one_fake(vault):
    with fake_model(summary_json("Falcon"), "[]",            # ingest: summary, page ops
                    "[]", "[]",                              # memory save: profile ops, wiki page ops
                    "[neutral] Sure.") as fake, \
            patch.object(speech, "synthesize", fake_tts):   # the typed turn's streamed reply
        filed = sources.ingest_note("Falcon is amber.")
        assert memory.save([{"role": "user", "content": "Anna is a colleague"}]) == []
        sent = run_turn(chatbot.handle_text, Session(), 1, "hi")

    assert filed["path"] == "sources/falcon"
    assert len(fake.calls) == 5 and fake._replies == []
    assert [bool(c.get("stream")) for c in fake.calls] == [False] * 4 + [True]
    assert "Falcon is amber." in fake.calls[0]["messages"][1]["content"]
    assert "Anna is a colleague" in fake.calls[2]["messages"][1]["content"]
    assert "Anna is a colleague" in fake.calls[3]["messages"][1]["content"]
    assert fake.calls[4]["messages"][-1] == {"role": "user", "content": "hi"}
    assert {c["model"] for c in fake.calls} == {core_llm.OMLX_MODEL}
    assert [s["purpose"] for s in trace_lines(kind="llm_call")] == ["ingest", "ingest", "reply"]
    assert any(m.get("type") == "speech_end" for m in sent)
