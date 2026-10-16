"""PRD 49 scenarios 1, 2, 3, 5, 6, 8: the event trace of a turn (docs/49-event-trace.md).

Scenario 4 is in tests/sources/test_sources.py, 7 in test_trace_profile.py, 9 in test_trace_report.py.
"""
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import chatbot
import llm
from core import llm as core_llm
from core import trace
from session import Session
from tests.conftest import (SENTINEL, RawStreamClient, StreamClient, ask_turn, fake_asr, fake_tts, run_turn, stream_chunks,
                            tool_call_chunk, trace_lines, trace_text)

pytestmark = pytest.mark.usefixtures("fresh_llm")

QUESTION = f"tell me about the {SENTINEL} project plans for next week"
REPLY = f"[neutral] The {SENTINEL} plans look fine to me."


def _typed(turn: int, text: str, client):
    with patch.object(core_llm, "_client", client), patch.object(chatbot, "synthesize", fake_tts):
        return run_turn(chatbot.handle_text, Session(), turn, text)


def _spoken(turn: int, text: str, client):
    with patch.object(core_llm, "_client", client), patch.object(chatbot, "synthesize", fake_tts), \
            patch.object(chatbot, "asr", fake_asr(text)):
        return run_turn(chatbot.handle_utterance, Session(), turn, np.zeros(10, dtype=np.float32))


# --------------------------------------------------------------- 1. a turn's events share its ids


def test_typed_and_spoken_turns_share_ids_and_hold_no_text(vault):
    _typed(1, QUESTION, StreamClient(REPLY))
    _spoken(1, QUESTION, StreamClient(REPLY))        # a new Session: a reconnect restarts the turn ids
    starts = trace_lines(kind="turn_start")
    assert [s["origin"] for s in starts] == ["typed", "spoken"] and {s["turn_id"] for s in starts} == {1}
    runs = [s["run_id"] for s in starts]
    assert runs[0] != runs[1]                         # unique in the process regardless of reconnects
    for run in runs:
        rows = [r for r in trace_lines() if r.get("run_id") == run]
        kinds = [r["kind"] for r in rows]
        assert {"turn_start", "recall", "llm_call", "turn_end"} <= set(kinds), kinds
        assert kinds.count("turn_start") == kinds.count("turn_end") == 1
        assert {r["turn_id"] for r in rows} == {1}
        assert rows[kinds.index("turn_start")]["text_len"] == len(QUESTION)
        end = rows[kinds.index("turn_end")]
        assert end["outcome"] == "ok" and end["reply_len"] > 0
    text = trace_text()
    assert SENTINEL not in text and "plans look fine" not in text


# --------------------------------------------------------------- 2. tool_call without its result


def test_a_tool_call_records_its_size_not_its_result(vault, capsys, monkeypatch):
    monkeypatch.setattr(llm, "run_tool", lambda name, args: f"RESULT {SENTINEL} body of the page")
    client = RawStreamClient([
        [tool_call_chunk(id="c1", name="search_wiki", arguments='{"query": "falcon"}')],
        stream_chunks("[neutral] Done."),
    ])
    with trace.turn(7):
        with patch.object(core_llm, "_client", client):
            list(llm.ask_events("search the wiki for falcon", origin="typed"))
    calls = trace_lines(kind="tool_call")
    assert len(calls) == 1
    call = calls[0]
    assert call["name"] == "search_wiki" and call["ok"] is True and call["result_len"] > 0
    assert call["arg_keys"] == ["query"] and call["turn_id"] == 7
    assert SENTINEL not in trace_text()
    assert SENTINEL not in capsys.readouterr().out


# --------------------------------------------------------------- 3. llm_call timings, tokens, span


def _usage_chunk(prompt: int, completion: int):
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


def test_llm_call_reads_usage_from_the_last_chunk(vault):
    client = RawStreamClient([stream_chunks("[neutral] ", "Hi there.") + [_usage_chunk(12, 3)]])
    with trace.turn(3) as root, patch.object(core_llm, "_client", client):
        list(llm.ask_events("hello", origin="typed"))
    assert client.calls[0]["stream_options"] == {"include_usage": True}
    (call,) = trace_lines(kind="llm_call")
    assert call["purpose"] == "reply" and call["model"] == core_llm.OMLX_MODEL
    assert (call["tokens_in"], call["tokens_out"]) == (12, 3)
    assert 0 <= call["ttft_ms"] <= call["ms"]
    assert call["span_id"] and call["parent_id"] == root
    assert call["duration_ms"] >= 0 and call["status"] == "ok" and call["start"]


def test_llm_call_without_usage_leaves_tokens_null(vault):
    with trace.turn(4):
        events, _ = ask_turn("hello", "[neutral] Hi.")
    (call,) = trace_lines(kind="llm_call")
    assert call["tokens_in"] is None and call["tokens_out"] is None
    assert call["ttft_ms"] is not None and call["status"] == "ok"


# --------------------------------------------------------------- 5. a turn that raises mid-stream


def test_a_turn_raising_mid_stream_is_an_error_event_and_an_error_message(vault):
    def broken():
        yield from stream_chunks("[neutral] Half a ")
        raise RuntimeError("stream dropped")

    sent = _typed(5, "hello there friend", RawStreamClient([broken()]))
    errors = trace_lines(kind="error")
    assert len(errors) == 1 and errors[0]["scope"] == "turn" and errors[0]["turn_id"] == 5
    assert errors[0]["exc_type"] == "RuntimeError" and errors[0]["traceback"]
    (end,) = trace_lines(kind="turn_end")
    assert end["outcome"] == "error"
    assert any(m["type"] == "error" and m["scope"] == "turn" for m in sent)
    assert [m for m in sent if m["type"] == "state"][-1]["value"] == "idle"


# --------------------------------------------------------------- 6. rotation, never raises


def test_rotation_keeps_at_most_five_backups(monkeypatch):
    monkeypatch.setattr(trace, "MAX_BYTES", 512 * 1024)
    pad = "x" * 190
    for i in range(30_000):                          # ~11 MB of events
        trace.event("batch_file", rel=pad, state="filed", seconds=i)
    folder = os.path.dirname(trace.paths()["events"])
    names = sorted(n for n in os.listdir(folder) if n.startswith("events.jsonl"))
    assert names == ["events.jsonl"] + [f"events.jsonl.{i}" for i in range(1, 6)]


def test_an_unwritable_dir_does_not_raise(tmp_path, monkeypatch, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("not a dir")
    monkeypatch.setattr(trace, "TRACE_DIR", str(blocker / "logs"))
    trace.reset()
    trace.event("turn_start", origin="typed", text_len=3, words=1)
    trace.event("turn_end", ms=1, outcome="ok")
    assert capsys.readouterr().err.count("[Trace] write failed") == 1      # rate-limited


def test_a_free_text_field_never_reaches_events(vault):
    trace.event("turn_start", origin="typed", text=SENTINEL, reply=SENTINEL, words=1,
                where="x" * 500, title="t" * 500)
    (row,) = trace_lines(kind="turn_start")
    assert "text" not in row and "reply" not in row and row["where"] == "other"
    assert row["title"] == "t" * trace.CAP


# --------------------------------------------------------------- 8. dev verbose file


def test_dev_turn_writes_prompt_and_turn_text_to_verbose_only(vault):
    assert trace.profile() == "dev"
    _typed(8, QUESTION, StreamClient(REPLY))
    rows = trace_lines("verbose")
    kinds = {r["kind"] for r in rows}
    assert {"prompt", "turn_text"} <= kinds
    assert all(r["turn_id"] == 8 for r in rows if r["kind"] in ("prompt", "turn_text"))
    (said,) = [r for r in rows if r["kind"] == "turn_text"]
    assert SENTINEL in said["user"] and SENTINEL in said["reply"]
    assert SENTINEL in trace_text("verbose") and SENTINEL not in trace_text()


# --------------------------------------------------------------- review: no free text in prod, even short


def _raising_tool(name, args):
    raise ValueError(f"user said {SENTINEL}")


def _tool_turn(turn: int):
    client = RawStreamClient([[tool_call_chunk(id="c1", name="search_wiki", arguments='{"query": "x"}')],
                              stream_chunks("[neutral] Done.")])
    return _typed(turn, "search the wiki for x please", client)


def _turn_raising(turn: int):
    def broken():
        yield from stream_chunks("[neutral] Half ")
        raise ValueError(f"user said {SENTINEL}")
    return _typed(turn, "hello there friend", RawStreamClient([broken()]))


def test_an_exception_text_never_reaches_prod_events(vault, monkeypatch, prod_profile):
    monkeypatch.setattr(llm, "run_tool", _raising_tool)
    _tool_turn(1)
    _turn_raising(2)
    errors = trace_lines(kind="error")
    assert len(errors) == 2 and all(e["message"] == e["code"] == "ValueError" for e in errors)
    assert [r["status"] for r in trace_lines(kind="tool_call")] == ["error"]
    assert SENTINEL not in trace_text()
    assert not os.path.exists(trace.paths()["verbose"])


def test_an_exception_text_goes_to_dev_verbose_only(vault, monkeypatch):
    monkeypatch.setattr(llm, "run_tool", _raising_tool)
    _tool_turn(1)
    _turn_raising(2)
    assert SENTINEL not in trace_text()
    texts = [r for r in trace_lines("verbose") if r["kind"] == "error_text"]
    assert len(texts) == 2 and all(SENTINEL in r["message"] for r in texts)


def test_vocabulary_fields_outside_their_values_become_other(vault):
    trace.event("batch_file", rel="a.pdf", state=f"filed {SENTINEL}", seconds=1)
    trace.event("error", scope="turn", where=SENTINEL, exc_type="X", message="X", code="X")
    trace.event("tool_call", name=SENTINEL, arg_keys=["query", f"q {SENTINEL}"], ok=True)
    trace.event("llm_call", purpose="reply", fallback_reason=SENTINEL)
    assert trace_lines(kind="batch_file")[0]["state"] == "other"
    assert trace_lines(kind="error")[0]["where"] == "other"
    call = trace_lines(kind="tool_call")[0]
    assert call["name"] == "other" and call["arg_keys"] == ["query", "other"]
    assert trace_lines(kind="llm_call")[0]["fallback_reason"] == "other"
    assert SENTINEL not in trace_text()
    assert SENTINEL in trace_text("verbose")                 # dev keeps the original for the agent


# --------------------------------------------------------------- review: a swallowed handler error is still an error


def test_a_briefing_whose_completion_raises_ends_the_turn_in_error(vault, monkeypatch):
    import services
    import wiki

    def boom(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(wiki, "changes_since", lambda since: {"pages": [1]})
    monkeypatch.setattr(wiki, "digest_empty", lambda d: False)
    monkeypatch.setattr(services, "llm_ok", lambda: True)
    monkeypatch.setattr(llm, "briefing", boom)
    sent = _typed(9, "what's new?", StreamClient())
    (end,) = trace_lines(kind="turn_end")
    assert end["outcome"] == "error"
    (err,) = trace_lines(kind="error")
    assert err["scope"] == "turn" and err["where"] == "briefing" and err["turn_id"] == 9
    assert [m for m in sent if m["type"] == "state"][-1]["value"] == "idle"


# --------------------------------------------------------------- exception sweep


def test_an_asr_failure_is_an_error_event_and_an_error_message(vault, prod_profile):
    def broken(audio, prompt=None):
        raise RuntimeError(f"asr down {SENTINEL}")

    with patch.object(chatbot, "asr", SimpleNamespace(transcribe=broken)):
        sent = run_turn(chatbot.handle_utterance, Session(), 6, np.zeros(10, dtype=np.float32))
    (err,) = trace_lines(kind="error")
    assert err["scope"] == "turn" and err["where"] == "asr" and err["exc_type"] == "RuntimeError"
    assert any(m["type"] == "error" and m["scope"] == "turn" and SENTINEL not in m["text"] for m in sent)
    assert [m for m in sent if m["type"] == "state"][-1]["value"] == "idle"
    assert SENTINEL not in trace_text()


def test_a_failed_background_memory_save_is_a_non_fatal_error_event(vault, prod_profile, monkeypatch):
    def broken():
        raise RuntimeError(f"model down {SENTINEL}")

    monkeypatch.setattr(llm, "save_memory", broken)
    monkeypatch.setattr(chatbot, "_push_saving", lambda on: None)
    monkeypatch.setattr(chatbot, "push_memory", lambda **k: None)
    chatbot._periodic_memory(3)
    (err,) = trace_lines(kind="error")
    assert err["where"] == "memory_save" and err["fatal_for"] == "none" and err["exc_type"] == "RuntimeError"
    assert SENTINEL not in trace_text()


def test_a_background_error_during_a_turn_does_not_fail_it(vault):
    with trace.turn(7):
        try:
            raise RuntimeError("memory down")
        except RuntimeError as exc:
            trace.error("memory", "memory_save", exc, fatal_for="none")
            trace.error("turn", "memory_save", exc, fatal_for="none")
        assert not trace.turn_failed()


@pytest.mark.parametrize("profile, shown", [("prod", False), ("dev", True)])
def test_stdout_tracebacks_carry_the_exception_text_only_in_dev(vault, monkeypatch, capsys, profile, shown):
    monkeypatch.setenv("TRACE_PROFILE", profile)
    trace.reset()
    try:
        raise RuntimeError(SENTINEL)
    except RuntimeError as exc:
        chatbot._turn_failed(lambda msg: None, "respond", exc, "The reply failed.")
    out = capsys.readouterr().out
    assert "RuntimeError" in out and (SENTINEL in out) is shown
