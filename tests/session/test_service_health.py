"""PRD 41 scenarios 1–4: the service probe, the `services` push, the turn refusal while the LLM is
down, and the client banner / Settings rows (node fixture)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

import asr
import chatbot
import llm
import services
from tests.conftest import spoken, typed


@pytest.fixture(autouse=True)
def fresh_services():
    services.reset()
    yield
    services.reset()


# ---------------------------------------------------------------- scenario 1: probe()


class FakeHTTP:
    """`requests.get` by URL suffix: an exception raises, a dict is the JSON body, None is a 200."""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls = []

    def __call__(self, url, **kw):
        self.calls.append((url, kw))
        for suffix, answer in self.routes.items():
            if url.endswith(suffix):
                if isinstance(answer, Exception):
                    raise answer
                return SimpleNamespace(raise_for_status=lambda: None, json=lambda: answer or {})
        raise requests.ConnectionError(f"unrouted {url}")


def probe_with(routes, backend="remote"):
    with patch.object(requests, "get", FakeHTTP(routes)), patch.object(asr, "BACKEND", backend), \
            patch.object(asr, "URL", "http://127.0.0.1:8020"), \
            patch.object(llm, "OMLX_BASE_URL", "http://localhost:8000/v1"), patch.object(llm, "OMLX_MODEL", "g"):
        return services.probe()


def test_probe_reports_both_ok_and_names_the_ports():
    got = probe_with({"/health": None, "/v1/models": {"data": [{"id": "g"}]}, ":8020/health": None})
    assert got["llm"]["ok"] and got["llm"]["label"] == "oMLX :8000" and "g" in got["llm"]["detail"]
    assert got["asr"]["ok"] and got["asr"]["label"] == "ASR :8020"
    assert got["llm"]["checked"][:2] == "20"


def test_probe_reports_llm_down_and_asr_down():
    got = probe_with({"8000/health": requests.ConnectionError("refused"), ":8020/health": requests.Timeout()})
    assert not got["llm"]["ok"] and "not answering" in got["llm"]["detail"] and ":8000" in got["llm"]["label"]
    assert not got["asr"]["ok"] and ":8020" in got["asr"]["label"]


def test_probe_reports_model_not_loaded():
    got = probe_with({"/health": None, "/v1/models": {"data": [{"id": "other"}]}, ":8020/health": None})
    assert not got["llm"]["ok"] and got["llm"]["detail"] == "model g not loaded"


def test_probe_sends_the_key_to_the_models_list_only():
    fake = FakeHTTP({"/health": None, "/v1/models": {"data": [{"id": "g"}]}})
    with patch.object(requests, "get", fake), patch.object(llm, "OMLX_MODEL", "g"), \
            patch.object(llm, "OMLX_API_KEY", "secret"), patch.object(asr, "BACKEND", "mlx"):
        got = services.probe()
    assert got["asr"] == {**got["asr"], "ok": True, "label": "ASR (in-process)"}
    by_url = {u.split("/")[-1]: kw for u, kw in fake.calls}
    assert "headers" not in by_url["health"] and by_url["models"]["headers"]["Authorization"] == "Bearer secret"


# ---------------------------------------------------------------- scenario 2: the push


def down(name="asr"):
    return {"ok": False, "label": "ASR :8020" if name == "asr" else "oMLX :8000", "detail": "not answering",
            "checked": "2026-08-30T10:00:00"}


def ok(name):
    return {"ok": True, "label": "ASR :8020" if name == "asr" else "oMLX :8000", "detail": "fine",
            "checked": "2026-08-30T10:00:30"}


def test_state_change_pushes_once_and_a_repeat_pushes_nothing():
    pushed = []
    with patch.object(services, "probe", lambda: {"llm": ok("llm"), "asr": down()}):
        services.refresh(pushed.append)
        services.refresh(pushed.append)        # same state, later `checked`: no push
    assert len(pushed) == 1
    assert pushed[0] == {"llm": ok("llm"), "asr": down(), "hint": services.START_HINT}
    with patch.object(services, "probe", lambda: {"llm": ok("llm"), "asr": ok("asr")}):
        services.refresh(pushed.append)
    assert len(pushed) == 2 and pushed[1]["asr"]["ok"]


def test_api_config_carries_the_last_probe():
    with patch.object(services, "probe", lambda: {"llm": ok("llm"), "asr": down()}):
        services.refresh()
    cfg = TestClient(chatbot.app).get("/api/config").json()
    assert cfg["services"] == services.message()
    assert cfg["services"]["asr"]["ok"] is False and cfg["services"]["hint"] == "scripts/voicebot start"


def test_the_change_reaches_the_open_socket_as_a_services_message():
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        with patch.object(services, "probe", lambda: {"llm": down("llm"), "asr": ok("asr")}):
            services.refresh(chatbot._push_services)
        msg = ws.receive_json()
        while msg["type"] != "services":
            msg = ws.receive_json()
    assert msg["llm"]["ok"] is False and msg["hint"] == services.START_HINT


def test_recheck_wakes_the_daemon_without_a_probe_of_its_own():
    assert not services._wake.is_set()
    services.recheck()
    assert services._wake.is_set()


# ---------------------------------------------------------------- scenario 3: turns while the LLM is down


def llm_down():
    return patch.object(services, "_state", {"llm": down("llm"), "asr": ok("asr")})


def test_typed_turn_is_refused_with_the_hint_and_no_thinking_state():
    with llm_down():
        sent, fake = typed("hello")
    types = [m["type"] for m in sent]
    assert "state" not in types or all(m["value"] != "thinking" for m in sent if m["type"] == "state")
    err = next(m for m in sent if m["type"] == "error")
    assert err["scope"] == "turn" and "run" in err["text"] and "scripts/voicebot start" in err["text"]
    assert "oMLX :8000" in err["text"]
    assert sent[-1] == {"type": "state", "value": "idle"}
    assert fake.calls == []                       # the model was never asked


def test_spoken_turn_is_refused_the_same_way():
    with llm_down():
        sent, fake = spoken("hello there")
    assert [m["value"] for m in sent if m["type"] == "state"] == ["processing", "idle"]
    assert next(m for m in sent if m["type"] == "error")["scope"] == "turn"
    assert fake.calls == []


def test_after_the_probe_flips_back_the_next_turn_proceeds():
    with llm_down():
        sent, _ = typed("hello")
    assert any(m["type"] == "error" for m in sent)
    with patch.object(services, "_state", {"llm": ok("llm"), "asr": ok("asr")}):
        sent, fake = typed("hello again", reply="[happy] Back.")
    assert not any(m["type"] == "error" for m in sent)
    assert [m["value"] for m in sent if m["type"] == "state"][0] == "thinking"
    assert next(m for m in sent if m["type"] == "transcript")["text"] == "Back."


def test_a_failed_completion_asks_for_an_immediate_reprobe():
    with patch.object(llm, "ask_events", side_effect=RuntimeError("connection refused")):
        sent, _ = typed("hello")
    assert sent[-1] == {"type": "state", "value": "idle"}
    assert services._wake.is_set()


def test_a_failed_remote_asr_call_asks_for_an_immediate_reprobe():
    with patch.object(chatbot, "asr", SimpleNamespace(transcribe=lambda a, prompt=None: "", last_error="down")):
        from tests.conftest import run_turn
        from session import Session
        import numpy as np
        run_turn(chatbot.handle_utterance, Session(), 1, np.zeros(10, dtype=np.float32))
    assert services._wake.is_set()


# ---------------------------------------------------------------- scenario 4: the client fixture


def test_client_banner_for_a_down_service_with_the_copy_button(r):
    assert r["svc_initial"]["hidden"]
    b = r["svc_asr_down"]
    assert not b["hidden"] and b["cls"] == "degraded"
    assert "voice input offline (ASR :8020)" in b["text"] and "scripts/voicebot start" in b["text"]
    assert b["copy"] == "scripts/voicebot start" and r["svc_copied"] == ["scripts/voicebot start"]


def test_client_banner_hides_when_both_are_ok_and_settings_rows_show_dots_and_times(r):
    assert r["svc_both_ok"]["hidden"]
    rows = {row["name"]: row for row in r["svc_rows_down"]}
    assert rows["LLM"]["dot"] == "svc-dot ok" and rows["LLM"]["copy"] == []
    assert rows["Voice input"]["dot"] == "svc-dot" and rows["Voice input"]["copy"] == ["scripts/voicebot start"]
    assert "checked 10:11:13" in rows["Voice input"]["value"]
    assert all(row["dot"] == "svc-dot ok" for row in r["svc_rows_ok"])


def test_client_backend_down_banner_on_close_1006_but_not_on_a_takeover(r):
    b = r["svc_backend_down"]
    assert not b["hidden"] and b["cls"] == "down"
    assert b["text"] == "Wanko's backend is not running — run: scripts/voicebot start"
    assert r["svc_after_4000"] == "opened in another tab — reload to use here"
