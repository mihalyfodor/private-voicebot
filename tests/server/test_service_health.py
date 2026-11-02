"""PRD 41 scenarios 1–4: the service probe, the `services` push, the turn refusal while the LLM is
down, and the client banner / Settings rows (node fixture)."""
from core import settings
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

from server import asr
from server import app
from server import panels
from server import turns as server_turns
from agent import turn as a_turn
from server import services
from tests.conftest import spoken, typed
from core import llm as core_llm


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
            patch.object(core_llm, "OMLX_BASE_URL", "http://localhost:8000/v1"), patch.object(core_llm, "OMLX_MODEL", "g"):
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
    with patch.object(requests, "get", fake), patch.object(core_llm, "OMLX_MODEL", "g"), \
            patch.object(core_llm, "OMLX_API_KEY", "secret"), patch.object(asr, "BACKEND", "mlx"):
        got = services.probe()
    assert got["asr"] == {**got["asr"], "ok": True, "label": "ASR (in-process)"}
    by_url = {u.split("/")[-1]: kw for u, kw in fake.calls}
    assert "headers" not in by_url["health"] and by_url["models"]["headers"]["Authorization"] == "Bearer secret"


# ---------------------------------------------------------------- PRD 47b scenario 6: the push service row


def test_remarkable_row_ok_down_and_absent():
    from ingest import remarkable
    with patch.object(remarkable, "URL", "http://127.0.0.1:8021"):
        got = probe_with({":8021/health": None, "/health": None, "/v1/models": {"data": [{"id": "g"}]}})
        assert got["remarkable"]["ok"] and got["remarkable"]["label"] == "reMarkable push :8021"
        assert got["remarkable"]["detail"] == "answering"
        got = probe_with({":8021/health": requests.ConnectionError("refused")})
        assert not got["remarkable"]["ok"] and ":8021" in got["remarkable"]["label"]
        assert got["remarkable"]["hint"] == ".venv/bin/python scripts/remarkable_server.py"
    with patch.object(remarkable, "URL", ""):
        assert "remarkable" not in probe_with({"/health": None, "/v1/models": {"data": [{"id": "g"}]}})


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
    cfg = TestClient(app.app).get("/api/config").json()
    assert cfg["services"] == services.message()
    assert cfg["services"]["asr"]["ok"] is False and cfg["services"]["hint"] == "scripts/voicebot start"


def test_the_change_reaches_the_open_socket_as_a_services_message():
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        with patch.object(services, "probe", lambda: {"llm": down("llm"), "asr": ok("asr")}):
            services.refresh(panels._push_services)
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
    with patch.object(a_turn, "ask_events", side_effect=RuntimeError("connection refused")):
        sent, _ = typed("hello")
    assert sent[-1] == {"type": "state", "value": "idle"}
    assert services._wake.is_set()


def test_a_failed_remote_asr_call_asks_for_an_immediate_reprobe():
    with patch.object(server_turns, "asr", SimpleNamespace(transcribe=lambda a, prompt=None: "", last_error="down")):
        from tests.conftest import run_turn
        from server.session import Session
        import numpy as np
        run_turn(server_turns.handle_utterance, Session(), 1, np.zeros(10, dtype=np.float32))
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


def test_client_remarkable_row_only_shows_when_the_services_message_carries_it(r):
    without = {row["name"] for row in r["svc_rows_without_remarkable"]}
    assert {"LLM", "Voice input"} <= without and "reMarkable push" not in without
    with_rm = {row["name"]: row for row in r["svc_rows_with_remarkable"]}
    assert {"LLM", "Voice input"} <= set(with_rm)
    rm = with_rm["reMarkable push"]
    assert rm["dot"] == "svc-dot ok" and "ready" in rm["value"]


def test_client_backend_down_banner_on_close_1006_but_not_on_a_takeover(r):
    b = r["svc_backend_down"]
    assert not b["hidden"] and b["cls"] == "down"
    assert b["text"] == "Wanko's backend is not running — run: scripts/voicebot start"
    assert r["svc_after_4000"] == "opened in another tab — reload to use here"


# ---------------------------------------------------------------- addendum 2026-08-30: the admin password


def settings_file():
    return json.load(open(settings.SETTINGS_PATH)) if os.path.exists(settings.SETTINGS_PATH) else {}


def test_set_and_clear_the_password_without_ever_echoing_it(monkeypatch):
    """Scenario 7: the reply and /api/config say only whether it is set; the file holds it."""
    from tests.conftest import send_setting
    monkeypatch.delenv("OMLX_ADMIN_PASSWORD", raising=False)
    client = TestClient(app.app)
    reply = send_setting("omlx_admin_password", "s3-not-for-logs", client)
    assert reply == {"type": "setting", "key": "omlx_admin_password",
                     "omlx_admin_password_set": True, "omlx_admin_password_source": "settings"}
    assert settings_file()["omlx_admin_password"] == "s3-not-for-logs"
    cfg = client.get("/api/config")
    assert cfg.json()["omlx_admin_password_set"] is True and cfg.json()["omlx_admin_password_source"] == "settings"
    assert "s3-not-for-logs" not in cfg.text and "s3-not-for-logs" not in json.dumps(reply)

    reply = send_setting("omlx_admin_password", "", client)
    assert reply["omlx_admin_password_set"] is False and reply["omlx_admin_password_source"] is None
    assert "omlx_admin_password" not in settings_file()
    assert client.get("/api/config").json()["omlx_admin_password_set"] is False
    assert send_setting("omlx_admin_password", 42, client)["type"] == "error"


def test_env_password_wins_over_settings(monkeypatch):
    """Scenario 8."""
    monkeypatch.delenv("OMLX_ADMIN_PASSWORD", raising=False)
    assert services.admin_password() == "" and services.admin_password_source() is None
    settings.save_setting("omlx_admin_password", "from-settings")
    assert services.admin_password() == "from-settings" and services.admin_password_source() == "settings"
    monkeypatch.setenv("OMLX_ADMIN_PASSWORD", "from-env")
    assert services.admin_password() == "from-env" and services.admin_password_source() == "env"
    assert TestClient(app.app).get("/api/config").json()["omlx_admin_password_source"] == "env"


def test_client_password_row_and_model_buttons(r):
    """Scenario 9."""
    assert (r["pw_initial"], r["pw_env"], r["pw_settings"], r["pw_cleared"]) == \
        ("not set", "set (from .env)", "set", "not set")
    assert r["pw_save_sent"] == [{"action": "set_setting", "key": "omlx_admin_password", "value": "hunter2"}]
    assert r["pw_input_after_save"] == ""
    assert r["pw_clear_sent"] == [{"action": "set_setting", "key": "omlx_admin_password", "value": ""}]
    assert r["model_sent"] == ["model_load", "model_unload"]


class FakeAdmin:
    """A `requests.Session` stand-in for the oMLX admin API: the bearer status, the login status."""

    def __init__(self, bearer=200, login=200):
        self.bearer, self.login, self.calls, self.logged_in = bearer, login, [], False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers))
        if url.endswith("/admin/api/login"):
            self.logged_in = self.login < 400
            return SimpleNamespace(status_code=self.login)
        return SimpleNamespace(status_code=200 if self.logged_in else self.bearer)


def model_load(fake, password, monkeypatch):
    if password is None:
        monkeypatch.delenv("OMLX_ADMIN_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("OMLX_ADMIN_PASSWORD", password)
    with patch.object(requests, "Session", lambda: fake), patch.object(core_llm, "OMLX_MODEL", "g"), \
            patch.object(services, "probe", lambda: {"llm": ok("llm"), "asr": ok("asr")}):
        client = TestClient(app.app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"action": "model_load"})
            while True:
                msg = ws.receive_json()
                if msg["type"] in ("services", "error"):
                    return msg


def test_model_load_with_the_bearer_accepted_answers_services(monkeypatch):
    """Scenario 10."""
    fake = FakeAdmin()
    msg = model_load(fake, None, monkeypatch)
    assert msg["type"] == "services" and msg["llm"]["ok"]
    assert [u for u, _, _ in fake.calls] == ["http://localhost:8000/admin/api/models/g/load"]
    assert fake.calls[0][2] == {"Authorization": f"Bearer {core_llm.OMLX_API_KEY}"}


def test_model_load_logs_in_with_the_password_when_the_bearer_is_refused(monkeypatch):
    fake = FakeAdmin(bearer=401)
    msg = model_load(fake, "pw", monkeypatch)
    assert msg["type"] == "services"
    assert [u.rsplit("/", 1)[-1] for u, _, _ in fake.calls] == ["load", "login", "load"]
    assert fake.calls[1][1] == {"password": "pw"}


def test_model_load_without_a_password_is_refused_before_any_login(monkeypatch):
    fake = FakeAdmin(bearer=401)
    msg = model_load(fake, None, monkeypatch)
    assert msg["type"] == "error" and msg["scope"] == "settings" and "password" in msg["text"]
    assert len(fake.calls) == 1


def test_model_load_with_a_wrong_password_reports_the_401(monkeypatch):
    fake = FakeAdmin(bearer=401, login=401)
    msg = model_load(fake, "wrong", monkeypatch)
    assert msg["type"] == "error" and msg["scope"] == "settings" and "401" in msg["text"]
    assert "wrong" not in msg["text"]


def test_model_unload_with_omlx_down_reports_it():
    def boom(*a, **k):
        raise requests.ConnectionError("refused")
    down = FakeAdmin()
    down.post = boom
    with patch.object(requests, "Session", lambda: down):
        with pytest.raises(services.AdminError, match="not answering"):
            services.model_op("unload")


def test_api_config_has_version_and_busy(monkeypatch):
    """PRD 58: a deploy reads which build runs and whether a turn is in flight."""
    monkeypatch.setenv("VOICEBOT_VERSION", "t123")
    body = TestClient(app.app).get("/api/config").json()
    assert body["version"] == "t123"
    assert body["busy"] is False
