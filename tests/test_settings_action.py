"""PRD 22: the generic `set_setting` action and what /api/config hands the UI at boot.

`set_setting` is deliberately narrow — one whitelist (chatbot.SETTABLE), one type per key —
so the UI can persist a layout preference without a new action per checkbox.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json

import pytest
from fastapi.testclient import TestClient

import avatars
import chatbot
import memory
import wiki


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(avatars, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(avatars, "_current", None)
    monkeypatch.delenv("AVATAR", raising=False)
    return tmp_path / "settings.json"


def settings_on_disk(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def send_setting(key, value):
    """Send one set_setting and return the first `setting`/`error` reply."""
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "set_setting", "key": key, "value": value})
        while True:
            msg = ws.receive_json()
            if msg.get("type") in ("setting", "error"):
                return msg


def test_panel_pinned_persists(isolated_settings):
    assert send_setting("panel_pinned", True) == {"type": "setting", "key": "panel_pinned", "value": True}
    assert settings_on_disk(isolated_settings)["panel_pinned"] is True
    assert send_setting("panel_pinned", False)["value"] is False
    assert settings_on_disk(isolated_settings)["panel_pinned"] is False


def test_unknown_key_is_refused(isolated_settings):
    reply = send_setting("vault_path", "/etc")
    assert reply["type"] == "error" and "vault_path" in reply["text"]
    assert "vault_path" not in settings_on_disk(isolated_settings)


def test_wrong_type_is_refused(isolated_settings):
    reply = send_setting("panel_pinned", "yes")
    assert reply["type"] == "error" and "bool" in reply["text"]
    assert "panel_pinned" not in settings_on_disk(isolated_settings)


def test_api_config_exposes_the_ui_settings(isolated_settings):
    avatars.save_setting("panel_pinned", True)
    cfg = TestClient(chatbot.app).get("/api/config").json()
    assert cfg["panel_pinned"] is True
    assert cfg["stale_days"] == memory.STALE_DAYS
    assert cfg["stale_days"]["people"] == 180
    assert cfg["vault"] == wiki.WIKI_PATH
    assert "screenshot.py" in cfg["screenshot_cmd"]


def test_api_config_defaults_to_unpinned():
    cfg = TestClient(chatbot.app).get("/api/config").json()
    assert cfg["panel_pinned"] is False
