"""PRD 22/22b: the generic `set_setting` action and what /api/config hands the UI at boot.

`set_setting` is deliberately narrow — one whitelist (chatbot.SETTABLE), one type per key.
PRD 22b emptied the whitelist (the contextual panel and its `panel_pinned` flag are gone);
the action and this test file stay so the next persisted preference is a one-line change.
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


def test_unknown_key_is_refused(isolated_settings):
    reply = send_setting("vault_path", "/etc")
    assert reply["type"] == "error" and "vault_path" in reply["text"]
    assert "vault_path" not in settings_on_disk(isolated_settings)


def test_panel_pinned_is_gone(isolated_settings):
    """PRD 22b: the panel flag is no longer a settable key, and nothing writes it."""
    assert "panel_pinned" not in chatbot.SETTABLE
    reply = send_setting("panel_pinned", True)
    assert reply["type"] == "error"
    assert settings_on_disk(isolated_settings) == {}


def test_whitelisted_key_persists_and_echoes(isolated_settings, monkeypatch):
    """The mechanism itself still works for whatever key is whitelisted next."""
    monkeypatch.setitem(chatbot.SETTABLE, "demo_flag", bool)
    assert send_setting("demo_flag", True) == {"type": "setting", "key": "demo_flag", "value": True}
    assert settings_on_disk(isolated_settings)["demo_flag"] is True
    reply = send_setting("demo_flag", "yes")
    assert reply["type"] == "error" and "bool" in reply["text"]


def test_api_config_exposes_the_ui_settings(isolated_settings):
    cfg = TestClient(chatbot.app).get("/api/config").json()
    assert "panel_pinned" not in cfg
    assert cfg["stale_days"] == memory.STALE_DAYS
    assert cfg["stale_days"]["people"] == 180
    assert cfg["vault"] == wiki.WIKI_PATH
    assert "screenshot.py" in cfg["screenshot_cmd"]


def test_index_html_has_no_panel_markup():
    """PRD 22b scenario 1: the contextual panel is gone from the DOM, not just hidden."""
    root = os.path.join(os.path.dirname(__file__), "..")
    html = open(os.path.join(root, "static", "index.html")).read()
    for dead in ('id="side-panel"', 'id="pin-btn"', 'id="scrim"', 'id="panel-wrap"'):
        assert dead not in html, dead
    for moved in ('id="memory-save-now"', 'id="wiki-ingest"', 'id="hands-free-toggle"',
                  'id="memory-reset"', 'id="sources-file"'):
        assert moved in html, moved


@pytest.mark.parametrize("raw, expected", [
    ("0", False), ("false", False), ("FALSE", False), ("no", False), ("No", False), (" 0 ", False),
    ("1", True), ("true", True), ("yes", True), ("", True),
])
def test_env_flag_parsing(monkeypatch, raw, expected):
    """OPEN_BROWSER=0/false/no keeps a restart from stealing focus; anything else opens a tab."""
    monkeypatch.setenv("OPEN_BROWSER", raw)
    assert chatbot.env_flag("OPEN_BROWSER") is expected


def test_env_flag_defaults_to_on(monkeypatch):
    monkeypatch.delenv("OPEN_BROWSER", raising=False)
    assert chatbot.env_flag("OPEN_BROWSER") is True
    assert chatbot.env_flag("OPEN_BROWSER", "0") is False
