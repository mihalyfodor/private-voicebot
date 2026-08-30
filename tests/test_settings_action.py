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


@pytest.fixture
def isolated_settings(file_sandbox):
    return file_sandbox / "settings.json"


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


# --------------------------------------------------------------- PRD 35: location


def test_weather_location_prefers_browser_over_env_over_default(isolated_settings, monkeypatch):
    from tools import weather

    for key in ("LOCATION_NAME", "LOCATION_LAT", "LOCATION_LON", "LOCATION_TIMEZONE"):
        monkeypatch.delenv(key, raising=False)
    assert weather.location()["source"] == "default"

    monkeypatch.setenv("LOCATION_LAT", "47.5"); monkeypatch.setenv("LOCATION_LON", "19.0")
    monkeypatch.setenv("LOCATION_NAME", "Budapest"); monkeypatch.setenv("LOCATION_TIMEZONE", "Europe/Budapest")
    env = weather.location()
    assert (env["source"], env["name"], env["lat"]) == ("env", "Budapest", 47.5)

    avatars.save_setting("location", {"lat": 46.771, "lon": 23.624, "tz": "Europe/Bucharest", "source": "browser"})
    got = weather.location()
    assert (got["source"], got["lat"], got["lon"], got["tz"]) == ("browser", 46.771, 23.624, "Europe/Bucharest")
    assert got["name"] == "Budapest"       # a name only ever comes from .env (no reverse geocoding)


def _send(payload):
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json(payload)
        while True:
            msg = ws.receive_json()
            if msg.get("type") in ("setting", "error"):
                return client, msg


def test_set_location_rounds_saves_and_shows_in_config(isolated_settings, monkeypatch):
    monkeypatch.delenv("LOCATION_NAME", raising=False)
    client, msg = _send({"action": "set_location", "lat": 46.77123, "lon": 23.62361, "tz": "Europe/Bucharest"})

    assert msg["type"] == "setting" and msg["key"] == "location"
    assert (msg["value"]["lat"], msg["value"]["lon"], msg["value"]["source"]) == (46.771, 23.624, "browser")
    assert settings_on_disk(isolated_settings)["location"] == {
        "lat": 46.771, "lon": 23.624, "tz": "Europe/Bucharest", "source": "browser"}
    assert client.get("/api/config").json()["location"]["source"] == "browser"


@pytest.mark.parametrize("payload", [
    {"action": "set_location", "lat": 95, "lon": 0, "tz": "UTC"},
    {"action": "set_location", "lat": "x", "lon": 0, "tz": "UTC"},
    {"action": "set_location", "lat": 1, "lon": 1},
])
def test_bad_locations_are_refused_and_nothing_is_saved(isolated_settings, payload):
    _client, msg = _send(payload)
    assert msg["type"] == "error"
    assert "location" not in settings_on_disk(isolated_settings)


@pytest.fixture(scope="module")
def r():
    import shutil, subprocess
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    root = os.path.join(os.path.dirname(__file__), "..")
    out = subprocess.run(["node", "tests/app_fixture.mjs"], cwd=root, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_the_page_offers_its_position_once_and_renders_the_row(r):
    assert r["location_sent"] == [{"action": "set_location", "lat": 46.7712, "lon": 23.6236, "tz": "Europe/Bucharest"}]
    assert r["location_shown"] == "46.771, 23.624 · Europe/Bucharest · from browser"
