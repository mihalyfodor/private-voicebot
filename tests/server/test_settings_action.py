"""PRD 22/22b: the generic `set_setting` action and what /api/config hands the UI at boot.

`set_setting` is deliberately narrow — one whitelist (chatbot.SETTABLE), one type per key.
PRD 22b emptied the whitelist (the contextual panel and its `panel_pinned` flag are gone);
the action and this test file stay so the next persisted preference is a one-line change.
"""
import os
import json

import pytest
import requests
from fastapi.testclient import TestClient

from core import avatars
from core import settings
from server.actions import settings as m_settings
from server import app
from memory import store as m_store
from vault import store
from tests.conftest import send_setting
from tools import weather as _weather_mod

REAL_REVERSE_GEOCODE = _weather_mod.reverse_geocode      # the geocoder tests call the real one


@pytest.fixture
def isolated_settings(file_sandbox):
    return file_sandbox / "settings.json"


@pytest.fixture(autouse=True)
def no_nominatim(monkeypatch):
    """No test in this module reaches Nominatim; the canned geocoder records its calls."""
    from tools import weather
    calls = []

    def fake(lat, lon):
        calls.append((lat, lon))
        return "Honolulu, United States"
    monkeypatch.setattr(weather, "reverse_geocode", fake)
    monkeypatch.delenv("LOCATION_NAME", raising=False)
    return calls


def settings_on_disk(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def _send(payload):
    """Send one arbitrary action on a fresh socket: (client, first `setting`/`error` reply)."""
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json(payload)
        while True:
            msg = ws.receive_json()
            if msg.get("type") in ("setting", "error"):
                return client, msg


def test_unknown_key_is_refused(isolated_settings):
    reply = send_setting("vault_path", "/etc")
    assert reply["type"] == "error" and "vault_path" in reply["text"]
    assert "vault_path" not in settings_on_disk(isolated_settings)


def test_panel_pinned_is_gone(isolated_settings):
    """PRD 22b: the panel flag is no longer a settable key, and nothing writes it."""
    reply = send_setting("panel_pinned", True)
    assert reply["type"] == "error"
    assert settings_on_disk(isolated_settings) == {}


def test_whitelisted_key_persists_and_echoes(isolated_settings, monkeypatch):
    """The mechanism itself still works for whatever key is whitelisted next."""
    monkeypatch.setitem(m_settings.SETTABLE, "demo_flag", bool)
    assert send_setting("demo_flag", True) == {"type": "setting", "key": "demo_flag", "value": True}
    assert settings_on_disk(isolated_settings)["demo_flag"] is True
    reply = send_setting("demo_flag", "yes")
    assert reply["type"] == "error" and "bool" in reply["text"]


def test_api_config_exposes_the_ui_settings(isolated_settings):
    cfg = TestClient(app.app).get("/api/config").json()
    assert "panel_pinned" not in cfg
    assert cfg["stale_days"] == m_store.STALE_DAYS
    assert cfg["stale_days"]["people"] == 180
    assert cfg["vault"] == store.WIKI_PATH
    assert "screenshot.py" in cfg["screenshot_cmd"]


def test_index_html_has_no_panel_markup():
    """PRD 22b scenario 1: the contextual panel is gone from the DOM, not just hidden."""
    root = os.path.join(os.path.dirname(__file__), "..", "..")
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
    assert app.env_flag("OPEN_BROWSER") is expected


def test_env_flag_defaults_to_on(monkeypatch):
    monkeypatch.delenv("OPEN_BROWSER", raising=False)
    assert app.env_flag("OPEN_BROWSER") is True
    assert app.env_flag("OPEN_BROWSER", "0") is False


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

    settings.save_setting("location", {"lat": 21.307, "lon": -157.858, "tz": "Pacific/Honolulu", "source": "browser"})
    got = weather.location()
    assert (got["source"], got["lat"], got["lon"], got["tz"]) == ("browser", 21.307, -157.858, "Pacific/Honolulu")
    assert got["name"] == "Budapest"       # .env LOCATION_NAME wins over the geocoded name
    monkeypatch.delenv("LOCATION_NAME")
    assert weather.location()["name"] == ""
    settings.save_setting("location", {"lat": 21.307, "lon": -157.858, "tz": "Pacific/Honolulu",
                                      "source": "browser", "name": "Honolulu, United States"})
    assert weather.location()["name"] == "Honolulu, United States"


def test_set_location_rounds_saves_and_shows_in_config(isolated_settings, monkeypatch):
    client, msg = _send({"action": "set_location", "lat": 21.3069, "lon": -157.8583, "tz": "Pacific/Honolulu"})

    assert msg["type"] == "setting" and msg["key"] == "location"
    assert (msg["value"]["lat"], msg["value"]["lon"], msg["value"]["source"]) == (21.307, -157.858, "browser")
    assert settings_on_disk(isolated_settings)["location"] == {
        "lat": 21.307, "lon": -157.858, "tz": "Pacific/Honolulu", "source": "browser", "name": "Honolulu, United States"}
    cfg = client.get("/api/config").json()["location"]
    assert (cfg["source"], cfg["name"]) == ("browser", "Honolulu, United States")


@pytest.mark.parametrize("payload", [
    {"action": "set_location", "lat": 95, "lon": 0, "tz": "UTC"},
    {"action": "set_location", "lat": "x", "lon": 0, "tz": "UTC"},
    {"action": "set_location", "lat": 1, "lon": 1},
])
def test_bad_locations_are_refused_and_nothing_is_saved(isolated_settings, payload):
    _client, msg = _send(payload)
    assert msg["type"] == "error"
    assert "location" not in settings_on_disk(isolated_settings)


def test_the_page_offers_its_position_once_and_renders_the_row(r):
    assert r["history_after_open"] == 0
    assert r["location_sent"] == [{"action": "set_location", "lat": 21.3069, "lon": -157.8583, "tz": "Pacific/Honolulu"}]
    assert r["location_shown"] == "Honolulu, United States · 21.307, -157.858 · Pacific/Honolulu · from browser"


# ----------------------------------------------------------- PRD 35 addendum: the assistant knows the place

NOMINATIM_REPLY = {"place_id": 1, "display_name": "Honolulu, Honolulu County, Hawaii, United States",
                   "address": {"city": "Honolulu", "county": "Honolulu County", "state": "Hawaii", "country": "United States", "country_code": "us"}}


class _Resp:
    def __init__(self, payload=None, status=200, broken=False):
        self._payload, self.status_code, self._broken = payload, status, broken

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        if self._broken:
            raise ValueError("not json")
        return self._payload


def test_reverse_geocode_parses_a_canned_reply_and_sends_the_agent(monkeypatch):
    from tools import weather
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, headers=headers, timeout=timeout)
        return _Resp(NOMINATIM_REPLY)
    monkeypatch.setattr(weather.requests, "get", fake_get)
    monkeypatch.setenv("WIKIPEDIA_CONTACT", "me@example.com")

    assert REAL_REVERSE_GEOCODE(21.307, -157.858) == "Honolulu, United States"
    assert seen["url"] == "https://nominatim.openstreetmap.org/reverse"
    assert (seen["params"]["format"], seen["params"]["zoom"], seen["timeout"]) == ("jsonv2", 10, 5)
    assert seen["headers"]["User-Agent"] == "voicebot/1.0 (personal assistant; me@example.com)"

    monkeypatch.delenv("WIKIPEDIA_CONTACT")
    REAL_REVERSE_GEOCODE(1, 1)
    assert seen["headers"]["User-Agent"] == "voicebot/1.0 (personal assistant)"


@pytest.mark.parametrize("get", [
    lambda *a, **k: (_ for _ in ()).throw(requests.Timeout("slow")),
    lambda *a, **k: _Resp(status=503),
    lambda *a, **k: _Resp(broken=True),
    lambda *a, **k: _Resp({"error": "Unable to geocode"}),
])
def test_reverse_geocode_failures_yield_an_empty_name(monkeypatch, get):
    from tools import weather
    monkeypatch.setattr(weather.requests, "get", get)
    assert REAL_REVERSE_GEOCODE(21.307, -157.858) == ""


def test_set_location_geocodes_once_and_skips_small_moves(isolated_settings, no_nominatim):
    _send({"action": "set_location", "lat": 21.307, "lon": -157.858, "tz": "Pacific/Honolulu"})
    assert no_nominatim == [(21.307, -157.858)]

    _send({"action": "set_location", "lat": 21.311, "lon": -157.852, "tz": "Pacific/Honolulu"})   # a reload
    assert len(no_nominatim) == 1
    assert settings_on_disk(isolated_settings)["location"]["name"] == "Honolulu, United States"

    _send({"action": "set_location", "lat": 47.5, "lon": 19.0, "tz": "Europe/Budapest"})        # a trip
    assert no_nominatim[-1] == (47.5, 19.0) and len(no_nominatim) == 2


def test_a_failed_geocode_keeps_the_coordinates_without_a_name(isolated_settings, monkeypatch):
    from tools import weather
    monkeypatch.setattr(weather, "reverse_geocode", lambda lat, lon: "")
    _client, msg = _send({"action": "set_location", "lat": 21.307, "lon": -157.858, "tz": "Pacific/Honolulu"})
    assert msg["type"] == "setting" and msg["value"]["name"] == ""
    assert settings_on_disk(isolated_settings)["location"]["name"] == ""


def test_the_system_prompt_names_the_place_after_set_location(isolated_settings, monkeypatch):
    from agent import conversation
    from agent import prompt as m_prompt
    import re
    for key in ("LOCATION_LAT", "LOCATION_LON", "LOCATION_TIMEZONE"):
        monkeypatch.delenv(key, raising=False)
    assert "The user is in" not in m_prompt.build_system_prompt(avatars.current())

    _send({"action": "set_location", "lat": 21.307, "lon": -157.858, "tz": "Pacific/Honolulu"})
    conversation.refresh_context()
    prompt = m_prompt.build_system_prompt(avatars.current())
    assert re.search(r"The user is in Honolulu, United States \(timezone Pacific/Honolulu, local time \d\d:\d\d\)\.",
                     prompt), prompt
    if conversation._conversation:
        assert "The user is in Honolulu, United States" in conversation._conversation[0]["content"]


def test_weather_text_uses_the_geocoded_name(isolated_settings, monkeypatch):
    from tools import weather
    settings.save_setting("location", {"lat": 21.307, "lon": -157.858, "tz": "Pacific/Honolulu",
                                      "source": "browser", "name": "Honolulu, United States"})
    forecast = {"current": {"temperature_2m": 21.4, "apparent_temperature": 20.0, "weathercode": 1,
                            "windspeed_10m": 8.0, "wind_direction_10m": 90},
                "daily": {"temperature_2m_max": [25.0], "temperature_2m_min": [14.0], "weathercode": [2],
                          "precipitation_sum": [0.0]}}
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: _Resp(forecast))
    assert "(in Honolulu, United States)" in weather.run({})


@pytest.mark.live
def test_live_nominatim_names_cluj(monkeypatch):
    name = REAL_REVERSE_GEOCODE(21.307, -157.858)
    print(f"\n[live] reverse_geocode(21.307, -157.858) -> {name!r}")
    assert "Honolulu" in name
