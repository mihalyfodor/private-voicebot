import os
import requests

import avatars

#: the hosts this tool talks to (docs/33-untrusted-text.md read-only policy): the forecast, and Nominatim
#: for one reverse-geocode per new position (PRD 35 addendum) — a fixed endpoint, never a free URL
HOSTS = ("api.open-meteo.com", "nominatim.openstreetmap.org")
#: a saved position within this distance (degrees, ~1 km) keeps its name instead of asking Nominatim again
GEOCODE_MOVE_THRESHOLD = 0.01

#: the built-in fallback when neither the browser (settings.json) nor `.env` says where the user is
DEFAULT_LOCATION = {"name": "Seychelles", "lat": -4.6796, "lon": 55.4920, "tz": "Indian/Mahe", "source": "default"}


def location() -> dict:
    """Where the weather is for: settings.json `location` (from the browser, PRD 35) → `LOCATION_*`
    env → the built-in default. Always `{name?, lat, lon, tz, source}`."""
    saved = avatars.load_settings().get("location")
    if isinstance(saved, dict) and "lat" in saved and "lon" in saved:
        return {"name": os.getenv("LOCATION_NAME", "") or str(saved.get("name") or ""),
                "lat": float(saved["lat"]), "lon": float(saved["lon"]),
                "tz": str(saved.get("tz") or os.getenv("LOCATION_TIMEZONE") or "auto"), "source": "browser"}
    if os.getenv("LOCATION_LAT") and os.getenv("LOCATION_LON"):
        return {"name": os.getenv("LOCATION_NAME", ""), "lat": float(os.getenv("LOCATION_LAT")),
                "lon": float(os.getenv("LOCATION_LON")),
                "tz": os.getenv("LOCATION_TIMEZONE", "auto"), "source": "env"}
    return dict(DEFAULT_LOCATION)


def _user_agent() -> str:
    ua = "voicebot/1.0 (personal assistant"
    contact = os.getenv("WIKIPEDIA_CONTACT", "").strip()
    return ua + (f"; {contact}" if contact else "") + ")"


def reverse_geocode(lat: float, lon: float) -> str:
    """`"City, Country"` for a position from Nominatim, or `""` on any failure (never raises)."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "jsonv2", "zoom": 10, "lat": lat, "lon": lon},
            headers={"User-Agent": _user_agent(), "Accept": "application/json"},
            timeout=5,
        )
        resp.raise_for_status()
        address = resp.json().get("address") or {}
    except Exception as exc:                       # noqa: BLE001 — a missing name is never the user's problem
        print(f"[Location] reverse geocode failed: {exc}")
        return ""
    place = next((address[k] for k in ("city", "town", "village", "municipality", "county") if address.get(k)), "")
    return ", ".join(p for p in (place, address.get("country", "")) if p)


def needs_geocode(saved, lat: float, lon: float) -> bool:
    """True when no name is stored or the position moved by more than `GEOCODE_MOVE_THRESHOLD`."""
    if not isinstance(saved, dict) or not saved.get("name"):
        return True
    try:
        return (abs(float(saved["lat"]) - lat) > GEOCODE_MOVE_THRESHOLD
                or abs(float(saved["lon"]) - lon) > GEOCODE_MOVE_THRESHOLD)
    except (KeyError, TypeError, ValueError):
        return True

WMO_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}

DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Returns the current weather and today's outlook at the user's location.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def run(args):
    where = location()
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": where["lat"],
            "longitude": where["lon"],
            "current": "temperature_2m,apparent_temperature,weathercode,windspeed_10m,wind_direction_10m",
            "daily": "temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum",
            "temperature_unit": "celsius",
            "windspeed_unit": "kmh",
            "timezone": where["tz"],
            "forecast_days": 1,
        },
        timeout=5,
    )
    data = resp.json()
    c = data["current"]
    d = data["daily"]
    dirs = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
    condition = WMO_CODES.get(c["weathercode"], "unknown")
    outlook = WMO_CODES.get(d["weathercode"][0], "unknown")
    temp = round(c['temperature_2m'])
    feels = round(c['apparent_temperature'])
    low = round(d['temperature_2m_min'][0])
    high = round(d['temperature_2m_max'][0])
    wind = round(c['windspeed_10m'])
    wind_dir = dirs[round(c['wind_direction_10m'] / 45) % 8]
    place = f"in {where['name']}" if where.get("name") else "at your location"
    return (
        f"Read out this full weather report naturally ({place}): "
        f"Currently {condition}, {temp} degrees Celsius, feels like {feels} degrees Celsius. "
        f"Today's low is {low} degrees Celsius and the high is {high} degrees Celsius. "
        f"Outlook: {outlook}. "
        f"Wind: {wind} kilometers per hour from the {wind_dir}."
    )
