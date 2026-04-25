import os
import requests

LOCATION_NAME = os.getenv("LOCATION_NAME", "Seychelles")
LOCATION_LAT = float(os.getenv("LOCATION_LAT", "-4.6796"))
LOCATION_LON = float(os.getenv("LOCATION_LON", "55.4920"))
LOCATION_TIMEZONE = os.getenv("LOCATION_TIMEZONE", "Indian/Mahe")

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
        "description": f"Returns the current weather in {LOCATION_NAME}.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def run(args):
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": LOCATION_LAT,
            "longitude": LOCATION_LON,
            "current": "temperature_2m,apparent_temperature,weathercode,windspeed_10m,wind_direction_10m",
            "daily": "temperature_2m_max,temperature_2m_min,weathercode,precipitation_sum",
            "temperature_unit": "celsius",
            "windspeed_unit": "kmh",
            "timezone": LOCATION_TIMEZONE,
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
    return (
        f"Read out this full weather report naturally: "
        f"Currently {condition}, {temp} degrees Celsius, feels like {feels} degrees Celsius. "
        f"Today's low is {low} degrees Celsius and the high is {high} degrees Celsius. "
        f"Outlook: {outlook}. "
        f"Wind: {wind} kilometers per hour from the {wind_dir}."
    )
