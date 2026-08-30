Status: done

# 35 — Location from the browser

## Overview
The weather tool used a hard-coded default (Seychelles) unless `LOCATION_*` was set in `.env`. The browser knows
where the user is; on connect the page offers its position once, the server stores it in settings.json, and the
weather tool reads it. `.env` stays as the fallback and the Settings view shows what is in use.

## User stories
- As a user, "how is the weather" answers for where I actually am, after one browser permission prompt.
- As a user, I can see in Settings which location and time zone the weather uses and where it came from.

## Technical approach
- `static/app.js`: on the first websocket open per page, `navigator.geolocation.getCurrentPosition` (maximumAge 6 h,
  low accuracy); success → `{"action":"set_location","lat","lon","tz"}` with `tz` from
  `Intl.DateTimeFormat().resolvedOptions().timeZone`; denial/absence → nothing (fallback stays).
- `chatbot.py`: action `set_location` validates lat ∈ [-90, 90], lon ∈ [-180, 180], tz a non-empty string, rounds to
  3 decimals (≈ 100 m, enough for weather, no street-level trail in settings.json) and saves
  `location: {lat, lon, tz, source: "browser"}`; replies `setting {key: "location", value}`; `/api/config` exposes
  `location` (resolved: browser or env, with `source`).
- `tools/weather.py`: `location()` → settings.json `location` if present, else `LOCATION_*` env (source `env`),
  else the built-in default; `run` uses it and names the place when a name is known ("in Cluj") or says "at your
  location"; the tool description no longer bakes a place name in at import.
- Settings view: a "Location" row under Tools: `46.770, 23.590 · Europe/Bucharest · from browser` (or env/default).

### Decisions
- Browser geolocation over IP lookup: no third-party call, one permission prompt, works offline.
- Store rounded coordinates: weather needs a city, not a street.
- No reverse geocoding (would need a service): the name comes only from `LOCATION_NAME` when set.

## Test scenarios
1. `weather.location()` prefers settings.json over env over the default, and reports `source`.
2. `set_location` with valid values → `setting` reply, settings.json updated, `/api/config.location` shows it;
   out-of-range or missing fields → `error`, nothing saved.
3. Fixture: on socket open the page sends `set_location` from a stubbed `navigator.geolocation`; without geolocation
   it sends nothing; the Settings row renders the resolved value from config and from a `setting` message.
4. Manual: allow location in the browser, ask for the weather, see the Settings row.

## Out of scope
Reverse geocoding, multiple saved places, per-tool time zones.
