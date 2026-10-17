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
  else the built-in default; `run` uses it and names the place when a name is known ("in Honolulu") or says "at your
  location"; the tool description no longer bakes a place name in at import.
- Settings view: a "Location" row under Tools: `21.307, -157.858 · Pacific/Honolulu · from browser` (or env/default).

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

## Also (2026-08-30): the assistant knows the place

Coordinates only fed the weather tool; the assistant itself had no idea where the user was ("where am I?" → "I don't
know"). Now `set_location` reverse-geocodes once and the place name reaches the system prompt.

- `tools/weather.py`: `reverse_geocode(lat, lon)` calls Nominatim `reverse?format=jsonv2&zoom=10` with
  `User-Agent: voicebot/1.0 (personal assistant[; WIKIPEDIA_CONTACT])`, 5 s timeout, and returns `"City, Country"`
  (city/town/village/municipality/county + country) or `""` on any failure — never an error to the user.
  `nominatim.openstreetmap.org` joins `HOSTS` (read-only, one fixed endpoint, per PRD 33).
- `chatbot.py` `set_location`: geocodes only when no name is stored yet or the coordinates moved by more than 0.01°
  (≈ 1 km) from the saved ones — a page reload does not hit Nominatim again. The result is stored as `name` in
  the settings `location`; a failed lookup keeps the coordinates without a name. `llm.refresh_context()` runs
  afterwards so the very next turn knows the place.
- `location()`: `LOCATION_NAME` from `.env` still wins when set; otherwise the stored `name`.
- `llm.py`: the system prompt gets one line `The user is in <name> (timezone <tz>, local time <HH:MM>).` built at
  prompt build/refresh time; omitted when no name is known. The weather tool keeps saying "in <name>".
- Settings row: `Honolulu, United States · 21.307, -157.858 · Pacific/Honolulu · from browser`; `/api/config.location`
  carries `name`.

### Test scenarios
5. `reverse_geocode` parses a canned Nominatim response into "City, Country"; a timeout/HTTP error/garbage body
   yields `""` and raises nothing.
6. `set_location` with a mocked geocoder stores `name`; sending nearly the same coordinates again (< 0.01°) does
   not call the geocoder and keeps the name; moving > 0.01° geocodes again; a failing geocoder saves the
   coordinates with `name: ""` and still replies `setting`.
7. After `set_location` + `refresh_context` the system prompt contains `The user is in Honolulu, United States
   (timezone Pacific/Honolulu, local time HH:MM)`; with no location the line is absent.
8. The weather text says "in Honolulu, United States" when the stored name is set and `.env` has none.
9. Fixture: the Settings row renders the name in front of the coordinates.
10. Host policy: `nominatim.openstreetmap.org` is in `ALLOWED_HOSTS` and the weather module declares it.
11. Live (`-m live`): Nominatim for 21.307, -157.858 returns a name containing "Honolulu".
