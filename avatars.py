"""Avatar profiles: server-side half (name, voice, persona), derived from character cards.

Client half lives in static/app.js PROFILES.
"""
import json
import os

import characters

# SETTINGS_PATH env overrides the location (the e2e tests point it at a temp dir)
SETTINGS_PATH = os.getenv("SETTINGS_PATH") or os.path.join(os.path.dirname(__file__), "settings.json")

DEFAULT = "wanko"
_current: str | None = None


def _build_avatars() -> dict:
    avatars = {}
    for key, card in characters.load_all().items():
        avatars[key] = {
            "name": card["name"],
            "tagline": card["tagline"],
            "description": card["tagline"],  # backward-compat alias
            "voice": card["voice"],
            "speed": card["speed"],
            "greeting": card["greeting"],
            "switch_greeting": card["switch_greeting"],
            "verbosity": card["verbosity"],
            "persona": characters.build_persona(card),
        }
    return avatars


AVATARS = _build_avatars()


def reload() -> dict:
    """Re-read cards from disk. Keeps the current key if still valid, else falls back to DEFAULT."""
    global AVATARS, _current
    AVATARS = _build_avatars()
    if _current is not None and _current not in AVATARS:
        _current = DEFAULT if DEFAULT in AVATARS else None
    return current()


SETTINGS_VERSION = 1


def load_settings() -> dict:
    """Read settings.json. A missing file is silently {}; a corrupt one warns and is {}."""
    try:
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        print(f"[avatars] warning: could not read {SETTINGS_PATH} ({e}); using defaults")
        return {}
    if not isinstance(data, dict):
        print(f"[avatars] warning: {SETTINGS_PATH} is not an object; using defaults")
        return {}
    return data


def save_setting(key: str, value) -> None:
    """Persist one setting atomically (tmp file + os.replace), stamping the schema version."""
    data = load_settings()
    data[key] = value
    data["version"] = SETTINGS_VERSION
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, SETTINGS_PATH)


def _load_saved() -> str | None:
    return load_settings().get("avatar")


def _resolve(key: str) -> str:
    """Map a requested key to a known one. Since PRD 23 there is only Wanko, so anything
    else (a stale settings.json, an old AVATAR env value) quietly resolves to the default."""
    key = (key or "").lower()
    if key in AVATARS:
        return key
    fallback = DEFAULT if DEFAULT in AVATARS else next(iter(AVATARS))
    print(f"[avatars] unknown avatar {key!r}; using {fallback}")
    return fallback


def current_key() -> str:
    global _current
    if _current is None:
        _current = _resolve(_load_saved() or os.getenv("AVATAR") or DEFAULT)
    return _current


def set_current(key: str) -> dict:
    global _current
    _current = _resolve(key)
    save_setting("avatar", _current)
    return current()


def current() -> dict:
    return {"key": current_key(), **AVATARS[current_key()]}


def listing() -> list:
    return [{"key": k, "name": v["name"], "description": v["tagline"]} for k, v in AVATARS.items()]
