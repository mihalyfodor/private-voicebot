"""Avatar profiles: server-side half (name, voice, persona), derived from character cards.

Client half lives in static/app.js PROFILES.
"""
import json
import os

import characters

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")

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


def _validate(key: str) -> str:
    key = (key or "").lower()
    if key not in AVATARS:
        raise ValueError(f"Unknown avatar {key!r}; valid: {', '.join(AVATARS)}")
    return key


def current_key() -> str:
    global _current
    if _current is None:
        saved = _load_saved()
        if saved:
            try:
                _current = _validate(saved)
            except ValueError:
                print(f"[avatars] warning: invalid saved avatar {saved!r}; falling back")
                _current = _validate(os.getenv("AVATAR") or DEFAULT)
        else:
            _current = _validate(os.getenv("AVATAR") or DEFAULT)
    return _current


def set_current(key: str) -> dict:
    global _current
    _current = _validate(key)
    save_setting("avatar", _current)
    return current()


def current() -> dict:
    return {"key": current_key(), **AVATARS[current_key()]}


def listing() -> list:
    return [{"key": k, "name": v["name"], "description": v["tagline"]} for k, v in AVATARS.items()]
