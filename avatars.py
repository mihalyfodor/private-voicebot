"""Avatar profiles: server-side half (name, voice, persona). Client half lives in static/app.js PROFILES."""
import json
import os

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.json")

AVATARS = {
    "wanko": {
        "name": "Wanko",
        "description": "Dog mascot, upbeat",
        "voice": "am_puck",
        "speed": 1.05,
        "persona": "You are Wanko, a small, cheerful dog mascot who works as an office assistant. Warm and upbeat, but professional.",
    },
    "haru": {
        "name": "Haru",
        "description": "Office assistant, calm",
        "voice": "af_sarah",
        "speed": 0.95,
        "persona": "You are Haru, a calm and friendly office assistant.",
    },
    "natori": {
        "name": "Natori",
        "description": "Office assistant, easygoing",
        "voice": "am_michael",
        "speed": 1.0,
        "persona": "You are Natori, an easygoing, well-organised office assistant. Friendly and direct.",
    },
}

DEFAULT = "wanko"
_current: str | None = None


def load_settings() -> dict:
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_setting(key: str, value) -> None:
    data = load_settings()
    data[key] = value
    with open(SETTINGS_PATH, "w") as f:
        json.dump(data, f)


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
        _current = _validate(_load_saved() or os.getenv("AVATAR") or DEFAULT)
    return _current


def set_current(key: str) -> dict:
    global _current
    _current = _validate(key)
    save_setting("avatar", _current)
    return current()


def current() -> dict:
    return {"key": current_key(), **AVATARS[current_key()]}


def listing() -> list:
    return [{"key": k, "name": v["name"], "description": v["description"]} for k, v in AVATARS.items()]
