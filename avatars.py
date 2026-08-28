"""Avatar profiles: server-side half (name, voice, persona). Client half lives in static/app.js PROFILES."""
import os

AVATARS = {
    "wanko": {
        "name": "Wanko",
        "voice": "am_puck",
        "speed": 1.05,
        "persona": "You are Wanko, a small, cheerful dog mascot who works as an office assistant. Warm and upbeat, but professional.",
    },
    "haru": {
        "name": "Haru",
        "voice": "af_sarah",
        "speed": 0.95,
        "persona": "You are Haru, a calm and friendly office assistant.",
    },
}

DEFAULT = "wanko"


def current_key() -> str:
    key = (os.getenv("AVATAR") or DEFAULT).lower()
    if key not in AVATARS:
        raise ValueError(f"Unknown AVATAR={key!r}; valid: {', '.join(AVATARS)}")
    return key


def current() -> dict:
    return {"key": current_key(), **AVATARS[current_key()]}
