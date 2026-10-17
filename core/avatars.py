"""Avatar profiles: server-side half (name, voice, persona), derived from character cards.

Client half lives in static/app.js PROFILES.
"""
import os

from core import characters
from core import trace
from core import settings

log = trace.logger("avatars")

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


def _load_saved() -> str | None:
    return settings.load_settings().get("avatar")


def _resolve(key: str) -> str:
    """Map a requested key to a known one. Since PRD 23 there is only Wanko, so anything
    else (a stale settings.json, an old AVATAR env value) quietly resolves to the default."""
    key = (key or "").lower()
    if key in AVATARS:
        return key
    fallback = DEFAULT if DEFAULT in AVATARS else next(iter(AVATARS))
    log.info(f"unknown avatar {key!r}; using {fallback}")
    return fallback


def current_key() -> str:
    global _current
    if _current is None:
        _current = _resolve(_load_saved() or os.getenv("AVATAR") or DEFAULT)
    return _current


def set_current(key: str) -> dict:
    global _current
    _current = _resolve(key)
    settings.save_setting("avatar", _current)
    return current()


def current() -> dict:
    return {"key": current_key(), **AVATARS[current_key()]}
