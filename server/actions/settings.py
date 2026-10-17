"""WebSocket actions of the settings panel: hands-free, backdrop, characters, verbosity, location, settings and the model (PRD 50 step 11, from chatbot.py)."""
import os
import signal
import asyncio
from core import trace
from agent import conversation as a_conversation
from agent import prompt as a_prompt
from core import avatars
from core import settings
from server import services
from tools import weather as weather_tool
from server import backdrops
from server import panels
from server import speech

log = trace.logger("App")


async def _action_set_hands_free(sess, msg, loop):

    value = bool(msg.get("value"))
    settings.save_setting("hands_free", value)
    panels.hands_free = value
    if sess.endpointer is not None:
        sess.endpointer.gated = False
        sess.endpointer.reset()
    sess.hearing = False
    await panels.send({"type": "hands_free", "value": value})
    if not value:
        await panels.send({"type": "listening", "value": "idle"})


async def _action_set_backdrop(sess, msg, loop):
    key = backdrops.validate(msg.get("key", ""))
    settings.save_setting("backdrop", key)
    await panels.send({"type": "backdrop", "key": key})


async def _action_reload_characters(sess, msg, loop):
    """The user edited characters/wanko.yaml: re-read the card, rebuild persona/voice/fillers."""
    speech.apply_avatar(avatars.reload()["key"])
    await panels.send({"type": "characters_reloaded", "avatar": speech.AVATAR["key"], "name": speech.AVATAR["name"]})


async def _action_set_verbosity(sess, msg, loop):
    a_prompt.set_verbosity(msg.get("value"))
    await panels.send({"type": "verbosity", "value": a_prompt.current_verbosity()})


#: Settings the UI may write straight to settings.json via `set_setting`, and the type each
#: one must have. Deliberately tiny: everything with side effects keeps its own action.
# key -> expected type. Empty since PRD 22b removed `panel_pinned`; the action stays so the
# next persisted UI preference is one line here instead of a new action.
#: `calendar_horizon_days` — how far a recurring invite is expanded into meeting stubs (PRD 31)
#: `recall_min_score` — the BM25-per-word floor a source passage must clear to be quoted (PRD 32)
#: `omlx_admin_password` — the oMLX admin password for load/unload (PRD 41 addendum); "" clears it and
#: the reply / config only ever say whether one is set
SETTABLE: dict[str, type] = {"calendar_horizon_days": int, "recall_min_score": float,
                             services.PASSWORD_KEY: str}


async def _action_set_location(sess, msg, loop):
    """The browser's position (PRD 35): validated, rounded to ~100 m, saved as `location`."""
    try:
        lat, lon = float(msg.get("lat")), float(msg.get("lon"))
    except (TypeError, ValueError):
        await panels._error("settings", "location needs numeric lat/lon")
        return
    tz = str(msg.get("tz") or "").strip()
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or not tz:
        await panels._error("settings", "location out of range or missing tz")
        return
    saved = settings.load_settings().get("location")
    if weather_tool.needs_geocode(saved, lat, lon):
        name = await asyncio.to_thread(weather_tool.reverse_geocode, lat, lon)   # "" when Nominatim fails
    else:
        name = saved["name"]
    value = {"lat": round(lat, 3), "lon": round(lon, 3), "tz": tz, "source": "browser", "name": name}
    settings.save_setting("location", value)
    await asyncio.to_thread(a_conversation.refresh_context)   # the next turn knows the place
    await panels.send({"type": "setting", "key": "location", "value": weather_tool.location()})


async def _action_set_setting(sess, msg, loop):
    """Persist one whitelisted UI setting and echo it back; unknown keys and wrong types are refused.

    The whole point is that a stored UI preference does not need an action of its own — anything
    with a side effect (avatar, verbosity, hands-free) still gets a dedicated handler.
    """
    key = msg.get("key")
    kind = SETTABLE.get(key)
    value = msg.get("value")
    if kind is None:
        await panels._error("settings", f"Unknown setting {key!r}")
        return
    ok = isinstance(value, kind) or (kind is float and isinstance(value, int) and not isinstance(value, bool))
    if not ok:
        await panels._error("settings", f"{key} needs a {kind.__name__}")
        return
    value = kind(value)
    if key == services.PASSWORD_KEY:
        _save_admin_password(value)
        await panels.send({"type": "setting", "key": key, **services.admin_password_status()})
        return
    settings.save_setting(key, value)
    await panels.send({"type": "setting", "key": key, "value": value})


def _save_admin_password(value: str) -> None:
    """Store the password (or drop the key for ""); the value itself is never echoed or printed."""
    value = value.strip()
    if value:
        settings.save_setting(services.PASSWORD_KEY, value)
        return
    data = settings.load_settings()
    if services.PASSWORD_KEY in data:
        del data[services.PASSWORD_KEY]
        settings.save_settings(data)


async def _action_model_load(sess, msg, loop):
    await _model_op("load")


async def _action_model_unload(sess, msg, loop):
    await _model_op("unload")


async def _model_op(op: str) -> None:
    """Load / unload the model through the oMLX admin API (what scripts/voicebot does), then
    re-probe and answer `services`; a refusal comes back on the settings status line."""
    try:
        await asyncio.to_thread(services.model_op, op)
    except services.AdminError as e:
        await panels._error("settings", str(e))
        return
    await asyncio.to_thread(services.refresh)
    await panels.send({"type": "services", **services.message()})


async def _action_shutdown(sess, msg, loop):
    log.info("shutdown requested from UI")
    os.kill(os.getpid(), signal.SIGTERM)  # graceful: triggers the shutdown hook
