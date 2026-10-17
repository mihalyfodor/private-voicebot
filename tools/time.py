from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tools import weather as _weather

#: no network at all (docs/33-untrusted-text.md read-only policy)
HOSTS = ()

DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Returns the current local date, weekday, time and timezone.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


def now() -> datetime:
    """The current moment in the user's timezone (the weather location's `tz`; the system zone
    when that is "auto" or unknown). Aware."""
    try:
        return datetime.now(ZoneInfo(_weather.location()["tz"]))
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return datetime.now().astimezone()


def run(args):
    t = now()
    return f"{t.strftime('%A, %d %B %Y, %H:%M:%S')} ({getattr(t.tzinfo, 'key', None) or t.tzname()})"
