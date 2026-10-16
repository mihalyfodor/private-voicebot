"""settings.json: load and save (moved from avatars.py, docs/50-modular-layout.md)."""
import json
import os

from core import config, trace

log = trace.logger("settings")

SETTINGS_PATH = config.SETTINGS_PATH

SETTINGS_VERSION = 1


def load_settings() -> dict:
    """Read settings.json. A missing file is silently {}; a corrupt one warns and is {}."""
    try:
        with open(SETTINGS_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.warning(f"warning: could not read {SETTINGS_PATH} ({e}); using defaults")
        return {}
    if not isinstance(data, dict):
        log.warning(f"warning: {SETTINGS_PATH} is not an object; using defaults")
        return {}
    return data


def save_setting(key: str, value) -> None:
    """Persist one setting atomically (tmp file + os.replace), stamping the schema version."""
    data = load_settings()
    data[key] = value
    save_settings(data)


def save_settings(data: dict) -> None:
    """Write the whole settings object atomically, stamping the schema version."""
    data["version"] = SETTINGS_VERSION
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, SETTINGS_PATH)
