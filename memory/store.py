"""Profile files and schema: paths, constants, load/write, normalise, migration from the v1 shortmem (docs/15-memory-v2.md; PRD 50 step 8, from memory.py)."""
import json
import os
import shutil
from core import config, trace
from core import vault_lock
from memory import identity as m_identity
from memory import render
from core import util

log = trace.logger("Memory")



# --- files -------------------------------------------------------------------------
_DIR = config.MEMORY_DIR   # MEMORY_DIR env: e2e tests use a temp dir

MEMORY_PATH = os.path.join(_DIR, "memory.json")            # the profile

OPS_LOG_PATH = os.path.join(_DIR, "memory_ops.jsonl")      # append-only audit log

SHORTMEM_PATH = os.path.join(_DIR, "shortmem.txt")         # v1, read once for migration


# --- schema ------------------------------------------------------------------------
VERSION = 2

SCALAR_SECTIONS = ("identity", "preferences")              # key -> string, plus "superseded" / "sources"

BOOKKEEPING_KEYS = ("superseded", "sources")               # scalar-section keys that are not facts

SOURCE_FIELD = "source"    # the document title an entry was learned from (docs/33-untrusted-text.md)

#: The one list-valued key in a scalar section: every other string that refers to the user
#: (other spellings, the full name, old nicknames, what the ASR tends to hear). docs/26.
ALIAS_KEY = "aliases"

ALIAS_PATH = f"identity.{ALIAS_KEY}"

LIST_SECTIONS = ("people", "projects", "recurring", "episodic")

LIST_KEY = {"people": "name", "projects": "name", "recurring": "what", "episodic": "text"}

CAPS = {"people": 30, "projects": 20, "recurring": 15, "episodic": 40}  # max entries per list

DEFAULT_TTL_DAYS = 30                                             # episodic lifetime when unstated


#: Days after the last `seen` date a list entry stops being rendered (docs/18-memory-decay.md).
#: Nothing is deleted: a stale entry stays on disk and comes back with a CONFIRM or a restore.
STALE_DAYS = {
    "people": int(os.getenv("MEMORY_STALE_PEOPLE", "180")),
    "projects": int(os.getenv("MEMORY_STALE_PROJECTS", "120")),
    "recurring": int(os.getenv("MEMORY_STALE_RECURRING", "90")),
}

STALE_FIELD = "stale"      # the one entry field that holds a boolean

SEEN_FIELD = "seen"        # "YYYY-MM-DD", last time the user confirmed the entry


_LOCK = vault_lock.LOCK   # the vault-wide lock; vault/ and ingest/ take the same one


UI_REASONS = ("ui", "ui:restore", "ui:reset")   # audit tags that mark a hand edit


PROFILE_NOTE = (
    "The block above holds facts about the USER you already know. Use them naturally, "
    "as a friend would (their name, preferences, ongoing projects). Don't recite the list "
    "or announce that you remembered; don't raise sensitive items unprompted. If a fact "
    "conflicts with what the user says now, believe the user."
)



# --------------------------------------------------------------------------- schema


def _empty_profile() -> dict:
    return {
        "version": VERSION,
        "updated": None,
        "identity": {},
        "preferences": {},
        "people": [],
        "projects": [],
        "recurring": [],
        "episodic": [],
        "meta": {"saves": 0},
    }



def _normalise(data: dict) -> dict:
    profile = _empty_profile()
    if not isinstance(data, dict):
        return profile
    for section in SCALAR_SECTIONS:
        value = data.get(section)
        if isinstance(value, dict):
            profile[section] = {
                k: (m_identity.alias_list(v) if m_identity._is_alias(section, k) else v)
                for k, v in value.items()
                if k in BOOKKEEPING_KEYS or isinstance(v, str) or m_identity._is_alias(section, k)
            }
    if not profile["identity"].get(ALIAS_KEY):
        profile["identity"].pop(ALIAS_KEY, None)   # an empty alias list is no fact at all
    profile["updated"] = data.get("updated")
    backfill = _seen_default(profile["updated"])
    for section in LIST_SECTIONS:
        value = data.get(section)
        if isinstance(value, list):
            profile[section] = [
                e if e.get(SEEN_FIELD) else {**e, SEEN_FIELD: backfill}
                for e in value if isinstance(e, dict)
            ]
    meta = data.get("meta")
    if isinstance(meta, dict):
        profile["meta"].update(meta)
    return profile



def _seen_default(updated) -> str:
    """The `seen` date for an entry that predates the field: the profile's own `updated` day."""
    return (render._parse_date(updated) or util.today()).isoformat()



def load_profile() -> dict:
    """The profile from memory.json, normalised to the v2 schema.

    Never raises and never returns None: a missing, unreadable or half-written file yields an
    empty profile, so a broken memory file can degrade the assistant but not break it.
    """
    if not os.path.exists(MEMORY_PATH):
        return _empty_profile()
    try:
        with open(MEMORY_PATH) as f:
            return _normalise(json.load(f))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        log.warning(f"warning: could not read {os.path.basename(MEMORY_PATH)} ({exc}); starting empty")
        _back_up_corrupt(exc)
        return _empty_profile()


def _back_up_corrupt(exc) -> None:
    """Copy an unreadable memory.json aside once, before the next save overwrites it.

    Copy, not move: a missing memory.json + shortmem.txt reads as "migrate from v1".
    """
    try:
        backup = f"{MEMORY_PATH}.corrupt-{int(os.path.getmtime(MEMORY_PATH))}"
        if not os.path.exists(backup):
            shutil.copy2(MEMORY_PATH, backup)
    except OSError:
        pass
    trace.error("memory", "load", exc, fatal_for="none")



def _write_profile(profile: dict) -> None:
    """Atomic write: full file to `.tmp`, fsync, then os.replace."""
    profile["version"] = VERSION
    profile["updated"] = util.stamp("%H:%M")
    tmp = MEMORY_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile, f, indent=1, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, MEMORY_PATH)
    m_identity._ASR_CACHE["key"] = None      # the profile changed; the ASR prompt is rebuilt on next use



def has_content(profile: dict) -> bool:
    """True when the profile holds at least one real fact (superseded history doesn't count)."""
    for section in SCALAR_SECTIONS:
        if any(k not in BOOKKEEPING_KEYS for k in profile.get(section, {})):
            return True
    return any(profile.get(section) for section in LIST_SECTIONS)
