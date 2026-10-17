"""The vault on disk: git, atomic file writes, the folder layout, frontmatter and path rules (docs/19-wiki-layer.md; PRD 50 step 6, from wiki.py)."""
import hashlib
import os
import re
import subprocess
from vault import index_db
from memory import render
from core import config, trace

log = trace.logger("Wiki")


# --- vault -------------------------------------------------------------------------

WIKI_PATH = config.WIKI_PATH

INDEX_FILE = "index.md"
LOG_FILE = "log.md"
SCHEMA_FILE = "schema.md"
DECISIONS_FILE = "decisions.md"
RECURRING_FILE = "recurring.md"          # where archived `recurring` profile entries land
#: where the assistant may file pages; the user may also use any other single-level folder
FOLDERS = ("people", "projects", "meetings", "sources", "notes")

#: raw sources (PRD 20) live here: immutable originals, never pages; text-kind ones are
#: chunk-indexed for recall (PRD 32), never ingested as pages
RAW_DIR = "raw"
RAW_SUBDIRS = ("inbox", "screens", "email", "news")
#: per-sha chunk cache for search inside sources (PRD 32); gitignored, rebuilt on demand
RAW_INDEX_DIR = "raw/.index"
#: the search index (docs/53-scalable-index.md): `<vault>/.index/search.sqlite`, gitignored
INDEX_DIR = index_db.INDEX_DIR

#: pages that are machinery, not knowledge — never listed, never searched, never ingested about
RESERVED = (INDEX_FILE, LOG_FILE, SCHEMA_FILE)

#: fixed frontmatter keys, written on every page (Bases/Dataview-friendly; docs/17-long-term-memory.md)
FIELDS = ("type", "name", "tags", "source", "updated", "seen")
#: keys a page may carry but does not have to — written only once something sets them.
#: `status` is the project pages' (docs/19-wiki-layer.md); `entry_from_page` reads it back
#: into the profile and the index one-liner shows it.
OPTIONAL_FIELDS = ("status",)
#: what a `set_field` op may write
SETTABLE_FIELDS = FIELDS + OPTIONAL_FIELDS

MAX_PAGE_BYTES = 20 * 1024      # a page the assistant may still append to
MAX_NOTE_CHARS = 600            # one `## Notes` bullet
MAX_READ_BYTES = 4 * 1024       # read_page() ceiling
TRUNCATED = "\n\n[... truncated]"
WIKI_INDEX_TOKENS = int(os.getenv("WIKI_INDEX_TOKENS", "600"))
MAP_LINK_WEIGHT = 2              # vault map rank: inbound links × this + recency (docs/54-vault-map-research.md)
MAP_RECENCY_HALF_LIFE_D = 30     # … recency = 0.5 ** (days since `updated` / this)
MAP_TOP_TAGS = 3                 # tags named on a folder's header
MAP_ONE_LINER_CHARS = 100        # a map row's one-liner is cut here ("…"), so a folder still shows several pages
RELATED_TOKENS = int(os.getenv("WIKI_RELATED_TOKENS", "200"))
INDEX_MAX_LINES = 200

SECTIONS = ("## Summary", "## Notes", "## Related", "## History")

SCHEMA_TEXT = """---
type: schema
name: schema
tags: []
source: bootstrap
updated: -
seen: -
---

# Schema

This vault is maintained by an assistant and edited by a human in Obsidian.

## Division of labour

- The **profile** (memory.json) is who the user is: name, preferences, the handful of people and
  projects currently in play. Small, always in context, it decays.
- This **wiki** is what we are working on: meetings, decisions, people and project pages.
  Unbounded, durable, read on demand.

## Page templates

people/<slug>.md
```
---
type: person
name: Anna
tags: []
source: session 2026-08-29
updated: 2026-08-29
seen: 2026-08-29
---

# Anna

## Summary
Colleague on the platform team, joined 2026-08.

## Notes

## Related

## History
```

projects/<slug>.md — same, `type: project`, plus a `status: green|amber|red|paused|done`
frontmatter key and a status line in the Summary.
meetings/YYYY-MM-DD.md — `type: meeting`, one page per day.
sources/<slug>.md — `type: source`, one per filed document (PRD 20); the immutable original
lives under `raw/` and is linked from the page's frontmatter.
decisions.md — one `## YYYY-MM-DD <title>` section per decision.

## Rules

- File only what the USER said. Never the assistant's own words, never a guess, never world facts.
- Worth a page: people, projects, meetings, decisions, anything the user would expect you to
  remember next month.
- Not worth anything: meals, weather, small talk, moods, chores. Filing nothing is normal.
- Every note and decision carries a `source`.
- Never rewrite or delete a page: append.
- Frontmatter keys are exactly: type, name, tags, source, updated, seen — plus `status` on project pages.
"""


def _git(*args, quiet: bool = False):
    """Run git inside the vault, quietly. A missing or unhappy git prints, never raises."""
    try:
        result = subprocess.run(
            ["git", "-c", "user.name=llm-wiki", "-c", "user.email=llm-wiki@localhost", *args],
            cwd=WIKI_PATH, capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.info(f"git {args[0] if args else ''} failed: {exc}")
        return None
    if result.returncode != 0 and not quiet:
        message = (result.stderr or result.stdout or "").strip().splitlines()
        if message:
            log.info(f"git {args[0] if args else ''}: {message[0]}")
    return result


def _path(*parts) -> str:
    return os.path.join(WIKI_PATH, *parts)


def _write(rel: str, text: str) -> None:
    """Atomic: the bytes land in a sibling temp file first, so a crash never leaves half a page."""
    full = _path(rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    tmp = full + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, full)
    if v_search._is_page(rel):
        v_search.index_page(rel, text)


def etag(text: str) -> str:
    """The content hash the editor round-trips (PRD 28): mtime lies after git and our own writes."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _read(rel: str) -> str:
    try:
        with open(_path(rel)) as f:
            return f.read()
    except OSError:
        return ""


def exists() -> bool:
    return os.path.isdir(WIKI_PATH)


def ensure_vault() -> str:
    """Create the vault layout (and its git repo) if it isn't there yet; returns WIKI_PATH.

    Called lazily on the first write — never at import, so importing the app can't litter the
    user's home directory. An existing `schema.md` is never rewritten: it is the user's to edit.
    """
    fresh = not os.path.isdir(WIKI_PATH)
    os.makedirs(WIKI_PATH, exist_ok=True)
    folders = list(FOLDERS) + [os.path.join(RAW_DIR, sub) for sub in RAW_SUBDIRS]
    for folder in folders:
        os.makedirs(_path(folder), exist_ok=True)
        keep = os.path.join(folder, ".gitkeep")
        if not os.path.exists(_path(keep)):
            _write(keep, "")
    if not os.path.exists(_path(SCHEMA_FILE)):
        _write(SCHEMA_FILE, SCHEMA_TEXT)
    if not os.path.exists(_path(LOG_FILE)):
        _write(LOG_FILE, "# Log\n\n")
    if not os.path.exists(_path(DECISIONS_FILE)):
        _write(DECISIONS_FILE, _page_text(
            {"type": "decisions", "name": "Decisions", "tags": [], "source": "bootstrap",
             "updated": render._today_iso(), "seen": render._today_iso()},
            "# Decisions\n",
        ))
    ignore = _read(".gitignore")     # `.index/` (PRD 53) carries its own `*` .gitignore: no tracked edit here
    if f"{RAW_INDEX_DIR}/" not in ignore.splitlines():
        _write(".gitignore", (ignore or ".obsidian/\n.trash/\n").rstrip("\n") + f"\n{RAW_INDEX_DIR}/\n")
    if not os.path.exists(_path(INDEX_FILE)):
        v_index.rebuild_index()
    if not os.path.isdir(_path(".git")):
        _git("init", "-q")
        _commit("wiki: initial vault")
    elif fresh:
        _commit("wiki: vault layout")
    return WIKI_PATH


def _commit(message: str) -> None:
    """One commit for the whole batch. "nothing to commit" is normal and stays silent."""
    _git("add", "-A")
    _git("commit", "-q", "-m", message, quiet=True)

# --- frontmatter -------------------------------------------------------------------
# A deliberately tiny YAML subset: `key: scalar` and `tags: [a, b]`. Enough for the six
# fixed keys, no dependency, and it round-trips whatever Obsidian writes for those keys.

_FM_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)$")


def parse_page(text: str) -> tuple:
    """``(frontmatter dict, body str)``. A page without frontmatter yields ``({}, text)``."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, text
    fm = {}
    for line in lines[1:end]:
        match = _FM_LINE.match(line)
        if not match:
            continue
        key, raw = match.group(1), match.group(2).strip()
        if raw.startswith("[") and raw.endswith("]"):
            fm[key] = [p.strip().strip("'\"") for p in raw[1:-1].split(",") if p.strip()]
        else:
            fm[key] = raw.strip("'\"")
    return fm, "\n".join(lines[end + 1:]).lstrip("\n")


def _fm_text(fm: dict) -> str:
    lines = ["---"]
    for key in FIELDS:
        value = fm.get(key, [] if key == "tags" else "")
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        else:
            lines.append(f"{key}: {value}")
    for key, value in fm.items():        # keys a human added in Obsidian survive a rewrite
        if key in FIELDS:
            continue
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines)


def _page_text(fm: dict, body: str) -> str:
    return f"{_fm_text(fm)}\n\n{body.lstrip()}"


def _edit_page(rel: str, body_fn=None, defaults: dict | None = None, **fields) -> None:
    """The one read-modify-write path for an existing page.

    Bumps `updated`, fills `defaults` in where a frontmatter key is missing, applies `fields`
    over the frontmatter and `body_fn` over the body. No op formats a whole page itself, so
    "never rewrite, only append" is enforced in one place.
    """
    fm, body = parse_page(_read(rel))
    fm["updated"] = render._today_iso()
    for key, value in (defaults or {}).items():
        fm.setdefault(key, value)
    fm.update(fields)
    _write(rel, _page_text(fm, body_fn(body) if body_fn else body))

# --- paths -------------------------------------------------------------------------

_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-")
    return slug or "untitled"


def normalise_path(path, user: bool = False) -> str | None:
    """``"people/Anna"`` -> ``"people/anna.md"``; ``None`` when the path is not vault-safe.

    Rejects anything absolute, anything with ``..``, anything deeper than ``folder/page.md``
    and any segment that is not a slug — the model proposes these paths, so they are untrusted.
    Assistant ops may only file under `FOLDERS`; a `user=True` op (the editor, PRD 28) may use
    any single-level folder except `raw/`.
    """
    raw = str(path or "").strip().replace("\\", "/").strip("/")
    if not raw or raw.startswith(".") or ":" in raw:
        return None
    if not raw.endswith(".md"):
        raw += ".md"
    parts = raw.split("/")
    if len(parts) > 2 or any(p in ("", ".", "..") for p in parts):
        return None
    parts = [p.lower() for p in parts]
    if len(parts) == 2 and (parts[0] == RAW_DIR if user else parts[0] not in FOLDERS):
        return None
    if not all(_SLUG.match(p) for p in parts):
        return None
    return "/".join(parts)


def wikilink(rel: str) -> str:
    """The public form of a page path: `people/anna.md` -> `people/anna`, as wikilinks write it."""
    return rel.removesuffix(".md")


def page_exists(rel: str) -> bool:
    return os.path.isfile(_path(rel))


def _has_decisions(text: str) -> bool:
    _, body = parse_page(text)
    return any(line.startswith("## ") for line in body.splitlines())


def page_paths() -> list:
    """Every knowledge page in the vault, sorted. Machinery and an empty decisions.md are out."""
    if not exists():
        return []
    found = []
    for root, dirs, files in os.walk(WIKI_PATH):
        dirs[:] = [d for d in dirs if not d.startswith(".") and
                   os.path.join(root, d) != _path(RAW_DIR)]
        for name in files:
            if not name.endswith(".md"):
                continue
            rel = os.path.relpath(os.path.join(root, name), WIKI_PATH).replace(os.sep, "/")
            if rel in RESERVED:
                continue
            if rel == DECISIONS_FILE and not _has_decisions(_read(rel)):
                continue
            found.append(rel)
    return sorted(found)


# Imported last: index and search read store's constants at import time, store reaches them only inside functions.
from vault import index as v_index  # noqa: E402
from vault import search as v_search  # noqa: E402
