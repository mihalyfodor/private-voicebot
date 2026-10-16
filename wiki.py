"""Wiki layer — an Obsidian-compatible markdown vault the assistant maintains.

The vault (``WIKI_PATH``, default ``~/llm-wiki``) is the *durable* half of long-term memory:
the profile in memory.py is "who the user is" (small, always in the prompt, decays), the vault
is "what we are working on" (unbounded, on demand, human co-edited in Obsidian).

Only ``index.md`` is loaded into the prompt; pages are reached through two tools
(``search_wiki``, ``read_page``), so recall costs nothing until it is needed.

The model never writes a file. At memory-save time :func:`ingest` asks it for a JSON array of
**page ops**; :func:`apply_page_ops` validates and applies them, regenerates ``index.md`` from
the pages' frontmatter, appends to ``log.md`` and makes one ``git commit``. Same lesson as
PRD 15: ops + validation, never a whole-file rewrite. ``git log`` inside the vault is the undo.

See docs/19-wiki-layer.md.
"""
import contextlib
import hashlib
import json
import os
import re
import subprocess
from collections import Counter

import index_db
import memory
from core import config, settings, trace, vault_lock
from core import llm as core_llm
from core.untrusted import DOCUMENT_RULE
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
    if _is_page(rel):
        index_page(rel, text)


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
             "updated": memory._today_iso(), "seen": memory._today_iso()},
            "# Decisions\n",
        ))
    ignore = _read(".gitignore")     # `.index/` (PRD 53) carries its own `*` .gitignore: no tracked edit here
    if f"{RAW_INDEX_DIR}/" not in ignore.splitlines():
        _write(".gitignore", (ignore or ".obsidian/\n.trash/\n").rstrip("\n") + f"\n{RAW_INDEX_DIR}/\n")
    if not os.path.exists(_path(INDEX_FILE)):
        rebuild_index()
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
    fm["updated"] = memory._today_iso()
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


# --- index -------------------------------------------------------------------------


def _summary_line(body: str) -> str:
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() != "## summary":
            continue
        for follow in lines[i + 1:]:
            if follow.startswith("## "):
                break
            if follow.strip():
                return follow.strip()
        break
    return ""


def _one_liner(fm: dict, body: str) -> str:
    """The description in `index.md`: frontmatter identity, then the first Summary line."""
    meta = ", ".join(str(fm[k]).strip() for k in ("name", "type", "status")
                     if str(fm.get(k, "")).strip())
    summary = _summary_line(body)
    return " — ".join(p for p in (meta, summary) if p)


def index_lines() -> list:
    """`index.md`'s lines, from the search index's `pages` table: no page is read."""
    if not exists():
        return []
    return [f"- [[{wikilink(rel)}]]" + (f" — {described}" if described else "")
            for rel, described in _db().index_rows(INDEX_MAX_LINES)]


def rebuild_index() -> str:
    """Regenerate `index.md` from the pages' frontmatter — a pure function of the vault."""
    lines = index_lines()
    text = "# Index\n\n" + ("\n".join(lines) + "\n" if lines else "(no pages yet)\n")
    _write(INDEX_FILE, text)
    return text


def index_text(max_tokens: int = WIKI_INDEX_TOKENS) -> str:
    """The index trimmed to `max_tokens` — what goes into the prompt and into the ingest call.

    Built from the search index's `pages` table (no page read) rather than read from `index.md`,
    so every write hook's change is already reflected. "" when there is nothing to show.
    """
    if not exists():
        return ""
    lines = index_lines()
    if not lines:
        return ""
    lines = ["# Index", ""] + lines
    while lines and memory.tokens_est("\n".join(lines)) > max_tokens:
        lines.pop()
    return "\n".join(lines).strip()


def related_context(text: str, k: int = 5) -> str:
    """The pages most related to `text`, as a prompt block for the ingest calls (PRD 44).

    The index is trimmed to `WIKI_INDEX_TOKENS`, so a page it cut is still named here; `sources/`
    pages are left out. Trimmed line by line to `RELATED_TOKENS`; "" when there is no hit.
    """
    hits = [h for h in search(text, k + 3) if not h["path"].startswith("sources/")][:k]
    lines = []
    for hit in hits:
        fm, body = parse_page(_read(hit["path"] + ".md"))
        line = f"- {hit['path']}: {fm.get('name') or hit['path'].rsplit('/', 1)[-1]}"
        summary = _summary_line(body)[:120]
        lines.append(line + (f" — {summary}" if summary else ""))
    head = "Existing related pages (update these rather than duplicating):"
    while lines and memory.tokens_est("\n".join([head] + lines)) > RELATED_TOKENS:
        lines.pop()
    return "\n".join([head] + lines) if lines else ""


def _log(entries: list) -> None:
    if not entries:
        return
    today = memory._today_iso()
    with open(_path(LOG_FILE), "a") as f:
        for entry in entries:
            f.write(f"- {today} {entry}\n")


# --- page ops ----------------------------------------------------------------------
# Each op returns (log line, None) when applied, or (None, reason) when refused.


def _check_size(rel: str, addition: str) -> str | None:
    size = len(_read(rel).encode()) + len(addition.encode())
    if size > MAX_PAGE_BYTES:
        return f"page {wikilink(rel)} would exceed {MAX_PAGE_BYTES // 1024} kB"
    return None


def _insert_into_section(body: str, heading: str, line: str) -> str:
    """Append `line` at the end of `heading`'s section, creating the section if needed."""
    lines = body.rstrip("\n").splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().lower() == heading.lower()), None)
    if start is None:
        return "\n".join(lines + ["", heading, line]).lstrip("\n") + "\n"
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    tail = end
    while tail > start + 1 and not lines[tail - 1].strip():
        tail -= 1
    return "\n".join(lines[:tail] + [line] + lines[tail:]) + "\n"


def _target(op: dict, key: str = "path", *, exists: bool | None = None,
            reserved: bool = False, user: bool = False) -> tuple:
    """Validate an op's (untrusted) path: ``(rel, None)`` or ``(None, reason)``.

    `exists=True` demands the page is there, `exists=False` demands it is not; `reserved=True`
    additionally refuses the machinery pages; `user=True` applies the editor's folder rule.
    Every path check goes through here, so the rejection reasons are worded the same way
    everywhere.
    """
    rel = normalise_path(op.get(key), user=user)
    if rel is None:
        return None, f"bad path {op.get(key)!r}"
    if reserved and rel in RESERVED:
        return None, f"{wikilink(rel)} is not a page"
    if exists is True and not page_exists(rel):
        return None, f"no such page {wikilink(rel)}"
    if exists is False and page_exists(rel):
        return None, f"{wikilink(rel)} already exists"
    return rel, None


def new_page(op: dict, source: str) -> tuple:
    """`{path, type, name, summary}` → a fresh page with the six frontmatter keys and sections."""
    rel, reason = _target(op, exists=False, reserved=True)
    if reason:
        return None, reason
    page_type = str(op.get("type") or "note").strip().lower()
    name = str(op.get("name") or "").strip() or os.path.basename(rel)[:-3]
    summary = str(op.get("summary") or "").strip()
    if len(summary) > MAX_NOTE_CHARS:
        return None, f"summary longer than {MAX_NOTE_CHARS} characters"
    _write(rel, _fresh_page(page_type, name, summary, source))
    return f"filed [[{wikilink(rel)}]] (source: {source})", None


def _fresh_page(page_type: str, name: str, summary: str, source: str) -> str:
    today = memory._today_iso()
    fm = {"type": page_type, "name": name, "tags": [], "source": source,
          "updated": today, "seen": today}
    return _page_text(fm, f"# {name}\n\n## Summary\n{summary}\n\n## Notes\n\n## Related\n")


def append_note(op: dict, source: str) -> tuple:
    """`{path, text, section?}` → one dated bullet at the end of `## Notes` (or another section)."""
    rel, reason = _target(op, exists=True, reserved=True)
    if reason:
        return None, reason
    text = " ".join(str(op.get("text") or "").split())
    if not text:
        return None, "empty note"
    if len(text) > MAX_NOTE_CHARS:
        return None, f"note longer than {MAX_NOTE_CHARS} characters"
    heading = str(op.get("section") or "## Notes")
    if heading not in SECTIONS:
        return None, f"unknown section {heading!r}"
    today = memory._today_iso()
    line = f"- {today} {text} (source: {source})"
    oversize = _check_size(rel, line)
    if oversize:
        return None, oversize
    _edit_page(rel, lambda body: _insert_into_section(body, heading, line),
               defaults={"seen": today})
    return f"noted on [[{wikilink(rel)}]] (source: {source})", None


def set_field(op: dict, source: str) -> tuple:
    """`{path, key, value}` → one frontmatter key, from the fixed set. The body is untouched."""
    rel, reason = _target(op, exists=True)
    if reason:
        return None, reason
    key = str(op.get("key") or "").strip()
    if key not in SETTABLE_FIELDS:
        return None, f"unknown frontmatter key {key!r}"
    value = op.get("value")
    if key == "tags":
        value = value if isinstance(value, list) else [p.strip() for p in str(value or "").split(",") if p.strip()]
    else:
        value = " ".join(str(value or "").split())
        if len(value) > MAX_NOTE_CHARS:
            return None, f"value longer than {MAX_NOTE_CHARS} characters"
    _edit_page(rel, **{key: value})     # `updated` is bumped unless it *is* the key being set
    return f"set {key} on [[{wikilink(rel)}]] (source: {source})", None


def add_decision(op: dict, source: str) -> tuple:
    """`{title, text}` → a dated `## YYYY-MM-DD <title>` section appended to `decisions.md`."""
    title = " ".join(str(op.get("title") or "").split())
    text = " ".join(str(op.get("text") or "").split())
    if not title:
        return None, "decision needs a title"
    if len(text) > MAX_NOTE_CHARS:
        return None, f"decision longer than {MAX_NOTE_CHARS} characters"
    today = memory._today_iso()
    block = f"\n## {today} {title}\n{text}\nsource: {source}\n"
    oversize = _check_size(DECISIONS_FILE, block)
    if oversize:
        return None, oversize
    _edit_page(DECISIONS_FILE, lambda body: body.rstrip("\n") + "\n" + block,
               defaults={"type": "decisions", "name": "Decisions"})
    return f"decided {title} in [[decisions]] (source: {source})", None


def link(op: dict, source: str) -> tuple:
    """`{path, target}` → a `[[target]]` bullet under `## Related`, once."""
    rel, target = normalise_path(op.get("path")), normalise_path(op.get("target"))
    if rel is None or target is None:
        return None, f"bad path {op.get('path')!r} -> {op.get('target')!r}"
    if not page_exists(rel):
        return None, f"no such page {wikilink(rel)}"
    if rel == target:
        return None, "a page cannot link to itself"
    mark = f"[[{wikilink(target)}]]"
    if mark in parse_page(_read(rel))[1]:
        return None, f"{wikilink(rel)} already links to {wikilink(target)}"
    _edit_page(rel, lambda body: _insert_into_section(body, "## Related", f"- {mark}"))
    return f"linked [[{wikilink(rel)}]] -> {mark} (source: {source})", None


OPS = {
    "new_page": new_page,
    "append_note": append_note,
    "set_field": set_field,
    "add_decision": add_decision,
    "link": link,
}

trace.vocab("kind", OPS)


def op_kind(op) -> str:
    """A rejected op's kind for the trace: one of OPS, else "other" (the op itself is document text)."""
    kind = str(op.get("op") if isinstance(op, dict) else "").strip().lower()
    return kind if kind in OPS else "other"


def log_rejected(bad_ops: list, out) -> list:
    """Log each rejected op on `out` by kind and reason code (the op text only at DEBUG); returns the
    trace's `ops_rejected` shape `[{kind, reason_code}]`."""
    shape = []
    for bad in bad_ops:
        short = {"kind": op_kind(bad.get("op")), "reason_code": trace.reason_code(bad.get("reason"))}
        out.info(f"rejected {short['kind']}: {short['reason_code']}")
        out.debug(f"rejected {bad['op']}: {bad['reason']}")
        shape.append(short)
    return shape


#: bumped by every applied batch, so callers can tell "did this save file anything?"
_applied_batches = 0


def applied_count() -> int:
    return _applied_batches


def apply_page_ops(ops, source_default: str = "", commit: bool = True) -> tuple:
    """Validate and apply `ops`; returns ``(applied, rejected)``.

    A batch that applied at least one op regenerates `index.md`, appends one `log.md` line per
    applied op and makes exactly one commit. A fully rejected batch touches nothing.

    `commit=False` skips only the commit: the caller (sources.ingest) is writing more files
    around this batch and wants one commit for the whole ingest.
    """
    if not isinstance(ops, list) or not ops:
        return [], []
    with vault_lock.LOCK:      # re-entrant: memory.save and sources.file_bytes already hold it
        return _apply_page_ops(ops, source_default, commit)


def _apply_page_ops(ops: list, source_default: str, commit: bool) -> tuple:
    global _applied_batches
    applied, rejected = [], []
    ensure_vault()
    source_default = source_default or f"session {memory._today_iso()}"
    entries = []
    for raw in ops:
        op = raw if isinstance(raw, dict) else {}
        kind = str(op.get("op") or "").strip().lower()
        handler = OPS.get(kind)
        if handler is None:
            rejected.append({"op": raw, "reason": f"unknown op {op.get('op')!r}"})
            continue
        source = " ".join(str(op.get("source") or "").split()) or source_default
        if not source:
            rejected.append({"op": raw, "reason": "source is required"})
            continue
        try:
            entry, reason = handler(op, source)
        except OSError as exc:
            entry, reason = None, str(exc)
        if reason:
            rejected.append({"op": raw, "reason": reason})
        else:
            applied.append({**op, "op": kind, "source": source})
            entries.append(entry)
    if applied:
        rebuild_index()
        _log(entries)
        if commit:
            _commit(f"wiki: {len(applied)} op{'s' if len(applied) != 1 else ''} ({source_default})")
        _applied_batches += 1
    return applied, rejected


# --- editor (PRD 28) ---------------------------------------------------------------
# The user is the only whole-page writer. Every call takes `vault_lock.LOCK` so an ingest and a
# save never interleave, and ends in exactly one commit; git is the undo.

USER_SOURCE = "user edit"
_UPDATED_LINE = re.compile(r"^updated:.*$", re.M)
_TYPES = {"people": "person", "projects": "project", "meetings": "meeting", "sources": "source"}


def _bump_updated(text: str) -> str:
    """The `updated:` line inside the frontmatter set to today; every other byte untouched."""
    end = text.find("\n---", 3)
    if not text.startswith("---") or end == -1:
        return text
    head = _UPDATED_LINE.sub(f"updated: {memory._today_iso()}", text[:end], count=1)
    return head + text[end:]


def _user_target(path, *, exists: bool, fixed: bool = True) -> tuple:
    """A user-supplied path: ``(rel, None)`` or ``(None, reason)``; `fixed` also refuses the pages
    the layer relies on (machinery + `decisions`)."""
    rel = normalise_path(path, user=True)
    if rel is not None and (rel in RESERVED or (fixed and rel == DECISIONS_FILE)):
        return None, f"{wikilink(rel)} is not an editable page"
    return _target({"path": path}, exists=exists, user=True)


def _user_commit(entry: str, message: str) -> None:
    rebuild_index()
    _log([entry])
    _commit(message)


def save_page(path: str, text: str, tag: str) -> tuple:
    """Whole-page save; ``(result, None)`` or ``(None, reason)``.

    `result` is ``{"etag": new}`` — or ``{"conflict": True, "text": current, "etag": current}``
    when `tag` no longer matches the bytes on disk. The body is written byte-for-byte except
    the `updated:` line; validation is the page contract (frontmatter keys, size).
    """
    with vault_lock.LOCK:
        rel, reason = _user_target(path, exists=True, fixed=False)
        if reason:
            return None, reason
        current = _read(rel)
        if etag(current) != str(tag or ""):
            return {"conflict": True, "text": current, "etag": etag(current)}, None
        text = str(text or "")
        if len(text.encode()) > MAX_PAGE_BYTES:
            return None, f"page larger than {MAX_PAGE_BYTES // 1024} kB"
        fm, _ = parse_page(text)
        missing = [key for key in FIELDS if key not in fm]
        if missing:
            return None, f"frontmatter missing {', '.join(missing)}"
        if text == current:
            return {"etag": etag(current)}, None
        text = _bump_updated(text)
        _write(rel, text)
        _user_commit(f"edited [[{wikilink(rel)}]] (source: {USER_SOURCE})",
                     f"wiki: edit {wikilink(rel)} ({USER_SOURCE})")
        return {"etag": etag(text)}, None


def create_page(path: str) -> tuple:
    """A fresh page at a user path (any single-level folder); ``(path, None)`` or ``(None, reason)``."""
    with vault_lock.LOCK:
        ensure_vault()
        rel, reason = _user_target(path, exists=False)
        if reason:
            return None, reason
        folder, base = os.path.split(rel)
        _write(rel, _fresh_page(_TYPES.get(folder, "note"), base[:-3], "", USER_SOURCE))
        _user_commit(f"created [[{wikilink(rel)}]] (source: {USER_SOURCE})",
                     f"wiki: create {wikilink(rel)} ({USER_SOURCE})")
        return wikilink(rel), None


CARD_SOURCE = "written answer"


def save_card(path: str, title: str, markdown: str, sources: list) -> tuple:
    """A written answer filed as a note (PRD 43): ``(path, None)`` or ``(None, reason)``.

    One validated user write (any single-level folder but `raw/`, no existing page) and one
    commit: frontmatter `type: note`, `name: <title>`, `source: written answer`, `sources:` the
    card's list; the body is the card markdown as it was rendered.
    """
    with vault_lock.LOCK:
        ensure_vault()
        rel, reason = _user_target(path, exists=False)
        if reason:
            return None, reason
        today = memory._today_iso()
        fm = {"type": "note", "name": title, "tags": [], "source": CARD_SOURCE, "updated": today,
              "seen": today, "sources": [str(x).replace(",", " ") for x in sources]}
        body = markdown.strip() + "\n"
        if not body.startswith("# "):
            body = f"# {title}\n\n" + body
        text = _page_text(fm, body)
        if len(text.encode()) > MAX_PAGE_BYTES:
            return None, f"page larger than {MAX_PAGE_BYTES // 1024} kB"
        _write(rel, text)
        _user_commit(f"filed [[{wikilink(rel)}]] (source: {CARD_SOURCE})",
                     f"wiki: save card {wikilink(rel)} ({CARD_SOURCE})")
        return wikilink(rel), None


def _link_pattern(target: str) -> re.Pattern:
    """`[[target]]`, `[[target|…]]`, `[[target#…]]` and `![[target…]]` — the exact path only."""
    return re.compile(r"(!?\[\[)" + re.escape(target) + r"(?=[\]|#])")


def rename_page(path: str, new_path: str) -> tuple:
    """Move a page and rewrite the links to it; ``({path, links}, None)`` or ``(None, reason)``.

    Rewritten: every page's exact-path links, plus the short form `[[<basename>…]]` when that
    basename is unique in the vault. Touched pages get a fresh `updated:`.
    """
    with vault_lock.LOCK:
        rel, reason = _user_target(path, exists=True)
        if reason:
            return None, reason
        new_rel, reason = _user_target(new_path, exists=False)
        if reason:
            return None, reason
        old, new = wikilink(rel), wikilink(new_rel)
        basename = os.path.basename(old)
        unique = sum(os.path.basename(p)[:-3] == basename for p in page_paths()) == 1
        patterns = [_link_pattern(old)]
        if unique and basename != old:
            patterns.append(_link_pattern(basename))
        touched = 0
        for other in page_paths():
            if other == rel:
                continue
            text = _read(other)
            rewritten = text
            for pattern in patterns:
                rewritten = pattern.sub(lambda m: m.group(1) + new, rewritten)
            if rewritten != text:
                _write(other, _bump_updated(rewritten))
                touched += 1
        moved = _bump_updated(_read(rel))
        _write(new_rel, moved)          # the new file first: a failing index hook never loses the page
        os.remove(_path(rel))
        unindex_page(rel)
        _user_commit(f"renamed [[{old}]] → [[{new}]] ({touched} links, source: {USER_SOURCE})",
                     f"wiki: rename {old} → {new} ({touched} links)")
        return {"path": new, "links": touched}, None


def delete_page(path: str) -> tuple:
    """Remove a page (one commit); ``(path, None)`` or ``(None, reason)``. Lint reports the broken links."""
    with vault_lock.LOCK:
        rel, reason = _user_target(path, exists=True)
        if reason:
            return None, reason
        _git("rm", "-q", "-f", "--", rel, quiet=True)
        if os.path.exists(_path(rel)):
            os.remove(_path(rel))
        unindex_page(rel)
        _user_commit(f"deleted [[{wikilink(rel)}]] (source: {USER_SOURCE})",
                     f"wiki: delete {wikilink(rel)}")
        return wikilink(rel), None


# --- search ------------------------------------------------------------------------
# BM25 over page bodies + frontmatter. Personal scale (≤ 200 pages), no embeddings, no new
# dependency, and deterministic — which is what makes it testable.

def _stem(word: str) -> str:
    """A light stem so "lockpick" finds "lockpicking" and "turret" "turrets" — no dependency."""
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word


def _tokenise(text: str) -> list:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())]


def _db() -> index_db.Index:
    """The vault's search index (docs/53-scalable-index.md). The first open in this process runs
    the consistency check, so a vault without an index file (or with a stale one) gets one."""
    idx = index_db.open_index(WIKI_PATH)
    if not idx.checked:
        idx.checked = True
        try:
            _check(idx)
        except BaseException:
            idx.checked = False
            raise
    return idx


def _hook_db() -> index_db.Index | None:
    """The index for a write hook; None when the vault does not exist yet (nothing to index)."""
    return _db() if exists() else None


def _is_page(rel: str) -> bool:
    """What :func:`page_paths` would list, by path alone (an empty decisions.md is decided on its text)."""
    parts = rel.split("/")
    return (rel.endswith(".md") and rel not in RESERVED and parts[0] != RAW_DIR
            and not any(p.startswith(".") for p in parts[:-1]))


def _sha(text: str) -> str:
    """Hex SHA-256 of a text."""
    return hashlib.sha256(text.encode()).hexdigest()


def _put_page(idx: index_db.Index, rel: str, text: str, mtime: int) -> None:
    """Write one page's row and postings from its text."""
    fm, body = parse_page(text)
    fields = {"type": str(fm.get("type") or ""), "name": str(fm.get("name") or ""),
              "tags": json.dumps(tags(text)), "updated": str(fm.get("updated") or ""),
              "one_liner": _one_liner(fm, body)}
    idx.put_page(rel, mtime, _sha(text), fields, _tokenise(text))


def index_page(rel: str, text: str | None = None) -> None:
    """Write hook: (re)index one page after its file was written (`_write` calls it)."""
    if not _is_page(rel) or (idx := _hook_db()) is None:
        return
    text = _read(rel) if text is None else text
    if not os.path.isfile(_path(rel)) or (rel == DECISIONS_FILE and not _has_decisions(text)):
        idx.drop_page(rel)
        return
    _put_page(idx, rel, text, os.stat(_path(rel)).st_mtime_ns)


def unindex_page(rel: str) -> None:
    """Write hook: a page file was removed (rename, delete, forget); its rows go."""
    if _is_page(rel) and (idx := _hook_db()) is not None:
        idx.drop_page(rel)


def _check(idx: index_db.Index, *, batch: bool | None = None) -> int:
    """Compare the index to the vault and repair the difference; returns the documents changed.

    Pages key on (mtime, sha): an unchanged mtime costs one stat, a changed one a read and a hash.
    Sources key on the manifest: shas gone are dropped, shas new are indexed from their chunk cache.
    """
    with idx.write() if (idx.fresh if batch is None else batch) else contextlib.nullcontext():
        stored = idx.page_stamps()
        changed = 0
        for rel in page_paths():
            mtime = os.stat(_path(rel)).st_mtime_ns
            known = stored.pop(rel, None)
            if known and known[0] == mtime:
                continue
            text = _read(rel)
            if known and known[1] == _sha(text):
                idx.touch_page(rel, mtime)
                continue
            _put_page(idx, rel, text, mtime)
            changed += 1
        for rel in stored:
            idx.drop_page(rel)
            changed += 1
        changed += _sync_sources(idx)
    idx.fresh = False
    return changed


def startup_check() -> int:
    """The consistency check, run once at app (and batch-import) start; returns the documents
    repaired. A vault that does not exist yet is left alone (it is created lazily)."""
    if not exists():
        return 0
    idx = index_db.open_index(WIKI_PATH)
    changed = _check(idx)
    idx.checked = True                  # only after it ran: a failed check is retried on the next open
    idx.set_meta("checked_at", memory._stamp("%H:%M:%S"))
    return changed


def reindex(path: str | None = None) -> str:
    """Rebuild the search index from scratch (pages on disk + the manifest); returns the file.

    `path` builds into another file and leaves the live one alone (the rebuild-equals-incremental
    test); otherwise the new file replaces the live index. `python -m wiki reindex` runs this.
    """
    if not exists():
        raise FileNotFoundError(f"no vault at {WIKI_PATH}")
    live = index_db.db_path(WIKI_PATH)
    target = path or live + ".new"
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(target + suffix):
            os.remove(target + suffix)
    idx = index_db.Index(target)
    try:
        _check(idx, batch=True)
    finally:
        idx.close()
    if path:
        return path
    index_db.replace(WIKI_PATH, target)
    _db()
    return live


def backlinks(path: str) -> list:
    """The pages (suffix-less) whose wikilinks point at `path`; alias/heading forms count.

    Only the pages the index says hold every word of the target are read."""
    rel = normalise_path(path, user=True)
    if rel is None or not exists():
        return []
    target = wikilink(rel)
    idx, words = _db(), set(_tokenise(target))
    candidates = idx.docs_with_all(index_db.PAGE, words) if words else idx.page_paths()
    found = []
    for other in sorted(candidates):
        if other != rel and target in links_in(_read(other)):
            found.append(wikilink(other))
    return found


_TAG = re.compile(r"(?<![\w/#&])#([A-Za-z][\w/-]*)")


def tags(text: str) -> list:
    """Frontmatter `tags:` merged with the body's `#tag`s, sorted, de-duplicated."""
    fm, body = parse_page(text)
    listed = fm.get("tags") or []
    found = {str(t).strip().lstrip("#") for t in (listed if isinstance(listed, list) else [listed])}
    found |= set(_TAG.findall(body))
    return sorted(t for t in found if t)


def _snippet(text: str, terms: set, width: int = 180) -> str:
    body = " ".join(parse_page(text)[1].split())
    lower = body.lower()
    at = min((lower.find(t) for t in terms if lower.find(t) >= 0), default=0)
    start = max(0, at - 40)
    return ("…" if start else "") + body[start:start + width] + ("…" if len(body) > start + width else "")


def _bm25(docs: list, terms: set, weights: dict | None = None) -> list:
    """BM25 score of `terms` against each token list in `docs` (0.0 where nothing matched);
    `weights` scales a term's contribution (default 1.0). In memory, for a handful of sections
    (:func:`best_section`); the vault-wide searches score from the index (`index_db.Index.bm25`)."""
    n = len(docs)
    weights = weights or {}
    avgdl = sum(len(t) for t in docs) / n
    df = Counter(term for tokens in docs for term in terms if term in tokens)
    scores = []
    for tokens in docs:
        counts, dl, score = Counter(tokens), len(tokens) or 1, 0.0
        for term in terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            score += index_db.bm25_term(weights.get(term, 1.0), freq, dl, avgdl, df[term], n)
        scores.append(score)
    return scores


def _page_subset(idx: index_db.Index, filters: dict | None):
    """The pages `filters` keeps — ``{type, tags, updated_from, updated_to}`` — or None (all)."""
    if not filters:
        return None
    want_type = filters.get("type")
    types = {want_type} if isinstance(want_type, str) else set(want_type or ())
    want_tags = filters.get("tags")
    tagset = {want_tags} if isinstance(want_tags, str) else set(want_tags or ())
    since, until = str(filters.get("updated_from") or ""), str(filters.get("updated_to") or "")
    keep = set()
    for rel, (kind, tag_json, updated) in idx.page_fields().items():
        if types and kind not in types:
            continue
        if tagset and not tagset <= set(json.loads(tag_json or "[]")):
            continue
        if (since and updated < since) or (until and updated > until):
            continue
        keep.add(rel)
    return keep


def search(query: str, k: int = 5, filters: dict | None = None) -> list:
    """Top-`k` pages for `query` as ``[{path, snippet, score}]`` (BM25, best first).

    `path` is the suffix-less form the model sees everywhere else (`people/anna`), so a hit can
    be handed straight to :func:`read_page`. Raw sources are a separate index: :func:`search_chunks`.
    `filters` (``{type, tags, updated_from, updated_to}``) narrows the pages scored; the BM25
    statistics are then the narrowed set's. Reads the index plus the `k` pages it returns.
    """
    terms = set(_tokenise(query))
    if not exists() or not terms:
        return []
    idx = _db()
    if not idx.page_lengths():
        return []
    scores = idx.bm25(index_db.PAGE, terms, None, _page_subset(idx, filters))
    ranked = sorted(((round(score, 3), wikilink(rel), rel) for rel, (score, _) in scores.items() if score > 0),
                    key=lambda r: (-r[0], r[1]))[:k]
    return [{"path": path, "snippet": _snippet(_read(rel), terms), "score": score}
            for score, path, rel in ranked]


# --- search inside sources (docs/32-source-recall.md) -------------------------------
# Text-kind raw files are split into ~800-token chunks, cached per sha in raw/.index/, and
# searched with the same BM25 as the pages. Pages and chunks are ranked separately.

CHUNK_TOKENS = 800
CHUNK_OVERLAP = 80
SNIPPET_CHARS = 600
MAX_SOURCE_READ = 3000
#: `recall_min_score` default — BM25 per content word, calibrated in tests/test_source_recall.py
RECALL_MIN_SCORE = 4.5
#: a content question matched on plain words only answers grounded from this score (the recall band,
#: docs/45-ask-before-lookup.md); below it a spoken turn asks first
RECALL_HIGH_SCORE = 8.0
#: query words that carry no topic; dropped before scoring so "the" cannot dilute a match
STOPWORDS = frozenset("""a an the and or but if so of to in on at by for from with about into over
after before as is are am was were be been being do does did have has had i me my mine we us our you
your he she it its they them their this that these those what which who whom how when where why
there here can could should would will shall may might must not no yes im ive dont cant lets s m t
re ll d ve keep keeps get got just really very some any all one doe up off per accord exactly thi thes""".split())   # thi/thes: this/these stemmed
#: a query word the sources write Title-case ("Covenant", "Whiterun") names a place, a person
#: or a quest; it counts up to this much more than a plain word ("cook", "weather"), which is
#: what separates "I'm at Covenant" from "what should we cook" when each matches one rare word
PROPER_BOOST = 2.0
#: a query word written Title-case at least this often in the sources is a proper noun for the
#: recall precision rule (PRD 34): one such match may carry a chunk, one plain word may not
PROPER_RATIO_MIN = 0.5
_WORD = re.compile(r"[A-Za-z0-9]+")


def _read_bytes(rel: str) -> bytes:
    with open(_path(rel), "rb") as f:
        return f.read()


_PARA = re.compile(r"(?:[^\n]*\S[^\n]*(?:\n|\Z))+")


def _blocks(text: str) -> list:
    """``[(start, block)]`` — runs of non-blank lines (paragraphs, headings), offsets into `text`."""
    return [(m.start(), m.group()) for m in _PARA.finditer(text)]


def _split_long(start: int, block: str, limit: int) -> list:
    """A block over `limit` chars is cut at line boundaries (or hard, for one endless line)."""
    parts, buf, buf_at, at = [], "", start, start
    for line in block.splitlines(keepends=True):
        if buf and len(buf) + len(line) > limit:
            parts.append((buf_at, buf))
            buf = ""
        while len(line) > limit:
            parts.append((at, line[:limit]))
            line, at = line[limit:], at + limit
        if not buf:
            buf_at = at
        buf += line
        at += len(line)
    if buf:
        parts.append((buf_at, buf))
    return parts


def _carry_from(buf_at: int, buf: str, tail: int) -> int:
    """Where the overlap carried into the next chunk starts: the last `tail` chars of the
    chunk, moved forward to a line start (or a word start when the tail is one long line)."""
    body = buf.rstrip()
    if not tail or not body:
        return buf_at + len(buf)
    at = max(0, len(body) - tail)
    cut = body.find("\n", at)
    if 0 <= cut < len(body) - 1:
        return buf_at + cut + 1
    space = body.find(" ", at)
    return buf_at + (space + 1 if 0 <= space < len(body) - 1 else at)


def chunk_text(text: str, tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list:
    """``[{n, start, text}]`` — ~`tokens`-token chunks on paragraph boundaries with `overlap`
    tokens of tail carried into the next chunk. Deterministic: same text, same chunks."""
    limit, tail = tokens * 4, overlap * 4
    pieces = [p for start, block in _blocks(text) for p in _split_long(start, block, limit)]
    chunks, buf, buf_at = [], "", 0
    for start, piece in pieces:
        if buf and start + len(piece) - buf_at > limit:
            chunks.append({"n": len(chunks) + 1, "start": buf_at, "text": buf.rstrip()})
            buf_at = _carry_from(buf_at, buf, tail)
            buf = text[buf_at:start]
        if not buf.strip():
            buf_at = start
        buf = text[buf_at:start + len(piece)]      # a slice of `text`, so offsets stay honest
    if buf.strip():
        chunks.append({"n": len(chunks) + 1, "start": buf_at, "text": buf.rstrip()})
    return chunks


def _chunkable(entry: dict) -> bool:
    ext = os.path.splitext(str(entry.get("path") or ""))[1].lower()
    return bool(ext) and ext not in (".png", ".jpg", ".jpeg", ".ics")


def _source_title(source: str, raw_rel: str) -> str:
    fm = parse_page(_read(f"{source}.md"))[0] if source else {}
    return str(fm.get("name") or os.path.splitext(os.path.basename(raw_rel))[0])


def _cache_rel(sha: str) -> str:
    return f"{RAW_INDEX_DIR}/{sha}.json"


def raw_chunks(raw_rel: str) -> dict:
    """The chunk record of one raw file: ``{sha, raw, source, title, chunks}``.

    Cached in `raw/.index/<sha>.json`; a cache hit reads no raw bytes. `raw_rel` may be the
    raw path or the source page path; a file that is not in the manifest raises KeyError.
    """
    import sources

    entry = next((e for e in sources.manifest_entries()
                  if raw_rel in (e.get("path"), e.get("page"))), None)
    if entry is None:
        raise KeyError(f"no such source: {raw_rel}")
    return _raw_chunks_for(entry)


def _raw_chunks_for(entry: dict) -> dict:
    """:func:`raw_chunks` for a manifest entry already in hand (a rebuild parses the manifest once)."""
    import sources

    sha, raw_rel, source = entry["sha256"], entry["path"], entry.get("page", "")
    cached = _read(_cache_rel(sha))
    if cached:
        try:
            record = json.loads(cached)
            if record.get("sha") == sha and record.get("raw") == raw_rel:
                return record
        except ValueError:
            pass
    text = sources.raw_text(raw_rel, _read_bytes(raw_rel)) if _chunkable(entry) else ""
    record = {"sha": sha, "raw": raw_rel, "source": source,
              "title": _source_title(source, raw_rel),
              "chunks": chunk_text(text, CHUNK_TOKENS, CHUNK_OVERLAP)}
    _write(_cache_rel(sha), json.dumps(record, ensure_ascii=False))
    return record


def drop_raw_chunks(sha: str) -> None:
    """Remove a forgotten source's cache file (nothing to do when it was never built)."""
    full = _path(_cache_rel(sha))
    if os.path.exists(full):
        os.remove(full)


def _record(sha: str) -> dict | None:
    """A source's chunk record from its `raw/.index/<sha>.json`, or None when there is none.
    The last few are kept parsed: a hit reads its chunk text from here."""
    key = (WIKI_PATH, sha)
    if key in _records:
        return _records[key]
    try:
        record = json.loads(_read(_cache_rel(sha)) or "null")
    except ValueError:
        record = None
    if record is not None and record.get("sha") != sha:
        record = None
    if record is not None:
        _records[key] = record
        while len(_records) > 16:
            _records.pop(next(iter(_records)))
    return record


_records: dict = {}


def index_source(record: dict) -> None:
    """Write hook: index one filed source's chunks (`record` is :func:`raw_chunks`' result)."""
    if (idx := _hook_db()) is not None:
        _put_source(idx, record)


def _put_source(idx: index_db.Index, record: dict) -> None:
    """Write one source's chunk rows (or drop it when it has no chunks)."""
    seen, titled = _proper_counts(c["text"] for c in record["chunks"])
    chunks = [(c["n"], c["start"], _tokenise(c["text"])) for c in record["chunks"]]
    if chunks:
        idx.put_source(record["sha"], record["raw"], record["source"], record["title"], chunks, seen, titled)
    else:
        idx.drop_source(record["sha"], seen, titled)


def unindex_source(sha: str) -> None:
    """Write hook: a forgotten source's chunks leave the index. Call it before
    :func:`drop_raw_chunks` — its proper-noun counts come from the chunk cache."""
    if (idx := _hook_db()) is not None:
        _drop_source(idx, sha)


def _drop_source(idx: index_db.Index, sha: str) -> None:
    """Remove one source's chunk rows, keeping the proper-noun counts right."""
    record = _record(sha)
    if record is not None:
        idx.drop_source(sha, *_proper_counts(c["text"] for c in record["chunks"]))
    elif idx.drop_source(sha, None, None):           # no cache left to subtract: recount the rest
        idx.replace_proper(*_proper_counts(c["text"] for other in idx.source_shas()
                                           for c in (_record(other) or {"chunks": []})["chunks"]))


def _sync_sources(idx: index_db.Index) -> int:
    """Bring the chunk index in line with the manifest; returns the sources changed. Skipped when the
    manifest bytes hash to what the last sync saw."""
    import sources

    digest = _sha(_read(sources.MANIFEST))
    if idx.get_meta("manifest") == digest:
        return 0
    entries = {e["sha256"]: e for e in sources.manifest_entries() if e.get("sha256") and _chunkable(e)}
    have = idx.source_shas()
    changed, skipped = 0, False
    for sha, raw in have.items():
        if sha not in entries or entries[sha]["path"] != raw:
            _drop_source(idx, sha)
            changed += 1
    for sha, entry in entries.items():
        if have.get(sha) == entry["path"]:
            continue
        try:
            record = _raw_chunks_for(entry)
        except Exception as exc:    # noqa: BLE001 — one unreadable source must not take the index down
            log.exception(f"search index: skipped {entry.get('path')} (cannot be chunked)")
            trace.error("index", "sync_sources", exc, fatal_for="none")
            skipped = True
            continue
        if record["chunks"]:
            _put_source(idx, record)
            changed += 1
    if not skipped:                 # a skipped source keeps its chance: the next sync retries it
        idx.set_meta("manifest", digest)
    return changed


def _chunk_index() -> index_db.Index | None:
    """The index, with its sources in line with the manifest: one stat per call, a re-read and a
    sync only when the manifest file changed (a batch import in another process, a hand edit)."""
    import sources

    if not exists():
        return None
    idx = _db()
    try:
        st = os.stat(_path(sources.MANIFEST))
        mark = (st.st_mtime_ns, st.st_size)
    except OSError:
        mark = None
    if getattr(idx, "manifest_mark", None) != mark:
        _sync_sources(idx)
        idx.manifest_mark = mark
    return idx


def _proper_noun(text: str, m) -> bool:
    """Title-case and mid-sentence ("of Covenant", not "Today you") — a name, not a sentence start."""
    word = m.group()
    if not (word[:1].isupper() and word[1:].islower()):
        return False
    prev = text[:m.start()].rstrip()[-1:]
    return prev.isalnum() or prev in ",;:"


def _proper_counts(texts) -> tuple:
    """``(seen, titled)``: stem -> occurrences, and how many of them are proper nouns, over `texts`.
    The index sums them per source; a stem's ratio is ``titled / seen`` over every chunk, and every
    stem the sources contain has a row, so the table doubles as the vocabulary (:func:`proper_phrases`)."""
    seen, titled = Counter(), Counter()
    for text in texts:
        for m in _WORD.finditer(text):
            stem = _stem(m.group().lower())
            seen[stem] += 1
            if _proper_noun(text, m):
                titled[stem] += 1
    return dict(seen), dict(titled)


def _proper_ratios(terms) -> dict:
    """stem -> share of its occurrences that are proper nouns, for the `terms` the sources hold."""
    idx = _chunk_index()
    return idx.proper_ratios(terms) if idx is not None else {}


def _term_weights(terms: set) -> dict:
    """1 + PROPER_BOOST × (share of the word's occurrences written Title-case)."""
    proper = _proper_ratios(terms)
    return {t: 1.0 + PROPER_BOOST * proper.get(t, 0.0) for t in terms}


def content_terms(query: str) -> set:
    """The query's tokens minus stopwords — what a match is scored and normalised on."""
    return {t for t in _tokenise(query) if t not in STOPWORDS}


def proper_terms(terms: set) -> set:
    """The query terms the sources write Title-case at least `PROPER_RATIO_MIN` of the time."""
    proper = _proper_ratios(terms)
    return {t for t in terms if proper.get(t, 0.0) >= PROPER_RATIO_MIN}


#: words that make a question but name no topic ("summarise this", "what does it say"): a
#: question with nothing else in it gets a page's head, not a chunk (docs/42-recall-precision-2.md)
GENERIC_WORDS = frozenset("""summarise summarize summary sum recap explain describe overview gist
tl dr tell page document source say said says talk about mean thing things content contents""".split())
#: title words that name no source on their own ("the guide", "this book")
GENERIC_TITLE_WORDS = frozenset("""guide guides book books note notes walkthrough faq manual
edition volume part game year new complete official""".split())


def topic_terms(query: str) -> set:
    """Content terms minus GENERIC_WORDS: what the question is actually about, or nothing."""
    return content_terms(query) - GENERIC_WORDS


def _title_terms() -> dict:
    """source page -> the content tokens of its title, over the indexed chunks."""
    idx = _chunk_index()
    if idx is None:
        return {}
    return idx.cached("title_terms", lambda: {source: content_terms(title)
                                              for source, _, title, *_ in idx.chunk_meta().values()})


def split_title_terms(terms: set) -> tuple:
    """``(topic, named)`` for the query's content terms: the terms left after dropping every
    filed source's title tokens (all of them when nothing would be left), and the sources the
    question names — two of a title's tokens, or one that no other title has and is not a
    GENERIC_TITLE_WORDS one (docs/42-recall-precision-2.md §1)."""
    titles = _title_terms()
    title_tokens = set().union(*titles.values()) if titles else set()
    topic = terms - title_tokens or terms
    named = set()
    for source, tokens in titles.items():
        matched = terms & tokens
        unique = {t for t in matched if t not in GENERIC_TITLE_WORDS
                  and not any(t in other for s, other in titles.items() if s != source)}
        if len(matched) >= 2 or unique:
            named.add(source)
    return topic, named


def proper_phrases(query: str) -> list:
    """The proper-noun bigrams in `query` as ``[(a, b)]`` — consecutive tokens where one is a
    proper term and the other is a proper term, a word the sources never use ("Lovelace"),
    or both are written Title-case ("Ada Lovelace"). A chunk must contain the pair whole for
    the proper term to count (docs/42-recall-precision-2.md §5)."""
    words = re.findall(r"[A-Za-z0-9]+", str(query))
    vocab = _proper_ratios({_stem(w.lower()) for w in words})
    pairs = []
    for first, second in zip(words, words[1:]):
        a, b = _stem(first.lower()), _stem(second.lower())
        if a in STOPWORDS or b in STOPWORDS:
            continue
        proper = {t for t in (a, b) if vocab.get(t, 0.0) >= PROPER_RATIO_MIN}
        if not proper:
            continue
        titled = first[:1].isupper() and second[:1].isupper()
        unseen = a not in vocab or b not in vocab
        if len(proper) == 2 or unseen or titled:
            pairs.append((a, b))
    return pairs


def _chunk_subset(meta: dict, named: set, filters: dict | None):
    """The chunk ids scored: the named sources' (all when none is named or they hold none),
    narrowed by `filters` ``{source, kind}``; None means every chunk."""
    import sources

    subset = {i for i, m in meta.items() if m[0] in named} or None if named else None
    if not filters:
        return subset
    keep = set(meta) if subset is None else subset
    want = filters.get("source")
    if want:
        want = {want} if isinstance(want, str) else set(want)
        keep = {i for i in keep if meta[i][0] in want or meta[i][1] in want}
    kind = filters.get("kind")
    if kind:
        kinds = {kind} if isinstance(kind, str) else set(kind)
        shas = {e.get("sha256") for e in sources.manifest_entries() if e.get("kind") in kinds}
        keep = {i for i in keep if meta[i][5] in shas}
    return keep


def _phrases_at(idx: index_db.Index, doc: int, pairs: list) -> list:
    """The `pairs` that occur adjacent in chunk `doc`, from the stored token positions."""
    found = []
    for a, b in pairs:
        after = {p - 1 for p in idx.positions(index_db.CHUNK, b, doc)}
        if after and after.intersection(idx.positions(index_db.CHUNK, a, doc)):
            found.append((a, b))
    return found


def search_chunks(query: str, k: int = 3, filters: dict | None = None) -> list:
    """Top-`k` raw chunks for `query`: ``[{source, raw, title, n, total, text, score, matched, phrases}]``.

    The query's title words never score a chunk (:func:`split_title_terms`): they only pick the
    sources searched. `score` is BM25 divided by the number of topic words, so a threshold means
    the same for a three-word and a twelve-word question (see :func:`recall_min_score`); `matched`
    lists the distinct topic words found in the chunk (the PRD 34 precision rule), `phrases`
    the query's proper bigrams the chunk holds whole, `named` whether the question named the
    chunk's source by title. `filters` (``{source, kind}``) narrows the chunks scored, with the
    narrowed set's BM25 statistics, as naming a source does.
    """
    idx = _chunk_index()
    terms = content_terms(query)
    if idx is None or not terms:
        return []
    meta = idx.chunk_meta()
    if not meta:
        return []
    terms, named = split_title_terms(terms)
    terms = terms - GENERIC_WORDS or terms                  # "what does it say about X": X scores
    subset = _chunk_subset(meta, named, filters)
    pairs = proper_phrases(query)
    scores = idx.bm25(index_db.CHUNK, terms, _term_weights(terms), subset)
    ranked = sorted(((round(score / len(terms), 3), doc, matched) for doc, (score, matched) in scores.items()
                     if score > 0), key=lambda r: (-r[0], meta[r[1]][1], meta[r[1]][3]))[:k]
    hits = []
    for score, doc, matched in ranked:
        source, raw, title, n, total, sha = meta[doc]
        record = _record(sha) or _raw_chunks_for({"sha256": sha, "path": raw, "page": source})
        hits.append({"source": source, "raw": raw, "title": title, "n": n, "total": total,
                     "text": record["chunks"][n - 1]["text"], "score": score, "matched": sorted(matched),
                     "phrases": _phrases_at(idx, doc, pairs), "named": source in named})
    return hits


def proper_match(hit: dict, proper: set, pairs: list) -> set:
    """The hit's matched proper terms that may carry it alone: a term inside a proper bigram the
    chunk does not hold whole is demoted to a common word (docs/42-recall-precision-2.md §5)."""
    partial = {t for a, b in pairs if (a, b) not in hit.get("phrases", []) for t in (a, b)}
    return (proper & set(hit["matched"])) - partial


def read_source(path: str, part) -> str:
    """Chunk `part` of a source with a neighbour either side, ≤ `MAX_SOURCE_READ` chars.

    `path` is the source page (`sources/guide`) or the raw path; a bad path or part number
    comes back as a plain error string the model can read.
    """
    rel = normalise_path(path, user=True)
    try:
        record = raw_chunks(wikilink(rel) if rel else str(path or ""))
    except KeyError:
        return f"no such source: {path}"
    chunks, total = record["chunks"], len(record["chunks"])
    try:
        n = int(part)
    except (TypeError, ValueError):
        n = 0
    if not 1 <= n <= total:
        return f"no part {part} in {path} (parts 1-{total})"
    middle = chunks[n - 1]["text"][:MAX_SOURCE_READ]
    room = (MAX_SOURCE_READ - len(middle)) // 2
    before = chunks[n - 2]["text"][-room:] if n > 1 and room else ""
    after = chunks[n]["text"][:room] if n < total and room else ""
    text = "\n".join(p for p in (before, middle, after) if p)[:MAX_SOURCE_READ]
    return f'{record["title"]} · part {n}/{total}\n{text}'


def recall_min_score() -> float:
    """`recall_min_score` from settings.json (default `RECALL_MIN_SCORE`)."""

    value = settings.load_settings().get("recall_min_score")
    return float(value) if isinstance(value, (int, float)) and value >= 0 else RECALL_MIN_SCORE


def read_page(path: str) -> str:
    """A page's text, capped at 4 kB with a marker. Unknown path → "no such page"."""
    rel = normalise_path(path)
    if rel is None or not page_exists(rel):
        return f"no such page: {path}"
    text = _read(rel)
    if len(text.encode()) > MAX_READ_BYTES:
        return text.encode()[:MAX_READ_BYTES].decode(errors="ignore") + TRUNCATED
    return text


def page_text(path: str) -> str | None:
    """The whole page for the editor (no 4 kB cap); None for an unknown path."""
    rel = normalise_path(path, user=True)
    if rel is None or not page_exists(rel):
        return None
    return _read(rel)



# --- focus context (docs/38-focus-context.md) ----------------------------------------
# The open page rides on the turn as its head plus the one `##` section (or, for a source,
# the one raw chunk) that best matches the question — BM25 with the same tokeniser as search.

_H2 = re.compile(r"^## .*$", re.M)


def page_title(path: str) -> str:
    """The `name:` of a page (its file stem when it has none); "" for an unknown path."""
    text = page_text(path)
    if text is None:
        return ""
    fm, _ = parse_page(text)
    return str(fm.get("name") or os.path.basename(normalise_path(path, user=True) or path))


def split_sections(body: str) -> tuple:
    """``(head, [section, ...])``: the text before the first `## ` heading and each `##`
    section with its heading line (the page's frontmatter is not part of `body`)."""
    starts = [m.start() for m in _H2.finditer(body)]
    if not starts:
        return body.strip(), []
    head = body[:starts[0]].strip()
    ends = starts[1:] + [len(body)]
    return head, [body[a:b].strip() for a, b in zip(starts, ends)]


def _best_match(texts: list, question: str) -> int | None:
    """Index of the text that scores best for the question's content words (BM25, earliest on
    a tie); None when there is nothing to score or no word occurs in any of them."""
    terms = content_terms(question)
    if not texts or not terms:
        return None
    scores = _bm25([_tokenise(t) for t in texts], terms)
    best = max(range(len(texts)), key=lambda i: (scores[i], -i))
    return best if scores[best] > 0 else None


def best_section(sections: list, question: str) -> str | None:
    """The section that scores best for `question`; None with fewer than two sections or
    when no content word of the question occurs in any of them."""
    at = _best_match(sections, question) if len(sections) > 1 else None
    return None if at is None else sections[at]


def best_chunk(path: str, question: str) -> dict | None:
    """The raw chunk of source page `path` that scores best for `question` as
    ``{title, n, total, text}``; None for a page without raw text or when nothing matches."""
    try:
        record = raw_chunks(path)
    except KeyError:
        return None
    chunks = record["chunks"]
    at = _best_match([c["text"] for c in chunks], question)
    if at is None:
        return None
    return {"title": record["title"], "n": chunks[at]["n"], "total": len(chunks), "text": chunks[at]["text"]}


_LINK = re.compile(r"\[\[([^\]|#]+)")


def links_in(text: str) -> list:
    """The wikilink targets in `text`, in order, de-duplicated."""
    seen = []
    for raw in _LINK.findall(text or ""):
        target = raw.strip()
        if target and target not in seen:
            seen.append(target)
    return seen


#: the recall tools tools/wiki.py registers — the prompt may only advertise them when they
#: are actually offered to the model this run (see :func:`tools_registered`).
RECALL_TOOLS = ("search_wiki", "read_page", "read_source")


def tools_registered() -> bool:
    """True when `search_wiki`/`read_page` are on the tool list the model is being given.

    `llm.TOOLS` is the live list: the eval harness empties it to run tool-free, and a prompt
    that tells a tool-less model to "use search_wiki" only teaches it to fake a tool call
    (docs/19-wiki-layer.md, Results).
    """
    try:
        import llm

        names = {(t.get("function") or {}).get("name") for t in llm.TOOLS if isinstance(t, dict)}
    except Exception:  # noqa: BLE001 — no llm (tests, scripts) means no tools
        return False
    return any(name in names for name in RECALL_TOOLS)


# --- briefing (docs/46-briefing.md) -------------------------------------------------
# "what's new?": what the vault, the sources and memory gained since a cutoff, from git
# subjects (every writer makes one commit per write), the manifest and the memory ops log.

_NOISE_COMMITS = ("wiki: lint", "wiki: initial vault", "wiki: vault layout")
_USER_COMMIT = re.compile(r"^wiki: (?:edit|create|save card|rename|delete) ")
_CHAT_COMMIT = re.compile(r"^wiki: \d+ ops? \(session [^)]*\)$")
BRIEFING_CAP = 8                 # items per card section, then "+n more"
_CHANGE = {"A": "new", "M": "edited", "D": "deleted"}
_MEMORY_OPS = {"ADD": "added", "UPDATE": "updated", "DELETE": "removed"}


def _news_commit(subject: str) -> bool:
    """False for the layer's own housekeeping (lint, vault creation) and the user's own writes
    (`_user_commit`: editor save / create / rename / delete, PRD 43 save card): never news."""
    return subject not in _NOISE_COMMITS and not _USER_COMMIT.match(subject)


def _news_path(path: str) -> bool:
    return path.endswith(".md") and path not in RESERVED and not path.startswith((f"{RAW_DIR}/", "sources/"))


def _page_changes(ts: str) -> tuple:
    """``(pages, chat_pages)`` from `git log --since=ts`; nothing when git or the repo is missing."""
    result = _git("log", f"--since={ts}", "--name-status", "--format=%x1e%s", quiet=True)
    if result is None or result.returncode != 0:
        return [], []
    seen: dict = {}                                      # path -> {"status": [..], "chat": bool}, newest first
    for record in result.stdout.split("\x1e")[1:]:
        lines = [line for line in record.splitlines() if line.strip()]
        subject = lines[0] if lines else ""
        if not _news_commit(subject):
            continue
        for line in lines[1:]:
            fields = line.split("\t")
            status = fields[0][:1]
            touched = [(fields[1], "D"), (fields[2], "A")] if status == "R" and len(fields) == 3 else \
                [(fields[-1], status)]
            for path, st in touched:
                if not _news_path(path):
                    continue
                entry = seen.setdefault(path, {"status": [], "chat": True})
                entry["status"].append(st)
                entry["chat"] = entry["chat"] and bool(_CHAT_COMMIT.match(subject))
    pages, chat = [], []
    for path, entry in seen.items():
        rel = wikilink(path)
        added, here = "A" in entry["status"], os.path.isfile(_path(path))
        if not here and added:
            continue                                     # born and gone inside the window
        change = "new" if added and here else "deleted" if not here else "edited"
        if entry["chat"] and here:
            chat.append(rel)
        else:
            pages.append({"path": rel, "title": page_title(rel) or os.path.basename(rel), "change": change})
    order = list(_CHANGE.values())
    pages.sort(key=lambda p: (order.index(p["change"]), p["path"]))
    return pages, sorted(chat)


def _source_changes(ts: str) -> list:
    import sources
    out = []
    for entry in sources.manifest_entries():
        at = str(entry.get("ingested_at") or "")
        if "T" not in at:                                # an old line: the date lives on the page
            at = sources._source_meta(entry)["when"].replace(" ", "T")
        if len(at) < 19 or at <= ts:
            continue
        out.append({"page": entry.get("page", ""), "path": entry.get("path", ""),
                    "kind": str(entry.get("kind") or ""), "at": at})
    return sorted(out, key=lambda s: s["at"])


def _memory_changes(ts: str) -> dict:
    counts = {"added": 0, "updated": 0, "removed": 0, "examples": []}
    try:
        with open(memory.OPS_LOG_PATH) as f:
            lines = f.readlines()
    except OSError:
        return counts
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = _MEMORY_OPS.get(str(rec.get("op") or ""))
        if not kind or str(rec.get("ts") or "") <= ts or rec.get("reason") in memory.UI_REASONS:
            continue
        counts[kind] += 1
        if len(counts["examples"]) < 2:
            counts["examples"].append({"op": rec["op"], "path": rec.get("path", ""), "value": rec.get("value")})
    return counts


def changes_since(ts: str) -> dict:
    """The briefing digest: pages, chat pages, sources and memory changes after `ts` (PRD 46)."""
    with vault_lock.LOCK:
        pages, chat = _page_changes(ts) if exists() else ([], [])
        return {"since": ts, "pages": pages, "chat_pages": chat, "sources": _source_changes(ts),
                "memory": _memory_changes(ts)}


def memory_total(digest: dict) -> int:
    """How many memory ops the digest counts (added + updated + removed)."""
    mem = digest["memory"]
    return mem["added"] + mem["updated"] + mem["removed"]


def digest_empty(digest: dict) -> bool:
    return not (digest["pages"] or digest["chat_pages"] or digest["sources"] or memory_total(digest))


def _day(d) -> str:
    return f"{d.day} {d.strftime('%b')}"


def briefing_when(ts: str) -> str:
    """"14:05" today, "yesterday at 14:05", else "12 Oct"."""
    from datetime import date, timedelta
    try:
        day = date.fromisoformat(ts[:10])
    except ValueError:
        return ts
    today = memory._today()
    if day == today:
        return ts[11:16]
    if day == today - timedelta(days=1):
        return f"yesterday at {ts[11:16]}"
    return _day(day)


def memory_value(value) -> str:
    """An op value as one short line: a dict's values joined by " · "."""
    if isinstance(value, dict):
        value = " · ".join(str(v) for v in value.values() if v not in (None, ""))
    return " ".join(str(value if value is not None else "").split())[:80]


def capped(items: list) -> tuple:
    """``(the first BRIEFING_CAP items, how many more)``."""
    return items[:BRIEFING_CAP], max(0, len(items) - BRIEFING_CAP)


def _capped(items: list) -> list:
    head, extra = capped(items)
    return head + ([f"+{extra} more"] if extra else [])


def briefing_card(digest: dict) -> str:
    """The briefing as a Markdown card: `## Pages` / `## Sources` / `## Memory`, empty ones left out."""
    out = [f"# Briefing {_day(memory._today())}"]
    pages = _capped([f"{p['change']}: " + (p["title"] if p["change"] == "deleted" else f"[[{p['path']}|{p['title']}]]")
                     for p in digest["pages"]])
    chat = digest["chat_pages"]
    if chat:
        more = f" (+{len(chat) - 2})" if len(chat) > 2 else ""
        pages.append("notes from our chats: " + ", ".join(f"[[{c}]]" for c in chat[:2]) + more)
    sections = [("Pages", pages),
                ("Sources", _capped([f"[[{s['page']}]] · {s['kind'] or 'file'}" for s in digest["sources"]]))]
    mem = digest["memory"]
    if memory_total(digest):
        lines = [f"{mem['added']} added, {mem['updated']} updated, {mem['removed']} removed"]
        lines += [f"{_MEMORY_OPS[e['op']]} {e['path']}: {memory_value(e['value'])}".rstrip(": ")
                  for e in mem["examples"]]
        sections.append(("Memory", lines))
    for heading, items in sections:
        if items:
            out += ["", f"## {heading}"] + [f"- {item}" for item in items]
    return "\n".join(out)


# --- listing / lint ----------------------------------------------------------------


def pages() -> list:
    """``[{path, title, type, updated}]`` for the Wiki tab, sorted by path.

    Paths are suffix-less (`people/anna`), the same form wikilinks and the tools use; the
    normalising on the way back in (:func:`normalise_path`) still accepts `.md`.
    """
    listing = []
    for rel in page_paths():
        text = _read(rel)
        fm, _ = parse_page(text)
        listing.append({
            "path": wikilink(rel),
            "title": str(fm.get("name") or os.path.basename(rel)[:-3]),
            "type": str(fm.get("type") or ""),
            "updated": str(fm.get("updated") or ""),
            "tags": tags(text),
        })
    return listing


REINDEX_HINT = " — run `python -m wiki reindex`"


def lint() -> list:
    """Findings, worst first: per page broken links, missing from the search index, bad frontmatter,
    size, a `source: raw/…` whose file is gone; then index rows whose page is gone (both mean
    `python -m wiki reindex`), raw files nothing cites, then duplicate names. `index.md` is a capped
    human view (200 lines), so it is not what a page is checked against (docs/53-scalable-index.md).

    A finding names its page the way the rest of the layer does, `people/anna`, or its raw file.
    """
    import sources

    if not exists():
        return []
    findings = []
    cited, labels = set(), []
    paths = page_paths()
    known = {wikilink(rel) for rel in paths} | {wikilink(r) for r in RESERVED}
    indexed = set(_db().page_paths())
    for rel in paths:
        text = _read(rel)
        fm, _ = parse_page(text)
        name = wikilink(rel)
        for target in links_in(text):
            if target not in known:
                findings.append(f"{name}: broken link [[{target}]]")
        if rel not in indexed:
            findings.append(f"{name}: missing from the search index{REINDEX_HINT}")
        missing = [key for key in FIELDS if key not in fm]
        if missing:
            findings.append(f"{name}: frontmatter missing {', '.join(missing)}")
        if len(text.encode()) > MAX_PAGE_BYTES:
            findings.append(f"{name}: larger than {MAX_PAGE_BYTES // 1024} kB")
        source = str(fm.get("source") or "")
        if source.startswith(RAW_DIR + "/") and not os.path.isfile(_path(source)):
            findings.append(f"{name}: source {source} no longer exists")
        cited |= {str(fm.get(key) or "") for key in ("source", "raw")}
        labels.append((name, " ".join(str(fm.get("name") or "").split())))
    findings += [f"{wikilink(rel)}: in the search index but not on disk{REINDEX_HINT}"
                 for rel in sorted(indexed - set(paths))]
    cited |= {e.get("path") for e in sources.manifest_entries()}
    cited |= {rel for e in sources.manifest_entries() for rel in e.get("pages") or []}   # PRD 47 PNGs
    for root, dirs, files in os.walk(_path(RAW_DIR)):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fname in sorted(files):
            rel = os.path.relpath(os.path.join(root, fname), WIKI_PATH).replace(os.sep, "/")
            if not fname.startswith(".") and rel != sources.MANIFEST and rel not in cited:
                findings.append(f"{rel}: not in the manifest and cited by no page")
    first = {}
    for name, label in sorted(labels):
        if not label or name.startswith("meetings/"):   # recurring invites share a name
            continue
        group = (name.startswith("sources/"), label.casefold())
        if group in first:
            findings.append(f'{name}: duplicate name "{label}" (also {first[group]})')
        first.setdefault(group, name)
    return findings


def lint_and_log() -> list:
    """`lint()` from the reflection cadence: only a run with findings earns a `log.md` line."""
    findings = lint()
    if findings:
        ensure_vault()
        summary = f"lint: {len(findings)} finding{'s' if len(findings) != 1 else ''} ({findings[0]})"
        _log([summary])
        _commit("wiki: lint")
    return findings


# --- ingest ------------------------------------------------------------------------


INGEST_SYSTEM = """You keep a markdown wiki of the USER's working life: people, projects,
meetings and decisions. You are given the wiki's schema, its index, and a transcript of the
conversation that just happened.

Reply with a JSON array of page operations and nothing else. Each operation is one of:
{"op":"new_page","path":"people/anna","type":"person","name":"Anna","summary":"one line"}
{"op":"append_note","path":"projects/atlas","text":"one sentence"}
{"op":"set_field","path":"projects/atlas","key":"status","value":"paused"}
{"op":"add_decision","title":"deploy on Fridays","text":"one or two sentences"}
{"op":"link","path":"people/anna","target":"projects/atlas"}

Paths are `people/<slug>`, `projects/<slug>`, `meetings/YYYY-MM-DD` or `decisions`.
Frontmatter keys are exactly: type, name, tags, source, updated, seen.

Rules:
- Only file what the USER actually said. Never the assistant's words, never a guess.
- %s
- File people, projects, meetings and decisions — things the user would expect you to remember
  in a month.
- Never file everyday trivia: meals, weather, moods, small talk, chores, commutes.
- Append to an existing page from the index instead of creating a near-duplicate.
- A note is one sentence, at most 600 characters.
- """ + DOCUMENT_RULE + """
- Nothing worth filing? Reply with exactly [].
Today is %s."""


def user_page_reason(op: dict, names) -> str | None:
    """Why `op` would file the user as a person — or None when it is fine.

    Only `new_page people/<slug>` and a `link` whose target is a people page can do that; the
    slug and the op's `name` are both checked, because the model spells either one.
    """
    kind = str(op.get("op") or "").strip().lower()
    if kind == "new_page":
        checked = [op.get("path"), op.get("name")]
    elif kind == "link":
        checked = [op.get("target")]
    else:
        return None
    rel = normalise_path(op.get("path") if kind == "new_page" else op.get("target"))
    if rel is None or not rel.startswith("people/"):
        return None
    checked.append(os.path.basename(rel)[:-3].replace("-", " "))
    for value in checked:
        text = " ".join(str(value or "").split())
        if "/" in text:
            continue
        if text and memory.is_user(text, names):
            return f"{text} is the user, not another person"
    return None


def reject_user_pages(ops, names) -> tuple:
    """``(kept, rejected)`` — the user never gets a people page of their own."""
    kept, rejected = [], []
    for raw in ops if isinstance(ops, list) else []:
        reason = user_page_reason(raw if isinstance(raw, dict) else {}, names)
        (rejected if reason else kept).append({"op": raw, "reason": reason} if reason else raw)
    return kept, rejected


def ingest(turns: list) -> list:
    """Ask the model for page ops for this session and apply them; returns the applied ops.

    Never raises: a wiki failure must not cost the user their profile save (see memory.save).
    """
    transcript = memory._transcript(turns)
    if not transcript.strip():
        return []
    names = memory.user_names()
    system = INGEST_SYSTEM % (memory.user_note(names), memory._today_iso())
    related = related_context(transcript)
    user = (f"Schema:\n{_read(SCHEMA_FILE) or SCHEMA_TEXT}\n\n"
            f"Index:\n{index_text(WIKI_INDEX_TOKENS) or '(empty wiki)'}\n\n"
            + (f"{related}\n\n" if related else "")
            + f"Transcript:\n{transcript}")
    try:
        raw = core_llm.complete(system, user, 800)
    except Exception as exc:  # noqa: BLE001
        log.info(f"ingest failed: {exc}")
        return []
    ops = memory._extract_json(raw)
    if not isinstance(ops, list):
        log.info("could not parse page ops; nothing filed")
        return []
    ops, about_user = reject_user_pages(ops, names)
    applied, rejected = apply_page_ops(ops, f"session {memory._today_iso()}")
    log_rejected(about_user + rejected, log)
    if applied:
        log.info(f"{len(applied)} page op{'s' if len(applied) != 1 else ''} applied")
    return applied


# --- bridge helpers (used by memory.archive / memory.pull) --------------------------


def page_for(section: str, entry: dict) -> tuple:
    """Where a profile entry of `section` archives to: ``(path, type, name)``."""
    if section == "people":
        name = str(entry.get("name") or "").strip()
        return f"people/{slugify(name)}.md", "person", name
    if section == "projects":
        name = str(entry.get("name") or "").strip()
        return f"projects/{slugify(name)}.md", "project", name
    return RECURRING_FILE, "recurring", "Recurring"


def history_line(section: str, entry: dict) -> str:
    """The `## History` bullet an archived entry leaves behind."""
    fields = {"people": ("name", "rel", "note"),
              "projects": ("name", "status", "note"),
              "recurring": ("what", "when", "note")}.get(section, ("name", "note"))
    parts = [str(entry[f]).strip() for f in fields if str(entry.get(f, "")).strip()]
    seen = str(entry.get(memory.SEEN_FIELD) or "").strip()
    text = "; ".join(parts) or section
    return f"{text} (last seen {seen})" if seen else text


def archive_entry(section: str, entry: dict) -> tuple:
    """Create or extend the wiki page for `entry`; ``(page, None)`` or ``(None, reason)``."""
    ensure_vault()
    rel, page_type, name = page_for(section, entry)
    source = f"archived from {section} {memory._today_iso()}"
    ops = []
    if not page_exists(rel):
        ops.append({"op": "new_page", "path": rel, "type": page_type,
                    "name": name or rel[:-3], "summary": history_line(section, entry),
                    "source": source})
    ops.append({"op": "append_note", "path": rel, "text": history_line(section, entry),
                "source": source, "section": "## History"})
    applied, rejected = apply_page_ops(ops, source)
    if not applied:
        return None, (rejected[0]["reason"] if rejected else "nothing to archive")
    return wikilink(rel), None


def entry_from_page(path: str, section: str) -> tuple:
    """A live profile entry built from a page's frontmatter; ``(entry, None)`` or ``(None, reason)``."""
    rel = normalise_path(path)
    if rel is None or not page_exists(rel):
        return None, f"no such page {path}"
    fm, body = parse_page(_read(rel))
    name = str(fm.get("name") or os.path.basename(rel)[:-3]).strip()
    summary = _summary_line(body)
    entry = {memory.SEEN_FIELD: memory._today_iso()}
    if section == "recurring":
        entry["what"] = name
        if summary:
            entry["when"] = summary[:120]
    else:
        entry["name"] = name
        status = str(fm.get("status") or "").strip()
        if section == "projects":
            entry["status"] = status or "active"
        if summary:
            entry["note"] = summary[:200]
    return entry, None


def cli_vault(env_file: str) -> str:
    """The vault `python -m wiki` works on: WIKI_PATH from the environment, else from `env_file`
    (the checkout's `.env`, so a dev checkout targets its dev vault), else the default."""
    from dotenv import dotenv_values

    value = os.environ.get("WIKI_PATH") or dotenv_values(env_file).get("WIKI_PATH") or "~/llm-wiki"
    return os.path.expanduser(value)


def main(argv) -> int:
    """`python -m wiki reindex`: rebuild the search index; refused (exit 2) while the app runs,
    because swapping the file under a running app would orphan its open handle."""
    import sys

    import batch_import

    if list(argv) != ["reindex"]:
        print("usage: python -m wiki reindex", file=sys.stderr)
        return 2
    running = batch_import.running_app()
    if running:
        print(f"refusing to reindex: {running}", file=sys.stderr)
        return 2
    print(f"reindexed {WIKI_PATH} -> {reindex()}")
    return 0


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    import wiki as _wiki      # the imported module, not this __main__ copy: the index handle lives there

    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env)         # the app's ports for the running-app guard
    _wiki.WIKI_PATH = _wiki.cli_vault(_env)
    sys.exit(_wiki.main(sys.argv[1:]))
