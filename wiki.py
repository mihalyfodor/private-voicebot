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
import math
import os
import re
import subprocess
from collections import Counter

import memory

# --- vault -------------------------------------------------------------------------

WIKI_PATH = os.path.expanduser(os.getenv("WIKI_PATH", "~/llm-wiki"))

INDEX_FILE = "index.md"
LOG_FILE = "log.md"
SCHEMA_FILE = "schema.md"
DECISIONS_FILE = "decisions.md"
RECURRING_FILE = "recurring.md"          # where archived `recurring` profile entries land
FOLDERS = ("people", "projects", "meetings", "sources")

#: raw sources (PRD 20) live here: immutable originals, never pages, never indexed
RAW_DIR = "raw"
RAW_SUBDIRS = ("inbox", "screens", "email", "news")

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
        print(f"[Wiki] git {args[0] if args else ''} failed: {exc}")
        return None
    if result.returncode != 0 and not quiet:
        message = (result.stderr or result.stdout or "").strip().splitlines()
        if message:
            print(f"[Wiki] git {args[0] if args else ''}: {message[0]}")
    return result


def _path(*parts) -> str:
    return os.path.join(WIKI_PATH, *parts)


def _write(rel: str, text: str) -> None:
    full = _path(rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(text)


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
    if not os.path.exists(_path(".gitignore")):
        _write(".gitignore", ".obsidian/\n.trash/\n")
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
        if key not in FIELDS:
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


def normalise_path(path) -> str | None:
    """``"people/Anna"`` -> ``"people/anna.md"``; ``None`` when the path is not vault-safe.

    Rejects anything absolute, anything with ``..``, anything deeper than ``folder/page.md``
    and any segment that is not a slug — the model proposes these paths, so they are untrusted.
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
    if len(parts) == 2 and parts[0] not in FOLDERS:
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
    lines = []
    for rel in page_paths():
        fm, body = parse_page(_read(rel))
        described = _one_liner(fm, body)
        lines.append(f"- [[{wikilink(rel)}]]" + (f" — {described}" if described else ""))
    return lines[:INDEX_MAX_LINES]


def rebuild_index() -> str:
    """Regenerate `index.md` from the pages' frontmatter — a pure function of the vault."""
    lines = index_lines()
    text = "# Index\n\n" + ("\n".join(lines) + "\n" if lines else "(no pages yet)\n")
    _write(INDEX_FILE, text)
    return text


def index_text(max_tokens: int = WIKI_INDEX_TOKENS) -> str:
    """The index trimmed to `max_tokens` — what goes into the prompt and into the ingest call.

    Computed from the pages rather than read from `index.md`, so a page the user edited in
    Obsidian since the last commit is already reflected. "" when there is nothing to show.
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
            reserved: bool = False) -> tuple:
    """Validate an op's (untrusted) path: ``(rel, None)`` or ``(None, reason)``.

    `exists=True` demands the page is there, `exists=False` demands it is not; `reserved=True`
    additionally refuses the machinery pages. Every op's path check goes through here, so the
    rejection reasons the model sees are worded the same way everywhere.
    """
    rel = normalise_path(op.get(key))
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
    today = memory._today_iso()
    fm = {"type": page_type, "name": name, "tags": [], "source": source,
          "updated": today, "seen": today}
    body = f"# {name}\n\n## Summary\n{summary}\n\n## Notes\n\n## Related\n"
    _write(rel, _page_text(fm, body))
    return f"filed [[{wikilink(rel)}]] (source: {source})", None


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
    global _applied_batches
    applied, rejected = [], []
    if not isinstance(ops, list) or not ops:
        return applied, rejected
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


# --- search ------------------------------------------------------------------------
# BM25 over page bodies + frontmatter. Personal scale (≤ 200 pages), no embeddings, no new
# dependency, and deterministic — which is what makes it testable.

_K1, _B = 1.5, 0.75
_index_cache: dict = {"sig": None, "docs": []}


def _tokenise(text: str) -> list:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _signature() -> tuple:
    return tuple(sorted((rel, os.path.getmtime(_path(rel))) for rel in page_paths()
                        if os.path.exists(_path(rel))))


def _corpus() -> list:
    """[(path, text, tokens)] — rebuilt only when a page's mtime changed (Obsidian edits count)."""
    sig = _signature()
    if _index_cache["sig"] != sig:
        docs = []
        for rel, _ in sig:
            text = _read(rel)
            docs.append((rel, text, _tokenise(text)))
        _index_cache["sig"], _index_cache["docs"] = sig, docs
    return _index_cache["docs"]


def _snippet(text: str, terms: set, width: int = 180) -> str:
    body = " ".join(parse_page(text)[1].split())
    lower = body.lower()
    at = min((lower.find(t) for t in terms if lower.find(t) >= 0), default=0)
    start = max(0, at - 40)
    return ("…" if start else "") + body[start:start + width] + ("…" if len(body) > start + width else "")


def search(query: str, k: int = 5) -> list:
    """Top-`k` pages for `query` as ``[{path, snippet, score}]`` (BM25, best first).

    `path` is the suffix-less form the model sees everywhere else (`people/anna`), so a hit can
    be handed straight to :func:`read_page`.
    """
    docs = _corpus()
    terms = set(_tokenise(query))
    if not docs or not terms:
        return []
    n = len(docs)
    avgdl = sum(len(t) for _, _, t in docs) / n
    df = Counter(term for _, _, tokens in docs for term in terms if term in tokens)
    scored = []
    for rel, text, tokens in docs:
        counts, dl, score = Counter(tokens), len(tokens) or 1, 0.0
        for term in terms:
            freq = counts.get(term, 0)
            if not freq:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * freq * (_K1 + 1) / (freq + _K1 * (1 - _B + _B * dl / avgdl))
        if score > 0:
            scored.append({"path": wikilink(rel), "snippet": _snippet(text, terms),
                           "score": round(score, 3)})
    return sorted(scored, key=lambda d: (-d["score"], d["path"]))[:k]


def read_page(path: str) -> str:
    """A page's text, capped at 4 kB with a marker. Unknown path → "no such page"."""
    rel = normalise_path(path)
    if rel is None or not page_exists(rel):
        return f"no such page: {path}"
    text = _read(rel)
    if len(text.encode()) > MAX_READ_BYTES:
        return text.encode()[:MAX_READ_BYTES].decode(errors="ignore") + TRUNCATED
    return text


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
RECALL_TOOLS = ("search_wiki", "read_page")


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


# --- listing / lint ----------------------------------------------------------------


def pages() -> list:
    """``[{path, title, type, updated}]`` for the Wiki tab, sorted by path.

    Paths are suffix-less (`people/anna`), the same form wikilinks and the tools use; the
    normalising on the way back in (:func:`normalise_path`) still accepts `.md`.
    """
    listing = []
    for rel in page_paths():
        fm, _ = parse_page(_read(rel))
        listing.append({
            "path": wikilink(rel),
            "title": str(fm.get("name") or os.path.basename(rel)[:-3]),
            "type": str(fm.get("type") or ""),
            "updated": str(fm.get("updated") or ""),
        })
    return listing


def lint() -> list:
    """Findings, worst first: broken links, pages missing from the index, bad frontmatter, size.

    Each finding names its page the way the rest of the layer does, `people/anna`.
    """
    if not exists():
        return []
    findings = []
    paths = page_paths()
    known = {wikilink(rel) for rel in paths} | {wikilink(r) for r in RESERVED}
    indexed = set(links_in(_read(INDEX_FILE)))
    for rel in paths:
        text = _read(rel)
        fm, _ = parse_page(text)
        name = wikilink(rel)
        for target in links_in(text):
            if target not in known:
                findings.append(f"{name}: broken link [[{target}]]")
        if name not in indexed:
            findings.append(f"{name}: missing from index.md")
        missing = [key for key in FIELDS if key not in fm]
        if missing:
            findings.append(f"{name}: frontmatter missing {', '.join(missing)}")
        if len(text.encode()) > MAX_PAGE_BYTES:
            findings.append(f"{name}: larger than {MAX_PAGE_BYTES // 1024} kB")
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


def ingest(turns: list, client, model: str) -> list:
    """Ask the model for page ops for this session and apply them; returns the applied ops.

    Never raises: a wiki failure must not cost the user their profile save (see memory.save).
    """
    transcript = memory._transcript(turns)
    if not transcript.strip():
        return []
    names = memory.user_names()
    system = INGEST_SYSTEM % (memory.user_note(names), memory._today_iso())
    user = (f"Schema:\n{_read(SCHEMA_FILE) or SCHEMA_TEXT}\n\n"
            f"Index:\n{index_text(WIKI_INDEX_TOKENS) or '(empty wiki)'}\n\n"
            f"Transcript:\n{transcript}")
    try:
        raw = memory._complete(client, model, system, user, 800)
    except Exception as exc:  # noqa: BLE001
        print(f"[Wiki] ingest failed: {exc}")
        return []
    ops = memory._extract_json(raw)
    if not isinstance(ops, list):
        print("[Wiki] could not parse page ops; nothing filed")
        return []
    ops, about_user = reject_user_pages(ops, names)
    applied, rejected = apply_page_ops(ops, f"session {memory._today_iso()}")
    for bad in about_user + rejected:
        print(f"[Wiki] rejected {bad['op']}: {bad['reason']}")
    if applied:
        print(f"[Wiki] {len(applied)} page op{'s' if len(applied) != 1 else ''} applied")
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
