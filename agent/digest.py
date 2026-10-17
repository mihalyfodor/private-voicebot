"""The briefing digest: what changed in the vault since a cutoff (docs/46-briefing.md; PRD 50 step 7, from wiki.py)."""
import json
import os
import re
from memory import store as m_store
from core import trace, vault_lock
from vault import store
from vault import focus
from core import util

log = trace.logger("Wiki")


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
    return path.endswith(".md") and path not in store.RESERVED and not path.startswith((f"{store.RAW_DIR}/", "sources/"))


def _housekeeping(subject: str, paths: list) -> bool:
    """The layer's own upkeep: a `_NOISE_COMMITS` subject, or a commit touching only index / log /
    schema (docs/51-run-cards.md counts these as the briefing's skipped commits)."""
    return subject in _NOISE_COMMITS or (bool(paths) and all(p in store.RESERVED for p in paths))


def _page_changes(ts: str) -> tuple:
    """``(pages, chat_pages, skipped)`` from `git log --since=ts`; nothing when git or the repo is
    missing. A page carries the short sha of its newest commit; `skipped` counts housekeeping commits."""
    result = store._git("log", f"--since={ts}", "--name-status", "--format=%x1e%h %s", quiet=True)
    if result is None or result.returncode != 0:
        return [], [], 0
    seen: dict = {}                                      # path -> {"status": [..], "chat": bool}, newest first
    skipped = 0
    for record in result.stdout.split("\x1e")[1:]:
        lines = [line for line in record.splitlines() if line.strip()]
        sha, _, subject = (lines[0] if lines else "").partition(" ")
        if _housekeeping(subject, [line.split("\t")[-1] for line in lines[1:]]):
            skipped += 1
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
                entry = seen.setdefault(path, {"status": [], "chat": True, "commit": sha})
                entry["status"].append(st)
                entry["chat"] = entry["chat"] and bool(_CHAT_COMMIT.match(subject))
    pages, chat = [], []
    for path, entry in seen.items():
        rel = store.wikilink(path)
        added, here = "A" in entry["status"], os.path.isfile(store._path(path))
        if not here and added:
            continue                                     # born and gone inside the window
        change = "new" if added and here else "deleted" if not here else "edited"
        if entry["chat"] and here:
            chat.append(rel)
        else:
            pages.append({"path": rel, "title": focus.page_title(rel) or os.path.basename(rel), "change": change,
                          "commit": entry["commit"]})
    order = list(_CHANGE.values())
    pages.sort(key=lambda p: (order.index(p["change"]), p["path"]))
    return pages, sorted(chat), skipped


def _source_changes(ts: str) -> list:
    from vault import manifest
    out = []
    for entry in manifest.manifest_entries():
        at = str(entry.get("ingested_at") or "")
        if "T" not in at:                                # an old line: the date lives on the page
            at = manifest._source_meta(entry)["when"].replace(" ", "T")
        if len(at) < 19 or at <= ts:
            continue
        out.append({"page": entry.get("page", ""), "path": entry.get("path", ""),
                    "kind": str(entry.get("kind") or ""), "at": at})
    return sorted(out, key=lambda s: s["at"])


def _memory_changes(ts: str) -> dict:
    counts = {"added": 0, "updated": 0, "removed": 0, "examples": []}
    try:
        with open(m_store.OPS_LOG_PATH) as f:
            lines = f.readlines()
    except OSError:
        return counts
    for line in lines:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = _MEMORY_OPS.get(str(rec.get("op") or ""))
        if not kind or str(rec.get("ts") or "") <= ts or rec.get("reason") in m_store.UI_REASONS:
            continue
        counts[kind] += 1
        if len(counts["examples"]) < 2:
            counts["examples"].append({"op": rec["op"], "path": rec.get("path", ""), "value": rec.get("value")})
    return counts


def changes_since(ts: str) -> dict:
    """The briefing digest: pages, chat pages, sources and memory changes after `ts` (PRD 46), plus
    `skipped`, the housekeeping commits left out (PRD 51)."""
    with vault_lock.LOCK:
        pages, chat, skipped = _page_changes(ts) if store.exists() else ([], [], 0)
        return {"since": ts, "pages": pages, "chat_pages": chat, "sources": _source_changes(ts),
                "memory": _memory_changes(ts), "skipped": skipped}


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
    today = util._today()
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
    out = [f"# Briefing {_day(util._today())}"]
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
