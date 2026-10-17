"""Bridge helpers used by memory.archive / memory.pull: a profile entry as a vault page and back (PRD 50 step 7, from wiki.py)."""
import os
from memory import store as m_store
from core import trace
from vault import index as v_index
from vault import pages as v_pages
from vault import store
from core import util

log = trace.logger("Wiki")


# --- bridge helpers (used by memory.archive / memory.pull) --------------------------


def page_for(section: str, entry: dict) -> tuple:
    """Where a profile entry of `section` archives to: ``(path, type, name)``."""
    if section == "people":
        name = str(entry.get("name") or "").strip()
        return f"people/{store.slugify(name)}.md", "person", name
    if section == "projects":
        name = str(entry.get("name") or "").strip()
        return f"projects/{store.slugify(name)}.md", "project", name
    return store.RECURRING_FILE, "recurring", "Recurring"


def history_line(section: str, entry: dict) -> str:
    """The `## History` bullet an archived entry leaves behind."""
    fields = {"people": ("name", "rel", "note"),
              "projects": ("name", "status", "note"),
              "recurring": ("what", "when", "note")}.get(section, ("name", "note"))
    parts = [str(entry[f]).strip() for f in fields if str(entry.get(f, "")).strip()]
    seen = str(entry.get(m_store.SEEN_FIELD) or "").strip()
    text = "; ".join(parts) or section
    return f"{text} (last seen {seen})" if seen else text


def archive_entry(section: str, entry: dict) -> tuple:
    """Create or extend the wiki page for `entry`; ``(page, None)`` or ``(None, reason)``."""
    store.ensure_vault()
    rel, page_type, name = page_for(section, entry)
    source = f"archived from {section} {util.today_iso()}"
    ops = []
    if not store.page_exists(rel):
        ops.append({"op": "new_page", "path": rel, "type": page_type,
                    "name": name or rel[:-3], "summary": history_line(section, entry),
                    "source": source})
    ops.append({"op": "append_note", "path": rel, "text": history_line(section, entry),
                "source": source, "section": "## History"})
    applied, rejected = v_pages.apply_page_ops(ops, source)
    if not applied:
        return None, (rejected[0]["reason"] if rejected else "nothing to archive")
    return store.wikilink(rel), None


def entry_from_page(path: str, section: str) -> tuple:
    """A live profile entry built from a page's frontmatter; ``(entry, None)`` or ``(None, reason)``."""
    rel = store.normalise_path(path)
    if rel is None or not store.page_exists(rel):
        return None, f"no such page {path}"
    fm, body = store.parse_page(store._read(rel))
    name = str(fm.get("name") or os.path.basename(rel)[:-3]).strip()
    summary = v_index._summary_line(body)
    entry = {m_store.SEEN_FIELD: util.today_iso()}
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
