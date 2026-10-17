"""Bridge helpers used by memory.archive / memory.pull, and the `python -m wiki reindex` entry points (PRD 50 step 7, from wiki.py)."""
import os
from memory import render
from memory import store as m_store
from core import trace
from vault import index as v_index
from vault import pages as v_pages
from vault import search as v_search
from vault import store

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
    source = f"archived from {section} {render._today_iso()}"
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
    entry = {m_store.SEEN_FIELD: render._today_iso()}
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

    value = os.environ.get("WIKI_PATH") or dotenv_values(env_file).get("WIKI_PATH") or "~/llm-wiki-dev"
    return os.path.expanduser(value)


def main(argv) -> int:
    """`python -m wiki reindex`: rebuild the search index; refused (exit 2) while the app runs,
    because swapping the file under a running app would orphan its open handle."""
    import sys

    from ingest import batch

    if list(argv) != ["reindex"]:
        print("usage: python -m wiki reindex", file=sys.stderr)
        return 2
    running = batch.running_app()
    if running:
        print(f"refusing to reindex: {running}", file=sys.stderr)
        return 2
    print(f"reindexed {store.WIKI_PATH} -> {v_search.reindex()}")
    return 0
