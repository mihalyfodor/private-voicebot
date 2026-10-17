"""The generated `index.md`, `log.md`, the one-liners and the vault map (PRD 50 step 6, from wiki.py)."""
import json
from collections import Counter
from datetime import date
import memory
from core import trace
from vault import search as v_search
from vault import store

log = trace.logger("Wiki")


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
    if not store.exists():
        return []
    return [f"- [[{store.wikilink(rel)}]]" + (f" — {described}" if described else "")
            for rel, described in v_search._db().index_rows(store.INDEX_MAX_LINES)]


def rebuild_index() -> str:
    """Regenerate `index.md` from the pages' frontmatter — a pure function of the vault."""
    lines = index_lines()
    text = "# Index\n\n" + ("\n".join(lines) + "\n" if lines else "(no pages yet)\n")
    store._write(store.INDEX_FILE, text)
    return text


def index_text(max_tokens: int = store.WIKI_INDEX_TOKENS) -> str:
    """The alphabetical index trimmed to `max_tokens` — the human `index.md` order, for tests and tools;
    the prompt and the ingest calls see :func:`vault_map` instead (PRD 54).

    Built from the search index's `pages` table (no page read) rather than read from `index.md`,
    so every write hook's change is already reflected. "" when there is nothing to show.
    """
    if not store.exists():
        return ""
    lines = index_lines()
    if not lines:
        return ""
    lines = ["# Index", ""] + lines
    while lines and memory.tokens_est("\n".join(lines)) > max_tokens:
        lines.pop()
    return "\n".join(lines).strip()


def _map_folder(rel: str) -> str:
    return rel.split("/", 1)[0] if "/" in rel else "(root)"


def _recency(updated: str, today: date) -> float:
    """0.5 ** (days since `updated` / half-life), 1.0 for today or a future date, 0.0 without a date."""
    try:
        age = (today - date.fromisoformat(str(updated)[:10])).days
    except ValueError:
        return 0.0
    return 0.5 ** (max(age, 0) / store.MAP_RECENCY_HALF_LIFE_D)


def _cut(text: str, limit: int = store.MAP_ONE_LINER_CHARS) -> str:
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def map_folders(today: date | None = None) -> list:
    """The vault map's model: ``[{folder, count, tags, rows}]`` (folders by page count, then name),
    each folder's `rows` ranked by inbound links × `MAP_LINK_WEIGHT` + recency, then path.

    A pure fold over the `pages` table (one-liner, type, name, tags, updated, links): no page is read.
    """
    if not store.exists():
        return []
    today = today or memory._today()
    rows = v_search._db().map_rows()
    inbound = Counter()
    for rel, *_rest, links in rows:
        own = store.wikilink(rel)
        for target in {str(t).strip().removesuffix(".md") for t in json.loads(links or "[]")}:
            if target != own:
                inbound[target] += 1
    folders = {}
    for rel, _type, _name, tag_json, updated, one_liner, _links in rows:
        path = store.wikilink(rel)
        rank = inbound[path] * store.MAP_LINK_WEIGHT + _recency(updated, today)
        folders.setdefault(_map_folder(rel), []).append(
            (rank, path, f"- [[{path}]]" + (f" — {_cut(one_liner)}" if one_liner else ""), json.loads(tag_json or "[]")))
    out = []
    for name, pages in folders.items():
        pages.sort(key=lambda p: (-p[0], p[1]))
        tag_counts = Counter(t for p in pages for t in p[3])
        top = [t for t, _ in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:store.MAP_TOP_TAGS]]
        out.append({"folder": name, "count": len(pages), "tags": top, "rows": [p[2] for p in pages]})
    return sorted(out, key=lambda f: (-f["count"], f["folder"]))


def _render_map(folders: list, shown: dict) -> str:
    lines = ["# Vault map"]
    for f in folders:
        if f["folder"] not in shown:
            continue
        n = shown[f["folder"]]
        lines.append(f"## {f['folder']} ({f['count']})" + (f" — tags: {', '.join(f['tags'])}" if f["tags"] else ""))
        lines += f["rows"][:n]
        if f["count"] > n:
            lines.append(f"+{f['count'] - n} more")
    return "\n".join(lines)


def vault_map(max_tokens: int = store.WIKI_INDEX_TOKENS, today: date | None = None) -> str:
    """The vault as the prompt and the ingest calls see it (PRD 54): per folder a header (page count, top
    tags), its best pages and "+n more", fitted round-robin into `max_tokens` — every folder's header and
    best page first, then the next page of each, so no folder vanishes while the cap allows its header.

    Replaces `index_text` at its three prompt sites; `index.md` stays the alphabetical file. "" when
    there are no pages (or not even one header fits).
    """
    folders = map_folders(today)
    if not folders:
        return ""

    def fits(shown):
        return memory.tokens_est(_render_map(folders, shown)) <= max_tokens

    shown = {}
    for f in folders:                                       # headers first, in folder order
        if not fits({**shown, f["folder"]: 0}):
            break
        shown[f["folder"]] = 0
    if not shown:
        return ""
    open_ = [f for f in folders if f["folder"] in shown]
    while open_:                                            # then one more page per folder per round
        for f in list(open_):
            name = f["folder"]
            if shown[name] >= f["count"] or not fits({**shown, name: shown[name] + 1}):
                open_.remove(f)
                continue
            shown[name] += 1
    return _render_map(folders, shown)


def related_context(text: str, k: int = 5) -> str:
    """The pages most related to `text`, as a prompt block for the ingest calls (PRD 44).

    The index is trimmed to `WIKI_INDEX_TOKENS`, so a page it cut is still named here; `sources/`
    pages are left out. Trimmed line by line to `RELATED_TOKENS`; "" when there is no hit.
    """
    hits = [h for h in v_search.search(text, k + 3) if not h["path"].startswith("sources/")][:k]
    lines = []
    for hit in hits:
        fm, body = store.parse_page(store._read(hit["path"] + ".md"))
        line = f"- {hit['path']}: {fm.get('name') or hit['path'].rsplit('/', 1)[-1]}"
        summary = _summary_line(body)[:120]
        lines.append(line + (f" — {summary}" if summary else ""))
    head = "Existing related pages (update these rather than duplicating):"
    while lines and memory.tokens_est("\n".join([head] + lines)) > store.RELATED_TOKENS:
        lines.pop()
    return "\n".join([head] + lines) if lines else ""


def _log(entries: list) -> None:
    if not entries:
        return
    today = memory._today_iso()
    with open(store._path(store.LOG_FILE), "a") as f:
        for entry in entries:
            f.write(f"- {today} {entry}\n")
