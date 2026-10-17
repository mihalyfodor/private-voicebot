"""Page ops (the only way the model writes) and the human editor entry points (PRD 28; PRD 50 step 6, from wiki.py)."""
import os
import re
import memory
from core import trace, vault_lock
from vault import index as v_index
from vault import search
from vault import store

log = trace.logger("Wiki")


# --- page ops ----------------------------------------------------------------------
# Each op returns (log line, None) when applied, or (None, reason) when refused.


def _check_size(rel: str, addition: str) -> str | None:
    size = len(store._read(rel).encode()) + len(addition.encode())
    if size > store.MAX_PAGE_BYTES:
        return f"page {store.wikilink(rel)} would exceed {store.MAX_PAGE_BYTES // 1024} kB"
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
    rel = store.normalise_path(op.get(key), user=user)
    if rel is None:
        return None, f"bad path {op.get(key)!r}"
    if reserved and rel in store.RESERVED:
        return None, f"{store.wikilink(rel)} is not a page"
    if exists is True and not store.page_exists(rel):
        return None, f"no such page {store.wikilink(rel)}"
    if exists is False and store.page_exists(rel):
        return None, f"{store.wikilink(rel)} already exists"
    return rel, None


def new_page(op: dict, source: str) -> tuple:
    """`{path, type, name, summary}` → a fresh page with the six frontmatter keys and sections."""
    rel, reason = _target(op, exists=False, reserved=True)
    if reason:
        return None, reason
    page_type = str(op.get("type") or "note").strip().lower()
    name = str(op.get("name") or "").strip() or os.path.basename(rel)[:-3]
    summary = str(op.get("summary") or "").strip()
    if len(summary) > store.MAX_NOTE_CHARS:
        return None, f"summary longer than {store.MAX_NOTE_CHARS} characters"
    store._write(rel, _fresh_page(page_type, name, summary, source))
    return f"filed [[{store.wikilink(rel)}]] (source: {source})", None


def _fresh_page(page_type: str, name: str, summary: str, source: str) -> str:
    today = memory._today_iso()
    fm = {"type": page_type, "name": name, "tags": [], "source": source,
          "updated": today, "seen": today}
    return store._page_text(fm, f"# {name}\n\n## Summary\n{summary}\n\n## Notes\n\n## Related\n")


def append_note(op: dict, source: str) -> tuple:
    """`{path, text, section?}` → one dated bullet at the end of `## Notes` (or another section)."""
    rel, reason = _target(op, exists=True, reserved=True)
    if reason:
        return None, reason
    text = " ".join(str(op.get("text") or "").split())
    if not text:
        return None, "empty note"
    if len(text) > store.MAX_NOTE_CHARS:
        return None, f"note longer than {store.MAX_NOTE_CHARS} characters"
    heading = str(op.get("section") or "## Notes")
    if heading not in store.SECTIONS:
        return None, f"unknown section {heading!r}"
    today = memory._today_iso()
    line = f"- {today} {text} (source: {source})"
    oversize = _check_size(rel, line)
    if oversize:
        return None, oversize
    store._edit_page(rel, lambda body: _insert_into_section(body, heading, line),
               defaults={"seen": today})
    return f"noted on [[{store.wikilink(rel)}]] (source: {source})", None


def set_field(op: dict, source: str) -> tuple:
    """`{path, key, value}` → one frontmatter key, from the fixed set. The body is untouched."""
    rel, reason = _target(op, exists=True)
    if reason:
        return None, reason
    key = str(op.get("key") or "").strip()
    if key not in store.SETTABLE_FIELDS:
        return None, f"unknown frontmatter key {key!r}"
    value = op.get("value")
    if key == "tags":
        value = value if isinstance(value, list) else [p.strip() for p in str(value or "").split(",") if p.strip()]
    else:
        value = " ".join(str(value or "").split())
        if len(value) > store.MAX_NOTE_CHARS:
            return None, f"value longer than {store.MAX_NOTE_CHARS} characters"
    store._edit_page(rel, **{key: value})     # `updated` is bumped unless it *is* the key being set
    return f"set {key} on [[{store.wikilink(rel)}]] (source: {source})", None


def add_decision(op: dict, source: str) -> tuple:
    """`{title, text}` → a dated `## YYYY-MM-DD <title>` section appended to `decisions.md`."""
    title = " ".join(str(op.get("title") or "").split())
    text = " ".join(str(op.get("text") or "").split())
    if not title:
        return None, "decision needs a title"
    if len(text) > store.MAX_NOTE_CHARS:
        return None, f"decision longer than {store.MAX_NOTE_CHARS} characters"
    today = memory._today_iso()
    block = f"\n## {today} {title}\n{text}\nsource: {source}\n"
    oversize = _check_size(store.DECISIONS_FILE, block)
    if oversize:
        return None, oversize
    store._edit_page(store.DECISIONS_FILE, lambda body: body.rstrip("\n") + "\n" + block,
               defaults={"type": "decisions", "name": "Decisions"})
    return f"decided {title} in [[decisions]] (source: {source})", None


def link(op: dict, source: str) -> tuple:
    """`{path, target}` → a `[[target]]` bullet under `## Related`, once."""
    rel, target = store.normalise_path(op.get("path")), store.normalise_path(op.get("target"))
    if rel is None or target is None:
        return None, f"bad path {op.get('path')!r} -> {op.get('target')!r}"
    if not store.page_exists(rel):
        return None, f"no such page {store.wikilink(rel)}"
    if rel == target:
        return None, "a page cannot link to itself"
    mark = f"[[{store.wikilink(target)}]]"
    if mark in store.parse_page(store._read(rel))[1]:
        return None, f"{store.wikilink(rel)} already links to {store.wikilink(target)}"
    store._edit_page(rel, lambda body: _insert_into_section(body, "## Related", f"- {mark}"))
    return f"linked [[{store.wikilink(rel)}]] -> {mark} (source: {source})", None


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
    store.ensure_vault()
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
        v_index.rebuild_index()
        v_index._log(entries)
        if commit:
            store._commit(f"wiki: {len(applied)} op{'s' if len(applied) != 1 else ''} ({source_default})")
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
    rel = store.normalise_path(path, user=True)
    if rel is not None and (rel in store.RESERVED or (fixed and rel == store.DECISIONS_FILE)):
        return None, f"{store.wikilink(rel)} is not an editable page"
    return _target({"path": path}, exists=exists, user=True)


def _user_commit(entry: str, message: str) -> None:
    v_index.rebuild_index()
    v_index._log([entry])
    store._commit(message)


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
        current = store._read(rel)
        if store.etag(current) != str(tag or ""):
            return {"conflict": True, "text": current, "etag": store.etag(current)}, None
        text = str(text or "")
        if len(text.encode()) > store.MAX_PAGE_BYTES:
            return None, f"page larger than {store.MAX_PAGE_BYTES // 1024} kB"
        fm, _ = store.parse_page(text)
        missing = [key for key in store.FIELDS if key not in fm]
        if missing:
            return None, f"frontmatter missing {', '.join(missing)}"
        if text == current:
            return {"etag": store.etag(current)}, None
        text = _bump_updated(text)
        store._write(rel, text)
        _user_commit(f"edited [[{store.wikilink(rel)}]] (source: {USER_SOURCE})",
                     f"wiki: edit {store.wikilink(rel)} ({USER_SOURCE})")
        return {"etag": store.etag(text)}, None


def create_page(path: str) -> tuple:
    """A fresh page at a user path (any single-level folder); ``(path, None)`` or ``(None, reason)``."""
    with vault_lock.LOCK:
        store.ensure_vault()
        rel, reason = _user_target(path, exists=False)
        if reason:
            return None, reason
        folder, base = os.path.split(rel)
        store._write(rel, _fresh_page(_TYPES.get(folder, "note"), base[:-3], "", USER_SOURCE))
        _user_commit(f"created [[{store.wikilink(rel)}]] (source: {USER_SOURCE})",
                     f"wiki: create {store.wikilink(rel)} ({USER_SOURCE})")
        return store.wikilink(rel), None


CARD_SOURCE = "written answer"


def save_card(path: str, title: str, markdown: str, sources: list, run_id: str = "") -> tuple:
    """A written answer filed as a note (PRD 43): ``(path, None)`` or ``(None, reason)``.

    One validated user write (any single-level folder but `raw/`, no existing page) and one
    commit: frontmatter `type: note`, `name: <title>`, `source: written answer`, `sources:` the
    card's list (+ `run_id:` for a research card, PRD 54); the body is the card markdown as it was rendered.
    """
    with vault_lock.LOCK:
        store.ensure_vault()
        rel, reason = _user_target(path, exists=False)
        if reason:
            return None, reason
        today = memory._today_iso()
        fm = {"type": "note", "name": title, "tags": [], "source": CARD_SOURCE, "updated": today,
              "seen": today, "sources": [str(x).replace(",", " ") for x in sources]}
        if run_id:
            fm["run_id"] = str(run_id)
        body = markdown.strip() + "\n"
        if not body.startswith("# "):
            body = f"# {title}\n\n" + body
        text = store._page_text(fm, body)
        if len(text.encode()) > store.MAX_PAGE_BYTES:
            return None, f"page larger than {store.MAX_PAGE_BYTES // 1024} kB"
        store._write(rel, text)
        _user_commit(f"filed [[{store.wikilink(rel)}]] (source: {CARD_SOURCE})",
                     f"wiki: save card {store.wikilink(rel)} ({CARD_SOURCE})")
        return store.wikilink(rel), None


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
        old, new = store.wikilink(rel), store.wikilink(new_rel)
        basename = os.path.basename(old)
        unique = sum(os.path.basename(p)[:-3] == basename for p in store.page_paths()) == 1
        patterns = [_link_pattern(old)]
        if unique and basename != old:
            patterns.append(_link_pattern(basename))
        touched = 0
        for other in store.page_paths():
            if other == rel:
                continue
            text = store._read(other)
            rewritten = text
            for pattern in patterns:
                rewritten = pattern.sub(lambda m: m.group(1) + new, rewritten)
            if rewritten != text:
                store._write(other, _bump_updated(rewritten))
                touched += 1
        moved = _bump_updated(store._read(rel))
        store._write(new_rel, moved)          # the new file first: a failing index hook never loses the page
        os.remove(store._path(rel))
        search.unindex_page(rel)
        _user_commit(f"renamed [[{old}]] → [[{new}]] ({touched} links, source: {USER_SOURCE})",
                     f"wiki: rename {old} → {new} ({touched} links)")
        return {"path": new, "links": touched}, None


def delete_page(path: str) -> tuple:
    """Remove a page (one commit); ``(path, None)`` or ``(None, reason)``. Lint reports the broken links."""
    with vault_lock.LOCK:
        rel, reason = _user_target(path, exists=True)
        if reason:
            return None, reason
        store._git("rm", "-q", "-f", "--", rel, quiet=True)
        if os.path.exists(store._path(rel)):
            os.remove(store._path(rel))
        search.unindex_page(rel)
        _user_commit(f"deleted [[{store.wikilink(rel)}]] (source: {USER_SOURCE})",
                     f"wiki: delete {store.wikilink(rel)}")
        return store.wikilink(rel), None
