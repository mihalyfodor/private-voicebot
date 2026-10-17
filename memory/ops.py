"""Validate and apply profile ops, the audit log, the editor entry points and source forgetting (docs/15-memory-v2.md, docs/16-memory-editor.md; PRD 50 step 8, from memory.py)."""
import contextlib
import copy
import difflib
import json
import re
import threading
from core import trace
from memory import identity as m_identity
from memory import render as m_render
from memory import store
from core import util

log = trace.logger("Memory")


DUP_RATIO = 0.85                                                  # _similar() threshold



def _transcript(turns: list) -> str:
    return "\n".join(
        f"{str(t.get('role', 'user')).capitalize()}: {t.get('content', '')}"
        for t in turns
        if isinstance(t, dict) and t.get("content")
    )



def extract_json(text: str, opener: str = "[", closer: str = "]"):
    """Parse the first `opener`...`closer` block, tolerating ```json fences and prose."""
    if not text:
        return None
    stripped = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip()).strip()
    for candidate in (stripped, text):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start == -1 or end <= start:
            continue
        try:
            return json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            continue
    return None



def _public(profile: dict) -> dict:
    """The profile without bookkeeping, for prompts."""
    return {k: v for k, v in profile.items() if k not in ("meta", "version", "updated")}



def _norm(text) -> str:
    """Lowercased, punctuation-free form used for all fact comparisons."""
    return re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower()).strip()



def _similar(a, b, ratio: float = DUP_RATIO) -> bool:
    """True when `a` and `b` are the same fact modulo casing/punctuation/small edits.

    The single fuzzy-match rule for the project — the eval harness imports it too, so the
    duplicate detection in reports uses exactly the same threshold as reflection.
    """
    a, b = _norm(a), _norm(b)
    return bool(a) and bool(b) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() > ratio)



def _find(entries: list, key_field: str, key_value) -> int:
    target = _norm(key_value)
    for i, entry in enumerate(entries):
        if _norm(entry.get(key_field)) == target:
            return i
    return -1



def _episodic_entry(value: dict) -> dict:
    entry = {
        "date": str(value.get("date") or util._today_iso())[:10],
        "text": str(value["text"]).strip(),
        "ttl_days": store.DEFAULT_TTL_DAYS,
        "importance": m_render._importance(value),
        store.SEEN_FIELD: util._today_iso(),
    }
    try:
        entry["ttl_days"] = max(1, int(value.get("ttl_days", store.DEFAULT_TTL_DAYS)))
    except (TypeError, ValueError):
        pass
    return entry



def _supersede(section: dict, key: str, old_value: str) -> None:
    superseded = section.setdefault("superseded", {})
    if not isinstance(superseded, dict):
        superseded = section["superseded"] = {}
    history = superseded.setdefault(key, [])
    if old_value not in history:
        history.append(old_value)



def _enforce_caps(profile: dict) -> list:
    """Trim lists to CAPS: episodics by score, the rest stale-first then oldest `seen` first."""
    dropped = []
    for section, cap in store.CAPS.items():
        entries = profile.get(section, [])
        if len(entries) <= cap:
            continue
        if section == "episodic":
            keep = sorted(entries, key=m_render._episodic_score, reverse=True)[:cap]
            profile[section] = [e for e in entries if e in keep]
        else:
            # evict the entries the user is least likely to miss: already stale ones, then the
            # ones confirmed longest ago; position only breaks ties.
            order = sorted(range(len(entries)),
                           key=lambda i: (not m_render.is_stale(entries[i]), m_render._seen_date(entries[i]), i))
            evicted = set(order[:len(entries) - cap])
            profile[section] = [e for i, e in enumerate(entries) if i not in evicted]
        dropped.append(f"{section}: dropped {len(entries) - cap} over cap {cap}")
    return dropped



_ENTRY_FIELD = re.compile(r"^(\w+)\[(\d+)\]\.(\w+)$")

_ENTRY = re.compile(r"^(\w+)\[(\d+)\]$")



def _parse_path(path: str) -> tuple:
    """Classify an op path once, for all three op kinds.

    Returns ``(kind, *parts)``, one of::

        ("entry_field", section, index, field)   projects[0].note
        ("entry", section, index)                projects[0]
        ("list", section)                        people[] / people
        ("scalar", section, key)                 identity.name
        ("bad", reason)                          anything else

    Models are inconsistent about the ``[]`` suffix, so a bare list-section name counts too.
    An explicit index only addresses one entry for UPDATE/DELETE; ADD always appends (that is
    what the models mean by ``people[0]``, and what the editor sends is a bare index).
    """
    match = _ENTRY_FIELD.match(path)
    if match and match.group(1) in store.LIST_SECTIONS:
        return "entry_field", match.group(1), int(match.group(2)), match.group(3)
    match = _ENTRY.match(path)
    if match and match.group(1) in store.LIST_SECTIONS:
        return "entry", match.group(1), int(match.group(2))
    base = re.sub(r"\[\d*\]$", "", path).strip()
    if base in store.LIST_SECTIONS:
        return "list", base
    if "." in path:
        section, _, key = path.partition(".")
        if section not in store.SCALAR_SECTIONS:
            return "bad", f"unknown section {section!r}"
        if not key:
            return "bad", "missing key"
        return "scalar", section, key
    return "bad", f"unknown path {path!r}"



def _is_scalar(value) -> bool:
    """A usable leaf value: a string or number that isn't blank.

    Booleans are deliberately excluded (they would stringify to "True"); the only boolean field
    in the schema is `stale`, handled by :func:`_apply_stale_op`.
    """
    return isinstance(value, (str, int, float)) and not isinstance(value, bool) \
        and bool(str(value).strip())



def _as_bool(value):
    """True/False for a JSON boolean or its string spelling, else None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None



def _touch(entry: dict) -> dict:
    """Mark `entry` as confirmed today (ADD / UPDATE / CONFIRM all count as a mention)."""
    entry[store.SEEN_FIELD] = util._today_iso()
    return entry



def _revive(entry: dict) -> None:
    """Seen today and not stale — the one way an entry comes back into the prompt."""
    _touch(entry)
    entry.pop(store.STALE_FIELD, None)



def _entries_at(profile, section, index, op):
    """``(entries, entry)`` for `section[index]`, or `(entries, None)` when the index is stale.

    The index comes from the UI's last snapshot or from the model, so it can point past the end
    after a concurrent delete; rejecting is how both callers handle that.
    """
    entries = profile.setdefault(section, [])
    if index >= len(entries):
        _reject(op, f"{section}[{index}] does not exist")
        return entries, None
    return entries, entries[index]



def _apply_entry_op(profile, section, index, kind, value, op) -> bool:
    """`people[1]` — one whole list entry addressed by position (what the editor sends)."""
    entries, entry = _entries_at(profile, section, index, op)
    if entry is None:
        return False
    if kind == "DELETE":
        entries.pop(index)
        return True
    if not isinstance(value, dict):
        _reject(op, f"{section}[{index}] needs an object value")
        return False
    clean = {k: str(v).strip() for k, v in value.items() if _is_scalar(v)}
    if not clean or all(_norm(entry.get(k)) == _norm(v) for k, v in clean.items()):
        return False
    entries[index] = _touch({**entry, **clean})
    return True



def _apply_stale_op(entry: dict, value, op) -> bool:
    """`people[1].stale` — the editor retiring (true) or restoring (false) one entry.

    The only boolean field in the schema, and the only op that can set it: staleness otherwise
    comes from :func:`_mark_stale`. Restoring also refreshes `seen`, exactly like a CONFIRM.
    """
    flag = _as_bool(value)
    if flag is None:
        _reject(op, f"{store.STALE_FIELD} needs true or false")
        return False
    if flag:
        if m_render.is_stale(entry):
            return False
        entry[store.STALE_FIELD] = True
        return True
    if not m_render.is_stale(entry) and entry.get(store.SEEN_FIELD) == util._today_iso():
        return False
    _revive(entry)
    return True



def _apply_entry_field_op(profile, section, index, field, kind, value, op) -> bool:
    """`projects[0].note` — a single field of one list entry (models reach for this form)."""
    _, entry = _entries_at(profile, section, index, op)
    if entry is None:
        return False
    if kind == "DELETE":
        if field not in entry:
            _reject(op, f"{section}[{index}] has no {field!r}")
            return False
        del entry[field]
        return True
    if field == store.STALE_FIELD:
        return _apply_stale_op(entry, value, op)
    if not _is_scalar(value):
        _reject(op, f"{section}[{index}].{field} needs a scalar value")
        return False
    if _norm(entry.get(field)) == _norm(value):
        return False
    entry[field] = str(value).strip()
    if field != store.SEEN_FIELD:
        _touch(entry)
    return True



def _apply_confirm(profile: dict, path: str, value, op) -> tuple:
    """`CONFIRM people[2]` — "still true, mentioned again": `seen` = today, `stale` cleared.

    Returns ``(applied, canonical_path)``. Addressed either by index (``people[2]``) or, like
    every other list op, by the identifying key in the value (``{"name": "Anna"}``) — the value
    wins when both are present. A stale entry is never refused: this is how it comes back.
    """
    target = _parse_path(path)
    if target[0] == "entry_field":                 # people[2].note — confirm the whole entry
        target = ("entry", target[1], target[2])
    if target[0] not in ("entry", "list"):
        _reject(op, f"CONFIRM needs a list entry path, got {path!r}")
        return False, path
    section = target[1]
    entries = profile.setdefault(section, [])
    key_field = store.LIST_KEY[section]
    if isinstance(value, dict) and value.get(key_field):
        index = _find(entries, key_field, value[key_field])
        path = f"{section}[]"
        if index == -1:
            _reject(op, f"no {section} entry named {value[key_field]!r}")
            return False, path
    elif target[0] == "entry" and target[2] < len(entries):
        index = target[2]
    else:
        _reject(op, f"{path} does not exist")
        return False, path
    _revive(entries[index])
    return True, path



def document_source(op: dict) -> str:
    """The document title an op attributes itself to, or "" for `conversation` / unattributed."""
    source = " ".join(str(op.get(store.SOURCE_FIELD) or "").split())
    return "" if source.lower() in ("", "conversation", "user", "none") else source[:120]



def _stamp_source(profile: dict, target: tuple, kind: str, value, source: str) -> None:
    """Record where a just-applied op came from: `source` on the list entry, or
    `<section>.sources[<key>]` for a scalar; a conversation op clears an older document stamp."""
    if target[0] == "scalar":
        _, section, key = target
        stamps = profile.setdefault(section, {}).setdefault("sources", {})
        if source and kind != "DELETE":
            stamps[key] = source
        else:
            stamps.pop(key, None)
        if not stamps:
            del profile[section]["sources"]
        return
    if kind == "DELETE" or not source:
        return
    section = target[1]
    entries = profile.get(section, [])
    key = value.get(store.LIST_KEY[section]) if isinstance(value, dict) else None
    # a value carrying the identifying key names the entry (how the model addresses lists);
    # otherwise the path's index does (people[2], people[2].note)
    index = _find(entries, store.LIST_KEY[section], key) if key else (target[2] if len(target) > 2 else -1)
    if 0 <= index < len(entries):
        entries[index][store.SOURCE_FIELD] = source



def apply_ops(profile: dict, ops: list, rejected: list | None = None) -> tuple:
    """Validate and apply LLM-proposed `ops` to `profile`; returns (profile, applied_ops).

    This is the only place the profile changes shape: the model proposes, Python decides.
    Anything malformed, unknown or already true is rejected (printed) and left out of
    `applied_ops`, so the ops log records exactly what really landed on disk.

    Pass a list as `rejected` to also collect the rejections as
    ``{"op": <the op>, "reason": "..."}`` — the memory editor shows the reason to the user.
    """
    applied = []
    with _collecting(rejected):
        for op in ops or []:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op", "")).strip().upper()
            path = str(op.get("path", "")).strip()
            value = op.get("value")
            if kind == "NOOP":
                continue
            if kind == "CONFIRM":
                ok, path = _apply_confirm(profile, path, value, op)
                if ok:
                    applied.append({"op": kind, "path": path, "value": value,
                                    "reason": op.get("reason") or "confirm"})
                continue
            if kind not in ("ADD", "UPDATE", "DELETE"):
                _reject(op, f"unknown op {kind!r}")
                continue

            target = _parse_path(path)
            if target[0] == "entry_field":
                _, section, index, field = target
                ok = _apply_entry_field_op(profile, section, index, field, kind, value, op)
            elif target[0] == "entry":
                _, section, index = target
                key_field = store.LIST_KEY[section]
                if kind == "ADD" or (isinstance(value, dict) and value.get(key_field)):
                    # ADD always appends, and a value carrying the identifying key wins over the
                    # index — that is how the LLM addresses entries (docs/15-memory-v2.md).
                    path = f"{section}[]"
                    ok = _apply_list_op(profile, section, kind, value, op)
                else:
                    ok = _apply_entry_op(profile, section, index, kind, value, op)
            elif target[0] == "list":
                section = target[1]
                path = f"{section}[]"    # log one canonical path whatever form the model used
                ok = _apply_list_op(profile, section, kind, value, op)
            elif target[0] == "scalar":
                ok = _apply_scalar_op(profile, target[1], target[2], kind, value, op)
            else:
                _reject(op, target[1])
                continue
            if not ok:
                continue
            source = document_source(op)
            _stamp_source(profile, target, kind, value, source)
            record = {"op": kind, "path": path, "value": value, "reason": op.get("reason", "")}
            if source:
                record[store.SOURCE_FIELD] = source
            applied.append(record)

    for note in _enforce_caps(profile):
        log.info(f"cap enforced — {note}")
    return profile, applied



def _apply_scalar_op(profile, section, key, kind, value, op) -> bool:
    """`identity.name` — one string under a scalar section; UPDATE keeps the old value in
    `superseded` so the editor can offer "restore"."""
    target = profile.setdefault(section, {})
    if m_identity._is_alias(section, key):
        return m_identity._apply_alias_op(target, kind, value, op)
    if section == "identity" and key == "name" and kind in ("ADD", "UPDATE") \
            and m_identity._alias_guard(target, value, op):
        _reject(op, "name matches a known alias; not changing the real name")
        return False
    if kind == "DELETE":
        if key not in target:
            _reject(op, "key not present")
            return False
        del target[key]
        return True
    if not isinstance(value, str) or not value.strip():
        _reject(op, f"{section}.{key} needs a non-empty string value, got {type(value).__name__}")
        return False
    value = value.strip()
    old = target.get(key)
    if _norm(old) == _norm(value):   # same fact, different casing/punctuation
        return False
    if kind == "UPDATE" and isinstance(old, str) and old:
        _supersede(target, key, old)
    target[key] = value
    return True



def _apply_list_op(profile, section, kind, value, op) -> bool:
    """`people[]` — a whole entry addressed by its identifying key (`name` / `what` / `text`),
    which is how the model refers to list entries; the entry is matched, appended or removed."""
    entries = profile.setdefault(section, [])
    key_field = store.LIST_KEY[section]
    if not isinstance(value, dict):
        _reject(op, f"{section}[] needs an object value")
        return False
    if not value.get(key_field):
        _reject(op, f"{section}[] needs a {key_field!r} key")
        return False

    if section == "episodic":
        if kind != "ADD":
            _reject(op, "episodic supports ADD only")
            return False
        entry = _episodic_entry(value)
        if _find(entries, "text", entry["text"]) != -1:
            return False
        entries.append(entry)
        return True

    index = _find(entries, key_field, value[key_field])
    clean = {k: v for k, v in value.items() if _is_scalar(v)}
    if kind == "DELETE":
        if index == -1:
            _reject(op, f"no {section} entry named {value[key_field]!r}")
            return False
        entries.pop(index)
        return True
    if index == -1:
        entries.append(_touch(clean))  # ADD, or an UPDATE of something we never stored
        return True
    if kind == "ADD" and {k: v for k, v in entries[index].items()
                          if k not in (store.SEEN_FIELD, store.STALE_FIELD, store.SOURCE_FIELD)} == clean:
        return False   # the very same entry again; a re-mention is a CONFIRM, not an ADD
    entries[index] = _touch({**entries[index], **clean})
    return True



#: Where `_reject` also files its reasons, when a caller asked to collect them. Thread-local so
#: a UI edit and a background save can never read each other's rejections.
_collector = threading.local()



@contextlib.contextmanager
def _collecting(rejected: list | None):
    """Route rejections of the ops applied in this block into `rejected` as well as stdout."""
    previous = getattr(_collector, "rejected", None)
    _collector.rejected = rejected
    try:
        yield
    finally:
        _collector.rejected = previous



def _reject(op: dict, reason: str) -> None:
    """Print the rejection and, inside `_collecting`, record it for the caller (the UI)."""
    log.info(f"rejected op {op.get('op')} {op.get('path')!r}: {reason}")
    rejected = getattr(_collector, "rejected", None)
    if rejected is not None:
        rejected.append({"op": op, "reason": reason})



def _log_ops(applied: list) -> None:
    if not applied:
        return
    stamp = util._stamp("%H:%M:%S")
    with open(store.OPS_LOG_PATH, "a") as f:
        for op in applied:
            f.write(json.dumps({"ts": stamp, **op}, ensure_ascii=False) + "\n")



# --------------------------------------------------------------------------- editor


def fact_count(profile: dict) -> int:
    """User-visible facts: scalar keys (superseded history excluded) + list entries."""
    total = sum(len([k for k in profile.get(s, {}) if k not in store.BOOKKEEPING_KEYS]) for s in store.SCALAR_SECTIONS)
    return total + sum(len(profile.get(s, [])) for s in store.LIST_SECTIONS)



def _shown(section: str, entry: dict, backfill: str) -> dict:
    """One list entry as the panel wants it: always a `seen` and a `stale`, plus the window
    (`stale_after`) so a row can dim as it ages, or `expired` for episodics."""
    shown = {**entry, store.SEEN_FIELD: entry.get(store.SEEN_FIELD) or backfill, store.STALE_FIELD: m_render.is_stale(entry)}
    if section == "episodic":
        shown["expired"] = m_render.is_expired(entry)
    else:
        shown["stale_after"] = store.STALE_DAYS[section]
    return shown



def snapshot(profile: dict) -> dict:
    """What the memory panel needs: the profile, the rendered block, the fact count.

    Stale and expired entries are kept (the panel groups/greys them) and carry ``stale`` /
    ``expired`` flags; those flags and the backfilled ``seen`` live in the returned copy only.
    """
    public = dict(_public(profile))
    backfill = store._seen_default(profile.get("updated"))
    for section in store.LIST_SECTIONS:
        public[section] = [_shown(section, e, backfill) for e in profile.get(section, [])]
    return {"profile": public, "rendered": m_render.render(profile), "count": fact_count(profile)}



def _cap_reason(profile: dict, op: dict) -> str | None:
    """Why a hand ADD to a full list must be refused.

    LLM ops silently evict the oldest entry when a list is over cap; a hand edit says so instead,
    so the user can decide what to delete.
    """
    if str(op.get("op", "")).strip().upper() != "ADD":
        return None
    target = _parse_path(str(op.get("path", "")).strip())
    if target[0] not in ("list", "entry"):
        return None
    section = target[1]
    cap = store.CAPS[section]
    if len(profile.get(section, [])) >= cap:
        return f"{section} is at its cap of {cap} — delete an entry first"
    return None



def _ui_op(op) -> dict:
    """One op from the panel, with its audit tag forced back into `UI_REASONS`."""
    op = op if isinstance(op, dict) else {}
    reason = str(op.get("reason") or "ui")
    return {"op": op.get("op"), "path": op.get("path"), "value": op.get("value"),
            "reason": reason if reason in store.UI_REASONS else "ui"}



def edit_many(ops: list) -> tuple:
    """Apply a batch of hand-made ops atomically; ``(profile, None)`` or ``(profile, error)``.

    The editor never writes memory.json itself: a UI edit walks the same validated
    load -> apply_ops -> write -> audit path as an LLM op, under the same lock, and is tagged
    ``ui`` / ``ui:restore`` in memory_ops.jsonl.

    All-or-nothing, which is what the drag-and-drop moves need (promoting an episodic is one
    ADD plus one DELETE and the profile must never show both entries or neither): every op is
    validated and applied to a *copy*, and the first one that fails aborts the batch with
    ``{"path": ..., "reason": ...}`` — nothing is written and nothing is logged.
    """
    ops = [_ui_op(op) for op in ops] if isinstance(ops, list) else []
    if not ops:
        return store.load_profile(), {"path": None, "reason": "no ops"}
    with store._LOCK:
        stored = store.load_profile()
        working = copy.deepcopy(stored)
        applied_all = []
        for single in ops:
            capped = _cap_reason(working, single)
            if capped:                   # checked before apply_ops: caps there evict silently
                _reject(single, capped)
                return stored, {"path": single["path"], "reason": capped}
            rejected = []
            working, applied = apply_ops(working, [single], rejected)
            if not applied:
                reason = rejected[0]["reason"] if rejected else "nothing changed"
                return stored, {"path": single["path"], "reason": reason}
            applied_all.extend(applied)
        store._write_profile(working)
        _log_ops(applied_all)
        return working, None



def edit(op: dict) -> tuple:
    """Apply exactly one hand-made op; returns ``(profile, None)`` or ``(profile, reason)``."""
    profile, error = edit_many([op])
    return profile, (error["reason"] if error else None)



ARCHIVABLE = ("people", "projects", "recurring")



def archive(section: str, index) -> tuple:
    """Move one profile entry into the wiki and mark it stale; ``(profile, error)``.

    The bridge PRD 18 promised: a fact leaving working memory is *filed*, not forgotten. The
    entry's fields and its last `seen` date become a `## History` bullet on the matching wiki
    page (creating that page if needed), and only once that succeeded is the profile entry
    retired through the normal :func:`edit_many` path. See docs/19-wiki-layer.md.
    """
    if section not in ARCHIVABLE:
        return store.load_profile(), {"path": section, "reason": f"cannot archive {section}"}
    try:
        index = int(index)
    except (TypeError, ValueError):
        return store.load_profile(), {"path": section, "reason": f"bad index {index!r}"}
    with store._LOCK:
        profile = store.load_profile()
        entries = profile.get(section, [])
        if not 0 <= index < len(entries):
            return profile, {"path": f"{section}[{index}]", "reason": f"{section}[{index}] does not exist"}
        entry = entries[index]
        try:
            from memory import bridge as v_bridge

            page, reason = v_bridge.archive_entry(section, entry)
        except Exception as exc:  # noqa: BLE001
            page, reason = None, str(exc)
        if reason:
            return profile, {"path": f"{section}[{index}]", "reason": f"wiki: {reason}"}
        if m_render.is_stale(entry):
            return profile, None        # already retired; the page is what mattered
        path = f"{section}[{index}].{store.STALE_FIELD}"
        profile, error = edit_many([{"op": "UPDATE", "path": path, "value": True, "reason": "ui"}])
        if error:
            error["reason"] = f"filed to {page} but could not retire the entry: {error['reason']}"
        return profile, error



def pull(path: str, section: str) -> tuple:
    """The reverse of :func:`archive`: a live entry built from a wiki page; ``(profile, error)``."""
    if section not in ARCHIVABLE:
        return store.load_profile(), {"path": path, "reason": f"cannot pull into {section}"}
    with store._LOCK:
        try:
            from memory import bridge as v_bridge

            entry, reason = v_bridge.entry_from_page(path, section)
        except Exception as exc:  # noqa: BLE001
            entry, reason = None, str(exc)
        if reason:
            return store.load_profile(), {"path": path, "reason": reason}
        profile = store.load_profile()
        key_field = store.LIST_KEY[section]
        found = _find(profile.get(section, []), key_field, entry.get(key_field))
        if found != -1 and m_render.is_stale(profile[section][found]):
            # the entry this page was archived from is still on disk: revive it in place
            # rather than growing a second copy (`stale: false` also refreshes `seen`).
            op = {"op": "UPDATE", "path": f"{section}[{found}].{store.STALE_FIELD}",
                  "value": False, "reason": "ui:restore"}
        else:
            op = {"op": "ADD", "path": f"{section}[]", "value": entry, "reason": "ui"}
        return edit_many([op])



def reset() -> dict:
    """Forget everything: write an empty profile and audit it as one `DELETE *` line."""
    with store._LOCK:
        profile = store._empty_profile()
        store._write_profile(profile)
        _log_ops([{"op": "DELETE", "path": "*", "value": None, "reason": "ui:reset"}])
        return profile



def source_entries(title: str) -> list:
    """The memory entries `forget_source(title)` would drop, as short lines (`people: Anna`,
    `identity.city: Kyoto`) — read-only, for an approval's proposal (docs/52-approvals.md)."""
    title = " ".join(str(title or "").split())
    if not title:
        return []
    profile = store.load_profile()
    out = [f"{section}: {e.get(store.LIST_KEY[section]) or ''}".rstrip()
           for section in store.LIST_SECTIONS for e in profile.get(section, []) if e.get(store.SOURCE_FIELD) == title]
    for section in store.SCALAR_SECTIONS:
        values = profile.get(section, {})
        stamps = values.get("sources") if isinstance(values.get("sources"), dict) else {}
        out += [f"{section}.{key}: {values[key]}" for key, s in stamps.items()
                if s == title and key in values and key not in store.BOOKKEEPING_KEYS]
    return out



def forget_source(title: str) -> int:
    """Drop every memory entry and scalar key learned from the document `title`; the count.

    Called by `sources.forget` under the vault lock — the same rule the wiki notes follow."""
    title = " ".join(str(title or "").split())
    if not title:
        return 0
    with store._LOCK:
        profile = store.load_profile()
        removed = 0
        for section in store.LIST_SECTIONS:
            kept = [e for e in profile.get(section, []) if e.get(store.SOURCE_FIELD) != title]
            removed += len(profile.get(section, [])) - len(kept)
            profile[section] = kept
        for section in store.SCALAR_SECTIONS:
            values = profile.get(section, {})
            stamps = values.get("sources") if isinstance(values.get("sources"), dict) else {}
            for key in [k for k, s in stamps.items() if s == title]:
                stamps.pop(key)
                if key in values and key not in store.BOOKKEEPING_KEYS:
                    del values[key]
                    removed += 1
            if not stamps:
                values.pop("sources", None)
        if removed:
            store._write_profile(profile)
            _log_ops([{"op": "DELETE", "path": f"source:{title}", "value": removed, "reason": "forget"}])
            log.info(f"forgot {removed} entr{'y' if removed == 1 else 'ies'} from \"{title}\"")
        return removed
