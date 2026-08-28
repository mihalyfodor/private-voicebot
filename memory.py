"""Memory v2 — a structured user profile with LLM-proposed ops applied in Python.

Two tiers live in ``memory.json``:

* the durable **profile** — ``identity`` / ``preferences`` (scalar strings) and the
  ``people`` / ``projects`` / ``recurring`` lists;
* dated **episodic** entries with a TTL, so "it rained today" never becomes a fact.

The LLM never rewrites the file. It proposes ops (``ADD`` / ``UPDATE`` / ``DELETE`` /
``NOOP``); :func:`apply_ops` validates and applies them, every applied op is appended to
``memory_ops.jsonl``, and every ``REFLECT_EVERY`` saves a :func:`reflect` pass merges
duplicates, expires episodics and promotes repeats — also validated, never blind.

See docs/15-memory-v2.md.
"""
import json
import os
import re
from datetime import date, datetime

_DIR = os.path.dirname(__file__)
MEMORY_PATH = os.path.join(_DIR, "memory.json")
OPS_LOG_PATH = os.path.join(_DIR, "memory_ops.jsonl")
SHORTMEM_PATH = os.path.join(_DIR, "shortmem.txt")  # v1, read once for migration

VERSION = 2
REFLECT_EVERY = int(os.getenv("MEMORY_REFLECT_EVERY", "5"))
BUDGET_TOKENS = int(os.getenv("MEMORY_BUDGET_TOKENS", "700"))
EPISODIC_SHOWN = 8
DEFAULT_TTL_DAYS = 30

SCALAR_SECTIONS = ("identity", "preferences")
LIST_SECTIONS = ("people", "projects", "recurring", "episodic")
LIST_KEY = {"people": "name", "projects": "name", "recurring": "what", "episodic": "text"}
CAPS = {"people": 30, "projects": 20, "recurring": 15, "episodic": 40}

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
                k: v for k, v in value.items()
                if k == "superseded" or isinstance(v, str)
            }
    for section in LIST_SECTIONS:
        value = data.get(section)
        if isinstance(value, list):
            profile[section] = [e for e in value if isinstance(e, dict)]
    meta = data.get("meta")
    if isinstance(meta, dict):
        profile["meta"].update(meta)
    profile["updated"] = data.get("updated")
    return profile


def load_profile() -> dict:
    """The profile from memory.json (schema defaults when missing or unreadable)."""
    if not os.path.exists(MEMORY_PATH):
        return _empty_profile()
    try:
        with open(MEMORY_PATH) as f:
            return _normalise(json.load(f))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[Memory] warning: could not read {os.path.basename(MEMORY_PATH)} ({exc}); starting empty")
        return _empty_profile()


def _write_profile(profile: dict) -> None:
    """Atomic write: full file to `.tmp`, fsync, then os.replace."""
    profile["version"] = VERSION
    profile["updated"] = datetime.now().strftime("%Y-%m-%dT%H:%M")
    tmp = MEMORY_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile, f, indent=1, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, MEMORY_PATH)


def has_content(profile: dict) -> bool:
    for section in SCALAR_SECTIONS:
        if any(k != "superseded" for k in profile.get(section, {})):
            return True
    return any(profile.get(section) for section in LIST_SECTIONS)


# --------------------------------------------------------------------------- render


def _today() -> date:
    return date.today()


def _parse_date(value) -> date | None:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _age_days(entry: dict) -> int:
    when = _parse_date(entry.get("date"))
    return 0 if when is None else max(0, (_today() - when).days)


def is_expired(entry: dict) -> bool:
    ttl = entry.get("ttl_days", DEFAULT_TTL_DAYS)
    try:
        ttl = int(ttl)
    except (TypeError, ValueError):
        ttl = DEFAULT_TTL_DAYS
    return _age_days(entry) > ttl


def _importance(entry: dict) -> int:
    try:
        return max(1, min(3, int(entry.get("importance", 2))))
    except (TypeError, ValueError):
        return 2


def _episodic_score(entry: dict) -> float:
    """recency x importance — a 2%/day decay times importance 1-3."""
    return _importance(entry) * (0.98 ** _age_days(entry))


def live_episodics(profile: dict) -> list:
    """Non-expired episodics, best first (recency x importance)."""
    live = [e for e in profile.get("episodic", []) if not is_expired(e)]
    return sorted(live, key=_episodic_score, reverse=True)


def tokens_est(text: str) -> int:
    return len(text) // 4


def _scalar_lines(profile: dict, section: str) -> list:
    """Only current values. Superseded ones stay in the file (undo, "you used to") but are
    deliberately not rendered: showing them made the model volunteer the stale value."""
    return [
        f"{key}: {value}"
        for key, value in profile.get(section, {}).items()
        if key != "superseded" and isinstance(value, str) and value.strip()
    ]


def _entry_line(section: str, entry: dict) -> str:
    key = entry.get(LIST_KEY[section], "")
    if section == "people":
        extra = " ".join(x for x in (f"({entry['rel']})" if entry.get("rel") else "",
                                     f"- {entry['note']}" if entry.get("note") else "") if x)
    elif section == "projects":
        extra = " ".join(x for x in (f"[{entry['status']}]" if entry.get("status") else "",
                                     f"- {entry['note']}" if entry.get("note") else "") if x)
    else:  # recurring
        extra = f"({entry['when']})" if entry.get("when") else ""
    return f"{key} {extra}".strip()


def _blocks(profile: dict) -> list:
    """(label, items) in fixed order; `items` are trimmed from the tail to fit the budget."""
    blocks = []
    for section in SCALAR_SECTIONS:
        items = _scalar_lines(profile, section)
        if items:
            blocks.append((section.capitalize(), items))
    for section in ("people", "projects", "recurring"):
        items = [line for line in (_entry_line(section, e) for e in profile.get(section, [])) if line]
        if items:
            blocks.append((section.capitalize(), items))
    episodics = live_episodics(profile)[:EPISODIC_SHOWN]
    if episodics:
        blocks.append(("Recent", [f"{e.get('date', '')}: {e.get('text', '')}".strip(": ")
                                  for e in episodics]))
    return blocks


def _text(blocks: list) -> str:
    lines = []
    for label, items in blocks:
        if label == "Recent":
            lines.append("Recent:")
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append(f"{label}: " + "; ".join(items))
    return "\n".join(lines)


def render(profile: dict, budget_tokens: int | None = None) -> str:
    """The profile block that goes into the prompt, trimmed to `budget_tokens`."""
    budget = BUDGET_TOKENS if budget_tokens is None else budget_tokens
    blocks = _blocks(profile)
    while blocks and tokens_est(_text(blocks)) > budget:
        blocks[-1][1].pop()          # drop the least important item of the last section
        if not blocks[-1][1]:
            blocks.pop()
    return _text(blocks)


def load(system_prompt: str) -> str:
    """`system_prompt` with the `<user_profile>` block appended (unchanged if empty)."""
    profile = load_profile()
    if not has_content(profile):
        return system_prompt
    block = render(profile, BUDGET_TOKENS)
    if not block.strip():
        return system_prompt
    return (
        f"{system_prompt}\n\n"
        f"<user_profile>\n{block}\n</user_profile>\n"
        f"{PROFILE_NOTE}"
    )


# --------------------------------------------------------------------------- ops


PROPOSE_SYSTEM = """You maintain a long-term profile of the USER for a conversational assistant.
Given the current profile (JSON) and a new conversation transcript, output the smallest set of
operations that brings the profile up to date.

Reply with a JSON array of operations and nothing else. Each operation is
{"op": "ADD"|"UPDATE"|"DELETE"|"NOOP", "path": "<path>", "value": <value>, "reason": "<short>"}

Paths:
  identity.<key>    a plain string. Use keys like name, location, occupation, birthday.
  preferences.<key> a plain string, e.g. {"op":"ADD","path":"preferences.coffee","value":"black"}
  people[]          value {"name": "...", "rel": "...", "note": "..."}
  projects[]        value {"name": "...", "status": "...", "note": "..."}
  recurring[]       value {"what": "...", "when": "..."}
  episodic[]        value {"date": "YYYY-MM-DD", "text": "...", "ttl_days": N, "importance": 1-3}

Rules:
- Only facts the USER stated about THEMSELVES, their people or their projects. Never facts about
  the assistant, never world facts, never anything the assistant merely said or guessed.
- Durable facts go to identity/preferences/people/projects/recurring. Anything time-bound or
  trivial (today's weather, what they ate, a passing mood) is skipped, or at most an episodic
  with ttl_days 1-7 and importance 1. Never store transients as identity or preferences.
- If a fact already in the profile changed, UPDATE that exact path. Never ADD a second copy.
- If a fact stopped being true and has no replacement, DELETE it.
- For UPDATE or DELETE on a list, the value must carry the identifying key (name for people and
  projects, what for recurring) so the right entry is matched.
- Learned nothing new? Reply with exactly [].
Today is %s."""

REFLECT_SYSTEM = """You tidy a user profile. Reply with the cleaned profile as JSON using exactly
the same schema and keys, and nothing else.

You may ONLY:
- merge near-duplicate entries in people, projects and recurring (keep the richer wording),
- drop episodic entries that are expired (date + ttl_days is in the past) or fully redundant,
- add ONE recurring entry {"what": "...", "when": "..."} when three or more episodic entries
  describe the same repeated activity.

You may NOT change identity or preferences, invent facts, or drop entries that are not duplicates.
Today is %s."""


def _transcript(turns: list) -> str:
    return "\n".join(
        f"{str(t.get('role', 'user')).capitalize()}: {t.get('content', '')}"
        for t in turns
        if isinstance(t, dict) and t.get("content")
    )


def _extract_json(text: str, opener: str = "[", closer: str = "]"):
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


def _complete(client, model: str, system: str, user: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def propose_ops(profile: dict, turns: list, client, model: str) -> list:
    """Ask the LLM for upsert ops against `profile`; [] on anything unparseable."""
    transcript = _transcript(turns)
    if not transcript.strip():
        return []
    system = PROPOSE_SYSTEM % _today().isoformat()
    user = (
        f"Current profile:\n{json.dumps(_public(profile), ensure_ascii=False, indent=1)}\n\n"
        f"New conversation:\n{transcript}"
    )
    try:
        content = _complete(client, model, system, user, 600)
    except Exception as exc:  # noqa: BLE001 — memory must never break a session
        print(f"[Memory] warning: op proposal failed ({exc})")
        return []
    ops = _extract_json(content)
    if not isinstance(ops, list):
        print(f"[Memory] warning: could not parse ops from model output: {content[:200]!r}")
        return []
    return [op for op in ops if isinstance(op, dict)]


def _public(profile: dict) -> dict:
    """The profile without bookkeeping, for prompts."""
    return {k: v for k, v in profile.items() if k not in ("meta", "version", "updated")}


def _norm(text) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower()).strip()


def _find(entries: list, key_field: str, key_value) -> int:
    target = _norm(key_value)
    for i, entry in enumerate(entries):
        if _norm(entry.get(key_field)) == target:
            return i
    return -1


def _episodic_entry(value: dict) -> dict:
    entry = {
        "date": str(value.get("date") or _today().isoformat())[:10],
        "text": str(value["text"]).strip(),
        "ttl_days": DEFAULT_TTL_DAYS,
        "importance": _importance(value),
    }
    try:
        entry["ttl_days"] = max(1, int(value.get("ttl_days", DEFAULT_TTL_DAYS)))
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
    """Trim lists to CAPS, dropping the least important / oldest first."""
    dropped = []
    for section, cap in CAPS.items():
        entries = profile.get(section, [])
        if len(entries) <= cap:
            continue
        if section == "episodic":
            keep = sorted(entries, key=_episodic_score, reverse=True)[:cap]
            profile[section] = [e for e in entries if e in keep]
        else:
            profile[section] = entries[-cap:]  # oldest out
        dropped.append(f"{section}: dropped {len(entries) - cap} over cap {cap}")
    return dropped


def _list_section(path: str) -> str | None:
    """`people[]`, `people`, `people[2]` -> "people"; anything else -> None.

    Models are inconsistent about the `[]` suffix, so a bare list-section name counts too.
    """
    base = re.sub(r"\[\d*\]$", "", path).strip()
    return base if base in LIST_SECTIONS else None


_ENTRY_FIELD = re.compile(r"^(\w+)\[(\d+)\]\.(\w+)$")


def _apply_entry_field_op(profile, section, index, field, kind, value, op) -> bool:
    """`projects[0].note` — a single field of one list entry (models reach for this form)."""
    entries = profile.setdefault(section, [])
    if index >= len(entries):
        _reject(op, f"{section}[{index}] does not exist")
        return False
    entry = entries[index]
    if kind == "DELETE":
        if field not in entry:
            _reject(op, f"{section}[{index}] has no {field!r}")
            return False
        del entry[field]
        return True
    if not isinstance(value, (str, int, float)) or not str(value).strip():
        _reject(op, f"{section}[{index}].{field} needs a scalar value")
        return False
    if _norm(entry.get(field)) == _norm(value):
        return False
    entry[field] = str(value).strip()
    return True


def apply_ops(profile: dict, ops: list) -> tuple:
    """Validate and apply `ops`; returns (profile, applied_ops). Rejects are printed."""
    applied = []
    for op in ops or []:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op", "")).strip().upper()
        path = str(op.get("path", "")).strip()
        value = op.get("value")
        if kind == "NOOP":
            continue
        if kind not in ("ADD", "UPDATE", "DELETE"):
            _reject(op, f"unknown op {kind!r}")
            continue

        field_match = _ENTRY_FIELD.match(path)
        section = _list_section(path)
        if field_match and field_match.group(1) in LIST_SECTIONS:
            section, index, field = field_match.group(1), int(field_match.group(2)), field_match.group(3)
            if not _apply_entry_field_op(profile, section, index, field, kind, value, op):
                continue
        elif section:
            path = f"{section}[]"
            if not _apply_list_op(profile, section, kind, value, op):
                continue
        elif "." in path:
            section, _, key = path.partition(".")
            if section not in SCALAR_SECTIONS:
                _reject(op, f"unknown section {section!r}")
                continue
            if not key:
                _reject(op, "missing key")
                continue
            if not _apply_scalar_op(profile, section, key, kind, value, op):
                continue
        else:
            _reject(op, f"unknown path {path!r}")
            continue
        applied.append({"op": kind, "path": path, "value": value, "reason": op.get("reason", "")})

    for note in _enforce_caps(profile):
        print(f"[Memory] cap enforced — {note}")
    return profile, applied


def _apply_scalar_op(profile, section, key, kind, value, op) -> bool:
    target = profile.setdefault(section, {})
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
    entries = profile.setdefault(section, [])
    key_field = LIST_KEY[section]
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
    clean = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and str(v).strip()}
    if kind == "DELETE":
        if index == -1:
            _reject(op, f"no {section} entry named {value[key_field]!r}")
            return False
        entries.pop(index)
        return True
    if index == -1:
        if kind == "UPDATE":
            entries.append(clean)  # update of something we never stored = add
            return True
        entries.append(clean)
        return True
    if kind == "ADD" and entries[index] == clean:
        return False
    entries[index] = {**entries[index], **clean}
    return True


def _reject(op: dict, reason: str) -> None:
    print(f"[Memory] rejected op {op.get('op')} {op.get('path')!r}: {reason}")


def _log_ops(applied: list) -> None:
    if not applied:
        return
    stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(OPS_LOG_PATH, "a") as f:
        for op in applied:
            f.write(json.dumps({"ts": stamp, **op}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- reflection


def reflect(profile: dict, client, model: str) -> dict:
    """Merge duplicates / expire episodics / promote repeats — validated, never blind."""
    system = REFLECT_SYSTEM % _today().isoformat()
    user = json.dumps(_public(profile), ensure_ascii=False, indent=1)
    try:
        content = _complete(client, model, system, user, 900)
    except Exception as exc:  # noqa: BLE001
        print(f"[Memory] warning: reflection failed ({exc})")
        return profile
    cleaned = _extract_json(content, "{", "}")
    if not isinstance(cleaned, dict):
        print(f"[Memory] warning: could not parse reflection output: {content[:200]!r}")
        return profile
    return _validate_reflection(profile, cleaned)


def _similar(a: str, b: str, ratio: float = 0.85) -> bool:
    import difflib
    a, b = _norm(a), _norm(b)
    return bool(a) and bool(b) and (a == b or difflib.SequenceMatcher(None, a, b).ratio() > ratio)


def _validate_reflection(original: dict, cleaned: dict) -> dict:
    """Accept removals/merges and new `recurring`; identity and preferences are copied verbatim."""
    result = _empty_profile()
    result["identity"] = dict(original.get("identity", {}))
    result["preferences"] = dict(original.get("preferences", {}))
    result["meta"] = dict(original.get("meta", {}))

    for section in ("people", "projects"):
        key_field = LIST_KEY[section]
        originals = original.get(section, [])
        kept = []
        for entry in cleaned.get(section, []) or []:
            if not isinstance(entry, dict) or not entry.get(key_field):
                continue
            if any(_similar(entry[key_field], o.get(key_field)) for o in originals):
                kept.append(entry)
        result[section] = kept

    # recurring: existing entries plus promotions from episodics
    recurring = [e for e in cleaned.get("recurring", []) or []
                 if isinstance(e, dict) and e.get("what")]
    result["recurring"] = recurring[:CAPS["recurring"]]

    originals = original.get("episodic", [])
    kept = []
    for entry in cleaned.get("episodic", []) or []:
        if not isinstance(entry, dict) or not entry.get("text"):
            continue
        if is_expired(entry):
            continue
        if any(_similar(entry["text"], o.get("text")) for o in originals):
            kept.append(entry)
    result["episodic"] = kept
    _enforce_caps(result)
    return result


# --------------------------------------------------------------------------- migration


def migrate_if_needed(client, model: str) -> dict:
    """Seed memory.json from shortmem.txt once. The text file is left untouched."""
    if os.path.exists(MEMORY_PATH) or not os.path.exists(SHORTMEM_PATH):
        return load_profile()
    try:
        with open(SHORTMEM_PATH) as f:
            raw = f.read()
    except OSError:
        return load_profile()
    lines = [
        line.strip() for line in raw.splitlines()
        if line.strip() and not re.fullmatch(r"-{2,}.*-{2,}", line.strip())
    ]
    if not lines:
        return load_profile()

    profile = _empty_profile()
    ops = propose_ops(profile, [{"role": "user", "content": "\n".join(lines)}], client, model)
    profile, applied = apply_ops(profile, ops)
    _write_profile(profile)
    _log_ops(applied)
    print(f"[Memory] migrated {len(applied)} facts from shortmem.txt (file kept)")
    return profile


# --------------------------------------------------------------------------- save


def save(session_turns: list, client, model: str) -> list:
    """Extract ops from this session, apply them, log them, reflect every REFLECT_EVERY saves."""
    profile = migrate_if_needed(client, model)
    ops = propose_ops(profile, session_turns, client, model)
    profile, applied = apply_ops(profile, ops)

    if not applied:
        print("\n[Nothing new to save]")
        return []

    meta = profile.setdefault("meta", {})
    meta["saves"] = int(meta.get("saves", 0)) + 1
    _write_profile(profile)
    _log_ops(applied)

    counts = {"ADD": 0, "UPDATE": 0, "DELETE": 0}
    for op in applied:
        counts[op["op"]] = counts.get(op["op"], 0) + 1
    print(f"\n[Memory] {len(applied)} ops applied "
          f"({counts['ADD']} add, {counts['UPDATE']} update, {counts['DELETE']} delete)")

    if meta["saves"] % REFLECT_EVERY == 0:
        before = sum(len(profile.get(s, [])) for s in LIST_SECTIONS)
        profile = reflect(profile, client, model)
        profile.setdefault("meta", {})["saves"] = meta["saves"]
        _write_profile(profile)
        after = sum(len(profile.get(s, [])) for s in LIST_SECTIONS)
        print(f"[Memory] reflection pass: {before} -> {after} entries")

    return applied
