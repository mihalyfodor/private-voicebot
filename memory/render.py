"""Render the profile for the prompt: dates, decay (stale / expired), episodic scoring and the token budget (docs/18-memory-decay.md; PRD 50 step 8, from memory.py)."""
import os
from datetime import date, datetime
from memory import identity as m_identity
from memory import store
from core import util



BUDGET_TOKENS = int(os.getenv("MEMORY_BUDGET_TOKENS", "700"))     # render() ceiling for the prompt

EPISODIC_SHOWN = 5                                                # episodic lines in the prompt

EPISODIC_MIN_IMPORTANCE = 2  # importance-1 entries stay on disk until TTL, but are not rendered



def _parse_date(value) -> date | None:
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None



def _age_days(entry: dict) -> int:
    when = _parse_date(entry.get("date"))
    return 0 if when is None else max(0, (util.today() - when).days)



def is_expired(entry: dict) -> bool:
    ttl = entry.get("ttl_days", store.DEFAULT_TTL_DAYS)
    try:
        ttl = int(ttl)
    except (TypeError, ValueError):
        ttl = store.DEFAULT_TTL_DAYS
    return _age_days(entry) > ttl



def is_stale(entry: dict) -> bool:
    """Stale = still on disk, no longer in the prompt. Absent flag means live."""
    return bool(entry.get(store.STALE_FIELD))



def _seen_date(entry: dict) -> date:
    """The entry's `seen` day; a missing/garbled one counts as the oldest possible."""
    return _parse_date(entry.get(store.SEEN_FIELD)) or date.min



def _live(entries: list) -> list:
    return [e for e in entries or [] if not is_stale(e)]



def _mark_stale(profile: dict) -> list:
    """Flag entries whose `seen` date is older than their section window; returns what changed.

    Pure Python and deterministic — the model is never asked whether a fact is still current,
    it just stops being rendered. Episodic entries have a TTL instead and are left alone.
    """
    marked = []
    today = util.today()
    for section, window in store.STALE_DAYS.items():
        for entry in profile.get(section, []):
            if is_stale(entry):
                continue
            seen = _parse_date(entry.get(store.SEEN_FIELD))
            if seen is None or (today - seen).days <= window:
                continue
            entry[store.STALE_FIELD] = True
            marked.append(f"{section}: {entry.get(store.LIST_KEY[section], '?')} (seen {entry[store.SEEN_FIELD]})")
    return marked



def _importance(entry: dict) -> int:
    try:
        return max(1, min(3, int(entry.get("importance", 2))))
    except (TypeError, ValueError):
        return 2



def _episodic_score(entry: dict) -> float:
    """recency x importance — a 2%/day decay times importance 1-3."""
    return _importance(entry) * (0.98 ** _age_days(entry))



def live_episodics(profile: dict, min_importance: int = 1) -> list:
    """Non-expired episodics, best first (recency x importance).

    `min_importance` filters out day-to-day noise: importance-1 entries may still sit on disk
    until their TTL expires, they just don't earn a line in the prompt.
    """
    live = [e for e in profile.get("episodic", [])
            if not is_expired(e) and not is_stale(e) and _importance(e) >= min_importance]
    return sorted(live, key=_episodic_score, reverse=True)



def _scalar_lines(profile: dict, section: str) -> list:
    """Only current values. Superseded ones stay in the file (undo, "you used to") but are
    deliberately not rendered: showing them made the model volunteer the stale value.

    `identity.nickname` and `identity.aliases` are folded into the name line as
    `name: Mihaly Fodor (call them: Michael; also: Misi, Mihály)` so the preferred form of
    address — and every other string that means "the user" — is impossible to miss. Empty
    clauses are omitted.
    """
    values = profile.get(section, {})
    lines = []
    is_identity = section == "identity"
    nickname = values.get("nickname") if is_identity else None
    nickname = nickname.strip() if isinstance(nickname, str) else ""
    aliases = m_identity.alias_list(values.get(store.ALIAS_KEY)) if is_identity else []
    name = values.get("name") if is_identity else None
    name = name.strip() if isinstance(name, str) else ""
    for key, value in values.items():
        if key in store.BOOKKEEPING_KEYS:
            continue
        if m_identity._is_alias(section, key):
            if aliases and not name:
                lines.append("also known as: " + ", ".join(aliases))
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        if key == "nickname":
            if not name:
                lines.append(f"call them: {value.strip()}")
            continue
        if key == "name":
            clauses = [c for c in (f"call them: {nickname}" if nickname else "",
                                   "also: " + ", ".join(aliases) if aliases else "") if c]
            suffix = f" ({'; '.join(clauses)})" if clauses else ""
            lines.append(f"name: {name}{suffix}")
            continue
        lines.append(f"{key}: {value.strip()}")
    return lines



def _entry_line(section: str, entry: dict) -> str:
    key = entry.get(store.LIST_KEY[section], "")
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
    for section in store.SCALAR_SECTIONS:
        items = _scalar_lines(profile, section)
        if items:
            blocks.append((section.capitalize(), items))
    for section in ("people", "projects", "recurring"):
        # stale entries stay on disk (and in fact_count) but leave the prompt — that is what
        # "forgetting" means here (docs/18-memory-decay.md).
        items = [line for line in (_entry_line(section, e) for e in _live(profile.get(section, []))) if line]
        if items:
            blocks.append((section.capitalize(), items))
    episodics = live_episodics(profile, EPISODIC_MIN_IMPORTANCE)[:EPISODIC_SHOWN]
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
    """The profile block that goes into the prompt, trimmed to `budget_tokens`.

    Read-only — rendering never changes what is stored. Sections come in a fixed order and are
    trimmed from the tail, so the most identifying facts survive the tightest budget. Superseded
    values and low-importance/expired episodics stay on disk but never reach the prompt.
    """
    budget = BUDGET_TOKENS if budget_tokens is None else budget_tokens
    blocks = _blocks(profile)
    while blocks and util.tokens_est(_text(blocks)) > budget:
        blocks[-1][1].pop()          # drop the least important item of the last section
        if not blocks[-1][1]:
            blocks.pop()
    return _text(blocks)
