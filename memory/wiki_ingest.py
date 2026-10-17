"""Session ingest: ask the model for page ops over a transcript and apply them (PRD 50 step 7, from wiki.py)."""
import os
from memory import identity
from memory import ops as m_ops
from core import trace
from core import llm as core_llm
from core.untrusted import DOCUMENT_RULE
from vault import index as v_index
from vault import pages as v_pages
from vault import store
from core import util

log = trace.logger("Wiki")


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
    rel = store.normalise_path(op.get("path") if kind == "new_page" else op.get("target"))
    if rel is None or not rel.startswith("people/"):
        return None
    checked.append(os.path.basename(rel)[:-3].replace("-", " "))
    for value in checked:
        text = " ".join(str(value or "").split())
        if "/" in text:
            continue
        if text and identity.is_user(text, names):
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
    transcript = m_ops._transcript(turns)
    if not transcript.strip():
        return []
    names = identity.user_names()
    system = INGEST_SYSTEM % (identity.user_note(names), util._today_iso())
    related = v_index.related_context(transcript)
    user = (f"Schema:\n{store._read(store.SCHEMA_FILE) or store.SCHEMA_TEXT}\n\n"
            f"Index:\n{v_index.vault_map(store.WIKI_INDEX_TOKENS) or '(empty wiki)'}\n\n"
            + (f"{related}\n\n" if related else "")
            + f"Transcript:\n{transcript}")
    try:
        raw = core_llm.complete(system, user, 800)
    except Exception as exc:  # noqa: BLE001
        log.info(f"ingest failed: {exc}")
        return []
    ops = m_ops.extract_json(raw)
    if not isinstance(ops, list):
        log.info("could not parse page ops; nothing filed")
        return []
    ops, about_user = reject_user_pages(ops, names)
    applied, rejected = v_pages.apply_page_ops(ops, f"session {util._today_iso()}")
    v_pages.log_rejected(about_user + rejected, log)
    if applied:
        log.info(f"{len(applied)} page op{'s' if len(applied) != 1 else ''} applied")
    return applied
