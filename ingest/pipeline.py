"""Raw ingestion — the office document stack feeds the wiki (docs/20-raw-ingestion.md).

vault/store.py and vault/pages.py own page mechanics (frontmatter, ops, index, commits). This module owns *sources*:
the bytes the user drops (files, pastes, screenshots), emails and news items they ask to file.

One ingest is always the same five steps, under `vault_lock.LOCK`, ending in one commit:

1. hash the bytes; a sha256 already in ``raw/manifest.jsonl`` means "already filed", stop.
2. copy the original, untouched, under ``raw/`` — the provenance the pages point back at.
3. extract text (md/txt as-is, pdf via pypdf, .eml via stdlib email, .msg via extract_msg,
   docx/pptx/xlsx via python-docx/python-pptx/openpyxl, .vtt via a small parser) or, for an
   image, a downscaled base64 data URL for the multimodal model. Text over `RAW_CHUNK_TOKENS`
   is sampled head/middle/tail (docs/31-office-formats.md).
4. one summary completion -> ``{title, summary, entities, topic, unreadable}`` -> ``sources/<slug>.md``.
5. one page-ops completion (PRD 19's op set, targets restricted to pages already in the index
   plus new people/projects pages), every op carrying ``source: raw/…``.

Two kinds take a different step 5: a *transcript* (.vtt, or a Teams .docx export) gets a
meeting completion -> ``meetings/<date>-<slug>`` plus notes on the attendees' pages; a
*calendar* (.ics) skips the model entirely and writes one deterministic stub per event.

Nothing here is ever triggered automatically: the popup, a tool call or the harness asks for it.
"""
import datetime as dt
import os
import re

from core.formats import (Extracted, _downscale, _image_part, docx_text, ics_events,
                          sample_text)
from memory import identity as m_identity
from memory import ops as m_ops
from core import llm as core_llm
from core import trace
from core.untrusted import DOCUMENT_RULE, untrusted
from core import vault_lock
from vault.manifest import _manifest_append, _manifest_write, manifest_entries, manifest_find, sha256
from vault import chunks
from memory import wiki_ingest as v_ingest
from vault import index as v_index
from vault import pages as v_pages
from vault import search as v_search
from vault import store as v_store
from core import util
from core import formats as c_formats
from vault import manifest as c_manifest

log = trace.logger("Sources")

# --- limits ------------------------------------------------------------------------


MAX_FILE_BYTES = 2 * 1024 * 1024        # eml/msg/ics
MAX_TEXT_BYTES = 50 * 1024 * 1024       # pdf/md/txt/docx/pptx/xlsx/vtt/epub and pasted text (sampled, so size is cheap)
MAX_IMAGE_BYTES = 4 * 1024 * 1024       # png/jpg


TEXT_EXTS = {".md": "md", ".markdown": "md", ".txt": "txt"}
DOC_EXTS = {".pdf": "pdf", ".eml": "email", ".msg": "email", ".ics": "calendar"}
OFFICE_EXTS = {".docx": "docx", ".pptx": "pptx", ".xlsx": "xlsx", ".vtt": "transcript", ".epub": "epub"}
#: extension -> the `kind` that lands in the source page's frontmatter
KINDS = {**TEXT_EXTS, **c_formats.IMAGE_EXTS, **DOC_EXTS, **OFFICE_EXTS}
#: the kinds whose bytes are cheap to hold because the model only ever sees a sample
LARGE_TEXT_EXTS = set(TEXT_EXTS) | set(OFFICE_EXTS) | {".pdf"}      # PRD 48: OneNote PDFs carry images

#: page folders an op may create in when the page is not in the index yet
NEW_PAGE_FOLDERS = ("people", "projects")
#: Kinds that are published works: their authors, editors and cited experts are not people the
#: user works with, so an ingest may only open/append a `people/` page for someone already in
#: the profile (PRD 31 addendum, 2026-08-30). A Teams .docx is a `transcript`, not in the set.
PUBLISHED_KINDS = {"epub", "pdf", "docx"}
PUBLISHED_NOTE = ("This document is a published work: its authors, editors and cited experts are "
                  "NOT people pages — only people the user actually works with are. Do not create "
                  "a `people/` page for anyone who merely wrote or is quoted in it.")
#: How many times the summary completion is asked before the raw file name is used as the title.
SUMMARY_ATTEMPTS = 2

#: intake names that say nothing about the content — the model's title wins over them
GENERIC_TITLES = ("note", "untitled", "pasted text", "drop", "document", "new document")




class IngestError(Exception):
    """A refusal the caller turns into an HTTP status (`status`) or an error string."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# --- prompts -----------------------------------------------------------------------

SUMMARY_SYSTEM = """You are filing one document into the USER's personal wiki.

Reply with a JSON object and nothing else:
{"title": "short title", "summary": "3-6 sentences", "entities": [{"name": "Anna", "type": "person"}], "topic": "work/project", "unreadable": false}

- `title` is a short descriptive name (3-8 words) the user would refer to the document by —
  describe what it is, never quote its first line or heading.
- `summary` is what the document says: facts, status, decisions, dates, numbers. No preamble.
- `entities` are the people, projects and organisations the document is about; `type` is one of
  person, project, org, other. At most 8, most important first.
- `topic` is one short lowercase tag for what the document is about, e.g. gaming, work/project,
  finance, health, travel, family, home, admin.
- Set `unreadable` to true (and leave the summary short) when the text is garbled, empty or
  clearly a failed extraction.
- For an image: describe what it shows and transcribe any text visible in it.
- %s Never list the user as an entity of type person.
- """ + DOCUMENT_RULE + """
Today is %s."""

OPS_SYSTEM = """You maintain the USER's markdown wiki. A new document has just been filed as a
source page; decide what the rest of the wiki should learn from it.

Reply with a JSON array of page operations and nothing else. Each operation is one of:
{"op":"new_page","path":"people/anna","type":"person","name":"Anna","summary":"one line"}
{"op":"append_note","path":"projects/atlas","text":"one sentence"}
{"op":"set_field","path":"projects/atlas","key":"status","value":"paused"}
{"op":"add_decision","title":"deploy on Fridays","text":"one or two sentences"}
{"op":"link","path":"people/anna","target":"projects/atlas"}

Rules:
- Only touch pages listed in the index, or create new `people/<slug>` / `projects/<slug>` pages.
- Never touch the source page itself; it is already written.
- A note is one sentence, at most 600 characters, and states a fact from the document.
- Prefer appending to an existing page over creating a near-duplicate.
- %s
- """ + DOCUMENT_RULE + """
- Nothing worth filing? Reply with exactly [].
Today is %s."""

MEETING_SYSTEM = """You are filing a meeting transcript into the USER's personal wiki.

Reply with a JSON object and nothing else:
{"title": "Falcon weekly sync", "date": "YYYY-MM-DD", "attendees": ["Anna Kovacs", "Bob"],
 "project": "Falcon", "decisions": ["one sentence each"], "actions": ["who does what by when"],
 "summary": "3-6 sentences"}

- `title` names the meeting the way the user would; `date` is the meeting date if the transcript
  says it, else today.
- `attendees` are the people who spoke or were named as present, full names as spelled.
- `project` is the one project the meeting was about, or "" when there is none.
- `decisions` are things that were agreed; `actions` are commitments with an owner. Empty lists
  are fine. At most 8 each, one sentence, at most 300 characters.
- %s
Today is %s."""


def extract(raw_rel: str, data: bytes, kind: str = "") -> Extracted:
    """The text (or image part) of a stored raw file — "" means "nothing extractable"."""
    ext = os.path.splitext(raw_rel)[1].lower()
    full = v_store._path(raw_rel)
    kind = kind or KINDS.get(ext, "txt")
    if ext in c_formats.IMAGE_EXTS:
        _downscale(full)
        return Extracted("", _image_part(full), kind)
    if ext == ".ics":
        events = ics_events(data, today=dt.date.fromisoformat(util.today_iso()))
        text = "\n".join(f"{e['date']} {e['time']} {e['title']} — {', '.join(e['attendees'])}"
                         for e in events)
        return Extracted(text, None, kind, util.tokens_est(text), util.tokens_est(text), events)
    if ext == ".docx":
        kind = docx_text(data)[1]
    shown, shown_tokens, total_tokens = sample_text(chunks.raw_text(raw_rel, data))
    return Extracted(shown, None, kind, shown_tokens, total_tokens)


# --- source page -------------------------------------------------------------------


def _free_path(folder: str, slug: str, ext: str) -> str:
    """`folder/slug.ext`, with `-2`, `-3`… when that name is taken by other bytes."""
    rel = f"{folder}/{slug}{ext}"
    n = 2
    while os.path.exists(v_store._path(rel)):
        rel = f"{folder}/{slug}-{n}{ext}"
        n += 1
    return rel


def _entity_line(entity: dict, user_names=()) -> str:
    """One `## Entities` bullet — a wikilink only when the page exists, so lint stays quiet.

    The profile owner is spelled out as "the user" and never linked: they have no people page
    (docs/26-identity-aliases.md) and one must not be invented for them here either.
    """
    name = " ".join(str(entity.get("name") or "").split())
    if not name:
        return ""
    if m_identity.is_user(name, user_names):
        return f"- the user ({name})"
    etype = " ".join(str(entity.get("type") or "").split()).lower()
    folder = {"person": "people", "project": "projects"}.get(etype)
    if folder:
        rel = f"{folder}/{v_store.slugify(name)}.md"
        if v_store.page_exists(rel):
            return f"- [[{v_store.wikilink(rel)}]] — {name} ({etype or 'entity'})"
    return f"- {name} ({etype or 'entity'})"


def _source_body(title: str, summary: str, entities: list, fed: list, user_names=()) -> str:
    lines = [f"# {title}", "", "## Summary", summary.strip() or "(no summary)", "", "## Entities"]
    lines += [line for line in (_entity_line(e, user_names) for e in entities
                                if isinstance(e, dict)) if line]
    lines += ["", "## Fed"]
    lines += [f"- [[{path}]]" for path in fed]
    return "\n".join(lines).rstrip() + "\n"


def _write_source_page(rel: str, *, title: str, kind: str, raw_rel: str, digest: str,
                       summary: str, entities: list, fed: list, source: str,
                       user_names=(), tags=(), extra: dict | None = None) -> None:
    today = util.today_iso()
    fm = {"type": "source", "name": title, "tags": list(tags), "source": source,
          "updated": today, "seen": today,
          "kind": kind, "raw": raw_rel, "sha256": digest, **(extra or {})}
    v_store._write(rel, v_store._page_text(fm, _source_body(title, summary, entities, fed, user_names)))


# --- meetings (docs/31-office-formats.md) --------------------------------------------


def _attendee_link(name: str, user_names=()) -> str:
    """`people/<slug>` for a colleague, the literal "the user" for the profile owner."""
    return "the user" if m_identity.is_user(name, user_names) else f"people/{v_store.slugify(name)}"


def _write_meeting_page(rel: str, *, title: str, date: str, time: str, attendees: list,
                        source_rel: str, raw_rel: str, summary: str, decisions=(), actions=(),
                        user_names=()) -> None:
    """A `meetings/<date>-<slug>` page: meeting frontmatter and the four fixed sections.

    Written here, not through the `new_page` op, because the op cannot carry `date` / `time` /
    `attendees` or the Decisions/Actions sections — the same arrangement the source page has.
    """
    today = util.today_iso()
    links = []
    for name in attendees:
        link = _attendee_link(name, user_names)
        if link not in links:
            links.append(link)
    fm = {"type": "meeting", "name": title, "tags": [], "source": raw_rel,
          "updated": today, "seen": today, "date": date, "time": time,
          "attendees": f"[{', '.join(links)}]"}
    lines = [f"# {title}", "", "## Summary", summary.strip() or "(no summary)",
             f"Source: [[{v_store.wikilink(source_rel)}]]", "", "## Decisions"]
    lines += [f"- {d}" for d in decisions]
    lines += ["", "## Actions"] + [f"- {a}" for a in actions]
    lines += ["", "## Notes", ""]
    v_store._write(rel, v_store._page_text(fm, "\n".join(lines)))


def _degraded(reason: str) -> None:
    """Note a degraded ingest: `ingest_end.degraded`, and why ("summary" beats "ops") for the reply."""
    trace.mark(degraded=True)
    if reason == "summary" or trace.marked("_degraded") is None:
        trace.mark(_degraded=reason)


def meeting_from_transcript(text: str, title: str, user_names=()) -> dict | None:
    """The meeting completion -> ``{title, date, attendees, project, decisions, actions,
    summary}``, tidied; None when the model cannot be reached or parsed."""
    try:
        system = MEETING_SYSTEM % (m_identity.user_note(user_names), util.today_iso())
        raw = core_llm.complete(system, f"Filename/title: {title}\n\nTranscript:\n{text}", 900, purpose="ingest")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"meeting completion failed: {exc}")
        trace.error("ingest", "meeting_from_transcript", exc)
        _degraded("ops")
        return None
    data = m_ops.extract_json(raw, "{", "}")
    if not isinstance(data, dict):
        log.info("could not parse the meeting; filing the transcript only")
        _degraded("ops")
        return None

    def strings(key, limit=8):
        values = data.get(key) if isinstance(data.get(key), list) else []
        return [" ".join(str(v).split())[:300] for v in values if str(v).strip()][:limit]

    date = str(data.get("date") or "").strip()
    if not c_manifest._DROP_DATE.fullmatch(date):
        date = util.today_iso()
    return {
        "title": " ".join(str(data.get("title") or title or "Meeting").split())[:120],
        "date": date,
        "attendees": strings("attendees", 20),
        "project": " ".join(str(data.get("project") or "").split())[:80],
        "decisions": strings("decisions"),
        "actions": strings("actions"),
        "summary": " ".join(str(data.get("summary") or "").split()),
    }


def _attendee_ops(meeting: dict, meeting_rel: str, user_names=()) -> list:
    """`new_page` (when missing) + `append_note` for every attendee and the project page.

    Proposed exactly like the model's ops would be, so `restrict_ops` and the wiki validate
    them the same way — including "the user never gets a people page".
    """
    link = f"[[{v_store.wikilink(meeting_rel)}]]"
    when = f"{meeting['title']}, {meeting['date']}"
    ops = []
    for name in meeting["attendees"]:
        if m_identity.is_user(name, user_names):
            continue
        path = f"people/{v_store.slugify(name)}"
        if not v_store.page_exists(f"{path}.md"):
            ops.append({"op": "new_page", "path": path, "type": "person", "name": name,
                        "summary": f"Met in {when}."})
        ops.append({"op": "append_note", "path": path, "text": f"Attended {link} ({when})."})
    if meeting["project"]:
        path = f"projects/{v_store.slugify(meeting['project'])}"
        what = meeting["decisions"][0] if meeting["decisions"] else meeting["summary"].split(". ")[0]
        if not v_store.page_exists(f"{path}.md"):
            ops.append({"op": "new_page", "path": path, "type": "project",
                        "name": meeting["project"], "summary": f"Discussed in {when}."})
        ops.append({"op": "append_note", "path": path,
                    "text": f"{link} ({meeting['date']}): {what}"[:v_store.MAX_NOTE_CHARS]})
    return ops


def _file_meeting(extracted, *, raw_rel: str, source_rel: str, title: str,
                  names) -> tuple:
    """A transcript's step 5: the meeting page plus the attendee/project ops. ``(meeting, ops)``."""
    meeting = meeting_from_transcript(extracted.text, title, names)
    if meeting is None:
        return "", []
    rel = _free_path("meetings", f"{meeting['date']}-{v_store.slugify(meeting['title'])}", ".md")
    _write_meeting_page(rel, title=meeting["title"], date=meeting["date"], time="",
                        attendees=meeting["attendees"], source_rel=source_rel, raw_rel=raw_rel,
                        summary=meeting["summary"], decisions=meeting["decisions"],
                        actions=meeting["actions"], user_names=names)
    return v_store.wikilink(rel), _attendee_ops(meeting, rel, names)


def _file_calendar(events: list, *, raw_rel: str, source_rel: str, names) -> tuple:
    """A calendar's step 5, no model involved: one stub per event, an `updated invite` note on
    a stub that already exists. ``(stubs written, ops for the existing ones)``."""
    written, ops = [], []
    for event in events:
        rel = f"meetings/{event['date']}-{v_store.slugify(event['title'])}.md"
        if v_store.wikilink(rel) in written:
            continue
        if v_store.page_exists(rel):
            ops.append({"op": "append_note", "path": v_store.wikilink(rel),
                        "text": (f"updated invite: {event['time'] or 'all day'}, "
                                 f"{len(event['attendees'])} attendees ([[{v_store.wikilink(source_rel)}]])")})
            continue
        _write_meeting_page(rel, title=event["title"], date=event["date"], time=event["time"],
                            attendees=event["attendees"], source_rel=source_rel, raw_rel=raw_rel,
                            summary=event["description"], user_names=names)
        written.append(v_store.wikilink(rel))
    return written, ops


def _calendar_summary(events: list, title: str) -> dict:
    """The deterministic stand-in for the summary completion: what the .ics holds."""
    if not events:
        return {"title": title, "summary": "unreadable — no events could be read from this file.",
                "entities": [], "unreadable": True}
    lines = [f"{e['date']} {e['time']} {e['title']}".strip() for e in events]
    return {"title": title, "summary": f"{len(events)} meeting(s): " + "; ".join(lines),
            "entities": [], "unreadable": False}


# --- ops ---------------------------------------------------------------------------


def restrict_ops(ops, raw_rel: str, user_names=(), kind: str = "", people=None) -> tuple:
    """``(kept, rejected)``: only index pages and new people/projects pages may be touched.

    For a published work (`kind` in PUBLISHED_KINDS) a new `people/` page — or a note to a
    people page that does not exist yet — is refused unless the name matches someone in the
    user's profile (`people`, default `memory.people_names()`): a book's author is not a
    colleague. Matching is `memory.is_user` against those names, so a mangled spelling of a
    real colleague still lands on their page.

    The model proposes these, so they are untrusted twice over — once for the path (wiki does
    that) and once for the *target*: an ingest may not invent a `meetings/` or `sources/` page.
    Every kept op is stamped with `source: raw/…`, the PRD's provenance rule.

    Ops that would file the *user* as a person are refused first, by the same wiki helper the
    session ingest uses: a document mentioning their own (possibly misspelled) name must not
    grow a `people/` page about them.
    """
    candidates, rejected = v_ingest.reject_user_pages(ops, user_names)
    if kind in PUBLISHED_KINDS and people is None:
        people = m_identity.people_names()
    kept = []
    for raw in candidates:
        op = raw if isinstance(raw, dict) else {}
        op_kind = str(op.get("op") or "").strip().lower()
        if op_kind == "add_decision":
            kept.append({**op, "source": raw_rel})
            continue
        checked = [op.get(key) for key in ("path", "target") if key in op] or [None]
        reason = next((r for r in (_target_allowed(p) for p in checked) if r), None)
        if not reason and kind in PUBLISHED_KINDS:
            reason = _author_page_reason(op, people or [])
        if reason:
            rejected.append({"op": raw, "reason": reason})
        else:
            kept.append({**op, "source": raw_rel})
    return kept, rejected


def _author_page_reason(op: dict, people: list) -> str | None:
    """Why a published work may not touch this people page: a `new_page people/*`, or an
    `append_note` to a people page not yet in the wiki, about someone not in the profile."""
    op_kind = str(op.get("op") or "").strip().lower()
    if op_kind not in ("new_page", "append_note"):
        return None
    rel = v_store.normalise_path(op.get("path"))
    if rel is None or not rel.startswith("people/") or v_store.page_exists(rel):
        return None
    slug_name = os.path.basename(rel)[:-3].replace("-", " ")
    names = [" ".join(str(v or "").split()) for v in (op.get("name"), slug_name)]
    if any(name and m_identity.is_user(name, people) for name in names):
        return None
    return (f"{names[0] or slug_name} is not in the user's profile; a published work's "
            "authors and cited experts are not people pages")


def _target_allowed(path) -> str | None:
    rel = v_store.normalise_path(path)
    if rel is None:
        return f"bad path {path!r}"
    if v_store.page_exists(rel):
        return None
    if rel.split("/")[0] not in NEW_PAGE_FOLDERS:
        return f"{v_store.wikilink(rel)} is not in the index and not a people/projects page"
    return None


def summarise(kind: str, text: str, image_part, hint: str, title: str,
              user_names=(), strict: bool = False) -> dict:
    """One completion -> ``{title, summary, entities, topic, unreadable}``; never raises
    (`strict`, PRD 48: an empty extraction or a failed completion raises instead of filing a stub).

    An empty extraction skips the call entirely: there is nothing to summarise, and the page is
    still worth creating (it carries the link to the raw file — PRD's "unreadable" risk).
    """
    fallback = {"title": title or "Untitled", "summary": "", "entities": [], "topic": "",
                "unreadable": True}
    if not text.strip() and image_part is None:
        if strict:
            raise IngestError(422, "no text could be extracted")
        fallback["summary"] = "unreadable — no text could be extracted from this file."
        return fallback
    context = [f"Kind: {kind}"]
    if title:
        context.append(f"Filename/title: {title}")
    if hint:
        context.append(f"User hint: {hint}")
    header = "\n".join(context)
    if image_part is not None:
        content = [{"type": "text", "text": f"{header}\n\nSummarise this image."}, image_part]
    else:
        content = f"{header}\n\nDocument:\n{untrusted(text, title or kind)}"
    system = SUMMARY_SYSTEM % (m_identity.user_note(user_names), util.today_iso())
    data = None
    for attempt in range(1, SUMMARY_ATTEMPTS + 1):
        try:
            raw = core_llm.complete(system, content, 700, purpose="ingest")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"summary failed: {exc}")
            trace.error("ingest", "summarise", exc)
            _degraded("summary")
            if strict:
                raise
            fallback["summary"] = "the summary model could not be reached; the raw file is filed."
            return fallback
        parsed = m_ops.extract_json(raw, "{", "}")
        if isinstance(parsed, dict):
            data = parsed
            if str(data.get("title") or "").strip():
                break
        if attempt < SUMMARY_ATTEMPTS:      # a garbled or untitled reply earns one more try
            log.info("summary came back without a title; asking once more")
    if not isinstance(data, dict):
        log.info("could not parse the summary; filing the raw file only")
        _degraded("summary")
        if strict:
            raise IngestError(422, "the summary could not be parsed")
        fallback["summary"] = "the summary could not be parsed; the raw file is filed."
        return fallback
    entities = [e for e in (data.get("entities") or []) if isinstance(e, dict)][:8]
    summary = " ".join(str(data.get("summary") or "").split())
    if strict and (data.get("unreadable") or not summary):
        raise IngestError(422, "the model found nothing readable in the file")
    return {
        "title": " ".join(str(data.get("title") or title or "Untitled").split())[:120],
        "summary": summary,
        "entities": entities,
        "topic": "-".join(str(data.get("topic") or "").lower().split())[:40],
        "unreadable": bool(data.get("unreadable")) or not summary,
    }


def propose_page_ops(source_rel: str, summary: dict, hint: str,
                     hint_page: str = "", user_names=(), kind: str = "") -> list:
    """Ask for page ops about a just-filed source; [] on anything unparseable.

    For a published work (`kind` in PUBLISHED_KINDS) the prompt adds that authors and cited
    experts are not people pages.
    """
    index = v_index.vault_map(v_store.WIKI_INDEX_TOKENS) or "(empty wiki)"
    entities = ", ".join(f"{e.get('name')} ({e.get('type')})" for e in summary.get("entities", []))
    related = v_index.related_context(" ".join(
        [str(summary.get("title") or ""), str(summary.get("summary") or "")]
        + [str(e.get("name") or "") for e in summary.get("entities", [])]))
    parts = [f"Index:\n{index}"] + ([related] if related else [])
    parts += [f"Source page: {v_store.wikilink(source_rel)}",
             f"Title: {summary.get('title')}",
             f"Summary:\n{untrusted(summary.get('summary'), summary.get('title'))}"]
    if entities:
        parts.append(f"Entities: {entities}")
    if hint:
        parts.append(f"User hint: {hint}")
    if hint_page:
        parts.append(f"The user filed this under: {hint_page}")
    try:
        rules = m_identity.user_note(user_names)
        if kind in PUBLISHED_KINDS:
            rules += "\n- " + PUBLISHED_NOTE
        system = OPS_SYSTEM % (rules, util.today_iso())
        raw = core_llm.complete(system, "\n\n".join(parts), 800, purpose="ingest")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"page ops failed: {exc}")
        trace.error("ingest", "propose_page_ops", exc)
        _degraded("ops")
        return []
    ops = m_ops.extract_json(raw)
    if not isinstance(ops, list):
        log.info("could not parse the page ops; filing the source page only")
        _degraded("ops")
        return []
    return ops


# --- the ingest ---------------------------------------------------------------------


def _store(raw_rel: str, data: bytes) -> None:
    full = v_store._path(raw_rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)


def _rename_generic_raw(raw_rel: str, title: str) -> str:
    """`raw/inbox/<date>-note.md` becomes `raw/inbox/<date>-<slug of title>.md` once the model
    has named the document; a real filename (`falcon-plan.docx`) is left alone. Returns the path
    the raw file now has (deduped like any other)."""
    folder, name = os.path.split(raw_rel)
    stem, ext = os.path.splitext(name)
    slug = v_store.slugify(title)
    if not is_generic_title(stem) or is_generic_title(title) or not slug:
        return raw_rel
    date = c_manifest._DROP_DATE.search(stem)
    new_rel = _free_path(folder, f"{date.group(1)}-{slug}" if date else slug, ext)
    os.replace(v_store._path(raw_rel), v_store._path(new_rel))
    return new_rel


def is_generic_title(title: str) -> bool:
    """A name that says nothing about the content (`note`, `untitled`, a stamped `note 2026-…`)."""
    words = " ".join(str(title or "").lower().replace("_", " ").replace("-", " ").split())
    words = c_manifest._DROP_DATE.sub("", words).strip(" -:0123456789")
    return not words or words in GENERIC_TITLES


def _summarise_raw(raw_rel: str, data: bytes, kind: str, hint: str,
                   title: str, names, on_stage=None, strict: bool = False,
                   unreadable: bool = False) -> tuple:
    """Extract the stored original and run the summary completion — steps 3 and 4 of an ingest.

    ``(summary, extracted)``: the kind the extraction settled on (`extracted.kind`) is the one
    that lands on the page. A calendar never reaches the model. `on_stage` (PRD 36) hears
    "extracting" and "summarising" as each step starts. `unreadable` (PRD 47: every notebook page
    came back empty) takes the no-text path: no summary call, no ops.
    """
    _stage(on_stage, "extracting")
    extracted = extract(raw_rel, data, kind)
    if unreadable:
        extracted.text = ""
    stem = os.path.splitext(os.path.basename(raw_rel))[0]
    label = title or ("" if is_generic_title(stem) else stem)
    if extracted.kind == "calendar":
        return _calendar_summary(extracted.events, title or stem), extracted
    _stage(on_stage, "summarising")
    return summarise(extracted.kind, extracted.text, extracted.image_part,
                     hint, label, names, strict), extracted


def _file_page(summary: dict, extracted, *, raw_rel: str, source_rel: str,
               digest: str, page_title: str, hint: str, page: str, source: str, names) -> list:
    """Write the source page and apply the proposed ops — step 5. Returns the pages fed.

    Shared by the first filing and a re-ingest; runs under the lock, commits nothing, so the
    caller owns the manifest line and the single commit.
    """
    kind = extracted.kind
    extra = {}
    if extracted.total_tokens > extracted.shown_tokens:
        extra = {"shown_tokens": extracted.shown_tokens, "total_tokens": extracted.total_tokens}

    def write_page(fed: list, meeting: str = "") -> None:
        """The source page, written once before the ops and once after (now with `## Fed`)."""
        _write_source_page(source_rel, title=page_title, kind=kind, raw_rel=raw_rel,
                           digest=digest, summary=summary.get("summary", ""),
                           entities=summary.get("entities", []), fed=fed,
                           source=source, user_names=names,
                           tags=[summary["topic"]] if summary.get("topic") else [],
                           extra={**extra, **({"meeting": meeting} if meeting else {})})

    write_page([])

    applied, stubs, meeting = [], [], ""
    if kind == "calendar":
        stubs, ops = _file_calendar(extracted.events, raw_rel=raw_rel, source_rel=source_rel,
                                    names=names)
        meeting = stubs[0] if stubs else ""
    elif kind == "transcript" and not summary.get("unreadable"):
        meeting, ops = _file_meeting(extracted, raw_rel=raw_rel,
                                     source_rel=source_rel, title=page_title, names=names)
    elif not summary.get("unreadable"):
        ops = propose_page_ops(source_rel, summary, hint, page, names, kind)
    else:
        ops = []
    trace.verbose("ingest_text", summary=summary.get("summary", ""), ops=ops)
    if ops:
        kept, rejected = restrict_ops(ops, raw_rel, names, kind)
        applied, more = v_pages.apply_page_ops(kept, raw_rel, commit=False)
        trace.mark(ops_rejected=v_pages.log_rejected(rejected + more, log))

    fed = sorted({v_store.wikilink(v_store.normalise_path(op["path"]))
                  for op in applied if op.get("path")}
                 | {"decisions" for op in applied if op["op"] == "add_decision"}
                 | set(stubs) | ({meeting} if meeting else set()))
    write_page(fed, meeting)     # rewritten now that the ops are in, listing what it actually fed
    trace.mark(ops_applied=len(applied), pages=[v_store.wikilink(source_rel)] + fed)
    return fed


def _stage(on_stage, stage: str) -> None:
    """Report an ingest stage to the optional progress callback (PRD 36); never raises."""
    if on_stage is None:
        return
    try:
        on_stage(stage)
    except Exception as exc:  # noqa: BLE001 — a progress push must not fail the ingest
        log.info(f"progress callback failed: {exc}")


def file_bytes(data: bytes, raw_rel: str, kind: str, *, title: str = "", hint: str = "",
               page: str = "", on_stage=None,
               source: str = "", strict: bool = False, digest: str = "", pages=None,
               unreadable: bool = False, extra: dict | None = None) -> dict:
    """The one ingest path: store, extract, summarise, file, commit once.

    Returns ``{path, raw, already}`` — `path` is the source page (`sources/<slug>`), `raw` the
    vault-relative original, `already` True when these exact bytes were filed before.
    `on_stage(stage)` hears "extracting" / "summarising" / "filing" as the steps start (PRD 36);
    an already-filed drop reports nothing. A notebook (PRD 47) passes its own `digest` (the
    pages', not the transcript's), the page PNGs as `pages` — stored next to the raw file under
    `<stem>/page-NN.png` and listed in the manifest — and `extra` manifest fields.
    """
    with vault_lock.LOCK:
        v_store.ensure_vault()
        names = m_identity.user_names()
        digest = digest or sha256(data)
        seen = manifest_find(digest)
        if seen:
            trace.mark(already=True, path=seen.get("page", ""))
            return {"path": seen.get("page", ""), "raw": seen.get("path", ""), "already": True}

        folder, name = os.path.split(raw_rel)
        stem, ext = os.path.splitext(name)
        raw_rel = _free_path(folder, stem, ext)
        _store(raw_rel, data)

        try:
            summary, extracted = _summarise_raw(raw_rel, data, kind, hint, title,
                                                names, on_stage, strict, unreadable)
        except Exception:
            if strict:          # PRD 48: nothing but the stored original exists yet; drop just that file
                os.remove(v_store._path(raw_rel))
            raise
        kind = extracted.kind
        page_title = title or summary.get("title") or stem
        raw_rel = _rename_generic_raw(raw_rel, page_title)
        page_rels = []
        for i, png in enumerate(pages or [], 1):
            page_rels.append(f"{os.path.splitext(raw_rel)[0]}/page-{i:02d}.png")
            _store(page_rels[-1], png)
        source_rel = _free_path("sources", v_store.slugify(page_title), ".md")
        _stage(on_stage, "filing")
        fed = _file_page(summary, extracted, raw_rel=raw_rel,
                         source_rel=source_rel, digest=digest, page_title=page_title,
                         hint=hint, page=page, source=source or f"user drop {util.today_iso()}",
                         names=names)
        _manifest_append({"sha256": digest, "path": raw_rel, "kind": kind,
                          "ingested_at": util.stamp("%H:%M:%S"),
                          "page": v_store.wikilink(source_rel), "pages_touched": fed,
                          **({"pages": page_rels} if page_rels else {}), **(extra or {})})
        chunks.index_source(chunks.raw_chunks(raw_rel))   # PRD 32 chunk cache, built while the bytes are hot; PRD 53 index
        v_index.rebuild_index()
        v_index._log([f"filed [[{v_store.wikilink(source_rel)}]] from {raw_rel}"
                   + (f" -> {', '.join(fed)}" if fed else "")])
        head = v_store.head_sha()
        v_store._commit(f"wiki: filed {raw_rel}")
        trace.mark(path=v_store.wikilink(source_rel), title=page_title)
        if v_store.head_sha() != head:                 # PRD 51: the run page's sha, only for a commit that happened
            trace.mark(commit=v_store.head_sha())
        return {"path": v_store.wikilink(source_rel), "raw": raw_rel, "already": False,
                "degraded": trace.marked("_degraded")}


def list_sources() -> list:
    """Every filed source, newest first (the manifest is append-only, so reversed = newest)."""
    return [c_manifest._source_meta(e) for e in reversed(manifest_entries())]


def _entry_or_raise(digest: str) -> dict:
    entry = manifest_find(digest)
    if entry is None:
        raise IngestError(404, f"no filed source with sha {str(digest)[:12]}")
    return entry


def reingest(digest: str) -> dict:
    """Re-run summary + ops from the stored raw file; same source page, same sha, one commit.

    Touched pages get fresh dated notes (the ops are re-applied, not diffed); the manifest
    line is replaced in place with the union of old and new `pages_touched`.
    """
    with vault_lock.LOCK:
        entry = _entry_or_raise(digest)
        raw_rel, source_rel = entry["path"], f"{entry['page']}.md"
        try:
            with open(v_store._path(raw_rel), "rb") as f:
                data = f.read()
        except OSError as exc:
            raise IngestError(410, f"the raw file {raw_rel} is gone") from exc
        names = m_identity.user_names()
        kind = str(entry.get("kind") or KINDS.get(os.path.splitext(raw_rel)[1].lower()) or "txt")
        old_fm = v_store.parse_page(v_store._read(source_rel))[0]
        summary, extracted = _summarise_raw(raw_rel, data, kind, "", "", names)
        page_title = str(old_fm.get("name") or summary.get("title") or "")
        fed = _file_page(summary, extracted, raw_rel=raw_rel,
                         source_rel=source_rel, digest=digest, page_title=page_title,
                         hint="", page="",
                         source=str(old_fm.get("source") or f"user drop {util.today_iso()}"),
                         names=names)
        fed = sorted(set(entry.get("pages_touched") or []) | set(fed))
        stamp = util.stamp("%H:%M:%S")
        _manifest_write([{**e, "pages_touched": fed, "ingested_at": stamp}
                         if e.get("sha256") == digest else e for e in manifest_entries()])
        v_index.rebuild_index()
        v_index._log([f"re-ingested [[{entry['page']}]] from {raw_rel}"
                   + (f" -> {', '.join(fed)}" if fed else "")])
        v_store._commit(f"wiki: re-ingested {raw_rel}")
        return {"path": entry["page"], "raw": raw_rel}


def _without_source_notes(body: str, raw_rel: str) -> str:
    """Drop the `## Notes` / `## History` bullets stamped `(source: <raw_rel>)`; nothing else."""
    marker = f"(source: {raw_rel})"
    out, section = [], None
    for line in body.splitlines():
        if line.startswith("## "):
            section = line.strip()
        if (section in ("## Notes", "## History") and line.lstrip().startswith("- ")
                and line.rstrip().endswith(marker)):
            continue
        out.append(line)
    return "\n".join(out).rstrip("\n") + "\n"


#: the notebook writer's page file name (`<stem>/page-NN.png`)
_PAGE_IMAGE = re.compile(r"page-\d{2,}\.png")


def _forget_targets(entry: dict) -> dict:
    """What forgetting `entry` touches, read-only: the source page, the raw file, its stored page
    files, the pages whose notes cite it and the source title the memory entries carry."""
    raw_rel = entry["path"]
    cleaned = []
    for fed in entry.get("pages_touched") or []:
        rel = v_store.normalise_path(fed)
        if rel is None or not v_store.page_exists(rel) or rel in cleaned:
            continue
        if f"(source: {raw_rel})" in (v_store._read(rel) or ""):
            cleaned.append(rel)
    files = set(entry.get("pages") or [])
    for folder in {os.path.dirname(rel) for rel in files}:          # a notebook's page folder, as it is now
        if os.path.isdir(v_store._path(folder)):                        # its page images only: a stray file stays
            files |= {f"{folder}/{name}" for name in os.listdir(v_store._path(folder)) if _PAGE_IMAGE.fullmatch(name)}
    return {"page": entry["page"], "raw": raw_rel, "files": sorted(files),
            "cleaned": cleaned, "title": c_manifest._source_meta(entry)["title"]}


#: `forget(expected=…)` found the plan changed since the user approved it (docs/52-approvals.md)
CHANGED_STATUS = 409


def _plan(digest: str, targets: dict) -> dict:
    """The proposal dict `forget_plan` returns (and `forget` compares with the approved one)."""
    return {"sha": digest, "title": targets["title"], "page": targets["page"], "raw": targets["raw"],
            "images": targets["files"], "cleaned": [v_store.wikilink(rel) for rel in targets["cleaned"]],
            "memory": m_ops.source_entries(targets["title"])}


def forget_plan(digest: str) -> dict:
    """The dry run of `forget` (docs/52-approvals.md): `{sha, title, page, raw, images: [a notebook's
    page files], cleaned: [page], memory: [entry text]}` — nothing is written. `IngestError` for an
    unknown sha."""
    with vault_lock.LOCK:
        return _plan(digest, _forget_targets(_entry_or_raise(digest)))


def forget(digest: str, expected: dict | None = None) -> dict:
    """Delete the source page and raw file, drop the manifest line, the notes it fed and the
    memory entries learned from it (docs/33-untrusted-text.md); one commit. What it removes is
    `forget_plan`'s list (the same `_forget_targets`); with `expected` (an approved plan) a plan
    that differs under the lock is refused with `CHANGED_STATUS`, nothing written."""
    with vault_lock.LOCK:
        targets = _forget_targets(_entry_or_raise(digest))
        if expected is not None and _plan(digest, targets) != expected:
            raise IngestError(CHANGED_STATUS, "the source changed since the proposal")
        raw_rel, page = targets["raw"], targets["page"]
        for rel in targets["cleaned"]:
            v_store._edit_page(rel, lambda body: _without_source_notes(body, raw_rel))
        cleaned = [v_store.wikilink(rel) for rel in targets["cleaned"]]
        for rel in (f"{page}.md", raw_rel, *targets["files"]):
            v_store._git("rm", "-q", "-f", "--", rel, quiet=True)
            if os.path.exists(v_store._path(rel)):
                os.remove(v_store._path(rel))
            v_search.unindex_page(rel)
        for folder in {os.path.dirname(rel) for rel in targets["files"]}:
            if os.path.isdir(v_store._path(folder)) and not os.listdir(v_store._path(folder)):
                os.rmdir(v_store._path(folder))
        _manifest_write([e for e in manifest_entries() if e.get("sha256") != digest])
        chunks.unindex_source(digest)
        chunks.drop_raw_chunks(digest)
        v_index.rebuild_index()
        v_index._log([f"forgot [[{page}]] and {raw_rel}"
                   + (f" (notes removed from {', '.join(cleaned)})" if cleaned else "")])
        v_store._commit(f"wiki: forgot {raw_rel}")
        forgotten = m_ops.forget_source(targets["title"])
        return {"path": page, "raw": raw_rel, "cleaned": cleaned, "memory": forgotten}


def _dated(slug: str) -> str:
    """`<today>-<slug>`; a slug that already starts with a date loses it (no double date)."""
    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", slug)
    return f"{util.today_iso()}-{v_store.slugify(slug)}"


def check_upload(filename: str, size: int, kind_hint: str = "file") -> tuple:
    """``(ext, kind)`` for an upload, or `IngestError` — the one type/size gate.

    The endpoint calls it on arrival (so an oversized drop is refused before any work happens)
    and `ingest_upload` calls it again for the callers that never went through HTTP.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    if kind_hint == "paste_image" and ext not in c_formats.IMAGE_EXTS:
        ext = ".png"
    kind = KINDS.get(ext)
    if kind is None:
        raise IngestError(415, f"unsupported file type {ext or filename!r}")
    limit = (MAX_IMAGE_BYTES if ext in c_formats.IMAGE_EXTS
             else MAX_TEXT_BYTES if ext in LARGE_TEXT_EXTS else MAX_FILE_BYTES)
    if size > limit:
        raise IngestError(413, f"{filename} is larger than {limit // (1024 * 1024)} MB")
    return ext, kind


def ingest_upload(data: bytes, filename: str, *, kind_hint: str = "file", title: str = "",
                  hint: str = "", page: str = "",
                  on_stage=None, source: str = "", strict: bool = False) -> dict:
    """One uploaded file (or pasted image). Raises `IngestError` for size/type refusals."""
    ext, kind = check_upload(filename, len(data), kind_hint)
    stem = title or os.path.splitext(os.path.basename(filename or ""))[0] or "drop"
    if is_generic_title(title):
        title = ""          # `note.txt`, `untitled`…: the summary's title names the page
    folder = "raw/screens" if ext in c_formats.IMAGE_EXTS else "raw/inbox"
    raw_rel = f"{folder}/{_dated(stem)}{ext}"
    with trace.ingest(kind, raw=raw_rel, bytes=len(data)):
        return file_bytes(data, raw_rel, kind, title=title, hint=hint, page=page,
                          on_stage=on_stage, source=source, strict=strict)


def check_note(text: str) -> str:
    """The stripped note body, or `IngestError` — the text counterpart of `check_upload`."""
    body = (text or "").strip()
    if not body:
        raise IngestError(400, "nothing to file: the text is empty")
    if len(body.encode()) > MAX_TEXT_BYTES:      # pasted text is sampled like any text kind (PRD 31)
        raise IngestError(413, f"the text is larger than {MAX_TEXT_BYTES // (1024 * 1024)} MB")
    return body


def ingest_note(text: str, *, title: str = "", hint: str = "", page: str = "", on_stage=None) -> dict:
    """Pasted or spoken free text -> `raw/inbox/<date>-<slug>.md`."""
    body = check_note(text)
    stem = title or "note"
    if is_generic_title(title):
        title = ""
    raw_rel = f"raw/inbox/{_dated(stem)}.md"
    with trace.ingest("note", raw=raw_rel, bytes=len(body.encode())):
        return file_bytes(body.encode(), raw_rel, "note", title=title, hint=hint, page=page,
                          on_stage=on_stage)


def ingest_path(path: str, *, title: str = "", hint: str = "", page: str = "") -> dict:
    """A file already on disk (the eval harness's `ingest:` turn, an AppleScript drop)."""
    with trace.ingest("path", raw=os.path.basename(path)):
        try:
            with open(os.path.expanduser(path), "rb") as f:
                data = f.read()
        except OSError as exc:
            raise IngestError(400, f"cannot read {path}: {exc}") from exc
        return ingest_upload(data, os.path.basename(path), title=title, hint=hint, page=page)


def ingest_email(message_id: str, markdown: str, *, hint: str = "", page: str = "") -> dict:
    """A fetched email -> `raw/email/<id>.md` (headers block + body, already rendered)."""
    ident = v_store.slugify(message_id) or _dated("email")
    subject = next((line[len("Subject: "):] for line in markdown.splitlines()
                    if line.startswith("Subject: ")), "").strip()
    with trace.ingest("email", raw=f"raw/email/{ident}.md", bytes=len(markdown.encode())):
        return file_bytes(markdown.encode(), f"raw/email/{ident}.md", "email",
                          title=subject, hint=hint, page=page)


def ingest_news(title: str, markdown: str, *, hint: str = "", page: str = "") -> dict:
    """A flagged news article -> `raw/news/<slug>.md`."""
    with trace.ingest("news", raw=f"raw/news/{v_store.slugify(title)}.md", bytes=len(markdown.encode())):
        return file_bytes(markdown.encode(), f"raw/news/{v_store.slugify(title)}.md", "news",
                          title=title, hint=hint, page=page)

