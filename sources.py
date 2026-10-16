"""Raw ingestion — the office document stack feeds the wiki (docs/20-raw-ingestion.md).

wiki.py owns page mechanics (frontmatter, ops, index, commits). This module owns *sources*:
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
import base64
import binascii
import dataclasses
import datetime as dt
import email
import email.policy
import hashlib
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
import io
import json
import os
import re
import subprocess

from core import settings
import memory
from core import trace
from core.untrusted import DOCUMENT_RULE, untrusted
from core import vault_lock
import wiki

log = trace.logger("Sources")

# --- limits ------------------------------------------------------------------------

#: how much extracted text the model is shown (the rest is sampled out, no chunk-and-merge)
RAW_CHUNK_TOKENS = int(os.getenv("RAW_CHUNK_TOKENS", "6000"))
#: the head / middle / tail shares of a sampled document
SAMPLE_SHARES = (0.60, 0.15, 0.25)
SKIPPED = "[… %d tokens skipped …]"

MAX_FILE_BYTES = 2 * 1024 * 1024        # eml/msg/ics
MAX_TEXT_BYTES = 50 * 1024 * 1024       # pdf/md/txt/docx/pptx/xlsx/vtt/epub and pasted text (sampled, so size is cheap)
MAX_IMAGE_BYTES = 4 * 1024 * 1024       # png/jpg
IMAGE_LONG_EDGE = 1568                  # what the multimodal model wants to see
#: below this, downscaling costs more than it saves (and keeps the tests off `sips`)
DOWNSCALE_ABOVE_BYTES = 256 * 1024

#: a PDF that yields less than this much text has no text layer we can use
MIN_PDF_CHARS = 20

MANIFEST = "raw/manifest.jsonl"

TEXT_EXTS = {".md": "md", ".markdown": "md", ".txt": "txt"}
IMAGE_EXTS = {".png": "screenshot", ".jpg": "screenshot", ".jpeg": "screenshot"}
DOC_EXTS = {".pdf": "pdf", ".eml": "email", ".msg": "email", ".ics": "calendar"}
OFFICE_EXTS = {".docx": "docx", ".pptx": "pptx", ".xlsx": "xlsx", ".vtt": "transcript", ".epub": "epub"}
#: extension -> the `kind` that lands in the source page's frontmatter
KINDS = {**TEXT_EXTS, **IMAGE_EXTS, **DOC_EXTS, **OFFICE_EXTS}
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

#: xlsx cut: rows per sheet, sheets per book, characters per cell (the cut is stated in the text)
XLSX_MAX_ROWS = 200
XLSX_MAX_SHEETS = 20
XLSX_MAX_CELL = 80

#: a Teams .docx transcript: at least this many `Speaker   0:00` paragraphs
TEAMS_MIN_TURNS = 3
DEFAULT_HORIZON_DAYS = 14


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


# --- llm ---------------------------------------------------------------------------


def _client_model() -> tuple:
    """The live oMLX client and model. Split out so tests can hand in a fake."""
    import llm

    return llm.client(), llm.OMLX_MODEL


def _ask(client, model: str, system: str, content, max_tokens: int) -> str:
    """One completion. `content` is a string, or the multimodal content-part list for an image."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    with trace.llm_span("ingest", messages, model=model, max_tokens=max_tokens, ttft_ms=None) as span:
        response = client.chat.completions.create(model=model, messages=messages, temperature=0,
                                                  max_tokens=max_tokens)
        span.update(trace.usage_fields(getattr(response, "usage", None)))
    return response.choices[0].message.content or ""


# --- extraction --------------------------------------------------------------------


def sample_text(text: str) -> tuple:
    """``(shown, shown_tokens, total_tokens)`` — `text` sampled to `RAW_CHUNK_TOKENS`.

    A document over the budget is shown as its head, a slice of its middle and its tail
    (`SAMPLE_SHARES`), each gap marked `[… N tokens skipped …]`: tagging and summarising only
    need the shape of the document, and one call is all it gets (no chunk-and-merge).
    """
    text = text or ""
    total = memory.tokens_est(text)
    if total <= RAW_CHUNK_TOKENS:
        return text, total, total
    budget = RAW_CHUNK_TOKENS * 4
    head_n, mid_n, tail_n = (int(budget * share) for share in SAMPLE_SHARES)
    mid_at = (len(text) - mid_n) // 2
    head, mid, tail = text[:head_n], text[mid_at:mid_at + mid_n], text[len(text) - tail_n:]
    gap1 = memory.tokens_est(text[head_n:mid_at])
    gap2 = memory.tokens_est(text[mid_at + mid_n:len(text) - tail_n])
    shown = "\n\n".join([head.rstrip(), SKIPPED % gap1, mid.strip(), SKIPPED % gap2, tail.lstrip()])
    return shown, memory.tokens_est(head + mid + tail), total


def cap_text(text: str) -> str:
    """`text` as the model will see it — the sample alone, without the token counts."""
    return sample_text(text)[0]


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)


def _strip_html(html: str) -> str:
    return " ".join(_TAGS.sub(" ", html or "").split())


def pdf_text(data: bytes) -> str:
    """Every page's text, joined. An unreadable PDF (no text layer) yields ""."""
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception:  # noqa: BLE001 — a broken PDF is "unreadable", not a crash
        log.warning("pdf extraction failed")
        return ""
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    return text if len(text.strip()) >= MIN_PDF_CHARS else ""


def eml_text(data: bytes) -> str:
    """`.eml` -> a headers block plus the text/plain part (falling back to stripped html)."""
    try:
        msg = email.message_from_bytes(data, policy=email.policy.default)
    except Exception:  # noqa: BLE001
        log.warning("eml parse failed")
        return ""
    plain, html = "", ""
    for part in (msg.walk() if msg.is_multipart() else [msg]):
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            body = part.get_content()
        except Exception:  # noqa: BLE001
            log.warning("eml part skipped")
            continue
        if ctype == "text/plain" and not plain:
            plain = str(body)
        elif ctype == "text/html" and not html:
            html = str(body)
    return _headers_block(
        {k: msg.get(k, "") for k in ("From", "To", "Date", "Subject")},
        plain.strip() or _strip_html(html),
    )


def msg_text(path: str) -> str:
    """Outlook `.msg` (a drag-out from Outlook for Mac) via extract_msg, same shape as `.eml`."""
    try:
        import extract_msg

        with extract_msg.openMsg(path) as msg:
            headers = {"From": msg.sender or "", "To": msg.to or "",
                       "Date": str(msg.date or ""), "Subject": msg.subject or ""}
            body = msg.body or ""
    except Exception:  # noqa: BLE001
        log.warning("msg extraction failed")
        return ""
    return _headers_block(headers, str(body).strip())


def _headers_block(headers: dict, body: str) -> str:
    lines = [f"{key}: {value}" for key, value in headers.items() if str(value).strip()]
    return "\n".join(lines) + "\n\n" + (body or "")


def _downscale(full: str) -> None:
    """Shrink an image in place to `IMAGE_LONG_EDGE` with Pillow (macOS `sips` when Pillow is
    missing); failures are ignored. Small files are left alone so a test never pays for this.
    """
    if os.path.getsize(full) <= DOWNSCALE_ABOVE_BYTES:
        return
    try:
        try:
            from PIL import Image
        except ImportError:
            subprocess.run(["sips", "-Z", str(IMAGE_LONG_EDGE), full], capture_output=True, timeout=20)
            return
        with Image.open(full) as im:
            if max(im.size) > IMAGE_LONG_EDGE:
                im.thumbnail((IMAGE_LONG_EDGE, IMAGE_LONG_EDGE))
                im.save(full)
    except Exception:
        log.warning("image downscale skipped")


def _image_part(full: str) -> dict:
    mime = "image/jpeg" if full.lower().endswith((".jpg", ".jpeg")) else "image/png"
    with open(full, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


# --- office formats (docs/31-office-formats.md) --------------------------------------


def _cell(value) -> str:
    text = " ".join(str(value if value is not None else "").split()).replace("|", "\\|")
    return text[:XLSX_MAX_CELL - 1] + "…" if len(text) > XLSX_MAX_CELL else text


def _table_lines(rows: list) -> list:
    """Markdown table lines from rows of cell strings; the first row is the header."""
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return out


_HEADING = re.compile(r"heading\s*(\d)", re.I)
_TEAMS_TURN = re.compile(r"^(.+?)\s{2,}(\d{1,2}:\d{2}(?::\d{2})?)\s*$")


def docx_paragraphs(data: bytes) -> list:
    """The body of a `.docx` in order: ``("p", text, level)``, ``("table", rows)`` and — a Teams
    export's `Speaker   0:12` lines — ``("turn", speaker)`` items."""
    import docx

    doc = docx.Document(io.BytesIO(data))
    items = []
    for block in doc.iter_inner_content():
        if hasattr(block, "rows"):
            items.append(("table", [[_cell(c.text) for c in row.cells] for row in block.rows]))
            continue
        turn = _TEAMS_TURN.match(block.text.strip())
        if turn:                                   # `Speaker   0:12` — before the spaces collapse
            items.append(("turn", turn.group(1).strip()))
            continue
        text = " ".join(block.text.split())
        if not text:
            continue
        match = _HEADING.search(getattr(block.style, "name", "") or "")
        level = int(match.group(1)) if match else 0
        if not level and (block.style.name or "").lower() == "title":
            level = 1
        items.append(("p", text, min(level, 6)))
    return items


def teams_turns(items: list) -> list:
    """``[(speaker, text)]`` when the paragraphs are a Teams transcript export, else [].

    Teams writes one `Speaker   0:12` paragraph per turn followed by what was said; anything
    else (a normal document) does not have `TEAMS_MIN_TURNS` such lines and takes the generic path.
    """
    turns, speaker = [], None
    for item in items:
        if item[0] == "turn":
            speaker = item[1]
        elif item[0] == "p" and speaker:
            turns.append((speaker, item[1]))
    return turns if len(turns) >= TEAMS_MIN_TURNS else []


def merge_turns(turns: list) -> list:
    """Consecutive turns of one speaker become one."""
    merged = []
    for speaker, text in turns:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, f"{merged[-1][1]} {text}")
        else:
            merged.append((speaker, text))
    return merged


def transcript_text(turns: list, duration: str = "") -> str:
    lines = [f"duration: {duration}"] if duration else []
    lines += [f"**{speaker}:** {text}" for speaker, text in merge_turns(turns)]
    return "\n\n".join(lines)


def docx_text(data: bytes) -> tuple:
    """``(markdown, kind)`` — `kind` is "transcript" for a Teams export, else "docx"."""
    try:
        items = docx_paragraphs(data)
    except Exception:  # noqa: BLE001 — a broken file is "unreadable", not a crash
        log.warning("docx extraction failed")
        return "", "docx"
    turns = teams_turns(items)
    if turns:
        return transcript_text(turns), "transcript"
    lines = []
    for item in items:
        if item[0] == "table":
            lines += _table_lines(item[1]) + [""]
        elif item[0] == "turn":
            lines += [item[1], ""]
        else:
            _, text, level = item
            lines += [("#" * level + " " + text) if level else text, ""]
    return "\n".join(lines).strip(), "docx"


def pptx_text(data: bytes) -> str:
    """One `## Slide N — title` section per slide, its text then its notes."""
    try:
        import pptx

        deck = pptx.Presentation(io.BytesIO(data))
        sections = []
        for n, slide in enumerate(deck.slides, start=1):
            title = " ".join(slide.shapes.title.text.split()) if slide.shapes.title is not None else ""
            body = []
            for shape in slide.shapes:
                if shape == slide.shapes.title or not shape.has_text_frame:
                    continue
                body += [" ".join(p.text.split()) for p in shape.text_frame.paragraphs if p.text.strip()]
            lines = [f"## Slide {n}" + (f" — {title}" if title else "")] + body
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                notes = " ".join(slide.notes_slide.notes_text_frame.text.split())
                if notes:
                    lines.append(f"Notes: {notes}")
            sections.append("\n".join(lines))
    except Exception:  # noqa: BLE001
        log.warning("pptx extraction failed")
        return ""
    return "\n\n".join(sections)


def xlsx_text(data: bytes) -> str:
    """Per sheet `## <name>` and a markdown table of the first `XLSX_MAX_ROWS` rows (cached values)."""
    try:
        import openpyxl

        book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sections = []
        for sheet in book.worksheets[:XLSX_MAX_SHEETS]:
            rows, total = [], 0
            for row in sheet.iter_rows(values_only=True):
                total += 1
                if total <= XLSX_MAX_ROWS:
                    rows.append([_cell(v) for v in row])
            lines = [f"## {sheet.title}"] + _table_lines(rows)
            if total > XLSX_MAX_ROWS:
                lines.append(f"({XLSX_MAX_ROWS} of {total} rows shown)")
            sections.append("\n".join(lines))
        if len(book.worksheets) > XLSX_MAX_SHEETS:
            sections.append(f"({XLSX_MAX_SHEETS} of {len(book.worksheets)} sheets shown)")
        book.close()
    except Exception:  # noqa: BLE001
        log.warning("xlsx extraction failed")
        return ""
    return "\n\n".join(sections)


_VTT_TIMING = re.compile(r"^(\d{1,2}:)?\d{2}:\d{2}[.,]\d{3}\s+-->\s+((\d{1,2}:)?\d{2}:\d{2})[.,]\d{3}")
_VTT_VOICE = re.compile(r"<v\s+([^>]+)>(.*?)(?:</v>|$)", re.S)
_VTT_TAG = re.compile(r"<[^>]+>")
_VTT_NAMED = re.compile(r"^([A-Za-z][\w .'-]{0,40}?):\s+(.*)$", re.S)


def vtt_text(data: bytes) -> str:
    """WebVTT -> `duration:` line plus `**Speaker:** text` turns, timestamps dropped.

    Speakers come from `<v Name>` voice spans (what Teams writes) or a leading `Name:`; cues
    without either are one "Speaker".
    """
    text = _decode(data)
    if not text.lstrip().startswith("WEBVTT"):
        return ""
    turns, end = [], ""
    for block in re.split(r"\n\s*\n", text.replace("\r", ""))[1:]:
        lines = [line.strip() for line in block.splitlines()]
        at = next((i for i, line in enumerate(lines) if _VTT_TIMING.match(line)), None)
        if at is None:
            continue
        end = _VTT_TIMING.match(lines[at]).group(2)
        body = " ".join(" ".join(lines[at + 1:]).split())
        if not body:
            continue
        voice, named = _VTT_VOICE.search(body), _VTT_NAMED.match(body)
        if voice:
            turns.append((voice.group(1).strip(), _VTT_TAG.sub("", voice.group(2)).strip()))
        elif named:
            turns.append((named.group(1).strip(), _VTT_TAG.sub("", named.group(2)).strip()))
        else:
            turns.append(("Speaker", _VTT_TAG.sub("", body).strip()))
    return transcript_text(turns, end) if turns else ""


def _ics_person(value) -> str:
    """An ATTENDEE/ORGANIZER -> its CN, else the mailbox."""
    if value is None:
        return ""
    name = str(getattr(value, "params", {}).get("CN") or "").strip()
    return name or str(value).replace("mailto:", "").strip()


def _ics_field(component, key: str) -> str:
    return " ".join(str(component.get(key) or "").split())


def _ics_people(component) -> tuple:
    """``(organizer, attendees)`` display names, the organizer first among the attendees."""
    organizer = _ics_person(component.get("ORGANIZER"))
    raw = component.get("ATTENDEE") or []
    named = [_ics_person(a) for a in (raw if isinstance(raw, list) else [raw])]
    return organizer, [organizer] * bool(organizer) + [a for a in named if a and a != organizer]


def _ics_time(start: dt.datetime, length: dt.timedelta | None) -> str:
    """`09:00–09:45` (or just `09:00` without an end) for a timed occurrence."""
    return start.strftime("%H:%M") + (f"–{(start + length).strftime('%H:%M')}" if length else "")


def _as_datetime(value) -> dt.datetime | None:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time.min)
    return None


def calendar_horizon_days() -> int:
    """`calendar_horizon_days` from settings.json (default `DEFAULT_HORIZON_DAYS`)."""
    value = settings.load_settings().get("calendar_horizon_days")
    return int(value) if isinstance(value, int) and value > 0 else DEFAULT_HORIZON_DAYS


def _occurrences(component, today: dt.date, horizon: int) -> list:
    """Every start datetime of a VEVENT: the one it names, or — recurring — those from `today`
    to `today + horizon` days."""
    start = _as_datetime(component.decoded("DTSTART", None))
    if start is None:
        return []
    rule = component.get("RRULE")
    if rule is None:
        return [start]
    from dateutil import rrule as du

    naive = start.replace(tzinfo=None)
    lo, hi = dt.datetime.combine(today, dt.time.min), dt.datetime.combine(today + dt.timedelta(days=horizon), dt.time.max)
    try:
        rules = du.rrulestr(rule.to_ical().decode(), dtstart=naive, forceset=True)
    except Exception:  # noqa: BLE001
        log.warning("ics recurrence rule skipped")
        return [start]
    exdates = component.get("EXDATE")
    exdates = exdates if isinstance(exdates, list) else [exdates] if exdates else []
    skipped = {_as_datetime(x.dt).replace(tzinfo=None)
               for ex in exdates for x in getattr(ex, "dts", [])}
    return [o.replace(tzinfo=start.tzinfo) for o in rules.between(lo, hi, inc=True) if o not in skipped]


def ics_events(data: bytes, today: dt.date | None = None, horizon: int | None = None) -> list:
    """``[{title, date, time, attendees, organizer, description, uid}]`` per occurrence, sorted.

    `attendees` are display names (CN or mailbox), the organizer first; nothing about the
    wiki here — `_write_meeting_page` decides what becomes a link.
    """
    today = today or dt.date.fromisoformat(memory._today_iso())
    horizon = calendar_horizon_days() if horizon is None else horizon
    try:
        import icalendar

        cal = icalendar.Calendar.from_ical(data)
    except Exception:  # noqa: BLE001
        log.warning("ics parse failed")
        return []
    events = []
    for component in cal.walk("VEVENT"):
        organizer, attendees = _ics_people(component)
        first, end = component.decoded("DTSTART", None), _as_datetime(component.decoded("DTEND", None))
        length = end - _as_datetime(first) if first is not None and end else None
        for start in _occurrences(component, today, horizon):
            events.append({
                "title": _ics_field(component, "SUMMARY") or "Untitled event",
                "date": start.date().isoformat(),
                "time": _ics_time(start, length) if isinstance(first, dt.datetime) else "",
                "attendees": attendees, "organizer": organizer,
                "description": _ics_field(component, "DESCRIPTION"),
                "uid": _ics_field(component, "UID"),
            })
    return sorted(events, key=lambda e: (e["date"], e["time"], e["title"]))


@dataclasses.dataclass
class Extracted:
    """What `extract` found: the sampled text *or* an image part, plus the kind it decided on
    (a Teams `.docx` becomes a transcript) and the token counts the source page records."""

    text: str = ""
    image_part: dict | None = None
    kind: str = ""
    shown_tokens: int = 0
    total_tokens: int = 0
    events: list = dataclasses.field(default_factory=list)   # a calendar's occurrences


class _HtmlText(HTMLParser):
    """XHTML → markdown-ish text: headings become `#` lines, block elements break lines, the
    rest is inline text. Scripts/styles are dropped."""
    _BLOCKS = {"p", "div", "li", "br", "tr", "section", "article", "blockquote", "pre", "table"}
    _HEAD = {"h1": "#", "h2": "##", "h3": "###", "h4": "####"}

    def __init__(self):
        super().__init__()
        self.parts, self._skip, self._head = [], 0, None

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self._HEAD:
            self._head = self._HEAD[tag]
            self.parts.append(f"\n\n{self._head} ")
        elif tag in self._BLOCKS:
            self.parts.append("\n" if tag == "br" else "\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in self._HEAD:
            self._head = None
            self.parts.append("\n\n")
        elif tag in self._BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", "".join(self.parts))).strip()


def epub_text(data: bytes) -> str:
    """An EPUB (zip of XHTML) as text in spine order; chapters keep their headings.

    Reads `META-INF/container.xml` → the OPF → its spine, then each XHTML item. A broken or
    non-EPUB zip yields "" (unreadable), never an exception.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            container = ET.fromstring(zf.read("META-INF/container.xml"))
            ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
            opf_path = container.find(".//c:rootfile", ns).attrib["full-path"]
            opf = ET.fromstring(zf.read(opf_path))
            ons = {"o": "http://www.idpf.org/2007/opf"}
            base = posixpath.dirname(opf_path)
            items = {i.attrib["id"]: i.attrib["href"] for i in opf.findall(".//o:manifest/o:item", ons)}
            spine = [items[r.attrib["idref"]] for r in opf.findall(".//o:spine/o:itemref", ons)
                     if r.attrib.get("idref") in items]
            chapters = []
            for href in spine:
                name = posixpath.normpath(posixpath.join(base, href)) if base else href
                try:
                    raw = zf.read(name).decode("utf-8", "replace")
                except KeyError:
                    continue
                parser = _HtmlText()
                parser.feed(raw)
                text = parser.text()
                if text:
                    chapters.append(text)
            return "\n\n".join(chapters)
    except Exception:  # noqa: BLE001 — a broken file is "unreadable", not a crash
        log.warning("epub extraction failed")
        return ""


def raw_text(raw_rel: str, data: bytes) -> str:
    """The whole extracted text of a stored text-kind raw file — nothing sampled out.

    What the chunk index (PRD 32) reads; :func:`extract` samples it for the model. Images and
    calendars have no text here ("").
    """
    ext = os.path.splitext(raw_rel)[1].lower()
    if ext in IMAGE_EXTS or ext == ".ics":
        return ""
    if ext == ".pdf":
        return pdf_text(data)
    if ext == ".eml":
        return eml_text(data)
    if ext == ".msg":
        return msg_text(wiki._path(raw_rel))
    if ext == ".docx":
        return docx_text(data)[0]
    if ext == ".pptx":
        return pptx_text(data)
    if ext == ".xlsx":
        return xlsx_text(data)
    if ext == ".vtt":
        return vtt_text(data)
    if ext == ".epub":
        return epub_text(data)
    return _decode(data)


def extract(raw_rel: str, data: bytes, kind: str = "") -> Extracted:
    """The text (or image part) of a stored raw file — "" means "nothing extractable"."""
    ext = os.path.splitext(raw_rel)[1].lower()
    full = wiki._path(raw_rel)
    kind = kind or KINDS.get(ext, "txt")
    if ext in IMAGE_EXTS:
        _downscale(full)
        return Extracted("", _image_part(full), kind)
    if ext == ".ics":
        events = ics_events(data)
        text = "\n".join(f"{e['date']} {e['time']} {e['title']} — {', '.join(e['attendees'])}"
                         for e in events)
        return Extracted(text, None, kind, memory.tokens_est(text), memory.tokens_est(text), events)
    if ext == ".docx":
        kind = docx_text(data)[1]
    shown, shown_tokens, total_tokens = sample_text(raw_text(raw_rel, data))
    return Extracted(shown, None, kind, shown_tokens, total_tokens)


# --- manifest ----------------------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_entries() -> list:
    """Every line of `raw/manifest.jsonl`, parsed; a garbled line is skipped, never fatal."""
    entries = []
    for line in (wiki._read(MANIFEST) or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def manifest_find(digest: str) -> dict | None:
    return next((e for e in manifest_entries() if e.get("sha256") == digest), None)


def _manifest_append(entry: dict) -> None:
    full = wiki._path(MANIFEST)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --- source page -------------------------------------------------------------------


def _free_path(folder: str, slug: str, ext: str) -> str:
    """`folder/slug.ext`, with `-2`, `-3`… when that name is taken by other bytes."""
    rel = f"{folder}/{slug}{ext}"
    n = 2
    while os.path.exists(wiki._path(rel)):
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
    if memory.is_user(name, user_names):
        return f"- the user ({name})"
    etype = " ".join(str(entity.get("type") or "").split()).lower()
    folder = {"person": "people", "project": "projects"}.get(etype)
    if folder:
        rel = f"{folder}/{wiki.slugify(name)}.md"
        if wiki.page_exists(rel):
            return f"- [[{wiki.wikilink(rel)}]] — {name} ({etype or 'entity'})"
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
    today = memory._today_iso()
    fm = {"type": "source", "name": title, "tags": list(tags), "source": source,
          "updated": today, "seen": today,
          "kind": kind, "raw": raw_rel, "sha256": digest, **(extra or {})}
    wiki._write(rel, wiki._page_text(fm, _source_body(title, summary, entities, fed, user_names)))


# --- meetings (docs/31-office-formats.md) --------------------------------------------


def _attendee_link(name: str, user_names=()) -> str:
    """`people/<slug>` for a colleague, the literal "the user" for the profile owner."""
    return "the user" if memory.is_user(name, user_names) else f"people/{wiki.slugify(name)}"


def _write_meeting_page(rel: str, *, title: str, date: str, time: str, attendees: list,
                        source_rel: str, raw_rel: str, summary: str, decisions=(), actions=(),
                        user_names=()) -> None:
    """A `meetings/<date>-<slug>` page: meeting frontmatter and the four fixed sections.

    Written here, not through the `new_page` op, because the op cannot carry `date` / `time` /
    `attendees` or the Decisions/Actions sections — the same arrangement the source page has.
    """
    today = memory._today_iso()
    links = []
    for name in attendees:
        link = _attendee_link(name, user_names)
        if link not in links:
            links.append(link)
    fm = {"type": "meeting", "name": title, "tags": [], "source": raw_rel,
          "updated": today, "seen": today, "date": date, "time": time,
          "attendees": f"[{', '.join(links)}]"}
    lines = [f"# {title}", "", "## Summary", summary.strip() or "(no summary)",
             f"Source: [[{wiki.wikilink(source_rel)}]]", "", "## Decisions"]
    lines += [f"- {d}" for d in decisions]
    lines += ["", "## Actions"] + [f"- {a}" for a in actions]
    lines += ["", "## Notes", ""]
    wiki._write(rel, wiki._page_text(fm, "\n".join(lines)))


def _degraded(reason: str) -> None:
    """Note a degraded ingest: `ingest_end.degraded`, and why ("summary" beats "ops") for the reply."""
    trace.mark(degraded=True)
    if reason == "summary" or trace.marked("_degraded") is None:
        trace.mark(_degraded=reason)


def meeting_from_transcript(client, model: str, text: str, title: str, user_names=()) -> dict | None:
    """The meeting completion -> ``{title, date, attendees, project, decisions, actions,
    summary}``, tidied; None when the model cannot be reached or parsed."""
    try:
        system = MEETING_SYSTEM % (memory.user_note(user_names), memory._today_iso())
        raw = _ask(client, model, system, f"Filename/title: {title}\n\nTranscript:\n{text}", 900)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"meeting completion failed: {exc}")
        trace.error("ingest", "meeting_from_transcript", exc)
        _degraded("ops")
        return None
    data = memory._extract_json(raw, "{", "}")
    if not isinstance(data, dict):
        log.info("could not parse the meeting; filing the transcript only")
        _degraded("ops")
        return None

    def strings(key, limit=8):
        values = data.get(key) if isinstance(data.get(key), list) else []
        return [" ".join(str(v).split())[:300] for v in values if str(v).strip()][:limit]

    date = str(data.get("date") or "").strip()
    if not _DROP_DATE.fullmatch(date):
        date = memory._today_iso()
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
    link = f"[[{wiki.wikilink(meeting_rel)}]]"
    when = f"{meeting['title']}, {meeting['date']}"
    ops = []
    for name in meeting["attendees"]:
        if memory.is_user(name, user_names):
            continue
        path = f"people/{wiki.slugify(name)}"
        if not wiki.page_exists(f"{path}.md"):
            ops.append({"op": "new_page", "path": path, "type": "person", "name": name,
                        "summary": f"Met in {when}."})
        ops.append({"op": "append_note", "path": path, "text": f"Attended {link} ({when})."})
    if meeting["project"]:
        path = f"projects/{wiki.slugify(meeting['project'])}"
        what = meeting["decisions"][0] if meeting["decisions"] else meeting["summary"].split(". ")[0]
        if not wiki.page_exists(f"{path}.md"):
            ops.append({"op": "new_page", "path": path, "type": "project",
                        "name": meeting["project"], "summary": f"Discussed in {when}."})
        ops.append({"op": "append_note", "path": path,
                    "text": f"{link} ({meeting['date']}): {what}"[:wiki.MAX_NOTE_CHARS]})
    return ops


def _file_meeting(client, model: str, extracted, *, raw_rel: str, source_rel: str, title: str,
                  names) -> tuple:
    """A transcript's step 5: the meeting page plus the attendee/project ops. ``(meeting, ops)``."""
    meeting = meeting_from_transcript(client, model, extracted.text, title, names)
    if meeting is None:
        return "", []
    rel = _free_path("meetings", f"{meeting['date']}-{wiki.slugify(meeting['title'])}", ".md")
    _write_meeting_page(rel, title=meeting["title"], date=meeting["date"], time="",
                        attendees=meeting["attendees"], source_rel=source_rel, raw_rel=raw_rel,
                        summary=meeting["summary"], decisions=meeting["decisions"],
                        actions=meeting["actions"], user_names=names)
    return wiki.wikilink(rel), _attendee_ops(meeting, rel, names)


def _file_calendar(events: list, *, raw_rel: str, source_rel: str, names) -> tuple:
    """A calendar's step 5, no model involved: one stub per event, an `updated invite` note on
    a stub that already exists. ``(stubs written, ops for the existing ones)``."""
    written, ops = [], []
    for event in events:
        rel = f"meetings/{event['date']}-{wiki.slugify(event['title'])}.md"
        if wiki.wikilink(rel) in written:
            continue
        if wiki.page_exists(rel):
            ops.append({"op": "append_note", "path": wiki.wikilink(rel),
                        "text": (f"updated invite: {event['time'] or 'all day'}, "
                                 f"{len(event['attendees'])} attendees ([[{wiki.wikilink(source_rel)}]])")})
            continue
        _write_meeting_page(rel, title=event["title"], date=event["date"], time=event["time"],
                            attendees=event["attendees"], source_rel=source_rel, raw_rel=raw_rel,
                            summary=event["description"], user_names=names)
        written.append(wiki.wikilink(rel))
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
    candidates, rejected = wiki.reject_user_pages(ops, user_names)
    if kind in PUBLISHED_KINDS and people is None:
        people = memory.people_names()
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
    rel = wiki.normalise_path(op.get("path"))
    if rel is None or not rel.startswith("people/") or wiki.page_exists(rel):
        return None
    slug_name = os.path.basename(rel)[:-3].replace("-", " ")
    names = [" ".join(str(v or "").split()) for v in (op.get("name"), slug_name)]
    if any(name and memory.is_user(name, people) for name in names):
        return None
    return (f"{names[0] or slug_name} is not in the user's profile; a published work's "
            "authors and cited experts are not people pages")


def _target_allowed(path) -> str | None:
    rel = wiki.normalise_path(path)
    if rel is None:
        return f"bad path {path!r}"
    if wiki.page_exists(rel):
        return None
    if rel.split("/")[0] not in NEW_PAGE_FOLDERS:
        return f"{wiki.wikilink(rel)} is not in the index and not a people/projects page"
    return None


def summarise(client, model: str, kind: str, text: str, image_part, hint: str, title: str,
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
    system = SUMMARY_SYSTEM % (memory.user_note(user_names), memory._today_iso())
    data = None
    for attempt in range(1, SUMMARY_ATTEMPTS + 1):
        try:
            raw = _ask(client, model, system, content, 700)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"summary failed: {exc}")
            trace.error("ingest", "summarise", exc)
            _degraded("summary")
            if strict:
                raise
            fallback["summary"] = "the summary model could not be reached; the raw file is filed."
            return fallback
        parsed = memory._extract_json(raw, "{", "}")
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


def propose_page_ops(client, model: str, source_rel: str, summary: dict, hint: str,
                     hint_page: str = "", user_names=(), kind: str = "") -> list:
    """Ask for page ops about a just-filed source; [] on anything unparseable.

    For a published work (`kind` in PUBLISHED_KINDS) the prompt adds that authors and cited
    experts are not people pages.
    """
    index = wiki.index_text(wiki.WIKI_INDEX_TOKENS) or "(empty wiki)"
    entities = ", ".join(f"{e.get('name')} ({e.get('type')})" for e in summary.get("entities", []))
    related = wiki.related_context(" ".join(
        [str(summary.get("title") or ""), str(summary.get("summary") or "")]
        + [str(e.get("name") or "") for e in summary.get("entities", [])]))
    parts = [f"Index:\n{index}"] + ([related] if related else [])
    parts += [f"Source page: {wiki.wikilink(source_rel)}",
             f"Title: {summary.get('title')}",
             f"Summary:\n{untrusted(summary.get('summary'), summary.get('title'))}"]
    if entities:
        parts.append(f"Entities: {entities}")
    if hint:
        parts.append(f"User hint: {hint}")
    if hint_page:
        parts.append(f"The user filed this under: {hint_page}")
    try:
        rules = memory.user_note(user_names)
        if kind in PUBLISHED_KINDS:
            rules += "\n- " + PUBLISHED_NOTE
        system = OPS_SYSTEM % (rules, memory._today_iso())
        raw = _ask(client, model, system, "\n\n".join(parts), 800)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"page ops failed: {exc}")
        trace.error("ingest", "propose_page_ops", exc)
        _degraded("ops")
        return []
    ops = memory._extract_json(raw)
    if not isinstance(ops, list):
        log.info("could not parse the page ops; filing the source page only")
        _degraded("ops")
        return []
    return ops


# --- the ingest ---------------------------------------------------------------------


def _store(raw_rel: str, data: bytes) -> None:
    full = wiki._path(raw_rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)


def _rename_generic_raw(raw_rel: str, title: str) -> str:
    """`raw/inbox/<date>-note.md` becomes `raw/inbox/<date>-<slug of title>.md` once the model
    has named the document; a real filename (`falcon-plan.docx`) is left alone. Returns the path
    the raw file now has (deduped like any other)."""
    folder, name = os.path.split(raw_rel)
    stem, ext = os.path.splitext(name)
    slug = wiki.slugify(title)
    if not is_generic_title(stem) or is_generic_title(title) or not slug:
        return raw_rel
    date = _DROP_DATE.search(stem)
    new_rel = _free_path(folder, f"{date.group(1)}-{slug}" if date else slug, ext)
    os.replace(wiki._path(raw_rel), wiki._path(new_rel))
    return new_rel


def is_generic_title(title: str) -> bool:
    """A name that says nothing about the content (`note`, `untitled`, a stamped `note 2026-…`)."""
    words = " ".join(str(title or "").lower().replace("_", " ").replace("-", " ").split())
    words = _DROP_DATE.sub("", words).strip(" -:0123456789")
    return not words or words in GENERIC_TITLES


def _summarise_raw(client, model: str, raw_rel: str, data: bytes, kind: str, hint: str,
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
    return summarise(client, model, extracted.kind, extracted.text, extracted.image_part,
                     hint, label, names, strict), extracted


def _file_page(client, model: str, summary: dict, extracted, *, raw_rel: str, source_rel: str,
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
        meeting, ops = _file_meeting(client, model, extracted, raw_rel=raw_rel,
                                     source_rel=source_rel, title=page_title, names=names)
    elif not summary.get("unreadable"):
        ops = propose_page_ops(client, model, source_rel, summary, hint, page, names, kind)
    else:
        ops = []
    trace.verbose("ingest_text", summary=summary.get("summary", ""), ops=ops)
    if ops:
        kept, rejected = restrict_ops(ops, raw_rel, names, kind)
        applied, more = wiki.apply_page_ops(kept, raw_rel, commit=False)
        trace.mark(ops_rejected=wiki.log_rejected(rejected + more, log))

    fed = sorted({wiki.wikilink(wiki.normalise_path(op["path"]))
                  for op in applied if op.get("path")}
                 | {"decisions" for op in applied if op["op"] == "add_decision"}
                 | set(stubs) | ({meeting} if meeting else set()))
    write_page(fed, meeting)     # rewritten now that the ops are in, listing what it actually fed
    trace.mark(ops_applied=len(applied), pages=[wiki.wikilink(source_rel)] + fed)
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
               page: str = "", client=None, model: str = "", on_stage=None,
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
    if client is None:
        client, model = _client_model()
    with vault_lock.LOCK:
        wiki.ensure_vault()
        names = memory.user_names()
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
            summary, extracted = _summarise_raw(client, model, raw_rel, data, kind, hint, title,
                                                names, on_stage, strict, unreadable)
        except Exception:
            if strict:          # PRD 48: nothing but the stored original exists yet; drop just that file
                os.remove(wiki._path(raw_rel))
            raise
        kind = extracted.kind
        page_title = title or summary.get("title") or stem
        raw_rel = _rename_generic_raw(raw_rel, page_title)
        page_rels = []
        for i, png in enumerate(pages or [], 1):
            page_rels.append(f"{os.path.splitext(raw_rel)[0]}/page-{i:02d}.png")
            _store(page_rels[-1], png)
        source_rel = _free_path("sources", wiki.slugify(page_title), ".md")
        _stage(on_stage, "filing")
        fed = _file_page(client, model, summary, extracted, raw_rel=raw_rel,
                         source_rel=source_rel, digest=digest, page_title=page_title,
                         hint=hint, page=page, source=source or f"user drop {memory._today_iso()}",
                         names=names)
        _manifest_append({"sha256": digest, "path": raw_rel, "kind": kind,
                          "ingested_at": memory._stamp("%H:%M:%S"),
                          "page": wiki.wikilink(source_rel), "pages_touched": fed,
                          **({"pages": page_rels} if page_rels else {}), **(extra or {})})
        wiki.raw_chunks(raw_rel)          # PRD 32: the chunk cache, built while the bytes are hot
        wiki.rebuild_index()
        wiki._log([f"filed [[{wiki.wikilink(source_rel)}]] from {raw_rel}"
                   + (f" -> {', '.join(fed)}" if fed else "")])
        wiki._commit(f"wiki: filed {raw_rel}")
        trace.mark(path=wiki.wikilink(source_rel), title=page_title)
        return {"path": wiki.wikilink(source_rel), "raw": raw_rel, "already": False,
                "degraded": trace.marked("_degraded")}


# --- the inbox (docs/25-sources-inbox.md) --------------------------------------------


def _manifest_write(entries: list) -> None:
    """Rewrite the whole manifest — the only time a line is replaced or dropped."""
    wiki._write(MANIFEST, "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))


_DROP_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _when(date: str, ingested_at: str) -> str:
    """`2026-08-30 09:12` from the ISO stamp (which already carries the date); the drop date alone otherwise."""
    if "T" in ingested_at:                       # full ISO stamp: it already carries the date
        return ingested_at[:16].replace("T", " ")
    return " ".join(part for part in (date, ingested_at) if part)


def _source_meta(entry: dict) -> dict:
    """One inbox row: the manifest line joined with its page's frontmatter."""
    page_rel = f"{entry.get('page', '')}.md"
    fm = wiki.parse_page(wiki._read(page_rel))[0] if wiki.page_exists(page_rel) else {}
    match = _DROP_DATE.search(str(fm.get("source") or ""))
    date = match.group(1) if match else str(fm.get("updated") or "")
    return {
        "sha": entry.get("sha256", ""),
        "path": entry.get("page", ""),
        "raw": entry.get("path", ""),
        "kind": str(fm.get("kind") or entry.get("kind") or ""),
        "title": str(fm.get("name") or os.path.basename(entry.get("page", ""))),
        "when": _when(date, str(entry.get("ingested_at") or "")),
        "fed": list(entry.get("pages_touched") or []),
        "meeting": str(fm.get("meeting") or ""),
    }


def list_sources() -> list:
    """Every filed source, newest first (the manifest is append-only, so reversed = newest)."""
    return [_source_meta(e) for e in reversed(manifest_entries())]


def _entry_or_raise(digest: str) -> dict:
    entry = manifest_find(digest)
    if entry is None:
        raise IngestError(404, f"no filed source with sha {str(digest)[:12]}")
    return entry


def reingest(digest: str, *, client=None, model: str = "") -> dict:
    """Re-run summary + ops from the stored raw file; same source page, same sha, one commit.

    Touched pages get fresh dated notes (the ops are re-applied, not diffed); the manifest
    line is replaced in place with the union of old and new `pages_touched`.
    """
    if client is None:
        client, model = _client_model()
    with vault_lock.LOCK:
        entry = _entry_or_raise(digest)
        raw_rel, source_rel = entry["path"], f"{entry['page']}.md"
        try:
            with open(wiki._path(raw_rel), "rb") as f:
                data = f.read()
        except OSError as exc:
            raise IngestError(410, f"the raw file {raw_rel} is gone") from exc
        names = memory.user_names()
        kind = str(entry.get("kind") or KINDS.get(os.path.splitext(raw_rel)[1].lower()) or "txt")
        old_fm = wiki.parse_page(wiki._read(source_rel))[0]
        summary, extracted = _summarise_raw(client, model, raw_rel, data, kind, "", "", names)
        page_title = str(old_fm.get("name") or summary.get("title") or "")
        fed = _file_page(client, model, summary, extracted, raw_rel=raw_rel,
                         source_rel=source_rel, digest=digest, page_title=page_title,
                         hint="", page="",
                         source=str(old_fm.get("source") or f"user drop {memory._today_iso()}"),
                         names=names)
        fed = sorted(set(entry.get("pages_touched") or []) | set(fed))
        stamp = memory._stamp("%H:%M:%S")
        _manifest_write([{**e, "pages_touched": fed, "ingested_at": stamp}
                         if e.get("sha256") == digest else e for e in manifest_entries()])
        wiki.rebuild_index()
        wiki._log([f"re-ingested [[{entry['page']}]] from {raw_rel}"
                   + (f" -> {', '.join(fed)}" if fed else "")])
        wiki._commit(f"wiki: re-ingested {raw_rel}")
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


def forget(digest: str) -> dict:
    """Delete the source page and raw file, drop the manifest line, the notes it fed and the
    memory entries learned from it (docs/33-untrusted-text.md); one commit."""
    with vault_lock.LOCK:
        entry = _entry_or_raise(digest)
        raw_rel, page = entry["path"], entry["page"]
        title = _source_meta(entry)["title"]
        cleaned = []
        for fed in entry.get("pages_touched") or []:
            rel = wiki.normalise_path(fed)
            if rel is None or not wiki.page_exists(rel):
                continue
            before = wiki._read(rel)
            if f"(source: {raw_rel})" not in before:
                continue
            wiki._edit_page(rel, lambda body: _without_source_notes(body, raw_rel))
            cleaned.append(wiki.wikilink(rel))
        for rel in (f"{page}.md", raw_rel, *(entry.get("pages") or [])):
            wiki._git("rm", "-q", "-f", "--", rel, quiet=True)
            if os.path.exists(wiki._path(rel)):
                os.remove(wiki._path(rel))
        for folder in {os.path.dirname(rel) for rel in entry.get("pages") or []}:
            if os.path.isdir(wiki._path(folder)) and not os.listdir(wiki._path(folder)):
                os.rmdir(wiki._path(folder))
        _manifest_write([e for e in manifest_entries() if e.get("sha256") != digest])
        wiki.drop_raw_chunks(digest)
        wiki.rebuild_index()
        wiki._log([f"forgot [[{page}]] and {raw_rel}"
                   + (f" (notes removed from {', '.join(cleaned)})" if cleaned else "")])
        wiki._commit(f"wiki: forgot {raw_rel}")
        forgotten = memory.forget_source(title)
        return {"path": page, "raw": raw_rel, "cleaned": cleaned, "memory": forgotten}


def _dated(slug: str) -> str:
    """`<today>-<slug>`; a slug that already starts with a date loses it (no double date)."""
    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", slug)
    return f"{memory._today_iso()}-{wiki.slugify(slug)}"


def check_upload(filename: str, size: int, kind_hint: str = "file") -> tuple:
    """``(ext, kind)`` for an upload, or `IngestError` — the one type/size gate.

    The endpoint calls it on arrival (so an oversized drop is refused before any work happens)
    and `ingest_upload` calls it again for the callers that never went through HTTP.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    if kind_hint == "paste_image" and ext not in IMAGE_EXTS:
        ext = ".png"
    kind = KINDS.get(ext)
    if kind is None:
        raise IngestError(415, f"unsupported file type {ext or filename!r}")
    limit = (MAX_IMAGE_BYTES if ext in IMAGE_EXTS
             else MAX_TEXT_BYTES if ext in LARGE_TEXT_EXTS else MAX_FILE_BYTES)
    if size > limit:
        raise IngestError(413, f"{filename} is larger than {limit // (1024 * 1024)} MB")
    return ext, kind


def ingest_upload(data: bytes, filename: str, *, kind_hint: str = "file", title: str = "",
                  hint: str = "", page: str = "", client=None, model: str = "",
                  on_stage=None, source: str = "", strict: bool = False) -> dict:
    """One uploaded file (or pasted image). Raises `IngestError` for size/type refusals."""
    ext, kind = check_upload(filename, len(data), kind_hint)
    stem = title or os.path.splitext(os.path.basename(filename or ""))[0] or "drop"
    if is_generic_title(title):
        title = ""          # `note.txt`, `untitled`…: the summary's title names the page
    folder = "raw/screens" if ext in IMAGE_EXTS else "raw/inbox"
    raw_rel = f"{folder}/{_dated(stem)}{ext}"
    with trace.ingest(kind, raw=raw_rel, bytes=len(data)):
        return file_bytes(data, raw_rel, kind, title=title, hint=hint, page=page,
                          client=client, model=model, on_stage=on_stage, source=source, strict=strict)


def check_note(text: str) -> str:
    """The stripped note body, or `IngestError` — the text counterpart of `check_upload`."""
    body = (text or "").strip()
    if not body:
        raise IngestError(400, "nothing to file: the text is empty")
    if len(body.encode()) > MAX_TEXT_BYTES:      # pasted text is sampled like any text kind (PRD 31)
        raise IngestError(413, f"the text is larger than {MAX_TEXT_BYTES // (1024 * 1024)} MB")
    return body


def ingest_note(text: str, *, title: str = "", hint: str = "", page: str = "",
                client=None, model: str = "", on_stage=None) -> dict:
    """Pasted or spoken free text -> `raw/inbox/<date>-<slug>.md`."""
    body = check_note(text)
    stem = title or "note"
    if is_generic_title(title):
        title = ""
    raw_rel = f"raw/inbox/{_dated(stem)}.md"
    with trace.ingest("note", raw=raw_rel, bytes=len(body.encode())):
        return file_bytes(body.encode(), raw_rel, "note", title=title, hint=hint, page=page,
                          client=client, model=model, on_stage=on_stage)


def ingest_path(path: str, *, title: str = "", hint: str = "", page: str = "",
                client=None, model: str = "") -> dict:
    """A file already on disk (the eval harness's `ingest:` turn, an AppleScript drop)."""
    with trace.ingest("path", raw=os.path.basename(path)):
        try:
            with open(os.path.expanduser(path), "rb") as f:
                data = f.read()
        except OSError as exc:
            raise IngestError(400, f"cannot read {path}: {exc}") from exc
        return ingest_upload(data, os.path.basename(path), title=title, hint=hint, page=page,
                             client=client, model=model)


def ingest_email(message_id: str, markdown: str, *, hint: str = "", page: str = "",
                 client=None, model: str = "") -> dict:
    """A fetched email -> `raw/email/<id>.md` (headers block + body, already rendered)."""
    ident = wiki.slugify(message_id) or _dated("email")
    subject = next((line[len("Subject: "):] for line in markdown.splitlines()
                    if line.startswith("Subject: ")), "").strip()
    with trace.ingest("email", raw=f"raw/email/{ident}.md", bytes=len(markdown.encode())):
        return file_bytes(markdown.encode(), f"raw/email/{ident}.md", "email",
                          title=subject, hint=hint, page=page, client=client, model=model)


def ingest_news(title: str, markdown: str, *, hint: str = "", page: str = "",
                client=None, model: str = "") -> dict:
    """A flagged news article -> `raw/news/<slug>.md`."""
    with trace.ingest("news", raw=f"raw/news/{wiki.slugify(title)}.md", bytes=len(markdown.encode())):
        return file_bytes(markdown.encode(), f"raw/news/{wiki.slugify(title)}.md", "news",
                          title=title, hint=hint, page=page, client=client, model=model)


def decode_data_url(value: str) -> bytes:
    """`data:image/png;base64,…` -> bytes (the paste path when it arrives as a string)."""
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise IngestError(415, "the pasted image could not be decoded") from exc

