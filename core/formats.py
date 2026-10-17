"""The pure extractors: bytes in, text out (docs/20-raw-ingestion.md, docs/31-office-formats.md).

pdf/eml/msg/docx/pptx/xlsx/vtt/ics/epub readers, head/middle/tail sampling and the image downscale.
Nothing here touches the vault; `ingest.pipeline` stores the raw bytes and picks the extractor.
"""
import base64
import dataclasses
import datetime as dt
import email
import email.policy
import io
import os
import posixpath
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser

from core import settings
from core import trace

log = trace.logger("Sources")

#: how much extracted text the model is shown (the rest is sampled out, no chunk-and-merge)
RAW_CHUNK_TOKENS = int(os.getenv("RAW_CHUNK_TOKENS", "6000"))
#: the head / middle / tail shares of a sampled document
SAMPLE_SHARES = (0.60, 0.15, 0.25)
SKIPPED = "[… %d tokens skipped …]"
IMAGE_LONG_EDGE = 1568                  # what the multimodal model wants to see
#: below this, downscaling costs more than it saves (and keeps the tests off `sips`)
DOWNSCALE_ABOVE_BYTES = 256 * 1024
#: a PDF that yields less than this much text has no text layer we can use
MIN_PDF_CHARS = 20
#: xlsx cut: rows per sheet, sheets per book, characters per cell (the cut is stated in the text)
XLSX_MAX_ROWS = 200
XLSX_MAX_SHEETS = 20
XLSX_MAX_CELL = 80
#: a Teams .docx transcript: at least this many `Speaker   0:00` paragraphs
TEAMS_MIN_TURNS = 3
DEFAULT_HORIZON_DAYS = 14


def _tokens_est(text: str) -> int:
    return len(text) // 4        # memory.tokens_est, without the upward import


def sample_text(text: str) -> tuple:
    """``(shown, shown_tokens, total_tokens)`` — `text` sampled to `RAW_CHUNK_TOKENS`.

    A document over the budget is shown as its head, a slice of its middle and its tail
    (`SAMPLE_SHARES`), each gap marked `[… N tokens skipped …]`: tagging and summarising only
    need the shape of the document, and one call is all it gets (no chunk-and-merge).
    """
    text = text or ""
    total = _tokens_est(text)
    if total <= RAW_CHUNK_TOKENS:
        return text, total, total
    budget = RAW_CHUNK_TOKENS * 4
    head_n, mid_n, tail_n = (int(budget * share) for share in SAMPLE_SHARES)
    mid_at = (len(text) - mid_n) // 2
    head, mid, tail = text[:head_n], text[mid_at:mid_at + mid_n], text[len(text) - tail_n:]
    gap1 = _tokens_est(text[head_n:mid_at])
    gap2 = _tokens_est(text[mid_at + mid_n:len(text) - tail_n])
    shown = "\n\n".join([head.rstrip(), SKIPPED % gap1, mid.strip(), SKIPPED % gap2, tail.lstrip()])
    return shown, _tokens_est(head + mid + tail), total


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
    today = today or dt.date.today()
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


IMAGE_EXTS = {".png": "screenshot", ".jpg": "screenshot", ".jpeg": "screenshot"}
