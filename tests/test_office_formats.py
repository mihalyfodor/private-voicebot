"""PRD 31 — office formats, calendar and transcripts through the same drop. Tmp vault, mocked LLM.

Scenario numbers are the PRD's; fixtures are generated here with the same libraries the
extractors use (no binaries checked in). 8 is the e2e/fixture render (tests/e2e/test_sources.py
and the fixture assertions at the bottom), 11 is manual.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime as dt
import io
import json
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

import avatars
import chatbot
import memory
import sources
import wiki
from tests.conftest import FakeClient
from tests.test_sources import SUMMARY, commits, fake, prompts, read, vault  # noqa: F401 — fixtures

ROOT = os.path.join(os.path.dirname(__file__), "..")
TODAY = memory._today_iso()


# --------------------------------------------------------------------------- fixture builders


def docx_bytes(blocks: list) -> bytes:
    """`blocks`: ("h1", text) / ("h2", text) / ("p", text) / ("table", rows) in order."""
    import docx

    doc = docx.Document()
    for kind, value in blocks:
        if kind == "table":
            table = doc.add_table(rows=len(value), cols=len(value[0]))
            for r, row in enumerate(value):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell
        elif kind.startswith("h"):
            doc.add_heading(value, level=int(kind[1]))
        else:
            doc.add_paragraph(value)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def pptx_bytes(slides: list) -> bytes:
    """`slides`: (title, body, notes)."""
    import pptx

    deck = pptx.Presentation()
    for title, body, notes in slides:
        slide = deck.slides.add_slide(deck.slide_layouts[1])      # title + content
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        if notes:
            slide.notes_slide.notes_text_frame.text = notes
    buf = io.BytesIO()
    deck.save(buf)
    return buf.getvalue()


def xlsx_bytes(sheets: dict) -> bytes:
    """`sheets`: name -> rows (lists)."""
    import openpyxl

    book = openpyxl.Workbook()
    book.remove(book.active)
    for name, rows in sheets.items():
        sheet = book.create_sheet(name)
        for row in rows:
            sheet.append(row)
    buf = io.BytesIO()
    book.save(buf)
    return buf.getvalue()


VTT = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
<v Anna Kovacs>We need the load test before go-live.</v>

2
00:00:04.500 --> 00:00:07.000
<v Anna Kovacs>Bob, can you own it?</v>

3
00:00:07.500 --> 00:00:10.000
<v Bob Nagy>Yes, by Friday.</v>

4
00:00:10.500 --> 00:12:34.000
<v Mihaly Fodor>Agreed, deploy stays on Fridays.</v>
"""

MEETING = json.dumps({
    "title": "Falcon go-live sync", "date": "2026-08-28",
    "attendees": ["Anna Kovacs", "Bob Nagy", "Mihaly Fodor"], "project": "Falcon",
    "decisions": ["Deploy stays on Fridays."],
    "actions": ["Bob runs the load test by Friday."],
    "summary": "The team agreed the load test gates the go-live.",
})


def ics_bytes(events: list) -> bytes:
    """`events`: dicts with summary, start (datetime), minutes, attendees, description, rrule?, uid."""
    import icalendar

    cal = icalendar.Calendar()
    cal.add("prodid", "-//test//")
    cal.add("version", "2.0")
    for spec in events:
        ev = icalendar.Event()
        ev.add("uid", spec["uid"])
        ev.add("summary", spec["summary"])
        ev.add("dtstart", spec["start"])
        ev.add("dtend", spec["start"] + dt.timedelta(minutes=spec.get("minutes", 30)))
        ev.add("description", spec.get("description", ""))
        org = icalendar.vCalAddress("mailto:anna@example.com")
        org.params["cn"] = "Anna Kovacs"
        ev.add("organizer", org)
        for name, mail in spec.get("attendees", []):
            att = icalendar.vCalAddress(f"mailto:{mail}")
            att.params["cn"] = name
            ev.add("attendee", att)
        if spec.get("rrule"):
            ev.add("rrule", spec["rrule"])
        cal.add_component(ev)
    return cal.to_ical()


def two_events(description="Agenda: status") -> bytes:
    today = dt.date.fromisoformat(TODAY)
    at = lambda d, h: dt.datetime.combine(d, dt.time(h, 0))  # noqa: E731
    return ics_bytes([
        {"uid": "one@x", "summary": "Falcon kickoff", "start": at(today + dt.timedelta(days=3), 10),
         "attendees": [("Bob Nagy", "bob@example.com"), ("Mihaly Fodor", "me@example.com")],
         "description": description},
        {"uid": "weekly@x", "summary": "Falcon weekly", "start": at(today, 9), "minutes": 45,
         "rrule": {"freq": "weekly"}, "attendees": [("Bob Nagy", "bob@example.com")]},
    ])


def meeting_pages() -> list:
    folder = os.path.join(wiki.WIKI_PATH, "meetings")
    return sorted(f for f in os.listdir(folder) if not f.startswith(".")) if os.path.isdir(folder) else []


def be_the_user():
    memory._write_profile(memory._normalise(
        {"identity": {"name": "Mihaly Fodor", "nickname": "Misi"}}))


# --------------------------------------------------------------- 1. docx


def test_docx_keeps_headings_and_table_rows(vault, fake):
    data = docx_bytes([("h1", "Falcon plan"), ("p", "Go-live is September 15."),
                       ("h2", "Risks"), ("table", [["Risk", "Owner"], ["load test", "Bob"]])])
    filed = sources.ingest_upload(data, "falcon-plan.docx")

    prompt = prompts(fake)[0]
    assert "# Falcon plan" in prompt and "## Risks" in prompt
    assert "Go-live is September 15." in prompt
    assert "| Risk | Owner |" in prompt and "| load test | Bob |" in prompt
    assert "kind: docx" in read(filed["path"] + ".md")
    assert sources.manifest_entries()[0]["kind"] == "docx"


# --------------------------------------------------------------- 2. pptx


def test_pptx_gives_one_section_per_slide_with_notes(vault, fake):
    data = pptx_bytes([("Kickoff", "Why Falcon", "remember the budget"),
                       ("Timeline", "Q3 build, Q4 launch", ""),
                       ("Asks", "Two engineers", "ask Anna")])
    filed = sources.ingest_upload(data, "falcon-deck.pptx")

    prompt = prompts(fake)[0]
    assert "## Slide 1 — Kickoff\nWhy Falcon\nNotes: remember the budget" in prompt
    assert "## Slide 2 — Timeline\nQ3 build, Q4 launch" in prompt
    assert "## Slide 3 — Asks\nTwo engineers\nNotes: ask Anna" in prompt
    assert "kind: pptx" in read(filed["path"] + ".md")


# --------------------------------------------------------------- 3. xlsx


def test_xlsx_caps_each_sheet_at_200_rows_and_says_so(vault, fake):
    rows = [["id", "value"]] + [[i, f"row {i}"] for i in range(1, 300)]      # 300 rows
    data = xlsx_bytes({"Data": rows, "Notes": [["k", "v"], ["p99", 940]]})
    filed = sources.ingest_upload(data, "falcon-numbers.xlsx")

    prompt = prompts(fake)[0]
    assert "## Data" in prompt and "## Notes" in prompt
    assert "| id | value |" in prompt and "| p99 | 940 |" in prompt
    assert "| 199 | row 199 |" in prompt and "| 200 | row 200 |" not in prompt
    assert f"({sources.XLSX_MAX_ROWS} of 300 rows shown)" in prompt
    assert "kind: xlsx" in read(filed["path"] + ".md")


# --------------------------------------------------------------- 4. vtt -> meeting page


def test_vtt_turns_are_merged_and_timestamps_dropped():
    text = sources.vtt_text(VTT.encode())

    assert text.startswith("duration: 00:12:34")
    assert "00:00:01" not in text and "-->" not in text
    assert "**Anna Kovacs:** We need the load test before go-live. Bob, can you own it?" in text
    assert "**Bob Nagy:** Yes, by Friday." in text
    assert text.count("**Anna Kovacs:**") == 1


def test_vtt_files_a_meeting_page_and_notes_the_attendees(vault, monkeypatch):
    be_the_user()
    client = FakeClient(SUMMARY, MEETING)
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))
    before = len(commits())

    filed = sources.ingest_upload(VTT.encode(), "falcon-sync.vtt")

    assert len(client.calls) == 2                       # summary + meeting, no generic page ops
    assert "Transcript:" in prompts(client)[1] and "**Bob Nagy:**" in prompts(client)[1]
    assert meeting_pages() == ["2026-08-28-falcon-go-live-sync.md"]
    page = read("meetings/2026-08-28-falcon-go-live-sync.md")
    fm, body = wiki.parse_page(page)
    assert fm["type"] == "meeting" and fm["date"] == "2026-08-28"
    assert fm["attendees"] == ["people/anna-kovacs", "people/bob-nagy", "the user"]
    assert fm["source"] == filed["raw"]
    assert "## Decisions\n- Deploy stays on Fridays." in body
    assert "## Actions\n- Bob runs the load test by Friday." in body
    assert f"[[{filed['path']}]]" in body
    for who in ("anna-kovacs", "bob-nagy"):
        assert "Attended [[meetings/2026-08-28-falcon-go-live-sync]]" in read(f"people/{who}.md")
    assert "[[meetings/2026-08-28-falcon-go-live-sync]]" in read("projects/falcon.md")
    assert not wiki.page_exists("people/mihaly-fodor.md")      # the user never gets a page
    source = read(filed["path"] + ".md")
    assert "kind: transcript" in source
    assert "meeting: meetings/2026-08-28-falcon-go-live-sync" in source
    assert "- [[meetings/2026-08-28-falcon-go-live-sync]]" in source
    assert sources.list_sources()[0]["meeting"] == "meetings/2026-08-28-falcon-go-live-sync"
    assert len(commits()) == before + 1
    assert wiki.lint() == []


# --------------------------------------------------------------- 5. Teams .docx detection


TEAMS = [("p", "Anna Kovacs   0:12"), ("p", "We need the load test before go-live."),
         ("p", "Bob Nagy   0:40"), ("p", "Yes, by Friday."),
         ("p", "Anna Kovacs   1:05"), ("p", "Great, deploy stays on Fridays.")]


def test_a_teams_docx_export_takes_the_transcript_path(vault, monkeypatch):
    client = FakeClient(SUMMARY, MEETING)
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))

    filed = sources.ingest_upload(docx_bytes(TEAMS), "teams-export.docx")

    assert "**Anna Kovacs:** We need the load test before go-live." in prompts(client)[0]
    assert "0:12" not in prompts(client)[0]
    assert "kind: transcript" in read(filed["path"] + ".md")
    assert sources.manifest_entries()[0]["kind"] == "transcript"
    assert meeting_pages() == ["2026-08-28-falcon-go-live-sync.md"]


def test_a_normal_docx_is_not_a_transcript(vault, fake):
    filed = sources.ingest_upload(docx_bytes([("h1", "Plan"), ("p", "Anna 0:12 is a time of day."),
                                              ("p", "Nothing else.")]), "plan.docx")

    assert "kind: docx" in read(filed["path"] + ".md")
    assert len(fake.calls) == 2 and "Transcript:" not in prompts(fake)[1]
    assert meeting_pages() == []


# --------------------------------------------------------------- 6. ics -> deterministic stubs


def test_ics_writes_stubs_for_the_horizon_without_the_model(vault, monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))
    today = dt.date.fromisoformat(TODAY)
    before = len(commits())

    filed = sources.ingest_upload(two_events(), "falcon-week.ics")

    assert client.calls == []
    expected = [f"{(today + dt.timedelta(days=d)).isoformat()}-falcon-weekly.md" for d in (0, 7, 14)]
    expected.append(f"{(today + dt.timedelta(days=3)).isoformat()}-falcon-kickoff.md")
    assert meeting_pages() == sorted(expected)
    fm, body = wiki.parse_page(read("meetings/" + expected[-1]))
    assert fm["type"] == "meeting" and fm["time"] == "10:00–10:30"
    assert fm["attendees"] == ["people/anna-kovacs", "people/bob-nagy", "people/mihaly-fodor"]
    assert fm["source"] == filed["raw"]
    assert "Agenda: status" in body and f"[[{filed['path']}]]" in body
    source = read(filed["path"] + ".md")
    assert "kind: calendar" in source and f"meeting: meetings/{expected[0][:-3]}" in source
    assert "4 meeting(s)" in source
    assert len(commits()) == before + 1
    assert wiki.lint() == []
    # the same bytes again: nothing happens
    again = sources.ingest_upload(two_events(), "falcon-week.ics")
    assert again["already"] is True and len(commits()) == before + 1


def test_ics_expands_recurring_events_to_the_configured_horizon(vault, monkeypatch, file_sandbox):
    monkeypatch.setattr(sources, "_client_model", lambda: (FakeClient(), "m"))
    avatars.save_setting("calendar_horizon_days", 28)

    sources.ingest_upload(two_events(), "falcon-week.ics")

    assert len([p for p in meeting_pages() if "weekly" in p]) == 5      # today, +7 … +28


def test_an_updated_invite_notes_the_existing_stub_once(vault, monkeypatch):
    monkeypatch.setattr(sources, "_client_model", lambda: (FakeClient(), "m"))
    sources.ingest_upload(two_events(), "falcon-week.ics")
    stubs = meeting_pages()
    before = len(commits())

    sources.ingest_upload(two_events(description="Agenda: go-live"), "falcon-week-v2.ics")

    assert meeting_pages() == stubs                         # nothing new, nothing rewritten
    kickoff = next(p for p in stubs if "kickoff" in p)
    page = read("meetings/" + kickoff)
    assert page.count("updated invite") == 1
    assert "Agenda: status" in page and "Agenda: go-live" not in page
    assert f"- {TODAY} updated invite: 10:00–10:30, 3 attendees" in page
    assert len(commits()) == before + 1
    assert wiki.lint() == []


def test_the_calendar_horizon_setting_round_trips(file_sandbox):
    assert sources.calendar_horizon_days() == 14
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "set_setting", "key": "calendar_horizon_days", "value": 28})
        while True:
            msg = ws.receive_json()
            if msg.get("type") in ("setting", "error"):
                break
    assert msg == {"type": "setting", "key": "calendar_horizon_days", "value": 28}
    assert sources.calendar_horizon_days() == 28
    assert client.get("/api/config").json()["calendar_horizon_days"] == 28


# --------------------------------------------------------------- 7. broken files


@pytest.mark.parametrize("name", ["bad.docx", "bad.pptx", "bad.xlsx", "bad.vtt", "bad.ics"])
def test_a_broken_file_of_each_kind_is_filed_as_unreadable(vault, fake, name):
    filed = sources.ingest_upload(b"\x00garbage\x01" * 8, name)

    page = read(filed["path"] + ".md")
    assert "unreadable" in page.lower()
    assert fake.calls == []
    assert meeting_pages() == []
    assert wiki.lint() == []


@pytest.mark.parametrize("name", ["empty.docx", "empty.vtt", "empty.ics"])
def test_an_empty_file_is_unreadable_too(vault, fake, name):
    filed = sources.ingest_upload(b"", name)
    assert "unreadable" in read(filed["path"] + ".md").lower()
    assert fake.calls == []


# --------------------------------------------------------------- 9. large text


def test_a_3mb_text_is_sampled_head_middle_tail(vault, fake):
    lines = [f"line {i} of the game FAQ about Falcon.\n" for i in range(90000)]   # ~3.3 MB
    data = "".join(lines).encode()
    assert len(data) > 3 * 1024 * 1024
    filed = sources.ingest_upload(data, "faq.txt")

    prompt = prompts(fake)[0]
    assert "line 0 of" in prompt and "line 89999 of" in prompt         # head and tail
    import re
    shown = [int(n) for n in re.findall(r"line (\d+) of", prompt)]
    assert any(30000 < n < 60000 for n in shown)                         # the middle slice
    assert prompt.count("tokens skipped") == 2
    assert memory.tokens_est(prompt) <= sources.RAW_CHUNK_TOKENS + 200
    fm, _ = wiki.parse_page(read(filed["path"] + ".md"))
    assert int(fm["shown_tokens"]) < int(fm["total_tokens"])
    assert int(fm["total_tokens"]) == memory.tokens_est(data.decode())


def test_an_oversized_png_is_still_refused(vault, fake):
    """Images keep PRD 20's 4 MB cap and pdf its 2 MB one; only the text kinds grew."""
    with pytest.raises(sources.IngestError) as exc:
        sources.check_upload("big.png", sources.MAX_IMAGE_BYTES + 1)
    assert exc.value.status == 413
    assert sources.check_upload("big.png", 3 * 1024 * 1024) == (".png", "screenshot")
    with pytest.raises(sources.IngestError):
        sources.check_upload("big.pdf", 3 * 1024 * 1024)               # pdf keeps the 2 MB cap
    assert sources.check_upload("big.docx", 9 * 1024 * 1024) == (".docx", "docx")


def test_a_short_text_records_no_token_counts(vault, fake):
    filed = sources.ingest_upload(b"short note about Falcon", "falcon.txt")
    fm, _ = wiki.parse_page(read(filed["path"] + ".md"))
    assert "shown_tokens" not in fm and "total_tokens" not in fm


# --------------------------------------------------------------- 10. pasted text is titled by the model


FAQ_SUMMARY = json.dumps({
    "title": "Falcon rocket FAQ", "summary": "Answers about the Falcon rocket game's missions.",
    "entities": [], "topic": "gaming", "unreadable": False,
})


def test_a_paste_is_named_by_the_model_and_tagged_with_its_topic(vault, monkeypatch):
    client = FakeClient(FAQ_SUMMARY, "[]")
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))
    text = ("Frequently Asked Questions about the Falcon rocket game, version 3, updated August 2026\n"
            "Q: How do I land? A: Slowly.")

    filed = sources.ingest_note(text)

    assert filed["path"] == "sources/falcon-rocket-faq"
    fm, _ = wiki.parse_page(read("sources/falcon-rocket-faq.md"))
    assert fm["name"] == "Falcon rocket FAQ" and fm["tags"] == ["gaming"]
    assert "Filename/title" not in prompts(client)[0]                 # no generic name offered


def test_a_generic_file_name_defers_to_the_model_too(vault, monkeypatch):
    client = FakeClient(FAQ_SUMMARY, "[]")
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))
    filed = sources.ingest_upload(b"Q: How do I land? A: Slowly.", "note.txt", title="untitled")
    assert filed["path"] == "sources/falcon-rocket-faq"
    assert filed["raw"].endswith("-falcon-rocket-faq.txt")     # PRD 32: renamed after the title


def test_a_real_file_name_still_wins(vault, fake):
    filed = sources.ingest_upload(b"Q: How do I land? A: Slowly.", "landing-guide.txt",
                                  title="landing-guide")
    assert filed["path"] == "sources/landing-guide"


# --------------------------------------------------------------- 8/10. the client fixture


@pytest.fixture(scope="module")
def r():
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    out = subprocess.run(["node", "tests/app_fixture.mjs"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_sources_view_shows_the_new_kind_chips_and_the_meeting_link(r):
    assert r["src_chips"] == ["calendar", "docx", "pptx", "transcript", "xlsx"]
    assert r["src_meeting_button"] is True and r["src_meeting_button_plain"] is False
    assert r["src_meeting_click"] == {"view": "wiki", "sent": ["meetings/2026-08-30-falcon-sync"]}


def test_intake_leaves_a_paste_untitled_but_keeps_a_file_name(r):
    assert r["intake_paste_title"] == {"value": ""}
    assert r["intake_file_title"] == "falcon-status"
    assert r["intake_generic_title"] == ""
    assert r["intake_docx_ok"] is True and r["intake_doc_ok"] is False
    html = open(os.path.join(ROOT, "static", "index.html")).read()
    assert 'id="intake-title" placeholder="model picks a title"' in html


def test_the_horizon_field_sends_set_setting(r):
    assert r["horizon_shown"] == "28"
    assert r["horizon_sent"] == [{"action": "set_setting", "key": "calendar_horizon_days", "value": 21}]


# --------------------------------------------------------------- 6b. ics_events details


def test_ics_events_honours_exdate_and_names_the_organizer_first():
    import icalendar

    today = dt.date.fromisoformat(TODAY)
    data = two_events()
    cal = icalendar.Calendar.from_ical(data)
    weekly = next(c for c in cal.walk("VEVENT") if c.get("UID") == "weekly@x")
    weekly.add("exdate", dt.datetime.combine(today + dt.timedelta(days=7), dt.time(9, 0)))
    events = sources.ics_events(cal.to_ical(), today=today, horizon=14)

    weekly_dates = [e["date"] for e in events if e["title"] == "Falcon weekly"]
    assert weekly_dates == [today.isoformat(), (today + dt.timedelta(days=14)).isoformat()]
    kickoff = next(e for e in events if e["title"] == "Falcon kickoff")
    assert kickoff["attendees"] == ["Anna Kovacs", "Bob Nagy", "Mihaly Fodor"]
    assert kickoff["time"] == "10:00–10:30" and kickoff["uid"] == "one@x"


# --- epub (PRD 31 addendum, 2026-08-30) -------------------------------------------------


def _epub(chapters: list[tuple[str, str]]) -> bytes:
    """A minimal EPUB: container.xml → content.opf → one XHTML per chapter, in spine order."""
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml",
                    '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                    '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                    '</rootfiles></container>')
        items = "".join(f'<item id="c{i}" href="ch{i}.xhtml" media-type="application/xhtml+xml"/>'
                        for i in range(len(chapters)))
        refs = "".join(f'<itemref idref="c{i}"/>' for i in reversed(range(len(chapters))))   # spine reversed
        zf.writestr("OEBPS/content.opf",
                    f'<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata/>'
                    f'<manifest>{items}</manifest><spine>{refs}</spine></package>')
        for i, (title, body) in enumerate(chapters):
            zf.writestr(f"OEBPS/ch{i}.xhtml",
                        f'<html xmlns="http://www.w3.org/1999/xhtml"><head><style>p{{}}</style></head>'
                        f'<body><h1>{title}</h1><p>{body}</p><h2>Notes</h2><ul><li>one</li><li>two</li></ul>'
                        f'<script>alert(1)</script></body></html>')
    return buf.getvalue()


def test_epub_extracts_chapters_in_spine_order_with_headings():
    data = _epub([("Chapter One", "Bounded contexts keep models apart."),
                  ("Chapter Two", "CQRS splits reads from writes.")])
    text = sources.epub_text(data)

    assert text.index("# Chapter Two") < text.index("# Chapter One")      # spine order, not file order
    assert "## Notes" in text and "- one" not in text and "one\ntwo" in text.replace("\n\n", "\n")
    assert "alert" not in text and "p{}" not in text                        # script/style dropped
    assert sources.KINDS[".epub"] == "epub" and ".epub" in sources.LARGE_TEXT_EXTS
    assert sources.raw_text("raw/inbox/x.epub", data) == text


def test_broken_epub_is_unreadable_not_a_crash():
    assert sources.epub_text(b"not a zip") == ""
    assert sources.epub_text(_epub([])) == ""


# --- authors are not people (PRD 31 addendum, 2026-08-30) -------------------------------


BOOK_SUMMARY = json.dumps({
    "title": "Domain-Driven Design",
    "summary": "Eric Evans explains bounded contexts; Anna Kovacs is thanked for the review.",
    "entities": [{"name": "Eric Evans", "type": "person"}, {"name": "Anna Kovacs", "type": "person"}],
    "topic": "work/architecture", "unreadable": False,
})

BOOK_OPS = json.dumps([
    {"op": "new_page", "path": "people/eric-evans", "type": "person", "name": "Eric Evans",
     "summary": "Author/Expert in domain-driven design."},
    {"op": "append_note", "path": "people/anna-kovacs", "text": "Reviewed the DDD book."},
    {"op": "new_page", "path": "people/martin-fowler", "type": "person", "name": "Martin Fowler",
     "summary": "Wrote the foreword."},
])


def _colleague_anna():
    memory._write_profile(memory._normalise(
        {"identity": {"name": "Mihaly Fodor", "nickname": "Misi"},
         "people": [{"name": "Anna Kovacs", "rel": "colleague", "note": "tech lead"}]}))
    wiki.apply_page_ops([{"op": "new_page", "path": "people/anna-kovacs", "type": "person",
                          "name": "Anna Kovacs", "summary": "Tech lead."}], "seed")


def test_an_epub_author_gets_no_people_page_but_a_colleague_gets_the_note(vault, monkeypatch, capsys):
    _colleague_anna()
    client = FakeClient(BOOK_SUMMARY, BOOK_OPS)
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))
    data = _epub([("Preface", "By Eric Evans. Thanks to Anna Kovacs for the review."),
                  ("Chapter One", "Bounded contexts keep models apart.")])

    filed = sources.ingest_upload(data, "ddd.epub")

    assert not wiki.page_exists("people/eric-evans.md")
    assert not wiki.page_exists("people/martin-fowler.md")
    assert "Reviewed the DDD book." in read("people/anna-kovacs.md")
    assert "kind: epub" in read(filed["path"] + ".md")
    assert sources.manifest_entries()[0]["pages_touched"] == ["people/anna-kovacs"]
    out = capsys.readouterr().out
    assert "Eric Evans is not in the user's profile" in out
    ops_system = client.calls[1]["messages"][0]["content"]
    assert "authors, editors and cited experts are NOT people pages" in ops_system
    assert "authors" not in client.calls[0]["messages"][0]["content"]     # summary prompt untouched


def test_restrict_ops_lets_a_published_work_touch_only_known_people(vault):
    wiki.apply_page_ops([{"op": "new_page", "path": "people/bob-nagy", "type": "person",
                          "name": "Bob Nagy", "summary": "Ops."}], "seed")
    ops = [
        {"op": "new_page", "path": "people/eric-evans", "type": "person", "name": "Eric Evans"},
        {"op": "new_page", "path": "people/ana-kovacs", "type": "person", "name": "Ana Kovacs"},
        {"op": "append_note", "path": "people/bob-nagy", "text": "existing page: fine"},
        {"op": "append_note", "path": "people/martin-fowler", "text": "no page yet: refused"},
        {"op": "new_page", "path": "projects/ddd", "type": "project", "name": "DDD"},
    ]
    people = ["Anna Kovacs"]

    kept, rejected = sources.restrict_ops(ops, "raw/inbox/ddd.epub", (), "epub", people)
    assert [op["path"] for op in kept] == ["people/ana-kovacs", "people/bob-nagy", "projects/ddd"]
    assert [bad["op"]["path"] for bad in rejected] == ["people/eric-evans", "people/martin-fowler"]
    assert all("authors and cited experts" in bad["reason"] for bad in rejected)

    # a plain markdown drop or a transcript is not a published work: nothing extra is refused
    for kind in ("md", "transcript", ""):
        kept, rejected = sources.restrict_ops(ops, "raw/inbox/ddd.md", (), kind, people)
        assert len(kept) == 5 and rejected == []


def test_a_teams_docx_transcript_still_files_the_attendees_as_people(vault, monkeypatch):
    be_the_user()
    client = FakeClient(SUMMARY, MEETING)
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))

    sources.ingest_upload(docx_bytes(TEAMS), "teams-export.docx")

    assert wiki.page_exists("people/anna-kovacs.md") and wiki.page_exists("people/bob-nagy.md")
    assert not wiki.page_exists("people/mihaly-fodor.md")
    assert "NOT people pages" not in client.calls[1]["messages"][0]["content"]


def test_a_docx_report_tells_the_model_its_authors_are_not_people(vault, monkeypatch):
    client = FakeClient(SUMMARY, "[]")
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))
    sources.ingest_upload(docx_bytes([("h1", "Report"), ("p", "Findings by Eric Evans.")]),
                          "report.docx")
    assert "NOT people pages" in client.calls[1]["messages"][0]["content"]


def test_a_summary_without_a_title_is_asked_once_more(vault, monkeypatch, capsys):
    client = FakeClient("{{garbled", FAQ_SUMMARY, "[]")
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))

    filed = sources.ingest_note("Q: How do I land? A: Slowly.")

    assert filed["path"] == "sources/falcon-rocket-faq"
    assert len(client.calls) == 3                          # summary, retry, page ops
    assert client.calls[0]["messages"] == client.calls[1]["messages"]
    assert "asking once more" in capsys.readouterr().out


def test_a_summary_garbled_twice_falls_back_to_the_stem(vault, monkeypatch):
    client = FakeClient("{{garbled", "still {{garbled")
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))

    filed = sources.ingest_upload(b"Q: How do I land? A: Slowly.", "landing-guide.txt")

    assert filed["path"].endswith("-landing-guide")        # named from the raw stem
    assert len(client.calls) == 2                          # unreadable: no page ops asked
    assert "could not be parsed" in read(filed["path"] + ".md")
