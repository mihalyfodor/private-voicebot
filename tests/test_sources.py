"""Unit tests for raw ingestion — tmp vault, mocked LLM. Mirrors docs/20-raw-ingestion.md.

Scenario numbers are the PRD's test scenarios (0-8); 7b is the live harness script
eval/scripts/sources.yaml and 9 is manual.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import struct
import subprocess
import zlib

import pytest
from fastapi.testclient import TestClient

import chatbot
import memory
import sources
import wiki
from tests.conftest import FakeClient


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def vault(wiki_sandbox, memory_sandbox):
    """A created, committed, empty vault (the autouse fixture only points WIKI_PATH at tmp).

    `memory_sandbox` comes along so an ingest reading the profile (for the user's names, PRD 26)
    can never see the real memory.json.
    """
    wiki.ensure_vault()
    return wiki_sandbox


@pytest.fixture
def fake(monkeypatch):
    """A FakeClient wired in as the module's client: summary call first, page ops second."""
    client = FakeClient(SUMMARY, OPS)
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "fake-model"))
    return client


SUMMARY = json.dumps({
    "title": "Falcon status",
    "summary": "Project Falcon is amber; the load test showed 940 ms p99.",
    "entities": [{"name": "Anna", "type": "person"}, {"name": "Falcon", "type": "project"}],
    "unreadable": False,
})

OPS = json.dumps([
    {"op": "new_page", "path": "projects/falcon", "type": "project", "name": "Falcon",
     "summary": "Platform project, amber."},
])


def commits() -> list:
    out = subprocess.run(["git", "log", "--oneline"], cwd=wiki.WIKI_PATH,
                         capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def read(rel: str) -> str:
    return wiki._read(rel)


def raw_files() -> list:
    """Every stored original, vault-relative (the manifest itself excluded)."""
    root = os.path.join(wiki.WIKI_PATH, "raw")
    found = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            rel = os.path.relpath(os.path.join(base, name), wiki.WIKI_PATH).replace(os.sep, "/")
            if not name.startswith(".") and rel != sources.MANIFEST:
                found.append(rel)
    return sorted(found)


def prompts(client: FakeClient) -> list:
    """The user content of every completion the client was asked for."""
    return [call["messages"][1]["content"] for call in client.calls]


def pdf_bytes(pages: list, with_text: bool = True) -> bytes:
    """A minimal, valid PDF: one page per entry in `pages`, or no text layer at all."""
    contents, objs, kids = [], [], []
    first_page_obj = 3
    for i, lines in enumerate(pages):
        if with_text:
            stream = ("BT /F1 14 Tf 40 700 Td " +
                      " ".join(f"({line}) Tj 0 -20 Td" for line in lines) + " ET\n")
        else:
            stream = "0 0 0 rg 40 40 100 100 re f\n"
        contents.append(stream)
        kids.append(f"{first_page_obj + i * 2} 0 R")
    objs.append("<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>")
    font_obj = first_page_obj + 2 * len(pages)
    for i, stream in enumerate(contents):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
                    f"/Contents {first_page_obj + i * 2 + 1} 0 R >>")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}endstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def png_bytes(width: int = 4, height: int = 4) -> bytes:
    """A valid greyscale PNG, built by hand — Pillow is not a dependency of this app."""
    raw = b"".join(b"\x00" + bytes([(x * 60) % 256 for x in range(width)]) for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


EML = """From: Anna Kovacs <anna@example.com>
To: Mihaly <mihaly@example.com>
Subject: Falcon go-live moved
Date: Fri, 29 Aug 2026 09:12:00 +0200
Content-Type: text/plain; charset="utf-8"

Hi, the go-live moved to September 15 after the load test. Anna
"""


# --------------------------------------------------------------- 1. a dropped markdown file


def test_a_dropped_markdown_file_is_stored_summarised_and_committed(vault, fake):
    before = len(commits())
    filed = sources.ingest_upload(b"# Falcon\n\nFalcon is amber.\n", "falcon-status.md",
                                  hint="this is the weekly status")

    assert filed["already"] is False
    assert filed["raw"].startswith("raw/inbox/") and filed["raw"].endswith(".md")
    assert filed["path"] == "sources/falcon-status"

    page = read("sources/falcon-status.md")
    assert "type: source" in page and "kind: md" in page
    assert f"raw: {filed['raw']}" in page
    assert "sha256: " + sources.sha256(b"# Falcon\n\nFalcon is amber.\n") in page
    assert "## Summary" in page and "940 ms" in page
    assert "## Entities" in page and "## Fed" in page
    assert "[[projects/falcon]]" in page          # the page it fed

    entries = sources.manifest_entries()
    assert len(entries) == 1
    assert entries[0]["path"] == filed["raw"]
    assert entries[0]["kind"] == "md"
    assert entries[0]["pages_touched"] == ["projects/falcon"]
    assert entries[0]["ingested_at"].startswith(memory._today_iso())

    assert len(commits()) == before + 1           # exactly one commit for the whole ingest
    assert "sources/falcon-status" in read("index.md")
    assert "this is the weekly status" in prompts(fake)[0]


def test_the_same_bytes_twice_file_nothing_the_second_time(vault, fake, monkeypatch):
    data = b"# Falcon\n\nFalcon is amber.\n"
    first = sources.ingest_upload(data, "falcon.md")
    after_first = (len(commits()), raw_files(), len(fake.calls))

    monkeypatch.setattr(sources, "_client_model", lambda: (FakeClient(SUMMARY, OPS), "m"))
    again = sources.ingest_upload(data, "falcon.md")

    assert again == {"path": first["path"], "raw": first["raw"], "already": True}
    assert (len(commits()), raw_files(), len(fake.calls)) == after_first
    assert len(sources.manifest_entries()) == 1


def test_a_second_document_with_the_same_name_gets_its_own_files(vault, fake, monkeypatch):
    monkeypatch.setattr(sources, "_client_model",
                        lambda: (FakeClient(SUMMARY, OPS, SUMMARY, "[]"), "m"))
    first = sources.ingest_upload(b"one", "falcon.md")
    second = sources.ingest_upload(b"two", "falcon.md")

    assert first["raw"] != second["raw"]
    assert first["path"] != second["path"]
    assert len(raw_files()) == 2


# --------------------------------------------------------------- 2. PDF extraction


def test_pdf_text_reaches_the_model(vault, fake):
    data = pdf_bytes([["Project Falcon is amber"], ["load test showed 940 ms"]])
    filed = sources.ingest_upload(data, "falcon.pdf")

    assert filed["raw"].endswith(".pdf")
    assert "kind: pdf" in read(filed["path"] + ".md")
    assert "Project Falcon is amber" in prompts(fake)[0]
    assert "load test showed 940 ms" in prompts(fake)[0]


def test_a_pdf_without_a_text_layer_is_filed_as_unreadable_with_no_ops(vault, fake):
    filed = sources.ingest_upload(pdf_bytes([["ignored"]], with_text=False), "scan.pdf")

    page = read(filed["path"] + ".md")
    assert "unreadable" in page.lower()
    assert fake.calls == []                       # nothing to summarise, no call made
    assert not wiki.page_exists("projects/falcon.md")
    assert os.path.isfile(os.path.join(wiki.WIKI_PATH, filed["raw"]))   # the raw link survives


# --------------------------------------------------------------- 3. restricted page ops


def test_ops_outside_the_index_and_people_projects_are_rejected(vault, monkeypatch):
    wiki.apply_page_ops([{"op": "new_page", "path": "meetings/2026-08-29", "type": "meeting",
                          "name": "Standup", "summary": "daily"}], "seed")
    ops = json.dumps([
        {"op": "new_page", "path": "people/anna", "type": "person", "name": "Anna",
         "summary": "Tech lead."},
        {"op": "append_note", "path": "meetings/2026-08-29", "text": "in the index, allowed"},
        {"op": "new_page", "path": "meetings/2026-09-01", "type": "meeting", "name": "New",
         "summary": "not in the index"},
        {"op": "new_page", "path": "sources/sneaky", "type": "source", "name": "Sneaky",
         "summary": "not allowed either"},
    ])
    client = FakeClient(SUMMARY, ops)
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))

    filed = sources.ingest_upload(b"a document", "doc.md")

    assert wiki.page_exists("people/anna.md")
    assert "in the index, allowed" in read("meetings/2026-08-29.md")
    assert not wiki.page_exists("meetings/2026-09-01.md")
    assert not wiki.page_exists("sources/sneaky.md")
    # every applied op cites the raw file
    assert f"(source: {filed['raw']})" in read("meetings/2026-08-29.md")
    assert f"source: {filed['raw']}" in read("people/anna.md")
    assert sorted(sources.manifest_entries()[0]["pages_touched"]) == [
        "meetings/2026-08-29", "people/anna"]


def test_restrict_ops_stamps_the_raw_path_on_everything_it_keeps(vault):
    kept, rejected = sources.restrict_ops([
        {"op": "new_page", "path": "projects/atlas"},
        {"op": "add_decision", "title": "deploy on Fridays", "text": "agreed"},
        {"op": "append_note", "path": "../etc/passwd", "text": "no"},
    ], "raw/inbox/2026-08-29-doc.md")

    assert [op["source"] for op in kept] == ["raw/inbox/2026-08-29-doc.md"] * 2
    assert len(rejected) == 1 and "bad path" in rejected[0]["reason"]


# --------------------------------------------------------------- 4. file_source


def test_file_source_note_writes_a_raw_note_and_a_page(vault, fake):
    from tools import sources as tool

    reply = tool.file_source("note", "Falcon go-live moved to September 15")

    raw = raw_files()
    assert len(raw) == 1 and raw[0].startswith("raw/inbox/") and raw[0].endswith(".md")
    assert "Falcon go-live moved to September 15" in read(raw[0])
    assert "kind: note" in read("sources/falcon-status.md")
    assert "filed as sources/falcon-status" in reply


def test_file_source_email_stores_the_headers_block(vault, fake, monkeypatch):
    from tools import gmail, sources as tool

    monkeypatch.setattr(gmail, "LAST_RESULTS",
                        {"18f0": {"from": "Anna Kovacs <anna@example.com>",
                                  "subject": "Falcon go-live moved", "snippet": ""}})
    monkeypatch.setattr(gmail, "fetch_message", lambda msg_id: EML.split("\n\n")[0] +
                        "\n\nthe go-live moved to September 15")

    reply = tool.file_source("email", "anna")

    assert "raw/email/18f0.md" in raw_files()
    stored = read("raw/email/18f0.md")
    assert stored.startswith("From: Anna Kovacs") and "Subject: Falcon go-live moved" in stored
    assert "kind: email" in read("sources/falcon-go-live-moved.md")
    assert "filed as sources/falcon-go-live-moved" in reply


def test_file_source_with_an_unknown_ref_writes_nothing(vault, fake, monkeypatch):
    from tools import gmail, sources as tool

    monkeypatch.setattr(gmail, "LAST_RESULTS", {})
    reply = tool.file_source("email", "anna")

    assert "couldn't tell which email" in reply
    assert raw_files() == [] and wiki.pages() == []
    assert fake.calls == []


def test_file_source_rejects_an_unknown_kind(vault, fake):
    from tools import sources as tool

    assert "not 'video'" in tool.file_source("video", "whatever")
    assert raw_files() == []


def test_file_source_news_uses_the_last_headlines(vault, fake, monkeypatch):
    from tools import news, sources as tool

    monkeypatch.setattr(news, "LAST_RESULTS",
                        [("Falcon launches", "https://example.com/falcon")])
    monkeypatch.setattr(news, "article_text", lambda url: "The Falcon rocket launched today.")

    reply = tool.file_source("news", "2")
    assert "couldn't tell which story" in reply

    reply = tool.file_source("news", "1")
    assert "raw/news/falcon-launches.md" in raw_files()
    assert "The Falcon rocket launched today." in read("raw/news/falcon-launches.md")
    assert "kind: news" in read("sources/falcon-launches.md")
    assert "filed as sources/falcon-launches" in reply


# --------------------------------------------------------------- 6. the size cap


def test_a_huge_document_is_capped_with_a_marker(vault, fake):
    huge = ("Falcon status line. " * 20000)      # ~100 k tokens
    sources.ingest_upload(huge.encode(), "huge.txt")

    prompt = prompts(fake)[0]
    assert prompt.count("tokens skipped") == 2          # head … middle … tail (PRD 31)
    assert memory.tokens_est(prompt) <= sources.RAW_CHUNK_TOKENS + 200   # + the small header


def test_cap_text_leaves_a_short_document_alone():
    assert sources.cap_text("short") == "short"


# --------------------------------------------------------------- 8. images


def test_a_pasted_png_goes_to_the_model_as_an_image_part(vault, fake):
    filed = sources.ingest_upload(png_bytes(), "screenshot.png", kind_hint="paste_image",
                                  hint="the Falcon dashboard")

    assert filed["raw"].startswith("raw/screens/") and filed["raw"].endswith(".png")
    assert "kind: screenshot" in read(filed["path"] + ".md")

    content = prompts(fake)[0]
    assert isinstance(content, list)
    image = next(part for part in content if part["type"] == "image_url")
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert "the Falcon dashboard" in content[0]["text"]


def test_an_image_over_the_cap_is_refused_before_anything_is_written(vault, fake):
    with pytest.raises(sources.IngestError) as exc:
        sources.ingest_upload(b"x" * (sources.MAX_IMAGE_BYTES + 1), "big.png")

    assert exc.value.status == 413
    assert raw_files() == [] and fake.calls == []


def test_an_unsupported_file_type_is_refused(vault, fake):
    with pytest.raises(sources.IngestError) as exc:
        sources.ingest_upload(b"MZ", "installer.exe")

    assert exc.value.status == 415
    assert raw_files() == []


# --------------------------------------------------------------- extraction units


def test_eml_extraction_keeps_the_headers_block(vault, fake):
    filed = sources.ingest_upload(EML.encode(), "note.eml")

    prompt = prompts(fake)[0]
    assert "From: Anna Kovacs <anna@example.com>" in prompt
    assert "Subject: Falcon go-live moved" in prompt
    assert "the go-live moved to September 15" in prompt
    assert "kind: email" in read(filed["path"] + ".md")


def test_ingest_path_reads_a_file_from_disk(vault, fake, tmp_path):
    doc = tmp_path / "falcon.md"
    doc.write_text("Falcon is amber.\n")

    filed = sources.ingest_path(str(doc), hint="status")

    assert filed["already"] is False
    assert "Falcon is amber." in read(filed["raw"])


def test_raw_files_are_never_listed_as_pages(vault, fake, monkeypatch):
    from tools import gmail, sources as tool

    monkeypatch.setattr(gmail, "LAST_RESULTS", {"18f0": {"from": "a", "subject": "s"}})
    monkeypatch.setattr(gmail, "fetch_message", lambda msg_id: "From: a\n\nbody")
    tool.file_source("email", "18f0")

    paths = [p["path"] for p in wiki.pages()]
    assert "raw/email/18f0" not in paths
    assert not any(p.startswith("raw/") for p in paths)
    assert wiki.lint() == []                      # the source page passes lint


# --------------------------------------------------------------- 0 & 5. the endpoint


@pytest.fixture
def http(vault, memory_sandbox, fake):
    """A TestClient with an open panel websocket, so `wiki` pushes can be observed."""
    chatbot.greeted = True
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        yield client, ws


def _post(client, **fields):
    files = fields.pop("files", None)
    return client.post("/ingest", data={"confirm": "true", "kind": "file", **fields}, files=files)


def test_ingest_without_confirm_is_refused(http):
    client, _ws = http
    response = client.post("/ingest", data={"kind": "file"},
                           files={"file": ("falcon.md", b"Falcon is amber.", "text/markdown")})

    assert response.status_code == 400
    assert wiki.pages() == []


def test_ingest_from_a_wrong_origin_is_refused(http):
    client, _ws = http
    response = client.post("/ingest", data={"confirm": "true"},
                           files={"file": ("falcon.md", b"x", "text/markdown")},
                           headers={"origin": "http://evil.example"})

    assert response.status_code == 403
    assert wiki.pages() == []


def test_ingest_files_the_upload_and_pushes_the_wiki(http):
    client, ws = http
    response = _post(client, hint="weekly status",
                     files={"file": ("falcon.md", b"Falcon is amber.", "text/markdown")})

    assert response.status_code == 200
    body = response.json()
    assert body["errors"] == []
    assert body["filed"] == [{"path": "sources/falcon-status",
                              "raw": body["filed"][0]["raw"], "already": False}]
    assert body["filed"][0]["raw"].startswith("raw/inbox/")

    pushed = ws.receive_json()
    while pushed.get("type") != "wiki":
        pushed = ws.receive_json()
    assert "sources/falcon-status" in [p["path"] for p in pushed["pages"]]


def test_ingest_of_the_same_bytes_reports_already(http):
    client, _ws = http
    files = {"file": ("falcon.md", b"Falcon is amber.", "text/markdown")}
    _post(client, files=files)
    again = _post(client, files={"file": ("falcon.md", b"Falcon is amber.", "text/markdown")})

    assert again.json()["filed"][0]["already"] is True


def test_ingest_refuses_an_oversized_file(http):
    client, _ws = http
    big = b"x" * (sources.MAX_TEXT_BYTES + 1)
    response = _post(client, files={"file": ("big.txt", big, "text/plain")})

    assert response.status_code == 413
    assert raw_files() == []


def test_ingest_refuses_an_unsupported_type(http):
    client, _ws = http
    response = _post(client, files={"file": ("installer.exe", b"MZ", "application/octet-stream")})

    assert response.status_code == 415


def test_ingest_of_pasted_text_needs_text(http):
    client, _ws = http
    assert client.post("/ingest", data={"confirm": "true", "kind": "paste_text",
                                        "text": "  "}).status_code == 400


def test_ingest_of_pasted_text_files_a_note(http):
    client, _ws = http
    response = client.post("/ingest", data={"confirm": "true", "kind": "paste_text",
                                            "text": "Falcon go-live moved to September 15"})

    assert response.status_code == 200
    assert response.json()["filed"][0]["already"] is False
    assert "kind: note" in read("sources/falcon-status.md")


def test_ingest_files_several_uploads_in_one_request(http, monkeypatch):
    client, _ws = http
    monkeypatch.setattr(sources, "_client_model",
                        lambda: (FakeClient(SUMMARY, "[]", SUMMARY, "[]"), "m"))
    response = client.post("/ingest", data={"confirm": "true", "kind": "file"}, files=[
        ("file", ("one.md", b"one", "text/markdown")),
        ("file", ("two.md", b"two", "text/markdown")),
    ])

    assert response.status_code == 200
    assert len(response.json()["filed"]) == 2
    assert len(raw_files()) == 2


# --------------------------------------------------------------- 10. the user is not a person
# PRD 26 scenario 5: a document naming the (mis-OCR'd) profile owner must not grow a people page.


USER_SUMMARY = json.dumps({
    "title": "Team screenshot",
    "summary": "Mihay Fodor and Elena agreed to ship on Friday.",
    "entities": [{"name": "Mihay Fodor", "type": "person"}, {"name": "Elena", "type": "person"}],
    "unreadable": False,
})

USER_OPS = json.dumps([
    {"op": "new_page", "path": "people/mihay-fodor", "type": "person", "name": "Mihay Fodor",
     "summary": "Ships on Fridays."},
    {"op": "new_page", "path": "people/elena", "type": "person", "name": "Elena",
     "summary": "Agreed to ship on Friday."},
    {"op": "link", "path": "people/elena", "target": "people/mihay-fodor"},
])


def test_a_document_naming_the_user_files_no_page_about_them(vault, monkeypatch, capsys):
    memory._write_profile(memory._normalise(
        {"identity": {"name": "Mihaly Fodor", "nickname": "Michael", "aliases": ["Misi"]}}))
    client = FakeClient(USER_SUMMARY, USER_OPS)
    monkeypatch.setattr(sources, "_client_model", lambda: (client, "m"))

    filed = sources.ingest_upload(b"Mihay Fodor and Elena met.", "screens.md")

    assert not wiki.page_exists("people/mihay-fodor.md")
    assert wiki.page_exists("people/elena.md")
    assert "[[people/mihay-fodor]]" not in read("people/elena.md")

    page = read(filed["path"] + ".md")
    assert "- the user (Mihay Fodor)" in page          # named, but never linked
    assert "[[people/elena]]" in page
    assert "is the user, not another person" in capsys.readouterr().out

    systems = [call["messages"][0]["content"] for call in client.calls]
    assert all("Mihaly Fodor, Michael, Misi" in system for system in systems)


def test_restrict_ops_rejects_pages_about_the_user(vault):
    names = ["Mihaly Fodor", "Michael", "Misi"]
    kept, rejected = sources.restrict_ops([
        {"op": "new_page", "path": "people/misi", "type": "person", "name": "Misi"},
        {"op": "new_page", "path": "people/mihay-fodor", "type": "person", "name": "Mihay Fodor"},
        {"op": "new_page", "path": "people/elena", "type": "person", "name": "Elena"},
        {"op": "link", "path": "people/elena", "target": "people/michael"},
    ], "raw/inbox/doc.md", names)

    assert [op["path"] for op in kept] == ["people/elena"]
    assert len(rejected) == 3
    assert all("is the user" in bad["reason"] for bad in rejected)
