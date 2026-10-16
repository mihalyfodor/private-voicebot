"""Unit tests for raw ingestion — tmp vault, mocked LLM. Mirrors docs/20-raw-ingestion.md.

Scenario numbers are the PRD's test scenarios (0-8); 7b is the live harness script
eval/scripts/sources.yaml and 9 is manual.
"""
import os

import json
import struct
import zlib

import pytest
from fastapi.testclient import TestClient

import batch_import
import chatbot
import memory
import notebooks
import sources
import wiki
from tests.conftest import SENTINEL, FakeClient, commits, read, trace_lines, trace_text, vault, write_page  # noqa: F401 — fixture
from core import llm as core_llm

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def fake(monkeypatch):
    """A FakeClient wired in as the module's client: summary call first, page ops second."""
    client = FakeClient(SUMMARY, OPS)
    monkeypatch.setattr(core_llm, "_client", client)
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


def raw_files() -> list:
    """Every stored original, vault-relative (the manifest itself excluded)."""
    root = os.path.join(wiki.WIKI_PATH, "raw")
    found = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            rel = os.path.relpath(os.path.join(base, name), wiki.WIKI_PATH).replace(os.sep, "/")
            if not name.startswith(".") and rel != sources.MANIFEST and "/.index/" not in rel:
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

    monkeypatch.setattr(core_llm, "_client", FakeClient(SUMMARY, OPS))
    again = sources.ingest_upload(data, "falcon.md")

    assert again == {"path": first["path"], "raw": first["raw"], "already": True}
    assert (len(commits()), raw_files(), len(fake.calls)) == after_first
    assert len(sources.manifest_entries()) == 1


def test_a_second_document_with_the_same_name_gets_its_own_files(vault, fake, monkeypatch):
    monkeypatch.setattr(core_llm, "_client", FakeClient(SUMMARY, OPS, SUMMARY, "[]"))
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
    monkeypatch.setattr(core_llm, "_client", client)

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
                              "raw": body["filed"][0]["raw"], "already": False, "degraded": None}]
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
    monkeypatch.setattr(core_llm, "_client", FakeClient(SUMMARY, "[]", SUMMARY, "[]"))
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
    monkeypatch.setattr(core_llm, "_client", client)

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


def test_a_big_paste_gets_past_the_form_parser(http):
    """Pasted text is a plain form field; the framework's 1 MB per-part default must not refuse it."""
    client, _ws = http
    text = "Fallout 4 FAQ\n" + ("Q: Where is Dogmeat?\nA: Red Rocket.\n" * 60000)   # ~2 MB
    assert len(text.encode()) > 1024 * 1024
    response = _post(client, kind="paste_text", text=text)

    assert response.status_code == 200, response.text
    assert response.json()["filed"]


# --------------------------------------------------------------- PRD 36: progress pushes


def _drain_until(ws, kind: str) -> list:
    """Every message up to and including the first of type `kind`."""
    got = [ws.receive_json()]
    while got[-1].get("type") != kind:
        got.append(ws.receive_json())
    return got


def test_ingest_pushes_a_progress_message_per_stage_and_item(http, monkeypatch):
    """PRD 36 scenario 1: extracting -> summarising -> filing per item, all before the wiki push."""
    client, ws = http
    monkeypatch.setattr(core_llm, "_client", FakeClient(SUMMARY, "[]", SUMMARY, "[]"))
    response = client.post("/ingest", data={"confirm": "true", "kind": "file"}, files=[
        ("file", ("one.md", b"one", "text/markdown")),
        ("file", ("two.md", b"two", "text/markdown")),
    ])
    assert response.status_code == 200

    got = _drain_until(ws, "wiki")
    progress = [(m["filename"], m["stage"], m["i"], m["n"])
                for m in got if m["type"] == "ingest_progress"]
    assert progress == [("one.md", "extracting", 1, 2), ("one.md", "summarising", 1, 2),
                        ("one.md", "filing", 1, 2), ("two.md", "extracting", 2, 2),
                        ("two.md", "summarising", 2, 2), ("two.md", "filing", 2, 2)]


def test_an_already_filed_drop_pushes_no_progress(http):
    """PRD 36 scenario 1b: the sha short-circuit reports `already` and stays silent on the socket."""
    client, ws = http
    _post(client, files={"file": ("falcon.md", b"Falcon is amber.", "text/markdown")})
    _drain_until(ws, "sources")

    again = _post(client, files={"file": ("falcon.md", b"Falcon is amber.", "text/markdown")})
    assert again.json()["filed"][0]["already"] is True

    ws.send_json({"action": "get_sources"})        # the next frame is this reply, nothing in between
    assert ws.receive_json()["type"] == "sources"


# --------------------------------------------------------------------------- PRD 40 scenario 4: Pillow downscale


def test_downscale_shrinks_a_large_png_with_pillow(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    path = str(tmp_path / "big.png")
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (1800, 2400, 3), dtype=np.uint8), "RGB").save(path)
    assert os.path.getsize(path) > sources.DOWNSCALE_ABOVE_BYTES

    def no_sips(*a, **k):
        raise AssertionError("sips must not be called")
    monkeypatch.setattr(sources.subprocess, "run", no_sips)

    sources._downscale(path)
    with Image.open(path) as im:
        assert max(im.size) == sources.IMAGE_LONG_EDGE
        assert im.size[0] == sources.IMAGE_LONG_EDGE
        assert abs(im.size[1] - round(1800 * sources.IMAGE_LONG_EDGE / 2400)) <= 1


def test_downscale_leaves_small_images_alone(tmp_path, monkeypatch):
    from PIL import Image
    path = str(tmp_path / "small.png")
    Image.new("RGB", (2000, 1000), "red").save(path)      # solid colour: tiny file, big pixels
    assert os.path.getsize(path) <= sources.DOWNSCALE_ABOVE_BYTES

    def no_sips(*a, **k):
        raise AssertionError("sips must not be called")
    monkeypatch.setattr(sources.subprocess, "run", no_sips)

    before = open(path, "rb").read()
    sources._downscale(path)
    assert open(path, "rb").read() == before


# --------------------------------------------------------------- PRD 44: vault health


def test_forgetting_a_source_leaves_a_dead_source_ref_for_lint(vault, fake):
    filed = sources.ingest_upload(b"# Falcon\n\nFalcon is amber.\n", "falcon-status.md")
    assert not any("no longer exists" in f for f in wiki.lint())

    sources.forget(sources.sha256(b"# Falcon\n\nFalcon is amber.\n"))

    assert f"projects/falcon: source {filed['raw']} no longer exists" in wiki.lint()


def test_the_page_ops_prompt_lists_a_page_the_index_cut(vault, monkeypatch):
    for i in range(10):
        write_page(f"people/p{i}", f"Person number {i}", "\n## Summary\na colleague of the user\n")
    write_page("projects/zephyr", "Zephyr", "\n## Summary\nThe Zephyr rollout.\n")
    wiki.rebuild_index()
    monkeypatch.setattr(wiki, "WIKI_INDEX_TOKENS", 40)
    client = FakeClient("[]")

    monkeypatch.setattr(core_llm, "_client", client)
    sources.propose_page_ops("sources/zephyr-plan",
                             {"title": "Zephyr rollout", "summary": "Plan.", "entities": []}, "")

    user = client.calls[0]["messages"][1]["content"]
    index_part, block = user.split("Index:\n")[1].split("Existing related pages", 1)
    assert "projects/zephyr" not in index_part
    assert "(update these rather than duplicating):" in block and "projects/zephyr" in block


# --------------------------------------------------------------- PRD 48: batch import


def docx_file(text: str) -> bytes:
    import docx
    import io

    doc = docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def text_pdf(n: int) -> bytes:
    return pdf_bytes([[f"Falcon status note number {n}, amber, load test 940 ms p99"]])


@pytest.fixture
def export(tmp_path):
    """An export folder next to the vault, never inside it."""
    folder = tmp_path / "export"
    folder.mkdir()
    return folder


@pytest.fixture
def idle_port(monkeypatch):
    """Nothing answers on PORT (a dev machine may run the app on the default)."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = str(s.getsockname()[1])
    for name in ("PORT", "DEV_PORT", "PROD_PORT"):
        monkeypatch.setenv(name, port)


def put(folder, rel: str, data: bytes) -> None:
    path = folder / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def run_batch(folder, client=None, **kw) -> tuple:
    lines = []
    core_llm._client = client
    res = batch_import.ingest_batch(str(folder), out=lines.append, **kw)
    return res, lines


def batch_log(folder) -> list:
    return [json.loads(line) for line in (folder / "batch.log").read_text().splitlines()]


def three_files(export) -> None:
    put(export, "a.pdf", text_pdf(1))
    put(export, "b.pdf", text_pdf(2))
    put(export, "work/falcon/kickoff.docx", docx_file("Falcon kickoff notes, amber status."))


def test_a_folder_is_filed_through_the_drop_path(vault, idle_port, export):
    three_files(export)
    client = FakeClient(*[SUMMARY, OPS] * 3)
    before = len(commits())

    res, lines = run_batch(export, client)

    assert res["exit"] == 0 and res["filed"] == 3
    assert len(sources.manifest_entries()) == 3
    assert len(commits()) == before + 3
    assert [l.split()[0] for l in lines if l.startswith("[")] == ["[1/3]", "[2/3]", "[3/3]"]
    assert "filed 3 · skipped 0 · failed 0" in lines[-1]
    docx_entry = next(e for e in sources.manifest_entries() if e["kind"] == "docx")
    fm, _ = wiki.parse_page(read(docx_entry["page"] + ".md"))
    assert fm["source"] == f"batch import {memory._today_iso()}"
    sent = prompts(client)
    assert all("User hint: work/falcon" in p for p in sent[4:6])
    assert not any("User hint" in p for p in sent[:4])


def test_a_rerun_finds_everything_already_filed(vault, idle_port, export):
    three_files(export)
    run_batch(export, FakeClient(*[SUMMARY, OPS] * 3))
    before = commits()
    client = FakeClient()

    res, lines = run_batch(export, client)

    assert res["already"] == 3 and res["filed"] == 0 and res["exit"] == 0
    assert client.calls == [] and commits() == before
    assert sum(" already " in l for l in lines) == 3


def test_a_pdf_without_readable_text_needs_ocr(vault, idle_port, export):
    put(export, "ink.pdf", pdf_bytes([["ignored"]], with_text=False))
    put(export, "thin.pdf", pdf_bytes([["ten chars!!"]] * 5))      # ~11 chars a page: over the total floor, under the average
    client = FakeClient()

    res, lines = run_batch(export, client)

    assert res["needs-ocr"] == 2 and res["filed"] == 0 and client.calls == []
    assert raw_files() == [] and sources.manifest_entries() == []
    assert not [f for f in os.listdir(os.path.join(wiki.WIKI_PATH, "sources")) if f.endswith(".md")]
    assert sum(" needs-ocr " in l for l in lines) == 2


def test_failures_do_not_stop_the_batch_and_are_logged(vault, idle_port, export):
    put(export, "a.pdf", text_pdf(1))
    put(export, "b.pdf", text_pdf(2))
    put(export, "c.pdf", text_pdf(3))
    put(export, "d.docx", b"not a zip at all")
    client = FakeClient(SUMMARY, OPS, RuntimeError("model fell over"), SUMMARY, OPS)

    res, _lines = run_batch(export, client)

    assert res["exit"] == 1 and res["filed"] == 2 and res["failed"] == 2
    assert len(sources.manifest_entries()) == 2
    assert raw_files() == [f for f in raw_files() if any(
        e["path"] == f for e in sources.manifest_entries())]      # a failed file leaves no orphan original
    rows = [r for r in batch_log(export) if "rel" in r]
    assert [(r["rel"], r["state"]) for r in rows] == [
        ("a.pdf", "filed"), ("b.pdf", "failed"), ("c.pdf", "filed"), ("d.docx", "failed")]
    assert "model fell over" in rows[1]["reason"] and rows[3]["reason"]
    assert wiki._git("status", "--porcelain").stdout.strip() == ""


def test_resume_skips_what_failed_and_a_plain_rerun_retries_it(vault, idle_port, export):
    put(export, "a.pdf", text_pdf(1))
    put(export, "b.pdf", text_pdf(2))
    run_batch(export, FakeClient(SUMMARY, OPS, RuntimeError("boom")))

    client = FakeClient(SUMMARY, OPS)
    res, _ = run_batch(export, client, resume=True)
    assert res["filed"] == 0 and res["already"] == 1 and client.calls == []

    res, _ = run_batch(export, client)
    assert res["filed"] == 1 and res["failed"] == 0 and res["exit"] == 0
    assert len(sources.manifest_entries()) == 2


def test_a_dry_run_lists_states_and_writes_nothing(vault, idle_port, export, monkeypatch):
    three_files(export)
    put(export, "ink.pdf", pdf_bytes([["x"]], with_text=False))
    monkeypatch.setattr(core_llm, "_client", FakeClient(SUMMARY, OPS))
    sources.ingest_upload(text_pdf(1), "a.pdf")
    head, entries, raw = commits(), sources.manifest_entries(), raw_files()
    client = FakeClient()

    res, lines = run_batch(export, client, dry_run=True)

    assert client.calls == [] and commits() == head
    assert sources.manifest_entries() == entries and raw_files() == raw
    assert not (export / "batch.log").exists()
    states = " ".join(lines)
    assert "would file" in states and " already " in states and " needs-ocr " in states
    assert res["would file"] == 2 and res["filed"] == 0


def test_limit_sleep_unsupported_and_a_large_pdf(vault, idle_port, export, monkeypatch):
    for n in range(3):
        put(export, f"n{n}.pdf", text_pdf(n))
    put(export, ".hidden.pdf", text_pdf(9))

    res, _ = run_batch(export, FakeClient(*[SUMMARY, OPS] * 3), limit=1, sleep=0)
    assert res["filed"] == 1 and len(sources.manifest_entries()) == 1
    put(export, "page.one", b"x")
    put(export, "scan.png", b"x")

    naps = []
    monkeypatch.setattr(batch_import, "_sleep", naps.append)
    res, _ = run_batch(export, FakeClient(*[SUMMARY, OPS] * 3), sleep=2)
    assert res["filed"] == 2 and res["already"] == 1 and res["unsupported"] == 2
    assert naps == [2, 2]                    # none after the already-filed n0
    put(export, "zlast.pdf", text_pdf(7))
    run_batch(export, FakeClient(SUMMARY, OPS), sleep=2)
    assert naps == [2, 2]                    # none after the last file

    big = pdf_bytes([[("Falcon load test notes. " * 40) for _ in range(3300)]])
    assert len(big) > 2 * 1024 * 1024
    put(export, "zbig.pdf", big)
    res, _ = run_batch(export, FakeClient(SUMMARY, OPS))
    assert res["filed"] == 1 and res["failed"] == 0


def test_start_guards_refuse_a_dirty_vault_or_a_running_app(vault, idle_port, export, monkeypatch):
    import http.server
    import threading

    three_files(export)
    (vault / "stray.md").write_text("half filed")
    client = FakeClient()
    res, lines = run_batch(export, client)
    assert res["exit"] == 2 and client.calls == [] and "stray.md" in "\n".join(lines)
    assert sources.manifest_entries() == []
    (vault / "stray.md").unlink()

    server = http.server.HTTPServer(("127.0.0.1", 0), http.server.SimpleHTTPRequestHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("PROD_PORT", str(server.server_address[1]))     # `.env` says PORT=8011, prod is on 8010
    try:
        res, lines = run_batch(export, client)
    finally:
        server.shutdown()
    assert res["exit"] == 2 and client.calls == []
    assert "app is running — scripts/voicebot stop" in "\n".join(lines)


def test_the_script_points_the_vault_at_the_given_directory(tmp_path, export):
    import subprocess
    import sys

    target = tmp_path / "other-vault"
    target.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=target, check=True)
    (target / "stray.md").write_text("x")
    env = {**os.environ, "MEMORY_DIR": str(tmp_path / "mem"), "PORT": "1", "TRACE_DIR": str(tmp_path)}
    env.pop("WIKI_PATH", None)

    done = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ingest_batch.py"),
                           str(export), "--vault", str(target), "--dry-run"],
                          capture_output=True, text=True, env=env, cwd=tmp_path)

    assert done.returncode == 2
    assert str(target) in done.stdout and "stray.md" in done.stdout
    bad = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ingest_batch.py"),
                          str(tmp_path / "missing"), "--dry-run"],
                         capture_output=True, text=True, env=env, cwd=tmp_path)
    assert bad.returncode == 2


def test_a_strict_failure_removes_only_the_stored_original(vault, idle_port, monkeypatch):
    (vault / "notes.md").write_text("unrelated edit")
    wiki._commit("unrelated")
    (vault / "scratch.md").write_text("written during the run")

    def boom(*a, **kw):
        raise RuntimeError("model fell over")

    monkeypatch.setattr(sources, "summarise", boom)
    with pytest.raises(RuntimeError):
        monkeypatch.setattr(core_llm, "_client", FakeClient())
        sources.ingest_upload(text_pdf(1), "a.pdf", strict=True)

    assert (vault / "scratch.md").exists() and (vault / "notes.md").exists()
    assert raw_files() == [] and sources.manifest_entries() == []


@pytest.mark.parametrize("reply", ["garbage", json.dumps({"title": "T", "summary": "x", "entities": [],
                                                           "unreadable": True})])
def test_strict_refuses_an_unparseable_or_unreadable_summary(vault, idle_port, export, reply):
    put(export, "a.pdf", text_pdf(1))

    res, lines = run_batch(export, FakeClient(reply, reply, OPS))

    assert res["failed"] == 1 and res["exit"] == 1 and res["filed"] == 0
    assert sources.manifest_entries() == [] and raw_files() == []
    assert not [f for f in os.listdir(os.path.join(wiki.WIKI_PATH, "sources")) if f.endswith(".md")]


# --------------------------------------------------------------------------- PRD 47: reMarkable notebooks

NOTE_TEXT = "Standup: Anna ships the Falcon load test on Friday."


def notebook_pages(n: int) -> list:
    return [png_bytes(4 + i, 4) for i in range(n)]       # distinct bytes per page


def notebook_summary(title: str = "Standup") -> str:
    return json.dumps({"title": title, "summary": f"{title}: handwritten notes.", "entities": [],
                       "topic": "work", "unreadable": False})


def notebook_client(pages: int, ops: str = "[]") -> FakeClient:
    """One readable transcription per page, then the summary and the page ops."""
    return FakeClient(*[NOTE_TEXT] * pages, notebook_summary(), ops)


def file_notebook(pages: list, client, **kw) -> dict:
    core_llm._client = client
    return notebooks.ingest_notebook(pages, title=kw.pop("title", "Standup"), **kw)


def test_a_notebook_is_transcribed_page_by_page_and_filed_once(vault):
    client = FakeClient(NOTE_TEXT, NOTE_TEXT + " Two.", NOTE_TEXT + " Three.", notebook_summary(), OPS)
    before = len(commits())
    stages = []
    filed = file_notebook(notebook_pages(3), client, ocr=["anna ships", "", "three"], device_id="n1",
                          on_stage=stages.append)

    calls = client.calls[:3]
    for call in calls:
        parts = call["messages"][1]["content"]
        assert [p["type"] for p in parts].count("image_url") == 1
        assert call["max_tokens"] == 600
    assert "transcribing 2/3" in stages
    stem = filed["raw"][:-len(".md")]
    assert filed["raw"].startswith("raw/remarkable/") and filed["raw"].endswith("-standup.md")
    pngs = [f"{stem}/page-0{i}.png" for i in (1, 2, 3)]
    assert [r for r in raw_files() if r.endswith(".png")] == pngs
    transcript = read(filed["raw"])
    for i in (1, 2, 3):
        assert f"## Page {i}" in transcript
    assert "> Apple Vision OCR:" in transcript and "anna ships" in transcript
    assert transcript.count("> Apple Vision OCR:") == 2                 # the empty entry adds no block
    fm, _ = wiki.parse_page(read(filed["path"] + ".md"))
    assert fm["kind"] == "notebook" and fm["source"].startswith("remarkable ")
    assert len(commits()) == before + 1
    entry = sources.manifest_find(filed["sha256"])
    assert entry["pages"] == pngs and entry["device_id"] == "n1" and entry["path"] == filed["raw"]
    assert filed["sha256"] == sources.sha256(b"".join(notebook_pages(3)))


def test_the_same_pages_again_are_already_filed_without_a_model_call(vault):
    file_notebook(notebook_pages(2), notebook_client(2, OPS))
    again = FakeClient()
    res = file_notebook(notebook_pages(2), again)

    assert res["already"] is True and again.calls == []


def test_a_short_page_is_marked_unreadable(vault):
    filed = file_notebook(notebook_pages(2), FakeClient(NOTE_TEXT, "ok", notebook_summary(), OPS))

    assert "[page 2: unreadable]" in read(filed["raw"])
    assert "ok" not in read(filed["raw"]).split("## Page 2", 1)[1]


def test_an_all_unreadable_notebook_files_its_page_without_ops(vault):
    client = FakeClient("ok", "", notebook_summary(), OPS)
    filed = file_notebook(notebook_pages(2), client)

    assert len(client.calls) == 2                       # the two transcriptions; no summary, no ops
    assert read(filed["path"] + ".md")
    assert not wiki.page_exists("projects/falcon")
    assert "[page 1: unreadable]" in read(filed["raw"]) and "[page 2: unreadable]" in read(filed["raw"])


def test_replaces_forgets_the_earlier_source_and_its_raw_files(vault):
    old = file_notebook(notebook_pages(2), notebook_client(2))
    old_pngs = sources.manifest_find(old["sha256"])["pages"]
    new = file_notebook(notebook_pages(3), notebook_client(3),
                        replaces=old["sha256"])

    assert sources.manifest_find(old["sha256"]) is None
    assert old["path"] != new["path"] and not read(old["path"] + ".md")
    for rel in [old["raw"], *old_pngs]:
        assert rel not in raw_files()
    assert new["raw"] in raw_files() and len([r for r in raw_files() if r.endswith(".png")]) == 3
    assert read(new["path"] + ".md")


def test_a_long_notebook_is_sampled_head_middle_tail(vault):
    page = ("Falcon load test notes, written by hand. " * 30).strip()      # ~1200 chars a page
    client = FakeClient(*[page] * 40, notebook_summary(), "[]")
    file_notebook(notebook_pages(40), client)

    summary_prompt = client.calls[40]["messages"][1]["content"]
    assert summary_prompt.count("tokens skipped") == 2


def test_more_than_50_pages_is_413_before_any_model_call(vault):
    client = FakeClient()
    with pytest.raises(sources.IngestError) as exc:
        file_notebook(notebook_pages(51), client)

    assert exc.value.status == 413 and client.calls == [] and raw_files() == []


def test_ingest_endpoint_files_a_notebook(http, monkeypatch):
    client, _ws = http
    fake = notebook_client(2)
    monkeypatch.setattr(core_llm, "_client", fake)
    pages = notebook_pages(2)
    files = [("file", (f"page-0{i}.png", data, "image/png")) for i, data in enumerate(pages, 1)]
    response = _post(client, kind="notebook", title="Standup", ocr=json.dumps(["", ""]),
                     device_id="n1", files=files)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["errors"] == [] and body["filed"][0]["already"] is False
    assert body["filed"][0]["sha256"] == sources.sha256(b"".join(pages))
    assert len([r for r in raw_files() if r.endswith(".png")]) == 2


def test_ingest_endpoint_refuses_a_notebook_page_that_is_not_png(http):
    client, _ws = http
    response = _post(client, kind="notebook", title="Standup",
                     files=[("file", ("page-01.jpg", png_bytes(), "image/jpeg"))])

    assert response.status_code == 415 and raw_files() == []


def test_a_failed_replace_is_retried_when_the_same_pages_come_again(vault, monkeypatch):
    v1 = file_notebook(notebook_pages(2), notebook_client(2), device_id="n1")
    real_forget, calls = sources.forget, []

    def flaky(digest):
        calls.append(digest)
        if len(calls) == 1:
            raise RuntimeError("git fell over")
        return real_forget(digest)

    monkeypatch.setattr(sources, "forget", flaky)
    v2_pages = notebook_pages(3)
    with pytest.raises(RuntimeError):
        file_notebook(v2_pages, notebook_client(3),
                      device_id="n1", replaces=v1["sha256"])
    assert sources.manifest_find(v1["sha256"]) is not None

    again = FakeClient()
    res = file_notebook(v2_pages, again, device_id="n1", replaces=v1["sha256"])
    assert res["already"] is True and again.calls == []
    assert [e["sha256"] for e in sources.manifest_entries()] == [sources.sha256(b"".join(v2_pages))]


def test_replaces_must_name_a_notebook_from_the_same_device(vault, monkeypatch):
    monkeypatch.setattr(core_llm, "_client", FakeClient(SUMMARY, "[]"))
    drop = sources.ingest_upload(b"# Falcon\n\nFalcon is amber.\n", "falcon.md")
    sha = sources.manifest_entries()[0]["sha256"]
    other = file_notebook(notebook_pages(1), notebook_client(1), device_id="n2")
    for bad in (sha, other["sha256"], "f" * 64):
        client = FakeClient()
        with pytest.raises(sources.IngestError) as exc:
            file_notebook(notebook_pages(2), client, device_id="n1", replaces=bad)
        assert exc.value.status == 400 and client.calls == []
    assert read(drop["path"] + ".md") and sources.manifest_find(other["sha256"])


def test_ingest_endpoint_refuses_replacing_a_dropped_file(http, monkeypatch):
    client, _ws = http
    drop = _post(client, files={"file": ("falcon.md", b"Falcon is amber.", "text/markdown")}).json()["filed"][0]
    sha = sources.manifest_entries()[0]["sha256"]
    response = _post(client, kind="notebook", title="Standup", device_id="n1", replaces=sha,
                     files=[("file", ("page-01.png", png_bytes(), "image/png"))])

    assert response.status_code == 400 and "replaces" in response.json()["detail"]
    assert read(drop["path"] + ".md") and sources.manifest_find(sha)


def test_a_notebook_page_without_the_png_magic_is_415(vault):
    with pytest.raises(sources.IngestError) as exc:
        file_notebook([b"GIF89a not a png"], FakeClient())
    assert exc.value.status == 415


# --------------------------------------------------------------------------- PRD 49 scenario 4: the ingest trace


def test_a_failed_ops_completion_is_a_degraded_ingest_with_one_error(vault, monkeypatch):
    monkeypatch.setattr(core_llm, "_client", FakeClient(SUMMARY, RuntimeError("ops model down")))
    res = sources.ingest_note("Falcon is amber; Anna runs the load test.", title="Falcon status")
    (start,) = trace_lines(kind="ingest_start")
    (end,) = trace_lines(kind="ingest_end")
    (err,) = trace_lines(kind="error")
    assert start["ingest_kind"] == "note" and start["raw"].startswith("raw/inbox/") and start["bytes"] > 0
    assert end["degraded"] is True and end["ops_applied"] == 0 and end["status"] == "ok"
    assert end["path"] == res["path"] and end["pages"] == [res["path"]]
    assert err["exc_type"] == "RuntimeError" and err["traceback"] and err["scope"] == "ingest"
    assert start["ingest_id"] == end["ingest_id"] == err["ingest_id"]
    assert start["span_id"] == end["span_id"] and start["run_id"] == end["run_id"] == err["run_id"]
    assert "amber" not in trace_text()


def test_a_degraded_ingest_says_so_in_its_result(vault, monkeypatch):
    monkeypatch.setattr(core_llm, "_client", FakeClient(SUMMARY, RuntimeError("ops model down")))
    res = sources.ingest_note("Falcon is amber; Anna runs the load test.", title="Falcon status")
    assert res["degraded"] == "ops"
    monkeypatch.setattr(core_llm, "_client", FakeClient(RuntimeError("down"), RuntimeError("down")))
    summary = sources.ingest_note("Charlie is red.", title="Charlie status")
    assert summary["degraded"] == "summary"
    monkeypatch.setattr(core_llm, "_client", FakeClient(SUMMARY, "[]"))
    ok = sources.ingest_note("Bravo is green.", title="Bravo status")
    assert ok["degraded"] is None


def test_an_extractor_fallback_warns_with_the_kind_only(capsys, monkeypatch):
    import pypdf

    def boom(*a, **k):
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr(pypdf, "PdfReader", boom)
    assert sources.pdf_text(b"%PDF-1.4") == ""
    out = capsys.readouterr().out
    assert "pdf extraction failed" in out and SENTINEL not in out


def test_ingest_run_id_from_the_form_groups_a_host_scripts_ingests(http):
    client, _ws = http
    for name in ("a.md", "b.md"):
        assert _post(client, run_id="sync-7", files={"file": (name, f"{name} body".encode(), "text/markdown")}
                     ).status_code == 200
    ends = trace_lines(kind="ingest_end")
    assert len(ends) == 2 and {e["run_id"] for e in ends} == {"sync-7"}
    assert len({e["ingest_id"] for e in ends}) == 2
    assert all(e["pages"] and all(isinstance(p, str) for p in e["pages"]) for e in ends)


def test_a_rejected_op_is_logged_by_kind_and_reason_code_only(vault, monkeypatch):
    ops = json.dumps([{"op": "new_page", "path": f"secrets/{SENTINEL.lower()}-plans", "type": "topic",
                       "name": f"{SENTINEL} plans", "summary": f"{SENTINEL} said this in the doc."}])
    monkeypatch.setattr(core_llm, "_client", FakeClient(SUMMARY, ops))
    sources.ingest_note("Falcon is amber.", title="Falcon status")
    (end,) = trace_lines(kind="ingest_end")
    assert end["ops_rejected"] == [{"kind": "new_page", "reason_code": "bad_path"}]
    text = trace_text()
    assert SENTINEL not in text and SENTINEL.lower() not in text
