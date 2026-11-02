"""PRD 47b scenarios 1-2 and 5 (service half): scripts/remarkable_server.py against the fake tablet."""
import base64
import http.client
import json
import os
import subprocess
import urllib.error
import urllib.request

import pytest

from tests.conftest import ROOT, free_port
from tests.prod.conftest import RENDER, serve_push

SERVER = os.path.join(ROOT, "scripts", "remarkable_server.py")


def post(url: str, body: dict) -> tuple:
    req = urllib.request.Request(f"{url}/push", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_push_refuses_browsers_and_non_json(push_service):
    body = json.dumps({"title": "x", "markdown": "x"}).encode()
    for headers, code in (({"Content-Type": "application/json", "Origin": "http://evil.example"}, 403),
                          ({"Content-Type": "text/plain"}, 415)):
        req = urllib.request.Request(f"{push_service}/push", data=body, method="POST", headers=headers)
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req, timeout=10)
        assert err.value.code == code


def _raw_post(push_service, length=None, origin=None):
    host, port = push_service.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=10)
    conn.putrequest("POST", "/push")
    conn.putheader("Content-Type", "application/json")
    if origin:
        conn.putheader("Origin", origin)
    if length is not None:
        conn.putheader("Content-Length", length)
    conn.endheaders(b'{"title": "x", "markdown": "x"}')
    resp = conn.getresponse()
    reply = json.loads(resp.read())
    conn.close()
    return resp.status, reply


def test_push_refuses_oversized_or_unsized_bodies_without_reading(push_service, fake_remarkable):
    status, reply = _raw_post(push_service, "999999999999")
    assert status == 413 and reply["ok"] is False and "body too large (max 20 MB)" in reply["error"]
    status, reply = _raw_post(push_service, "abc")
    assert status == 400 and reply == {"ok": False, "error": "bad Content-Length"}
    status, reply = _raw_post(push_service, "-5")
    assert status == 400 and reply["error"] == "bad Content-Length"
    status, reply = _raw_post(push_service)
    assert status == 411 and reply == {"ok": False, "error": "Content-Length required"}
    # check order: a browser Origin is refused before the length is looked at
    status, _ = _raw_post(push_service, "999999999999", origin="http://evil.example")
    assert status == 403
    assert fake_remarkable.uploads == []


def test_health(push_service):
    assert json.loads(urllib.request.urlopen(f"{push_service}/health").read()) == {"ok": True}


@RENDER
def test_push_uploads_one_pdf_named_for_the_title(push_service, fake_remarkable):
    status, reply = post(push_service, {"title": "Plan", "markdown": "# Plan\n\nstep one\nstep two"})
    assert status == 200 and reply["ok"] is True and reply["pages"] == 1 and reply["name"] == "Plan.pdf"
    (up,) = fake_remarkable.uploads
    assert up["filename"] == "Plan.pdf" and up["content_type"] == "application/pdf"
    assert up["bytes"].startswith(b"%PDF")


@RENDER
def test_a_long_markdown_renders_to_several_pages(push_service, fake_remarkable):
    body = "\n".join(f"line {i}" for i in range(150))
    status, reply = post(push_service, {"title": "Long", "markdown": body})
    assert status == 200 and reply["pages"] >= 3
    pdf = fake_remarkable.uploads[0]["bytes"]
    out = subprocess.run(["pdftotext", "-", "-"], input=pdf, capture_output=True).stdout.decode()
    assert "Long" in out and "line 149" in out


@RENDER
def test_a_ready_made_pdf_is_uploaded_as_is(push_service, fake_remarkable):
    pdf = b"%PDF-1.4 fake"
    status, reply = post(push_service, {"title": "Mine", "pdf": base64.b64encode(pdf).decode()})
    assert status == 200 and fake_remarkable.uploads[0]["bytes"] == pdf


def test_bad_input_is_400_and_uploads_nothing(push_service, fake_remarkable):
    assert post(push_service, {"title": "Empty", "markdown": "  \n"})[0] == 400
    assert post(push_service, {"markdown": "x"})[0] == 400
    assert fake_remarkable.uploads == []


@RENDER
def test_an_unreachable_tablet_is_503_not_connected():
    proc, url = serve_push(f"http://127.0.0.1:{free_port()}")
    try:
        status, reply = post(url, {"title": "Plan", "markdown": "# Plan\nx"})
    finally:
        proc.kill()
    assert status == 503 and reply["ok"] is False and reply["error"].startswith("reMarkable not connected")
