"""PRD 47b scenarios 1-2 and 5 (service half): scripts/remarkable_server.py against the fake tablet."""
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from tests.conftest import ROOT, free_port

SERVER = os.path.join(ROOT, "scripts", "remarkable_server.py")
RENDER = pytest.mark.skipif(shutil.which("cupsfilter") is None or shutil.which("pdftotext") is None,
                            reason="cupsfilter / pdftotext are not installed")


def serve_push(device: str):
    """The push service on a free loopback port, its tablet at `device` -> ``(process, base url)``."""
    port = free_port()
    proc = subprocess.Popen([sys.executable, SERVER], env={**os.environ, "REMARKABLE_PORT": str(port),
                                                           "REMARKABLE_URL": device},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return proc, url
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("push service did not start")


@pytest.fixture
def push_service(fake_remarkable):
    proc, url = serve_push(fake_remarkable.url)
    yield url
    proc.kill()


def post(url: str, body: dict) -> tuple:
    req = urllib.request.Request(f"{url}/push", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


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
