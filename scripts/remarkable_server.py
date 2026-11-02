#!/usr/bin/env python3
"""Host reMarkable push service (PRD 47b): the container cannot reach USB, so it asks this.

    POST /push   JSON {title, markdown} (or {title, pdf: base64})  -> {ok, pages, name}
    GET  /health                                                   -> {ok: true}

Stdlib only; binds 127.0.0.1:$REMARKABLE_PORT (8021), no auth beyond loopback. `title\\n\\nbody` is
rendered as plain monospaced text with `cupsfilter`, then uploaded to the tablet's USB web interface
(`$REMARKABLE_URL`, default http://10.11.99.1). Errors are `{ok: false, error}`: 503 when the tablet is
unreachable, 400 on bad input, 502 when rendering or the upload failed. Nothing is kept between requests.
"""
import base64
import binascii
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import remarkable_sync as sync

# A PDF arrives base64 inside the JSON (4/3 overhead): 20 MB of body is a ~15 MB PDF, far above a note or a scan.
MAX_BODY = int(os.getenv("REMARKABLE_MAX_BODY_MB", "20")) * 1024 * 1024
HOST = "127.0.0.1"     # loopback only: Docker reaches it through host.docker.internal, the LAN does not


class PushError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def render(title: str, markdown: str) -> bytes:
    """`title`, a blank line and the markdown as monospaced text -> PDF bytes (`cupsfilter`)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "card.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{title}\n\n{markdown}\n")
        try:
            out = subprocess.run(["cupsfilter", "-m", "application/pdf", path], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PushError(502, f"could not render the PDF ({exc.__class__.__name__})") from exc
    if not out.stdout.startswith(b"%PDF"):
        raise PushError(502, "could not render the PDF")
    return out.stdout


def page_count(pdf: bytes) -> int:
    """Pages from `pdfinfo`, else the `/Type /Page` count."""
    try:
        info = subprocess.run(["pdfinfo", "-"], input=pdf, capture_output=True, timeout=30).stdout.decode()
        m = re.search(r"^Pages:\s+(\d+)", info, re.M)
        if m:
            return int(m.group(1))
    except (OSError, subprocess.TimeoutExpired):
        pass
    return max(1, len(re.findall(rb"/Type\s*/Page\b", pdf)))


def upload(device: str, name: str, pdf: bytes) -> None:
    try:
        sync.ensure_device(device)
    except sync.Idle as exc:
        raise PushError(503, str(exc)) from exc
    body, ctype = sync._multipart({}, [(name, pdf, "application/pdf")])
    req = urllib.request.Request(f"{device}/upload", data=body, method="POST", headers={"Content-Type": ctype})
    try:
        urllib.request.urlopen(req, timeout=120).read()
    except urllib.error.HTTPError as exc:
        raise PushError(502, f"the tablet refused the upload (HTTP {exc.code})") from exc
    except OSError as exc:
        raise PushError(503, sync.LINES["offline"].format(host=urllib.parse.urlparse(device).hostname or device)) from exc


def push(payload: dict, device: str) -> dict:
    title = re.sub(r'[\x00-\x1f/\\"]+', " ", str(payload.get("title") or "")).strip()
    if not title:
        raise PushError(400, "a title is required")
    if payload.get("pdf"):
        try:
            pdf = base64.b64decode(str(payload["pdf"]), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PushError(400, "pdf is not base64") from exc
        if not pdf.startswith(b"%PDF"):
            raise PushError(400, "pdf is not a PDF")
    else:
        markdown = str(payload.get("markdown") or "")
        if not markdown.strip():
            raise PushError(400, "nothing to send")
        pdf = render(title, markdown)
    name = f"{title}.pdf"
    upload(device, name, pdf)
    return {"ok": True, "pages": page_count(pdf), "name": name}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/push":
            return self._send(404, {"ok": False, "error": "not found"})
        # A browser page can reach loopback: no Origin (curl, the app server) and a JSON type
        # (not a no-preflight form/text post) or nothing is pushed.
        if self.headers.get("Origin"):
            return self._send(403, {"ok": False, "error": "browser requests are not accepted"})
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._send(415, {"ok": False, "error": "Content-Type must be application/json"})
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return self._send(411, {"ok": False, "error": "Content-Length required"})
        try:
            length = int(raw_length)
        except ValueError:
            length = -1
        if length < 0:
            return self._send(400, {"ok": False, "error": "bad Content-Length"})
        if length > MAX_BODY:
            return self._send(413, {"ok": False, "error": f"body too large (max {MAX_BODY // 1024 // 1024} MB)"})
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError
        except ValueError:
            return self._send(400, {"ok": False, "error": "body must be a JSON object"})
        try:
            self._send(200, push(payload, os.getenv("REMARKABLE_URL", "http://10.11.99.1")))
        except PushError as exc:
            self._send(exc.status, {"ok": False, "error": str(exc)})


if __name__ == "__main__":
    ThreadingHTTPServer((HOST, int(os.getenv("REMARKABLE_PORT", "8021"))), Handler).serve_forever()
