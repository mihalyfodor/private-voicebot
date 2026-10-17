#!/usr/bin/env python3
"""Pull changed reMarkable notebooks into the vault over the USB web interface (PRD 47).

    scripts/remarkable_sync.py pull [--device URL] [--app URL] [--state PATH]

Stdlib only; run from the checkout's venv on the host (the tablet is a host USB interface, the
container is not). Lists the `$REMARKABLE_FOLDER` (`wiki-inbox`) folder on the tablet
(`--device`, `$REMARKABLE_URL`, default http://10.11.99.1), renders every notebook whose
`ModifiedClient` changed since the last pull to 150 dpi PNGs with `pdftoppm`, and POSTs the pages
to the app's `/ingest` as one `kind=notebook` item (`--app`, `$VOICEBOT_URL`, default
http://127.0.0.1:8010). Optional per-page OCR cross-check: `scripts/vision_ocr <png>` (or
`$REMARKABLE_OCR_BIN`) when executable, off with `REMARKABLE_OCR=0`. State: `--state`, default
~/.cache/voicebot/remarkable.json, written per notebook only after the app filed it. The trace profile
rule is the app's (docs/49-event-trace.md); each pull's notebooks share one `run_id`.
Exit 0 done (or tablet not connected), 1 the app was unreachable or a notebook failed. Never
writes to the tablet.
"""
import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from core import trace  # noqa: E402 — the repo's core/trace.py (stdlib only)
DEFAULT_STATE = os.path.join("~", ".cache", "voicebot", "remarkable.json")

#: every line `pull` prints, in one place
LINES = {
    "offline": "reMarkable not connected ({host})",
    "no_folder": 'no folder "{folder}" on the reMarkable',
    "nothing": "nothing new",
    "app_down": "voicebot app not reachable ({app}): {error}",
    "failed": 'failed "{name}": {error}',
    "done": '{verb} "{name}" ({pages} pages) -> {path}',
}


class AppDown(Exception):
    """The app did not answer at all: every further notebook would fail the same way."""


class Idle(Exception):
    """Nothing to pull this run (tablet absent, no folder, nothing changed): one line, exit 0."""


def _get(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _listing(device: str, ident: str = "") -> list:
    return json.loads(_get(f"{device}/documents/{ident}", 10))


# --------------------------------------------------------------------------- state


def load_state(path: str) -> dict:
    """The state file; missing or garbled is empty."""
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(path: str, state: dict) -> None:
    """Atomic: temp file in the same folder, then `os.replace`; a failed write leaves no temp."""
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=folder, prefix=".remarkable-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def record(state_path: str, state: dict, doc: dict, name: str, pages: int, sha: str) -> None:
    """One filed notebook -> its state entry, saved at once."""
    state[doc["ID"]] = {"modified": doc.get("ModifiedClient"), "pages": pages, "sha": sha, "name": name}
    save_state(state_path, state)


# --------------------------------------------------------------------------- list, download


def ensure_device(device: str) -> None:
    """`Idle` ("reMarkable not connected (host)") when the tablet does not answer."""
    try:
        _get(f"{device}/documents/", 1)
    except (OSError, ValueError) as exc:
        raise Idle(LINES["offline"].format(host=urllib.parse.urlparse(device).hostname or device)) from exc


def list_changed(device: str, state: dict) -> list:
    """The notebooks in `$REMARKABLE_FOLDER` whose `ModifiedClient` differs from `state`; `Idle`
    when the tablet does not answer, the folder is missing or nothing changed."""
    folder_name = os.getenv("REMARKABLE_FOLDER", "wiki-inbox")
    ensure_device(device)
    folder = next((d for d in _listing(device) if d.get("Type") == "CollectionType"
                   and d.get("VissibleName") == folder_name), None)
    if folder is None:
        raise Idle(LINES["no_folder"].format(folder=folder_name))
    todo = [d for d in _listing(device, folder["ID"])
            if d.get("Type") == "DocumentType" and d.get("fileType", "notebook") == "notebook"
            and d.get("ModifiedClient") != (state.get(d["ID"]) or {}).get("modified")]
    if not todo:
        raise Idle(LINES["nothing"])
    return todo


def download(device: str, ident: str) -> bytes:
    """The notebook as the tablet renders it: one PDF."""
    return _get(f"{device}/download/{ident}/pdf", 300)


# --------------------------------------------------------------------------- rasterise, OCR


def render_pages(pdf: bytes, folder: str) -> list:
    """`pdftoppm -r 150 -png` -> the page PNGs' bytes, in page order."""
    src = os.path.join(folder, "notebook.pdf")
    with open(src, "wb") as f:
        f.write(pdf)
    subprocess.run(["pdftoppm", "-r", "150", "-png", src, os.path.join(folder, "page")],
                   check=True, capture_output=True, timeout=300)
    names = [n for n in os.listdir(folder) if n.startswith("page") and n.endswith(".png")]
    names.sort(key=lambda n: int(n[len("page-"):-len(".png")]))
    out = []
    for name in names:
        with open(os.path.join(folder, name), "rb") as f:
            out.append(f.read())
    return out


def ocr_bin() -> str:
    """The OCR helper to call, or "" (disabled, missing or not executable)."""
    if os.getenv("REMARKABLE_OCR", "1") == "0":
        return ""
    path = os.getenv("REMARKABLE_OCR_BIN") or os.path.join(HERE, "vision_ocr")
    return path if os.path.isfile(path) and os.access(path, os.X_OK) else ""


def ocr_page(helper: str, png: bytes, folder: str, n: int) -> str:
    """The helper's stdout for one page; "" on any failure."""
    if not helper:
        return ""
    path = os.path.join(folder, f"ocr-{n:02d}.png")
    with open(path, "wb") as f:
        f.write(png)
    try:
        done = subprocess.run([helper, path], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def rasterise(pdf: bytes, helper: str) -> tuple:
    """``(page PNGs, one OCR string per page)``; `RuntimeError` when the PDF has no pages."""
    with tempfile.TemporaryDirectory() as tmp:
        pages = render_pages(pdf, tmp)
        if not pages:
            raise RuntimeError("the PDF rendered no pages")
        return pages, [ocr_page(helper, p, tmp, n) for n, p in enumerate(pages, 1)]


# --------------------------------------------------------------------------- post


def _multipart(fields: dict, files: list) -> tuple:
    boundary = uuid.uuid4().hex
    out = []
    for name, value in fields.items():
        out += [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode(),
                str(value).encode(), b"\r\n"]
    for filename, data, *ctype in files:
        out += [f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {ctype[0] if ctype else 'image/png'}\r\n\r\n".encode(), data, b"\r\n"]
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def post_notebook(app: str, fields: dict, pages: list) -> dict:
    """One `/ingest` request -> its `filed[0]` (with a `sha256`); raises `AppDown` or `RuntimeError`."""
    body, ctype = _multipart(fields, [(f"page-{i:02d}.png", p) for i, p in enumerate(pages, 1)])
    req = urllib.request.Request(f"{app}/ingest", data=body, method="POST",
                                 headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            reply = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:200]
        raise RuntimeError(f"app answered {exc.code}: {detail}") from exc
    except (urllib.error.URLError, ConnectionError) as exc:
        raise AppDown(str(getattr(exc, "reason", exc))) from exc
    if reply.get("errors") or not reply.get("filed"):
        raise RuntimeError("; ".join(reply.get("errors") or ["nothing filed"]))
    filed = reply["filed"][0]
    if not filed.get("sha256"):
        raise RuntimeError("the app reply carries no sha256")
    return filed


def notebook_fields(doc: dict, name: str, ocr: list, prev: dict) -> dict:
    fields = {"confirm": "true", "kind": "notebook", "title": name, "ocr": json.dumps(ocr),
              "device_id": doc["ID"]}
    if prev.get("sha"):
        fields["replaces"] = prev["sha"]       # the app ignores it when the pages are unchanged
    return fields


# --------------------------------------------------------------------------- pull


def pull(device: str, app: str, state_path: str, out=print) -> int:
    state = load_state(state_path)
    try:
        todo = list_changed(device, state)
    except Idle as idle:
        out(str(idle))
        return 0
    helper, failed = ocr_bin(), 0
    run_id = time.strftime("rm-%Y%m%d-%H%M%S")        # one run for this pull's ingests (PRD 49, page: PRD 51)
    for doc in todo:
        name, prev = doc.get("VissibleName") or doc["ID"], state.get(doc["ID"]) or {}
        try:
            pages, ocr = rasterise(download(device, doc["ID"]), helper)
            filed = post_notebook(app, {**notebook_fields(doc, name, ocr, prev), "run_id": run_id}, pages)
        except AppDown as exc:
            out(LINES["app_down"].format(app=app, error=exc))
            return 1
        except Exception as exc:  # noqa: BLE001 — one bad notebook must not stop the rest
            out(LINES["failed"].format(name=name, error=exc))
            failed += 1
            continue
        record(state_path, state, doc, name, len(pages), filed["sha256"])
        out(LINES["done"].format(verb="already" if filed.get("already") else "filed", name=name,
                                 pages=len(pages), path=filed.get("path", "")))
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pull changed reMarkable notebooks into the vault.")
    ap.add_argument("command", choices=["pull"])
    ap.add_argument("--device", default=os.getenv("REMARKABLE_URL", "http://10.11.99.1"),
                    help="tablet USB web interface (default $REMARKABLE_URL or http://10.11.99.1)")
    ap.add_argument("--app", default=os.getenv("VOICEBOT_URL", "http://127.0.0.1:8010"),
                    help="the voicebot app (default $VOICEBOT_URL or http://127.0.0.1:8010)")
    ap.add_argument("--state", default=DEFAULT_STATE, help=f"state file (default {DEFAULT_STATE})")
    args = ap.parse_args(argv)
    trace.configure(out=sys.stderr)                   # PRD 49: the app's profile rule
    return pull(args.device.rstrip("/"), args.app.rstrip("/"), os.path.expanduser(args.state))


if __name__ == "__main__":
    sys.exit(main())
