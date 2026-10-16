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
~/.cache/voicebot/remarkable.json, written per notebook only after the app filed it.
Exit 0 done (or tablet not connected), 1 the app was unreachable or a notebook failed. Never
writes to the tablet.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATE = os.path.join("~", ".cache", "voicebot", "remarkable.json")


class AppDown(Exception):
    """The app did not answer at all: every further notebook would fail the same way."""


def _get(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _listing(device: str, ident: str = "") -> list:
    return json.loads(_get(f"{device}/documents/{ident}", 10))


def load_state(path: str) -> dict:
    """The state file; missing or garbled is empty."""
    try:
        with open(path) as f:
            state = json.load(f)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(path: str, state: dict) -> None:
    """Atomic: temp file in the same folder, then rename."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".remarkable-")
    with os.fdopen(fd, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


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


def _multipart(fields: dict, files: list) -> tuple:
    boundary = uuid.uuid4().hex
    out = []
    for name, value in fields.items():
        out += [f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode(),
                str(value).encode(), b"\r\n"]
    for filename, data in files:
        out += [f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                "Content-Type: image/png\r\n\r\n".encode(), data, b"\r\n"]
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def post_notebook(app: str, fields: dict, pages: list) -> dict:
    """One `/ingest` request -> its `filed[0]`; raises `AppDown` or `RuntimeError`."""
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
    return reply["filed"][0]


def pull(device: str, app: str, state_path: str, out=print) -> int:
    folder_name = os.getenv("REMARKABLE_FOLDER", "wiki-inbox")
    host = urllib.parse.urlparse(device).hostname or device
    try:
        _get(f"{device}/documents/", 1)
    except (OSError, ValueError):
        out(f"reMarkable not connected ({host})")
        return 0
    folder = next((d for d in _listing(device) if d.get("Type") == "CollectionType"
                   and d.get("VissibleName") == folder_name), None)
    if folder is None:
        out(f'no folder "{folder_name}" on the reMarkable')
        return 0
    state = load_state(state_path)
    todo = [d for d in _listing(device, folder["ID"])
            if d.get("Type") == "DocumentType" and d.get("fileType", "notebook") == "notebook"
            and d.get("ModifiedClient") != (state.get(d["ID"]) or {}).get("modified")]
    if not todo:
        out("nothing new")
        return 0
    helper, failed = ocr_bin(), 0
    for doc in todo:
        name, prev = doc.get("VissibleName") or doc["ID"], state.get(doc["ID"]) or {}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pages = render_pages(_get(f"{device}/download/{doc['ID']}/pdf", 300), tmp)
                if not pages:
                    raise RuntimeError("the PDF rendered no pages")
                ocr = [ocr_page(helper, p, tmp, n) for n, p in enumerate(pages, 1)]
            fields = {"confirm": "true", "kind": "notebook", "title": name, "ocr": json.dumps(ocr),
                      "device_id": doc["ID"]}
            if prev.get("sha"):
                fields["replaces"] = prev["sha"]       # the app ignores it when the pages are unchanged
            filed = post_notebook(app, fields, pages)
            sha = filed.get("sha256")
            if not sha:
                raise RuntimeError("the app reply carries no sha256")
        except AppDown as exc:
            out(f"voicebot app not reachable ({app}): {exc}")
            return 1
        except Exception as exc:  # noqa: BLE001 — one bad notebook must not stop the rest
            out(f'failed "{name}": {exc}')
            failed += 1
            continue
        state[doc["ID"]] = {"modified": doc.get("ModifiedClient"), "pages": len(pages), "sha": sha, "name": name}
        save_state(state_path, state)
        verb = "already" if filed.get("already") else "filed"
        out(f'{verb} "{name}" ({len(pages)} pages) -> {filed.get("path", "")}')
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
    return pull(args.device.rstrip("/"), args.app.rstrip("/"), os.path.expanduser(args.state))


if __name__ == "__main__":
    sys.exit(main())
