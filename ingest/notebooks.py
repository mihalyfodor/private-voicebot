"""reMarkable notebooks — page PNGs in, one transcript source out (PRD 47).

The sender is scripts/remarkable_sync.py via `POST /ingest kind=notebook`. The digest is the
pages' bytes in order, checked before any model call; each page gets one transcription
completion outside the lock; the transcript `.md` then takes the text path of
`pipeline.file_bytes`, the PNGs stored beside it. `replaces` (an earlier notebook's sha, same
`device_id`) is forgotten after the new filing.

See docs/47-remarkable-pull.md.
"""
import json
import os
import tempfile

from ingest import remarkable
from core import formats
from vault import manifest
from ingest import pipeline
from core import llm as core_llm
from core import trace
from ingest.pipeline import IngestError
from core.untrusted import DOCUMENT_RULE
from core import util

#: PNG pages, each under `pipeline.MAX_IMAGE_BYTES`
MAX_NOTEBOOK_PAGES = 50
TRANSCRIBE_TOKENS = 600
TRANSCRIBE_PROMPT = ("Transcribe any text visible on this handwritten page as markdown; `[?]` for illegible "
                     "words; reply with the text only.")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def check_page(filename: str, size: int) -> None:
    """`IngestError` unless one page is a `.png` within `pipeline.MAX_IMAGE_BYTES`."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext != ".png":
        raise IngestError(415, f"a notebook page must be a PNG, not {ext or filename!r}")
    pipeline.check_upload(filename, size)


def check_notebook(pages: list) -> None:
    """`IngestError` unless 1..`MAX_NOTEBOOK_PAGES` PNG pages each fit `MAX_IMAGE_BYTES`."""
    if not pages:
        raise IngestError(400, "nothing to file: the notebook has no pages")
    if len(pages) > MAX_NOTEBOOK_PAGES:
        raise IngestError(413, f"a notebook is at most {MAX_NOTEBOOK_PAGES} pages, not {len(pages)}")
    for i, png in enumerate(pages, 1):
        check_page(f"page-{i:02d}.png", len(png))
        if not png.startswith(PNG_MAGIC):
            raise IngestError(415, f"notebook page {i} is not a PNG")


def check_replaces(replaces: str, device_id: str) -> None:
    """`IngestError` 400 unless `replaces` is a filed notebook from the same `device_id`."""
    entry = manifest.manifest_find(replaces)
    if entry is None:
        raise IngestError(400, f"replaces: no filed source with sha {replaces[:12]}")
    if entry.get("kind") != "notebook" or entry.get("device_id", "") != device_id:
        raise IngestError(400, f"replaces: {replaces[:12]} is not a notebook from this device")


def check_request(names: list, pages: list, ocr: str, replaces: str, device_id: str) -> dict:
    """The `/ingest` gate for one notebook: every refusal before a worker thread is spent.

    Returns the item's ``{pages, ocr, replaces, device_id}``; `ocr` arrives as the JSON text of
    a list of strings. A `replaces` is checked only when it would be acted on (new pages).
    """
    for name, data in zip(names, pages):
        check_page(name or "", len(data))
    check_notebook(pages)
    digest = manifest.sha256(b"".join(pages))
    if replaces and replaces != digest and not manifest.manifest_find(digest):
        check_replaces(replaces, device_id)      # refused here: 400, nothing forgotten
    try:
        texts = json.loads(ocr or "[]")
    except ValueError:
        texts = None
    if not isinstance(texts, list) or not all(isinstance(x, str) for x in texts):
        raise IngestError(400, "ocr must be a JSON list of strings")
    return {"pages": pages, "ocr": texts, "replaces": replaces, "device_id": device_id}


def _transcribe(png: bytes, folder: str, n: int) -> str:
    """One page -> its markdown, or `[page N: unreadable]` under `MIN_PDF_CHARS`."""
    full = os.path.join(folder, f"page-{n:02d}.png")
    with open(full, "wb") as f:
        f.write(png)
    pipeline._downscale(full)
    text = core_llm.complete(DOCUMENT_RULE,
                             [{"type": "text", "text": TRANSCRIBE_PROMPT}, pipeline._image_part(full)],
                             TRANSCRIBE_TOKENS, purpose="ingest").strip()
    return text if len(text) >= formats.MIN_PDF_CHARS else f"[page {n}: unreadable]"


def _ocr_block(text: str) -> str:
    lines = [line.rstrip() for line in (text or "").strip().splitlines()]
    return "\n".join(["> Apple Vision OCR:", *(f"> {line}" if line else ">" for line in lines)])


def _transcript(texts: list, ocr: list) -> str:
    """`## Page N` per page, plus an Apple Vision block where `ocr` has text for it."""
    blocks = []
    for n, text in enumerate(texts, 1):
        extra = ocr[n - 1] if n <= len(ocr) else ""
        blocks.append(f"## Page {n}\n\n{text}" + (f"\n\n{_ocr_block(extra)}" if str(extra).strip() else ""))
    return "\n\n".join(blocks) + "\n"


def _skip_step(title: str, pages: list) -> None:
    """The run page's Did-not row for a notebook's unreadable pages (docs/51-run-cards.md): a
    `skip` step naming the page numbers, never their text."""
    detail = "; ".join(f"page {n}: unreadable" for n in pages)
    if len(detail) > trace.CAP:
        detail = f"{len(pages)} pages unreadable"
    with trace.span("step", name="skip", count=len(pages), detail=detail, reason="unreadable", title=title) as step:
        step["status"] = "skipped"


def ingest_notebook(pages: list, *, title: str, ocr=None, replaces: str = "", device_id: str = "",
                    on_stage=None) -> dict:
    """Page PNGs in order -> one transcript source; `pipeline.file_bytes`'s result plus `sha256`.

    `ocr` (one string per page, "" = none) adds an Apple Vision block per page. `replaces` is
    forgotten after the new filing — and on an `already` filing too, so a forget that failed is
    retried by the next pull; there an sha already gone is done, not an error.
    """
    check_notebook(pages)
    with trace.ingest("notebook", bytes=sum(len(p) for p in pages), pages=len(pages)):
        digest = manifest.sha256(b"".join(pages))
        replaces = "" if replaces == digest else replaces
        seen = manifest.manifest_find(digest)
        if seen:
            if replaces and manifest.manifest_find(replaces):
                check_replaces(replaces, device_id)
                pipeline.forget(replaces)
            trace.mark(already=True, path=seen.get("page", ""))
            return {"path": seen.get("page", ""), "raw": seen.get("path", ""), "already": True, "sha256": digest}
        if replaces:
            check_replaces(replaces, device_id)
        texts = []
        with tempfile.TemporaryDirectory() as folder:
            for n, png in enumerate(pages, 1):
                pipeline._stage(on_stage, f"transcribing {n}/{len(pages)}")
                texts.append(_transcribe(png, folder, n))
        bad = [n for n, t in enumerate(texts, 1) if t == f"[page {n}: unreadable]"]
        if bad:
            _skip_step(title, bad)
        unreadable = len(bad) == len(texts)
        raw_rel = f"raw/remarkable/{pipeline._dated(title or 'notebook')}.md"
        res = pipeline.file_bytes(_transcript(texts, list(ocr or [])).encode(), raw_rel, "notebook",
                                 title=title, on_stage=on_stage,
                                 source=f"remarkable {util.today_iso()}", digest=digest, pages=pages,
                                 unreadable=unreadable, extra={"device_id": device_id} if device_id else None)
        if replaces and manifest.manifest_find(replaces):
            pipeline.forget(replaces)
        return {**res, "sha256": digest}


# --- push (PRD 47b, docs/52-approvals.md) ----------------
#: the local page estimate of a pushed card: lines / PUSH_LINES_PER_PAGE, at least one
PUSH_LINES_PER_PAGE = 60


def _card_title(card: dict) -> str:
    return str(card.get("title") or "Card")


def push_plan(card: dict) -> dict:
    """What "send that to my reMarkable" would upload, read-only: `{card_id, title, pages}`
    (`pages` is the local estimate, docs/47b-remarkable-push.md)."""
    lines = len(str(card.get("markdown") or "").splitlines())
    return {"card_id": str(card["id"]), "title": _card_title(card),
            "pages": max(1, -(-lines // PUSH_LINES_PER_PAGE))}


def push_available() -> bool:
    """Is the push service configured (`$REMARKABLE_PUSH_URL`)? Unset: the command is not offered."""
    return bool(remarkable.URL)


def push(card: dict) -> dict:
    """Upload `card` to the tablet through the host push service: its `{ok, pages, name}`."""
    return remarkable.push(_card_title(card), str(card["markdown"]))
