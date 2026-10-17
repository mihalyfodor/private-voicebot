"""POST /ingest: files, pasted text and images, notebooks and host-script runs (PRD 50 step 11, from chatbot.py)."""
import re
import asyncio
from core import trace
import threading
from memory import ops as m_ops
from memory import store as m_store
from ingest import notebooks
from ingest import pipeline
from fastapi import HTTPException, Request
from starlette.datastructures import UploadFile
from server import app as s_app
from server import panels
from server.actions import wiki
from server import ws

log = trace.logger("App")


INGEST_KINDS = ("file", "paste_text", "paste_image", "notebook")


def _push_ingest_progress(filename: str, stage: str, i: int, n: int) -> None:
    """Tell the intake popup which stage item `i` of `n` is in (PRD 36); no panel → no-op."""
    send_sync = panels._panel_sender()
    if send_sync is not None:
        send_sync({"type": "ingest_progress", "filename": filename, "stage": stage, "i": i, "n": n})


def _file_one(item: dict, on_stage=None) -> dict:
    """One popup item -> `sources`. Runs on a worker thread: it does IO, git and two LLM calls."""
    if item["kind"] == "notebook":          # PRD 47: all pages of one reMarkable notebook
        return notebooks.ingest_notebook(item["pages"], title=item["title"], ocr=item["ocr"],
                                         replaces=item["replaces"], device_id=item["device_id"],
                                         on_stage=on_stage)
    if item["kind"] == "paste_text":
        return pipeline.ingest_note(item["text"], title=item["title"], hint=item["hint"],
                                   page=item["page"], on_stage=on_stage)
    return pipeline.ingest_upload(item["data"], item["filename"], kind_hint=item["kind"],
                                 title=item["title"], hint=item["hint"], page=item["page"],
                                 on_stage=on_stage)


#: host-script run ids this process has seen: the first ingest of one opens its run (`run start`)
_host_runs: set = set()
_host_runs_lock = threading.Lock()


def _ingest_all(items: list, run_id: str = "") -> dict:
    """File every item of one popup; a failure on one is reported, the rest still land.

    Each stage of each item is pushed to the popup as `ingest_progress` (PRD 36); a host
    script's `run_id` groups the items' trace events under its run (PRD 49).
    """
    with trace.run(run_id):
        if run_id:
            with _host_runs_lock:
                fresh = run_id not in _host_runs
                _host_runs.add(run_id)
            if fresh:                       # its end is derived on the run page (PRD 51)
                trace.run_event("remarkable" if any(i["kind"] == "notebook" for i in items) else "ingest", "start")
        return _ingest_items(items)


def _ingest_items(items: list) -> dict:
    filed, errors = [], []
    for i, item in enumerate(items, 1):
        name = item.get("filename") or "text"
        try:
            filed.append(_file_one(
                item, lambda stage, name=name, i=i: _push_ingest_progress(name, stage, i, len(items))))
        except pipeline.IngestError as exc:
            errors.append(exc.message)
        except Exception:  # noqa: BLE001 — one bad drop must not 500 the popup
            # the ingest span already traced it (type + code); only the log gets the traceback
            trace.logger("Ingest").exception("filing a drop failed")
            errors.append(f"{item.get('filename') or 'the text'}: filing failed")
    return {"filed": filed, "errors": errors}


#: a host script's optional `run_id` form field (PRD 49): short, no spaces
RUN_ID = re.compile(r"[\w.:-]{1,64}")
#: pasted text travels as a plain form field; Starlette caps those at 1 MB unless told otherwise
INGEST_MAX_PART = pipeline.MAX_TEXT_BYTES + 64 * 1024


@s_app.app.post("/ingest")
async def ingest(request: Request):
    """The intake popup's File button — the only HTTP write path (docs/20-raw-ingestion.md).

    Nothing is filed without `confirm=true`: the popup is the only sender, and the guard is what
    keeps a future "helpful" auto-path from writing to the vault behind the user's back.
    The form is parsed by hand so a big pasted text is not refused by the framework's 1 MB
    per-part default before `sources` gets to apply its own cap. `kind=notebook` (PRD 47,
    scripts/remarkable_sync.py) is one item: every `file` part a page PNG, plus `ocr` (JSON list,
    one string per page), `replaces` (an earlier sha) and `device_id`.
    """
    if not ws._origin_allowed(request.headers.get("origin")):
        raise HTTPException(status_code=403, detail="bad origin")
    form = await request.form(max_part_size=INGEST_MAX_PART)
    confirm = str(form.get("confirm") or "")
    kind = str(form.get("kind") or "file")
    title, hint, page = (str(form.get(k) or "") for k in ("title", "hint", "page"))
    text = str(form.get("text") or "")
    run_id = str(form.get("run_id") or "")
    if run_id and not RUN_ID.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="bad run_id")
    file = [f for f in form.getlist("file") if isinstance(f, UploadFile)]
    if str(confirm).lower() != "true":
        raise HTTPException(status_code=400, detail="ingest needs an explicit confirm=true")
    if kind not in INGEST_KINDS:
        raise HTTPException(status_code=400, detail=f"unknown kind {kind!r}")

    fields = {"kind": kind, "title": title, "hint": hint, "page": page}
    items = []
    try:
        if kind == "notebook":
            pages = [await upload.read() for upload in file]
            ocr, replaces, device_id = (str(form.get(k) or "") for k in ("ocr", "replaces", "device_id"))
            checked = notebooks.check_request([u.filename for u in file], pages, ocr, replaces, device_id)
            items.append({**fields, **checked, "filename": title or "notebook"})
            file = []
        for upload in file or []:
            data = await upload.read()
            # refuse type/size here as well as in sources, so a bad drop costs no worker thread
            pipeline.check_upload(upload.filename or "", len(data), kind)
            items.append({**fields, "data": data, "filename": upload.filename or "drop",
                          "text": ""})
        if not items and kind == "paste_text":
            items.append({**fields, "data": b"", "filename": "",
                          "text": pipeline.check_note(text)})
    except pipeline.IngestError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from exc
    if not items:
        raise HTTPException(status_code=400, detail="nothing to file")

    before = m_ops.snapshot(m_store.load_profile())
    result = await asyncio.to_thread(_ingest_all, items, run_id)
    if any(not f.get("already") for f in result["filed"]):
        await wiki._send_wiki()
    if m_ops.snapshot(m_store.load_profile()) != before:
        await panels.send(panels._memory_msg(m_store.load_profile()))
    return result
