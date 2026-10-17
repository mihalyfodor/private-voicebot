"""Batch import — file a folder of PDF/DOCX (a OneNote export) through the drop path (PRD 48).

One file at a time through `pipeline.ingest_upload(..., strict=True)`: the manifest's sha guard
makes a re-run a no-op, `<folder>/batch.log` remembers skips and failures for `resume`. The CLI
is scripts/ingest_batch.py; the app must be stopped (see `_refusal`).

See docs/48-batch-import.md.
"""
import contextlib
import io
import json
import os
import subprocess
import time

import memory
from core import formats
from vault import manifest
from ingest import pipeline
from core import trace
import wiki

BATCH_LOG = "batch.log"
_sleep = time.sleep
_RETRY_STATES = ("needs-ocr", "unsupported", "too-large", "failed")
_STATES = ("filed", "would file", "already", "needs-ocr", "unsupported", "too-large", "resume", "failed")
_SKIPS = ("already", "needs-ocr", "unsupported", "too-large", "resume")
#: a skipped file's reason code on its run step (docs/51-run-cards.md, `runs.REASONS`)
_SKIP_REASONS = {"already": "already-filed", "needs-ocr": "needs-ocr", "unsupported": "unsupported",
                 "too-large": "too-large"}


def _step(state: str, row: dict) -> dict:
    """The `step` fields of one file's outcome: skipped with its reason, error, or ok."""
    if state == "resume":
        state = row.get("reason", "")
        return {"status": "skipped", **({"reason": _SKIP_REASONS[state]} if state in _SKIP_REASONS else {})}
    if state in _SKIP_REASONS:
        return {"status": "skipped", "reason": _SKIP_REASONS[state]}
    if state == "failed":
        return {"status": "error", "reason": f"failed-{row.get('exc_type') or 'Exception'}"}
    return {}


def pdf_chars(data: bytes) -> tuple:
    """``(characters, pages)`` of the text layer; raises on a PDF pypdf cannot open."""
    import pypdf

    pages = [(page.extract_text() or "").strip() for page in pypdf.PdfReader(io.BytesIO(data)).pages]
    return sum(len(p) for p in pages), len(pages)


def _refusal(folder: str, exts: set) -> str:
    """Why a run must not start (exit 2, nothing written), or "" when it may.

    A missing folder or unknown `--kinds`; a vault whose `git status --porcelain` is not clean (a
    half-filed crash leftover would be swept into the next commit — the paths are listed); or the
    app answering on prod (8010), dev (8011) or PORT whatever `.env` says (`vault_lock.LOCK` is per
    process, so two writers would race on the manifest and the git index).
    """
    if not os.path.isdir(folder):
        return f"not a folder: {folder}"
    if not exts or not exts <= set(pipeline.KINDS):
        return f"unknown --kinds {','.join(sorted(exts))}"
    if os.path.isdir(os.path.join(wiki.WIKI_PATH, ".git")):
        done = subprocess.run(["git", "status", "--porcelain"], cwd=wiki.WIKI_PATH,
                              capture_output=True, text=True)
        dirty = [line for line in done.stdout.splitlines() if line.strip()]
        if dirty:
            return ("the vault has uncommitted changes (a half-filed crash leftover?) — clean them "
                    "first:\n" + "\n".join("  " + line for line in dirty))
    return running_app()


def running_app() -> str:
    """"app is running — …" when the app answers on prod (8010), dev (8011) or `PORT`; "" otherwise.
    Also the guard of `python -m wiki reindex` (PRD 53)."""
    import urllib.error
    import urllib.request

    ports = {os.getenv("PROD_PORT") or "8010", os.getenv("DEV_PORT") or "8011", os.getenv("PORT") or ""} - {""}
    for port in sorted(ports):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).close()
        except urllib.error.HTTPError:
            pass                                # an error page is still an app answering
        except (OSError, ValueError):
            continue
        return f"app is running — scripts/voicebot stop (port {port})"
    return ""


def _walk(folder: str) -> list:
    found = []
    for base, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if not name.startswith(".") and not (base == folder and name == BATCH_LOG):
                found.append(os.path.relpath(os.path.join(base, name), folder).replace(os.sep, "/"))
    return sorted(found)


def _last_states(log: str) -> dict:
    last = {}
    if os.path.exists(log):
        for line in open(log).read().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "rel" in row:
                last[row["rel"]] = row.get("state", "")
    return last


def _log(log: str, row: dict) -> None:
    with open(log, "a") as f:
        f.write(json.dumps(row) + "\n")


def _now() -> str:
    return memory._stamp("%Y-%m-%dT%H:%M:%S")


def _clock(seconds: float) -> str:
    s = int(seconds)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s" if m else f"{s}s"


def _state_line(i: int, n: int, rel: str, row: dict, seconds: float) -> str:
    """`[i/n] <state>  <rel><tail>`: a filed file names its page, a skip its reason."""
    state, reason = row["state"], row.get("reason", "")
    if state == "resume":
        label, tail = "skip ", f" (last run: {reason})"
    elif state == "filed":
        label, tail = state, f" -> {row['title'] or row['page']} ({row['pages_touched']} pages, {seconds:.0f} s)"
    elif state == "failed":
        label, tail = f"{state} — {reason}", ""
    else:
        label, tail = state, f" ({reason})" if reason else ""
    return f"[{i}/{n}] {label}  {rel}{tail}"


def _one(path: str, rel: str, kinds: set, dry_run: bool) -> dict:
    """One file's outcome `{state, ...}`; only a real filing touches the vault."""
    ext = os.path.splitext(rel)[1].lower()
    if ext not in kinds:
        return {"state": "unsupported"}
    try:
        pipeline.check_upload(rel, os.path.getsize(path))
    except pipeline.IngestError as exc:
        return {"state": "too-large", "reason": exc.message}
    reached = False
    try:
        with open(path, "rb") as f:
            data = f.read()
        digest = manifest.sha256(data)
        seen = manifest.manifest_find(digest)
        if seen:
            return {"state": "already", "sha256": digest, "page": seen.get("page", ""),
                    "pages_touched": len(seen.get("pages_touched") or [])}
        note = ""
        if ext == ".pdf":
            chars, pages = pdf_chars(data)
            per_page = chars // max(pages, 1)
            note = f"{per_page} chars/page"
            if per_page < formats.MIN_PDF_CHARS:
                return {"state": "needs-ocr", "sha256": digest, "reason": note}
        if dry_run:
            return {"state": "would file", "sha256": digest, "reason": note}
        reached = True
        res = pipeline.ingest_upload(data, os.path.basename(rel), hint=os.path.dirname(rel),
                                    source=f"batch import {memory._today_iso()}", strict=True)
        entry = manifest.manifest_find(digest) or {}
        fm, _ = wiki.parse_page((wiki._read(res["path"] + ".md") or ""))
        return {"state": "filed", "sha256": digest, "page": res["path"],
                "title": fm.get("name", ""), "pages_touched": len(entry.get("pages_touched") or [])}
    except Exception as exc:  # noqa: BLE001 — one bad file must not stop the batch
        trace.logger("Batch").exception("filing a file failed")
        if not reached:                     # past that point the ingest span already traced it
            trace.error("ingest", "batch_import", exc, fatal_for="none")
        return {"state": "failed", "reason": str(exc) or type(exc).__name__, "reached": reached,
                "exc_type": type(exc).__name__}


def ingest_batch(folder: str, *, kinds=("pdf", "docx"), resume: bool = False, dry_run: bool = False,
                 limit: int = 0, sleep: float = 0, out=print) -> dict:
    """File every `kinds` file under `folder` through the drop path, one at a time (PRD 48).

    Returns the per-state counts plus `exit` (0 clean, 1 something failed, 2 refused start) and
    `seconds`. Idempotent through the manifest's sha guard; `batch.log` in `folder` remembers
    skips and failures for `resume`.
    """
    exts = {"." + k.strip().lower().lstrip(".") for k in kinds if k.strip()}
    counts = dict.fromkeys(_STATES, 0)
    out(f"vault {wiki.WIKI_PATH}")
    if refuse := _refusal(folder, exts):
        out(refuse)
        return {**counts, "exit": 2, "seconds": 0}

    if not dry_run:
        wiki.startup_check()               # PRD 53: the index matches the vault before anything is filed
    files = _walk(folder)
    log = os.path.join(folder, BATCH_LOG)
    last = _last_states(log) if resume else {}
    started = time.time()
    out(f"{len(files)} files" + (" · dry run" if dry_run else ""))
    if not dry_run:
        _log(log, {"run": _now(), "started": started, "vault": wiki.WIKI_PATH,
                   "args": {"folder": folder, "kinds": sorted(exts), "resume": resume,
                            "limit": limit, "sleep": sleep}})
    run_id = None if dry_run else time.strftime("batch-%Y%m%d-%H%M%S")      # its run page (PRD 51)
    with trace.run(run_id):
        if run_id:
            trace.run_event("batch", "start")
        failure = "failed-Exception"
        try:
            failure = _files(folder, files, exts, last, counts, resume=resume, dry_run=dry_run, limit=limit,
                             sleep=sleep, out=out, log=log)
        except BaseException as exc:       # Ctrl-C included: the run still ends, failed (PRD 51 review)
            failure = f"failed-{type(exc).__name__}"
            raise
        finally:
            if run_id:
                trace.run_event("batch", "end", "failed" if failure else "done", failure or "ok")
    skipped = {k: counts[k] for k in _SKIPS}
    detail = ", ".join(f"{k} {v}" for k, v in skipped.items() if v)
    elapsed = time.time() - started
    head = f"would file {counts['would file']}" if dry_run else f"filed {counts['filed']}"
    out(f"{head} · skipped {sum(skipped.values())}" + (f" ({detail})" if detail else "")
        + f" · failed {counts['failed']} · {_clock(elapsed)}")
    return {**counts, "exit": 1 if counts["failed"] else 0, "seconds": round(elapsed)}


def _file_step(rel: str, dry_run: bool):
    """One file's `step` span of the run; a dry run is no run, so it gets a plain dict."""
    return contextlib.nullcontext({}) if dry_run else trace.span("step", name="file", rel=rel)


def _files(folder: str, files: list, exts: set, last: dict, counts: dict, *, resume: bool, dry_run: bool,
           limit: int, sleep: float, out, log: str) -> str:
    """File `files` one at a time, each a `step` of the run; returns the first failure's reason
    code ("" when none failed)."""
    reached, failure = 0, ""
    for i, rel in enumerate(files, 1):
        if resume and last.get(rel) in _RETRY_STATES:
            counts["resume"] += 1
            row = {"state": "resume", "reason": last[rel]}
            with _file_step(rel, dry_run) as step:
                step.update(_step("resume", row))
            out(_state_line(i, len(files), rel, row, 0))
            continue
        t0 = time.time()
        with _file_step(rel, dry_run) as step:
            row = _one(os.path.join(folder, rel), rel, exts, dry_run)
            step.update(_step(row["state"], row))
        state, seconds = row["state"], round(time.time() - t0, 1)
        counts[state] += 1
        if state == "failed" and not failure:
            failure = step["reason"]
        trace.event("batch_file", rel=rel, state=state, seconds=seconds)
        out(_state_line(i, len(files), rel, row, seconds))
        if not dry_run:
            _log(log, {"rel": rel, "sha256": row.get("sha256", ""), "state": state,
                       "title": row.get("title", ""), "page": row.get("page", ""),
                       "pages_touched": row.get("pages_touched", 0), "seconds": seconds,
                       "reason": row.get("reason", ""), "at": _now()})
        if state == "filed" or row.get("reached"):
            reached += 1
            if limit and reached >= limit:
                break
            if sleep > 0 and i < len(files):
                _sleep(sleep)
    return failure
