"""Runs (docs/51-run-cards.md): a multi-step workflow's card fields and its server-rendered page.

A run is the events sharing one `run_id` (docs/49-event-trace.md): a `run` start/end pair from the
app's workflows (briefing, write-up, batch import; a host script's `/ingest` run gets only the
start, its end is derived here), `step` spans and whatever else the run emitted (ingests, model
calls, errors). `load(run_id)` reads them back from events.jsonl and its rotated `.1…5`;
`build(events)` turns them into the run dict; `render(run)` is the page: f-strings with
`html.escape` on every value, built from counts, titles, paths, shas and reason codes only — the
commit lines come from `git show --stat` (never a patch), so no page body reaches the HTML.
`REASONS` is the one sentence per reason code that both the card and the page show.

Stdlib + core; `wiki` is imported where a commit line is read, so scripts/trace_report.py (which
imports `read` from here) stays light.
"""
import html
import json
import os
import re
from datetime import datetime

from core import trace

_LABELS = {"remarkable": "reMarkable pull", "batch": "Batch import", "push": "reMarkable push"}      # the rest: capitalised
#: workflow -> label; the keys are `trace.FIELDS["workflow"]` (the JS `RUN_WORKFLOWS` is the other copy)
WORKFLOWS = {w: _LABELS.get(w, w.capitalize()) for w in sorted(trace.FIELDS["workflow"])}
#: reason code -> the sentence the card's failed step and the page's outcome line show
REASONS = {
    "ok": "Finished.",
    "nothing-to-do": "Nothing new to report.",
    "housekeeping": "Housekeeping commits (lint, index, log) are not news.",
    "failed-model": "The model returned nothing usable.",
    "failed-offline": "The model server was not reachable.",
    "needs-ocr": "No text layer: the PDF needs OCR first.",
    "unreadable": "The page could not be read.",
    "unsupported": "That file type is not imported.",
    "too-large": "The file is over the size limit.",
    "already-filed": "Already filed earlier (the same bytes).",
    "rejected-op": "A proposed page change was refused.",
    "rejected": "Rejected: nothing was changed.",
    "expired": "Not approved: the next turn moved on, nothing was changed.",
    "changed": "It changed since the proposal: nothing was changed.",
    # research (docs/54-vault-map-research.md): the governor's cut, written from what was read
    "budget-time": "budget: time",
    "budget-reads": "budget: reads",
    "nothing": "nothing",
}
FAILED = "Failed with {exc_type}."          # `failed-<exc_type>`
OFFLINE_TYPES = ("APIConnectionError", "APITimeoutError")
MAX_CARD_STEPS = 6
OPEN = {"id": "open", "label": "Open run", "kind": "open"}
COPY = {"id": "copy", "label": "Copy", "kind": "copy"}
FOOTER = "Runs age out with log rotation (5 × 10 MB)."
SHA = re.compile(r"[0-9a-f]{4,40}")
ORIGINS = {"source": "source", "wanko": "Wanko", "user": "you"}
_SOURCE_COMMIT = re.compile(r"^wiki: (?:filed|re-ingested|forgot) ")


def reason_sentence(code: str | None) -> str:
    """The sentence of a reason code ("" for none)."""
    code = code or ""
    if code in REASONS:
        return REASONS[code]
    if code.startswith("failed-"):
        return FAILED.format(exc_type=code[len("failed-"):])
    return code


def failure_reason(exc: BaseException) -> str:
    """`failed-offline` for a connection / timeout error, else `failed-<exc_type>`."""
    name = type(exc).__name__
    if isinstance(exc, (ConnectionError, TimeoutError)) or name in OFFLINE_TYPES:
        return "failed-offline"
    return f"failed-{name}"


# ------------------------------------------------------------------ the card side


def card_steps(steps: list) -> list:
    """At most MAX_CARD_STEPS: the rest fold into one "+n more" row (error if any folded one failed)."""
    if len(steps) <= MAX_CARD_STEPS:
        return list(steps)
    head, rest = steps[:MAX_CARD_STEPS - 1], steps[MAX_CARD_STEPS - 1:]
    status = "error" if any(s.get("status") == "error" for s in rest) else "ok"
    return head + [{"label": f"+{len(rest)} more", "status": status}]


class Run:
    """One app workflow inside the current turn: `run start` now, one `run end` later; collects
    the card's steps `{label, status, detail?}` (the trace's `step` spans are the callers')."""

    def __init__(self, workflow: str):
        self.workflow, self.run_id, self.steps, self.status = workflow, trace.current_run(), [], "running"
        trace.run_event(workflow, "start")

    def step(self, label: str, status: str = "ok", detail: str = "") -> None:
        self.steps.append({"label": label, "status": status, **({"detail": detail} if detail else {})})

    def end(self, status: str, reason: str) -> None:
        if self.status == "running":
            self.status = status
            trace.run_event(self.workflow, "end", status, reason)

    def fail(self, label: str, reason: str) -> None:
        """The step `label` failed with `reason`: its row carries the reason sentence; the run ends failed."""
        self.step(label, "error", reason_sentence(reason))
        self.end("failed", reason)

    def card(self, actions: list, steps: list | None = None, fold: bool = True) -> dict:
        """The card's run fields (docs/51-run-cards.md, Data model). `fold=False` keeps every step row
        (a research card: the governor bounds its rows, PRD 54)."""
        steps = list(self.steps if steps is None else steps)
        return {"run_id": self.run_id, "workflow": self.workflow, "status": self.status,
                "steps": card_steps(steps) if fold else steps, "actions": actions,
                "run_url": f"/runs/{self.run_id}"}


# ------------------------------------------------------------------ reading


def read(path: str) -> list:
    """Every event line of `path` and its rotated `.5` … `.1` files, oldest first; junk skipped."""
    rows = []
    for name in [f"{path}.{i}" for i in range(trace.BACKUPS, 0, -1)] + [path]:
        if not os.path.exists(name):
            continue
        with open(name, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue                    # a torn last line (the writer is mid-append) or junk
                if isinstance(row, dict) and isinstance(row.get("kind"), str):
                    rows.append(row)
    return rows


def load(run_id: str, path: str | None = None) -> list:
    """The events of one run (empty when it never existed or aged out of the rotation)."""
    return [r for r in read(path or trace.paths()["events"]) if r.get("run_id") == run_id]


# ------------------------------------------------------------------ building


def _order(row: dict) -> str:
    return row.get("start") or row.get("ts") or ""


def _when(ts):
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def commit_line(sha: str) -> tuple:
    """``(subject, stat summary)`` of a vault commit from `git show --stat` (never a patch);
    ("", "") for a bad sha or a commit git does not know."""
    if not SHA.fullmatch(str(sha or "")):
        return "", ""
    from vault import store

    result = store._git("show", "--stat", "--format=%h %s", sha, quiet=True)
    if result is None or result.returncode != 0:
        return "", ""
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    subject = lines[0].partition(" ")[2] if lines else ""
    summary = lines[-1] if len(lines) > 1 and " changed" in lines[-1] else ""
    return subject, summary


def origin(subject: str) -> str:
    """Who made a commit, from its subject: the user's editor, an ingest, else Wanko's page ops."""
    from agent import digest

    if digest._USER_COMMIT.match(subject):
        return "user"
    if _SOURCE_COMMIT.match(subject):
        return "source"
    return "wanko"


def _workflow(run_id: str, events: list) -> str:
    start = next((e for e in events if e["kind"] == "run" and e.get("phase") == "start"), None)
    if start and start.get("workflow") in WORKFLOWS:
        return start["workflow"]
    if run_id.startswith("rm-"):
        return "remarkable"
    if run_id.startswith("batch-"):
        return "batch"
    return "ingest" if any(e["kind"] == "ingest_end" for e in events) else ""


def _skipped_item(e: dict) -> str:
    what = e.get("detail") or e.get("rel") or e.get("name") or ""
    return f"{e['title']} — {what}" if e.get("title") and what else (e.get("title") or what)


def build(events: list) -> dict:
    """The run dict (docs/51-run-cards.md, Data model) from one run's events."""
    events = sorted(events, key=_order)
    run_id = str(events[0].get("run_id") or "")
    workflow = _workflow(run_id, events)
    end = next((ev for ev in reversed(events) if ev["kind"] == "run" and ev.get("phase") == "end"), None)
    ingests = [ev for ev in events if ev["kind"] == "ingest_end"]
    steps = [ev for ev in events if ev["kind"] == "step"]
    proposal = next((ev for ev in steps if ev.get("name") == "propose"), None)        # an approval (PRD 52)
    errors = [{"exc_type": ev.get("exc_type") or "", "first_line": (str(ev.get("message") or "").splitlines() or [""])[0]}
              for ev in events if ev["kind"] == "error"]
    if end:
        status, reason = end.get("status") or "done", end.get("reason") or ""
    elif ingests:                              # a host script's run: its end is derived (no POST from the script)
        failed = errors or any(ev.get("status") == "error" for ev in ingests)
        status = "failed" if failed else "done"
        reason = f"failed-{errors[0]['exc_type'] or 'error'}" if errors else "failed-error" if failed else "ok"
    elif proposal:
        status, reason = "awaiting", ""
    else:
        status, reason = "running", ""

    pages, seen = [], set()

    def touched(path, sha, title=""):
        if path and (path, sha) not in seen:
            seen.add((path, sha))
            pages.append({"path": path, "commit": sha or "", "title": title or ""})

    sources, memory, did_not = [], {"added": 0, "updated": 0, "removed": 0}, []
    for ev in steps:
        for page in ev.get("pages") or []:
            if isinstance(page, dict):
                touched(page.get("path"), page.get("commit"), page.get("title"))
        for src in ev.get("sources") or []:
            if isinstance(src, dict) and src.get("title"):
                sources.append({"title": src["title"], "part": src.get("part") or ""})
        counts = ev.get("memory") if isinstance(ev.get("memory"), dict) else {}
        for key in memory:
            memory[key] += int(counts.get(key) or 0)
        if ev.get("status") == "skipped":
            did_not.append({"item": _skipped_item(ev), "reason": ev.get("reason") or ""})
    for ev in ingests:
        if ev.get("already"):
            did_not.append({"item": ev.get("title") or ev.get("path") or "an ingest", "reason": "already-filed"})
            continue
        for path in ev.get("pages") or []:
            touched(path, ev.get("commit"))
        if ev.get("title") or ev.get("path"):
            sources.append({"title": ev.get("title") or ev.get("path"), "part": ""})
        if ev.get("ops_rejected"):
            n = len(ev["ops_rejected"])
            did_not.append({"item": f"{n} page change{'s' if n != 1 else ''} for {ev.get('title') or ev.get('path') or 'an ingest'}",
                            "reason": "rejected-op"})
    lines = {}
    for page in pages:
        sha = page["commit"]
        if sha not in lines:
            lines[sha] = commit_line(sha)
        subject, page["stat"] = lines[sha]
        page["origin"] = origin(subject) if subject else ""

    timeline = []
    for ev in events:
        if not ev.get("span_id") or "duration_ms" not in ev or ev["kind"] == "ingest_start":
            continue
        label = {"step": ev.get("name"), "ingest_end": f"ingest {ev.get('title') or ev.get('path') or ''}".strip(),
                 "llm_call": f"model · {ev.get('purpose') or ''}".strip(" ·"),
                 "tool_call": f"tool · {ev.get('name') or ''}".strip(" ·")}.get(ev["kind"]) or ev["kind"]
        timeline.append({"label": label, "ms": ev.get("duration_ms"), "status": ev.get("status") or "",
                         "count": ev.get("count")})
    applied = next((ev for ev in steps if ev.get("name") == "apply" and ev.get("commit")), None)
    commit = {"sha": applied["commit"], "subject": commit_line(applied["commit"])[0]} if applied else None
    first, last = _when(events[0].get("ts")), _when((end or events[-1]).get("ts"))
    return {"run_id": run_id, "workflow": workflow, "start": events[0].get("ts") or "",
            "end": (end or {}).get("ts"), "seconds": (last - first).total_seconds() if first and last else None,
            "status": status, "reason": reason,
            "steps": [{"name": ev.get("name"), "status": ev.get("status"), "count": ev.get("count"),
                       "detail": ev.get("detail")} for ev in steps],
            "timeline": timeline,
            "did": {"pages": pages, "sources": sources, "memory": memory},
            "did_not": did_not, "errors": errors,
            "proposal": None if proposal is None else {
                "items": [p for p in proposal.get("proposal") or [] if isinstance(p, dict)],
                "detail": proposal.get("detail") or ""},
            "commit": commit}


# ------------------------------------------------------------------ rendering


def e(value) -> str:
    """Every value on the page goes through here."""
    return html.escape("" if value is None else str(value), quote=True)


def _local(ts: str) -> str:
    when = _when(ts)
    return when.astimezone().strftime("%Y-%m-%d %H:%M:%S") if when else ts


def _duration(seconds) -> str:
    if seconds is None:
        return "–"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds) // 60} min {int(seconds) % 60:02d} s"


def _list(items: list, empty: str) -> str:
    if not items:
        return f'<p class="run-none">{e(empty)}</p>'
    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def _page_row(p: dict) -> str:
    who = ORIGINS.get(p.get("origin") or "", "")
    sha = f'<code class="run-sha">{e(p["commit"])}</code> ' if p.get("commit") else ""
    stat = f' · <span class="run-stat">{e(p["stat"])}</span>' if p.get("stat") else ""
    label = f' <em class="run-origin">{e(who)}</em>' if who else ""
    name = f"{e(p['title'])} ({e(p['path'])})" if p.get("title") else e(p["path"])
    return f"{sha}{name}{stat}{label}"


def _proposal(run: dict) -> str:
    """An approval's Proposal section (what it would change) and, once approved, its Commit (PRD 52)."""
    out = ""
    if run.get("proposal") is not None:
        p = run["proposal"]
        rows = [e(f'{i["title"]} ({i["path"]})' if i.get("title") and i.get("path") else i.get("title") or i.get("path"))
                for i in p["items"]] + ([e(p["detail"])] if p.get("detail") else [])
        out += f'<section class="run-section run-proposal"><h2>Proposal</h2>{_list(rows, "Nothing listed.")}</section>\n'
    if run.get("commit"):
        c = run["commit"]
        out += (f'<section class="run-section run-commit"><h2>Commit</h2><p><code class="run-sha">{e(c["sha"])}</code> '
                f'{e(c["subject"])}</p></section>\n')
    return out


def render(run: dict) -> str:
    """The run page: header, Proposal / Commit (an approval), Timeline, Did, Did not, Errors, footer —
    every value escaped."""
    label = WORKFLOWS.get(run["workflow"], run["workflow"] or "Run")
    timeline = [f'{e(t["label"])} · {e(t["ms"])} ms · <span class="run-status-{e(t["status"])}">{e(t["status"])}</span>'
                + (f' · {e(t["count"])}' if t.get("count") is not None else "") for t in run["timeline"]]
    did = run["did"]
    sources = [e(f'{s["title"]} · part {s["part"]}' if s.get("part") else s["title"]) for s in did["sources"]]
    mem = did["memory"]
    memory = (f'<p class="run-memory">Memory: {e(mem["added"])} added, {e(mem["updated"])} updated, '
              f'{e(mem["removed"])} removed</p>') if any(mem.values()) else ""
    did_not = [f'{e(d["item"])} — {e(reason_sentence(d["reason"]))}' if d.get("reason") else e(d["item"])
               for d in run["did_not"]]
    errors = [f'<code>{e(x["exc_type"])}</code>' + (f': {e(x["first_line"])}' if x["first_line"] and
                                                     x["first_line"] != x["exc_type"] else "")
              for x in run["errors"]]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(label)} · run {e(run["run_id"])}</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body class="run-page">
<main class="run">
<header class="run-header">
<h1>{e(label)}</h1>
<p class="run-meta">{e(_local(run["start"]))} · {e(_duration(run["seconds"]))} ·
<span class="run-chip run-{e(run["status"])}">{e(run["status"])}</span>
<span class="run-reason">{e(reason_sentence(run["reason"]))}</span></p>
<p class="run-id">run <code>{e(run["run_id"])}</code></p>
</header>
{_proposal(run)}<section class="run-section"><h2>Timeline</h2>{_list(timeline, "No steps recorded.")}</section>
<section class="run-section"><h2>Did</h2>
<h3>Pages</h3>{_list([_page_row(p) for p in did["pages"]], "No pages touched.")}
<h3>Sources</h3>{_list(sources, "No sources.")}
{memory}</section>
<section class="run-section"><h2>Did not</h2>{_list(did_not, "Nothing left out.")}</section>
<section class="run-section"><h2>Errors</h2>{_list(errors, "No errors.")}</section>
<footer class="run-footer">{e(FOOTER)}</footer>
</main>
</body>
</html>
"""


def missing(run_id: str) -> str:
    """The 404 page: one line."""
    line = f"No run {run_id} (it may have aged out of the log)."
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>{e(line)}</title>'
            f'<link rel="stylesheet" href="/static/style.css"></head>'
            f'<body class="run-page"><main class="run"><p class="run-none">{e(line)}</p></main></body></html>\n')
