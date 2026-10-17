"""Event trace (docs/49-event-trace.md): one JSONL stream of actionable events, never text.

Two profiles, from `TRACE_PROFILE` (default `dev` from a checkout; docker-compose.yml forces `prod`):
`prod` writes `events.jsonl` only (titles, paths, scores, lengths, timings, ids); `dev` writes the
same plus `verbose.jsonl` (prompts, turn text, tool results). Both live in `TRACE_DIR` (default the
repo root; `/data/logs` in the container) and rotate at 10 MB x 5.

`event(kind, **fields)` adds the envelope (`ts`, `kind`, `profile`, `pid`, `turn_id`, `ingest_id`)
and keeps only the field names in FIELDS, each by its rule there. `span(kind)` adds the span shape
(`span_id`, `parent_id`, `start`, `duration_ms`, `status`) under the current turn's or ingest's root
span; `run_id` groups a turn or an ingest (or a host script's run) across restarts. The writer never
raises: a failed write is one rate-limited stderr line.

Also the stdout side: `logger(name)` prints `[name] message` to the current stdout; DEBUG lines
(turn text) only in dev.
"""
import contextlib
import contextvars
import itertools
import json
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

from core import config

ROOT = config.ROOT
TRACE_DIR = config.TRACE_DIR     # dev: repo-local, gitignored
PROFILES = ("prod", "dev")
DOCKERENV = "/.dockerenv"
PROD_DATA = "~/.voicebot"
DATA_VARS = ("DATA", "WIKI_PATH", "MEMORY_DIR")
MAX_BYTES = 10 * 1024 * 1024
BACKUPS = 5
CAP = 200                  # longest string an event keeps
TRACEBACK_CAP = 4096
TRACEBACK_FRAMES = 20
LIST_CAP = 20
COMPLAIN_EVERY = 60.0      # seconds between two stderr lines about a failed write

# ------------------------------------------------------------------ the allow-list

#: field rules: ANY keeps a string up to CAP chars (longer is dropped), CUT cuts it to CAP (titles,
#: paths), TAIL keeps its last TRACEBACK_CAP chars; a set is a fixed vocabulary and a pattern a
#: shape — a string outside either is "other" in events (the original goes to verbose.jsonl in dev).
#: `vocab()` adds the values other modules own (tool names, op kinds).
ANY, CUT, TAIL = "any", "cut", "tail"
IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,39}")

#: every field name an event may carry (the twelve kinds, envelope, span shape); others are dropped
FIELDS = {
    # turn_start
    "origin": {"typed", "spoken", "briefing", "card"}, "text_len": ANY, "words": ANY,
    # recall
    "band": ANY, "page": CUT, "hits": ANY, "source": CUT, "n": ANY, "score": ANY, "named": ANY, "offer": ANY,
    # tool_call
    "name": {"llm", "asr"}, "arg_keys": IDENTIFIER, "ms": ANY, "result_len": ANY, "ok": ANY,
    # llm_call
    "purpose": ANY, "model": ANY, "max_tokens": ANY, "ttft_ms": ANY, "tokens_in": ANY, "tokens_out": ANY,
    "fallback_reason": {"not_covered"},
    # turn_end
    "reply_len": ANY, "tools": set(), "ctx_pct": ANY, "outcome": ANY,
    # ingest_start
    "ingest_kind": ANY, "raw": ANY, "bytes": ANY, "pages": ANY,
    # ingest_end (`kind` / `reason_code` inside ops_rejected)
    "path": CUT, "title": CUT, "ops_applied": ANY, "ops_rejected": ANY, "kind": set(),
    "reason_code": {"bad_path", "not_indexed", "author_page", "unknown_op", "no_source", "user_page", "other"},
    "already": ANY, "degraded": ANY,
    # briefing
    "items": ANY, "sources": ANY,
    # batch_file
    "rel": ANY, "state": {"filed", "would file", "already", "needs-ocr", "unsupported", "too-large", "resume", "failed"},
    "seconds": ANY,
    # memory_save
    "facts_added": ANY, "facts_dropped": ANY,
    # error
    "scope": ANY, "exc_type": ANY, "message": ANY, "code": ANY, "traceback": TAIL, "fatal_for": {"none"},
    "where": {"respond", "briefing", "write_up", "save_note", "new_topic", "ingest", "summarise",
              "propose_page_ops", "meeting_from_transcript", "asr", "canned_turn", "memory_save", "ws",
              "batch_import", "sync_sources", "startup_check", "approval", "research"},   # sync/startup: the search index (PRD 53)
    # service / ws
    "event": ANY, "profile": ANY, "version": ANY, "pid": ANY,
    # span shape, run id
    "span_id": ANY, "parent_id": ANY, "start": ANY, "duration_ms": ANY, "status": ANY, "run_id": ANY,
    # run / step (docs/51-run-cards.md): `detail` is a short fact string, ANY not CUT — a longer one is dropped
    "workflow": {"briefing", "write-up", "remarkable", "batch", "ingest", "forget", "push", "research"}, "phase": {"start", "end"},
    "reason": re.compile(r"ok|nothing-to-do|housekeeping|needs-ocr|unreadable|unsupported|too-large|already-filed"
                         r"|rejected-op|rejected|expired|changed|budget-time|budget-reads|nothing"
                         r"|failed-model|failed-offline|failed-[A-Za-z_][A-Za-z0-9_]{0,59}"),   # runs.REASONS
    "count": ANY, "detail": ANY, "commit": re.compile(r"[0-9a-f]{4,40}"), "skipped": ANY,
    "part": ANY, "memory": ANY, "added": ANY, "updated": ANY, "removed": ANY,
    # decision (docs/52-approvals.md): the approval kind is its `workflow`; `proposal` = the `{path, title}` rows shown
    "decision": {"approve", "reject", "expired", "changed"}, "via": {"voice", "typed", "button", "reconnect", "disconnect"}, "proposal": ANY,
}
#: step names (a `step` span's `name`, beside the tool names `vocab` adds)
STEPS = {"gather", "skip", "phrase", "context", "write", "file", "propose", "apply",
         "plan", "search", "judge", "read", "verify", "budget"}       # research (docs/54-vault-map-research.md)
FIELDS["name"] |= STEPS
#: rejection texts → a stable reason_code (sources.restrict_ops, wiki ops, memory.reject_user_pages)
REASONS = [(re.compile(p), code) for p, code in (
    (r"^bad path", "bad_path"), (r"is not in the index", "not_indexed"), (r"is not in the user's profile", "author_page"),
    (r"^unknown op", "unknown_op"), (r"^source is required", "no_source"), (r"is the user, not another person", "user_page"))]

# ------------------------------------------------------------------ context: the one place ids live

_turn = contextvars.ContextVar("turn_id", default=None)
_turn_state = contextvars.ContextVar("turn_state", default=None)      # {"failed": True} once a handler failed
_ingest = contextvars.ContextVar("ingest", default=None)     # (ingest_id, state dict)
_root = contextvars.ContextVar("root_span", default=None)
_run = contextvars.ContextVar("run_id", default=None)
_ids = itertools.count(1)
_runs = itertools.count(1)     # run ids: process-wide, so a reconnect (turn ids restart) never reuses one
_lock = threading.RLock()
_state: dict = {}


@contextlib.contextmanager
def _bind(*pairs):
    """Set `(contextvar, value)` pairs for the body; reset them after, last set first."""
    tokens = [(var, var.set(value)) for var, value in pairs]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def _new_run() -> str:
    return f"{os.getpid()}-{next(_runs)}"


def _span_id() -> str:
    return os.urandom(8).hex()


def _iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def ms_since(t0: float) -> int:
    """Whole milliseconds since `t0` (a `time.monotonic()` reading)."""
    return round((time.monotonic() - t0) * 1000)


# ------------------------------------------------------------------ profile


def decide(env=None) -> tuple:
    """``(profile, reason)``: the profile this process runs under and why it must not start ("" = it may).

    `TRACE_PROFILE` picks it (default dev). An unknown value fails closed: the profile is prod and the
    reason names the value, so `configure()` refuses it while a lazy writer (an import, a test) runs
    as prod. dev refuses inside a container (`/.dockerenv`) or with any of DATA_VARS under
    ~/.voicebot — dev never touches prod data.
    """
    env = os.environ if env is None else env
    requested = (env.get("TRACE_PROFILE") or "dev").strip().lower()
    if requested not in PROFILES:
        return "prod", f"unknown TRACE_PROFILE {requested!r} (prod|dev)"
    if requested != "dev":
        return requested, ""
    if os.path.exists(DOCKERENV):
        return requested, "TRACE_PROFILE=dev inside a container; set TRACE_PROFILE=prod"
    prod = os.path.realpath(os.path.expanduser(PROD_DATA))
    for name in DATA_VARS:
        value = env.get(name)
        if not value:
            continue
        path = os.path.realpath(os.path.expanduser(value))
        if path == prod or path.startswith(prod + os.sep):
            return requested, f"TRACE_PROFILE=dev with {name}={value} under {PROD_DATA}; set TRACE_PROFILE=prod"
    return requested, ""


# ------------------------------------------------------------------ stdout logging


def _exc_text(exc_info) -> str:
    """dev: the full traceback; prod: the type and frame locations only (the events' policy, no text)."""
    if profile() == "dev":
        return "".join(traceback.format_exception(*exc_info)).rstrip()
    exc = exc_info[1]
    frames = "".join(f'  File "{f.filename}", line {f.lineno}, in {f.name}\n'
                     for f in traceback.extract_tb(exc.__traceback__, limit=-TRACEBACK_FRAMES))
    return f"{frames}{type(exc).__name__}"


class _Stdout(logging.Handler):
    """`[name] message` to whatever sys.stdout is now (so pytest's capsys sees it)."""

    def emit(self, record):
        try:
            text = f"[{record.name.split('.', 1)[-1]}] {record.getMessage()}"
            if record.exc_info:
                text += "\n" + _exc_text(record.exc_info)
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 — a closed stdout must not break a turn
            pass


_log_root = logging.getLogger("vb")
_log_root.propagate = False
if not _log_root.handlers:
    _log_root.addHandler(_Stdout())


def _set_level(profile: str) -> None:
    _log_root.setLevel(logging.DEBUG if profile == "dev" else logging.INFO)


_set_level(decide()[0])


def logger(name: str) -> logging.Logger:
    """The stdout logger whose lines read `[name] …`."""
    return logging.getLogger("vb." + name)


log = logger("Trace")


# ------------------------------------------------------------------ startup


def _version() -> str:
    if os.getenv("VOICEBOT_VERSION"):
        return os.environ["VOICEBOT_VERSION"]
    try:
        out = subprocess.run(["git", "-C", ROOT, "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=2)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _setup(profile: str | None = None) -> None:
    """Pick the profile once per process and write `service start` as the first event."""
    with _lock:
        if _state.get("profile"):
            return
        _state.update(profile=profile or decide()[0], dir=TRACE_DIR, handlers={}, complained=0.0)
        _set_level(_state["profile"])
    event("service", event="start", profile=_state["profile"], version=_version(), pid=os.getpid())


def configure(out=None) -> None:
    """Startup: refuse (exit 2, one stderr line) a dev process against prod data, else print the profile.

    A host script passes `out=sys.stderr` so the profile line stays out of its own output; its events
    go to `TRACE_DIR` like the app's (never a folder derived from the vault).
    """
    profile, reason = decide()
    if reason:
        sys.stderr.write(f"[Trace] refusing to start: {reason}\n")
        sys.exit(2)
    _setup(profile)
    line = f"profile={profile} events={paths()['events']}"
    if profile == "dev":
        line += f" verbose={paths()['verbose']}"
    if out is None:
        log.info(line)
    else:
        out.write(f"[Trace] {line}\n")


def profile() -> str:
    return _state.get("profile") or decide()[0]


def paths() -> dict:
    folder = _state.get("dir") or TRACE_DIR
    return {"events": os.path.join(folder, "events.jsonl"), "verbose": os.path.join(folder, "verbose.jsonl")}


def reset() -> None:
    """Close the files and forget the profile (tests; the next event starts a fresh stream)."""
    with _lock:
        for handler in (_state.get("handlers") or {}).values():
            handler.close()
        _state.clear()


# ------------------------------------------------------------------ writer


class _Handler(logging.handlers.RotatingFileHandler):
    def handleError(self, record):
        _complain(sys.exc_info()[1])


def _complain(exc) -> None:
    now = time.monotonic()
    if now - _state.get("complained", 0.0) < COMPLAIN_EVERY and _state.get("complained"):
        return
    _state["complained"] = now
    with contextlib.suppress(Exception):
        sys.stderr.write(f"[Trace] write failed: {exc}\n")


def vocab(field: str, values) -> None:
    """Add the values `field` may take (tool names from llm, op kinds from wiki)."""
    FIELDS[field].update(values)
    if field == "name":
        FIELDS["tools"].update(values)


def reason_code(reason) -> str:
    """The stable code of a rejection reason text; "other" when none matches."""
    text = str(reason or "")
    return next((code for pattern, code in REASONS if pattern.search(text)), "other")


_DROP = object()


def _clean(fields, name: str = ""):
    """`fields` by the FIELDS rules: unknown names dropped, nested dicts / lists the same way
    (lists cut to LIST_CAP); `_DROP` = leave the value out."""
    if isinstance(fields, dict):
        return {k: v for k, v in ((k, _clean(v, k)) for k, v in fields.items() if k in FIELDS) if v is not _DROP}
    if isinstance(fields, (list, tuple, set)):
        return [v for v in (_clean(v, name) for v in list(fields)[:LIST_CAP]) if v is not _DROP]
    if not isinstance(fields, str):
        return fields
    rule = FIELDS.get(name, ANY)
    if rule is TAIL:
        return fields[-TRACEBACK_CAP:]
    if isinstance(rule, (set, re.Pattern)):
        if rule.fullmatch(fields) if isinstance(rule, re.Pattern) else fields in rule:
            return fields
        verbose("vocab_miss", field=name, value=fields)
        return "other"
    if len(fields) > CAP:
        return fields[:CAP] if rule is CUT else _DROP
    return fields


def _envelope(kind: str) -> dict:
    rec = {"ts": _iso(), "kind": kind, "profile": _state.get("profile"), "pid": os.getpid()}
    if _run.get() is not None:
        rec["run_id"] = _run.get()
    if _turn.get() is not None:
        rec["turn_id"] = _turn.get()
    if _ingest.get() is not None:
        rec["ingest_id"] = _ingest.get()[0]
    return rec


def _emit(stream: str, kind: str, fields: dict) -> None:
    """The one writer path: events are cleaned, verbose lines exist only in dev; never raises."""
    try:
        _setup()
        if stream == "verbose" and _state["profile"] != "dev":
            return                                  # prod never opens the file
        if stream == "events":
            fields = _clean(fields)
        record = {**fields, **_envelope(kind)}      # the envelope always wins
        with _lock:
            handler = _state["handlers"].get(stream)
            if handler is None:
                path = paths()[stream]
                os.makedirs(os.path.dirname(path), exist_ok=True)
                handler = _Handler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS, delay=True, encoding="utf-8")
                _state["handlers"][stream] = handler
        handler.handle(logging.makeLogRecord({"msg": json.dumps(record, default=str, ensure_ascii=False)}))
    except Exception as exc:  # noqa: BLE001 — the trace never breaks a turn
        _complain(exc)


def event(kind: str, /, **fields) -> None:
    """One events.jsonl line."""
    _emit("events", kind, fields)


def verbose(kind: str, /, **fields) -> None:
    """One verbose.jsonl line in dev (prompts, turn text, tool results)."""
    _emit("verbose", kind, fields)


def error(scope: str, where: str, exc: BaseException | None = None, fatal_for: str | None = None) -> None:
    """An `error` event for `exc` (default: the one being handled): its type as `message` and
    `code`, the frame locations (no source lines). The exception text goes to verbose.jsonl in dev.
    `fatal_for="none"`: a background failure nothing waited on (it never fails a turn)."""
    exc = exc or sys.exc_info()[1]
    if exc is None:
        return
    if _ingest.get() is not None:
        _ingest.get()[1].setdefault("_seen", set()).add(id(exc))
    if scope == "turn" and fatal_for is None:
        fail_turn()
    frames = "".join(f'  File "{f.filename}", line {f.lineno}, in {f.name}\n'
                     for f in traceback.extract_tb(exc.__traceback__, limit=-TRACEBACK_FRAMES))
    code = type(exc).__name__
    event("error", scope=scope, where=where, exc_type=code, message=code, code=code,
          traceback=frames[-TRACEBACK_CAP:], **({"fatal_for": fatal_for} if fatal_for else {}))
    verbose("error_text", scope=scope, where=where, message=str(exc))


def fail_turn() -> None:
    """Mark the current turn failed (a handler that swallowed its exception); no-op outside a turn."""
    state = _turn_state.get()
    if state is not None:
        state["failed"] = True


def turn_failed() -> bool:
    return bool((_turn_state.get() or {}).get("failed"))


# ------------------------------------------------------------------ units and spans


def root_span() -> str | None:
    return _root.get()


def current_run() -> str | None:
    """The `run_id` events are grouped under right now (a turn's, an ingest's, a host script's)."""
    return _run.get()


def run_event(workflow: str, phase: str, status: str | None = None, reason: str | None = None) -> None:
    """A `run` point event (docs/51-run-cards.md): `start` opens a workflow's run, `end` closes it
    with a status (done/nothing/failed) and a reason code (`runs.REASONS`)."""
    event("run", workflow=workflow, phase=phase,
          **{k: v for k, v in (("status", status), ("reason", reason)) if v is not None})


@contextlib.contextmanager
def turn(turn_id):
    """Every event inside carries `turn_id` (per session) and a fresh `run_id` (`<pid>-<n>`, unique in
    the process); a fresh root span unless this turn is already open."""
    if _turn.get() == turn_id and _root.get():
        yield _root.get()
        return
    with _bind((_turn, turn_id), (_root, _span_id()), (_run, _new_run()), (_turn_state, {})):
        yield _root.get()


@contextlib.contextmanager
def run(run_id: str | None):
    """Group what happens inside under a caller's `run_id` (a host script's `/ingest` field)."""
    if not run_id:
        yield
        return
    with _bind((_run, run_id)):
        yield


@contextlib.contextmanager
def span(kind: str, /, **fields):
    """A child span of the current root: yields a dict the body fills; one event on exit with
    `span_id`, `parent_id`, `start`, `duration_ms`, `status` (error when the body raised; a body
    that finished may set its own, e.g. a `step` that was `skipped`)."""
    out = dict(fields)
    parent, start, t0, status = _root.get(), _iso(), time.monotonic(), "ok"
    try:
        yield out
    except GeneratorExit:               # an abandoned stream is closed, not failed
        raise
    except BaseException:
        status = "error"
        raise
    finally:
        ms = ms_since(t0)
        out.setdefault("ms", ms)
        chosen = out.pop("status", None)
        status = chosen if chosen and status == "ok" else status
        event(kind, span_id=_span_id(), parent_id=parent, start=start, duration_ms=ms, status=status, **out)


def llm_span(purpose: str, messages, /, **fields):
    """The `llm_call` span of one completion; its prompt goes to verbose.jsonl (dev) first."""
    verbose("prompt", purpose=purpose, messages=messages)
    return span("llm_call", purpose=purpose, **fields)


def usage_fields(usage) -> dict:
    """`tokens_in` / `tokens_out` from a completion's `usage` (null when there is none)."""
    return {"tokens_in": getattr(usage, "prompt_tokens", None), "tokens_out": getattr(usage, "completion_tokens", None)}


def marked(key: str, default=None):
    """What `mark` noted under `key` on the current ingest (`default` outside one)."""
    current = _ingest.get()
    return current[1].get(key, default) if current is not None else default


def mark(**fields) -> None:
    """Note outcome fields (`degraded`, `ops_applied`, …) on the current ingest; no-op outside one."""
    current = _ingest.get()
    if current is not None:
        current[1].update(fields)


@contextlib.contextmanager
def ingest(kind: str, **fields):
    """One ingest unit: `ingest_start` now, `ingest_end` on exit with what `mark` noted.

    A nested entry point (ingest_path → ingest_upload) joins the outer unit. The run is the
    caller's (a turn, a host script's `run_id`) or its own `<pid>-<n>` from the same counter as turns.
    """
    if _ingest.get() is not None:
        yield _ingest.get()[1]
        return
    iid = f"{os.getpid()}-{next(_ids)}"
    state = {"ops_applied": 0, "ops_rejected": [], "pages": [], "already": False, "degraded": False}
    sid, parent, start, t0 = _span_id(), _root.get(), _iso(), time.monotonic()
    shape = {"span_id": sid, "parent_id": parent, "start": start}
    with _bind((_ingest, (iid, state)), (_root, sid), (_run, _run.get() or _new_run())):
        event("ingest_start", ingest_kind=kind, **shape, **fields)
        status = "ok"
        try:
            yield state
        except BaseException as exc:
            status = "error"
            if id(exc) not in state.get("_seen", ()):
                error("ingest", "ingest", exc)
            raise
        finally:
            ms = ms_since(t0)
            event("ingest_end", **shape, duration_ms=ms, status=status, ms=ms,
                  **{k: v for k, v in state.items() if not k.startswith("_")})
