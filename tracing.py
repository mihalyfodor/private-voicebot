"""Event trace (docs/49-event-trace.md): one JSONL stream of actionable events, never text.

Two profiles, from `TRACE_PROFILE` (default `dev` from a checkout; docker-compose.yml forces `prod`):
`prod` writes `events.jsonl` only (titles, paths, scores, lengths, timings, ids); `dev` writes the
same plus `verbose.jsonl` (prompts, turn text, tool results). Both live in `TRACE_DIR` (default the
repo root; `/data/logs` in the container) and rotate at 10 MB x 5.

`event(kind, **fields)` adds the envelope (`ts`, `kind`, `profile`, `pid`, `turn_id`, `ingest_id`)
and keeps only allow-listed field names; a string over 200 chars is dropped unless its name is in
NAMED (then it is cut). `span(kind)` adds the span shape (`span_id`, `parent_id`, `start`,
`duration_ms`, `status`) under the current turn's or ingest's root span; `run_id` groups a turn or an
ingest (or a host script's run) across restarts. The writer never raises:
a failed write is one rate-limited stderr line.

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

ROOT = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.expanduser(os.getenv("TRACE_DIR") or os.path.join(ROOT, "logs"))     # dev: repo-local, gitignored
PROFILES = ("prod", "dev")
DOCKERENV = "/.dockerenv"
PROD_DATA = "~/docker-data"
DATA_VARS = ("DATA", "WIKI_PATH", "MEMORY_DIR")
MAX_BYTES = 10 * 1024 * 1024
BACKUPS = 5
CAP = 200                  # longest string an event keeps
TRACEBACK_CAP = 4096
TRACEBACK_FRAMES = 20
LIST_CAP = 20
COMPLAIN_EVERY = 60.0      # seconds between two stderr lines about a failed write

#: every field name an event may carry (envelope, span shape and the twelve kinds); others are dropped
FIELDS = {
    "origin", "text_len", "words",                                          # turn_start
    "band", "page", "hits", "source", "n", "score", "named", "offer",       # recall
    "name", "arg_keys", "ms", "result_len", "ok",                           # tool_call
    "purpose", "model", "max_tokens", "ttft_ms", "tokens_in", "tokens_out", "fallback_reason",  # llm_call
    "reply_len", "tools", "ctx_pct", "outcome",                             # turn_end
    "ingest_kind", "raw", "bytes", "pages",                                 # ingest_start
    "path", "title", "ops_applied", "ops_rejected", "kind", "reason_code", "already", "degraded",  # ingest_end
    "items", "sources",                                                     # briefing
    "rel", "state", "seconds",                                              # batch_file
    "facts_added", "facts_dropped",                                         # memory_save
    "scope", "where", "exc_type", "message", "code", "traceback",           # error
    "event", "profile", "version", "pid", "code",                           # service / ws
    "span_id", "parent_id", "start", "duration_ms", "status", "run_id",     # span shape, run id
}
#: string fields cut to CAP instead of dropped when longer (titles and paths)
NAMED = {"title", "path", "page", "source"}
#: fields whose values come from a fixed vocabulary: anything else is "other" in events (and the
#: original goes to verbose.jsonl in dev). `vocab()` adds the values other modules own (tool names, op kinds).
VOCAB = {
    "where": {"respond", "briefing", "write_up", "save_note", "new_topic", "ingest", "summarise",
              "propose_page_ops", "meeting_from_transcript"},
    "state": {"filed", "would file", "already", "needs-ocr", "unsupported", "too-large", "resume", "failed"},
    "name": {"llm", "asr"},
    "fallback_reason": {"not_covered"},
    "reason_code": {"bad_path", "not_indexed", "author_page", "unknown_op", "no_source", "user_page", "other"},
    "kind": set(),
    "origin": {"typed", "spoken", "briefing", "card"},
    "tools": set(),
    "arg_keys": set(),
}
IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]{0,39}")
#: rejection texts → a stable reason_code (sources.restrict_ops, wiki ops, memory.reject_user_pages)
REASONS = [(re.compile(p), code) for p, code in (
    (r"^bad path", "bad_path"), (r"is not in the index", "not_indexed"), (r"is not in the user's profile", "author_page"),
    (r"^unknown op", "unknown_op"), (r"^source is required", "no_source"), (r"is the user, not another person", "user_page"))]

_turn = contextvars.ContextVar("turn_id", default=None)
_ingest = contextvars.ContextVar("ingest", default=None)     # (ingest_id, state dict)
_root = contextvars.ContextVar("root_span", default=None)
_run = contextvars.ContextVar("run_id", default=None)
_ids = itertools.count(1)
_runs = itertools.count(1)     # run ids: process-wide, so a reconnect (turn ids restart) never reuses one
_lock = threading.RLock()
_state: dict = {}


# ------------------------------------------------------------------ stdout logging


class _Stdout(logging.Handler):
    """`[name] message` to whatever sys.stdout is now (so pytest's capsys sees it)."""

    def emit(self, record):
        try:
            text = f"[{record.name.split('.', 1)[-1]}] {record.getMessage()}"
            if record.exc_info:
                text += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 — a closed stdout must not break a turn
            pass


_log_root = logging.getLogger("vb")
_log_root.propagate = False
if not _log_root.handlers:
    _log_root.addHandler(_Stdout())


def logger(name: str) -> logging.Logger:
    """The stdout logger whose lines read `[name] …`."""
    return logging.getLogger("vb." + name)


log = logger("Trace")


def _env_profile() -> str:
    """The profile `TRACE_PROFILE` asks for; an unknown value fails closed to prod."""
    value = (os.getenv("TRACE_PROFILE") or "dev").strip().lower()
    return value if value in PROFILES else "prod"


_log_root.setLevel(logging.DEBUG if _env_profile() == "dev" else logging.INFO)


# ------------------------------------------------------------------ profile


def refusal(profile: str, env=None) -> str:
    """Why this process must not start under `profile`, or "" (dev never touches prod data)."""
    env = os.environ if env is None else env
    if profile not in PROFILES:
        return f"unknown TRACE_PROFILE {profile!r} (prod|dev)"
    if profile != "dev":
        return ""
    if os.path.exists(DOCKERENV):
        return "TRACE_PROFILE=dev inside a container; set TRACE_PROFILE=prod"
    prod = os.path.realpath(os.path.expanduser(PROD_DATA))
    for name in DATA_VARS:
        value = env.get(name)
        if not value:
            continue
        path = os.path.realpath(os.path.expanduser(value))
        if path == prod or path.startswith(prod + os.sep):
            return f"TRACE_PROFILE=dev with {name}={value} under {PROD_DATA}; set TRACE_PROFILE=prod"
    return ""


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
        _state.update(profile=profile or _env_profile(), dir=TRACE_DIR, handlers={}, complained=0.0)
        _log_root.setLevel(logging.DEBUG if _state["profile"] == "dev" else logging.INFO)
    event("service", event="start", profile=_state["profile"], version=_version(), pid=os.getpid())


def configure(out=None) -> None:
    """Startup: refuse (exit 2, one stderr line) a dev process against prod data, else print the profile.

    A host script passes `out=sys.stderr` so the profile line stays out of its own output; its events
    go to `TRACE_DIR` like the app's (never a folder derived from the vault).
    """
    profile = (os.getenv("TRACE_PROFILE") or "dev").strip().lower()
    reason = refusal(profile)
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
    return _state.get("profile") or _env_profile()


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


def _write(stream: str, record: dict) -> None:
    with _lock:
        handler = _state["handlers"].get(stream)
        if handler is None:
            path = paths()[stream]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handler = _Handler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS, delay=True, encoding="utf-8")
            _state["handlers"][stream] = handler
    handler.handle(logging.makeLogRecord({"msg": json.dumps(record, default=str, ensure_ascii=False)}))


def vocab(field: str, values) -> None:
    """Add the values `field` may take (tool names from llm, op kinds from wiki)."""
    VOCAB.setdefault(field, set()).update(values)
    if field == "name":
        VOCAB["tools"].update(values)


def reason_code(reason) -> str:
    """The stable code of a rejection reason text; "other" when none matches."""
    text = str(reason or "")
    return next((code for pattern, code in REASONS if pattern.search(text)), "other")


def _word(name: str, value: str):
    """A vocabulary field's value, or "other" (the original to verbose.jsonl in dev)."""
    ok = IDENTIFIER.fullmatch(value) if name == "arg_keys" else value in VOCAB[name]
    if ok:
        return value
    verbose("vocab_miss", field=name, value=value)
    return "other"


def _clean(value, name: str = ""):
    """Allow-listed names only, vocabulary fields mapped, strings capped; nested dicts / lists the
    same way. `_DROP` = drop it."""
    if isinstance(value, dict):
        return {k: v for k, v in ((k, _clean(v, k)) for k, v in value.items() if k in FIELDS) if v is not _DROP}
    if isinstance(value, (list, tuple, set)):
        items = [_clean(v, name) for v in list(value)[:LIST_CAP]]
        return [v for v in items if v is not _DROP]
    if isinstance(value, str):
        if name in VOCAB:
            return _word(name, value)
        if name == "traceback":
            return value[-TRACEBACK_CAP:]
        if len(value) > CAP:
            return value[:CAP] if name in NAMED else _DROP
    return value


_DROP = object()


def _envelope(kind: str) -> dict:
    rec = {"ts": datetime.now().astimezone().isoformat(timespec="milliseconds"), "kind": kind,
           "profile": _state.get("profile"), "pid": os.getpid()}
    if _run.get() is not None:
        rec["run_id"] = _run.get()
    if _turn.get() is not None:
        rec["turn_id"] = _turn.get()
    if _ingest.get() is not None:
        rec["ingest_id"] = _ingest.get()[0]
    return rec


def event(kind: str, /, **fields) -> None:
    """One events.jsonl line; never raises."""
    try:
        _setup()
        _write("events", {**_clean(fields), **_envelope(kind)})     # the envelope always wins
    except Exception as exc:  # noqa: BLE001 — the trace never breaks a turn
        _complain(exc)


def verbose(kind: str, /, **fields) -> None:
    """One verbose.jsonl line in dev (prompts, turn text, tool results); prod never opens the file."""
    try:
        _setup()
        if _state["profile"] != "dev":
            return
        _write("verbose", {**fields, **_envelope(kind)})
    except Exception as exc:  # noqa: BLE001
        _complain(exc)


def error(scope: str, where: str, exc: BaseException | None = None) -> None:
    """An `error` event for `exc` (default: the one being handled): its type as `message` and
    `code`, the frame locations (no source lines). The exception text goes to verbose.jsonl in dev."""
    exc = exc or sys.exc_info()[1]
    if exc is None:
        return
    state = _ingest.get()[1] if _ingest.get() else None
    if state is not None:
        state.setdefault("_seen", set()).add(id(exc))
    if scope == "turn":
        fail_turn()
    frames = "".join(f'  File "{f.filename}", line {f.lineno}, in {f.name}\n'
                     for f in traceback.extract_tb(exc.__traceback__, limit=-TRACEBACK_FRAMES))
    code = type(exc).__name__
    event("error", scope=scope, where=where, exc_type=code, message=code, code=code,
          traceback=frames[-TRACEBACK_CAP:])
    verbose("error_text", scope=scope, where=where, message=str(exc))


_turn_state = contextvars.ContextVar("turn_state", default=None)


def fail_turn() -> None:
    """Mark the current turn failed (a handler that swallowed its exception); no-op outside a turn."""
    state = _turn_state.get()
    if state is not None:
        state["failed"] = True


def turn_failed() -> bool:
    return bool((_turn_state.get() or {}).get("failed"))


# ------------------------------------------------------------------ units and spans


def _span_id() -> str:
    return os.urandom(8).hex()


def _iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def root_span() -> str | None:
    return _root.get()


@contextlib.contextmanager
def turn(turn_id):
    """Every event inside carries `turn_id` (per session) and a fresh `run_id` (`<pid>-<n>`, unique in
    the process); a fresh root span unless this turn is already open."""
    if _turn.get() == turn_id and _root.get():
        yield _root.get()
        return
    tokens = (_turn.set(turn_id), _root.set(_span_id()), _run.set(f"{os.getpid()}-{next(_runs)}"),
              _turn_state.set({}))
    try:
        yield _root.get()
    finally:
        _turn_state.reset(tokens[3])
        _run.reset(tokens[2])
        _root.reset(tokens[1])
        _turn.reset(tokens[0])


@contextlib.contextmanager
def run(run_id: str | None):
    """Group what happens inside under a caller's `run_id` (a host script's `/ingest` field)."""
    if not run_id:
        yield
        return
    token = _run.set(run_id)
    try:
        yield
    finally:
        _run.reset(token)


@contextlib.contextmanager
def span(kind: str, /, **fields):
    """A child span of the current root: yields a dict the body fills; one event on exit with
    `span_id`, `parent_id`, `start`, `duration_ms`, `status` (error when the body raised)."""
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
        ms = round((time.monotonic() - t0) * 1000)
        out.setdefault("ms", ms)
        event(kind, span_id=_span_id(), parent_id=parent, start=start, duration_ms=ms, status=status, **out)


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
    n = next(_ids)
    iid = f"{os.getpid()}-{n}"
    state = {"ops_applied": 0, "ops_rejected": [], "pages": [], "already": False, "degraded": False}
    sid, parent, start, t0 = _span_id(), _root.get(), _iso(), time.monotonic()
    tokens = (_ingest.set((iid, state)), _root.set(sid), _run.set(_run.get() or f"{os.getpid()}-{next(_runs)}"))
    shape = {"span_id": sid, "parent_id": parent, "start": start}
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
        ms = round((time.monotonic() - t0) * 1000)
        event("ingest_end", **shape, duration_ms=ms, status=status, ms=ms,
              **{k: v for k, v in state.items() if not k.startswith("_")})
        _run.reset(tokens[2])
        _root.reset(tokens[1])
        _ingest.reset(tokens[0])
