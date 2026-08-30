"""Service health (PRD 41): is the LLM (oMLX) and the speech service (ASR) there?

`probe()` asks each service once; `refresh()` does that and remembers the answer; `start()`
re-probes on a daemon thread every PROBE_INTERVAL_S (and right away after `recheck()`, which the
turn path calls when an LLM / ASR call fails). A change of state goes to `on_change` — chatbot
pushes it to the page as `services`. `snapshot()` is what `/api/config` carries.

Each service is `{ok, label, detail, checked}`: `label` names the host piece and its port
("oMLX :8000", "ASR :8020"), `detail` says what was seen, `checked` is the probe time (ISO,
seconds). Before the first probe every service counts as ok, so nothing is refused on a cold start.
"""
import os
import threading
from datetime import datetime
from urllib.parse import urlparse

import requests

import asr
import llm

PROBE_INTERVAL_S = 30
TIMEOUT_S = 3
#: The command the banner tells the user to run; the container says the same by default.
START_HINT = os.getenv("START_HINT", "scripts/voicebot start")

_lock = threading.Lock()
_state: dict = {}            # the last probe, {} before the first
_wake = threading.Event()    # set by recheck(): the daemon probes now instead of at the next tick
_thread: threading.Thread | None = None


def _port_label(url: str, name: str) -> str:
    port = urlparse(url).port or (443 if url.startswith("https") else 80)
    return f"{name} :{port}"


def _llm_host() -> str:
    """OMLX_BASE_URL without its /v1 path — where /health lives."""
    u = urlparse(llm.OMLX_BASE_URL)
    return f"{u.scheme}://{u.netloc}"


def _result(ok: bool, label: str, detail: str) -> dict:
    return {"ok": ok, "label": label, "detail": detail,
            "checked": datetime.now().isoformat(timespec="seconds")}


def probe_llm() -> dict:
    """GET /health on the oMLX host, then /v1/models with the key: the configured model must be listed."""
    label = _port_label(llm.OMLX_BASE_URL, "oMLX")
    host = _llm_host()
    try:
        requests.get(f"{host}/health", timeout=TIMEOUT_S).raise_for_status()
    except requests.RequestException as exc:
        return _result(False, label, f"not answering ({exc.__class__.__name__})")
    try:
        r = requests.get(f"{llm.OMLX_BASE_URL.rstrip('/')}/models", timeout=TIMEOUT_S,
                         headers={"Authorization": f"Bearer {llm.OMLX_API_KEY}"})
        r.raise_for_status()
        ids = [m.get("id") for m in r.json().get("data", [])]
    except (requests.RequestException, ValueError) as exc:
        return _result(False, label, f"models list failed ({exc.__class__.__name__})")
    if llm.OMLX_MODEL not in ids:
        return _result(False, label, f"model {llm.OMLX_MODEL} not loaded")
    return _result(True, label, f"model {llm.OMLX_MODEL} loaded")


def probe_asr() -> dict:
    """GET /health on the ASR service when the backend is remote; in-process MLX is always there."""
    if asr.BACKEND != "remote":
        return _result(True, "ASR (in-process)", "mlx-whisper in this process")
    label = _port_label(asr.URL, "ASR")
    try:
        requests.get(f"{asr.URL}/health", timeout=TIMEOUT_S).raise_for_status()
    except requests.RequestException as exc:
        return _result(False, label, f"not answering ({exc.__class__.__name__})")
    return _result(True, label, "answering")


def probe() -> dict:
    return {"llm": probe_llm(), "asr": probe_asr()}


def snapshot() -> dict:
    """The last probe (a copy), `{}` before the first."""
    with _lock:
        return {k: dict(v) for k, v in _state.items()}


def message() -> dict:
    """The `services` message / the `/api/config.services` payload: the probe plus the hint."""
    return {**snapshot(), "hint": START_HINT}


def llm_ok() -> bool:
    """False only when the last probe saw the LLM down; unknown (never probed) counts as ok."""
    with _lock:
        return _state.get("llm", {}).get("ok", True)


def _same(a: dict, b: dict) -> bool:
    """Equal apart from the probe time: what "no change" means for the push."""
    strip = lambda s: {k: {kk: vv for kk, vv in v.items() if kk != "checked"} for k, v in s.items()}
    return strip(a) == strip(b)


def refresh(on_change=None) -> dict:
    """Probe now, remember it, tell `on_change(message)` when the state differs from the last."""
    global _state
    fresh = probe()
    with _lock:
        changed = not _same(_state, fresh)
        _state = fresh
    if changed:
        for name, svc in fresh.items():
            print(f"[Services] {name}: {'ok' if svc['ok'] else 'DOWN'} — {svc['label']} {svc['detail']}")
        if on_change is not None:
            on_change(message())
    return fresh


def recheck() -> None:
    """Ask the daemon for an immediate re-probe (after a failed LLM / ASR call)."""
    _wake.set()


def _loop(on_change) -> None:
    while True:
        try:
            refresh(on_change)
        except Exception as e:      # a probe bug must not kill the daemon
            print(f"[Services] probe failed: {e}")
        _wake.wait(timeout=PROBE_INTERVAL_S)
        _wake.clear()


def start(on_change=None) -> None:
    """Start the re-probe daemon once; the first probe runs right away on that thread."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(target=_loop, args=(on_change,), daemon=True, name="services")
    _thread.start()


def reset() -> None:
    """Forget the last probe (tests)."""
    global _state
    with _lock:
        _state = {}
    _wake.clear()
