"""The FastAPI app, its startup and shutdown, the config and run-page routes, and `main()` (PRD 50 step 11, from chatbot.py)."""
import os
import asyncio
from core import trace
import threading
import webbrowser
from agent import conversation as a_conversation
from agent import prompt as a_prompt
from memory import store as m_store
from agent import runs
from vault import chunks
from vault import search as v_search
from vault import store as v_store
from core import config, formats
from core import settings
from server import services
from tools import weather as weather_tool
from server import backdrops
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from server import panels
from server import speech

log = trace.logger("App")


PORT = int(os.getenv("PORT", "8010"))
HOST = os.getenv("HOST", "127.0.0.1")  # loopback by default; set HOST=0.0.0.0 to expose on the LAN
STATIC_DIR = os.path.join(config.ROOT, "static")


# ---------------------------------------------------------------- app


app = FastAPI()


@app.on_event("startup")
async def _on_startup():
    """The search index's consistency check (docs/53-scalable-index.md): an edit made while the app
    was down, or a write that missed its hook, is repaired before the first turn."""
    try:
        repaired = v_search.startup_check()
    except Exception as e:      # noqa: BLE001 — the app still starts; search repairs on its next open
        log.exception("search index check failed")
        trace.error("index", "startup_check", e, fatal_for="none")
        return
    if repaired:
        log.info(f"search index: repaired {repaired} documents")


@app.on_event("shutdown")
async def _on_shutdown():
    """Runs on graceful shutdown (SIGTERM/SIGINT/UI shutdown) — the place session memory is saved."""
    try:
        a_conversation.save_memory()
    except Exception as e:
        panels._memory_failed(e)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def _no_static_cache(request: Request, call_next):
    """Static files are served fresh: fetch-loaded model json / backdrops must not go stale."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


def current_backdrop() -> str:
    try:
        return backdrops.validate(settings.load_settings().get("backdrop") or backdrops.DEFAULT)
    except ValueError:
        return backdrops.DEFAULT


#: Shown read-only in the Settings view — the command that takes a screenshot of the app.
SCREENSHOT_CMD = ".venv/bin/python scripts/screenshot.py /tmp/shot.png"


@app.get("/api/config")
async def api_config():
    """Everything the UI needs at boot: what to render, what is switched on, what to display."""
    return {
        # who is on stage (Wanko only since PRD 23) and the backdrop behind her
        "avatar": speech.AVATAR["key"], "name": speech.AVATAR["name"],
        "backdrop": current_backdrop(), "backdrops": backdrops.listing(),
        # session state the UI mirrors
        "hands_free": panels.hands_free,
        "verbosity": a_prompt.current_verbosity(),
        # what deploys check: which build is running, and whether a turn is in flight (PRD 58)
        "version": trace._version(), "busy": panels.is_busy(),
        # read-only facts shown in the Settings view
        "stale_days": dict(m_store.STALE_DAYS),
        "calendar_horizon_days": formats.calendar_horizon_days(),
        "recall_min_score": chunks.recall_min_score(),
        "location": weather_tool.location(),
        "vault": v_store.WIKI_PATH,
        "screenshot_cmd": SCREENSHOT_CMD,
        # the last service probe (PRD 41); {} plus the hint before the first one
        "services": services.message(),
        # whether the oMLX admin password is known and where from (never the value)
        **services.admin_password_status(),
    }


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/runs/{run_id}")
async def run_page(run_id: str, request: Request):
    """One run's page (docs/51-run-cards.md), built from its events and the vault's git — never
    page text; the same origin check as `/ingest`. An unknown, aged-out or malformed id is a 404."""
    if not ws._origin_allowed(request.headers.get("origin")):
        raise HTTPException(status_code=403, detail="bad origin")
    events = await asyncio.to_thread(runs.load, run_id) if s_ingest.RUN_ID.fullmatch(run_id) else []
    if not events:
        return HTMLResponse(runs.missing(run_id), status_code=404)
    return HTMLResponse(await asyncio.to_thread(lambda: runs.render(runs.build(events))))


def env_flag(name: str, default: str = "1") -> bool:
    """Env var as a boolean: `0`, `false`, `no` (any case) are off, everything else is on."""
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no")


# Off with OPEN_BROWSER=0: a restart during development should not steal focus.
OPEN_BROWSER = env_flag("OPEN_BROWSER")


def open_browser():
    import time
    time.sleep(1)
    webbrowser.open(f"http://localhost:{PORT}")


def main() -> None:
    """`python chatbot.py`: load Kokoro, start the service probe, serve until stopped."""
    from kokoro_onnx import Kokoro
    log.info("Loading Kokoro TTS...")
    speech.kokoro = Kokoro(speech.KOKORO_MODEL, speech.KOKORO_VOICES)
    services.start(panels._push_services)
    log.info("Ready.")
    if OPEN_BROWSER:
        threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    log.info("Exiting...")
    trace.event("service", event="stop")


# Imported last: these modules register their routes on `app`, which must exist by then, and read this module's
# constants only inside functions.
from server import ingest as s_ingest  # noqa: E402
from server import ws  # noqa: E402
