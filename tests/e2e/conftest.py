"""PRD 29 — Playwright e2e fixtures: one real server per session, headless Chromium per test.

The server is `chatbot.py` as a subprocess against a temp vault, temp settings/memory files and a
free port, so nothing here can touch ~/llm-wiki, settings.json or memory.json. Screenshots go to
/tmp/e2e/<test>-<name>.png.
"""
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SHOTS = pathlib.Path("/tmp/e2e")
PAGE_TIMEOUT_MS = 10_000
START_TIMEOUT_S = 90        # Kokoro load + uvicorn
# Same flags as scripts/screenshot.py: software WebGL so the Live2D stage renders headless.
CHROMIUM_ARGS = ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist",
                 "--autoplay-policy=no-user-gesture-required"]


ASSETS = ["static/vendor/pixi.min.js", "static/models/wanko"]


@pytest.fixture(scope="session", autouse=True)
def fetched_assets():
    """Fail early and clearly: without pixi + the Wanko model the page only shows console errors."""
    missing = [rel for rel in ASSETS if not (ROOT / rel).exists()]
    if missing:
        pytest.fail(f"missing {', '.join(missing)} — run scripts/fetch_assets.sh", pytrace=False)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def seed_page(page_type: str, name: str, tags: list, body: str) -> str:
    return (f"---\ntype: {page_type}\nname: {name}\ntags: [{', '.join(tags)}]\nsource: seed\n"
            f"updated: 2026-08-01\nseen: 2026-08-01\n---\n\n{body}")


SEED = {
    "people/anna.md": seed_page("person", "Anna", ["platform", "team"],
        "# Anna\n\n## Summary\nColleague on the [[projects/atlas|Atlas project]], works with [[people/bob]].\n\n"
        "## Notes\n- 2026-08-01: kickoff #platform\n\n## Related\n\n## History\n"),
    "projects/atlas.md": seed_page("project", "Atlas", ["platform"],
        "# Atlas\n\n## Summary\nData platform rebuild, led by [[people/anna]] and [[people/anna#Summary]].\n\n"
        "## Notes\n\n## Related\n- ![[people/anna]]\n\n## History\n"),
    "notes/scratch.md": seed_page("note", "Scratch", ["misc"],
        "# Scratch\n\n## Summary\nLoose ends, nothing to do with the platform.\n\n## Notes\n\n## Related\n\n## History\n"),
}


def seed_vault(vault: pathlib.Path) -> None:
    for rel, text in SEED.items():
        path = vault / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def wait_for(url: str, timeout: float, proc: subprocess.Popen) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}")
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"server did not answer on {url} within {timeout}s")


@pytest.fixture(scope="session")
def server():
    """(base_url, vault_path) of a chatbot.py started against a temp vault + temp settings/memory."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="voicebot-e2e-"))
    vault = tmp / "wiki"
    seed_vault(vault)
    (tmp / "memory").mkdir()
    port = free_port()
    env = {**os.environ, "WIKI_PATH": str(vault), "SETTINGS_PATH": str(tmp / "settings.json"),
           "MEMORY_DIR": str(tmp / "memory"), "SESSION_LOG": str(tmp / "session.log"), "GMAIL_DIR": str(tmp),
           "PORT": str(port), "OPEN_BROWSER": "0", "PYTHONUNBUFFERED": "1"}
    log = open(tmp / "server.log", "w")
    proc = subprocess.Popen([sys.executable, "chatbot.py"], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT)
    base_url = f"http://127.0.0.1:{port}"
    try:
        wait_for(base_url + "/", START_TIMEOUT_S, proc)
    except Exception:
        proc.terminate()
        log.close()
        raise RuntimeError((tmp / "server.log").read_text())
    yield base_url, vault
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    if not os.environ.get("E2E_KEEP"):
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(args=CHROMIUM_ARGS)
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    """A fresh tab on the app, already loaded; fails the test on any page/console error."""
    base_url, _ = server
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    pg = context.new_page()
    pg.set_default_timeout(PAGE_TIMEOUT_MS)
    errors = []
    pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    pg.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error" else None)
    pg.goto(base_url + "/", timeout=30_000)
    yield pg
    context.close()
    assert not errors, "browser errors:\n" + "\n".join(errors)


@pytest.fixture
def shot(page, request):
    """shot("name") -> /tmp/e2e/<test>-<name>.png (overwritten)."""
    SHOTS.mkdir(parents=True, exist_ok=True)
    test = request.node.name

    def take(name: str) -> pathlib.Path:
        out = SHOTS / f"{test}-{name}.png"
        page.screenshot(path=str(out))
        return out

    return take
