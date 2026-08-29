"""Headless screenshot of the running app for visual checks (dev only).

    .venv/bin/python scripts/screenshot.py /tmp/chat.png [--view memory] [--width 1440] [--height 900]
"""
import argparse, sys
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8010/")
ap.add_argument("--view", default=None, help="workspace tab to open, e.g. memory")
ap.add_argument("--width", type=int, default=1440)
ap.add_argument("--height", type=int, default=900)
ap.add_argument("--wait", type=int, default=2500, help="ms to wait for the avatar to load")
a = ap.parse_args()

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
    page = browser.new_page(viewport={"width": a.width, "height": a.height})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(a.url)
    page.wait_for_timeout(a.wait)
    if a.view:
        page.click(f'.view-tab[data-view="{a.view}"]')
        page.wait_for_timeout(800)
    page.screenshot(path=a.out)
    browser.close()
print(f"saved {a.out}")
for e in errors:
    print("console error:", e, file=sys.stderr)
