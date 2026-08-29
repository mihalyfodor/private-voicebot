"""Headless screenshot of the running app for visual checks (dev only).

    .venv/bin/python scripts/screenshot.py /tmp/chat.png [--view memory] [--panel] [--pin]
        [--drag FROM TO] [--drop-file /tmp/note.md] [--width 1440] [--height 900]
"""
import argparse, base64, mimetypes, pathlib, sys
from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8010/")
ap.add_argument("--view", default=None, help="rail item to open, e.g. memory / wiki / sources / settings")
ap.add_argument("--panel", action="store_true", help="open the contextual panel for the view")
ap.add_argument("--pin", action="store_true", help="open the panel and pin it (pushes the layout)")
ap.add_argument("--width", type=int, default=1440)
ap.add_argument("--height", type=int, default=900)
ap.add_argument("--drag", nargs=2, metavar=("FROM", "TO"), default=None,
                help="drag one selector onto another before the screenshot")
ap.add_argument("--drop-file", default=None,
                help="dispatch a synthetic file drop of this path onto #workspace before the screenshot")
ap.add_argument("--wait", type=int, default=2500, help="ms to wait for the avatar to load")
a = ap.parse_args()

# Playwright's set_input_files only works on <input type=file>; a drop zone needs a real
# DragEvent, so the file is rebuilt from base64 inside the page and put on a DataTransfer.
DROP_JS = """
([name, mime, b64]) => {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  const file = new File([buf], name, { type: mime });
  const dt = new DataTransfer();
  dt.items.add(file);
  const target = document.getElementById('workspace');
  for (const type of ['dragenter', 'dragover', 'drop']) {
    target.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: dt }));
  }
}
"""

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist"])
    page = browser.new_page(viewport={"width": a.width, "height": a.height})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)

    def click_rail(name, wait=800):
        page.click(f'.rail-item[data-view="{name}"]')
        page.wait_for_timeout(wait)

    page.goto(a.url)
    page.wait_for_timeout(a.wait)
    view = a.view or "chat"
    if a.view:
        click_rail(view)
    # The panel is toggled by clicking the rail item you are already on; the pin button
    # is inside it, so --pin implies --panel. NB: pinning is persisted in settings.json.
    if a.panel or a.pin:
        if page.is_hidden("#panel-wrap"):
            click_rail(view, wait=400)
        if bool(page.locator("#pin-btn.on").count()) != a.pin:
            page.click("#pin-btn")
        page.wait_for_timeout(800)
    if a.drop_file:
        path = pathlib.Path(a.drop_file)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        page.evaluate(DROP_JS, [path.name, mime, base64.b64encode(path.read_bytes()).decode()])
        page.wait_for_timeout(800)
    if a.drag:
        page.drag_and_drop(a.drag[0], a.drag[1])
        page.wait_for_timeout(800)
    page.screenshot(path=a.out)
    browser.close()
print(f"saved {a.out}")
for e in errors:
    print("console error:", e, file=sys.stderr)
