"""PRD 25 — the Sources inbox in a real browser, offline: a manifest line, a raw file, a source
page and a fed project page are seeded straight into the server's temp vault (no LLM needed).

Run: python -m pytest -m e2e tests/e2e
"""
import json

import pytest

from tests.e2e.conftest import seed_page

pytestmark = pytest.mark.e2e

RAW = "raw/inbox/2026-08-20-falcon.md"
SHA = "ab" * 32
NOTE_FROM_SOURCE = f"- 2026-08-20 Load test showed 940 ms p99. (source: {RAW})"
NOTE_KEPT = "- 2026-08-21 Kickoff went fine. (source: user note 2026-08-21)"


def seed_source(vault) -> None:
    (vault / "raw/inbox").mkdir(parents=True, exist_ok=True)
    (vault / RAW).write_text("# Falcon\n\nload test: 940 ms p99\n")
    (vault / "raw/manifest.jsonl").write_text(json.dumps({
        "sha256": SHA, "path": RAW, "kind": "md", "ingested_at": "09:12:00",
        "page": "sources/falcon-status", "pages_touched": ["projects/falcon"]}) + "\n")
    (vault / "sources").mkdir(exist_ok=True)
    (vault / "sources/falcon-status.md").write_text(
        "---\ntype: source\nname: Falcon status\ntags: []\nsource: user drop 2026-08-20\n"
        f"updated: 2026-08-20\nseen: 2026-08-20\nkind: md\nraw: {RAW}\nsha256: {SHA}\n---\n"
        "# Falcon status\n\n## Summary\nProject Falcon is amber; the load test showed 940 ms p99.\n\n"
        "## Entities\n- [[projects/falcon]] — Falcon (project)\n\n## Fed\n- [[projects/falcon]]\n")
    (vault / "projects").mkdir(exist_ok=True)
    (vault / "projects/falcon.md").write_text(seed_page("project", "Falcon", [],
        f"# Falcon\n\n## Summary\nPlatform project, amber.\n\n## Notes\n{NOTE_FROM_SOURCE}\n{NOTE_KEPT}\n\n"
        "## Related\n\n## History\n"))


def open_sources_empty(page):
    page.click('.rail-item[data-view="sources"]')
    page.wait_for_selector(".source-row", state="detached")


def open_sources(page):
    page.click('.rail-item[data-view="sources"]')
    page.wait_for_selector("#wiki-list .source-row")


# --------------------------------------------------------------------------- 5. list + select

def test_sources_list_shows_the_seeded_source(page, shot, server):
    _, vault = server
    seed_source(vault)
    open_sources(page)
    row = page.locator(".source-row").first
    assert "Falcon status" in row.inner_text()
    assert "2026-08-20 09:12:00 · 1 page fed" in row.inner_text()
    assert row.locator(".source-kind").inner_text() == "md"
    assert page.locator(".source-kinds .chip").all_inner_texts() == ["md"]
    row.click()
    page.wait_for_selector(".source-forget")
    assert "940 ms p99" in page.locator("#wiki-text").inner_text()
    assert page.locator(".source-open-raw").get_attribute("href").endswith("file=" + RAW.replace("/", "%2F"))
    shot("selected")


# --------------------------------------------------------------------------- 5. forget

def test_forget_removes_the_source_and_only_its_note(page, shot, server):
    """PRD 52: Forget proposes (no dialog); the toast leads to the awaiting card, Approve deletes."""
    _, vault = server
    seed_source(vault)
    open_sources(page)
    page.locator(".source-row").first.click()
    page.wait_for_selector(".source-forget")
    dialogs = []
    page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
    page.click(".source-forget")
    page.wait_for_selector("#intake-toast.toast-link")
    assert (vault / RAW).exists()                    # a proposal deletes nothing
    page.click("#intake-toast")
    page.wait_for_selector(".msg.card .card-approve")
    shot("forget-proposal")
    page.click(".msg.card .card-approve")
    page.wait_for_selector(".msg.card .run-chip.run-done")
    assert dialogs == []
    open_sources_empty(page)
    assert page.locator("#wiki-list").inner_text().strip() == "nothing filed yet"
    shot("forgotten")
    assert not (vault / "sources/falcon-status.md").exists()
    assert not (vault / RAW).exists()
    assert (vault / "raw/manifest.jsonl").read_text() == ""
    falcon = (vault / "projects/falcon.md").read_text()
    assert NOTE_FROM_SOURCE not in falcon
    assert NOTE_KEPT in falcon


# --------------------------------------------------------------------------- rail badge

def test_rail_badge_counts_new_sources_until_the_view_is_opened(page, shot):
    page.click('.rail-item[data-view="wiki"]')
    page.wait_for_selector("#wiki-list .wiki-row")
    page.evaluate("setSources({sources: [{sha: 'new1', path: 'sources/x', raw: 'raw/inbox/x.md', "
                  "kind: 'note', title: 'X', when: '2026-08-29', fed: []}]})")
    badge = page.locator('.rail-item[data-view="sources"] .rail-badge')
    assert badge.is_visible() and badge.inner_text() == "1"
    shot("badge")
    page.click('.rail-item[data-view="sources"]')
    assert badge.is_hidden()


# --------------------------------------------------------------------------- PRD 31: drop a .docx

# Playwright's set_input_files only works on <input type=file>; the workspace is a drop zone, so
# the file is rebuilt from base64 inside the page and dispatched as a DragEvent (same as
# scripts/screenshot.py --drop-file).
DROP_JS = """
(files) => {
  const dt = new DataTransfer();
  for (const [name, mime, b64] of files) {
    const bin = atob(b64);
    const buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    dt.items.add(new File([buf], name, { type: mime }));
  }
  const target = document.getElementById('workspace');
  for (const type of ['dragenter', 'dragover', 'drop']) {
    target.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: dt }));
  }
}
"""
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def drop_files(page, files: list) -> None:
    """One drop of several `(name, data, mime)` files onto the workspace."""
    import base64
    page.evaluate(DROP_JS, [[name, mime, base64.b64encode(data).decode()] for name, data, mime in files])


def drop_file(page, name: str, data: bytes, mime: str) -> None:
    drop_files(page, [(name, data, mime)])


def test_a_dropped_docx_lands_in_sources_with_its_kind(page, shot, server):
    """PRD 31 scenario 8: a generated .docx through the real drop → intake → /ingest → Sources.

    No model is running behind the e2e server, so the page is filed as unreadable — the raw
    file, the manifest line and the `docx` kind chip are what this checks.
    """
    from tests.ingest.test_office_formats import docx_bytes
    _, vault = server
    data = docx_bytes([("h1", "Falcon plan"), ("p", "Go-live is September 15."),
                       ("table", [["Risk", "Owner"], ["load test", "Bob"]])])

    drop_file(page, "falcon-plan.docx", data, DOCX_MIME)
    page.wait_for_selector("#intake:not([hidden])")
    assert page.input_value("#intake-title") == "falcon-plan"          # the file's own name
    assert "falcon-plan.docx" in page.locator("#intake-items").inner_text()
    shot("intake")
    page.click("#intake-file")
    page.wait_for_selector("#intake", state="hidden", timeout=60_000)   # the model call fails fast

    open_sources(page)
    row = page.locator(".source-row").first
    assert row.locator(".source-kind").inner_text() == "docx"
    assert "docx" in page.locator(".source-kinds .chip").all_inner_texts()
    row.click()
    page.wait_for_selector(".source-forget")
    shot("docx-filed")
    raw = [line for line in (vault / "raw/manifest.jsonl").read_text().splitlines() if line]
    entry = json.loads(raw[-1])
    assert entry["kind"] == "docx" and entry["path"].endswith("-falcon-plan.docx")
    assert (vault / entry["path"]).read_bytes() == data
    assert "kind: docx" in (vault / f"{entry['page']}.md").read_text()


# --------------------------------------------------------------------------- PRD 36: a two-file batch


def test_a_two_file_drop_files_one_at_a_time_and_shows_each_row(page, shot, server):
    """PRD 36 scenario 3: two generated .md files → two rows reach `filed`, the popup stays
    open with its summary line, Close hides it, both land in Sources."""
    _, vault = server
    drop_files(page, [("eagle-notes.md", b"# Eagle\n\nEagle kicked off on Monday.\n", "text/markdown"),
                      ("hawk-notes.md", b"# Hawk\n\nHawk ships on Friday.\n", "text/markdown")])
    page.wait_for_selector("#intake:not([hidden])")
    assert page.locator("#intake-note").inner_text().endswith("already filed is a no-op")
    page.click("#intake-file")
    page.wait_for_function("document.querySelectorAll('#intake .intake-item.st-filed').length === 2",
                           timeout=60_000)
    states = page.locator("#intake .intake-item-state").all_inner_texts()
    assert all(s.startswith("filed → ") for s in states) and len(states) == 2
    assert page.locator("#intake-status").inner_text().startswith("2 filed ·")
    assert page.inner_text("#intake-cancel") == "Close"
    shot("batch-filed")
    page.click("#intake-cancel")
    page.wait_for_selector("#intake", state="hidden")

    open_sources(page)
    titles = page.locator(".source-row").all_inner_texts()
    assert any("eagle" in t.lower() for t in titles) and any("hawk" in t.lower() for t in titles)
    lines = [json.loads(line) for line in (vault / "raw/manifest.jsonl").read_text().splitlines() if line]
    # raw/inbox/<date>-<name>: one manifest line per file, in drop order
    assert [e["path"].split("-", 3)[-1] for e in lines[-2:]] == ["eagle-notes.md", "hawk-notes.md"]
