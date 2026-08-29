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
    _, vault = server
    seed_source(vault)
    open_sources(page)
    page.locator(".source-row").first.click()
    page.wait_for_selector(".source-forget")
    page.once("dialog", lambda d: d.accept())
    page.click(".source-forget")
    page.wait_for_selector(".source-row", state="detached")
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
