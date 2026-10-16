"""PRD 38 scenario 8 — focus context in a real browser against the real model: a guide filed as a
source, "summarise this" with it open names it; "go back to that page" from Chat reopens it.

Run: python -m pytest -m "e2e and live" tests/e2e/test_focus.py -s
The guide is a read-only copy of one raw file from ~/llm-wiki-dev (skipped when it is missing).
"""
import hashlib
import json
import pathlib

import pytest

from tests.e2e.test_app import IDLE, omlx_up, wait_idle

pytestmark = [pytest.mark.e2e, pytest.mark.live]

GUIDE = pathlib.Path("~/llm-wiki-dev/raw/inbox/2026-08-30-morrowind-game-of-the-year-edition-guide.md").expanduser()
RAW = "raw/inbox/2026-08-30-morrowind-guide.md"
PAGE = "sources/morrowind-guide"
GUIDE_BYTES = 200_000       # the first part of the guide is plenty for one focus block


def seed_guide(vault: pathlib.Path) -> None:
    text = GUIDE.read_bytes()[:GUIDE_BYTES].decode(errors="ignore")
    (vault / "raw/inbox").mkdir(parents=True, exist_ok=True)
    (vault / RAW).write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    with open(vault / "raw/manifest.jsonl", "a") as f:
        f.write(json.dumps({"sha256": sha, "path": RAW, "kind": "md", "ingested_at": "09:12:00",
                            "page": PAGE, "pages_touched": []}) + "\n")
    (vault / "sources").mkdir(exist_ok=True)
    (vault / (PAGE + ".md")).write_text(
        "---\ntype: source\nname: Morrowind guide\ntags: [gaming]\nsource: user drop 2026-08-30\n"
        f"updated: 2026-08-30\nseen: 2026-08-30\nkind: md\nraw: {RAW}\nsha256: {sha}\n---\n"
        "# Morrowind guide\n\n## Summary\nA walkthrough of The Elder Scrolls III: Morrowind, Game of the "
        "Year Edition: the main quest step by step, from Seyda Neen to the Tribunal and Bloodmoon expansions.\n\n"
        "## Entities\n\n## Fed\n")


def ask(page, text: str, shot, name: str) -> str:
    """Type `text`, wait for the reply to finish, return the assistant's last line."""
    before = page.locator("#history .msg.assistant").count()
    page.fill("#composer-input", text)
    page.keyboard.press("Enter")
    page.wait_for_function(
        f"document.getElementById('status').textContent.startsWith({IDLE!r})"
        f" && document.querySelectorAll('#history .msg.assistant').length > {before}", timeout=180_000)
    shot(name)
    return page.locator("#history .msg.assistant .text").last.inner_text().strip()


def test_summarise_this_names_the_open_guide_and_go_back_reopens_it(page, shot, server):
    if not omlx_up():
        pytest.skip("oMLX not reachable on localhost:8000")
    if not GUIDE.exists():
        pytest.skip(f"{GUIDE} missing")
    _, vault = server
    seed_guide(vault)
    wait_idle(page)

    page.click('.rail-item[data-view="sources"]')
    page.wait_for_selector("#wiki-list .source-row")
    page.evaluate(f"selectWikiPage({PAGE!r})")
    page.wait_for_function("document.getElementById('composer-input').placeholder === 'ask about Morrowind guide…'")
    shot("guide-open")

    summary = ask(page, "summarise this", shot, "summary")
    print(f"\nsummarise this -> {summary}")
    assert "morrowind" in summary.lower()

    page.click('.rail-item[data-view="chat"]')
    page.wait_for_function("document.getElementById('composer-input').placeholder === 'type a message…'")
    back = ask(page, "go back to that page", shot, "back")
    print(f"go back to that page -> {back}")
    assert page.evaluate("currentView") == "sources"
    assert page.evaluate("wikiSelected") == PAGE
    assert page.evaluate("document.getElementById('composer-input').placeholder") == "ask about Morrowind guide…"
