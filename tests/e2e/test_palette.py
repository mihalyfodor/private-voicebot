"""PRD 30 — the ⌘K palette in a real browser (scenario 7).

Run: python -m pytest -m e2e tests/e2e
"""
import pytest

pytestmark = pytest.mark.e2e


def test_cmd_k_type_a_page_enter_opens_it_in_the_wiki(page, shot):
    page.wait_for_function("document.getElementById('status').textContent.startsWith('type a message')",
                           timeout=30_000)
    assert page.locator("#palette").is_hidden()
    page.keyboard.press("Meta+k")
    page.wait_for_selector("#palette:not([hidden])")
    assert page.evaluate("document.activeElement.id") == "palette-input"
    page.keyboard.type("atlas")
    first = page.locator(".palette-row").first
    assert first.locator(".palette-label").inner_text() == "Atlas"
    assert first.locator(".palette-detail").inner_text() == "projects/atlas"
    shot("open")
    page.keyboard.press("Enter")
    page.wait_for_selector("#palette[hidden]", state="attached")
    page.wait_for_selector("#view-wiki:not([hidden])")
    page.wait_for_selector("#wiki-text h1")
    assert page.locator("#wiki-text h1").inner_text() == "Atlas"
    assert page.locator('.rail-item[data-view="wiki"].active').count() == 1
    shot("opened")
    # ⌘K again from the wiki view opens an empty palette; Esc closes it and runs nothing
    page.keyboard.press("Meta+k")
    page.wait_for_selector("#palette:not([hidden])")
    assert page.locator("#palette-input").input_value() == ""
    page.keyboard.press("Escape")
    page.wait_for_selector("#palette[hidden]", state="attached")
    assert page.locator("#wiki-text h1").inner_text() == "Atlas"
