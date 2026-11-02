"""PRD 29 — browser e2e over a real chatbot.py. Scenario numbers are the PRD's.

Run: python -m pytest -m e2e tests/e2e        (live turn: -m "e2e and live")
"""
import subprocess

import pytest

pytestmark = pytest.mark.e2e

IDLE = "type a message or talk"


# --------------------------------------------------------------------------- helpers

def commits(vault) -> list:
    out = subprocess.run(["git", "log", "--oneline"], cwd=vault, capture_output=True, text=True)
    return [l for l in out.stdout.splitlines() if l.strip()]


def wait_idle(page, timeout=30_000):
    page.wait_for_function(f"document.getElementById('status').textContent.startsWith({IDLE!r})", timeout=timeout)


def open_wiki(page, path=None):
    """Open the Wiki view (waits for the list), optionally a page (waits for its editor to be ready)."""
    page.click('.rail-item[data-view="wiki"]')
    page.wait_for_selector("#wiki-list .wiki-row")
    if path:
        page.evaluate(f"selectWikiPage({path!r})")
        page.wait_for_selector(".wiki-edit")


def start_edit(page, extra_line: str):
    page.click(".wiki-edit")
    page.wait_for_selector("#wiki-editor:not([hidden])")
    page.focus("#wiki-editor")
    page.keyboard.press("Control+End")
    page.keyboard.type("\n" + extra_line)
    page.wait_for_selector(".wiki-save:not([disabled])")


def wait_view_mode(page):
    page.wait_for_selector("#wiki-editor[hidden]", state="attached")
    page.wait_for_selector("#wiki-text:not([hidden])")


def accept_dialog(page, value=None):
    page.once("dialog", lambda d: d.accept(value) if value is not None else d.accept())


# --------------------------------------------------------------------------- 1. load

def test_app_loads(page, shot):
    assert page.locator("canvas#stage").count() == 1
    assert page.locator("#avatar-note").is_hidden()
    wait_idle(page)   # the greeting turn has finished (canned, Kokoro only)
    page.wait_for_timeout(1500)   # let the model finish its first frames
    shot("loaded")
    assert page.locator("#history .msg.assistant").count() == 1


# --------------------------------------------------------------------------- 2. render

def test_wiki_page_renders(page, shot):
    open_wiki(page, "people/anna")
    shot("wiki")
    text = page.locator("#wiki-text")
    assert text.locator("h1").inner_text() == "Anna"
    assert "Summary" in text.locator("h2").first.inner_text()
    alias = text.locator(".wiki-link", has_text="Atlas project")
    assert alias.count() == 1 and alias.get_attribute("data-target") == "projects/atlas"
    assert text.locator(".wiki-link.missing[data-target='people/bob']").count() == 1
    assert page.locator("#wiki-backlinks").is_visible()
    assert "projects/atlas" in page.locator("#wiki-backlinks").inner_text()
    assert page.locator("#wiki-fm").is_visible()


# --------------------------------------------------------------------------- 3. save

def test_edit_and_save(page, shot, server):
    _, vault = server
    open_wiki(page, "projects/atlas")
    before = commits(vault)
    start_edit(page, "- typed via keyboard")
    shot("editing")
    page.keyboard.press("Meta+s")
    wait_view_mode(page)
    shot("saved-cmd-s")
    disk = (vault / "projects/atlas.md").read_text()
    assert "- typed via keyboard" in disk
    assert len(commits(vault)) == len(before) + 1
    assert "edit projects/atlas" in commits(vault)[0]

    start_edit(page, "- saved via button")
    page.click(".wiki-save")
    wait_view_mode(page)
    shot("saved-button")
    disk = (vault / "projects/atlas.md").read_text()
    assert "- typed via keyboard" in disk and "- saved via button" in disk
    assert len(commits(vault)) == len(before) + 2
    assert "seed" in disk and "[[people/anna#Summary]]" in disk   # byte-for-byte apart from updated:


# --------------------------------------------------------------------------- 4. conflict

def test_stale_etag_conflict(page, shot, server):
    _, vault = server
    open_wiki(page, "notes/scratch")
    start_edit(page, "- browser edit")
    on_disk = vault / "notes/scratch.md"
    original = on_disk.read_text()
    on_disk.write_text(original.replace("Loose ends", "Loose ends (changed on disk)"))
    before = commits(vault)
    page.keyboard.press("Meta+s")
    page.wait_for_selector("#wiki-conflict:not([hidden])")
    shot("conflict")
    assert page.locator("#wiki-conflict").is_visible()
    assert "changed on disk" in on_disk.read_text() and "browser edit" not in on_disk.read_text()
    assert commits(vault) == before
    # reload takes the disk version and drops the draft
    page.click("#wiki-conflict-reload")
    page.wait_for_selector("#wiki-conflict[hidden]", state="attached")
    assert "changed on disk" in page.locator("#wiki-editor").input_value()


# --------------------------------------------------------------------------- 5. red link

def test_red_link_creates_page(page, shot, server):
    _, vault = server
    open_wiki(page, "people/anna")
    red = page.locator(".wiki-link.missing[data-target='people/bob']")
    accept_dialog(page)
    red.click()
    page.wait_for_selector(".wiki-title:has-text('bob')")
    page.wait_for_selector(".wiki-edit")
    shot("created")
    assert (vault / "people/bob.md").exists()
    assert "type: person" in (vault / "people/bob.md").read_text()
    assert "create people/bob" in commits(vault)[0]
    page.evaluate("selectWikiPage('people/anna')")
    page.wait_for_selector(".wiki-link[data-target='people/bob']:not(.missing)")
    assert page.locator(".wiki-link.missing").count() == 0


# --------------------------------------------------------------------------- 8. tags

def test_tags_group_filters(page, shot):
    open_wiki(page)
    rows = page.locator("#wiki-list .wiki-row:not(.wiki-tag-row)")
    assert rows.count() >= 3
    page.locator(".wiki-tag-row", has_text="#team").click()
    page.wait_for_selector(".wiki-tag-row.selected")
    shot("filtered")
    titles = page.locator("#wiki-list .wiki-row:not(.wiki-tag-row) .wiki-row-title").all_inner_texts()
    assert titles == ["Anna"]
    page.locator(".wiki-tag-row", has_text="#team").click()
    page.wait_for_function("!document.querySelector('.wiki-tag-row.selected')")
    assert page.locator("#wiki-list .wiki-row:not(.wiki-tag-row)").count() >= 3


# --------------------------------------------------------------------------- 9. search

def test_search_returns_hits(page, shot):
    open_wiki(page)
    page.fill("#wiki-search", "rebuild")
    page.wait_for_selector(".wiki-hit")
    shot("hits")
    hits = page.locator(".wiki-hit .wiki-row-title").all_inner_texts()
    assert hits == ["Atlas"]
    assert "rebuild" in page.locator(".wiki-hit-snippet").first.inner_text()
    page.fill("#wiki-search", "")
    page.wait_for_selector(".wiki-group-title:has-text('people')")


# --------------------------------------------------------------------------- 6. rename

def test_rename_rewrites_links(page, shot, server):
    _, vault = server
    open_wiki(page, "people/anna")
    page.click(".wiki-more")
    shot("menu")
    accept_dialog(page, "people/anna-k")
    page.locator(".wiki-menu-item", has_text="Rename/move").click()
    page.wait_for_selector(".wiki-path:has-text('people/anna-k')")
    page.wait_for_selector(".wiki-edit")
    shot("renamed")
    assert not (vault / "people/anna.md").exists()
    assert (vault / "people/anna-k.md").exists()
    atlas = (vault / "projects/atlas.md").read_text()
    assert "[[people/anna]]" not in atlas and "[[people/anna#Summary]]" not in atlas
    assert "[[people/anna-k]]" in atlas and "[[people/anna-k#Summary]]" in atlas and "![[people/anna-k]]" in atlas
    assert "rename people/anna" in commits(vault)[0]
    assert "projects/atlas" in page.locator("#wiki-backlinks").inner_text()


# --------------------------------------------------------------------------- 7. delete

def test_delete_page(page, shot, server):
    _, vault = server
    open_wiki(page, "notes/scratch")
    before = commits(vault)
    page.click(".wiki-more")
    accept_dialog(page)
    page.locator(".wiki-menu-item", has_text="Delete").click()
    page.wait_for_selector("#wiki-page-head:has-text('pick a page')")
    page.wait_for_function("!Array.from(document.querySelectorAll('.wiki-row-title')).some(e => e.textContent === 'Scratch')")
    shot("deleted")
    assert not (vault / "notes/scratch.md").exists()
    assert len(commits(vault)) == len(before) + 1
    assert "delete notes/scratch" in commits(vault)[0]


# --------------------------------------------------------------------------- 10. live

def omlx_up() -> bool:
    """Reachable at all — a 401 (key required) still means the server is there."""
    import urllib.error, urllib.request
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2).read()
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


@pytest.mark.live
def test_live_text_turn(page, shot):
    if not omlx_up():
        pytest.skip("oMLX not reachable on localhost:8000")
    wait_idle(page)
    page.wait_for_function("document.querySelectorAll('#history .msg.assistant').length >= 1")  # the greeting lands first
    page.fill("#composer-input", "Say hello in one short sentence.")
    page.keyboard.press("Enter")
    seen = set()
    page.wait_for_function("document.getElementById('status').textContent !== " + repr(IDLE))
    deadline = 120_000
    step = 250
    while deadline > 0:
        status = page.locator("#status").text_content()   # not inner_text: CSS uppercases it
        for name in ("processing", "thinking", "speaking"):
            if status.startswith(name) and name not in seen:
                seen.add(name)
                shot(name)
        if status.startswith(IDLE) and page.locator("#history .msg.assistant").count() >= 2:
            break
        page.wait_for_timeout(step)
        deadline -= step
    assert deadline > 0, "reply never finished"
    shot("reply")
    reply = page.locator("#history .msg.assistant .text").last.inner_text()
    assert reply.strip()
    assert "thinking" in seen or "speaking" in seen


# --------------------------------------------------------------------------- PRD 51, scenario 9

@pytest.mark.live
def test_live_briefing_card_opens_its_run_page(page, shot):
    """"what's new?" -> the run card's "Open run" opens /runs/<id> in a new tab, headed "Briefing"
    (docs/51-run-cards.md scenario 9). No model needed when nothing changed (status `nothing`)."""
    wait_idle(page)
    page.fill("#composer-input", "what's new?")
    page.keyboard.press("Enter")
    link = page.locator("#history .msg.card .card-open").last
    link.wait_for(timeout=120_000)
    assert link.get_attribute("target") == "_blank" and link.get_attribute("href").startswith("/runs/")
    shot("card")
    with page.context.expect_page() as opened:
        link.click()
    run_page = opened.value
    run_page.wait_for_load_state()
    assert run_page.locator("h1").inner_text() == "Briefing"
    assert run_page.locator(".run-chip").inner_text().lower() in ("done", "nothing", "failed")
    run_page.screenshot(path="/tmp/e2e/run-page.png", full_page=True)
