"""PRD 28 — wiki as a light editor. Tmp vault, no LLM. Scenario numbers are the PRD's (1-11).

The browser half (7, 8 and the editor flow) runs static/app.js in node against the DOM shim in
tests/app_fixture.mjs; this file asserts on the JSON it prints.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import shutil
import subprocess
import threading
import time

import pytest

import memory
import vault_lock
import wiki

ROOT = os.path.join(os.path.dirname(__file__), "..")
TODAY = memory._today_iso()


@pytest.fixture
def vault(wiki_sandbox, memory_sandbox):
    wiki.ensure_vault()
    return wiki_sandbox


@pytest.fixture
def editor(vault, panel):
    """The shared websocket client over a created vault."""
    return panel


def commits(path: str = "") -> list:
    args = ["git", "log", "--oneline"] + (["--", path] if path else [])
    out = subprocess.run(args, cwd=wiki.WIKI_PATH, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def read(rel: str) -> str:
    return wiki._read(rel)


def make(path, name=None, page_type="person", summary="Colleague."):
    op = {"op": "new_page", "path": path, "type": page_type,
          "name": name or path.split("/")[-1].title(), "summary": summary}
    applied, rejected = wiki.apply_page_ops([op], "s")
    assert applied, rejected


def with_body(rel: str, extra: str) -> str:
    """The page's current text with `extra` appended to the body — what a user save sends."""
    return read(rel).rstrip("\n") + "\n" + extra


# --------------------------------------------------------------- 1. round trip


def test_save_preserves_every_byte_except_the_updated_line(vault, monkeypatch):
    make("people/anna")
    make("projects/atlas", page_type="project")
    text = with_body("people/anna.md", "\n## Notes\n- see [[projects/atlas|the rewrite]] and "
                     "[[projects/atlas#Summary]]; ![[people/anna]] #infra `code #not-a-tag`\n")
    monkeypatch.setattr(memory, "_today_iso", lambda: "2031-01-01")

    result, reason = wiki.save_page("people/anna", text, wiki.etag(read("people/anna.md")))

    assert reason is None
    saved = read("people/anna.md")
    assert saved.replace("updated: 2031-01-01", f"updated: {TODAY}") == text
    assert "updated: 2031-01-01" in saved
    assert result == {"etag": wiki.etag(saved)}


# --------------------------------------------------------------- 2. etag: conflict vs commit


def test_stale_etag_is_a_conflict_and_a_fresh_one_commits_once(editor):
    make("people/anna")
    before, n = read("people/anna.md"), len(commits())
    text = with_body("people/anna.md", "\n## Notes\n- edited by hand\n")

    stale = editor.send({"action": "wiki_save", "path": "people/anna", "text": text, "etag": "nope"})
    assert stale == {"type": "wiki_conflict", "path": "people/anna", "text": before,
                     "etag": wiki.etag(before)}
    assert read("people/anna.md") == before
    assert len(commits()) == n

    page, listing = editor.send_many(
        {"action": "wiki_save", "path": "people/anna", "text": text, "etag": wiki.etag(before)}, 2)
    assert page["type"] == "wiki_page" and listing["type"] == "wiki"
    assert "- edited by hand" in read("people/anna.md")
    assert page["etag"] == wiki.etag(read("people/anna.md")) != wiki.etag(before)
    assert len(commits()) == n + 1
    assert f"- {TODAY} edited [[people/anna]] (source: user edit)" in read("log.md")
    assert "[[people/anna]]" in read("index.md")


def test_index_is_regenerated_from_the_saved_summary(vault):
    make("people/anna", summary="Old summary.")
    text = read("people/anna.md").replace("Old summary.", "New summary.")
    wiki.save_page("people/anna", text, wiki.etag(read("people/anna.md")))
    assert "New summary." in read("index.md") and "Old summary." not in read("index.md")


def test_saving_unchanged_text_is_a_no_op(vault):
    make("people/anna")
    n = len(commits())
    result, reason = wiki.save_page("people/anna", read("people/anna.md"), wiki.etag(read("people/anna.md")))
    assert reason is None and result == {"etag": wiki.etag(read("people/anna.md"))}
    assert len(commits()) == n


# --------------------------------------------------------------- 3. validation


def test_missing_frontmatter_key_or_oversize_is_rejected(vault):
    make("people/anna")
    before = read("people/anna.md")
    tag = wiki.etag(before)

    result, reason = wiki.save_page("people/anna", before.replace("seen: ", "when: "), tag)
    assert result is None and reason == "frontmatter missing seen"
    result, reason = wiki.save_page("people/anna", with_body("people/anna.md", "x" * 21 * 1024), tag)
    assert result is None and "20 kB" in reason
    result, reason = wiki.save_page("people/anna", "no frontmatter at all", tag)
    assert result is None and reason.startswith("frontmatter missing")
    assert read("people/anna.md") == before
    assert wiki.save_page("index", "---\n---\n", "x") == (None, "index is not an editable page")
    assert wiki.save_page("people/nobody", "", "x") == (None, "no such page people/nobody")


# --------------------------------------------------------------- 4. waits for the lock


def _hold_lock_and(change, seconds=0.4):
    """A fake memory.save: holds `memory._LOCK`, runs `change`, keeps holding for `seconds`."""
    entered = threading.Event()

    def run():
        with memory._LOCK:
            entered.set()
            change()
            time.sleep(seconds)

    thread = threading.Thread(target=run)
    thread.start()
    assert entered.wait(timeout=5)
    return thread


def test_save_waits_for_an_ingest_and_then_sees_its_bytes(vault):
    make("people/anna")
    tag = wiki.etag(read("people/anna.md"))
    text = with_body("people/anna.md", "\n## Notes\n- mine\n")

    def ingest_changes_the_page():
        wiki.apply_page_ops([{"op": "append_note", "path": "people/anna", "text": "theirs"}], "s")

    thread = _hold_lock_and(ingest_changes_the_page)
    started = time.monotonic()
    result, reason = wiki.save_page("people/anna", text, tag)
    thread.join()
    assert time.monotonic() - started >= 0.3          # it waited for the lock
    assert reason is None and result["conflict"] is True
    assert "theirs" in result["text"] and "- mine" not in read("people/anna.md")

    thread = _hold_lock_and(lambda: None)             # an ingest that left the page alone
    result, reason = wiki.save_page("people/anna", with_body("people/anna.md", "- mine\n"), result["etag"])
    thread.join()
    assert reason is None and "etag" in result and "conflict" not in result
    assert "- mine" in read("people/anna.md")


# --------------------------------------------------------------- 5. create from a missing link


def test_create_from_a_missing_link_clears_the_lint_finding(vault):
    make("projects/atlas", page_type="project")
    wiki.save_page("projects/atlas", with_body("projects/atlas.md", "\nOwner: [[people/anna]]\n"),
                   wiki.etag(read("projects/atlas.md")))
    assert wiki.lint() == ["projects/atlas: broken link [[people/anna]]"]
    n = len(commits())

    path, reason = wiki.create_page("people/anna")
    assert (path, reason) == ("people/anna", None)
    fm, body = wiki.parse_page(read("people/anna.md"))
    assert all(key in fm for key in wiki.FIELDS)
    assert fm["type"] == "person" and fm["source"] == "user edit" and body.startswith("# anna")
    assert wiki.lint() == []
    assert len(commits()) == n + 1
    assert wiki.create_page("people/anna") == (None, "people/anna already exists")


def test_user_pages_may_use_any_folder_but_assistant_ops_may_not(vault):
    assert wiki.create_page("notes/foo") == ("notes/foo", None)
    assert wiki.parse_page(read("notes/foo.md"))[0]["type"] == "note"
    assert wiki.create_page("ideas/bar") == ("ideas/bar", None)
    assert wiki.create_page("raw/x") == (None, "bad path 'raw/x'")
    assert wiki.create_page("a/b/c") == (None, "bad path 'a/b/c'")
    assert wiki.create_page("schema") == (None, "schema is not an editable page")

    applied, rejected = wiki.apply_page_ops(
        [{"op": "new_page", "path": "ideas/baz", "type": "note", "name": "Baz"},
         {"op": "append_note", "path": "ideas/bar", "text": "hi"}], "s")
    assert applied == []
    assert [r["reason"] for r in rejected] == ["bad path 'ideas/baz'", "bad path 'ideas/bar'"]
    # notes/ is now a regular folder for the assistant too
    applied, _ = wiki.apply_page_ops([{"op": "append_note", "path": "notes/foo", "text": "hi"}], "s")
    assert len(applied) == 1


def test_wiki_create_action_opens_the_new_page(editor):
    page, listing = editor.send_many({"action": "wiki_create", "path": "meetings/2030-01-02"}, 2)
    assert page["type"] == "wiki_page" and page["path"] == "meetings/2030-01-02"
    assert page["etag"] == wiki.etag(read("meetings/2030-01-02.md")) and page["backlinks"] == []
    assert [p["path"] for p in listing["pages"]] == ["meetings/2030-01-02"]
    assert editor.send({"action": "wiki_create", "path": "raw/x"}) == {"type": "error", "text": "bad path 'raw/x'"}


# --------------------------------------------------------------- 6. backlinks


def test_backlinks_are_the_pages_whose_links_in_contain_the_path(editor):
    make("people/anna")
    make("projects/atlas", page_type="project")
    make("projects/beta", page_type="project")
    make("meetings/2030-01-01", page_type="meeting")
    for rel, line in (("projects/atlas.md", "[[people/anna|Anna K]]"),
                      ("projects/beta.md", "[[people/anna#Summary]] and [[projects/atlas]]"),
                      ("meetings/2030-01-01.md", "![[projects/atlas]]")):
        wiki.save_page(wiki.wikilink(rel), with_body(rel, f"\nsee {line}\n"), wiki.etag(read(rel)))

    expected = sorted(wiki.wikilink(r) for r in wiki.page_paths()
                      if "people/anna" in wiki.links_in(read(r)) and r != "people/anna.md")
    assert sorted(wiki.backlinks("people/anna")) == expected == ["projects/atlas", "projects/beta"]
    assert sorted(wiki.backlinks("projects/atlas")) == ["meetings/2030-01-01", "projects/beta"]
    assert wiki.backlinks("nobody") == [] and wiki.backlinks("../x") == []

    msg = editor.send({"action": "read_wiki_page", "path": "people/anna"})
    assert msg["backlinks"] == ["projects/atlas", "projects/beta"]
    assert msg["etag"] == wiki.etag(read("people/anna.md"))
    assert msg["text"] == read("people/anna.md")


def test_read_wiki_page_sends_the_whole_page_not_the_tool_cap(editor):
    make("people/anna")
    text = with_body("people/anna.md", "\n" + "long line\n" * 800)      # ~7 kB > MAX_READ_BYTES
    wiki.save_page("people/anna", text, wiki.etag(read("people/anna.md")))
    msg = editor.send({"action": "read_wiki_page", "path": "people/anna"})
    assert msg["text"] == read("people/anna.md") and wiki.TRUNCATED not in msg["text"]


# --------------------------------------------------------------- 7. tags


def test_tags_merge_frontmatter_and_body(editor):
    make("people/anna")
    text = read("people/anna.md").replace("tags: []", "tags: [team, ops]")
    text = text.rstrip("\n") + \
        "\n\n#infra and #team again; not a#b, not https://x.io/#frag, not `#5`, not # heading\n"
    wiki.save_page("people/anna", text, wiki.etag(read("people/anna.md")))

    assert wiki.tags(read("people/anna.md")) == ["infra", "ops", "team"]
    assert wiki.tags("---\ntype: x\n---\nno tags") == []
    assert wiki.pages()[0]["tags"] == ["infra", "ops", "team"]
    msg = editor.send({"action": "get_wiki"})
    assert msg["pages"][0]["tags"] == ["infra", "ops", "team"]


def test_wiki_search_action_uses_bm25(editor):
    make("people/anna", summary="Runs the Atlas release.")
    make("projects/beta", page_type="project", summary="Something else.")
    msg = editor.send({"action": "wiki_search", "query": "atlas release"})
    assert msg["type"] == "wiki_search" and msg["query"] == "atlas release"
    assert [h["path"] for h in msg["hits"]] == ["people/anna"]
    assert editor.send({"action": "wiki_search", "query": "  "})["hits"] == []


# --------------------------------------------------------------- 7/8. the client fixture


@pytest.fixture(scope="module")
def fixture_render():
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    out = subprocess.run(["node", "tests/app_fixture.mjs"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_tags_group_filters_the_list(fixture_render):
    r = fixture_render
    assert r["wiki_list"] == ["# people", "Anna", "# projects", "Atlas", "# notes", "todo",
                              "# tags", "#infra", "#team"]
    assert r["wiki_list_infra"] == ["# projects", "Atlas", "# tags", "#infra", "#team"]
    assert r["wiki_list_after"] == r["wiki_list"]


def test_alias_link_renders_the_alias_and_navigates_to_the_page(fixture_render):
    r = fixture_render
    assert r["wiki_links"] == [
        {"text": "Atlas project", "cls": "wiki-link", "target": "projects/atlas"},
        {"text": "projects/atlas#Summary", "cls": "wiki-link", "target": "projects/atlas"},
        {"text": "people/missing", "cls": "wiki-link missing", "target": "people/missing"},
        {"text": "![[notes/todo]]", "cls": "wiki-link embed", "target": "notes/todo"},
    ]
    assert r["wiki_alias_nav"] == ["people/anna", "projects/atlas"]
    assert r["wiki_backlinks"] == ["projects/atlas"]
    assert r["wiki_fm_hidden"] is False


def test_red_link_click_confirms_then_sends_wiki_create(fixture_render):
    r = fixture_render
    assert r["wiki_create_sent"] == [{"action": "wiki_create", "path": "people/missing"}]
    assert r["wiki_create_confirm"] == 1


def test_editor_saves_with_the_etag_and_handles_a_conflict(fixture_render):
    r = fixture_render
    assert r["wiki_save_disabled_clean"] is True
    assert r["wiki_editor_shown"] is True
    assert r["wiki_save_disabled_dirty"] is False
    [save] = r["wiki_save_sent"]
    assert (save["action"], save["path"], save["etag"]) == ("wiki_save", "people/anna", "e1")
    assert save["text"].endswith("![[notes/todo]].\nmore\n")
    assert r["wiki_conflict_shown"] is True
    [again] = r["wiki_overwrite_sent"]
    assert again["etag"] == "e2" and again["text"] == save["text"]
    assert r["wiki_conflict_hidden_after"] is True
    assert r["wiki_mode_after_save"] == "view"
    assert r["wiki_escape"] == {"confirmed": ["confirm", "Discard unsaved changes?"], "editor_hidden": True}


def test_today_new_page_and_search_send_their_actions(fixture_render):
    r = fixture_render
    assert r["wiki_tools_sent"][0]["action"] == "wiki_create"
    assert r["wiki_tools_sent"][0]["path"].startswith("meetings/20")
    assert r["wiki_tools_sent"][1:] == [{"action": "wiki_create", "path": "notes/scratch"},
                                        {"action": "wiki_search", "query": "atlas"}]
    assert r["wiki_search_rows"] == ["# results", "Atlas"]


def test_marked_is_vendored_with_its_licence():
    assert os.path.isfile(os.path.join(ROOT, "static", "vendor", "marked.umd.js"))
    assert "MIT" in open(os.path.join(ROOT, "static", "vendor", "LICENSE-marked.md")).read()
    assert "/static/vendor/marked.umd.js" in open(os.path.join(ROOT, "static", "index.html")).read()


# --------------------------------------------------------------- 9. every wiki write holds the lock


class RecordingLock:
    """An RLock that knows whether it is held, so a patched `_write` can tell."""

    def __init__(self):
        self._lock = threading.RLock()
        self.depth = 0

    def __enter__(self):
        self._lock.acquire()
        self.depth += 1

    def __exit__(self, *exc):
        self.depth -= 1
        self._lock.release()


@pytest.fixture
def held(monkeypatch):
    """Patches vault_lock.LOCK and records, per `wiki._write` call, whether the lock was held."""
    lock, seen = RecordingLock(), []
    real = wiki._write
    monkeypatch.setattr(vault_lock, "LOCK", lock)
    monkeypatch.setattr(wiki, "_write", lambda rel, text: (seen.append(lock.depth > 0), real(rel, text))[1])
    return seen


def test_wiki_note_and_wiki_save_write_under_the_memory_lock(editor, held):
    make("people/anna")
    held.clear()
    page, _ = editor.send_many({"action": "wiki_note", "path": "people/anna", "text": "hello"}, 2)
    assert page["type"] == "wiki_page" and held and all(held)

    held.clear()
    text = with_body("people/anna.md", "- more\n")
    editor.send_many({"action": "wiki_save", "path": "people/anna", "text": text, "etag": page["etag"]}, 2)
    assert held and all(held)

    for call in (lambda: wiki.create_page("notes/x"),
                 lambda: wiki.rename_page("notes/x", "notes/y"),
                 lambda: wiki.delete_page("notes/y")):
        held.clear()
        assert call()[1] is None
        assert held and all(held)


# --------------------------------------------------------------- 10. rename


LINK_FORMS = "[[people/anna]] [[people/anna|Anna K]] [[people/anna#Summary]] ![[people/anna]] [[anna]]"


def test_rename_rewrites_all_link_forms_and_the_unique_short_form(editor):
    make("people/anna")
    make("projects/atlas", page_type="project")
    make("projects/beta", page_type="project")
    wiki.save_page("projects/atlas", with_body("projects/atlas.md", f"\n{LINK_FORMS}\n"),
                   wiki.etag(read("projects/atlas.md")))
    wiki.save_page("projects/beta", with_body("projects/beta.md", "\nno links here\n"),
                   wiki.etag(read("projects/beta.md")))
    beta, n = read("projects/beta.md"), len(commits())

    page, listing = editor.send_many({"action": "wiki_rename", "path": "people/anna",
                                      "new_path": "people/anna-k"}, 2)
    assert page["type"] == "wiki_page" and page["path"] == "people/anna-k"
    assert not os.path.exists(os.path.join(wiki.WIKI_PATH, "people/anna.md"))
    assert "# Anna" in read("people/anna-k.md")
    assert ("[[people/anna-k]] [[people/anna-k|Anna K]] [[people/anna-k#Summary]] ![[people/anna-k]] "
            "[[people/anna-k]]") in read("projects/atlas.md")
    assert read("projects/beta.md") == beta                      # untouched pages stay untouched
    assert len(commits()) == n + 1
    assert commits()[0].endswith("wiki: rename people/anna → people/anna-k (1 links)")
    assert wiki.lint() == []
    assert [p["path"] for p in listing["pages"]] == ["people/anna-k", "projects/atlas", "projects/beta"]
    assert page["backlinks"] == ["projects/atlas"]


def test_rename_leaves_the_short_form_alone_when_the_basename_is_not_unique(vault):
    make("people/anna")
    make("notes/anna", page_type="note")
    make("projects/atlas", page_type="project")
    wiki.save_page("projects/atlas", with_body("projects/atlas.md", f"\n{LINK_FORMS}\n"),
                   wiki.etag(read("projects/atlas.md")))

    result, reason = wiki.rename_page("people/anna", "people/anna-k")
    assert reason is None and result == {"path": "people/anna-k", "links": 1}
    assert ("[[people/anna-k]] [[people/anna-k|Anna K]] [[people/anna-k#Summary]] ![[people/anna-k]] "
            "[[anna]]") in read("projects/atlas.md")


def test_rename_refuses_an_existing_target_and_fixed_pages(editor):
    make("people/anna")
    make("people/bob")
    assert wiki.rename_page("people/anna", "people/bob") == (None, "people/bob already exists")
    assert wiki.rename_page("people/anna", "raw/anna") == (None, "bad path 'raw/anna'")
    assert wiki.rename_page("people/nobody", "people/x") == (None, "no such page people/nobody")
    assert wiki.rename_page("decisions", "notes/d") == (None, "decisions is not an editable page")
    assert editor.send({"action": "wiki_rename", "path": "people/anna", "new_path": "people/bob"}) == \
        {"type": "error", "text": "people/bob already exists"}
    assert os.path.exists(os.path.join(wiki.WIKI_PATH, "people/anna.md"))


# --------------------------------------------------------------- 11. delete


def test_delete_removes_the_page_and_lint_names_the_broken_links(editor):
    make("people/anna")
    make("projects/atlas", page_type="project")
    wiki.save_page("projects/atlas", with_body("projects/atlas.md", "\n[[people/anna|Anna]]\n"),
                   wiki.etag(read("projects/atlas.md")))
    n = len(commits())

    listing = editor.send({"action": "wiki_delete", "path": "people/anna"})
    assert listing["type"] == "wiki"
    assert not os.path.exists(os.path.join(wiki.WIKI_PATH, "people/anna.md"))
    assert [p["path"] for p in listing["pages"]] == ["projects/atlas"]
    assert "[[people/anna]]" not in read("index.md")
    assert listing["lint"] == ["projects/atlas: broken link [[people/anna]]"]
    assert len(commits()) == n + 1 and commits()[0].endswith("wiki: delete people/anna")
    assert len(commits("people/anna.md")) >= 2          # git log still shows the file's history
    assert f"- {TODAY} deleted [[people/anna]] (source: user edit)" in read("log.md")
    assert editor.send({"action": "wiki_delete", "path": "people/anna"}) == \
        {"type": "error", "text": "no such page people/anna"}
    assert wiki.delete_page("index") == (None, "index is not an editable page")
