"""Unit tests for the wiki layer — tmp vault, mocked LLM. Mirrors docs/19-wiki-layer.md.

Scenario numbers below are the PRD's test scenarios (1-9); scenario 10 is the live harness
script eval/scripts/wiki.yaml and 11 is manual. The briefing digest + card are docs/46-briefing.md scenarios 1-2.
"""
import os

import json

import pytest

from agent import fold
from core import util
from memory import identity
from memory import ops as m_ops
from memory import render
from memory import store as m_store
from agent import digest as a_digest
from memory import bridge
from memory import wiki_ingest as m_ingest
from vault import chunks
from vault import focus
from vault import index as v_index
from vault import lint
from vault import pages as v_pages
from vault import search
from vault import store
from core import llm as core_llm
from tests.conftest import FakeClient, commits, log_ops, read, seed, write_page
from core import toolset


TURNS = [{"role": "user", "content": "we agreed to deploy on Fridays; Anna runs the release"}]


def a_page(path="people/anna", name="Anna", summary="Colleague on the platform team."):
    return {"op": "new_page", "path": path, "type": "person", "name": name, "summary": summary}


# --------------------------------------------------------------- 1. first run


def test_first_run_creates_the_vault(wiki_sandbox):
    assert not store.exists()
    store.ensure_vault()

    for rel in ("schema.md", "index.md", "log.md", "decisions.md", ".gitignore"):
        assert os.path.isfile(os.path.join(store.WIKI_PATH, rel)), rel
    for folder in store.FOLDERS:
        assert os.path.isdir(os.path.join(store.WIKI_PATH, folder))
    assert ".obsidian/" in read(".gitignore")
    assert "## Rules" in read("schema.md")
    assert os.path.isdir(os.path.join(store.WIKI_PATH, ".git"))
    assert len(commits()) == 1


def test_ensure_vault_is_idempotent(vault):
    store._write("people/anna.md", "keep me")
    store.ensure_vault()
    assert read("people/anna.md") == "keep me"
    assert len(commits()) == 1


def test_nothing_is_created_at_import(wiki_sandbox):
    """Reading the index of a vault that isn't there must not bring one into existence."""
    assert v_index.index_text() == ""
    assert lint.pages() == []
    assert lint.lint() == []
    assert not os.path.exists(store.WIKI_PATH)


# --------------------------------------------------------------- 2. page ops


def test_new_page_writes_frontmatter_and_sections(vault):
    applied, rejected = v_pages.apply_page_ops([a_page()], "session 2026-08-29")
    assert (len(applied), rejected) == (1, [])

    fm, body = store.parse_page(read("people/anna.md"))
    assert fm["type"] == "person" and fm["name"] == "Anna" and fm["tags"] == []
    assert fm["source"] == "session 2026-08-29"
    assert fm["updated"] == util.today_iso() and fm["seen"] == util.today_iso()
    assert "# Anna" in body
    assert "Colleague on the platform team." in body
    for section in ("## Summary", "## Notes", "## Related"):
        assert section in body


def test_append_note_adds_a_dated_bullet_under_notes(vault):
    v_pages.apply_page_ops([a_page()], "s")
    v_pages.apply_page_ops([{"op": "append_note", "path": "people/anna",
                          "text": "moved to the release team"}], "session x")
    _, body = store.parse_page(read("people/anna.md"))
    notes = body.split("## Notes")[1].split("## Related")[0]
    assert f"- {util.today_iso()} moved to the release team (source: session x)" in notes


def test_set_field_touches_only_frontmatter(vault):
    v_pages.apply_page_ops([a_page()], "s")
    before_body = store.parse_page(read("people/anna.md"))[1]
    applied, _ = v_pages.apply_page_ops([{"op": "set_field", "path": "people/anna",
                                       "key": "tags", "value": ["team", "release"]}], "s")
    fm, body = store.parse_page(read("people/anna.md"))
    assert applied and fm["tags"] == ["team", "release"] and body == before_body


def test_set_field_can_set_status_on_a_project_page(vault, memory_sandbox):
    """PRD 19 gives project pages a `status`; it must be settable and must survive a pull."""
    v_pages.apply_page_ops([{"op": "new_page", "path": "projects/falcon", "type": "project",
                          "name": "Falcon", "summary": "The platform rewrite."}], "s")

    applied, rejected = v_pages.apply_page_ops([{"op": "set_field", "path": "projects/falcon",
                                              "key": "status", "value": "amber"}], "s")

    assert applied and rejected == []
    assert store.parse_page(read("projects/falcon.md"))[0]["status"] == "amber"
    assert "Falcon, project, amber" in read("index.md")
    entry, reason = bridge.entry_from_page("projects/falcon", "projects")
    assert reason is None and entry["status"] == "amber"


def test_a_page_without_a_status_does_not_grow_one(vault):
    """`status` is optional: only project pages that were given one carry the key."""
    v_pages.apply_page_ops([a_page()], "s")
    v_pages.apply_page_ops([{"op": "append_note", "path": "people/anna", "text": "hi"}], "s")

    assert "status:" not in read("people/anna.md")


def test_ensure_vault_never_rewrites_an_edited_schema(vault):
    store._write(store.SCHEMA_FILE, "---\ntype: schema\n---\n\n# My own schema\n")

    store.ensure_vault()

    assert read(store.SCHEMA_FILE) == "---\ntype: schema\n---\n\n# My own schema\n"


def test_add_decision_appends_a_dated_section(vault):
    v_pages.apply_page_ops([{"op": "add_decision", "title": "deploy on Fridays",
                          "text": "Ship at 10:00, Anna runs it."}], "session 2026-08-29")
    text = read("decisions.md")
    assert f"## {util.today_iso()} deploy on Fridays" in text
    assert "Ship at 10:00, Anna runs it." in text
    assert "source: session 2026-08-29" in text


def test_link_adds_a_wikilink_once(vault):
    v_pages.apply_page_ops([a_page(), a_page("projects/atlas", "Atlas", "The platform rewrite.")], "s")
    op = {"op": "link", "path": "people/anna", "target": "projects/atlas"}
    assert v_pages.apply_page_ops([op], "s")[0]
    assert "[[projects/atlas]]" in read("people/anna.md")
    applied, rejected = v_pages.apply_page_ops([op], "s")     # second time is a no-op
    assert applied == [] and "already links" in rejected[0]["reason"]


@pytest.mark.parametrize("op, expect", [
    ({"op": "new_page", "path": "../escape", "name": "x"}, "bad path"),
    ({"op": "new_page", "path": "/etc/passwd", "name": "x"}, "bad path"),
    ({"op": "new_page", "path": "a/b/c", "name": "x"}, "bad path"),
    ({"op": "new_page", "path": "nowhere/thing", "name": "x"}, "bad path"),
    ({"op": "new_page", "path": "index", "name": "x"}, "not a page"),
    ({"op": "append_note", "path": "people/ghost", "text": "hi"}, "no such page"),
    ({"op": "set_field", "path": "people/anna", "key": "colour", "value": "red"}, "unknown frontmatter key"),
    ({"op": "add_decision", "text": "no title"}, "needs a title"),
    ({"op": "burn_it_down", "path": "people/anna"}, "unknown op"),
])
def test_invalid_ops_are_rejected_and_change_nothing(vault, op, expect):
    v_pages.apply_page_ops([a_page()], "s")
    before = {rel: read(rel) for rel in ("index.md", "log.md", "people/anna.md", "decisions.md")}
    commits_before = len(commits())

    applied, rejected = v_pages.apply_page_ops([op], "s")

    assert applied == []
    assert expect in rejected[0]["reason"]
    assert {rel: read(rel) for rel in before} == before
    assert len(commits()) == commits_before


def test_new_page_on_an_existing_page_is_rejected(vault):
    v_pages.apply_page_ops([a_page()], "s")
    applied, rejected = v_pages.apply_page_ops([a_page(summary="different")], "s")
    assert applied == [] and "already exists" in rejected[0]["reason"]
    assert "different" not in read("people/anna.md")


def test_oversized_note_is_rejected(vault):
    v_pages.apply_page_ops([a_page()], "s")
    applied, rejected = v_pages.apply_page_ops(
        [{"op": "append_note", "path": "people/anna", "text": "x" * 601}], "s")
    assert applied == [] and "600 characters" in rejected[0]["reason"]


def test_append_to_an_oversized_page_is_rejected(vault):
    v_pages.apply_page_ops([a_page()], "s")
    store._write("people/anna.md", read("people/anna.md") + "\nfiller " * 4000)
    applied, rejected = v_pages.apply_page_ops(
        [{"op": "append_note", "path": "people/anna", "text": "one more"}], "s")
    assert applied == [] and "20 kB" in rejected[0]["reason"]


def test_a_missing_source_falls_back_to_the_batch_default(vault):
    v_pages.apply_page_ops([a_page()], "session 2026-08-29")
    assert "source: session 2026-08-29" in read("people/anna.md")


# --------------------------------------------------------------- 3. batches


def test_a_batch_makes_one_commit_and_one_log_line_per_op(vault):
    commits_before = len(commits())
    ops = [
        a_page(),
        a_page("projects/atlas", "Atlas", "The platform rewrite."),
        {"op": "append_note", "path": "people/anna", "text": "runs the release"},
        {"op": "add_decision", "title": "deploy on Fridays", "text": "Ship at 10:00."},
    ]
    applied, rejected = v_pages.apply_page_ops(ops, "session 2026-08-29")

    assert len(applied) == 4 and rejected == []
    assert len(commits()) == commits_before + 1
    log_lines = [line for line in read("log.md").splitlines() if line.startswith("- ")]
    assert len(log_lines) == 4
    assert all(util.today_iso() in line for line in log_lines)
    assert any("filed [[people/anna]]" in line for line in log_lines)

    index = read("index.md")
    assert "- [[people/anna]]" in index and "- [[projects/atlas]]" in index
    assert "- [[decisions]]" in index
    assert "[[index]]" not in index and "[[log]]" not in index and "[[schema]]" not in index


def test_index_is_a_pure_function_of_the_pages(vault):
    v_pages.apply_page_ops([a_page(), a_page("projects/atlas", "Atlas", "The rewrite.")], "s")
    first = read("index.md")
    store._write("index.md", "# Index\n\ngarbage\n")
    assert v_index.rebuild_index() == first


def test_a_partly_bad_batch_applies_the_good_ops(vault):
    applied, rejected = v_pages.apply_page_ops([a_page(), {"op": "append_note", "path": "nope/x", "text": "y"}], "s")
    assert len(applied) == 1 and len(rejected) == 1
    assert store.page_exists("people/anna.md")


# --------------------------------------------------------------- 4. search / read


def test_search_ranks_the_decisions_page_first(vault):
    v_pages.apply_page_ops([
        a_page(),
        a_page("projects/atlas", "Atlas", "The platform rewrite, unrelated to shipping."),
        {"op": "add_decision", "title": "deploy on Fridays",
         "text": "We decided the deploy happens on Friday mornings."},
    ], "s")
    hits = search.search("deploy decision")
    assert hits and hits[0]["path"] == "decisions"
    assert "deploy" in hits[0]["snippet"].lower()


def test_search_returns_at_most_k_hits_and_nothing_for_a_miss(vault):
    v_pages.apply_page_ops([a_page(f"people/p{i}", f"P{i}", "colleague") for i in range(7)], "s")
    assert len(search.search("colleague")) == 5
    assert search.search("submarine") == []


def test_search_sees_a_page_edited_on_disk(vault):
    v_pages.apply_page_ops([a_page()], "s")
    assert search.search("kayaking") == []
    store._write("people/anna.md", read("people/anna.md") + "\n- she is into kayaking\n")
    assert [h["path"] for h in search.search("kayaking")] == ["people/anna"]


def test_read_page_truncates_at_four_kb(vault):
    v_pages.apply_page_ops([a_page()], "s")
    store._write("people/anna.md", read("people/anna.md") + "\n" + "long " * 2000)
    text = chunks.read_page("people/anna")
    assert text.endswith(store.TRUNCATED)
    assert len(text.encode()) <= store.MAX_READ_BYTES + len(store.TRUNCATED)


def test_read_page_reports_an_unknown_path(vault):
    assert chunks.read_page("people/nobody").startswith("no such page")
    assert chunks.read_page("../../etc/passwd").startswith("no such page")


def test_the_tools_wrap_search_and_read(vault):
    from tools import run_tool

    v_pages.apply_page_ops([a_page()], "s")
    assert "people/anna:" in run_tool("search_wiki", {"query": "colleague platform"})
    assert "# Anna" in run_tool("read_page", {"path": "people/anna"})
    assert run_tool("search_wiki", {"query": "submarine"}) == "no matching wiki pages"


# --------------------------------------------------------------- 5. prompt block


def test_load_injects_the_wiki_index(vault, memory_sandbox):
    v_pages.apply_page_ops([a_page()], "s")
    prompt = identity.load("BASE")
    assert "<wiki_index>" in prompt and "[[people/anna]]" in prompt
    assert "read_page" in prompt and "search_wiki" in prompt
    assert "<wiki_index>\n# Vault map\n## people (1)" in prompt                # the map, not index.md (PRD 54)
    block = prompt.split("<wiki_index>")[1].split("</wiki_index>")[0]
    assert util.tokens_est(block) <= store.WIKI_INDEX_TOKENS


def test_the_tool_hint_is_dropped_when_the_tools_are_not_registered(vault, memory_sandbox,
                                                                    monkeypatch):
    """PRD 19 Results: tool-free runs (the harness sets `toolset.TOOLS = []`) get the index only.

    Advertising search_wiki to a model that has not been offered it only teaches it to type a
    tool call at the user.
    """

    v_pages.apply_page_ops([a_page()], "s")
    monkeypatch.setattr(toolset, "TOOLS", [])

    prompt = identity.load("BASE")

    assert not focus.tools_registered()
    assert "<wiki_index>" in prompt and "[[people/anna]]" in prompt
    assert "search_wiki" not in prompt and "read_page" not in prompt
    assert prompt.rstrip().endswith("</wiki_index>")


def test_the_hint_names_the_tools_that_are_actually_registered():

    assert focus.tools_registered()
    assert set(focus.RECALL_TOOLS) <= {t["function"]["name"] for t in toolset.TOOLS}
    assert all(name in identity.WIKI_NOTE for name in focus.RECALL_TOOLS)


def test_an_empty_vault_adds_no_block(vault, memory_sandbox):
    assert identity.load("BASE") == "BASE"
    assert "<wiki_index>" not in identity.load("BASE")


def test_the_index_block_is_trimmed_to_its_budget(vault, memory_sandbox, monkeypatch):
    v_pages.apply_page_ops([a_page(f"people/p{i}", f"Person number {i}", "a colleague of the user")
                         for i in range(40)], "s")
    monkeypatch.setattr(store, "WIKI_INDEX_TOKENS", 50)
    block = identity.load("BASE").split("<wiki_index>")[1].split("</wiki_index>")[0]
    assert util.tokens_est(block) <= 50


# --------------------------------------------------------------- 6. lint


def _git_fails_on(monkeypatch, verb, code=128, stdout="", stderr="fatal: boom\nhint"):
    """`git <verb>` dies; every other git call succeeds (architecture review #12)."""
    import subprocess
    real = subprocess.run

    def fake(cmd, *a, **k):
        if verb in cmd:
            return subprocess.CompletedProcess(cmd, code, stdout, stderr)
        return real(cmd, *a, **k)
    monkeypatch.setattr(subprocess, "run", fake)


def test_lint_surfaces_vault_git_failures_first(vault, monkeypatch):
    v_pages.apply_page_ops([a_page()], "s")
    _git_fails_on(monkeypatch, "commit", stderr="fatal: unable to auto-detect\nhint")
    store._commit("one")
    store._commit("two")

    findings = lint.lint()
    assert findings[0] == "vault git: 2 failed commands this run (last: fatal: unable to auto-detect)"


def test_nothing_to_commit_is_not_a_failure(vault, monkeypatch):
    _git_fails_on(monkeypatch, "commit", code=1, stdout="nothing to commit, working tree clean", stderr="")
    store._commit("x")
    assert store.GIT_FAILURES["count"] == 0
    assert not [f for f in lint.lint() if f.startswith("vault git")]


def test_a_git_timeout_is_counted_with_a_short_message(vault, monkeypatch):
    import subprocess

    def boom(cmd, *a, **k):
        raise subprocess.TimeoutExpired(cmd, 20)
    monkeypatch.setattr(subprocess, "run", boom)
    store._git("commit", "-m", "x", quiet=True)
    assert store.GIT_FAILURES == {"count": 1, "last": "git commit timed out after 20s"}


def test_lint_flags_a_broken_link_and_a_page_missing_from_the_index(vault):
    v_pages.apply_page_ops([a_page()], "s")
    store._write("people/anna.md", read("people/anna.md") + "\n- see [[projects/ghost]]\n")
    search._db().drop_page("people/anna.md")           # a write that missed its hook (PRD 53)

    findings = lint.lint()
    assert any("broken link [[projects/ghost]]" in f for f in findings)
    assert any("missing from the search index" in f for f in findings)
    assert all(f.startswith("people/anna:") for f in findings)   # suffix-less, like wikilinks

    search.reindex()
    findings = lint.lint()
    assert not any("search index" in f for f in findings)
    assert any("broken link" in f for f in findings)


def test_lint_checks_the_search_index_not_the_capped_index_md(vault):
    """PRD 53 scenario 5: 250 pages are no lint finding; index.md stays at 200 lines; a page on
    disk the index lacks, and an index row whose page is gone, are reported."""
    v_pages.apply_page_ops([a_page(f"notes/n{i:03d}", f"Note {i:03d}", f"Note number {i}.")
                         for i in range(250)], "s")
    assert len(store.page_paths()) == 250
    assert not [f for f in lint.lint() if "index" in f]
    assert len([line for line in read("index.md").splitlines() if line.startswith("- [[")]) == store.INDEX_MAX_LINES

    write_page("notes/stray", "Stray", "## Summary\nWritten past the hooks.\n")
    search._db().drop_page("notes/stray.md")
    os.remove(os.path.join(store.WIKI_PATH, "notes/n000.md"))
    findings = [f for f in lint.lint() if "index" in f]
    assert findings == ["notes/stray: missing from the search index — run `python -m vault reindex`",
                        "notes/n000: in the search index but not on disk — run `python -m vault reindex`"]


def test_lint_flags_missing_frontmatter_keys(vault):
    store._write("people/bare.md", "---\ntype: person\n---\n\n# Bare\n")
    findings = lint.lint()
    assert any("frontmatter missing" in f and "name" in f for f in findings)


def test_lint_and_log_only_writes_when_there_are_findings(vault):
    v_pages.apply_page_ops([a_page()], "s")
    before = read("log.md")
    assert lint.lint_and_log() == []
    assert read("log.md") == before

    store._write("people/anna.md", read("people/anna.md") + "\n- see [[projects/ghost]]\n")
    assert lint.lint_and_log()
    assert "lint:" in read("log.md")


# PRD 44: dead source refs, uncited raw files, duplicate names


def stamped(rel, source, name="Anna", extra=""):
    write_page(rel, name, "\n## Summary\nx\n", source, extra)
    v_index.rebuild_index()


def test_lint_ignores_sources_that_are_not_raw_paths(vault):
    stamped("people/anna", "session 2026-08-29")
    stamped("projects/answer", "written answer", "Answer",
            "sources:\n  - Falcon status · part 1/2\n")
    assert lint.lint() == []


def test_lint_flags_an_uncited_raw_file_only(vault):
    stray = "raw/inbox/stray.pdf"
    store._write(stray, "x")
    store._write("raw/manifest.jsonl", "")
    store._write("raw/.index/abc.json", "{}")
    finding = f"{stray}: not in the manifest and cited by no page"
    assert lint.lint() == [finding]

    store._write("raw/manifest.jsonl", json.dumps({"path": stray, "sha256": "a"}) + "\n")
    assert lint.lint() == []
    store._write("raw/manifest.jsonl", "")
    stamped("projects/falcon", stray, "Falcon")
    assert lint.lint() == []


def test_lint_counts_a_source_page_raw_key_as_a_citation(vault):
    stray = "raw/inbox/stray.pdf"
    store._write(stray, "x")
    store._write("raw/manifest.jsonl", "")
    assert lint.lint() == [f"{stray}: not in the manifest and cited by no page"]
    stamped("sources/stray", "upload", "Stray", f"raw: {stray}\n")
    assert lint.lint() == []


def test_lint_exempts_meetings_from_duplicate_names(vault):
    stamped("meetings/2026-10-16-falcon-weekly", "s", "Falcon weekly")
    stamped("meetings/2026-10-23-falcon-weekly", "s", "Falcon weekly")
    assert lint.lint() == []


def test_lint_flags_a_duplicate_name_once_and_groups_sources_apart(vault):
    stamped("people/anna", "s", " anna ")
    stamped("people/anna-2", "s", "Anna")
    stamped("sources/falcon", "s", "Falcon")
    stamped("projects/falcon", "s", "Falcon")
    assert lint.lint() == ['people/anna-2: duplicate name "Anna" (also people/anna)']


def test_lint_and_log_commits_a_duplicate_name_and_the_wiki_message_lists_it(wiki_panel):
    stamped("people/anna", "s")
    stamped("people/anna-2", "s")
    before, log = len(commits()), read("log.md")

    findings = lint.lint_and_log()

    assert findings == ['people/anna-2: duplicate name "Anna" (also people/anna)']
    assert read("log.md").count("lint:") == log.count("lint:") + 1
    assert len(commits()) == before + 1 and "wiki: lint" in commits()[0]
    assert wiki_panel.send({"action": "get_wiki"})["lint"] == findings


# PRD 44: related pages


def test_related_context_lists_non_source_hits_and_is_empty_without_any(vault):
    assert v_index.related_context("Falcon launch") == ""
    write_page("projects/falcon", "Falcon", "\n## Summary\nThe Falcon launch project.\n")
    write_page("sources/falcon-plan", "Falcon plan", "\n## Summary\nFalcon launch plan.\n")
    block = v_index.related_context("Falcon launch", k=5)
    assert block.startswith("Existing related pages (update these rather than duplicating):")
    assert "- projects/falcon: Falcon — The Falcon launch project." in block
    assert "sources/" not in block


def test_related_context_stays_within_its_token_cap(vault):
    for i in range(50):
        write_page(f"projects/falcon-{i}", f"Falcon launch {i}",
                   "\n## Summary\n" + "Falcon launch details. " * 20 + "\n")
    block = v_index.related_context("Falcon launch", k=50)
    assert block and util.tokens_est(block) <= store.RELATED_TOKENS


def test_session_ingest_names_a_page_the_index_cut(vault, monkeypatch):
    for i in range(10):
        write_page(f"people/p{i}", f"Person number {i}", "\n## Summary\na colleague of the user\n")
    write_page("projects/zephyr", "Zephyr", "\n## Summary\nThe Zephyr rollout.\n")
    v_index.rebuild_index()
    monkeypatch.setattr(store, "WIKI_INDEX_TOKENS", 40)
    client = FakeClient("[]")

    monkeypatch.setattr(core_llm, "_client", client)
    m_ingest.ingest([{"role": "user", "content": "the Zephyr rollout slipped a week"}])

    user = client.calls[0]["messages"][1]["content"]
    index_part, block = user.split("Index:\n")[1].split("Existing related pages", 1)
    assert "projects/zephyr" not in index_part
    assert "(update these rather than duplicating):" in block and "projects/zephyr" in block


def test_session_ingest_over_320_pages_updates_zephyr_from_the_map(memory_sandbox, monkeypatch):
    """PRD 54 scenario 3: the map names `projects/zephyr` (z sorts last) next to `related_context`; the
    proposed op appends to it and no duplicate page appears."""
    from tests.perf.make_vault import make_vault

    make_vault(store.WIKI_PATH, pages=320, sources_n=0, seed=3)
    write_page("projects/zephyr", "Zephyr", "\n## Summary\nThe Zephyr budget rollout.\n", extra="")
    for i in range(5):
        rel = f"notes/p{4 * i + 3:05d}.md"
        store._write(rel, read(rel) + "\n- see [[projects/zephyr]]\n")
    pages_before = len(store.page_paths())
    client = FakeClient(json.dumps([{"op": "append_note", "path": "projects/zephyr",
                                     "text": "The Zephyr budget slipped a week."}]))
    monkeypatch.setattr(core_llm, "_client", client)

    applied = m_ingest.ingest([{"role": "user", "content": "the Zephyr budget rollout slipped a week"}])

    user = client.calls[0]["messages"][1]["content"]
    index_part, block = user.split("Index:\n")[1].split("Existing related pages", 1)
    assert "[[projects/zephyr]]" in index_part and index_part.startswith("# Vault map")
    assert "projects/zephyr" in block
    assert [op["path"] for op in applied] == ["projects/zephyr"]
    assert "The Zephyr budget slipped a week." in read("projects/zephyr.md")
    assert len(store.page_paths()) == pages_before


# --------------------------------------------------------------- 7. ingest


def test_ingest_files_the_ops_the_model_proposes(vault, monkeypatch):
    client = FakeClient(json.dumps([
        {"op": "new_page", "path": "people/anna", "type": "person", "name": "Anna",
         "summary": "Runs the release."},
        {"op": "add_decision", "title": "deploy on Fridays", "text": "Ship at 10:00."},
    ]))
    commits_before = len(commits())

    monkeypatch.setattr(core_llm, "_client", client)
    applied = m_ingest.ingest(TURNS)

    assert len(applied) == 2
    assert store.page_exists("people/anna.md")
    assert "deploy on Fridays" in read("decisions.md")
    assert len(commits()) == commits_before + 1
    system = client.calls[0]["messages"][0]["content"]
    user = client.calls[0]["messages"][1]["content"]
    assert "page operation" in system
    assert "## Rules" in user and "Transcript:" in user


@pytest.mark.parametrize("content", ["NOTHING AT ALL", "", None, "{}", "[]"])
def test_ingest_of_garbage_changes_nothing(vault, content, monkeypatch):
    before = {rel: read(rel) for rel in ("index.md", "log.md", "decisions.md")}
    commits_before = len(commits())

    monkeypatch.setattr(core_llm, "_client", FakeClient(content))
    assert m_ingest.ingest(TURNS) == []

    assert {rel: read(rel) for rel in before} == before
    assert len(commits()) == commits_before
    assert store.page_paths() == []


def test_ingest_survives_a_broken_client(vault, capsys, monkeypatch):
    monkeypatch.setattr(core_llm, "_client", FakeClient(RuntimeError("connection refused")))
    assert m_ingest.ingest(TURNS) == []
    assert "ingest failed" in capsys.readouterr().out


def test_ingest_skips_the_call_for_an_empty_transcript(vault, monkeypatch):
    client = FakeClient("[]")
    monkeypatch.setattr(core_llm, "_client", client)
    assert m_ingest.ingest([]) == []
    assert client.calls == []


def test_save_ingests_after_the_profile_ops(memory_sandbox, wiki_sandbox, monkeypatch):
    profile_ops = json.dumps([{"op": "ADD", "path": "identity.name", "value": "Mihaly"}])
    page_ops = json.dumps([{"op": "new_page", "path": "people/anna", "type": "person",
                            "name": "Anna", "summary": "Runs the release."}])
    client = FakeClient(profile_ops, page_ops)

    monkeypatch.setattr(core_llm, "_client", client)
    applied = fold.save(TURNS)

    assert [op["path"] for op in applied] == ["identity.name"]
    assert store.page_exists("people/anna.md")


def test_a_wiki_failure_never_loses_the_profile_save(memory_sandbox, wiki_sandbox, monkeypatch):
    monkeypatch.setattr(m_ingest, "ingest", lambda *a: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    ops = json.dumps([{"op": "ADD", "path": "identity.name", "value": "Mihaly"}])

    monkeypatch.setattr(core_llm, "_client", FakeClient(ops))
    assert fold.save(TURNS)
    assert m_store.load_profile()["identity"]["name"] == "Mihaly"


# --------------------------------------------------------------- 8. archive / pull


def test_archive_files_a_person_and_marks_the_entry_stale(vault, memory_sandbox):

    seed(people=[{"name": "Anna", "rel": "colleague", "note": "platform team",
                  m_store.SEEN_FIELD: "2026-01-05"}])

    profile, error = m_ops.archive("people", 0)

    assert error is None
    _, body = store.parse_page(read("people/anna.md"))
    history = body.split("## History")[1]
    assert "Anna; colleague; platform team" in history
    assert "last seen 2026-01-05" in history
    assert profile["people"][0][m_store.STALE_FIELD] is True
    assert m_store.load_profile()["people"][0][m_store.STALE_FIELD] is True


def test_archive_appends_to_an_existing_page(vault, memory_sandbox):

    v_pages.apply_page_ops([a_page()], "s")
    seed(people=[{"name": "Anna", "rel": "colleague", m_store.SEEN_FIELD: "2026-01-05"}])

    _, error = m_ops.archive("people", 0)

    assert error is None
    assert read("people/anna.md").count("# Anna") == 1
    assert "## History" in read("people/anna.md")


def test_archive_of_a_recurring_entry_lands_on_one_shared_page(vault, memory_sandbox):

    seed(recurring=[{"what": "gym", "when": "Tue/Thu", m_store.SEEN_FIELD: "2026-02-02"}])
    _, error = m_ops.archive("recurring", 0)
    assert error is None
    assert "gym; Tue/Thu" in read(store.RECURRING_FILE)


def test_archive_refuses_a_bad_target(vault, memory_sandbox):
    _, error = m_ops.archive("people", 3)
    assert "does not exist" in error["reason"]
    _, error = m_ops.archive("episodic", 0)
    assert "cannot archive" in error["reason"]


def test_pull_recreates_a_live_entry_from_a_page(vault, memory_sandbox):

    seed(people=[{"name": "Anna", "rel": "colleague", m_store.SEEN_FIELD: "2026-01-05"}])
    m_ops.archive("people", 0)

    profile, error = m_ops.pull("people/anna", "people")

    assert error is None
    fresh = [e for e in profile["people"] if not render.is_stale(e)]
    assert len(fresh) == 1
    assert fresh[0]["name"] == "Anna"
    assert fresh[0][m_store.SEEN_FIELD] == util.today_iso()


def test_pull_reports_an_unknown_page(vault, memory_sandbox):
    _, error = m_ops.pull("people/nobody", "people")
    assert "no such page" in error["reason"]


# --------------------------------------------------------------- 9. user edits


def test_a_user_edit_reaches_the_next_ingest_and_the_index(vault, memory_sandbox, monkeypatch):
    monkeypatch.setattr(core_llm, "_client", FakeClient(json.dumps([a_page()])))
    m_ingest.ingest(TURNS)
    assert "Anna" in read("index.md")

    # the user opens the page in Obsidian and corrects it
    fm, body = store.parse_page(read("people/anna.md"))
    fm["name"] = "Anna Kovacs"
    store._write("people/anna.md", store._page_text(fm, body + "\n- she prefers Anna K\n"))

    client = FakeClient("[]")
    monkeypatch.setattr(core_llm, "_client", client)
    m_ingest.ingest(TURNS)
    context = client.calls[0]["messages"][1]["content"]
    assert "Anna Kovacs" in context

    v_index.rebuild_index()
    assert "Anna Kovacs" in read("index.md")


# --------------------------------------------------------------- actions (PRD scenario 11)


@pytest.fixture
def wiki_panel(vault, panel):
    """The shared `panel` websocket client over a created vault (`vault` sandboxes the profile too)."""
    return panel


def test_get_wiki_lists_pages_with_lint_and_vault(wiki_panel):
    v_pages.apply_page_ops([a_page()], "s")
    msg = wiki_panel.send({"action": "get_wiki"})
    assert msg["type"] == "wiki"
    assert msg["vault"] == store.WIKI_PATH
    assert msg["lint"] == []
    assert [p["path"] for p in msg["pages"]] == ["people/anna"]
    assert msg["pages"][0] == {"path": "people/anna", "title": "Anna", "type": "person",
                               "updated": util.today_iso(), "tags": []}


def test_read_wiki_page_returns_text_and_links(wiki_panel):
    v_pages.apply_page_ops([a_page(), a_page("projects/atlas", "Atlas", "The rewrite."),
                         {"op": "link", "path": "people/anna", "target": "projects/atlas"}], "s")
    msg = wiki_panel.send({"action": "read_wiki_page", "path": "people/anna"})
    assert msg["type"] == "wiki_page" and msg["path"] == "people/anna"
    assert "# Anna" in msg["text"]
    assert msg["links"] == ["projects/atlas"]


def test_wiki_note_appends_and_refreshes_both_panes(wiki_panel):
    v_pages.apply_page_ops([a_page()], "s")
    page, listing = wiki_panel.send_many(
        {"action": "wiki_note", "path": "people/anna", "text": "she prefers async standups"}, 2)
    assert page["type"] == "wiki_page" and listing["type"] == "wiki"
    assert f"user note {util.today_iso()}" in page["text"]
    assert "she prefers async standups" in page["text"]


def test_wiki_note_on_a_missing_page_errors(wiki_panel):
    msg = wiki_panel.send({"action": "wiki_note", "path": "people/ghost", "text": "hi"})
    assert msg["type"] == "error" and "no such page" in msg["text"]


def test_archive_memory_action_replies_with_memory_and_wiki(wiki_panel):

    seed(people=[{"name": "Anna", "rel": "colleague", m_store.SEEN_FIELD: "2026-01-05"}])
    mem, listing = wiki_panel.send_many(
        {"action": "archive_memory", "section": "people", "index": 0}, 2)
    assert mem["type"] == "memory" and mem["profile"]["people"][0]["stale"] is True
    assert [p["path"] for p in listing["pages"]] == ["people/anna"]


def test_pull_wiki_action_replies_with_memory_and_wiki(wiki_panel):
    v_pages.apply_page_ops([a_page()], "s")
    mem, listing = wiki_panel.send_many(
        {"action": "pull_wiki", "path": "people/anna", "section": "people"}, 2)
    assert mem["type"] == "memory"
    assert mem["profile"]["people"][0]["name"] == "Anna"
    assert listing["type"] == "wiki"


def test_a_bad_archive_comes_back_as_memory_error(wiki_panel):
    msg = wiki_panel.send({"action": "archive_memory", "section": "people", "index": 9})
    assert msg["type"] == "memory_error" and "does not exist" in msg["reason"]


# PRD 26: the session ingest knows who the user is and never files them as a person.
def test_ingest_never_opens_a_people_page_about_the_user(vault, capsys, monkeypatch):
    m_store._write_profile(m_store._normalise(
        {"identity": {"name": "Mihaly Fodor", "nickname": "Michael", "aliases": ["Misi"]}}))
    ops = json.dumps([
        {"op": "new_page", "path": "people/michael", "type": "person", "name": "Michael",
         "summary": "The user."},
        {"op": "new_page", "path": "people/anna", "type": "person", "name": "Anna",
         "summary": "Runs the release."},
    ])
    client = FakeClient(ops)

    monkeypatch.setattr(core_llm, "_client", client)
    applied = m_ingest.ingest(TURNS)

    assert [op["path"] for op in applied] == ["people/anna"]
    assert not store.page_exists("people/michael.md")
    assert "is the user, not another person" in capsys.readouterr().out
    system = client.calls[0]["messages"][0]["content"]
    assert "The user (profile owner) is: Mihaly Fodor, Michael, Misi" in system


def test_the_ingest_prompt_says_so_even_with_an_empty_profile(vault, monkeypatch):
    client = FakeClient("[]")
    monkeypatch.setattr(core_llm, "_client", client)
    m_ingest.ingest(TURNS)
    assert identity.NO_USER_NOTE in client.calls[0]["messages"][0]["content"]


def test_a_page_filed_mid_session_reaches_the_next_turn(vault, memory_sandbox):
    """The system prompt is built at session start; `llm.refresh_context` re-reads the index so a
    source dropped a minute ago is not invisible until the next restart."""
    from agent import conversation

    conversation.reset()
    conversation._conversation.append({"role": "user", "content": "hi"})
    assert "[[people/anna]]" not in conversation._conversation[0]["content"]

    v_pages.apply_page_ops([a_page()], "s")
    conversation.refresh_context()

    assert "[[people/anna]]" in conversation._conversation[0]["content"]
    assert conversation._conversation[1] == {"role": "user", "content": "hi"}   # history kept


# --------------------------------------------------------------- PRD 46: the briefing digest

CUTOFF = "2026-10-10T12:00:00"


def at(monkeypatch, when: str) -> None:
    """Every vault commit from here on is dated `when` (`wiki._git` inherits the env)."""
    monkeypatch.setenv("GIT_COMMITTER_DATE", when)
    monkeypatch.setenv("GIT_AUTHOR_DATE", when)


def test_briefing_lists_only_the_pages_changed_after_the_cutoff(vault, monkeypatch):
    at(monkeypatch, "2026-10-09T09:00:00")
    v_pages.apply_page_ops([a_page(), a_page("projects/falcon", "Falcon"), a_page("notes/old", "Old")], "seed")
    at(monkeypatch, "2026-10-11T09:00:00")
    write_page("people/anna", "Anna", "Runs the Friday deploy.\n")
    store._commit("wiki: edited people/anna")
    write_page("notes/plan", "Plan", "Ship it.\n")
    store._commit("wiki: created notes/plan")
    os.remove(os.path.join(store.WIKI_PATH, "notes/old.md"))
    store._commit("wiki: tidy notes/old")
    v_pages.apply_page_ops([a_page("people/ben", "Ben")], "session 2026-10-11")
    v_index._log(["lint: 1 finding"])
    store._commit("wiki: lint")

    digest = a_digest.changes_since(CUTOFF)

    assert {(p["path"], p["title"], p["change"]) for p in digest["pages"]} == {
        ("notes/plan", "Plan", "new"), ("people/anna", "Anna", "edited"), ("notes/old", "old", "deleted")}
    assert digest["chat_pages"] == ["people/ben"]
    paths = [p["path"] for p in digest["pages"]] + digest["chat_pages"]
    assert not any(p in ("index", "log", "schema") or p.startswith(("raw/", "sources/")) for p in paths)
    assert digest["since"] == CUTOFF and not a_digest.digest_empty(digest)


def test_briefing_with_git_missing_has_no_pages(wiki_sandbox):
    digest = a_digest.changes_since(CUTOFF)
    assert digest["pages"] == [] and digest["chat_pages"] == [] and a_digest.digest_empty(digest)


def test_briefing_counts_later_sources_and_memory_ops_but_not_hand_edits(vault):
    from vault import manifest
    write_page("sources/old-split", "Old split", "x\n", source="user drop 2026-10-11")
    write_page("sources/older-split", "Older split", "x\n", source="user drop 2026-10-01")
    entries = [
        {"sha256": "a", "path": "raw/inbox/a.md", "kind": "text", "ingested_at": "2026-10-09T08:00:00",
         "page": "sources/a"},
        {"sha256": "b", "path": "raw/inbox/b.pdf", "kind": "pdf", "ingested_at": "2026-10-11T08:00:00",
         "page": "sources/b"},
        {"sha256": "c", "path": "raw/inbox/c.md", "kind": "text", "ingested_at": "09:30:00",
         "page": "sources/old-split"},
        {"sha256": "d", "path": "raw/inbox/d.md", "kind": "text", "ingested_at": "09:30:00",
         "page": "sources/older-split"},
    ]
    store._write(manifest.MANIFEST, "".join(json.dumps(e) + "\n" for e in entries))
    ops = [
        {"ts": "2026-10-09T08:00:00", "op": "ADD", "path": "people[]", "value": {"name": "Old"}, "reason": "said"},
        {"ts": "2026-10-11T08:00:00", "op": "ADD", "path": "people[]", "value": {"name": "Wife", "rel": "spouse"},
         "reason": "said"},
        {"ts": "2026-10-11T08:01:00", "op": "ADD", "path": "preferences.coffee", "value": "black", "reason": "said"},
        {"ts": "2026-10-11T08:02:00", "op": "DELETE", "path": "people[0]", "value": None, "reason": "ui"},
        {"ts": "2026-10-11T08:03:00", "op": "CONFIRM", "path": "people[1]", "value": None, "reason": "again"},
        {"ts": "2026-10-11T08:04:00", "op": "UPDATE", "path": "identity.city", "value": "Bern", "reason": "moved"},
    ]
    log_ops(*ops)

    digest = a_digest.changes_since(CUTOFF)

    assert [(s["page"], s["kind"], s["at"]) for s in digest["sources"]] == [
        ("sources/b", "pdf", "2026-10-11T08:00:00"), ("sources/old-split", "text", "2026-10-11T09:30:00")]
    mem = digest["memory"]
    assert (mem["added"], mem["updated"], mem["removed"]) == (2, 1, 0)
    assert [(e["op"], e["path"]) for e in mem["examples"]] == [("ADD", "people[]"), ("ADD", "preferences.coffee")]
    card = a_digest.briefing_card(digest)
    assert "## Sources\n- [[sources/b]] · pdf\n" in card and "## Pages" not in card
    assert "## Memory\n- 2 added, 1 updated, 0 removed\n" in card and "Wife · spouse" in card


def test_the_briefing_card_caps_a_section_at_eight_items():
    digest = {"since": CUTOFF, "pages": [{"path": f"notes/p{i}", "title": f"P{i}", "change": "edited"}
                                         for i in range(10)],
              "chat_pages": ["people/a", "people/b", "people/c"], "sources": [],
              "memory": {"added": 0, "updated": 0, "removed": 0, "examples": []}}
    card = a_digest.briefing_card(digest)
    pages = card.split("## Pages\n", 1)[1]
    assert pages.count("- edited: [[notes/p") == 8 and "- +2 more" in pages
    assert "- notes from our chats: [[people/a]], [[people/b]] (+1)" in pages
    assert card.startswith("# Briefing ") and "## Sources" not in card and "## Memory" not in card


@pytest.mark.parametrize("delta, expected", [
    (0, "14:05"), (1, "yesterday at 14:05"), (4, None)])
def test_briefing_when_reads_like_speech(delta, expected):
    from datetime import date, timedelta
    day = date.today() - timedelta(days=delta)
    got = a_digest.briefing_when(f"{day.isoformat()}T14:05:09")
    assert got == (expected or f"{day.day} {day.strftime('%b')}")


def test_the_users_own_editor_saves_and_saved_cards_are_not_news(vault, monkeypatch):
    at(monkeypatch, "2026-10-09T09:00:00")
    v_pages.apply_page_ops([a_page()], "seed")
    at(monkeypatch, "2026-10-11T09:00:00")
    text = read("people/anna.md")
    assert v_pages.save_page("people/anna", text + "\nMore.\n", store.etag(text))[1] is None
    assert v_pages.create_page("notes/mine")[1] is None
    assert v_pages.save_card("notes/2026-10-11-briefing-11-oct", "Briefing 11 Oct", "# Briefing 11 Oct\n", [])[1] is None
    v_pages.apply_page_ops([a_page("projects/falcon", "Falcon")], "seed")

    digest = a_digest.changes_since(CUTOFF)

    assert [(p["path"], p["change"]) for p in digest["pages"]] == [("projects/falcon", "new")]
    assert digest["chat_pages"] == []


def test_lint_counts_a_notebooks_page_pngs_as_cited(vault, monkeypatch):
    """PRD 47: the manifest's `pages` list cites the notebook's PNGs; a replace leaves none behind."""
    from ingest import notebooks
    from tests.conftest import FakeClient

    summary = json.dumps({"title": "Standup", "summary": "Standup: notes.", "entities": [], "unreadable": False})
    text = "Standup: Anna ships the load test."
    monkeypatch.setattr(core_llm, "_client", FakeClient(text, text, summary, "[]"))
    old = notebooks.ingest_notebook([b"\x89PNG\r\n\x1a\n1", b"\x89PNG\r\n\x1a\n2"], title="Standup", device_id="n1")
    assert not [f for f in lint.lint() if ".png" in f]
    monkeypatch.setattr(core_llm, "_client", FakeClient(text, summary, "[]"))
    notebooks.ingest_notebook([b"\x89PNG\r\n\x1a\n3"], title="Standup", device_id="n1", replaces=old["sha256"])
    assert not [f for f in lint.lint() if ".png" in f]
