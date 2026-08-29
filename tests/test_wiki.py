"""Unit tests for the wiki layer — tmp vault, mocked LLM. Mirrors docs/19-wiki-layer.md.

Scenario numbers below are the PRD's test scenarios (1-9); scenario 10 is the live harness
script eval/scripts/wiki.yaml and 11 is manual.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import subprocess

import pytest

import memory
import wiki
from tests.conftest import FakeClient


TURNS = [{"role": "user", "content": "we agreed to deploy on Fridays; Anna runs the release"}]


@pytest.fixture
def vault(wiki_sandbox):
    """A created, committed, empty vault (the autouse fixture only points WIKI_PATH at tmp)."""
    wiki.ensure_vault()
    return wiki_sandbox


def commits() -> list:
    out = subprocess.run(["git", "log", "--oneline"], cwd=wiki.WIKI_PATH,
                         capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def read(rel: str) -> str:
    return wiki._read(rel)


def a_page(path="people/anna", name="Anna", summary="Colleague on the platform team."):
    return {"op": "new_page", "path": path, "type": "person", "name": name, "summary": summary}


# --------------------------------------------------------------- 1. first run


def test_first_run_creates_the_vault(wiki_sandbox):
    assert not wiki.exists()
    wiki.ensure_vault()

    for rel in ("schema.md", "index.md", "log.md", "decisions.md", ".gitignore"):
        assert os.path.isfile(os.path.join(wiki.WIKI_PATH, rel)), rel
    for folder in wiki.FOLDERS:
        assert os.path.isdir(os.path.join(wiki.WIKI_PATH, folder))
    assert ".obsidian/" in read(".gitignore")
    assert "## Rules" in read("schema.md")
    assert os.path.isdir(os.path.join(wiki.WIKI_PATH, ".git"))
    assert len(commits()) == 1


def test_ensure_vault_is_idempotent(vault):
    wiki._write("people/anna.md", "keep me")
    wiki.ensure_vault()
    assert read("people/anna.md") == "keep me"
    assert len(commits()) == 1


def test_nothing_is_created_at_import(wiki_sandbox):
    """Reading the index of a vault that isn't there must not bring one into existence."""
    assert wiki.index_text() == ""
    assert wiki.pages() == []
    assert wiki.lint() == []
    assert not os.path.exists(wiki.WIKI_PATH)


# --------------------------------------------------------------- 2. page ops


def test_new_page_writes_frontmatter_and_sections(vault):
    applied, rejected = wiki.apply_page_ops([a_page()], "session 2026-08-29")
    assert (len(applied), rejected) == (1, [])

    fm, body = wiki.parse_page(read("people/anna.md"))
    assert fm["type"] == "person" and fm["name"] == "Anna" and fm["tags"] == []
    assert fm["source"] == "session 2026-08-29"
    assert fm["updated"] == memory._today_iso() and fm["seen"] == memory._today_iso()
    assert "# Anna" in body
    assert "Colleague on the platform team." in body
    for section in ("## Summary", "## Notes", "## Related"):
        assert section in body


def test_append_note_adds_a_dated_bullet_under_notes(vault):
    wiki.apply_page_ops([a_page()], "s")
    wiki.apply_page_ops([{"op": "append_note", "path": "people/anna",
                          "text": "moved to the release team"}], "session x")
    _, body = wiki.parse_page(read("people/anna.md"))
    notes = body.split("## Notes")[1].split("## Related")[0]
    assert f"- {memory._today_iso()} moved to the release team (source: session x)" in notes


def test_set_field_touches_only_frontmatter(vault):
    wiki.apply_page_ops([a_page()], "s")
    before_body = wiki.parse_page(read("people/anna.md"))[1]
    applied, _ = wiki.apply_page_ops([{"op": "set_field", "path": "people/anna",
                                       "key": "tags", "value": ["team", "release"]}], "s")
    fm, body = wiki.parse_page(read("people/anna.md"))
    assert applied and fm["tags"] == ["team", "release"] and body == before_body


def test_add_decision_appends_a_dated_section(vault):
    wiki.apply_page_ops([{"op": "add_decision", "title": "deploy on Fridays",
                          "text": "Ship at 10:00, Anna runs it."}], "session 2026-08-29")
    text = read("decisions.md")
    assert f"## {memory._today_iso()} deploy on Fridays" in text
    assert "Ship at 10:00, Anna runs it." in text
    assert "source: session 2026-08-29" in text


def test_link_adds_a_wikilink_once(vault):
    wiki.apply_page_ops([a_page(), a_page("projects/atlas", "Atlas", "The platform rewrite.")], "s")
    op = {"op": "link", "path": "people/anna", "target": "projects/atlas"}
    assert wiki.apply_page_ops([op], "s")[0]
    assert "[[projects/atlas]]" in read("people/anna.md")
    applied, rejected = wiki.apply_page_ops([op], "s")     # second time is a no-op
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
    wiki.apply_page_ops([a_page()], "s")
    before = {rel: read(rel) for rel in ("index.md", "log.md", "people/anna.md", "decisions.md")}
    commits_before = len(commits())

    applied, rejected = wiki.apply_page_ops([op], "s")

    assert applied == []
    assert expect in rejected[0]["reason"]
    assert {rel: read(rel) for rel in before} == before
    assert len(commits()) == commits_before


def test_new_page_on_an_existing_page_is_rejected(vault):
    wiki.apply_page_ops([a_page()], "s")
    applied, rejected = wiki.apply_page_ops([a_page(summary="different")], "s")
    assert applied == [] and "already exists" in rejected[0]["reason"]
    assert "different" not in read("people/anna.md")


def test_oversized_note_is_rejected(vault):
    wiki.apply_page_ops([a_page()], "s")
    applied, rejected = wiki.apply_page_ops(
        [{"op": "append_note", "path": "people/anna", "text": "x" * 601}], "s")
    assert applied == [] and "600 characters" in rejected[0]["reason"]


def test_append_to_an_oversized_page_is_rejected(vault):
    wiki.apply_page_ops([a_page()], "s")
    wiki._write("people/anna.md", read("people/anna.md") + "\nfiller " * 4000)
    applied, rejected = wiki.apply_page_ops(
        [{"op": "append_note", "path": "people/anna", "text": "one more"}], "s")
    assert applied == [] and "20 kB" in rejected[0]["reason"]


def test_a_missing_source_falls_back_to_the_batch_default(vault):
    wiki.apply_page_ops([a_page()], "session 2026-08-29")
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
    applied, rejected = wiki.apply_page_ops(ops, "session 2026-08-29")

    assert len(applied) == 4 and rejected == []
    assert len(commits()) == commits_before + 1
    log_lines = [line for line in read("log.md").splitlines() if line.startswith("- ")]
    assert len(log_lines) == 4
    assert all(memory._today_iso() in line for line in log_lines)
    assert any("filed [[people/anna]]" in line for line in log_lines)

    index = read("index.md")
    assert "- [[people/anna]]" in index and "- [[projects/atlas]]" in index
    assert "- [[decisions]]" in index
    assert "[[index]]" not in index and "[[log]]" not in index and "[[schema]]" not in index


def test_index_is_a_pure_function_of_the_pages(vault):
    wiki.apply_page_ops([a_page(), a_page("projects/atlas", "Atlas", "The rewrite.")], "s")
    first = read("index.md")
    wiki._write("index.md", "# Index\n\ngarbage\n")
    assert wiki.rebuild_index() == first


def test_a_partly_bad_batch_applies_the_good_ops(vault):
    applied, rejected = wiki.apply_page_ops([a_page(), {"op": "append_note", "path": "nope/x", "text": "y"}], "s")
    assert len(applied) == 1 and len(rejected) == 1
    assert wiki.page_exists("people/anna.md")


# --------------------------------------------------------------- 4. search / read


def test_search_ranks_the_decisions_page_first(vault):
    wiki.apply_page_ops([
        a_page(),
        a_page("projects/atlas", "Atlas", "The platform rewrite, unrelated to shipping."),
        {"op": "add_decision", "title": "deploy on Fridays",
         "text": "We decided the deploy happens on Friday mornings."},
    ], "s")
    hits = wiki.search("deploy decision")
    assert hits and hits[0]["path"] == "decisions"
    assert "deploy" in hits[0]["snippet"].lower()


def test_search_returns_at_most_k_hits_and_nothing_for_a_miss(vault):
    wiki.apply_page_ops([a_page(f"people/p{i}", f"P{i}", "colleague") for i in range(7)], "s")
    assert len(wiki.search("colleague")) == 5
    assert wiki.search("submarine") == []


def test_search_sees_a_page_edited_on_disk(vault):
    wiki.apply_page_ops([a_page()], "s")
    assert wiki.search("kayaking") == []
    wiki._write("people/anna.md", read("people/anna.md") + "\n- she is into kayaking\n")
    assert [h["path"] for h in wiki.search("kayaking")] == ["people/anna"]


def test_read_page_truncates_at_four_kb(vault):
    wiki.apply_page_ops([a_page()], "s")
    wiki._write("people/anna.md", read("people/anna.md") + "\n" + "long " * 2000)
    text = wiki.read_page("people/anna")
    assert text.endswith(wiki.TRUNCATED)
    assert len(text.encode()) <= wiki.MAX_READ_BYTES + len(wiki.TRUNCATED)


def test_read_page_reports_an_unknown_path(vault):
    assert wiki.read_page("people/nobody").startswith("no such page")
    assert wiki.read_page("../../etc/passwd").startswith("no such page")


def test_the_tools_wrap_search_and_read(vault):
    from tools import run_tool

    wiki.apply_page_ops([a_page()], "s")
    assert "people/anna:" in run_tool("search_wiki", {"query": "colleague platform"})
    assert "# Anna" in run_tool("read_page", {"path": "people/anna"})
    assert run_tool("search_wiki", {"query": "submarine"}) == "no matching wiki pages"


# --------------------------------------------------------------- 5. prompt block


def test_load_injects_the_wiki_index(vault, memory_sandbox):
    wiki.apply_page_ops([a_page()], "s")
    prompt = memory.load("BASE")
    assert "<wiki_index>" in prompt and "[[people/anna]]" in prompt
    assert "read_page" in prompt and "search_wiki" in prompt
    block = prompt.split("<wiki_index>")[1].split("</wiki_index>")[0]
    assert memory.tokens_est(block) <= wiki.WIKI_INDEX_TOKENS


def test_the_tool_hint_is_dropped_when_the_tools_are_not_registered(vault, memory_sandbox,
                                                                    monkeypatch):
    """PRD 19 Results: tool-free runs (the harness sets `llm.TOOLS = []`) get the index only.

    Advertising search_wiki to a model that has not been offered it only teaches it to type a
    tool call at the user.
    """
    import llm

    wiki.apply_page_ops([a_page()], "s")
    monkeypatch.setattr(llm, "TOOLS", [])

    prompt = memory.load("BASE")

    assert not wiki.tools_registered()
    assert "<wiki_index>" in prompt and "[[people/anna]]" in prompt
    assert "search_wiki" not in prompt and "read_page" not in prompt
    assert prompt.rstrip().endswith("</wiki_index>")


def test_the_hint_names_the_tools_that_are_actually_registered():
    import llm

    assert wiki.tools_registered()
    assert set(wiki.RECALL_TOOLS) <= {t["function"]["name"] for t in llm.TOOLS}
    assert all(name in memory.WIKI_NOTE for name in wiki.RECALL_TOOLS)


def test_an_empty_vault_adds_no_block(vault, memory_sandbox):
    assert memory.load("BASE") == "BASE"
    assert "<wiki_index>" not in memory.load("BASE")


def test_the_index_block_is_trimmed_to_its_budget(vault, memory_sandbox, monkeypatch):
    wiki.apply_page_ops([a_page(f"people/p{i}", f"Person number {i}", "a colleague of the user")
                         for i in range(40)], "s")
    monkeypatch.setattr(wiki, "WIKI_INDEX_TOKENS", 50)
    block = memory.load("BASE").split("<wiki_index>")[1].split("</wiki_index>")[0]
    assert memory.tokens_est(block) <= 50


# --------------------------------------------------------------- 6. lint


def test_lint_flags_a_broken_link_and_a_page_missing_from_the_index(vault):
    wiki.apply_page_ops([a_page()], "s")
    wiki._write("people/anna.md", read("people/anna.md") + "\n- see [[projects/ghost]]\n")
    wiki._write("index.md", "# Index\n\n(no pages yet)\n")

    findings = wiki.lint()
    assert any("broken link [[projects/ghost]]" in f for f in findings)
    assert any("missing from index.md" in f for f in findings)
    assert all(f.startswith("people/anna:") for f in findings)   # suffix-less, like wikilinks

    wiki.rebuild_index()
    findings = wiki.lint()
    assert not any("missing from index.md" in f for f in findings)
    assert any("broken link" in f for f in findings)


def test_lint_flags_missing_frontmatter_keys(vault):
    wiki._write("people/bare.md", "---\ntype: person\n---\n\n# Bare\n")
    findings = wiki.lint()
    assert any("frontmatter missing" in f and "name" in f for f in findings)


def test_lint_and_log_only_writes_when_there_are_findings(vault):
    wiki.apply_page_ops([a_page()], "s")
    before = read("log.md")
    assert wiki.lint_and_log() == []
    assert read("log.md") == before

    wiki._write("people/anna.md", read("people/anna.md") + "\n- see [[projects/ghost]]\n")
    assert wiki.lint_and_log()
    assert "lint:" in read("log.md")


# --------------------------------------------------------------- 7. ingest


def test_ingest_files_the_ops_the_model_proposes(vault):
    client = FakeClient(json.dumps([
        {"op": "new_page", "path": "people/anna", "type": "person", "name": "Anna",
         "summary": "Runs the release."},
        {"op": "add_decision", "title": "deploy on Fridays", "text": "Ship at 10:00."},
    ]))
    commits_before = len(commits())

    applied = wiki.ingest(TURNS, client, "m")

    assert len(applied) == 2
    assert wiki.page_exists("people/anna.md")
    assert "deploy on Fridays" in read("decisions.md")
    assert len(commits()) == commits_before + 1
    system = client.calls[0]["messages"][0]["content"]
    user = client.calls[0]["messages"][1]["content"]
    assert "page operation" in system
    assert "## Rules" in user and "Transcript:" in user


@pytest.mark.parametrize("content", ["NOTHING AT ALL", "", None, "{}", "[]"])
def test_ingest_of_garbage_changes_nothing(vault, content):
    before = {rel: read(rel) for rel in ("index.md", "log.md", "decisions.md")}
    commits_before = len(commits())

    assert wiki.ingest(TURNS, FakeClient(content), "m") == []

    assert {rel: read(rel) for rel in before} == before
    assert len(commits()) == commits_before
    assert wiki.page_paths() == []


def test_ingest_survives_a_broken_client(vault, capsys):
    assert wiki.ingest(TURNS, FakeClient(RuntimeError("connection refused")), "m") == []
    assert "ingest failed" in capsys.readouterr().out


def test_ingest_skips_the_call_for_an_empty_transcript(vault):
    client = FakeClient("[]")
    assert wiki.ingest([], client, "m") == []
    assert client.calls == []


def test_save_ingests_after_the_profile_ops(memory_sandbox, wiki_sandbox):
    profile_ops = json.dumps([{"op": "ADD", "path": "identity.name", "value": "Mihaly"}])
    page_ops = json.dumps([{"op": "new_page", "path": "people/anna", "type": "person",
                            "name": "Anna", "summary": "Runs the release."}])
    client = FakeClient(profile_ops, page_ops)

    applied = memory.save(TURNS, client, "m")

    assert [op["path"] for op in applied] == ["identity.name"]
    assert wiki.page_exists("people/anna.md")


def test_a_wiki_failure_never_loses_the_profile_save(memory_sandbox, wiki_sandbox, monkeypatch):
    monkeypatch.setattr(wiki, "ingest", lambda *a: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    ops = json.dumps([{"op": "ADD", "path": "identity.name", "value": "Mihaly"}])

    assert memory.save(TURNS, FakeClient(ops), "m")
    assert memory.load_profile()["identity"]["name"] == "Mihaly"


# --------------------------------------------------------------- 8. archive / pull


def test_archive_files_a_person_and_marks_the_entry_stale(vault, memory_sandbox):
    from tests.conftest import seed

    seed(people=[{"name": "Anna", "rel": "colleague", "note": "platform team",
                  memory.SEEN_FIELD: "2026-01-05"}])

    profile, error = memory.archive("people", 0)

    assert error is None
    _, body = wiki.parse_page(read("people/anna.md"))
    history = body.split("## History")[1]
    assert "Anna; colleague; platform team" in history
    assert "last seen 2026-01-05" in history
    assert profile["people"][0][memory.STALE_FIELD] is True
    assert memory.load_profile()["people"][0][memory.STALE_FIELD] is True


def test_archive_appends_to_an_existing_page(vault, memory_sandbox):
    from tests.conftest import seed

    wiki.apply_page_ops([a_page()], "s")
    seed(people=[{"name": "Anna", "rel": "colleague", memory.SEEN_FIELD: "2026-01-05"}])

    _, error = memory.archive("people", 0)

    assert error is None
    assert read("people/anna.md").count("# Anna") == 1
    assert "## History" in read("people/anna.md")


def test_archive_of_a_recurring_entry_lands_on_one_shared_page(vault, memory_sandbox):
    from tests.conftest import seed

    seed(recurring=[{"what": "gym", "when": "Tue/Thu", memory.SEEN_FIELD: "2026-02-02"}])
    _, error = memory.archive("recurring", 0)
    assert error is None
    assert "gym; Tue/Thu" in read(wiki.RECURRING_FILE)


def test_archive_refuses_a_bad_target(vault, memory_sandbox):
    _, error = memory.archive("people", 3)
    assert "does not exist" in error["reason"]
    _, error = memory.archive("episodic", 0)
    assert "cannot archive" in error["reason"]


def test_pull_recreates_a_live_entry_from_a_page(vault, memory_sandbox):
    from tests.conftest import seed

    seed(people=[{"name": "Anna", "rel": "colleague", memory.SEEN_FIELD: "2026-01-05"}])
    memory.archive("people", 0)

    profile, error = memory.pull("people/anna", "people")

    assert error is None
    fresh = [e for e in profile["people"] if not memory.is_stale(e)]
    assert len(fresh) == 1
    assert fresh[0]["name"] == "Anna"
    assert fresh[0][memory.SEEN_FIELD] == memory._today_iso()


def test_pull_reports_an_unknown_page(vault, memory_sandbox):
    _, error = memory.pull("people/nobody", "people")
    assert "no such page" in error["reason"]


# --------------------------------------------------------------- 9. user edits


def test_a_user_edit_reaches_the_next_ingest_and_the_index(vault, memory_sandbox):
    wiki.ingest(TURNS, FakeClient(json.dumps([a_page()])), "m")
    assert "Anna" in read("index.md")

    # the user opens the page in Obsidian and corrects it
    fm, body = wiki.parse_page(read("people/anna.md"))
    fm["name"] = "Anna Kovacs"
    wiki._write("people/anna.md", wiki._page_text(fm, body + "\n- she prefers Anna K\n"))

    client = FakeClient("[]")
    wiki.ingest(TURNS, client, "m")
    context = client.calls[0]["messages"][1]["content"]
    assert "Anna Kovacs" in context

    wiki.rebuild_index()
    assert "Anna Kovacs" in read("index.md")


# --------------------------------------------------------------- actions (PRD scenario 11)


@pytest.fixture
def wiki_panel(vault, memory_sandbox, panel):
    """The shared `panel` websocket client, with a created vault and a sandboxed profile."""
    return panel


def test_get_wiki_lists_pages_with_lint_and_vault(wiki_panel):
    wiki.apply_page_ops([a_page()], "s")
    msg = wiki_panel.send({"action": "get_wiki"})
    assert msg["type"] == "wiki"
    assert msg["vault"] == wiki.WIKI_PATH
    assert msg["lint"] == []
    assert [p["path"] for p in msg["pages"]] == ["people/anna"]
    assert msg["pages"][0] == {"path": "people/anna", "title": "Anna",
                               "type": "person", "updated": memory._today_iso()}


def test_read_wiki_page_returns_text_and_links(wiki_panel):
    wiki.apply_page_ops([a_page(), a_page("projects/atlas", "Atlas", "The rewrite."),
                         {"op": "link", "path": "people/anna", "target": "projects/atlas"}], "s")
    msg = wiki_panel.send({"action": "read_wiki_page", "path": "people/anna"})
    assert msg["type"] == "wiki_page" and msg["path"] == "people/anna"
    assert "# Anna" in msg["text"]
    assert msg["links"] == ["projects/atlas"]


def test_wiki_note_appends_and_refreshes_both_panes(wiki_panel):
    wiki.apply_page_ops([a_page()], "s")
    page, listing = wiki_panel.send_many(
        {"action": "wiki_note", "path": "people/anna", "text": "she prefers async standups"}, 2)
    assert page["type"] == "wiki_page" and listing["type"] == "wiki"
    assert f"user note {memory._today_iso()}" in page["text"]
    assert "she prefers async standups" in page["text"]


def test_wiki_note_on_a_missing_page_errors(wiki_panel):
    msg = wiki_panel.send({"action": "wiki_note", "path": "people/ghost", "text": "hi"})
    assert msg["type"] == "error" and "no such page" in msg["text"]


def test_archive_memory_action_replies_with_memory_and_wiki(wiki_panel):
    from tests.conftest import seed

    seed(people=[{"name": "Anna", "rel": "colleague", memory.SEEN_FIELD: "2026-01-05"}])
    mem, listing = wiki_panel.send_many(
        {"action": "archive_memory", "section": "people", "index": 0}, 2)
    assert mem["type"] == "memory" and mem["profile"]["people"][0]["stale"] is True
    assert [p["path"] for p in listing["pages"]] == ["people/anna"]


def test_pull_wiki_action_replies_with_memory_and_wiki(wiki_panel):
    wiki.apply_page_ops([a_page()], "s")
    mem, listing = wiki_panel.send_many(
        {"action": "pull_wiki", "path": "people/anna", "section": "people"}, 2)
    assert mem["type"] == "memory"
    assert mem["profile"]["people"][0]["name"] == "Anna"
    assert listing["type"] == "wiki"


def test_a_bad_archive_comes_back_as_memory_error(wiki_panel):
    msg = wiki_panel.send({"action": "archive_memory", "section": "people", "index": 9})
    assert msg["type"] == "memory_error" and "does not exist" in msg["reason"]
