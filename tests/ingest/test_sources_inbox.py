"""PRD 25 — the Sources inbox: list / reingest / forget + the WS actions. Tmp vault, mocked LLM.

Scenario numbers are the PRD's. Scenario 0 (no `people/` page for the user) is PRD 26's work and
lives in tests/ingest/test_sources.py; 5 is manual.
"""
import os

import json

import pytest
from unittest.mock import patch

from server import panels
from server import turns
from core import util
from ingest import pipeline
from vault import manifest
from vault import lint
from vault import pages
from vault import store
from tests.conftest import FakeClient, commits, read
from tests.ingest.conftest import OPS, SUMMARY
from core import llm as core_llm

RE_SUMMARY = json.dumps({
    "title": "Falcon status",
    "summary": "Project Falcon is green again; p99 is down to 310 ms.",
    "entities": [{"name": "Falcon", "type": "project"}],
    "unreadable": False,
})

RE_OPS = json.dumps([
    {"op": "append_note", "path": "projects/falcon", "text": "p99 down to 310 ms after the fix."},
])


def _file_falcon(monkeypatch) -> dict:
    """The PRD 20 fixture drop: `projects/falcon` is created by the ops, one commit."""
    client = FakeClient(SUMMARY, OPS)
    monkeypatch.setattr(core_llm, "_client", client)
    return pipeline.ingest_upload(b"# Falcon\nload test: 940 ms p99", "falcon.md")


def _sha(filed: dict) -> str:
    return next(e["sha256"] for e in manifest.manifest_entries() if e["path"] == filed["raw"])


# ------------------------------------------------------------ 1. list_sources, newest first


def test_list_sources_is_newest_first_with_the_pages_fed(vault, monkeypatch):
    first = _file_falcon(monkeypatch)
    client = FakeClient(json.dumps({"title": "Lunch menu", "summary": "Soup.", "entities": []}), "[]")
    monkeypatch.setattr(core_llm, "_client", client)
    second = pipeline.ingest_upload(b"soup of the day", "menu.txt")

    rows = pipeline.list_sources()

    assert [r["path"] for r in rows] == [second["path"], first["path"]]
    assert rows[1] == {
        "sha": _sha(first), "path": "sources/falcon-status", "raw": first["raw"], "kind": "md",
        "title": "Falcon status", "when": rows[1]["when"], "fed": ["projects/falcon"],
        "meeting": "",
    }
    assert rows[1]["when"].startswith(util.today_iso())
    assert rows[0]["kind"] == "txt" and rows[0]["fed"] == []


def test_list_sources_is_empty_without_a_manifest(vault):
    assert pipeline.list_sources() == []


# ------------------------------------------------------------ 2. reingest


def test_reingest_rewrites_the_source_page_adds_notes_and_commits_once(vault, monkeypatch):
    filed = _file_falcon(monkeypatch)
    sha = _sha(filed)
    before = len(commits())

    monkeypatch.setattr(core_llm, "_client", FakeClient(RE_SUMMARY, RE_OPS))
    result = pipeline.reingest(sha)

    assert result == {"path": "sources/falcon-status", "raw": filed["raw"]}
    page = read("sources/falcon-status.md")
    assert "green again" in page and "940 ms" not in page          # summary replaced
    assert f"sha256: {sha}" in page and "[[projects/falcon]]" in page
    falcon = read("projects/falcon.md")
    assert f"- {util.today_iso()} p99 down to 310 ms after the fix. (source: {filed['raw']})" in falcon
    assert len(commits()) == before + 1
    entries = manifest.manifest_entries()
    assert len(entries) == 1 and entries[0]["sha256"] == sha
    assert entries[0]["pages_touched"] == ["projects/falcon"]
    assert "re-ingested [[sources/falcon-status]]" in read("log.md")


def test_reingest_of_an_unknown_sha_is_refused(vault, monkeypatch):
    with pytest.raises(pipeline.IngestError) as exc:
        monkeypatch.setattr(core_llm, "_client", FakeClient())
        pipeline.reingest("0" * 64)
    assert exc.value.status == 404


# ------------------------------------------------------------ 3. forget


def test_forget_removes_page_raw_manifest_line_and_only_its_notes(vault, monkeypatch):
    filed = _file_falcon(monkeypatch)
    sha = _sha(filed)
    monkeypatch.setattr(core_llm, "_client", FakeClient(RE_SUMMARY, RE_OPS))
    pipeline.reingest(sha)
    pages.apply_page_ops([{"op": "append_note", "path": "projects/falcon",
                          "text": "Kept: said in a meeting.", "source": "session 2026-08-29"}],
                        "session 2026-08-29", commit=True)
    assert "(source: session 2026-08-29)" in read("projects/falcon.md")
    before = len(commits())

    result = pipeline.forget(sha)

    assert result == {"path": "sources/falcon-status", "raw": filed["raw"],
                      "cleaned": ["projects/falcon"], "memory": 0}
    assert not store.page_exists("sources/falcon-status.md")
    assert not os.path.exists(store._path(filed["raw"]))
    assert manifest.manifest_entries() == []
    falcon = read("projects/falcon.md")
    assert f"(source: {filed['raw']})" not in falcon
    assert "Kept: said in a meeting. (source: session 2026-08-29)" in falcon
    assert "## Summary\nPlatform project, amber." in falcon       # the page itself survives
    assert len(commits()) == before + 1
    assert lint.lint() == [f"projects/falcon: source {filed['raw']} no longer exists"]   # PRD 44
    assert "[[sources/falcon-status]]" not in read("index.md")


def test_without_source_notes_drops_only_matching_bullets_in_notes_and_history():
    body = ("# P\n\n## Summary\nsee (source: raw/x.md)\n\n## Notes\n- 2026-08-01 a (source: raw/x.md)\n"
            "- 2026-08-02 b (source: raw/y.md)\n\n## Related\n- [[q]]\n\n## History\n"
            "- 2026-08-03 c (source: raw/x.md)\n")
    out = pipeline._without_source_notes(body, "raw/x.md")
    assert out == ("# P\n\n## Summary\nsee (source: raw/x.md)\n\n## Notes\n"
                   "- 2026-08-02 b (source: raw/y.md)\n\n## Related\n- [[q]]\n\n## History\n")


# ------------------------------------------------------------ 4. actions through the Panel


@pytest.fixture
def inbox(vault, fake, panel):
    """The Panel fixture, also listening for `sources` messages, over a vault with one filing."""
    panel.WANTED = panel.WANTED + ("sources",)
    filed = pipeline.ingest_upload(b"# Falcon\nload test: 940 ms p99", "falcon.md")
    return panel, filed


def test_get_sources_round_trips_through_the_panel(inbox):
    panel, filed = inbox
    reply = panel.send({"action": "get_sources"})
    assert reply["type"] == "sources"
    assert [s["path"] for s in reply["sources"]] == [filed["path"]]
    assert reply["sources"][0]["fed"] == ["projects/falcon"]


def test_forget_source_proposes_and_changes_nothing(inbox):
    """PRD 52: the action no longer deletes; it starts the approval (a turn) — nothing to confirm."""
    panel, filed = inbox
    sha = _sha(filed)
    with patch.object(turns, "propose_turn") as propose:
        panel.ws.send_json({"action": "forget_source", "sha": sha})
        panel.send({"action": "get_sources"})            # the action has been handled by now
        panels.controller.join_idle()
    assert propose.call_args.args[2:4] == ("forget", sha)
    assert manifest.manifest_entries()[0]["sha256"] == sha


def test_forget_source_with_an_unknown_sha_is_an_error(inbox):
    panel, _ = inbox
    reply = panel.send({"action": "forget_source", "sha": "nope"})
    assert reply["type"] == "error" and reply["scope"] == "sources" and "no filed source" in reply["text"]


def test_reingest_action_refreshes_the_page_list(inbox):
    panel, filed = inbox
    wiki_msg, srcs = panel.send_many({"action": "reingest_source", "sha": _sha(filed)}, 2)
    assert wiki_msg["type"] == "wiki" and srcs["type"] == "sources"
    assert srcs["sources"][0]["path"] == filed["path"]


def test_reingest_action_with_an_unknown_sha_is_an_error(inbox):
    panel, _ = inbox
    reply = panel.send({"action": "reingest_source", "sha": "nope"})
    assert reply["type"] == "error" and "no filed source" in reply["text"]
