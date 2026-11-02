"""PRD 53 — the persisted search index (index_db.py). Scenario numbers are the PRD's.

The goldens (scenarios 1 and 6) were recorded on the pre-PRD in-memory `_bm25` path; regenerate them
only on purpose: `GOLDEN_UPDATE=1 .venv/bin/python -m pytest tests/vault/test_index_db.py -k golden`.
"""
import json
import os
import subprocess
import sys
import re

import pytest

from vault import index_db
from ingest import pipeline
from vault import manifest
from vault import chunks as v_chunks
from vault import index
from vault import pages as m_pages
from vault import search as v_search
from vault import store
from tests.conftest import ROOT, commits, file_faq, read, trace_lines
from core import util
from vault import __main__ as vault_cli

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fixtures")
SEARCH_GOLDEN = os.path.join(FIXTURES, "search_golden.json")
INDEX_GOLDEN = os.path.join(FIXTURES, "index_text_golden.txt")
UPDATE = os.getenv("GOLDEN_UPDATE") == "1"

#: FAQ-section phrasings (conftest FALLOUT_SECTIONS / SKYRIM_SECTIONS), the seeded pages, and misses
QUERIES = (
    "I'm at Covenant, what about Honest Dan?",
    "where is the SAFE test in Covenant",
    "how do I get to Sanctuary Hills with Codsworth",
    "Preston Garvey at the Red Rocket",
    "lockpicking sweet spot bobby pin",
    "how does the tumbler work when lockpicking in Skyrim",
    "where is Jarl Balgruuf in Whiterun",
    "Dragonsreach Companions Jorrvaskr",
    "Riverwood and the Golden Claw from Alvor",
    "joining the Thieves Guild in Riften with Brynjolf",
    "what does the Fallout guide say about hacking a terminal",
    "what does the Skyrim guide say about the Sanctuary",
    "Dark Brotherhood Astrid Falkreath black door",
    "power armor fusion core repair bay",
    "who is Ada Lovelace",
    "what should we cook tonight",
    "Anna runs the Friday deploy",
    "Falcon load test amber",
    "summarise the Fallout 4 guide",
    "quantum chromodynamics lecture",
)

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _undated(value):
    """Today's date sits in raw paths and frontmatter; the golden must not depend on the day it ran."""
    if isinstance(value, str):
        return _DATE.sub("DATE", value)
    if isinstance(value, list):
        return [_undated(v) for v in value]
    if isinstance(value, tuple):
        return [_undated(v) for v in value]
    if isinstance(value, dict):
        return {k: _undated(v) for k, v in value.items()}
    return value


def _snapshot() -> dict:
    out = {}
    for query in QUERIES:
        pages = [{"path": h["path"], "score": h["score"], "snippet": h["snippet"]}
                 for h in v_search.search(query, 5)]
        chunks = [{"source": h["source"], "raw": h["raw"], "title": h["title"], "n": h["n"],
                   "total": h["total"], "score": h["score"], "matched": h["matched"],
                   "phrases": [list(p) for p in h["phrases"]], "named": h["named"],
                   "text_head": h["text"][:80], "text_len": len(h["text"])}
                  for h in v_chunks.search_chunks(query, 3)]
        out[query] = {"search": pages, "search_chunks": chunks}
    return _undated(out)


# --------------------------------------------------------------- 1. golden scores


def test_search_and_search_chunks_match_the_recorded_golden(both):
    got = _snapshot()
    if UPDATE:
        with open(SEARCH_GOLDEN, "w") as f:
            json.dump(got, f, indent=1, ensure_ascii=False, sort_keys=True)
            f.write("\n")
    with open(SEARCH_GOLDEN) as f:
        golden = json.load(f)
    assert sum(bool(v["search_chunks"]) for v in golden.values()) >= 15     # the golden is not vacuous
    assert sum(bool(v["search"]) for v in golden.values()) >= 8
    for query in QUERIES:
        assert got[query] == golden[query], query


# --------------------------------------------------------------- 6. index_text golden


def test_index_text_matches_the_recorded_golden(both, monkeypatch):
    reads = []
    real_read = store._read
    monkeypatch.setattr(store, "_read", lambda rel: reads.append(rel) or real_read(rel))
    got = _undated(index.index_text())
    assert reads == []                                      # the pages table, not the pages
    monkeypatch.setattr(store, "_read", real_read)
    if UPDATE:
        with open(INDEX_GOLDEN, "w") as f:
            f.write(got)
    with open(INDEX_GOLDEN) as f:
        assert got == f.read()
    assert got.count("\n- [[") >= 4


# --------------------------------------------------------------- helpers


def rows(sql: str, *args) -> list:
    return v_search._db().conn.execute(sql, args).fetchall()


def df(corpus: str, term: str) -> int:
    found = rows("SELECT df FROM terms WHERE corpus=? AND term=?", corpus, term)
    return found[0][0] if found else 0


def sha_of(result: dict) -> str:
    return next(e["sha256"] for e in manifest.manifest_entries() if e["path"] == result["raw"])


def note(path: str, name: str, summary: str) -> dict:
    return {"op": "new_page", "path": path, "type": "note", "name": name, "summary": summary}


ZYX = "Field notes\n\nThe zyxwombat lives in the Quorbel marsh and eats reeds at dusk.\n"


# --------------------------------------------------------------- 2. incremental ingest


def test_filing_one_source_adds_postings_for_its_chunks_only(fallout):
    before = v_search._db().dump()
    zyx = file_faq(ZYX, "Marsh notes", "marsh.md")
    after = v_search._db().dump()

    sha = sha_of(zyx)
    old_chunks = [p for p in before["postings"] if p[0] == "chunk"]
    new_chunks = [p for p in after["postings"] if p[0] == "chunk"]
    assert set(old_chunks) - set(new_chunks) == set()                # nothing else was touched
    added = set(new_chunks) - set(old_chunks)
    assert added and {p[2][0] for p in added} == {sha}               # doc = (sha, n)
    assert [c for c in after["chunks"] if c not in before["chunks"]] == \
        [c for c in after["chunks"] if c[2] == sha]

    hits = v_chunks.search_chunks("where does the zyxwombat live", 3)
    assert hits[0]["raw"] == zyx["raw"] and "zyxwombat" in hits[0]["text"]


# --------------------------------------------------------------- 3. remove paths


def test_rename_delete_and_forget_leave_no_rows_behind(seeded_vault):
    m_pages.apply_page_ops([note("notes/first", "First", "The zephyr crossing."),
                         note("notes/second", "Second", "Another zephyr crossing.")], "s")
    assert df("page", "zephyr") == 2

    assert m_pages.rename_page("notes/first", "notes/moved")[1] is None
    assert rows("SELECT * FROM pages WHERE path='notes/first.md'") == []
    assert rows("SELECT * FROM postings WHERE corpus='page' AND doc='notes/first.md'") == []
    assert {h["path"] for h in v_search.search("zephyr")} == {"notes/moved", "notes/second"}

    assert m_pages.delete_page("notes/second")[1] is None
    assert rows("SELECT * FROM pages WHERE path='notes/second.md'") == []
    assert rows("SELECT * FROM postings WHERE corpus='page' AND doc='notes/second.md'") == []
    assert df("page", "zephyr") == 1

    zyx = file_faq(ZYX, "Marsh notes", "marsh.md")
    sha = sha_of(zyx)
    ids = [i for (i,) in rows("SELECT id FROM chunks WHERE sha=?", sha)]
    assert ids and df("chunk", "zyxwombat") == len(ids)
    pipeline.forget(sha)
    assert rows("SELECT * FROM chunks WHERE sha=?", sha) == []
    assert rows(f"SELECT * FROM postings WHERE corpus='chunk' AND doc IN ({','.join('?' * len(ids))})", *ids) == []
    assert rows("SELECT * FROM pages WHERE path=?", zyx["path"] + ".md") == []
    assert df("chunk", "zyxwombat") == 0 and rows("SELECT * FROM proper WHERE term='zyxwombat'") == []
    assert v_chunks.search_chunks("zyxwombat") == []


# --------------------------------------------------------------- 4. rebuild equals incremental


def test_reindex_from_scratch_equals_the_incrementally_reached_state(seeded_vault, tmp_path):
    m_pages.apply_page_ops([note("notes/a", "A", "Alpha zephyr."), note("notes/b", "B", "Beta [[notes/a]].")], "s")
    first = file_faq(ZYX, "Marsh notes", "marsh.md")                                   # 1
    second = file_faq("Quarry log\n\nThe Quorbel quarry floods in spring.\n", "Quarry", "q.md")   # 2
    m_pages.save_page("notes/a", read("notes/a.md") + "\n- more zephyr\n", store.etag(read("notes/a.md")))  # 3
    m_pages.rename_page("notes/b", "notes/c")                                                 # 4
    m_pages.delete_page("people/anna")                                                        # 5
    m_pages.create_page("notes/fresh")                                                        # 6
    m_pages.apply_page_ops([{"op": "append_note", "path": "projects/falcon", "note": "Quorbel kickoff."}], "s")  # 7
    pipeline.forget(sha_of(first))                                                          # 8
    m_pages.save_card("notes/card", "Card", "Quorbel answer.", ["q.md"])                      # 9
    m_pages.apply_page_ops([{"op": "add_decision", "decision": "Ship Quorbel."}], "s")        # 10
    assert second["raw"] and v_search.search("quorbel")

    rebuilt = index_db.Index(v_search.reindex(str(tmp_path / "rebuilt.sqlite")))
    try:
        assert rebuilt.dump() == v_search._db().dump()
    finally:
        rebuilt.close()
    assert v_search._db().dump()["pages"]                                  # not two empty files


def test_a_deleted_index_is_rebuilt_on_the_next_search(seeded_vault):
    before = v_search._db().dump()
    os.remove(index_db.db_path(store.WIKI_PATH))
    assert [h["path"] for h in v_search.search("anna")] == ["people/anna"]
    assert v_search._db().dump() == before
    assert v_search.reindex() == index_db.db_path(store.WIKI_PATH)
    assert v_search._db().dump() == before


# --------------------------------------------------------------- 7. startup check


def test_an_edit_made_while_the_app_was_down_is_repaired_at_startup(seeded_vault):
    full = os.path.join(store.WIKI_PATH, "people/anna.md")
    with open(full, "a") as f:                       # no hook ran: the app was down
        f.write("\n- Anna keeps a pet axolotl.\n")
    later = os.stat(full).st_mtime + 10
    os.utime(full, (later, later))
    assert v_search.search("axolotl") == []              # no per-search sweep (docs/53-scalable-index.md)

    index_db.close()                                 # the app restarts
    assert v_search.startup_check() == 1
    assert [h["path"] for h in v_search.search("axolotl")] == ["people/anna"]
    assert v_search.startup_check() == 0                 # nothing left to repair


# --------------------------------------------------------------- 8. filters


def test_search_filters_narrow_pages_and_chunks(seeded_vault):
    m_pages.apply_page_ops([note("notes/anna-lunch", "Anna lunch", "Lunch with Anna."),
                         {"op": "new_page", "path": "people/anna-b", "type": "person", "name": "Anna B",
                          "summary": "Anna from accounting."}], "s")
    store._write("people/anna-old.md", "---\ntype: person\nname: Anna Old\ntags: [archive]\nsource: s\n"
                "updated: 2020-01-01\nseen: 2020-01-01\n---\n# Anna Old\n\n## Summary\nAnna, years ago.\n")
    everything = {h["path"] for h in v_search.search("anna", 10)}
    assert {"notes/anna-lunch", "people/anna-old"} <= everything

    people = [h["path"] for h in v_search.search("anna", 10, filters={"type": "person"})]
    assert set(people) == {"people/anna", "people/anna-b", "people/anna-old"}
    recent = [h["path"] for h in v_search.search("anna", 10, filters={"type": "person", "updated_from": "2026-01-01"})]
    assert set(recent) == {"people/anna", "people/anna-b"}
    assert [h["path"] for h in v_search.search("anna", 10, filters={"tags": ["archive"]})] == ["people/anna-old"]

    zyx = file_faq(ZYX, "Marsh notes", "marsh.md")
    file_faq("Marsh diary\n\nAnother marsh, another dusk.\n", "Diary", "diary.md")
    assert len({h["raw"] for h in v_chunks.search_chunks("what happens at dusk", 5)}) == 2
    narrowed = v_chunks.search_chunks("what happens at dusk", 5, filters={"source": zyx["path"]})
    assert {h["raw"] for h in narrowed} == {zyx["raw"]}
    assert v_chunks.search_chunks("what happens at dusk", 5, filters={"kind": "pdf"}) == []
    kind = next(e["kind"] for e in manifest.manifest_entries() if e["path"] == zyx["raw"])
    assert len({h["raw"] for h in v_chunks.search_chunks("what happens at dusk", 5, filters={"kind": kind})}) == 2


# --------------------------------------------------------------- 9. cache invalidation + crash


def test_a_changed_sha_reindexes_that_source_alone(seeded_vault, monkeypatch):
    first = file_faq(ZYX, "Marsh notes", "marsh.md")
    second = file_faq("Quarry log\n\nThe Quorbel quarry floods in spring.\n", "Quarry", "q.md")
    kept = rows("SELECT id, sha FROM chunks WHERE raw=?", first["raw"])
    old_sha = sha_of(second)

    data = b"Quarry log\n\nThe Quorbel quarry dried out; the vexillum stayed.\n"
    with open(os.path.join(store.WIKI_PATH, second["raw"]), "wb") as f:
        f.write(data)
    manifest._manifest_write([{**e, "sha256": manifest.sha256(data)} if e["path"] == second["raw"] else e
                             for e in manifest.manifest_entries()])
    built = []
    real = v_chunks._raw_chunks_for
    monkeypatch.setattr(v_chunks, "_raw_chunks_for", lambda entry: built.append(entry["path"]) or real(entry))

    assert v_chunks.search_chunks("vexillum")[0]["raw"] == second["raw"]
    assert built == [second["raw"]]                                   # that source alone
    assert rows("SELECT id, sha FROM chunks WHERE raw=?", first["raw"]) == kept
    assert rows("SELECT * FROM chunks WHERE sha=?", old_sha) == []


def test_a_killed_update_leaves_the_prior_state_intact(seeded_vault, monkeypatch):
    idx = v_search._db()
    before = idx.dump()

    def killed(*args, **kw):
        raise RuntimeError("killed mid-transaction")
    with monkeypatch.context() as m:                     # undoes only this patch, never the vault sandbox
        m.setattr(idx, "_add_doc", killed)
        with pytest.raises(RuntimeError):
            store._write("people/anna.md", read("people/anna.md") + "\n- Anna moved to the axolotl team.\n")

    assert idx.dump() == before                          # the old postings were deleted, then rolled back
    assert idx.integrity() == "ok"
    index_db.close()
    assert v_search.startup_check() == 1                     # the restart repairs what the crash left
    assert [h["path"] for h in v_search.search("axolotl")] == ["people/anna"]


# --------------------------------------------------------------- review fixes (round 1)


def test_a_failing_index_hook_never_loses_a_renamed_page(seeded_vault, monkeypatch):
    original = read("people/anna.md")

    def broken(*args, **kw):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(index_db.Index, "drop_page", broken)
    monkeypatch.setattr(index_db.Index, "put_page", broken)
    with pytest.raises(RuntimeError):
        m_pages.rename_page("people/anna", "people/anna-k")
    kept = read("people/anna.md") or read("people/anna-k.md")
    assert kept and "Colleague on the platform team" in kept and kept.split("updated:")[0] == original.split("updated:")[0]


def test_a_source_that_cannot_be_chunked_is_skipped_not_fatal(seeded_vault):
    file_faq(ZYX, "Marsh notes", "marsh.md")
    manifest._manifest_append({"sha256": "f" * 64, "path": "raw/inbox/gone.md", "kind": "md", "page": "sources/gone"})
    index_db.close()                                       # the next open runs the startup check over it

    assert [h["path"] for h in v_search.search("anna")] == ["people/anna"]
    assert "[[people/anna]]" in index.index_text()
    assert v_chunks.search_chunks("where does the zyxwombat live")
    store._write("notes/after.md", read("people/anna.md").replace("Anna", "Bertha"))
    assert [h["path"] for h in v_search.search("bertha")] == ["notes/after"]
    assert "f" * 64 not in v_search._db().source_shas()
    assert v_search.startup_check() == 0
    assert [e for e in trace_lines(kind="error") if e["scope"] == "index" and e["where"] == "sync_sources"]


def test_a_skipped_source_is_retried_once_its_raw_file_is_restored(seeded_vault):
    file_faq(ZYX, "Marsh notes", "marsh.md")
    sha = "e" * 64
    manifest._manifest_append({"sha256": sha, "path": "raw/inbox/later.md", "kind": "md", "page": "sources/later"})
    index_db.close()
    assert v_chunks.search_chunks("quokkaword") == []
    assert sha not in v_search._db().source_shas()             # skipped, and the manifest hash was not stored
    store._write("raw/inbox/later.md", "The quokkaword lives on the island.")
    index_db.close()                                       # the next startup check retries it
    assert v_search.startup_check() == 1
    assert v_chunks.search_chunks("quokkaword")
    assert sha in v_search._db().source_shas()


def test_the_startup_event_logs_an_index_failure_and_carries_on(monkeypatch):
    import asyncio

    from server import app
    monkeypatch.setattr(v_search, "startup_check", lambda: (_ for _ in ()).throw(OSError("disk full")))
    asyncio.run(app._on_startup())
    assert [e for e in trace_lines(kind="error") if e["where"] == "startup_check"]


def test_a_swapped_index_file_is_picked_up_by_an_open_handle(seeded_vault, tmp_path):
    old = v_search._db()
    built = v_search.reindex(str(tmp_path / "next.sqlite"))
    fresh = index_db.Index(built)
    fresh.put_page("notes/new.md", 1, "x", {}, ["swapped"])
    fresh.close()
    live = index_db.db_path(store.WIKI_PATH)
    for suffix in ("-wal", "-shm"):                        # what another process's reindex does (index_db.replace)
        os.remove(live + suffix)
    os.replace(built, live)
    now = index_db.open_index(store.WIKI_PATH)
    assert now is not old
    assert "notes/new.md" in now.page_paths()


def test_the_reindex_cli_refuses_while_the_app_runs(seeded_vault, monkeypatch, capsys):
    monkeypatch.setattr(util, "running_app", lambda: "app is running — scripts/voicebot stop (port 8011)")
    before = v_search._db().dump()
    assert vault_cli.main(["reindex"]) == 2
    assert "app is running" in capsys.readouterr().err
    assert v_search._db().dump() == before


def test_the_reindex_cli_reads_the_vault_from_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f"WIKI_PATH={tmp_path / 'from-dotenv'}\n")
    monkeypatch.delenv("WIKI_PATH", raising=False)
    assert vault_cli.cli_vault(str(env)) == str(tmp_path / "from-dotenv")


def test_the_eval_harness_import_never_opens_the_env_vault(tmp_path):
    home = tmp_path / "home"
    (home / "llm-wiki").mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in ("WIKI_PATH", "MEMORY_DIR", "SETTINGS_PATH", "TRACE_DIR")}
    env.update(HOME=str(home), WIKI_PATH=str(home / "llm-wiki"))
    subprocess.run([sys.executable, "-c", "import eval.harness"], cwd=ROOT, env=env, check=True,
                   capture_output=True, timeout=120)
    assert not (home / "llm-wiki" / ".index").exists()


def test_opening_the_index_leaves_a_committed_vault_clean(vault):
    store._write(".gitignore", ".obsidian/\nraw/.index/\n")          # an existing vault's ignore file
    store._commit("wiki: old ignore")
    index_db.close()
    store.ensure_vault()
    assert v_search.search("anything") == [] and os.path.exists(index_db.db_path(store.WIKI_PATH))
    assert subprocess.run(["git", "status", "--porcelain"], cwd=store.WIKI_PATH, capture_output=True,
                          text=True).stdout == ""
    assert commits()


def test_a_failed_startup_check_is_retried(seeded_vault, monkeypatch):
    index_db.close()
    real = v_search._check
    monkeypatch.setattr(v_search, "_check", lambda idx, **kw: (_ for _ in ()).throw(OSError("io")))
    with pytest.raises(OSError):
        v_search.startup_check()
    monkeypatch.setattr(v_search, "_check", real)
    assert index_db.open_index(store.WIKI_PATH).checked is False


def test_backlinks_without_indexable_words_scan_every_page(seeded_vault, monkeypatch):
    store._write("notes/linker.md", read("projects/falcon.md") + "\n- see [[people/anna]]\n")
    real = v_search._tokenise
    monkeypatch.setattr(v_search, "_tokenise", lambda text: [] if text == "people/anna" else real(text))
    assert v_search.backlinks("people/anna") == ["notes/linker"]


def test_terms_df_equals_the_postings_count(seeded_vault):
    file_faq(ZYX, "Marsh notes", "marsh.md")
    assert rows("SELECT t.corpus, t.term, t.df FROM terms t WHERE t.df != (SELECT COUNT(*) FROM postings p "
                "WHERE p.corpus = t.corpus AND p.term = t.term)") == []
