"""PRD 54 — the vault map (docs/54-vault-map-research.md, scenarios 1, 2 and 4)."""
from datetime import date

from vault import index_db
from core import util
from vault import index as v_index
from vault import pages
from vault import search
from vault import store
from tests.perf.make_vault import make_vault

TODAY = date(2026, 10, 16)


def page(rel: str, name: str, kind: str, summary: str, updated: str, tags=(), links=()) -> None:
    """One page written through the app's write path (`_write` runs the index hook)."""
    fm = {"type": kind, "name": name, "tags": list(tags), "source": "seed", "updated": updated, "seen": updated}
    related = "".join(f"- [[{t}]]\n" for t in links)
    body = f"# {name}\n\n## Summary\n{summary}\n\n## Notes\n\n## Related\n{related}\n## History\n"
    store._write(rel + ".md", store._page_text(fm, body))


def twelve_pages() -> None:
    """3 folders: people (5), projects (4), meetings (3); links and dates decide the ranking."""
    page("people/anna", "Anna", "person", "sister, lives in Lyon", "2026-10-01", ["family"], ["projects/falcon"])
    page("people/bob", "Bob", "person", "climbing partner", "2026-10-15", ["climbing"], ["people/anna"])
    page("people/cara", "Cara", "person", "manager", "2026-09-01", ["work"], ["people/anna", "projects/falcon"])
    page("people/dan", "Dan", "person", "neighbour", "2026-10-16", ["family"])
    page("people/eve", "Eve", "person", "old friend", "2025-01-01", ["family", "work"])
    page("projects/falcon", "Falcon", "project", "load-test project", "2026-08-01", ["work"], ["people/cara"])
    page("projects/atlas", "Atlas", "project", "paused migration", "2026-10-10", ["work"])
    page("projects/zephyr", "Zephyr", "project", "budget review", "2026-10-12", ["work", "money"])
    page("projects/beta", "Beta", "project", "", "2026-10-12")
    page("meetings/2026-10-01", "Standup", "meeting", "weekly sync", "2026-10-01", [], ["projects/zephyr"])
    page("meetings/2026-10-08", "Standup", "meeting", "weekly sync", "2026-10-08")
    page("meetings/2026-10-15", "Retro", "meeting", "quarter retro", "2026-10-15", ["work"])


GOLDEN = """# Vault map
## people (5) — tags: family, work, climbing
- [[people/anna]] — Anna, person — sister, lives in Lyon
- [[people/cara]] — Cara, person — manager
+3 more
## projects (4) — tags: work, money
- [[projects/falcon]] — Falcon, project — load-test project
- [[projects/zephyr]] — Zephyr, project — budget review
+2 more
## meetings (3) — tags: work
- [[meetings/2026-10-15]] — Retro, meeting — quarter retro
- [[meetings/2026-10-08]] — Standup, meeting — weekly sync
+1 more"""


# --------------------------------------------------------------- 1. golden


def test_a_twelve_page_vault_maps_to_the_golden(vault):
    twelve_pages()
    # anna (2 inbound) > cara (1) > dan, bob (recency); falcon (2) > zephyr (1) > beta (10-12) > atlas (10-10);
    # at 124 tokens two rounds fit and people's third page (round three) does not
    assert v_index.vault_map(124, today=TODAY) == GOLDEN


def test_round_robin_gives_every_folder_its_header_and_best_page_first(vault):
    twelve_pages()
    small = v_index.vault_map(100, today=TODAY)
    assert "## people (5)" in small and "## projects (4)" in small and "## meetings (3)" in small
    assert "[[people/anna]]" in small and "[[projects/falcon]]" in small and "[[meetings/2026-10-15]]" in small
    assert util.tokens_est(small) <= 100
    assert small.splitlines()[-1].startswith("+")              # the last folder ends with "+n more"


def test_an_empty_vault_has_no_map(vault):
    assert v_index.vault_map() == ""


# --------------------------------------------------------------- 2. 320 pages: the cut is real


def test_the_map_names_zephyr_and_every_folder_where_the_alphabetical_cut_does_not(vault):
    path = store.WIKI_PATH + "-generated"
    made = make_vault(path, pages=320, sources_n=0, seed=3)
    assert made["pages"] == 320
    saved, store.WIKI_PATH = store.WIKI_PATH, path
    try:
        page("projects/zephyr", "Zephyr", "project", "the Zephyr budget", "2026-10-16", ["money"])
        for i in range(5):                                    # five pages link it
            rel = f"notes/p{4 * i + 3:05d}"
            store._write(rel + ".md", store._read(rel + ".md") + "\n- see [[projects/zephyr]]\n")
        got = v_index.vault_map(600)
        assert util.tokens_est(got) <= 600
        assert "[[projects/zephyr]]" in got
        for folder in ("people", "projects", "meetings", "notes"):
            assert f"## {folder} (" in got
        cut = v_index.index_text(600)
        assert "projects/zephyr" not in cut                   # z sorts last: the alphabetical cut hides it
    finally:
        store.WIKI_PATH = saved
        index_db.close()


# --------------------------------------------------------------- 4. incremental rows


def test_writing_one_page_rereads_only_its_row_and_the_map_follows(vault, monkeypatch):
    twelve_pages()
    assert "— budget review" in v_index.vault_map(600, today=TODAY)
    puts, reads = [], []
    real_put, real_read = index_db.Index.put_page, store._read
    monkeypatch.setattr(index_db.Index, "put_page", lambda self, rel, *a: (puts.append(rel), real_put(self, rel, *a)))
    monkeypatch.setattr(store, "_read", lambda rel: (reads.append(rel), real_read(rel))[1])

    page("projects/zephyr", "Zephyr", "project", "budget approved", "2026-10-16", ["work", "money"])
    got = v_index.vault_map(600, today=TODAY)

    assert puts == ["projects/zephyr.md"]                     # one row rebuilt
    assert reads == []                                        # the map reads no page
    assert "- [[projects/zephyr]] — Zephyr, project — budget approved" in got
    assert "budget review" not in got


def test_deleting_a_page_drops_its_row_and_the_folder_count(vault):
    twelve_pages()
    assert "## meetings (3)" in v_index.vault_map(600, today=TODAY)
    path, reason = pages.delete_page("meetings/2026-10-08")
    assert reason is None
    got = v_index.vault_map(600, today=TODAY)
    assert "## meetings (2)" in got and "[[meetings/2026-10-08]]" not in got


# --------------------------------------------------------------- fix round 1


def test_a_long_one_liner_is_cut_in_the_map_row(vault):
    twelve_pages()
    page("projects/atlas", "Atlas", "project", "paused migration " * 20, "2026-10-10", ["work"])
    row = next(line for line in v_index.vault_map(600, today=TODAY).splitlines() if "[[projects/atlas]]" in line)
    assert len(row) <= len("- [[projects/atlas]] — ") + store.MAP_ONE_LINER_CHARS and row.endswith("…")


def test_a_version_1_index_file_is_rebuilt_once_with_links(vault, monkeypatch):
    import sqlite3

    twelve_pages()
    index_db.close()
    path = index_db.db_path(store.WIKI_PATH)
    conn = sqlite3.connect(path)                  # make it a schema-1 file: no links column
    conn.executescript("DROP TABLE pages; CREATE TABLE pages(path TEXT PRIMARY KEY, mtime INTEGER, sha TEXT, "
                       "type TEXT, name TEXT, tags TEXT, updated TEXT, one_liner TEXT, length INTEGER);"
                       "UPDATE meta SET value='1' WHERE key='schema';")
    conn.commit()
    conn.close()
    puts = []
    real = index_db.Index.put_page
    monkeypatch.setattr(index_db.Index, "put_page", lambda self, rel, *a: (puts.append(rel), real(self, rel, *a)))
    assert "+3 more" in v_index.vault_map(124, today=TODAY) and v_index.vault_map(124, today=TODAY) == GOLDEN
    assert len(puts) == 12
    cols = [r[1] for r in search._db().conn.execute("PRAGMA table_info(pages)")]
    assert "links" in cols
    index_db.close()
    puts.clear()
    assert v_index.vault_map(124, today=TODAY) == GOLDEN and puts == []      # reopened: nothing re-indexed


def test_three_typed_turns_never_build_the_map(vault, monkeypatch, fresh_llm):
    from tests.conftest import typed

    twelve_pages()
    calls = []
    real = v_index.vault_map
    monkeypatch.setattr(v_index, "vault_map", lambda *a, **kw: (calls.append(1), real(*a, **kw))[1])
    for text in ("hello there", "how are you", "tell me a joke"):
        typed(text)
    assert calls == []
