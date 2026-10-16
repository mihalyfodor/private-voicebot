"""PRD 53 scenario 10: the index perf gates on the generated vault (marker `perf`, minutes).

    scripts/check.sh -m perf tests/perf
"""
import statistics
import time

import pytest

import index_db
import wiki
from tests.conftest import fake_model
from tests.perf.make_vault import make_vault, queries
import sources

pytestmark = pytest.mark.perf


def p95(samples: list) -> float:
    return statistics.quantiles(samples, n=20)[-1]


def timed(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t0) * 1000


def test_search_stays_fast_on_5k_pages_and_1k_sources(tmp_path, monkeypatch):
    vault = tmp_path / "perf-vault"
    made = make_vault(str(vault))
    monkeypatch.setattr(wiki, "WIKI_PATH", str(vault))
    index_db.close()
    wiki.startup_check()
    size_mb = round(sum(p.stat().st_size for p in (vault / ".index").iterdir()) / 1e6, 1)

    asks = queries(200)
    wiki.search(asks[0]), wiki.search_chunks(asks[0])          # warm the per-generation caches
    chunk_ms = [timed(wiki.search_chunks, q, 3) for q in asks]
    page_ms = [timed(wiki.search, q, 5) for q in asks]
    hits = sum(bool(wiki.search_chunks(q, 3)) for q in asks[:20])

    spent = []
    real = wiki.index_source
    monkeypatch.setattr(wiki, "index_source", lambda record: spent.append(timed(real, record)))
    text = "Extra handbook\n\n" + " ".join(queries(400, seed=7))
    with fake_model('{"title": "Extra handbook", "summary": "x", "entities": [], "topic": "t", "unreadable": false}', "[]"):
        sources.ingest_upload(text.encode(), "extra.md")

    print(f"\nPERF vault {made} index {size_mb} MB | p95 search_chunks {p95(chunk_ms):.1f} ms "
          f"(max {max(chunk_ms):.1f}) | p95 search {p95(page_ms):.1f} ms (max {max(page_ms):.1f}) | "
          f"one-source update {spent[0]:.1f} ms")
    assert made["pages"] >= 5000 and made["sources"] == 1000 and made["chunks"] >= 14000
    assert hits >= 10
    assert p95(chunk_ms) < 50
    assert p95(page_ms) < 20
    assert spent and spent[0] < 200


@pytest.mark.live
def test_the_battery_totals_equal_the_prd_43_run():
    pytest.skip("needs the live model: run tests/qa/stress.py against oMLX and compare correct/partial/wrong "
                "with the PRD 43 run by hand (docs/53-scalable-index.md scenario 10)")
