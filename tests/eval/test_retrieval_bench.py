"""PRD 55 phase 1: the retrieval bench (BM25 vs embeddings vs RRF hybrid) on a tiny vault with a stub embedder.

No network: `FakeEmbedder` replaces the /v1/embeddings call with deterministic bag-of-words vectors (a
synonym table makes "lamp" land next to "lantern"), and one test serves the real HTTP contract from a
local stub server.
"""
import json
import re
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from vault import store
from vault import chunks as v_chunks
from eval import retrieval_bench as rb
from tests.conftest import file_faq


@pytest.fixture(autouse=True)
def small_vault_floor(monkeypatch):
    """Eight chunks give every word a tiny IDF: the BM25 floor is lowered so the verbatim question passes it."""
    monkeypatch.setattr(v_chunks, "RECALL_MIN_SCORE", 0.5)


SYNONYMS = {"lamp": "lantern", "brought": "carried", "stew": "cook"}
DIM = 64


def vector(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for w in re.findall(r"[a-z]+", text.lower()):
        v[zlib.crc32(SYNONYMS.get(w, w).encode()) % DIM] += 1.0
    return v / max(float(np.linalg.norm(v)), 1e-9)


class FakeEmbedder(rb.Embedder):
    def __init__(self, delay: float = 0.0):
        super().__init__("http://127.0.0.1:1")
        self.delay, self.calls = delay, []

    def raw(self, texts: list) -> np.ndarray:
        self.calls.append(len(texts))
        if len(texts) == 1 and self.delay:
            time.sleep(self.delay)
        return np.stack([vector(t) for t in texts])


FILLER = "rope anchor galley harbour timber cargo sailor barrel tide mast".split()


def prose(seed: int, extra: str, paragraphs: int = 14) -> str:
    out = []
    for i in range(paragraphs):
        words = [FILLER[(seed * 7 + i * 3 + j) % len(FILLER)] for j in range(70)]
        out.append(" ".join(words) + (f" {extra}" if i == paragraphs // 2 else "") + ".")
    return "\n\n".join(out)


def tiny_vault(vault):
    file_faq("Harbour notes\n\n" + prose(1, "Oakhaven keeper hid the lantern under the old stairs."), "Harbour notes", "a.md")
    file_faq("Galley notes\n\n" + prose(2, "We cook stew every evening in the galley."), "Galley notes", "b.md")
    return rb.Corpus("tiny", str(vault))


def holding(corpus, needle: str) -> set:
    return {(raw, n) for raw, n, text in corpus.rows if needle in text}


def questions(corpus) -> list:
    lantern = holding(corpus, "hid the lantern")
    return [{"id": "Q1", "group": "battery", "text": "Where did the Oakhaven keeper hide the lantern?", "expect": lantern},
            {"id": "Q2", "group": "paraphrase", "text": "Where did the Oakhaven keeper hide the lamp?", "expect": lantern},
            {"id": "N1", "group": "negative", "text": "what should we cook", "expect": set()},
            {"id": "N2", "group": "negative", "text": "Are you listening, Uncle?", "expect": set()}]


# --------------------------------------------------------------------------- pure pieces

def test_rrf_golden_order_and_score():
    fused = rb.rrf(["a", "b", "c"], ["c", "a", "d"])
    assert [k for k, _s in fused] == ["a", "c", "b", "d"]
    assert fused[0][1] == 1 / 61 + 1 / 62


def hit(raw="r", n=1, matched=(), score=9.0, phrases=(), named=False, cos=None):
    return {"raw": raw, "n": n, "matched": list(matched), "score": score, "phrases": list(phrases), "named": named, "cos": cos}


def test_gate_needs_two_words_or_a_proper_term_and_skips_neighbours_and_low_scores():
    one, two, proper = hit(n=1, matched=["cook"]), hit(n=5, matched=["a", "b"]), hit(n=9, matched=["vivec"])
    assert rb.gate([one], set(), [], 4.5) == []
    assert rb.gate([one, two], set(), [], 4.5) == [two]
    assert rb.gate([proper], {"vivec"}, [], 4.5) == [proper]
    assert rb.gate([hit(n=5, matched=["a", "b"], score=1.0)], set(), [], 4.5) == []        # under the floor
    assert rb.gate([hit(n=5, matched=["a", "b"], score=1.0, named=True)], set(), [], 4.5)  # a named source has none
    assert rb.gate([two, hit(n=6, matched=["a", "b"])], set(), [], 4.5) == [two]           # a neighbour of a kept chunk
    assert rb.gate([one], set(), [], 4.5, tau=0.5) == []                                     # no cosine: no bonus
    assert rb.gate([hit(matched=["cook"], cos=0.7)], set(), [], 4.5, tau=0.5)               # relaxed: cosine counts once


def test_questions_skip_followups_count_empty_expectations_as_negatives_and_dedupe(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"questions": [
        {"id": "A1", "group": "battery", "text": "x", "expected": [{"raw": "r", "part": 2}, {"raw": "r", "part": 2}]},
        {"id": "A2", "group": "battery", "text": "yes", "expected": [{"raw": "r", "part": 2}]},
        {"id": "A9", "group": "battery", "text": "later", "expected": [{"raw": "r", "part": 1}], "note": "follow-up turn"},
        {"id": "A13", "group": "battery", "text": "not covered", "expected": []},
        {"id": "N1", "group": "negative", "text": "hi", "expected": []}]}))
    got = {q["id"]: q for q in rb.load_questions(str(path))}
    assert sorted(got) == ["A1", "A13", "N1"]
    assert got["A1"]["expect"] == {("r", 2)} and got["A1"]["group"] == "battery"
    assert got["A13"]["group"] == "negative"


def test_the_shipped_question_set_loads():
    qs = rb.load_questions()
    assert {q["group"] for q in qs} == {"battery", "paraphrase", "negative"}
    assert not {q["id"] for q in qs} & rb.FOLLOW_UPS
    assert sum(q["group"] == "paraphrase" for q in qs) == 30


# --------------------------------------------------------------------------- embedding cache and HTTP contract

def test_chunk_embeddings_are_cached_by_text(tmp_path):
    emb, base = FakeEmbedder(), str(tmp_path / "cache" / "emb")
    texts = [f"chunk number {i} about {w}" for i, w in enumerate(FILLER)]
    first = rb.embed_chunks(texts, emb, base)
    assert first.shape == (len(texts), DIM) and sum(emb.calls) == len(texts)
    emb.calls.clear()
    rb.embed_chunks(texts, emb, base)
    assert emb.calls == []                                                    # second run: nothing embedded
    rb.embed_chunks(texts + ["one more chunk"], emb, base)
    assert emb.calls == [1]                                                   # only the new text


def test_embedder_speaks_the_openai_contract():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            data = [{"index": i, "embedding": [3.0, 4.0, 0.0]} for i, _t in enumerate(body["input"])]
            payload = json.dumps({"data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        vec = rb.Embedder(f"http://127.0.0.1:{server.server_port}").query("hello")
    finally:
        server.shutdown()
    assert seen[0][0] == "/v1/embeddings" and seen[0][1]["input"] == [rb.QUERY_PREFIX + "hello"]
    assert np.allclose(vec, [0.6, 0.8, 0.0])                                  # L2-normalised client-side


# --------------------------------------------------------------------------- the bench on a tiny vault

def test_table_has_one_row_per_method_and_group_with_every_column_filled(vault):
    corpus = tiny_vault(vault)
    res = rb.run_corpus(corpus, questions(corpus), FakeEmbedder(), str(vault.parent / "cache"))
    table = rb.render([res])
    rows = [line for line in table.splitlines() if line.startswith("| tiny |") and "| bm25 p50" not in line]
    assert len(rows) == len(rb.METHODS) * len(rb.GROUPS) + 1                   # + the latency table's one row
    header = table.splitlines()[0]
    for col in ("hit@1", "hit@2", "violations", "p95 ms"):
        assert col in header
    assert all("None" not in line for line in rows)
    by = {(r["method"], r["group"]): r for r in res["rows"]}
    assert by[("bm25", "battery")]["hit2"] == 1.0                                # the verbatim question is found
    assert by[("hybrid", "battery")]["hit2"] == 1.0
    assert "paraphrase hit@2 gain" in rb.verdict([res])


def test_the_gate_rejects_what_an_ungated_embedding_would_emit(vault):
    corpus = tiny_vault(vault)
    res = rb.run_corpus(corpus, questions(corpus), FakeEmbedder(), str(vault.parent / "cache"))
    by = {(r["method"], r["group"]): r for r in res["rows"]}
    assert by[("embeddings", "negative")]["violations"] >= 1                     # "cook" alone is no match
    assert by[("embeddings", "negative")]["negatives_hit"] == 2                  # a cosine top-k always emits
    for method in ("bm25", "embeddings-gated", "hybrid", "hybrid-nofloor"):
        assert by[(method, "negative")]["violations"] == 0
    assert by[("hybrid", "negative")]["negatives_hit"] == 0


def test_a_dead_embed_server_leaves_the_bm25_rows(vault):
    corpus = tiny_vault(vault)
    res = rb.run_corpus(corpus, questions(corpus), rb.Embedder("http://127.0.0.1:1"), str(vault.parent / "cache"))
    assert res["available"] is False
    assert {r["method"] for r in res["rows"]} == {"bm25"}
    assert "embeddings unavailable" in rb.render([res])


def test_latency_includes_the_embed_call_and_times_the_dot_separately(vault):
    corpus = tiny_vault(vault)
    res = rb.run_corpus(corpus, questions(corpus), FakeEmbedder(delay=0.01), str(vault.parent / "cache"))
    lat = res["latency"]
    assert lat["serial_p50"] >= 10 and lat["serial_p95"] >= lat["serial_p50"]
    assert lat["dot_p50"] < 10 and lat["parallel_p95"] >= lat["parallel_p50"] >= 0
    assert store.WIKI_PATH == corpus.path
