"""PRD 55 phase 1: BM25 vs embeddings vs RRF hybrid on one labelled question set, as a markdown table.

    .venv/bin/python -m eval.retrieval_bench [--dev-vault ~/llm-wiki-dev] [--synthetic-sources 400]

Offline measurement, never imported by the app. BM25 is today's `wiki.search_chunks` plus the
`llm._recall_hits` precision rules (two matched topic words or `wiki.proper_match`); embeddings come
from the llama-server on :8082 (`POST /v1/embeddings`), cosine top-k with numpy; hybrid is reciprocal
rank fusion (k=60) of the BM25 and embedding top-20, then the same gate. The vaults are measured as
copies under --workdir (the dev vault is only read); the embedding matrix is cached there too.
"""
import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("WIKI_PATH", os.path.join(tempfile.gettempdir(), f"retrieval-bench-{os.getpid()}", "no-vault"))

import numpy as np          # noqa: E402
import requests             # noqa: E402

from vault import index_db
from vault import chunks as v_chunks
from vault import search
from vault import store

QUESTIONS = os.path.join(ROOT, "eval", "retrieval_questions.json")
EMBED_URL = "http://127.0.0.1:8082"
MODEL = "qwen3-embedding-0.6b-q8"
QUERY_PREFIX = "Instruct: Given a question, retrieve passages that answer it\nQuery: "
RRF_K = 60
TOP = 20                    # each ranking's depth into the fusion
CANDIDATES = 4              # what `_recall_hits` is asked for (llm.RECALL_CANDIDATES) before its gate
MAX_KEPT = 2                # llm.RECALL_MAX_CHUNKS
BATCH = 32
MAX_CHARS = 6000            # an embedding input is cut here (chunks are ~3 200 chars)
TAU = 0.55                  # relaxed hybrid: cosine >= TAU counts as one matched word (information only)
GROUPS = ("battery", "paraphrase", "negative")
METHODS = ("bm25", "embeddings", "embeddings-gated", "hybrid", "hybrid-nofloor", "hybrid-relaxed")
FOLLOW_UPS = {"A2", "A6", "C4"}


class EmbedDown(Exception):
    """The embed server did not answer."""


# --------------------------------------------------------------------------- question set

def load_questions(path: str = QUESTIONS) -> list:
    """``[{id, group, text, expect: {(raw, part)}}]``. Follow-ups are dropped; a question with nothing
    expected (the negatives, A13/A14) is group "negative": no hit may be kept for it."""
    out = []
    for q in json.load(open(path))["questions"]:
        if q["id"] in FOLLOW_UPS or "follow-up" in q.get("note", ""):
            continue
        expect = {(e["raw"], int(e["part"])) for e in q.get("expected", [])}
        out.append({"id": q["id"], "group": q["group"] if expect else "negative", "text": q["text"], "expect": expect})
    return out


# --------------------------------------------------------------------------- embedding

class Embedder:
    def __init__(self, url: str = EMBED_URL):
        self.url = url.rstrip("/")

    def raw(self, texts: list) -> np.ndarray:
        try:
            r = requests.post(f"{self.url}/v1/embeddings", json={"input": [t[:MAX_CHARS] for t in texts]}, timeout=120)
            r.raise_for_status()
            rows = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise EmbedDown(str(exc)) from exc
        m = np.asarray([d["embedding"] for d in rows], dtype=np.float32)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    def query(self, text: str) -> np.ndarray:
        return self.raw([QUERY_PREFIX + text])[0]


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def embed_chunks(texts: list, embedder: Embedder, cache_base: str, log=lambda s: None) -> np.ndarray:
    """float16 [len(texts), dim], rows aligned with `texts`; vectors cached by text hash in
    `<cache_base>.npy` + `.json`, so a re-run embeds only what is new."""
    have = {}
    if os.path.exists(cache_base + ".json") and os.path.exists(cache_base + ".npy"):
        meta = json.load(open(cache_base + ".json"))
        if meta.get("model") == MODEL:
            have = dict(zip(meta["rows"], np.load(cache_base + ".npy")))
    keys = [_key(t) for t in texts]
    todo = sorted({k: i for i, k in enumerate(keys) if k not in have}.values())
    for start in range(0, len(todo), BATCH):
        part = todo[start:start + BATCH]
        for i, vec in zip(part, embedder.raw([texts[i] for i in part])):
            have[keys[i]] = vec.astype(np.float16)
        if (start // BATCH) % 10 == 0:
            log(f"embedded {start + len(part)}/{len(todo)}")
    if todo:
        rows = list(have)
        os.makedirs(os.path.dirname(cache_base), exist_ok=True)
        np.save(cache_base + ".npy", np.stack([have[k] for k in rows]))
        json.dump({"model": MODEL, "rows": rows}, open(cache_base + ".json", "w"))
    return np.stack([have[k] for k in keys])


# --------------------------------------------------------------------------- corpus

class Corpus:
    """A vault copy opened through `wiki`, its chunks listed once: `rows[i] = (raw, n, text)`."""

    def __init__(self, name: str, path: str):
        self.name, self.path = name, path
        store.WIKI_PATH = path
        index_db.close()
        idx = v_chunks._chunk_index()
        meta = idx.chunk_meta() if idx else {}
        self.rows = []
        for source, raw, _title, n, _total, sha in sorted(meta.values(), key=lambda m: (m[1], m[3])):
            record = v_chunks._record(sha) or v_chunks._raw_chunks_for({"sha256": sha, "path": raw, "page": source})
            self.rows.append((raw, n, record["chunks"][n - 1]["text"]))
        self.at = {(raw, n): i for i, (raw, n, _t) in enumerate(self.rows)}
        self.matrix = None

    def activate(self) -> None:
        store.WIKI_PATH = self.path


# --------------------------------------------------------------------------- ranking, fusion, gate

def rrf(*rankings: list, k: int = RRF_K) -> list:
    """``[(key, score)]``, best first: score(d) = sum over rankings of 1 / (k + rank), rank from 1.
    Ties keep the order of first appearance."""
    score, first = {}, {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            score[key] = score.get(key, 0.0) + 1.0 / (k + rank)
            first.setdefault(key, len(first))
    return sorted(score.items(), key=lambda kv: (-kv[1], first[kv[0]]))


def passes(cand: dict, proper: set, pairs: list, bonus: int = 0) -> bool:
    """The PRD 34 precision rule: two matched topic words (`bonus` extra for the relaxed hybrid) or a proper
    term whose bigram the chunk holds whole."""
    return len(cand["matched"]) + bonus >= 2 or bool(v_chunks.proper_match(cand, proper, pairs))


def gate(cands: list, proper: set, pairs: list, floor: float, tau: float | None = None) -> list:
    """`llm._recall_hits`' rules over ranked candidates: up to MAX_KEPT kept, in order, a BM25 score under
    `floor` rejected (unless the source is named; a candidate without a BM25 score has none), never a
    neighbour of a kept chunk, and the precision rule. `tau`: cosine >= tau counts as one matched word."""
    kept = []
    for c in cands:
        if len(kept) >= MAX_KEPT:
            break
        if c["score"] is not None and c["score"] < floor and not c["named"]:
            continue
        if any(h["raw"] == c["raw"] and abs(h["n"] - c["n"]) == 1 for h in kept):
            continue
        bonus = 1 if tau is not None and c["cos"] is not None and c["cos"] >= tau else 0
        if passes(c, proper, pairs, bonus):
            kept.append(c)
    return kept


class Query:
    """One question's evidence: its terms, BM25 hits and embedding ranking."""

    def __init__(self, corpus: Corpus, text: str):
        self.text = text
        terms = v_chunks.content_terms(text)
        self.proper = v_chunks.proper_terms(terms)
        self.pairs = v_chunks.proper_phrases(text)
        topic, _named = v_chunks.split_title_terms(terms)
        self.topic = (topic - v_chunks.GENERIC_WORDS) or topic
        self.corpus = corpus
        self.bm25: list = []
        self.cos = None

    def describe(self, raw: str, n: int, hit=None) -> dict:
        """The candidate dict for chunk (raw, n); a BM25 hit brings its own score/matched/phrases/named."""
        cos = None if self.cos is None else float(self.cos[self.corpus.at[(raw, n)]])
        if hit is not None:
            return {"raw": raw, "n": n, "score": hit["score"], "matched": hit["matched"], "phrases": hit["phrases"],
                    "named": hit.get("named", False), "cos": cos}
        tokens = search._tokenise(self.corpus.rows[self.corpus.at[(raw, n)]][2])
        adjacent = set(zip(tokens, tokens[1:]))
        return {"raw": raw, "n": n, "score": None, "matched": sorted(self.topic & set(tokens)),
                "phrases": [p for p in self.pairs if tuple(p) in adjacent], "named": False, "cos": cos}

    def _by_bm25(self) -> dict:
        return {(h["raw"], h["n"]): h for h in self.bm25}

    def _top_cos(self, depth: int) -> list:
        return [self.corpus.rows[int(i)][:2] for i in np.argsort(-self.cos)[:depth]]

    def bm25_candidates(self, depth: int) -> list:
        return [self.describe(h["raw"], h["n"], h) for h in self.bm25[:depth]]

    def embedding_candidates(self, depth: int) -> list:
        by = self._by_bm25()
        return [self.describe(raw, n, by.get((raw, n))) for raw, n in self._top_cos(depth)]

    def hybrid_candidates(self, depth: int) -> list:
        by = self._by_bm25()
        fused = rrf([(h["raw"], h["n"]) for h in self.bm25[:TOP]], self._top_cos(TOP))
        return [self.describe(raw, n, by.get((raw, n))) for (raw, n), _s in fused[:depth]]


def emitted(method: str, q: Query, floor: float, tau: float = TAU) -> list:
    """The hits `method` would put in front of the model for `q`."""
    if method == "bm25":
        return gate(q.bm25_candidates(CANDIDATES), q.proper, q.pairs, floor)
    if method == "embeddings":
        return q.embedding_candidates(MAX_KEPT)                 # no gate: the raw cosine top-k
    if method == "embeddings-gated":
        return gate(q.embedding_candidates(CANDIDATES), q.proper, q.pairs, floor)
    if method == "hybrid":
        return gate(q.hybrid_candidates(CANDIDATES), q.proper, q.pairs, floor)
    if method == "hybrid-nofloor":                              # information: the BM25 score floor dropped
        return gate(q.hybrid_candidates(CANDIDATES), q.proper, q.pairs, 0.0)
    if method == "hybrid-relaxed":
        return gate(q.hybrid_candidates(CANDIDATES), q.proper, q.pairs, floor, tau)
    raise ValueError(method)


# --------------------------------------------------------------------------- measurement

def pct(samples: list, p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


def _timed_ms(fn, *args):
    t = time.perf_counter()
    out = fn(*args)
    return out, (time.perf_counter() - t) * 1000


def run_corpus(corpus: Corpus, questions: list, embedder: Embedder, cache_dir: str, log=lambda s: None) -> dict:
    """Run every question the five ways on `corpus`; returns ``{corpus, rows, latency, available}``."""
    corpus.activate()
    floor = v_chunks.RECALL_MIN_SCORE
    available = True
    try:
        corpus.matrix = embed_chunks([t for _r, _n, t in corpus.rows], embedder,
                                     os.path.join(cache_dir, f"emb-{MODEL}-{corpus.name}"), log).astype(np.float32)
    except EmbedDown:
        available = False
    asks, serial, parallel, dots, bm25_ms = [], [], [], [], []
    pool = ThreadPoolExecutor(max_workers=1)
    for spec in questions:
        q = Query(corpus, spec["text"])
        v_chunks.search_chunks(spec["text"], k=TOP)                 # warm, then the timed call
        q.bm25, solo = _timed_ms(v_chunks.search_chunks, spec["text"], TOP)
        bm25_ms.append(solo)
        if available:
            try:
                vec, embed_ms = _timed_ms(embedder.query, spec["text"])       # serial: BM25, then embed, then rank
                q.cos, dot_ms = _timed_ms(lambda v: corpus.matrix @ v, vec)
                _, fuse_ms = _timed_ms(lambda: rrf([(h["raw"], h["n"]) for h in q.bm25[:TOP]], q._top_cos(TOP)))
                serial.append(embed_ms + dot_ms + fuse_ms)
                dots.append(dot_ms + fuse_ms)
                t = time.perf_counter()                         # parallel: the embed runs alongside BM25
                fut = pool.submit(embedder.query, spec["text"])
                v_chunks.search_chunks(spec["text"], k=TOP)
                fut.result()
                parallel.append(max(0.0, (time.perf_counter() - t) * 1000 + dot_ms + fuse_ms - solo))
            except EmbedDown:
                available = False
        asks.append((spec, q))
    pool.shutdown()
    methods = METHODS if available else ("bm25",)
    outs = {m: {spec["id"]: emitted(m, q, floor) for spec, q in asks} for m in methods}
    qs = {spec["id"]: q for spec, q in asks}
    rows = []
    for method in methods:
        added = [0.0] if method == "bm25" else serial
        for group in GROUPS:
            specs = [s for s, _q in asks if s["group"] == group]
            if not specs:
                continue
            hits = [outs[method][s["id"]] for s in specs]
            row = {"corpus": corpus.name, "method": method, "group": group, "n": len(specs)}
            if group != "negative":
                for k in (1, 2):
                    row[f"hit{k}"] = sum(any((c["raw"], c["n"]) in s["expect"] for c in h[:k])
                                         for s, h in zip(specs, hits)) / len(specs)
            row["violations"] = sum(any(not passes(c, qs[s["id"]].proper, qs[s["id"]].pairs) for c in h)
                                    for s, h in zip(specs, hits))
            row["negatives_hit"] = sum(bool(h) for h in hits) if group == "negative" else None
            row["negatives_hit_ids"] = sorted(s["id"] for s, h in zip(specs, hits) if h) if group == "negative" else []
            row["p50_ms"], row["p95_ms"] = pct(added, 0.5), pct(added, 0.95)
            rows.append(row)
    latency = {"bm25_p50": pct(bm25_ms, .5), "bm25_p95": pct(bm25_ms, .95),
               "serial_p50": pct(serial, .5), "serial_p95": pct(serial, .95),
               "parallel_p50": pct(parallel, .5), "parallel_p95": pct(parallel, .95),
               "dot_p50": pct(dots, .5), "dot_p95": pct(dots, .95), "chunks": len(corpus.rows)}
    return {"corpus": corpus.name, "rows": rows, "latency": latency, "available": available}


def dot_at_scale(rows: int = 65000, dim: int = 1024) -> float:
    """Median ms of one float32 matrix-vector dot + top-k at `rows` x `dim` (the 10x-dev-vault case)."""
    m = np.random.default_rng(0).standard_normal((rows, dim), dtype=np.float32)
    v = m[0].copy()
    times = []
    for _ in range(9):
        t = time.perf_counter()
        np.argpartition(-(m @ v), TOP)[:TOP]
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


# --------------------------------------------------------------------------- report

def _f(x, as_pct=False):
    if x is None:
        return "-"
    return f"{x * 100:.0f}%" if as_pct else (f"{x:.1f}" if isinstance(x, float) else str(x))


def render(results: list) -> str:
    """The markdown report: one row per corpus x method x group, then the latency table."""
    out = ["| corpus | method | group | n | hit@1 | hit@2 | violations | negatives hit | p50 ms | p95 ms |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for res in results:
        for r in res["rows"]:
            out.append(f"| {r['corpus']} | {r['method']} | {r['group']} | {r['n']} | {_f(r.get('hit1'), True)} | "
                       f"{_f(r.get('hit2'), True)} | {r['violations']} | {_f(r['negatives_hit'])} | "
                       f"{_f(r['p50_ms'])} | {_f(r['p95_ms'])} |")
    for res in results:
        if not res["available"]:
            out.append(f"\n{res['corpus']}: embeddings unavailable (embed server did not answer); BM25 rows only.")
    out += ["", "Added latency over BM25 (ms): serial = embed then dot + fuse, parallel = embed alongside BM25.", "",
            "| corpus | chunks | bm25 p50 | bm25 p95 | serial p50 | serial p95 | parallel p50 | parallel p95 | dot+fuse p50 |",
            "|---|---|---|---|---|---|---|---|---|"]
    for res in results:
        lat = res["latency"]
        out.append(f"| {res['corpus']} | {lat['chunks']} | {lat['bm25_p50']:.1f} | {lat['bm25_p95']:.1f} | "
                   f"{lat['serial_p50']:.1f} | {lat['serial_p95']:.1f} | {lat['parallel_p50']:.1f} | "
                   f"{lat['parallel_p95']:.1f} | {lat['dot_p50']:.2f} |")
    return "\n".join(out)


def verdict(results: list) -> str:
    """The PRD 55 adoption bar, judged on the gated hybrid, per corpus."""
    lines = []
    for res in results:
        if not res["available"]:
            continue
        by = {(r["method"], r["group"]): r for r in res["rows"]}
        gain = (by[("hybrid", "paraphrase")]["hit2"] - by[("bm25", "paraphrase")]["hit2"]) * 100
        new = sorted(set(by[("hybrid", "negative")]["negatives_hit_ids"]) - set(by[("bm25", "negative")]["negatives_hit_ids"]))
        lat = res["latency"]
        lines.append(f"- {res['corpus']}: paraphrase hit@2 gain {gain:+.0f} pts (bar >= +15: {'ok' if gain >= 15 else 'MISS'}); "
                     f"new negatives with a kept hit {len(new)} {new} (bar 0: {'ok' if not new else 'MISS'}); "
                     f"added p95 serial {lat['serial_p95']:.1f} / parallel {lat['parallel_p95']:.1f} ms "
                     f"(bar <= 30: {'ok' if lat['serial_p95'] <= 30 else 'serial MISS'})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- vaults

def copy_dev_vault(src: str, workdir: str) -> str:
    """Read-only copy of the dev vault (no .git) under `workdir`; reused when already there."""
    dest = os.path.join(workdir, "dev-vault")
    if not os.path.isdir(dest):
        shutil.copytree(os.path.expanduser(src), dest, ignore=shutil.ignore_patterns(".git"))
    return dest


def make_synthetic(workdir: str, sources: int) -> tuple:
    """``(vault path, facts)``: PRD 53's generator with --prose, in a sandbox env, reused when present."""
    vault, facts_file = os.path.join(workdir, "synthetic-vault"), os.path.join(workdir, "synthetic-facts.json")
    if not os.path.isdir(vault):
        sandbox = os.path.join(workdir, "synthetic-sandbox")
        env = {**os.environ, "TRACE_DIR": sandbox, "SETTINGS_PATH": os.path.join(sandbox, "settings.json"),
               "MEMORY_DIR": os.path.join(sandbox, "memory"), "GMAIL_DIR": sandbox}
        os.makedirs(sandbox, exist_ok=True)
        done = subprocess.run([sys.executable, "-m", "tests.perf.make_vault", vault, "--prose", "--pages", "200",
                               "--sources", str(sources)], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        json.dump(json.loads(done.stdout.strip().splitlines()[-1])["facts"], open(facts_file, "w"))
    return vault, json.load(open(facts_file))


def synthetic_questions(corpus: Corpus, facts: list) -> list:
    """The tiny labelled set from what the generator wrote: each planted fact asked verbatim and reworded
    (expected = the chunks holding the planted sentence), plus chit-chat naming a planted name."""
    out = []
    for i, f in enumerate(facts):
        expect = {(raw, n) for raw, n, text in corpus.rows if f["plant"] in text}
        out.append({"id": f"S{i + 1}", "group": "battery", "text": f["question"], "expect": expect})
        out.append({"id": f"SP{i + 1}", "group": "paraphrase", "text": f["paraphrase"], "expect": expect})
    for i, f in enumerate(facts[:4]):
        out.append({"id": f"SN{i + 1}", "group": "negative", "text": f"{f['names'][0]} is so annoying", "expect": set()})
    out += [{"id": "SN5", "group": "negative", "text": "what should we cook", "expect": set()},
            {"id": "SN6", "group": "negative", "text": "How long should I steep green tea?", "expect": set()}]
    return [q for q in out if q["expect"] or q["group"] == "negative"]


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-vault", default="~/llm-wiki-dev")
    ap.add_argument("--synthetic-sources", type=int, default=400, help="0 skips the synthetic vault")
    ap.add_argument("--workdir", default=os.path.join(tempfile.gettempdir(), "retrieval-bench"))
    ap.add_argument("--embed-url", default=EMBED_URL)
    ap.add_argument("--questions", default=QUESTIONS)
    ap.add_argument("--skip-dev", action="store_true")
    ap.add_argument("--scale-dot", action="store_true", help="also time a 65 000 x 1024 dot (10x the dev vault)")
    args = ap.parse_args(argv)
    os.makedirs(args.workdir, exist_ok=True)
    embedder = Embedder(args.embed_url)

    def log(s):
        print(s, file=sys.stderr, flush=True)

    results = []
    if not args.skip_dev:
        corpus = Corpus("dev", copy_dev_vault(args.dev_vault, args.workdir))
        results.append(run_corpus(corpus, load_questions(args.questions), embedder, args.workdir, log))
    if args.synthetic_sources:
        vault, facts = make_synthetic(args.workdir, args.synthetic_sources)
        corpus = Corpus("synthetic", vault)
        results.append(run_corpus(corpus, synthetic_questions(corpus, facts), embedder, args.workdir, log))
    print(render(results))
    print(f"\ntau (relaxed hybrid) = {TAU}; candidates into the gate = {CANDIDATES}; RRF k = {RRF_K}, depth {TOP}.")
    if args.scale_dot:
        print(f"dot + top-k at 65 000 x 1024 (float32): {dot_at_scale():.1f} ms median")
    print("\nAdoption bar (gated hybrid):\n" + verdict(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
