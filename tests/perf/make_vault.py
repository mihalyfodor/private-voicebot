"""The shared synthetic-vault generator (docs/53-scalable-index.md scenario 10; PRDs 54 and 55 reuse it).

`make_vault(path)` writes a seeded vault: `pages` knowledge pages through `wiki._write` (the page
write hook) and `sources` text sources through the real filing path (`sources.ingest_upload`) with
the model faked, ~15 chunks each. Same seed, same vault. Git is switched off while it runs: the
history of a generated vault is not what is measured, and 1k commits would dominate the time.

Call it with the app's data paths sandboxed (the conftest autouse fixtures do that for a test):
filing reads the profile and writes trace events like a real drop.
"""
import argparse
import contextlib
import json
import os
import random
import time
from unittest.mock import patch

import sources
import wiki
from tests.conftest import fake_model

SYLLABLES = ("ka", "lo", "mi", "ner", "sto", "pha", "dun", "rel", "vi", "gor", "tal", "shi", "bra", "en",
             "ox", "qui", "zar", "mel", "tor", "ian", "ul", "dre", "fen", "wa", "sel", "cor", "ith", "um")
FOLDERS = ("people", "projects", "meetings", "notes")
TYPES = {"people": "person", "projects": "project", "meetings": "meeting", "notes": "note"}
CHUNK_CHARS = 2850          # text per chunk: ~15 chunks of `wiki.CHUNK_TOKENS` (with overlap) per source


def vocabulary(seed: int = 0, size: int = 20000) -> list:
    """`size` distinct lowercase pseudo-words, deterministic for `seed`; earlier = more frequent."""
    rng = random.Random(seed)
    words, seen = [], set()
    while len(words) < size:
        word = "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4)))
        if word not in seen:
            seen.add(word)
            words.append(word)
    return words


def _zipf(rng: random.Random, vocab: list, cum: list) -> str:
    return vocab[_bisect(cum, rng.random() * cum[-1])]


def _bisect(cum: list, x: float) -> int:
    lo, hi = 0, len(cum)
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return min(lo, len(cum) - 1)


def _weights(vocab: list) -> list:
    cum, total = [], 0.0
    for rank in range(len(vocab)):
        total += 1.0 / (rank + 1)
        cum.append(total)
    return cum


def _text(rng: random.Random, vocab: list, cum: list, names: list, chars: int) -> str:
    """Paragraphs of Zipf words with the `names` dropped in Title-case mid-sentence."""
    paragraphs, size = [], 0
    while size < chars:
        words = []
        for _ in range(rng.randint(60, 110)):
            words.append(rng.choice(names) if names and rng.random() < 0.05 else _zipf(rng, vocab, cum))
        sentence = " ".join(words)
        paragraphs.append(sentence[:1].upper() + sentence[1:] + ".")
        size += len(paragraphs[-1]) + 2
    return "\n\n".join(paragraphs) + "\n"


# --prose (PRD 55): readable sentences about invented names, no FAQ keywords. FILLER_* never overlap FACT_*.
FILLER_VERBS = ("watched", "followed", "counted", "weighed", "visited", "praised", "doubted", "gathered", "avoided",
                "described", "remembered", "inspected", "greeted", "ignored", "ferried", "sketched")
FILLER_NOUNS = ("meadow", "harbour", "orchard", "workshop", "cellar", "village", "market", "chimney", "pasture",
                "corridor", "festival", "quarry", "lighthouse", "granary", "courtyard", "stairwell")
FILLER_ADJS = ("quiet", "narrow", "ancient", "crowded", "distant", "narrow", "modest", "windy", "forgotten")
FILLER_PREPS = ("near", "beyond", "behind", "across", "inside", "along")
FACT_VERBS = (("carried", "brought"), ("hid", "concealed"), ("measured", "gauged"), ("repaired", "mended"),
              ("sold", "traded"), ("buried", "interred"), ("painted", "coloured"))
FACT_NOUNS = (("lantern", "lamp"), ("ledger", "register"), ("barrel", "cask"), ("tapestry", "wall hanging"),
              ("anvil", "forge block"), ("telescope", "spyglass"), ("violin", "fiddle"))


def _prose(rng: random.Random, names: list, chars: int) -> str:
    """Paragraphs of 4-7 template sentences over invented `names` (Title-case, mid-sentence)."""
    paragraphs, size = [], 0
    while size < chars:
        sentences = []
        for _ in range(rng.randint(4, 7)):
            a, b = rng.choice(names), rng.choice(names)
            sentences.append(f"{a} {rng.choice(FILLER_VERBS)} the {rng.choice(FILLER_ADJS)} {rng.choice(FILLER_NOUNS)} "
                             f"{rng.choice(FILLER_PREPS)} the {rng.choice(FILLER_NOUNS)} with {b} and the old keeper.")
        paragraphs.append(" ".join(sentences))
        size += len(paragraphs[-1]) + 2
    return "\n\n".join(paragraphs) + "\n"


def _facts(rng: random.Random, vocab: list, k: int) -> list:
    """`k` planted facts: ``{plant, question, paraphrase, names}``; the paraphrase swaps the verb and
    noun for synonyms (the names stay: a question cannot avoid them)."""
    facts = []
    for i in range(k):
        (verb, syn_verb), (noun, syn_noun) = FACT_VERBS[i % len(FACT_VERBS)], FACT_NOUNS[(i * 3 + i // 7) % len(FACT_NOUNS)]
        a, b = _names(rng, vocab, 2)
        facts.append({"plant": f"{a} {verb} the {noun} to {b} because the harvest had failed.",
                      "question": f"Who {verb} the {noun} to {b}?",
                      "paraphrase": f"Which person {syn_verb} the {syn_noun} for {b}?",
                      "names": [a, b]})
    return facts


def _names(rng: random.Random, vocab: list, k: int) -> list:
    return [rng.choice(vocab[2000:]).capitalize() for _ in range(k)]


@contextlib.contextmanager
def _vault(path: str):
    saved = wiki.WIKI_PATH
    wiki.WIKI_PATH = str(path)
    try:
        with patch.object(wiki, "_git", lambda *a, **kw: None):
            yield
    finally:
        wiki.WIKI_PATH = saved


def make_vault(path, pages: int = 5000, sources_n: int = 1000, seed: int = 0, chunks: int = 15,
               prose: bool = False, facts: int = 12) -> dict:
    """Generate the vault at `path` (must not exist yet); returns ``{pages, sources, chunks, seconds}``.

    `prose` writes the sources as readable sentences over invented names (no Zipf word salad) and plants
    `facts` one-sentence facts in them, returned as ``facts`` (see :func:`_facts`)."""
    if os.path.exists(path) and os.listdir(path):
        raise FileExistsError(f"{path} is not empty")
    rng = random.Random(seed)
    vocab = vocabulary(seed)
    cum = _weights(vocab)
    started = time.time()
    planted = _facts(rng, vocab, facts) if prose else []
    plant_at = {(f * sources_n) // len(planted): f for f in range(len(planted))}
    with _vault(path):
        wiki.ensure_vault()
        today = "2026-10-16"
        for i in range(pages):
            folder = FOLDERS[i % len(FOLDERS)]
            names = _names(rng, vocab, 3)
            fm = {"type": TYPES[folder], "name": f"{names[0]} {i}", "tags": [rng.choice(vocab[:200])],
                  "source": "generated", "updated": today, "seen": today}
            body = (f"# {names[0]} {i}\n\n## Summary\n{_text(rng, vocab, cum, names, 120).strip()}\n\n"
                    f"## Notes\n{_text(rng, vocab, cum, names, rng.randint(400, 1600))}\n## Related\n\n## History\n")
            wiki._write(f"{folder}/p{i:05d}.md", wiki._page_text(fm, body))
        for i in range(sources_n):
            names = _names(rng, vocab, 6)
            title = f"{names[0]} {names[1]} handbook {i}"
            if prose:
                body = _prose(rng, names, CHUNK_CHARS * chunks).split("\n\n")
                if i in plant_at:
                    body.insert(len(body) // 2, planted[plant_at[i]]["plant"])
                text = f"{title}\n\n" + "\n\n".join(body)
            else:
                text = f"{title}\n\n" + _text(rng, vocab, cum, names, CHUNK_CHARS * chunks)
            summary = json.dumps({"title": title, "summary": f"{title}: generated.", "entities": [],
                                  "topic": "generated", "unreadable": False})
            with fake_model(summary, "[]"):
                sources.ingest_upload(text.encode(), f"src-{i:05d}.md")
        wiki.rebuild_index()
        idx = wiki._db()
        counts = {"pages": len(idx.page_lengths()), "sources": len(idx.source_shas()),
                  "chunks": len(idx.chunk_lengths())}
    return {**counts, "seconds": round(time.time() - started, 1), **({"facts": planted} if prose else {})}


def queries(n: int = 200, seed: int = 0) -> list:
    """`n` questions over the generated vocabulary: 2–5 mid-frequency words, a third with a name."""
    rng = random.Random(seed + 1)
    vocab = vocabulary(seed)
    out = []
    for i in range(n):
        words = [rng.choice(vocab[50:5000]) for _ in range(rng.randint(2, 5))]
        if i % 3 == 0:
            words.insert(rng.randint(0, len(words)), rng.choice(vocab[2000:]).capitalize())
        out.append("what about " + " ".join(words))
    return out


def main(argv: list | None = None) -> None:
    """``python -m tests.perf.make_vault OUT [--prose] [--sources N] [--pages N]``: generate a vault
    (the caller sandboxes WIKI_PATH/SETTINGS_PATH/TRACE_DIR/MEMORY_DIR, as `eval/retrieval_bench.py` does).
    Prints the result as JSON on stdout."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("out")
    ap.add_argument("--prose", action="store_true", help="readable invented-name prose with planted facts")
    ap.add_argument("--sources", type=int, default=1000)
    ap.add_argument("--pages", type=int, default=5000)
    ap.add_argument("--chunks", type=int, default=15, help="chunks per source")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    made = make_vault(args.out, pages=args.pages, sources_n=args.sources, seed=args.seed, chunks=args.chunks,
                      prose=args.prose)
    print(json.dumps(made))


if __name__ == "__main__":
    main()
