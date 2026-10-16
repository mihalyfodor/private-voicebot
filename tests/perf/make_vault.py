"""The shared synthetic-vault generator (docs/53-scalable-index.md scenario 10; PRDs 54 and 55 reuse it).

`make_vault(path)` writes a seeded vault: `pages` knowledge pages through `wiki._write` (the page
write hook) and `sources` text sources through the real filing path (`sources.ingest_upload`) with
the model faked, ~15 chunks each. Same seed, same vault. Git is switched off while it runs: the
history of a generated vault is not what is measured, and 1k commits would dominate the time.

Call it with the app's data paths sandboxed (the conftest autouse fixtures do that for a test):
filing reads the profile and writes trace events like a real drop.
"""
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


def make_vault(path, pages: int = 5000, sources_n: int = 1000, seed: int = 0, chunks: int = 15) -> dict:
    """Generate the vault at `path` (must not exist yet); returns ``{pages, sources, chunks, seconds}``."""
    if os.path.exists(path) and os.listdir(path):
        raise FileExistsError(f"{path} is not empty")
    rng = random.Random(seed)
    vocab = vocabulary(seed)
    cum = _weights(vocab)
    started = time.time()
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
            text = f"{title}\n\n" + _text(rng, vocab, cum, names, CHUNK_CHARS * chunks)
            summary = json.dumps({"title": title, "summary": f"{title}: generated.", "entities": [],
                                  "topic": "generated", "unreadable": False})
            with fake_model(summary, "[]"):
                sources.ingest_upload(text.encode(), f"src-{i:05d}.md")
        wiki.rebuild_index()
        idx = wiki._db()
        counts = {"pages": len(idx.page_lengths()), "sources": len(idx.source_shas()),
                  "chunks": len(idx.chunk_lengths())}
    return {**counts, "seconds": round(time.time() - started, 1)}


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
