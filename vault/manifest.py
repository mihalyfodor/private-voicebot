"""`raw/manifest.jsonl`: one line per filed source (docs/20-raw-ingestion.md). Read, append, rewrite."""
import hashlib
import json
import os

from vault import store

MANIFEST = "raw/manifest.jsonl"

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_entries() -> list:
    """Every line of `raw/manifest.jsonl`, parsed; a garbled line is skipped, never fatal."""
    entries = []
    for line in (store._read(MANIFEST) or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def manifest_find(digest: str) -> dict | None:
    return next((e for e in manifest_entries() if e.get("sha256") == digest), None)


def _manifest_append(entry: dict) -> None:
    full = store._path(MANIFEST)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _manifest_write(entries: list) -> None:
    """Rewrite the whole manifest — the only time a line is replaced or dropped."""
    store._write(MANIFEST, "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
