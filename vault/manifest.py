"""`raw/manifest.jsonl`: one line per filed source (docs/20-raw-ingestion.md). Read, append, rewrite."""
import hashlib
import json
import os
import re

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


# --- the inbox (docs/25-sources-inbox.md) --------------------------------------------


_DROP_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _when(date: str, ingested_at: str) -> str:
    """`2026-08-30 09:12` from the ISO stamp (which already carries the date); the drop date alone otherwise."""
    if "T" in ingested_at:                       # full ISO stamp: it already carries the date
        return ingested_at[:16].replace("T", " ")
    return " ".join(part for part in (date, ingested_at) if part)


def _source_meta(entry: dict) -> dict:
    """One inbox row: the manifest line joined with its page's frontmatter."""
    page_rel = f"{entry.get('page', '')}.md"
    fm = store.parse_page(store._read(page_rel))[0] if store.page_exists(page_rel) else {}
    match = _DROP_DATE.search(str(fm.get("source") or ""))
    date = match.group(1) if match else str(fm.get("updated") or "")
    return {
        "sha": entry.get("sha256", ""),
        "path": entry.get("page", ""),
        "raw": entry.get("path", ""),
        "kind": str(fm.get("kind") or entry.get("kind") or ""),
        "title": str(fm.get("name") or os.path.basename(entry.get("page", ""))),
        "when": _when(date, str(entry.get("ingested_at") or "")),
        "fed": list(entry.get("pages_touched") or []),
        "meeting": str(fm.get("meeting") or ""),
    }
