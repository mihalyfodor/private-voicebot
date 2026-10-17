"""Page listing and lint (PRD 44; PRD 50 step 7, from wiki.py)."""
import os
from core import trace
from vault import index as v_index
from vault import search as v_search
from vault import store
from vault import focus

log = trace.logger("Wiki")


# --- listing / lint ----------------------------------------------------------------


def pages() -> list:
    """``[{path, title, type, updated}]`` for the Wiki tab, sorted by path.

    Paths are suffix-less (`people/anna`), the same form wikilinks and the tools use; the
    normalising on the way back in (:func:`normalise_path`) still accepts `.md`.
    """
    listing = []
    for rel in store.page_paths():
        text = store._read(rel)
        fm, _ = store.parse_page(text)
        listing.append({
            "path": store.wikilink(rel),
            "title": str(fm.get("name") or os.path.basename(rel)[:-3]),
            "type": str(fm.get("type") or ""),
            "updated": str(fm.get("updated") or ""),
            "tags": v_search.tags(text),
        })
    return listing


REINDEX_HINT = " — run `python -m wiki reindex`"


def lint() -> list:
    """Findings, worst first: per page broken links, missing from the search index, bad frontmatter,
    size, a `source: raw/…` whose file is gone; then index rows whose page is gone (both mean
    `python -m wiki reindex`), raw files nothing cites, then duplicate names. `index.md` is a capped
    human view (200 lines), so it is not what a page is checked against (docs/53-scalable-index.md).

    A finding names its page the way the rest of the layer does, `people/anna`, or its raw file.
    """
    from vault import manifest

    if not store.exists():
        return []
    findings = []
    cited, labels = set(), []
    paths = store.page_paths()
    known = {store.wikilink(rel) for rel in paths} | {store.wikilink(r) for r in store.RESERVED}
    indexed = set(v_search._db().page_paths())
    for rel in paths:
        text = store._read(rel)
        fm, _ = store.parse_page(text)
        name = store.wikilink(rel)
        for target in focus.links_in(text):
            if target not in known:
                findings.append(f"{name}: broken link [[{target}]]")
        if rel not in indexed:
            findings.append(f"{name}: missing from the search index{REINDEX_HINT}")
        missing = [key for key in store.FIELDS if key not in fm]
        if missing:
            findings.append(f"{name}: frontmatter missing {', '.join(missing)}")
        if len(text.encode()) > store.MAX_PAGE_BYTES:
            findings.append(f"{name}: larger than {store.MAX_PAGE_BYTES // 1024} kB")
        source = str(fm.get("source") or "")
        if source.startswith(store.RAW_DIR + "/") and not os.path.isfile(store._path(source)):
            findings.append(f"{name}: source {source} no longer exists")
        cited |= {str(fm.get(key) or "") for key in ("source", "raw")}
        labels.append((name, " ".join(str(fm.get("name") or "").split())))
    findings += [f"{store.wikilink(rel)}: in the search index but not on disk{REINDEX_HINT}"
                 for rel in sorted(indexed - set(paths))]
    cited |= {e.get("path") for e in manifest.manifest_entries()}
    cited |= {rel for e in manifest.manifest_entries() for rel in e.get("pages") or []}   # PRD 47 PNGs
    for root, dirs, files in os.walk(store._path(store.RAW_DIR)):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fname in sorted(files):
            rel = os.path.relpath(os.path.join(root, fname), store.WIKI_PATH).replace(os.sep, "/")
            if not fname.startswith(".") and rel != manifest.MANIFEST and rel not in cited:
                findings.append(f"{rel}: not in the manifest and cited by no page")
    first = {}
    for name, label in sorted(labels):
        if not label or name.startswith("meetings/"):   # recurring invites share a name
            continue
        group = (name.startswith("sources/"), label.casefold())
        if group in first:
            findings.append(f'{name}: duplicate name "{label}" (also {first[group]})')
        first.setdefault(group, name)
    return findings


def lint_and_log() -> list:
    """`lint()` from the reflection cadence: only a run with findings earns a `log.md` line."""
    findings = lint()
    if findings:
        store.ensure_vault()
        summary = f"lint: {len(findings)} finding{'s' if len(findings) != 1 else ''} ({findings[0]})"
        v_index._log([summary])
        store._commit("wiki: lint")
    return findings
