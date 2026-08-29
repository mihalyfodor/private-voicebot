"""Raw ingestion — the office document stack feeds the wiki (docs/20-raw-ingestion.md).

wiki.py owns page mechanics (frontmatter, ops, index, commits). This module owns *sources*:
the bytes the user drops (files, pastes, screenshots), emails and news items they ask to file.

One ingest is always the same five steps, under `memory._LOCK`, ending in one commit:

1. hash the bytes; a sha256 already in ``raw/manifest.jsonl`` means "already filed", stop.
2. copy the original, untouched, under ``raw/`` — the provenance the pages point back at.
3. extract text (md/txt as-is, pdf via pypdf, .eml via stdlib email, .msg via extract_msg) or,
   for an image, a downscaled base64 data URL for the multimodal model.
4. one summary completion -> ``{title, summary, entities, unreadable}`` -> ``sources/<slug>.md``.
5. one page-ops completion (PRD 19's op set, targets restricted to pages already in the index
   plus new people/projects pages), every op carrying ``source: raw/…``.

Nothing here is ever triggered automatically: the popup, a tool call or the harness asks for it.
"""
import base64
import binascii
import email
import email.policy
import hashlib
import io
import json
import os
import re
import subprocess

import memory
import wiki

# --- limits ------------------------------------------------------------------------

#: how much extracted text the model is shown (the rest is dropped, no chunk-and-merge)
RAW_CHUNK_TOKENS = int(os.getenv("RAW_CHUNK_TOKENS", "6000"))
TRUNCATED = "\n\n[truncated]"

MAX_FILE_BYTES = 2 * 1024 * 1024        # md/txt/pdf/eml/msg
MAX_IMAGE_BYTES = 4 * 1024 * 1024       # png/jpg
IMAGE_LONG_EDGE = 1568                  # what the multimodal model wants to see
#: below this, downscaling costs more than it saves (and keeps the tests off `sips`)
DOWNSCALE_ABOVE_BYTES = 256 * 1024

#: a PDF that yields less than this much text has no text layer we can use
MIN_PDF_CHARS = 20

MANIFEST = "raw/manifest.jsonl"

TEXT_EXTS = {".md": "md", ".markdown": "md", ".txt": "txt"}
IMAGE_EXTS = {".png": "screenshot", ".jpg": "screenshot", ".jpeg": "screenshot"}
DOC_EXTS = {".pdf": "pdf", ".eml": "email", ".msg": "email"}
#: extension -> the `kind` that lands in the source page's frontmatter
KINDS = {**TEXT_EXTS, **IMAGE_EXTS, **DOC_EXTS}

#: page folders an op may create in when the page is not in the index yet
NEW_PAGE_FOLDERS = ("people", "projects")


class IngestError(Exception):
    """A refusal the caller turns into an HTTP status (`status`) or an error string."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# --- prompts -----------------------------------------------------------------------

SUMMARY_SYSTEM = """You are filing one document into the USER's personal wiki.

Reply with a JSON object and nothing else:
{"title": "short title", "summary": "3-6 sentences", "entities": [{"name": "Anna", "type": "person"}], "unreadable": false}

- `title` names the document the way the user would refer to it.
- `summary` is what the document says: facts, status, decisions, dates, numbers. No preamble.
- `entities` are the people, projects and organisations the document is about; `type` is one of
  person, project, org, other. At most 8, most important first.
- Set `unreadable` to true (and leave the summary short) when the text is garbled, empty or
  clearly a failed extraction.
- For an image: describe what it shows and transcribe any text visible in it.
- %s Never list the user as an entity of type person.
Today is %s."""

OPS_SYSTEM = """You maintain the USER's markdown wiki. A new document has just been filed as a
source page; decide what the rest of the wiki should learn from it.

Reply with a JSON array of page operations and nothing else. Each operation is one of:
{"op":"new_page","path":"people/anna","type":"person","name":"Anna","summary":"one line"}
{"op":"append_note","path":"projects/atlas","text":"one sentence"}
{"op":"set_field","path":"projects/atlas","key":"status","value":"paused"}
{"op":"add_decision","title":"deploy on Fridays","text":"one or two sentences"}
{"op":"link","path":"people/anna","target":"projects/atlas"}

Rules:
- Only touch pages listed in the index, or create new `people/<slug>` / `projects/<slug>` pages.
- Never touch the source page itself; it is already written.
- A note is one sentence, at most 600 characters, and states a fact from the document.
- Prefer appending to an existing page over creating a near-duplicate.
- %s
- Nothing worth filing? Reply with exactly [].
Today is %s."""


# --- llm ---------------------------------------------------------------------------


def _client_model() -> tuple:
    """The live oMLX client and model. Split out so tests can hand in a fake."""
    import llm

    return llm.client(), llm.OMLX_MODEL


def _ask(client, model: str, system: str, content, max_tokens: int) -> str:
    """One completion. `content` is a string, or the multimodal content-part list for an image."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# --- extraction --------------------------------------------------------------------


def cap_text(text: str) -> str:
    """`text` trimmed to `RAW_CHUNK_TOKENS` with a marker — the model never sees more."""
    text = text or ""
    if memory.tokens_est(text) <= RAW_CHUNK_TOKENS:
        return text
    return text[: RAW_CHUNK_TOKENS * 4].rstrip() + TRUNCATED


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.S | re.I)


def _strip_html(html: str) -> str:
    return " ".join(_TAGS.sub(" ", html or "").split())


def pdf_text(data: bytes) -> str:
    """Every page's text, joined. An unreadable PDF (no text layer) yields ""."""
    try:
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 — a broken PDF is "unreadable", not a crash
        print(f"[Sources] pdf extraction failed: {exc}")
        return ""
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    return text if len(text.strip()) >= MIN_PDF_CHARS else ""


def eml_text(data: bytes) -> str:
    """`.eml` -> a headers block plus the text/plain part (falling back to stripped html)."""
    try:
        msg = email.message_from_bytes(data, policy=email.policy.default)
    except Exception as exc:  # noqa: BLE001
        print(f"[Sources] eml parse failed: {exc}")
        return ""
    plain, html = "", ""
    for part in (msg.walk() if msg.is_multipart() else [msg]):
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            body = part.get_content()
        except Exception:  # noqa: BLE001
            continue
        if ctype == "text/plain" and not plain:
            plain = str(body)
        elif ctype == "text/html" and not html:
            html = str(body)
    return _headers_block(
        {k: msg.get(k, "") for k in ("From", "To", "Date", "Subject")},
        plain.strip() or _strip_html(html),
    )


def msg_text(path: str) -> str:
    """Outlook `.msg` (a drag-out from Outlook for Mac) via extract_msg, same shape as `.eml`."""
    try:
        import extract_msg

        with extract_msg.openMsg(path) as msg:
            headers = {"From": msg.sender or "", "To": msg.to or "",
                       "Date": str(msg.date or ""), "Subject": msg.subject or ""}
            body = msg.body or ""
    except Exception as exc:  # noqa: BLE001
        print(f"[Sources] msg extraction failed: {exc}")
        return ""
    return _headers_block(headers, str(body).strip())


def _headers_block(headers: dict, body: str) -> str:
    lines = [f"{key}: {value}" for key, value in headers.items() if str(value).strip()]
    return "\n".join(lines) + "\n\n" + (body or "")


def _downscale(full: str) -> None:
    """Shrink an image in place to `IMAGE_LONG_EDGE` using macOS `sips`; failures are ignored.

    Pillow is not a dependency of this app, and the cap (4 MB) already bounds the upload; small
    images are left alone so a test never shells out.
    """
    if os.path.getsize(full) <= DOWNSCALE_ABOVE_BYTES:
        return
    try:
        subprocess.run(["sips", "-Z", str(IMAGE_LONG_EDGE), full],
                       capture_output=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[Sources] downscale skipped: {exc}")


def _image_part(full: str) -> dict:
    mime = "image/jpeg" if full.lower().endswith((".jpg", ".jpeg")) else "image/png"
    with open(full, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def extract(raw_rel: str, data: bytes) -> tuple:
    """``(text, image_part)`` for a stored raw file — exactly one of the two is set.

    Text is already capped at `RAW_CHUNK_TOKENS`; "" means "nothing extractable" (unreadable).
    """
    ext = os.path.splitext(raw_rel)[1].lower()
    full = wiki._path(raw_rel)
    if ext in IMAGE_EXTS:
        _downscale(full)
        return "", _image_part(full)
    if ext == ".pdf":
        return cap_text(pdf_text(data)), None
    if ext == ".eml":
        return cap_text(eml_text(data)), None
    if ext == ".msg":
        return cap_text(msg_text(full)), None
    return cap_text(_decode(data)), None


# --- manifest ----------------------------------------------------------------------


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_entries() -> list:
    """Every line of `raw/manifest.jsonl`, parsed; a garbled line is skipped, never fatal."""
    entries = []
    for line in (wiki._read(MANIFEST) or "").splitlines():
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
    full = wiki._path(MANIFEST)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --- source page -------------------------------------------------------------------


def _free_path(folder: str, slug: str, ext: str) -> str:
    """`folder/slug.ext`, with `-2`, `-3`… when that name is taken by other bytes."""
    rel = f"{folder}/{slug}{ext}"
    n = 2
    while os.path.exists(wiki._path(rel)):
        rel = f"{folder}/{slug}-{n}{ext}"
        n += 1
    return rel


def _entity_line(entity: dict, user_names=()) -> str:
    """One `## Entities` bullet — a wikilink only when the page exists, so lint stays quiet.

    The profile owner is spelled out as "the user" and never linked: they have no people page
    (docs/26-identity-aliases.md) and one must not be invented for them here either.
    """
    name = " ".join(str(entity.get("name") or "").split())
    if not name:
        return ""
    if memory.is_user(name, user_names):
        return f"- the user ({name})"
    etype = " ".join(str(entity.get("type") or "").split()).lower()
    folder = {"person": "people", "project": "projects"}.get(etype)
    if folder:
        rel = f"{folder}/{wiki.slugify(name)}.md"
        if wiki.page_exists(rel):
            return f"- [[{wiki.wikilink(rel)}]] — {name} ({etype or 'entity'})"
    return f"- {name} ({etype or 'entity'})"


def _source_body(title: str, summary: str, entities: list, fed: list, user_names=()) -> str:
    lines = [f"# {title}", "", "## Summary", summary.strip() or "(no summary)", "", "## Entities"]
    lines += [line for line in (_entity_line(e, user_names) for e in entities
                                if isinstance(e, dict)) if line]
    lines += ["", "## Fed"]
    lines += [f"- [[{path}]]" for path in fed]
    return "\n".join(lines).rstrip() + "\n"


def _write_source_page(rel: str, *, title: str, kind: str, raw_rel: str, digest: str,
                       summary: str, entities: list, fed: list, source: str,
                       user_names=()) -> None:
    today = memory._today_iso()
    fm = {"type": "source", "name": title, "tags": [], "source": source,
          "updated": today, "seen": today,
          "kind": kind, "raw": raw_rel, "sha256": digest}
    wiki._write(rel, wiki._page_text(fm, _source_body(title, summary, entities, fed, user_names)))


# --- ops ---------------------------------------------------------------------------


def restrict_ops(ops, raw_rel: str, user_names=()) -> tuple:
    """``(kept, rejected)``: only index pages and new people/projects pages may be touched.

    The model proposes these, so they are untrusted twice over — once for the path (wiki does
    that) and once for the *target*: an ingest may not invent a `meetings/` or `sources/` page.
    Every kept op is stamped with `source: raw/…`, the PRD's provenance rule.

    An op that would file the *user* as a person is refused too: a document mentioning their
    own (possibly misspelled) name must not grow a `people/` page about them.
    """
    kept, rejected = [], []
    for raw in ops if isinstance(ops, list) else []:
        op = raw if isinstance(raw, dict) else {}
        kind = str(op.get("op") or "").strip().lower()
        about_user = wiki.user_page_reason(op, user_names)
        if about_user:
            rejected.append({"op": raw, "reason": about_user})
            continue
        if kind == "add_decision":
            kept.append({**op, "source": raw_rel})
            continue
        checked = [op.get(key) for key in ("path", "target") if key in op] or [None]
        reason = next((r for r in (_target_allowed(p) for p in checked) if r), None)
        if reason:
            rejected.append({"op": raw, "reason": reason})
        else:
            kept.append({**op, "source": raw_rel})
    return kept, rejected


def _target_allowed(path) -> str | None:
    rel = wiki.normalise_path(path)
    if rel is None:
        return f"bad path {path!r}"
    if wiki.page_exists(rel):
        return None
    if rel.split("/")[0] not in NEW_PAGE_FOLDERS:
        return f"{wiki.wikilink(rel)} is not in the index and not a people/projects page"
    return None


def summarise(client, model: str, kind: str, text: str, image_part, hint: str, title: str,
              user_names=()) -> dict:
    """One completion -> ``{title, summary, entities, unreadable}``; never raises.

    An empty extraction skips the call entirely: there is nothing to summarise, and the page is
    still worth creating (it carries the link to the raw file — PRD's "unreadable" risk).
    """
    fallback = {"title": title or "Untitled", "summary": "", "entities": [], "unreadable": True}
    if not text.strip() and image_part is None:
        fallback["summary"] = "unreadable — no text could be extracted from this file."
        return fallback
    context = [f"Kind: {kind}"]
    if title:
        context.append(f"Filename/title: {title}")
    if hint:
        context.append(f"User hint: {hint}")
    header = "\n".join(context)
    if image_part is not None:
        content = [{"type": "text", "text": f"{header}\n\nSummarise this image."}, image_part]
    else:
        content = f"{header}\n\nDocument:\n{text}"
    try:
        system = SUMMARY_SYSTEM % (wiki.user_note(user_names), memory._today_iso())
        raw = _ask(client, model, system, content, 700)
    except Exception as exc:  # noqa: BLE001
        print(f"[Sources] summary failed: {exc}")
        fallback["summary"] = "the summary model could not be reached; the raw file is filed."
        return fallback
    data = memory._extract_json(raw, "{", "}")
    if not isinstance(data, dict):
        print("[Sources] could not parse the summary; filing the raw file only")
        fallback["summary"] = "the summary could not be parsed; the raw file is filed."
        return fallback
    entities = [e for e in (data.get("entities") or []) if isinstance(e, dict)][:8]
    summary = " ".join(str(data.get("summary") or "").split())
    return {
        "title": " ".join(str(data.get("title") or title or "Untitled").split())[:120],
        "summary": summary,
        "entities": entities,
        "unreadable": bool(data.get("unreadable")) or not summary,
    }


def propose_page_ops(client, model: str, source_rel: str, summary: dict, hint: str,
                     hint_page: str = "", user_names=()) -> list:
    """Ask for page ops about a just-filed source; [] on anything unparseable."""
    index = wiki.index_text(wiki.WIKI_INDEX_TOKENS) or "(empty wiki)"
    entities = ", ".join(f"{e.get('name')} ({e.get('type')})" for e in summary.get("entities", []))
    parts = [f"Index:\n{index}", f"Source page: {wiki.wikilink(source_rel)}",
             f"Title: {summary.get('title')}", f"Summary:\n{summary.get('summary')}"]
    if entities:
        parts.append(f"Entities: {entities}")
    if hint:
        parts.append(f"User hint: {hint}")
    if hint_page:
        parts.append(f"The user filed this under: {hint_page}")
    try:
        system = OPS_SYSTEM % (wiki.user_note(user_names), memory._today_iso())
        raw = _ask(client, model, system, "\n\n".join(parts), 800)
    except Exception as exc:  # noqa: BLE001
        print(f"[Sources] page ops failed: {exc}")
        return []
    ops = memory._extract_json(raw)
    return ops if isinstance(ops, list) else []


# --- the ingest ---------------------------------------------------------------------


def _store(raw_rel: str, data: bytes) -> None:
    full = wiki._path(raw_rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)


def file_bytes(data: bytes, raw_rel: str, kind: str, *, title: str = "", hint: str = "",
               page: str = "", client=None, model: str = "") -> dict:
    """The one ingest path: store, extract, summarise, file, commit once.

    Returns ``{path, raw, already}`` — `path` is the source page (`sources/<slug>`), `raw` the
    vault-relative original, `already` True when these exact bytes were filed before.
    """
    if client is None:
        client, model = _client_model()
    with memory._LOCK:
        wiki.ensure_vault()
        names = memory.user_names()
        digest = sha256(data)
        seen = manifest_find(digest)
        if seen:
            return {"path": seen.get("page", ""), "raw": seen.get("path", ""), "already": True}

        folder, name = os.path.split(raw_rel)
        stem, ext = os.path.splitext(name)
        raw_rel = _free_path(folder, stem, ext)
        _store(raw_rel, data)

        text, image_part = extract(raw_rel, data)
        summary = summarise(client, model, kind, text, image_part, hint, title or stem, names)
        page_title = title or summary.get("title") or stem
        source_rel = _free_path("sources", wiki.slugify(page_title), ".md")

        def write_page(fed: list) -> None:
            """The source page, written once before the ops and once after (now with `## Fed`)."""
            _write_source_page(source_rel, title=page_title, kind=kind, raw_rel=raw_rel,
                               digest=digest, summary=summary.get("summary", ""),
                               entities=summary.get("entities", []), fed=fed,
                               source=f"user drop {memory._today_iso()}", user_names=names)

        write_page([])

        applied = []
        if not summary.get("unreadable"):
            ops = propose_page_ops(client, model, source_rel, summary, hint, page, names)
            kept, rejected = restrict_ops(ops, raw_rel, names)
            for bad in rejected:
                print(f"[Sources] rejected {bad['op']}: {bad['reason']}")
            applied, more = wiki.apply_page_ops(kept, raw_rel, commit=False)
            for bad in more:
                print(f"[Sources] rejected {bad['op']}: {bad['reason']}")

        fed = sorted({wiki.wikilink(wiki.normalise_path(op["path"]))
                      for op in applied if op.get("path")}
                     | {"decisions" for op in applied if op["op"] == "add_decision"})
        write_page(fed)     # rewritten now that the ops are in, listing what it actually fed
        _manifest_append({"sha256": digest, "path": raw_rel, "kind": kind,
                          "ingested_at": memory._stamp("%H:%M:%S"),
                          "page": wiki.wikilink(source_rel), "pages_touched": fed})
        wiki.rebuild_index()
        wiki._log([f"filed [[{wiki.wikilink(source_rel)}]] from {raw_rel}"
                   + (f" -> {', '.join(fed)}" if fed else "")])
        wiki._commit(f"wiki: filed {raw_rel}")
        return {"path": wiki.wikilink(source_rel), "raw": raw_rel, "already": False}


def _dated(slug: str) -> str:
    return f"{memory._today_iso()}-{wiki.slugify(slug)}"


def check_upload(filename: str, size: int, kind_hint: str = "file") -> tuple:
    """``(ext, kind)`` for an upload, or `IngestError` — the one type/size gate.

    The endpoint calls it on arrival (so an oversized drop is refused before any work happens)
    and `ingest_upload` calls it again for the callers that never went through HTTP.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    if kind_hint == "paste_image" and ext not in IMAGE_EXTS:
        ext = ".png"
    kind = KINDS.get(ext)
    if kind is None:
        raise IngestError(415, f"unsupported file type {ext or filename!r}")
    limit = MAX_IMAGE_BYTES if ext in IMAGE_EXTS else MAX_FILE_BYTES
    if size > limit:
        raise IngestError(413, f"{filename} is larger than {limit // (1024 * 1024)} MB")
    return ext, kind


def ingest_upload(data: bytes, filename: str, *, kind_hint: str = "file", title: str = "",
                  hint: str = "", page: str = "", client=None, model: str = "") -> dict:
    """One uploaded file (or pasted image). Raises `IngestError` for size/type refusals."""
    ext, kind = check_upload(filename, len(data), kind_hint)
    stem = title or os.path.splitext(os.path.basename(filename or ""))[0] or "drop"
    folder = "raw/screens" if ext in IMAGE_EXTS else "raw/inbox"
    raw_rel = f"{folder}/{_dated(stem)}{ext}"
    return file_bytes(data, raw_rel, kind, title=title, hint=hint, page=page,
                      client=client, model=model)


def check_note(text: str) -> str:
    """The stripped note body, or `IngestError` — the text counterpart of `check_upload`."""
    body = (text or "").strip()
    if not body:
        raise IngestError(400, "nothing to file: the text is empty")
    if len(body.encode()) > MAX_FILE_BYTES:
        raise IngestError(413, f"the text is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    return body


def ingest_note(text: str, *, title: str = "", hint: str = "", page: str = "",
                client=None, model: str = "") -> dict:
    """Pasted or spoken free text -> `raw/inbox/<date>-<slug>.md`."""
    body = check_note(text)
    stem = title or "note"
    raw_rel = f"raw/inbox/{_dated(stem)}.md"
    return file_bytes(body.encode(), raw_rel, "note", title=title, hint=hint, page=page,
                      client=client, model=model)


def ingest_path(path: str, *, title: str = "", hint: str = "", page: str = "",
                client=None, model: str = "") -> dict:
    """A file already on disk (the eval harness's `ingest:` turn, an AppleScript drop)."""
    try:
        with open(os.path.expanduser(path), "rb") as f:
            data = f.read()
    except OSError as exc:
        raise IngestError(400, f"cannot read {path}: {exc}") from exc
    return ingest_upload(data, os.path.basename(path), title=title, hint=hint, page=page,
                         client=client, model=model)


def ingest_email(message_id: str, markdown: str, *, hint: str = "", page: str = "",
                 client=None, model: str = "") -> dict:
    """A fetched email -> `raw/email/<id>.md` (headers block + body, already rendered)."""
    ident = wiki.slugify(message_id) or _dated("email")
    subject = next((line[len("Subject: "):] for line in markdown.splitlines()
                    if line.startswith("Subject: ")), "").strip()
    return file_bytes(markdown.encode(), f"raw/email/{ident}.md", "email",
                      title=subject, hint=hint, page=page, client=client, model=model)


def ingest_news(title: str, markdown: str, *, hint: str = "", page: str = "",
                client=None, model: str = "") -> dict:
    """A flagged news article -> `raw/news/<slug>.md`."""
    return file_bytes(markdown.encode(), f"raw/news/{wiki.slugify(title)}.md", "news",
                      title=title, hint=hint, page=page, client=client, model=model)


def decode_data_url(value: str) -> bytes:
    """`data:image/png;base64,…` -> bytes (the paste path when it arrives as a string)."""
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise IngestError(415, "the pasted image could not be decoded") from exc
