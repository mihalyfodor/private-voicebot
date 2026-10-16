"""`file_source` — the voice way into raw ingestion (docs/20-raw-ingestion.md).

"file that email from Anna", "save that article", "remember this: …". The tool resolves the
reference against what the assistant last read out (gmail / news), or takes the note verbatim,
and hands the bytes to sources.py, which stores them under `raw/` and files a source page.
"""
import sources
from core import trace
log = trace.logger("Sources")

#: the vault only; the email / news bodies come from the tools that read them
HOSTS = ()

DEFINITION = {
    "type": "function",
    "function": {
        "name": "file_source",
        "description": (
            "Files something into the user's wiki as a source: an email they just heard "
            "('file that email from Anna'), a news article from the last headlines "
            "('save the second story'), or a note they dictate ('remember this: ...'). "
            "Use it only when the user asks for it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["email", "news", "note"],
                         "description": "What is being filed."},
                "ref": {"type": "string", "description": (
                    "For email: the sender, subject or position in the last inbox read-out. "
                    "For news: the headline, its number or the URL. For note: the text itself."
                )},
                "hint": {"type": "string", "description": "Optional one line of context."},
            },
            "required": ["kind", "ref"],
        },
    },
}


def _file_email(ref: str, hint: str) -> dict:
    from tools import gmail

    msg_id = gmail.resolve_ref(ref)
    if not msg_id:
        raise sources.IngestError(404, f"I couldn't tell which email {ref!r} is — "
                                       "read the inbox first, then say which one.")
    return sources.ingest_email(msg_id, gmail.fetch_message(msg_id), hint=hint)


def _file_news(ref: str, hint: str) -> dict:
    from tools import news

    title, url = news.resolve_ref(ref)
    if not url:
        raise sources.IngestError(404, f"I couldn't tell which story {ref!r} is — "
                                       "read the headlines first, then say which one.")
    body = f"{title}\n{url}\n\n{news.article_text(url)}"
    return sources.ingest_news(title, body, hint=hint)


def file_source(kind: str, ref: str, hint: str = "") -> str:
    """``(kind, ref)`` -> a spoken confirmation, or a clear error string. Never raises."""
    kind = str(kind or "").strip().lower()
    ref = str(ref or "").strip()
    if not ref:
        return "there is nothing to file: tell me which one, or what to write down."
    try:
        if kind == "email":
            filed = _file_email(ref, hint)
        elif kind == "news":
            filed = _file_news(ref, hint)
        elif kind == "note":
            filed = sources.ingest_note(ref, hint=hint)
        else:
            return f"I can file an email, a news story or a note — not {kind!r}."
    except sources.IngestError as exc:
        return exc.message
    except Exception as exc:  # noqa: BLE001 — a filing failure is spoken, never a crashed turn
        log.info(f"file_source failed: {exc}")
        return f"I couldn't file that: {exc}"
    _push_panel()
    if filed.get("already"):
        return f"already filed — it's on {filed['path']}."
    return f"filed as {filed['path']} (raw: {filed['raw']})."


def _push_panel() -> None:
    """Refresh an open Wiki tab through the turn's bound context, if any. No turn, no panel, no problem."""
    try:
        from tools import registry   # function-local: registry imports this module

        ctx = registry.current()
        if ctx is not None:
            ctx.push_panel()
    except Exception:  # noqa: BLE001
        pass


def run(args):
    args = args or {}
    return file_source(args.get("kind"), args.get("ref"), str(args.get("hint") or ""))
