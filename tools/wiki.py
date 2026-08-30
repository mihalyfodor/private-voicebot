"""Wiki recall tools: the model's only way into the vault (docs/19-wiki-layer.md)."""
import wiki
from untrusted import untrusted

#: the vault only (docs/33-untrusted-text.md read-only policy)
HOSTS = ()

DEFINITION_SEARCH = {
    "type": "function",
    "function": {
        "name": "search_wiki",
        "description": (
            "Searches the user's wiki (people, projects, meetings, decisions) and the text of "
            "filed documents (guides, FAQs, notes). Returns the best matching pages, then "
            "passages from inside documents as '<title> · part n/N'. Use it whenever the user "
            "asks about past work, a meeting, a decision, someone they mentioned before, or "
            "anything a filed document might cover."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in a few words."}
            },
            "required": ["query"],
        },
    },
}

DEFINITION_READ = {
    "type": "function",
    "function": {
        "name": "read_page",
        "description": (
            "Returns the full text of one wiki page, e.g. 'people/anna' or 'decisions'. "
            "Use the path shown in the wiki index or in a search_wiki result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The page path, e.g. people/anna."}
            },
            "required": ["path"],
        },
    },
}


DEFINITION_READ_SOURCE = {
    "type": "function",
    "function": {
        "name": "read_source",
        "description": (
            "Returns one part of a filed document with the text around it, e.g. part 12 of "
            "'sources/fallout-4-guide'. Use the path and part number from a search_wiki passage; "
            "ask for part+1 to keep reading."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "The source page path, e.g. sources/guide."},
                "part": {"type": "integer", "description": "The part number from search_wiki."},
            },
            "required": ["path", "part"],
        },
    },
}


def snippet_lines(chunks: list) -> list:
    """`<title> · part n/N (path): <text>` for each chunk hit, text capped at SNIPPET_CHARS,
    each one its own document block."""
    lines = []
    for c in chunks:
        text = " ".join(c["text"].split())
        if len(text) > wiki.SNIPPET_CHARS:
            text = text[:wiki.SNIPPET_CHARS].rstrip() + "…"
        lines.append(untrusted(f"{c['title']} · part {c['n']}/{c['total']} ({c['source']}): {text}", c["title"]))
    return lines


def run_search(args):
    query = str((args or {}).get("query") or "").strip()
    if not query:
        return "no query given"
    hits = wiki.search(query, k=5)
    chunks = wiki.search_chunks(query, k=3)
    if not hits and not chunks:
        return "no matching wiki pages"
    pages = [untrusted("\n".join(f"{h['path']}: {h['snippet']}" for h in hits), "wiki pages")] if hits else []
    return "\n".join(pages + snippet_lines(chunks))


def run_read(args):
    path = str((args or {}).get("path") or "")
    text = wiki.read_page(path)
    return text if text.startswith("no such page") else untrusted(text, path)


def run_read_source(args):
    args = args or {}
    text = wiki.read_source(str(args.get("path") or ""), args.get("part"))
    if text.startswith(("no such source", "no part ")):
        return text
    return untrusted(text, text.split(" · part ", 1)[0])
