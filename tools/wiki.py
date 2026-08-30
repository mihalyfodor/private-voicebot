"""Wiki recall tools: the model's only way into the vault (docs/19-wiki-layer.md)."""
import wiki

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
    """`<title> · part n/N (path): <text>` for each chunk hit, text capped at SNIPPET_CHARS."""
    lines = []
    for c in chunks:
        text = " ".join(c["text"].split())
        if len(text) > wiki.SNIPPET_CHARS:
            text = text[:wiki.SNIPPET_CHARS].rstrip() + "…"
        lines.append(f"{c['title']} · part {c['n']}/{c['total']} ({c['source']}): {text}")
    return lines


def run_search(args):
    query = str((args or {}).get("query") or "").strip()
    if not query:
        return "no query given"
    hits = wiki.search(query, k=5)
    chunks = wiki.search_chunks(query, k=3)
    if not hits and not chunks:
        return "no matching wiki pages"
    return "\n".join([f"{h['path']}: {h['snippet']}" for h in hits] + snippet_lines(chunks))


def run_read(args):
    return wiki.read_page(str((args or {}).get("path") or ""))


def run_read_source(args):
    args = args or {}
    return wiki.read_source(str(args.get("path") or ""), args.get("part"))
