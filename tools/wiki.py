"""Wiki recall tools: the model's only way into the vault (docs/19-wiki-layer.md)."""
import wiki

DEFINITION_SEARCH = {
    "type": "function",
    "function": {
        "name": "search_wiki",
        "description": (
            "Searches the user's wiki (people, projects, meetings, decisions) and returns the "
            "best matching pages with a snippet. Use it whenever the user asks about past work, "
            "a meeting, a decision or someone they mentioned before."
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


def run_search(args):
    query = str((args or {}).get("query") or "").strip()
    if not query:
        return "no query given"
    hits = wiki.search(query, k=5)
    if not hits:
        return "no matching wiki pages"
    return "\n".join(f"{h['path']}: {h['snippet']}" for h in hits)


def run_read(args):
    return wiki.read_page(str((args or {}).get("path") or ""))
