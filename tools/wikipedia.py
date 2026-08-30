"""Wikipedia lookups (docs/37-wikipedia-routing.md): search + summary, and one named section.

Fixed hosts only, 5 s timeouts, one request per second, results cached per title for the session.
Every failure comes back as a plain error string — the model reports it, nothing raises.
"""
import os
import re
import time
import urllib.parse

import requests

#: the only hosts this module ever talks to (the PRD 33 read-only policy test checks this list)
HOSTS = ("en.wikipedia.org",)
API_URL = "https://en.wikipedia.org/w/api.php"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
TIMEOUT = 5
MIN_INTERVAL = 1.0
RETRY_AFTER = 3.0
SEARCH_LIMIT = 3
SUMMARY_CHARS = 1200
SECTION_CHARS = 3000
VERSION = "1.0"

DEFINITION_GET = {
    "type": "function",
    "function": {
        "name": "get_wikipedia",
        "description": ("Searches English Wikipedia for a topic and returns the best article's title, a "
                        "one-paragraph summary and its URL. Only after the user agreed to check Wikipedia, "
                        "or when they ask for it outright."),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The topic to look up, e.g. 'Ada Lovelace'."}},
            "required": ["query"],
        },
    },
}

DEFINITION_READ = {
    "type": "function",
    "function": {
        "name": "read_wikipedia",
        "description": "Reads one named section of a Wikipedia article (by article title and section heading).",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The exact article title from get_wikipedia."},
                "section": {"type": "string", "description": "The section heading to read, e.g. 'Early life'."},
            },
            "required": ["title", "section"],
        },
    },
}

_cache: dict = {}
_last_request = 0.0


def _headers() -> dict:
    ua = f"voicebot/{VERSION} (personal assistant"
    contact = os.getenv("WIKIPEDIA_CONTACT", "").strip()
    if contact:
        ua += f"; {contact}"
    return {"User-Agent": ua + ")", "Accept": "application/json"}


def _get(url: str, params: dict | None = None) -> dict:
    """One throttled GET against a fixed host; the parsed JSON, or raises requests.RequestException."""
    global _last_request
    if urllib.parse.urlsplit(url).hostname not in HOSTS:
        raise requests.RequestException(f"host not allowed: {url}")
    for attempt in (1, 2):
        wait = MIN_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=TIMEOUT)
            if resp.status_code == 429 and attempt == 1:     # rate limited: one retry after a pause
                time.sleep(RETRY_AFTER)
                continue
            resp.raise_for_status()
            return resp.json()
        finally:
            _last_request = time.monotonic()


def _search(query: str) -> list[str]:
    key = ("search", query.strip().lower())
    if key not in _cache:
        data = _get(API_URL, {"action": "query", "list": "search", "srsearch": query, "srlimit": SEARCH_LIMIT,
                              "format": "json"})
        _cache[key] = [hit["title"] for hit in data.get("query", {}).get("search", [])]
    return _cache[key]


def _summary(title: str) -> dict:
    key = ("summary", title)
    if key not in _cache:
        data = _get(SUMMARY_URL + urllib.parse.quote(title.replace(" ", "_"), safe=""))
        _cache[key] = {
            "title": data.get("title") or title,
            "extract": (data.get("extract") or "").strip()[:SUMMARY_CHARS],
            "url": (data.get("content_urls") or {}).get("desktop", {}).get("page")
                   or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
        }
    return _cache[key]


def _sections(title: str) -> list[dict]:
    key = ("sections", title)
    if key not in _cache:
        data = _get(API_URL, {"action": "parse", "page": title, "prop": "sections", "format": "json"})
        if "error" in data:
            raise requests.RequestException(data["error"].get("info", "parse error"))
        _cache[key] = data.get("parse", {}).get("sections", [])
    return _cache[key]


_MARKUP = [
    (re.compile(r"\{\{[^{}]*\}\}"), ""),                       # templates (innermost; run twice)
    (re.compile(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", re.S), ""),  # references
    (re.compile(r"<[^>]+>"), ""),                              # other tags
    (re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]"), r"\1"),    # [[target|label]] → label
    (re.compile(r"\[https?://\S+ ([^\]]*)\]"), r"\1"),          # [url label] → label
    (re.compile(r"'{2,}"), ""),                                # bold/italic quotes
    (re.compile(r"^==+\s*(.*?)\s*==+$", re.M), r"\1"),          # headings
]


def _plain(wikitext: str) -> str:
    text = wikitext
    for _ in range(2):
        text = _MARKUP[0][0].sub("", text)
    for pattern, repl in _MARKUP[1:]:
        text = pattern.sub(repl, text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _section_text(title: str, section: dict) -> str:
    key = ("section", title, section["index"])
    if key not in _cache:
        data = _get(API_URL, {"action": "parse", "page": title, "section": section["index"], "prop": "wikitext",
                              "format": "json"})
        raw = data.get("parse", {}).get("wikitext", {}).get("*", "")
        _cache[key] = _plain(raw)[:SECTION_CHARS]
    return _cache[key]


def run_get_wikipedia(args) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return "Wikipedia lookup failed: no query given."
    try:
        titles = _search(query)
        if not titles:
            return f"No Wikipedia article found for '{query}'."
        page = _summary(titles[0])
    except (requests.RequestException, ValueError) as e:
        return f"Wikipedia lookup failed: {e}"
    others = ", ".join(titles[1:])
    return (f"Wikipedia article \"{page['title']}\" ({page['url']}): {page['extract']}"
            + (f"\nOther matches: {others}." if others else "")
            + "\nAnswer the question from this and say once, in passing, that it is per Wikipedia.")


def run_read_wikipedia(args) -> str:
    title = str(args.get("title") or "").strip()
    wanted = str(args.get("section") or "").strip()
    if not title or not wanted:
        return "Wikipedia lookup failed: need both a title and a section."
    try:
        sections = _sections(title)
        match = next((s for s in sections if s.get("line", "").strip().lower() == wanted.lower()), None)
        if match is None:
            names = ", ".join(s.get("line", "") for s in sections[:12])
            return f"No section '{wanted}' in \"{title}\". Sections: {names}."
        text = _section_text(title, match)
    except (requests.RequestException, ValueError) as e:
        return f"Wikipedia lookup failed: {e}"
    return (f"Wikipedia article \"{title}\", section \"{match['line']}\": {text}"
            "\nAnswer from this and say once, in passing, that it is per Wikipedia.")
