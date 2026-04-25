import requests
import xml.etree.ElementTree as ET
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
        if tag in ("p", "li", "h1", "h2", "h3"):
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self):
        return " ".join(" ".join(self._parts).split())


DEFINITION_NEWS = {
    "type": "function",
    "function": {
        "name": "get_news",
        "description": "Returns the latest BBC world news headlines with URLs.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

DEFINITION_NEWS_DETAIL = {
    "type": "function",
    "function": {
        "name": "get_news_detail",
        "description": "Fetches the full text of a BBC news article by URL for a deeper summary.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The BBC article URL to fetch."}
            },
            "required": ["url"],
        },
    },
}


def run_get_news(args):
    resp = requests.get(
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=5,
    )
    root = ET.fromstring(resp.content)
    items = root.findall(".//item")[:5]
    entries = [(item.findtext("title", "").strip(), item.findtext("link", "").strip()) for item in items]
    lines = " | ".join(f"{i+1}. {t} (URL: {u})" for i, (t, u) in enumerate(entries))
    return "Read out these top 5 BBC world news headlines naturally, without mentioning the URLs: " + lines


def run_get_news_detail(args):
    url = args.get("url", "")
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
    parser = _TextExtractor()
    parser.feed(resp.text)
    text = parser.get_text()[:4000]
    return f"Give a detailed spoken summary of this BBC article for a voice listener — cover the key facts, context, and any notable quotes, in around 150 to 200 words: {text}"
