from tools import time as _time
from tools import weather as _weather
from tools import gmail as _gmail
from tools import news as _news
from tools import wiki as _wiki
from tools import sources as _sources
from tools import wikipedia as _wikipedia

#: every host any tool may talk to — the read-only policy test checks each module against it
ALLOWED_HOSTS = frozenset(_wikipedia.HOSTS + _weather.HOSTS + _news.HOSTS + _gmail.HOSTS)
#: the tools whose result is free text somebody else wrote; each wraps it in document tags
READ_TOOLS = ("search_wiki", "read_page", "read_source", "get_emails", "get_news", "get_news_detail",
              "get_wikipedia", "read_wikipedia")

TOOLS = [
    _time.DEFINITION,
    _news.DEFINITION_NEWS,
    _news.DEFINITION_NEWS_DETAIL,
    _gmail.DEFINITION,
    _weather.DEFINITION,
    _wiki.DEFINITION_SEARCH,
    _wiki.DEFINITION_READ,
    _wiki.DEFINITION_READ_SOURCE,
    _sources.DEFINITION,
    _wikipedia.DEFINITION_GET,
    _wikipedia.DEFINITION_READ,
]

_DISPATCH = {
    "get_time":        _time.run,
    "get_news":        _news.run_get_news,
    "get_news_detail": _news.run_get_news_detail,
    "get_weather":     _weather.run,
    "get_emails":      _gmail.run,
    "search_wiki":     _wiki.run_search,
    "read_page":       _wiki.run_read,
    "read_source":     _wiki.run_read_source,
    "file_source":     _sources.run,
    "get_wikipedia":   _wikipedia.run_get_wikipedia,
    "read_wikipedia":  _wikipedia.run_read_wikipedia,
}


def run_tool(name, args):
    fn = _DISPATCH.get(name)
    return fn(args) if fn else "unknown tool"
