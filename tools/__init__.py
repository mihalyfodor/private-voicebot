from tools import time as _time
from tools import weather as _weather
from tools import gmail as _gmail
from tools import news as _news

TOOLS = [
    _time.DEFINITION,
    _news.DEFINITION_NEWS,
    _news.DEFINITION_NEWS_DETAIL,
    _gmail.DEFINITION,
    _weather.DEFINITION,
]

_DISPATCH = {
    "get_time":        _time.run,
    "get_news":        _news.run_get_news,
    "get_news_detail": _news.run_get_news_detail,
    "get_weather":     _weather.run,
    "get_emails":      _gmail.run,
}


def run_tool(name, args):
    fn = _DISPATCH.get(name)
    return fn(args) if fn else "unknown tool"
