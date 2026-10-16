"""The one model-call chokepoint (docs/50-modular-layout.md): the oMLX client and the three call shapes.

`stream` (the chat turn's streamed completion), `completion` (a raw response) and `complete` (one
plain system+user completion -> its text). Tests patch `_client` here with one fake.
"""
import time

from openai import OpenAI

from core import config, trace

OMLX_BASE_URL = config.OMLX_BASE_URL
OMLX_API_KEY = config.OMLX_API_KEY
OMLX_MODEL = config.OMLX_MODEL
_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY)
    return _client


def stream(purpose: str, fallback_reason: str | None = None, **kw):
    """A streamed completion (usage asked for) whose chunks pass through; one `llm_call` span when
    it ends: ms, ttft_ms (first content or tool delta), tokens from the last chunk's usage or null."""
    with trace.llm_span(purpose, kw.get("messages"), model=kw.get("model"), max_tokens=kw.get("max_tokens"),
                          fallback_reason=fallback_reason) as span:
        t0, ttft, usage = time.monotonic(), None, None
        chunks = client().chat.completions.create(stream=True, stream_options={"include_usage": True}, **kw)
        try:
            for chunk in chunks:
                usage = getattr(chunk, "usage", None) or usage
                if ttft is None and chunk.choices:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "content", None) or getattr(delta, "tool_calls", None):
                        ttft = trace.ms_since(t0)
                yield chunk
        except GeneratorExit:
            getattr(chunks, "close", lambda: None)()        # an abandoned generation stops at the server
            raise
        finally:
            span.update(ms=trace.ms_since(t0), ttft_ms=ttft, **trace.usage_fields(usage))


def completion(purpose: str, **kw):
    """A plain completion with its `llm_call` span (no ttft; tokens when the response has usage)."""
    with trace.llm_span(purpose, kw.get("messages"), model=kw.get("model"), max_tokens=kw.get("max_tokens"),
                          ttft_ms=None) as span:
        resp = client().chat.completions.create(**kw)
        span.update(trace.usage_fields(getattr(resp, "usage", None)))
        return resp


def complete(system: str, user, max_tokens: int, *, purpose: str | None = None, model: str | None = None) -> str:
    """One plain completion -> its text: messages [system, user] (`user` may be the multimodal
    content-part list), temperature 0, model defaulting to OMLX_MODEL. With `purpose` it runs under
    an `llm_call` span (sources' "ingest"); without one it is unspanned (memory, wiki ingest)."""
    kw = {"model": model or OMLX_MODEL,
          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
          "temperature": 0, "max_tokens": max_tokens}
    if purpose is None:
        response = client().chat.completions.create(**kw)
    else:
        response = completion(purpose, **kw)
    return response.choices[0].message.content or ""
