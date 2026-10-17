"""The context budget: model context length, token estimate, history trimming, the usage meter and its 90 % notice (docs/39-context-meter.md; PRD 50 step 9, from llm.py)."""
import json
import os
import time
from core import trace
from core import llm as core_llm
from tools import registry
from agent import conversation as a_conversation
from agent import prompt

log = trace.logger("llm")


CONTEXT_BUDGET = int(os.getenv("CONTEXT_BUDGET", "16000"))
_model_context_len_cache: int | None = None
_model_context_len_warned = False
_model_context_len_failed_at: float | None = None   # a failed lookup is not retried for RETRY_S
_MODEL_CONTEXT_RETRY_S = 60


def model_context_len() -> int | None:
    """Lazily query /v1/models for OMLX_MODEL's max_model_len. Returns None on any failure."""
    global _model_context_len_cache, _model_context_len_warned, _model_context_len_failed_at
    if _model_context_len_cache is not None:
        return _model_context_len_cache
    if _model_context_len_failed_at is not None and time.monotonic() - _model_context_len_failed_at < _MODEL_CONTEXT_RETRY_S:
        return None
    _model_context_len_failed_at = time.monotonic()
    try:
        models = core_llm.client().models.list()
        for m in models:
            if getattr(m, "id", None) == core_llm.OMLX_MODEL:
                max_len = getattr(m, "max_model_len", None)
                if max_len is None:
                    dumped = m.model_dump() if hasattr(m, "model_dump") else {}
                    max_len = dumped.get("max_model_len")
                if max_len:
                    _model_context_len_cache = int(max_len)
                    _model_context_len_failed_at = None
                    return _model_context_len_cache
        return None
    except Exception as e:
        if not _model_context_len_warned:
            _model_context_len_warned = True
            log.warning(f"warning: could not query model context length ({e}); using CONTEXT_BUDGET only")
        return None


def context_budget() -> int:
    model_len = model_context_len()
    if model_len:
        return min(CONTEXT_BUDGET, model_len)
    return CONTEXT_BUDGET


def estimate_tokens(messages: list) -> int:
    """Rough token count: ~4 characters per token, plus ~4 tokens of per-message framing."""
    total = 0
    for m in messages:
        size = len(str(m.get("content") or ""))
        if m.get("tool_calls"):
            size += len(json.dumps(m["tool_calls"]))
        total += size // 4 + 4
    return total


#: `estimate_tokens` counts the messages only; the tool schemas ride on every request too.
#: the 90 % crossing sentence (docs/39-context-meter.md), spoken once per crossing
CONTEXT_WARNING = "I'm close to the end of what I can hold — say new topic to start fresh."
CONTEXT_WARN_PCT, CONTEXT_REARM_PCT = 90, 75
#: user turns the last `trim_history` dropped, and whether the 90 % sentence was said this crossing
_last_trimmed = 0
_context_warned = False


def message_budget() -> int:
    """Tokens left for the messages: the context budget minus the reply reserve and the tool
    schemas. `trim_history` trims to it and `context_usage` measures against the same terms."""
    return context_budget() - prompt._effective_max_tokens() - registry.tokens()


#: `trim_history` keeps the tool results of the last KEEP_TOOL_TURNS user turns; an older one longer
#: than STUB_MIN_CHARS becomes a one-line stub before any turn is dropped (docs/57-context-hygiene.md)
KEEP_TOOL_TURNS, STUB_MIN_CHARS = 2, 400
STUB_TEXT = "[{tool} result removed to save room — call {tool} again if needed]"


def _tool_name(call_id: str) -> str:
    """The tool the assistant message with `call_id` called; "tool" if none matches."""
    for m in a_conversation._conversation:
        for call in m.get("tool_calls") or []:
            if call.get("id") == call_id:
                return call["function"]["name"]
    return "tool"


def stub_old_tool_results() -> int:
    """Replace the long tool results before the second-last user message with a re-call stub;
    returns how many. The assistant `tool_calls` stay, so no pairing is orphaned."""
    conv = a_conversation._conversation
    users = [i for i, m in enumerate(conv) if m.get("role") == "user"]
    if len(users) < KEEP_TOOL_TURNS:
        return 0
    stubbed = 0
    for m in conv[:users[-KEEP_TOOL_TURNS]]:
        if m.get("role") == "tool" and len(str(m.get("content") or "")) > STUB_MIN_CHARS:
            m["content"] = STUB_TEXT.format(tool=_tool_name(m.get("tool_call_id")))
            stubbed += 1
    return stubbed


def trim_history() -> int:
    """Drop the oldest turns until the conversation fits the budget; returns the user turns dropped.

    Index 0 (the system message) and the final turn (normally the latest user message) are
    never dropped. Over budget, old long tool results are stubbed first (a stub is not a dropped
    turn: `_last_trimmed` ignores it). An assistant turn with tool_calls takes its tool results with it, so no
    tool result is ever left orphaned.
    """
    global _last_trimmed
    budget = message_budget()
    if estimate_tokens(a_conversation._conversation) > budget:
        stubbed = stub_old_tool_results()
        if stubbed:
            log.info(f"stubbed {stubbed} tool results")
    dropped = []
    while len(a_conversation._conversation) > 2 and estimate_tokens(a_conversation._conversation) > budget:
        turn = a_conversation._conversation.pop(1)
        dropped.append(turn)
        if turn.get("tool_calls"):
            while len(a_conversation._conversation) > 2 and a_conversation._conversation[1].get("role") == "tool":
                del a_conversation._conversation[1]
    users = sum(1 for t in dropped if t.get("role") == "user")
    _last_trimmed = users or int(bool(dropped))                  # a lone assistant turn still counts
    if dropped:
        log.info(f"trimmed conversation history to fit context budget ({budget} tokens)")
    return _last_trimmed


def context_usage() -> dict:
    """`{used, budget, pct, trimmed}`: what the model holds, the budget, and the user turns the
    last trim forgot (docs/39-context-meter.md)."""
    used = estimate_tokens(a_conversation._conversation) + registry.tokens()
    budget = context_budget()
    return {"used": used, "budget": budget, "pct": round(100 * used / budget), "trimmed": _last_trimmed}


def context_notice() -> str:
    """CONTEXT_WARNING the first time usage reaches 90 %, "" otherwise; re-armed under 75 %."""
    global _context_warned
    pct = context_usage()["pct"]
    if pct >= CONTEXT_WARN_PCT and not _context_warned:
        _context_warned = True
        return CONTEXT_WARNING
    if pct < CONTEXT_REARM_PCT:
        _context_warned = False
    return ""
