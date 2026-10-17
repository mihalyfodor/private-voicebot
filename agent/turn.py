"""One turn: the streamed model call, the tool round, source and offer bookkeeping, `ask_events` / `ask_stream` / `ask` (PRD 50 step 10, from llm.py)."""
import json
from typing import Iterator
from core import trace
from core import llm as core_llm
from tools import registry
from core.untrusted import documents_in
from agent import context as a_context
from agent import offers
from agent import prompt as a_prompt
from agent import recall as a_recall
from agent import conversation as a_conversation
from agent import written as a_written

log = trace.logger("llm")
tool_log = trace.logger("Tool")


run_tool = registry.run_tool
trace.vocab("name", [t.name for t in registry.REGISTRY if t.schema] + ["dig_deeper"])


def _note_documents(text: str) -> None:
    """Record the document blocks in a sent message or tool result for memory attribution."""
    for name in documents_in(text):
        if name not in a_conversation._sources_seen:
            a_conversation._sources_seen.append(name)


def _tool_round(tool_calls: list[dict], content: str = "") -> list:
    """Execute accumulated tool calls and append the assistant + tool messages to the conversation.

    `tool_calls` is a list of plain dicts: {"id", "name", "arguments"} (arguments is a JSON string).
    Returns the tool result texts (the write-up context, PRD 43).
    """
    results = []
    a_conversation._conversation.append({
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"] or "{}"},
            }
            for tc in tool_calls
        ],
    })
    for tc in tool_calls:
        name = tc["name"]
        try:
            args = json.loads(tc["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        a_conversation._last_tool_calls.append({"name": name, "args": args})
        keys = sorted(args) if isinstance(args, dict) else []
        with trace.span("tool_call", name=name, arg_keys=keys, ok=False) as span:
            result = run_tool(name, args)
            span.update(result_len=len(str(result)), ok=True)
        tool_log.info(f"{name}({', '.join(keys)})")          # never the result (PRD 49)
        trace.verbose("tool_result", name=name, args=args, result=str(result))
        offers._note_tool_sources(name, args, str(result))
        _note_documents(str(result))
        results.append(str(result))
        a_conversation._conversation.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
    return results


def _accumulate_tool_calls(pending: dict, fragments) -> None:
    """Merge OpenAI-style streamed tool_call fragments into `pending` keyed by index."""
    for frag in fragments:
        index = getattr(frag, "index", None) or 0
        entry = pending.setdefault(index, {"id": None, "name": None, "arguments": ""})
        if getattr(frag, "id", None):
            entry["id"] = entry["id"] or frag.id
        func = getattr(frag, "function", None)
        if func is None:
            continue
        if getattr(func, "name", None):
            entry["name"] = entry["name"] or func.name
        if getattr(func, "arguments", None):
            entry["arguments"] += func.arguments


def _messages(deeper: bool, written: bool = False) -> list:
    """The conversation as sent: on a deeper turn the system rule reads `long` for this completion;
    on a written turn (typed, grounded, content — PRD 43) the short-answer clause reads full too."""
    if not deeper or not a_conversation._conversation:
        return a_conversation._conversation
    system = a_conversation._conversation[0]["content"].replace(a_prompt.VERBOSITY_RULES[a_prompt._verbosity], a_prompt.VERBOSITY_RULES["long"], 1)
    if written:
        system = system.replace(a_prompt.OFFER_CLAUSE, a_prompt.WRITTEN_CLAUSE, 1)
    return [{"role": "system", "content": system}] + a_conversation._conversation[1:]


def ask_events(user_text: str, origin: str = "user", note: str = "") -> Iterator[tuple[str, object]]:
    """Yield ("tool_calls", [names]) before tools run, then ("delta", text) chunks.

    The conversation and memory see typed and spoken input identically (PRD 24); `origin`
    tags the session log line, sets the answer length (PRD 43) and, on a spoken turn, the
    recall band (PRD 45). `note` is a one-turn system remark (a page was just opened for the
    user, PRD 38) appended to this turn's user message only.
    """


    conv_len = len(a_conversation._conversation)
    a_conversation._evidence, a_conversation.last_band = (None, []), None
    turns_len = len(a_conversation._session_turns)
    # An offer lives exactly one turn: a short "yes" takes it, anything else drops it.
    offer, a_conversation._deeper_offer = a_conversation._deeper_offer, None
    yes = bool(offer) and offers.is_affirmative(user_text)
    lookup = offer.get("wikipedia") if yes else None            # the question to look up (PRD 37)
    asked = offer.get("recall") if yes else None                # the earlier question to check (PRD 45)
    deeper = offers.deeper_block(offer) if yes and not (lookup or asked) else ""
    a_conversation._turn_sources = [(offer["source"], offer["title"], p) for p in offer["parts"]] if deeper else []
    a_conversation._turn_pages = []
    # The recall / source block rides on this turn's user message only; the session turns (what
    # memory and the wiki ingest see) keep the user's own words.
    # A page open in the UI takes the place of recall (PRD 38); Chat and Memory recall as before.
    focused = "" if deeper or lookup or asked else a_recall.focus_block(user_text)
    band, offer_text, recall_offer = "", "", None
    if deeper or lookup or focused:
        extra = deeper or ("\n\n" + offers.WIKIPEDIA_NOTE if lookup else focused)
    elif asked:                                                 # "yes" to the recall offer: forced high
        band, extra = "high", a_recall.recall_for(asked) + "\n\n" + a_recall.RECALL_YES_NOTE.format(question=asked)
    elif origin == "typed":
        extra = a_recall.recall_for(user_text)
    else:
        # A spoken turn asks before it looks up (PRD 45): only a high band quotes, weak evidence
        # gets one short offer, never twice in a row.
        page, hits = a_recall._recall_evidence(user_text)
        band = a_recall.recall_band(user_text, page, hits)
        extra = a_recall._recall_block(page, hits) if band == "high" else ""
        if band != "high":
            a_conversation._turn_pages = []
        if band == "medium" and not (offer or {}).get("recall"):
            extra = "\n\n" + a_recall.RECALL_ASK_NOTE
            recall_offer = {"recall": user_text, "title": a_recall.offer_title(hits[0]["title"])}
            offer_text = a_recall.RECALL_OFFER.format(title=recall_offer["title"])
    # A plain Chat turn may be re-asked once with a server-side lookup when the reply starts by
    # claiming the documents do not cover the question (PRD 42): its first characters are held.
    # A medium turn never is: its offer replaces the fallback.
    holding = not (deeper or lookup or focused or asked or band == "medium") and a_recall.lookup_worthy(user_text)
    if a_recall.focus["view"] == "memory" and not (deeper or lookup):
        extra += "\n\n" + a_recall.MEMORY_FOCUS_NOTE
    if note:
        extra += "\n\n" + note
    # Length by origin (PRD 43): a typed content question answered from filed text runs long
    # and drops the offer; a spoken one keeps the short rule and the offer.
    typed_content = origin == "typed" and a_written.content_question(user_text)
    written = typed_content and bool(extra.strip())
    if written:
        extra = a_written.written_note(extra)
    _note_documents(extra)
    user_msg = {"role": "user", "content": user_text + extra}
    a_conversation._conversation.append(user_msg)
    a_conversation._session_turns.append({"role": "user", "content": user_text})
    a_conversation._last_tool_calls = []
    a_context.trim_history()

    long = bool(deeper or lookup or written)
    tool_texts: list = []

    try:
        if deeper or asked:
            yield ("tool_calls", ["dig_deeper"])               # a filler while the grounded answer starts
        # Streamed first pass: text deltas go out immediately, tool-call fragments accumulate.
        # After a "yes" to the Wikipedia offer the text is held back instead: the pass is there
        # for the model to name the topic, and get_wikipedia runs either way.
        first = core_llm.stream("reply", model=core_llm.OMLX_MODEL, messages=_messages(long, written), tools=a_prompt.TOOLS,
                        max_tokens=a_prompt._budget(long))
        parts: list[str] = []
        pending: dict = {}
        held: list[str] = []                                    # text not yet yielded (fallback hold)
        retry = ""
        for chunk in first:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text and lookup:
                parts.append(text)                              # held back: get_wikipedia runs anyway
            elif text and offer_text:
                parts.append(text)                              # a medium reply: held whole (short)
            elif text and holding:
                parts.append(text)
                held.append(text)
                if len("".join(held)) >= a_recall.FALLBACK_HOLD_CHARS:
                    holding, retry = False, a_recall.fallback_block(user_text, "".join(held))
                    if retry:
                        break
                    for t in held:
                        yield ("delta", t)
                    held = []
            elif text:
                parts.append(text)
                yield ("delta", text)
            fragments = getattr(delta, "tool_calls", None)
            if fragments:
                _accumulate_tool_calls(pending, fragments)
        if holding and not pending:                             # a short reply: decide on all of it
            retry = a_recall.fallback_block(user_text, "".join(held))
            if not retry:
                for t in held:
                    yield ("delta", t)
                held = []

        if lookup and not any(tc["name"] == "get_wikipedia" for tc in pending.values()):
            pending = {0: {"id": "wikipedia_offer", "name": "get_wikipedia",
                           "arguments": json.dumps({"query": lookup})}}
            parts = []                                          # held back, never spoken
        if retry:
            # The held reply is dropped unspoken; the turn is asked once more with the lookup.
            log.info("fallback lookup: the reply claimed the documents do not cover it")
            getattr(first, "close", lambda: None)()             # stop the abandoned generation
            yield ("tool_calls", ["search_wiki"])
            if typed_content:                                   # grounded after all: the policy applies
                written = long = True
                retry = a_written.written_note(retry)
            user_msg["content"] += retry
            _note_documents(retry)
            parts = []
            stream = core_llm.stream("fallback", fallback_reason="not_covered", model=core_llm.OMLX_MODEL,
                             messages=_messages(long, written), max_tokens=a_prompt._budget(long))
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    yield ("delta", delta)
        elif pending:
            tool_calls = [pending[i] for i in sorted(pending)]
            if parts and not lookup and not held and not offer_text:
                log.warning("warning: text before tool call was spoken")
            parts = []  # spoken or held back; never the reply
            yield ("tool_calls", [tc["name"] for tc in tool_calls])
            tool_texts = _tool_round(tool_calls)
            if typed_content and any(tc["name"] in a_written.GROUNDING_TOOLS for tc in tool_calls):
                written = long = True
            stream = core_llm.stream("tool_round", model=core_llm.OMLX_MODEL, messages=_messages(long, written),
                             max_tokens=a_prompt._budget(long))
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    parts.append(delta)
                    yield ("delta", delta)
        if offer_text and not pending:
            # the held medium reply goes out without any offer the model ended it with, then the
            # recall offer: the turn's only one
            text = offers.strip_offer("".join(parts))
            parts = [text, " " + offer_text]
            yield ("delta", text)
            yield ("delta", " " + offer_text)
        reply = "".join(parts)
    except Exception:
        del a_conversation._conversation[conv_len:]
        del a_conversation._session_turns[turns_len:]
        raise

    # The turn is done: the block was for this completion only, the history keeps the bare
    # words (PRD 39) — the reply quotes the source, follow-ups use read_source / dig deeper.
    block = user_msg["content"][len(user_text):]
    user_msg["content"] = user_text
    a_conversation._conversation.append({"role": "assistant", "content": reply})
    a_conversation._session_turns.append({"role": "assistant", "content": reply})
    a_conversation._deeper_offer = recall_offer if offer_text and not pending else offers._offer_from_turn(reply, user_text)
    a_conversation._last_context = {"question": asked or user_text, "extra": block, "tools": tool_texts, "reply": reply,
                     "sources": list(a_conversation._turn_sources), "pages": list(a_conversation._turn_pages)}
    a_conversation.last_band = band or None
    trace.event("recall", band=a_conversation.last_band, page=(a_conversation._evidence[0] or {}).get("path"),
                offer=offers._offer_kind(bool(offer_text and not pending)),
                hits=[{"source": h.get("source"), "n": h.get("n"), "score": round(float(h.get("score") or 0), 3),
                       "named": bool(h.get("named"))} for h in a_conversation._evidence[1][:5]])


def ask_stream(user_text: str) -> Iterator[str]:
    """Yield only the text deltas of the assistant reply."""
    for kind, payload in ask_events(user_text):
        if kind == "delta":
            yield payload


def ask(user_text: str) -> str:
    return "".join(ask_stream(user_text))
