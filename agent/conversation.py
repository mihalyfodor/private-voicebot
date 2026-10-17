"""Conversation state: the turns, the one-turn slots, reset, save to memory, and `new_thread` (docs/39-context-meter.md; PRD 50 step 10, from llm.py)."""
import re
from agent import fold
from memory import identity
import avatars
from core import trace
from core import llm as core_llm
from splitter import strip_tag
from agent import context as a_context
from agent import prompt as a_prompt
from agent import recall as a_recall

log = trace.logger("llm")


_conversation: list = []
_session_turns: list = []
_last_tool_calls: list = []
#: {"source", "title", "parts"} while the last reply ended with a DEEPER_OFFERS phrase,
#: {"wikipedia": question} while it ended with WIKIPEDIA_OFFER, else None
_deeper_offer: dict | None = None
#: (source, title, part) of every passage this turn used: recall hits, search_wiki snippets, read_source
_turn_sources: list = []
#: the `source` names of every document block this session has read (recall, deeper, tool
#: results) since the last successful memory save — the memory proposal is asked to attribute
#: its ops against this list (docs/33-untrusted-text.md)
_sources_seen: list = []
#: the (page, hits) the last recall search found, and the spoken turn's band — the `recall` event (PRD 49)
_evidence: tuple = (None, [])
last_band: str | None = None
#: the wiki / Wikipedia page titles this turn drew on (page recall, the focused page, get_wikipedia)
_turn_pages: list = []
#: what the last answer was built from, for `write_up` (docs/43-written-answers.md):
#: {"question", "extra" (the recall/deeper/focus block), "tools" (tool result texts), "sources", "pages"},
#: plus "card" (ready markdown) after a briefing (docs/46-briefing.md)
_last_context: dict | None = None
#: every card this conversation wrote, by id; `last_card` is the one "save that" means
_cards: dict = {}
last_card: dict | None = None
#: run cards issued but not (yet) a turn of the conversation — a write-up running or failed (PRD 51):
#: id -> how many `dialogue()` lines preceded it, so a reconnect replays it in place
_loose: dict = {}
_card_seq = 0              # every card issued takes the next id; only `reset` starts over


def reset():
    global _conversation, _session_turns, _last_tool_calls, _deeper_offer, _turn_sources, _sources_seen
    global _turn_pages, _last_context, _card_seq
    _conversation = [{"role": "system", "content": identity.load(a_prompt.SYSTEM_PROMPT)}]
    _session_turns = []
    _last_tool_calls = []
    _deeper_offer = None
    _turn_sources = []
    _turn_pages = []
    _sources_seen = []
    _last_context = None
    a_context._last_trimmed = 0
    a_context._context_warned = False
    _forget_cards()
    _card_seq = 0


def _forget_cards() -> None:
    global last_card
    _cards.clear()
    _loose.clear()
    last_card = None


def _rebuild_system_prompt(note: str = "") -> None:
    """Rebuild SYSTEM_PROMPT from the current avatar and swap `_conversation[0]` in place."""

    a_prompt.SYSTEM_PROMPT = a_prompt.build_system_prompt(avatars.current())
    system = {"role": "system", "content": identity.load(a_prompt.SYSTEM_PROMPT) + note}
    if _conversation:
        _conversation[0] = system
    else:
        _conversation.append(system)


def refresh_context() -> None:
    """Re-read the profile and wiki index into the system prompt, keeping the conversation.

    Called after anything that changes the vault mid-session (a drop, an edit, an ingest), so
    a page filed a minute ago is visible to the very next turn instead of the next restart.
    """
    if _conversation:
        _rebuild_system_prompt()


def set_avatar():
    """Swap the persona in place, keeping the conversation history."""
    a = avatars.current()
    note = (f"\n\n(Note: from now on you are {a['name']}. Earlier assistant turns in this conversation "
            f"were spoken by a previous character; continue naturally as {a['name']}.)")
    _rebuild_system_prompt(note if len(_conversation) > 1 else "")


def record_user(text: str) -> None:
    """Append a user turn that took no completion (an approval's request or answer, PRD 52)."""
    _conversation.append({"role": "user", "content": text})
    _session_turns.append({"role": "user", "content": text})


def record_assistant(text: str) -> None:
    """Append a canned (non-LLM-generated) assistant turn to the conversation/session history."""
    _conversation.append({"role": "assistant", "content": text})
    _session_turns.append({"role": "assistant", "content": text})


def get_session_turns() -> list:
    return _session_turns


def get_last_tool_calls() -> list:
    return _last_tool_calls


def has_unsaved_turns() -> bool:
    """True when this session has turns that no save has folded into memory.json yet."""
    return bool(_session_turns)


def save_memory():
    """Fold this session into the memory.json profile. Idempotent: turns are cleared once saved.

    When the save could not reach the model (memory.save returns None, or raises) the turns go
    back in front of anything said meanwhile, so the next save — or the shutdown one — retries.
    """
    global _session_turns, _sources_seen
    if not _session_turns:
        return
    turns, _session_turns = _session_turns, []
    sources, _sources_seen = _sources_seen, []

    def restore():
        global _session_turns, _sources_seen
        _session_turns = turns + _session_turns
        _sources_seen = sources + [s for s in _sources_seen if s not in sources]

    try:
        applied = fold.save(turns, sources=sources)
    except Exception:
        restore()
        raise
    if applied is None:
        restore()


# --- new topic (docs/39-context-meter.md) ---------------------------------------------
SUMMARY_TOKENS = 150
SUMMARY_PROMPT = ("Summarise the conversation above in one short paragraph of plain prose, at most 100 "
                  "words: what we talked about, decisions made, open threads. No lists, no preamble, "
                  "no emotion tag.")
NEW_TOPIC_ACK = "Fresh start — I kept the gist above."
EARLIER_EMPTY = "Nothing was discussed before this point."


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]


def _summarise(turns: list) -> str:
    """One ≤ SUMMARY_TOKENS completion over the turns; on failure the turns' first sentences
    (one error print, never blocks)."""
    if not turns:
        return EARLIER_EMPTY
    try:
        resp = core_llm.completion(
            "summary", model=core_llm.OMLX_MODEL, messages=turns + [{"role": "user", "content": SUMMARY_PROMPT}],
            max_tokens=SUMMARY_TOKENS,
        )
        summary = strip_tag(resp.choices[0].message.content or "").strip()
        if not summary:
            raise ValueError("empty summary")
    except Exception as e:
        log.exception("summary failed; keeping the turns' first sentences")
        trace.error("memory", "new_topic", e, fatal_for="none")
        summary = " ".join(_first_sentence(strip_tag(t["content"])) for t in turns)
    return summary[:SUMMARY_TOKENS * 4].strip()


def dialogue() -> list:
    """The user and assistant lines the model holds, as `{"role", "content"}` copies: tool
    rounds and empty assistant turns left out."""
    return [{"role": m["role"], "content": m["content"]} for m in _conversation[1:]
            if m.get("role") in ("user", "assistant") and m.get("content")]


def new_thread() -> str:
    """Fold memory, then restart the conversation on a short visible summary; returns it.

    `_conversation` becomes [system, `<earlier>summary</earlier>` (user), NEW_TOPIC_ACK]; the
    deeper offer, the turn's sources, the remembered page and the trim count are cleared.
    """
    global _deeper_offer, _turn_sources, _last_context
    turns = dialogue()
    _last_context = None
    try:
        save_memory()
    except Exception as e:
        log.exception("memory save failed")
        trace.error("memory", "memory_save", e, fatal_for="none")
    summary = _summarise(turns)
    _conversation[1:] = [{"role": "user", "content": f"<earlier>{summary}</earlier>"},
                         {"role": "assistant", "content": NEW_TOPIC_ACK}]
    _deeper_offer, _turn_sources, a_recall.last_page = None, [], None
    a_context._last_trimmed, a_context._context_warned = 0, False
    _forget_cards()
    _rebuild_system_prompt()            # the vault map is computed here and at start, not per turn (PRD 54)
    return summary


def get_sources_seen() -> list:
    return _sources_seen


reset()
