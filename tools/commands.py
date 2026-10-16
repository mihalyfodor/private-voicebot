"""The deterministic turn commands: the trigger regexes the registry routes on, and their predicates.

Not NLU: each is matched on the whole (whitespace-normalised) turn, typed or spoken. The bodies that
run them drive TTS and turn state, so they live in the server and are bound with `registry.handler`.
"""
import re

#: no network: the read-only policy test checks every tools module declares its hosts
HOSTS = ()

# --- find / open pages from chat (docs/38-focus-context.md) ---------------------------
OPEN_PAGE = re.compile(
    r"^(?:please\s+)?(?:find|open|show(?:\s+me)?)\s+(?:the\s+|a\s+)?(?:(?:wiki\s+)?(?:page|note|source)\s+)?"
    r"(?:(?:about|on|for)\s+)?(?P<query>.+?)(?:,?\s+please)?[\s?.!]*$", re.I)
BACK_TO_PAGE = re.compile(
    r"^(?:(?:go|take me)\s+)?back\s+to\s+(?:that|the\s+last|the)\s+page[\s?.!]*$"
    r"|^what\s+were\s+we\s+looking\s+at[\s?.!]*$", re.I)
#: either page phrasing — the `open_page` registry record's trigger
OPEN_PAGE_ANY = re.compile(f"(?:{OPEN_PAGE.pattern})|(?:{BACK_TO_PAGE.pattern})", re.I)

# --- new topic (docs/39-context-meter.md): the exact phrase or the typed `/new` --------
NEW_TOPIC = re.compile(r"^(?:new\s+topic|/new)[\s?.!]*$", re.I)

# --- briefing (docs/46-briefing.md) ---------------------------------------------------
BRIEF = re.compile(
    r"^(?:please,?\s+)?(?:what(?:['’]?s|\s+is)\s+new|anything\s+new|brief\s+me|what\s+(?:has\s+)?changed"
    r"|what\s+did\s+i\s+miss|/brief)(?:,?\s+please)?[\s?.!]*$", re.I)

# --- written answers (docs/43-written-answers.md) --------------------------------------
SAVE_NOTE = re.compile(
    r"^(?:please\s+)?save\s+(?:that|this|it)(?:\s+(?:as|to|in)\s+(?:a\s+|the\s+)?(?:note|wiki|wiki\s+note|note\s+in\s+the\s+wiki))?"
    r"(?:,?\s+please)?[\s?.!]*$", re.I)
#: "write it up", "write that up", "written version", "write it out" — and the typed `/write`
WANTS_WRITEUP = re.compile(r"\bwrite\s+(?:it|that|this)\s+(?:up|out)\b|\bwritten\s+version\b|^/write\s*$", re.I)

# --- approvals (docs/52-approvals.md) ---------------------------------------------------
#: the words a source title is spoken with; "forget <title>" needs one at the end ("forget my address" and
#: "forget Anna" are memory requests for the model, not a source)
SOURCE_KINDS = r"source|guide|notebook|file|book|document|pdf|email|article|notes?"
#: "forget the Morrowind guide", "remove the Falcon status source" — never "forget it/about/to/what…",
#: and never "don't forget…" (anchored at the start)
FORGET = re.compile(
    r"^(?:please\s+)?(?:forget|remove)\s+(?:the\s+)?"
    r"(?!(?:it|that|this|them|everything|about|to|what|me|him|her|you|i|we|my)\b)"
    rf"(?P<title>.+?\s(?:{SOURCE_KINDS}))(?:,?\s+please)?[\s?.!]*$", re.I)
#: "send that to my reMarkable", "push it to the tablet" (PRD 47b's target: the last card)
PUSH = re.compile(
    r"^(?:please\s+)?(?:send|push)\s+(?:this|that|it|the\s+card)\s+to\s+(?:my\s+|the\s+)?(?:remarkable|tablet)"
    r"(?:,?\s+please)?[\s?.!]*$", re.I)


def normalise(text) -> str:
    return " ".join(str(text or "").split())


def page_command(text: str):
    """``("open", query)``, ``("back", None)`` or None for a turn that is not a page command."""
    text = normalise(text)
    if BACK_TO_PAGE.match(text):
        return ("back", None)
    m = OPEN_PAGE.match(text)
    if m and m.group("query"):
        return ("open", m.group("query"))
    return None


def new_topic_command(text: str) -> bool:
    """True for the exact phrase "new topic" (spoken or typed) or the typed `/new` (PRD 39)."""
    return bool(NEW_TOPIC.match(normalise(text)))


def briefing_command(text: str) -> bool:
    """True for "what's new?", "brief me", "anything new", "what changed", "what did I miss", `/brief`."""
    return bool(BRIEF.match(normalise(text)))


def save_note_command(text: str) -> bool:
    """True for "save that as a note" / "save this to the wiki" / "save it" (PRD 43)."""
    return bool(SAVE_NOTE.match(normalise(text)))


def wants_writeup(text: str) -> bool:
    """"write it up", "yes, write that up please", "give me the written version", typed `/write`."""
    return bool(WANTS_WRITEUP.search(normalise(text)))


def forget_command(text: str) -> str | None:
    """The source title a "forget <title>" / "remove the <title> source" turn names, else None (PRD 52)."""
    m = FORGET.match(normalise(text))
    if not m:
        return None
    title = m.group("title").strip()
    return re.sub(r"\s+source$", "", title, flags=re.I)


def push_command(text: str) -> bool:
    """True for "send that to my reMarkable" / "push it to the tablet" (PRD 52, the 47b command)."""
    return bool(PUSH.match(normalise(text)))
