"""Who the user is, in one place (docs/26-identity-aliases.md): aliases, `is_user`, the ASR prompt and the system-prompt block (PRD 50 step 8, from memory.py)."""
import os
import unicodedata
from core import trace
from memory import ops as m_ops
from memory import render as m_render
from memory import store as m_store
from core import util

log = trace.logger("Memory")


USER_RATIO = 0.8            # is_user() threshold — looser than DUP_RATIO: ASR/OCR mangle names

USER_MIN_TOKEN = 3          # a token shorter than this never matches on its own

ASR_PROMPT_TOKENS = 40      # ceiling for the whisper initial_prompt built from the profile



# --------------------------------------------------------------------------- identity

# Who the user is, in one place (docs/26-identity-aliases.md). "Is this the user?" is asked all
# over the app — by the extraction guard that refuses to overwrite the real name with an alias,
# by the wiki ingest that must not open a people page about the profile owner, by the source
# page's entity list and by the ASR prompt — so the alias list, the matching, the ops that
# maintain the list and the prompt lines that describe it all live together here.


def _is_alias(section: str, key: str) -> bool:
    """True for the one path whose value is a list rather than a string: identity.aliases."""
    return section == "identity" and key == m_store.ALIAS_KEY



def alias_list(value) -> list:
    """Aliases as a clean list of strings.

    Accepts a list (the stored form) or a comma-separated string, which is what the memory
    editor's scalar row sends and what a model tends to propose. Blanks are dropped and
    duplicates removed case-insensitively, first spelling wins.
    """
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return []
    out = []
    for part in parts:
        text = " ".join(str(part).split())
        if text and not any(m_ops._norm(text) == m_ops._norm(kept) for kept in out):
            out.append(text)
    return out



def user_names(profile: dict | None = None) -> list:
    """Every string that means "the user": name, nickname and aliases, in that order.

    `alias_list` does the tidying — blanks dropped, duplicates removed case-insensitively.
    """
    identity = (m_store.load_profile() if profile is None else profile).get("identity") or {}
    return alias_list([identity.get("name") or "", identity.get("nickname") or "",
                       *alias_list(identity.get(m_store.ALIAS_KEY))])



def people_names(profile: dict | None = None) -> list:
    """The names of the people in the profile (live entries only) — who the user actually knows.

    The source ingest uses it to tell a colleague from a book's author (PRD 31 addendum).
    """
    entries = m_render._live((m_store.load_profile() if profile is None else profile).get("people") or [])
    return alias_list([entry.get("name") or "" for entry in entries])



def _fold(text) -> str:
    """Lowercased, diacritic-free, punctuation-free — "Mihály" and "mihaly" are one string.

    `_norm` with the accents stripped first and runs of spaces collapsed, so folded strings can
    be compared word by word.
    """
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(m_ops._norm(plain).split())



def _long_tokens(text: str) -> list:
    """The words of a folded name that could identify someone on their own."""
    return [t for t in text.split() if len(t) >= USER_MIN_TOKEN]



def _shares_a_name(target: str, names: list) -> bool:
    """A bare "Misi" against "Misi Fodor", or "Mihaly Fodor" against the bare alias "Mihaly".

    One side must be a single word: two full names that merely share a surname are two people.
    """
    for name in names:
        if " " not in target and target in _long_tokens(name):
            return True
        if " " not in name and name in _long_tokens(target):
            return True
    return False



def _almost_a_name(target: str, names: list) -> bool:
    """The ASR/OCR case — "Mihay Fodor" for "Mihaly Fodor".

    The project's one fuzzy rule (`_similar`) at the looser USER_RATIO: the whole string first,
    then token against token, so a single mangled word is enough.
    """
    for name in names:
        if m_ops._similar(target, name, USER_RATIO):
            return True
        if any(m_ops._similar(token, name_token, USER_RATIO)
               for name_token in _long_tokens(name) for token in _long_tokens(target)):
            return True
    return False



def is_user(text, names: list | None = None) -> bool:
    """True when `text` names the profile owner: the same name, one of its words, or a
    near-miss spelling of either — tried in that order.

    Deliberately generous, and only for the user: an OCR'd "Mihay Fodor", a whisper "Michael"
    and a bare "Misi" must all be recognised, because the cost of missing one is a duplicate
    people page (or an overwritten real name). Other people are matched exactly, as before.
    """
    target = _fold(text)
    if not target:
        return False
    folded = [f for f in (_fold(n) for n in (user_names() if names is None else names)) if f]
    return (target in folded
            or _shares_a_name(target, folded)
            or _almost_a_name(target, folded))



#: The "who is the user" line every ingest prompt carries so the model never opens a people page
#: about the profile owner. `%s` is the comma-separated list of names.
USER_NOTE = ("The user (profile owner) is: %s; treat any of these as the user, never create a "
             "people page for them.")

NO_USER_NOTE = "Never create a people page for the user themselves."



def user_note(names=None) -> str:
    """The "who is the user" line for an ingest prompt, whatever the profile knows."""
    names = [n for n in (user_names() if names is None else names) if str(n).strip()]
    return USER_NOTE % ", ".join(names) if names else NO_USER_NOTE



# --- alias ops (applied from _apply_scalar_op) --------------------------------------


def _apply_alias_op(identity: dict, kind: str, value, op) -> bool:
    """`identity.aliases` — the one list-valued scalar key.

    ADD appends one alias (or a comma-separated few), UPDATE replaces the whole list and DELETE
    clears it. No `superseded` history: the list is the history, and restoring one old spelling
    out of a replaced list would be meaningless.
    """
    current = alias_list(identity.get(m_store.ALIAS_KEY))
    if kind == "DELETE":
        if m_store.ALIAS_KEY not in identity:
            m_ops._reject(op, "key not present")
            return False
        del identity[m_store.ALIAS_KEY]
        return True
    incoming = alias_list(value)
    if not incoming:
        m_ops._reject(op, f"{m_store.ALIAS_PATH} needs one alias or a comma-separated list")
        return False
    wanted = alias_list(current + incoming) if kind == "ADD" else incoming
    if [m_ops._norm(a) for a in wanted] == [m_ops._norm(a) for a in current]:
        return False        # nothing new (duplicates are ignored case-insensitively)
    identity[m_store.ALIAS_KEY] = wanted
    return True



def _alias_guard(identity: dict, value, op) -> bool:
    """True when this write to `identity.name` would replace the real name with an alias.

    The classic failure: whisper hears "Michael", the extractor proposes
    `UPDATE identity.name "Michael"` and the real name is gone. An alias is a name for the
    user, never *the* name — unless we have no name at all yet.
    """
    current = " ".join(str(identity.get("name") or "").split())
    if not current:
        return False        # nothing to protect yet
    names = user_names({"identity": identity})
    return is_user(value, names) and not is_user(value, [current])



# --- asr biasing --------------------------------------------------------------------

#: The last built ASR prompt, keyed on (path, mtime) — every save/edit/reset rewrites the file.
_ASR_CACHE = {"key": None, "text": ""}



def _asr_prompt_text(profile: dict) -> str:
    """`Names: …` — the user's names plus the people they talk about, capped at ~40 tokens."""
    people = [entry.get("name") or "" for entry in m_render._live(profile.get("people", []))]
    names = alias_list(user_names(profile) + people)
    while names and util.tokens_est("Names: " + ", ".join(names) + ".") > ASR_PROMPT_TOKENS:
        names.pop()
    return "Names: " + ", ".join(names) + "." if names else ""



def asr_prompt(profile: dict | None = None) -> str:
    """The whisper `initial_prompt` biasing the transcript towards names it keeps mangling.

    Cached: the chat path asks for it on every utterance, and the profile only changes when
    :func:`save` / :func:`edit` / :func:`reset` rewrite memory.json (which is what the cache
    key notices).
    """
    if profile is not None:
        return _asr_prompt_text(profile)
    try:
        key = (m_store.MEMORY_PATH, os.path.getmtime(m_store.MEMORY_PATH))
    except OSError:
        key = (m_store.MEMORY_PATH, None)
    if _ASR_CACHE["key"] != key:
        _ASR_CACHE.update(key=key, text=_asr_prompt_text(m_store.load_profile()))
    return _ASR_CACHE["text"]



WIKI_NOTE = (
    "Those pages can be opened with the read_page tool; use search_wiki when the user asks about "
    "past work, meetings, decisions, someone they mentioned before, or anything they filed — "
    "documents, guides, emails and notes live under sources/. search_wiki also returns passages "
    "from inside filed documents as '<title> · part n/N'; read_source(path, part) continues "
    "reading one. If the index lists a page on the topic, read it before answering; never guess "
    "or deny having it."
)



def _wiki_block() -> str:
    """The `<wiki_index>` block, or "" when there is no vault / no pages (docs/19-wiki-layer.md).

    The tool hint is only appended when the recall tools are actually registered for this run:
    the eval harness runs tool-free, and telling a tool-less model to "use search_wiki" just
    makes it type a tool call at the user (PRD 19, Results).

    Imported lazily and never allowed to raise: the wiki is an add-on to the prompt, and a
    broken vault must not be able to stop a conversation from starting.
    """
    try:
        from vault import focus
        from vault import index as v_index
        from vault import store as v_store

        index = v_index.vault_map(v_store.WIKI_INDEX_TOKENS)
        note = f"\n{WIKI_NOTE}" if focus.tools_registered() else ""
    except Exception as exc:  # noqa: BLE001
        log.warning(f"warning: wiki index unavailable ({exc})")
        return ""
    if not index.strip():
        return ""
    return f"\n\n<wiki_index>\n{index}\n</wiki_index>{note}"



def load(system_prompt: str) -> str:
    """`system_prompt` with the `<user_profile>` and `<wiki_index>` blocks appended.

    The one entry point the chat path uses to read memory. Either block is omitted when it
    would be empty, so a fresh install gets exactly the prompt it had before memory existed.
    """
    profile = m_store.load_profile()
    prompt = system_prompt
    block = m_render.render(profile, m_render.BUDGET_TOKENS) if m_store.has_content(profile) else ""
    if block.strip():
        prompt = (
            f"{system_prompt}\n\n"
            f"<user_profile>\n{block}\n</user_profile>\n"
            f"{m_store.PROFILE_NOTE}"
        )
    return prompt + _wiki_block()
