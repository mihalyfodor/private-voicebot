"""Approvals (docs/52-approvals.md): one pending write per session, resolved by the next turn or a card button.

A *kind* (`KINDS`) is a write that waits for the user: `plan(target)` is its read-only proposal,
`apply(proposal)` the write itself, `spoken(proposal)` the one question line. `propose` plans, opens
the run (PRD 51: `run start`, a `propose` step), issues the awaiting card and sets `sess.pending`
(replacing — and expiring — any older one). The next turn decides (`is_approval` / `is_rejection`,
anything else `expire`s it silently); a card button decides through `resolve_pending(..., run_id=)`,
which refuses a run_id that is not the pending one. The approval binds the proposal's hash: the plan
is recomputed before the write, and a different hash writes nothing.

A root module for now; it moves to `agent/approvals.py` at PRD 50 step 9.
"""
import hashlib
import json
import re
import threading
import time
from typing import Callable, NamedTuple

import llm
from ingest import notebooks
import runs
from ingest import pipeline
import wiki
from core import trace
from tools import commands

#: guards every pop / replace of `sess.pending`: the turn worker (propose, resolve) and the event loop
#: (expire on reconnect / disconnect) both write it
_SLOT = threading.Lock()

EXPIRED = "that request expired"
CHANGED = "that changed since I asked — ask again"
NOTHING_TO_SEND = "nothing to send"
REJECTED_LINE = "Left it."
APPROVE = {"id": "approve", "label": "Approve", "kind": "approve"}
REJECT = {"id": "reject", "label": "Reject", "kind": "reject"}
DECISIONS = ("approve", "reject")

#: a yes for any kind: one of these first ("yeah", "sure", "ok" are not enough) ...
_YES_LEADS = frozenset("yes approve approved confirm confirmed".split())
_YES_PHRASES = (("go", "ahead"), ("do", "it"))
#: ... then only these (or the kind's own verb naming the proposal: "yes, delete it")
_YES_TAIL = frozenset("please go ahead do it thanks thank you ok okay".split())
#: a kind's own verbs: they approve only that kind, and only when they name the proposal
KIND_VERBS = {"forget": frozenset("forget delete remove".split()), "push": frozenset("send push".split())}
#: what names the proposal after the verb: "it", "the source" / "the card", or its title
_OBJECTS = {"forget": (("it",), ("the", "source"), ("that", "source")), "push": (("it",), ("the", "card"), ("that", "card"))}
_TRAILING = frozenset("please thanks thank you now".split())
APPROVAL_MAX_WORDS = 12
#: a short explicit no ...
_REJECTION_LEADS = frozenset("no nope nah cancel reject don dont stop leave keep".split())
_REJECTION_WORDS = _REJECTION_LEADS | frozenset("t it that this thanks thank you please".split())
REJECTION_MAX_WORDS = 4
#: ... and these openings are a no whatever follows: "forget it, thanks", "never mind that"
_REJECTION_OPENINGS = (("forget", "it"), ("forget", "that"), ("forget", "this"), ("forget", "about"),
                       ("never", "mind"), ("nevermind",))


def _words(text: str) -> list:
    return re.findall(r"[a-z]+", str(text or "").lower())


def _names_proposal(rest: list, pending: dict) -> bool:
    """Do the words after a kind's verb name the pending proposal — "it", "the source", its title?"""
    kind = pending["kind"]
    if kind == "push" and rest[-3:-1] in (["to", "my"], ["to", "the"]) and rest[-1] in ("remarkable", "tablet"):
        rest = rest[:-3]
    if tuple(rest) in _OBJECTS[kind]:
        return True
    title = _words((pending.get("proposal") or {}).get("title"))
    rest = rest[1:] if rest[:1] == ["the"] else rest
    rest = rest[:-1] if rest[-1:] == [{"forget": "source", "push": "card"}[kind]] and rest[:-1] else rest
    return bool(title) and rest == title


def is_approval(text: str, pending: dict | None = None) -> bool:
    """An explicit yes to `pending`: "yes", "yes please", "go ahead", "confirm", "approve" for any
    kind; the kind's own verb ("delete it", "remove the source", "forget the Morrowind guide"; "send
    it" for a push) only when it names the proposal. "forget it/that/this…", bare "forget" / "delete",
    "yeah sure", "ok" never are."""
    words = _words(text)
    if not words or len(words) > APPROVAL_MAX_WORDS or is_rejection(text):
        return False
    while len(words) > 1 and words[-1] in _TRAILING:
        words = words[:-1]
    verbs = KIND_VERBS.get((pending or {}).get("kind"), frozenset())
    if words[0] in _YES_LEADS or tuple(words[:2]) in _YES_PHRASES:
        rest = words[1:] if words[0] in _YES_LEADS else words[2:]
        if all(w in _YES_TAIL for w in rest):
            return True
        return bool(rest) and rest[0] in verbs and len(rest) > 1 and _names_proposal(rest[1:], pending)
    return words[0] in verbs and len(words) > 1 and _names_proposal(words[1:], pending)


def is_rejection(text: str) -> bool:
    """"no", "no thanks", "nope", "cancel", "don't", "leave it" (≤ 4 words); and anything opening
    "forget it/that/this/about" or "never mind", whatever follows."""
    words = _words(text)
    if any(tuple(words[:len(o)]) == o for o in _REJECTION_OPENINGS):
        return True
    return (0 < len(words) <= REJECTION_MAX_WORDS and words[0] in _REJECTION_LEADS
            and all(w in _REJECTION_WORDS for w in words))


def via_of(origin: str) -> str:
    """The `decision.via` of a turn's origin: a typed turn is `typed`, a spoken one `voice`."""
    return "typed" if origin == "typed" else "voice"


def digest(proposal: dict) -> str:
    """sha256 of the proposal's canonical JSON — what the approval is bound to."""
    return hashlib.sha256(json.dumps(proposal, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _n(count: int, noun: str, plural: str = "") -> str:
    return f"{count} {noun if count == 1 else plural or noun + 's'}"


# ------------------------------------------------------------------ the kinds


class Kind(NamedTuple):
    plan: Callable          # target -> proposal dict, read-only
    apply: Callable         # proposal -> result dict (`commit`: the short sha when it committed, else "")
    spoken: Callable        # proposal -> the one question line
    key: str                # the proposal field naming the target (re-planned before the write)
    title: Callable         # proposal -> the card title
    body: Callable          # proposal -> the card's markdown
    rows: Callable          # proposal -> the run page's Proposal rows ({path?, title?}; no text beyond that)
    detail: Callable        # proposal -> a short fact line (counts) while it waits
    applied: Callable       # (proposal, apply result) -> the same facts once written
    done: str               # the spoken line after the write


def _committed(write) -> str:
    """Run `write()`; the vault's new short HEAD sha when it moved, else ""."""
    before = wiki.head_sha()
    write()
    after = wiki.head_sha()
    return after if after != before else ""


# --- forget (a source, PRD 25/33) -----------------------------------------------------


def _forget_apply(proposal: dict) -> dict:
    """`pipeline.forget` re-checks the approved plan under the vault lock (no window after the hash check)."""
    out = {}
    out["commit"] = _committed(lambda: out.update(pipeline.forget(proposal["sha"], expected=proposal)))
    return out


def _images(p: dict) -> str:
    return _n(len(p.get("images") or []), "page image")


def _forget_spoken(p: dict) -> str:
    parts = [p["title"] or "the source", "its raw file"]
    if p.get("images"):
        parts.append(f"its {_images(p)}")
    if p["cleaned"]:
        parts.append(f"its notes on {_n(len(p['cleaned']), 'page')}")
    if p["memory"]:
        parts.append(_n(len(p["memory"]), "memory entry", "memory entries"))
    return f"That removes {', '.join(parts[:-1])} and {parts[-1]}. Go ahead?"


def _forget_body(p: dict) -> str:
    lines = [f"Forget the source **{p['title']}**? This removes:", "",
             f"- the source page `{p['page']}`", f"- the raw file `{p['raw']}`"]
    if p.get("images"):
        lines.append(f"- its {_images(p)}")
    if p["cleaned"]:
        lines.append("- its notes on " + ", ".join(f"`{c}`" for c in p["cleaned"]))
    if p["memory"]:
        lines.append(f"- {_n(len(p['memory']), 'memory entry', 'memory entries')}:")
        lines += [f"  - {entry}" for entry in p["memory"]]
    return "\n".join(lines)


def _forget_facts(p: dict, cleaned: str, removed: str) -> str:
    facts = ([_images(p)] if p.get("images") else []) + [
        f"{_n(len(p['cleaned']), 'page')} {cleaned}",
        _n(len(p["memory"]), "memory entry", "memory entries") + removed]
    return ", ".join(facts)


FORGET = Kind(
    plan=lambda sha: pipeline.forget_plan(sha), apply=_forget_apply, spoken=_forget_spoken, key="sha",
    title=lambda p: f"Forget {p['title']}", body=_forget_body,
    rows=lambda p: ([{"path": p["page"], "title": p["title"]}, {"path": p["raw"]}]
                    + [{"path": i} for i in p.get("images") or []] + [{"path": c} for c in p["cleaned"]]),
    detail=lambda p: _forget_facts(p, "to clean", ""),
    applied=lambda p, r: _forget_facts(p, "cleaned", " removed"),
    done="Forgotten.")


# --- push (a card to the reMarkable, PRD 47b) -----------------------------------------


def _card(card_id: str) -> dict:
    card = llm.get_card(card_id)
    if not card or not card.get("markdown"):
        raise ValueError(NOTHING_TO_SEND)
    return card


NOT_AVAILABLE = "The reMarkable push service is not configured."


def _push_apply(proposal: dict) -> dict:
    result = notebooks.push(_card(proposal["card_id"]))
    return {**(result if isinstance(result, dict) else {}), "commit": ""}


PUSH = Kind(
    plan=lambda card_id: notebooks.push_plan(_card(card_id)), apply=_push_apply,
    spoken=lambda p: f"Send the card {p['title']} (about {_n(p['pages'], 'page')}) to your reMarkable?",
    key="card_id", title=lambda p: f"Send {p['title']}",
    body=lambda p: f"Send the card **{p['title']}** (~{_n(p['pages'], 'page')}) to your reMarkable?",
    rows=lambda p: [{"title": p["title"]}], detail=lambda p: f"~{_n(p['pages'], 'page')}",
    applied=lambda p, r: (f"uploaded '{p['title']}', {_n(r['pages'], 'page')}" if r.get("pages")
                          else f"sent, ~{_n(p['pages'], 'page')}"), done="Sent to your reMarkable.")

#: kind -> its plan / apply / spoken (+ the card and run-page shapes); PRD 47b's push is one entry
KINDS: dict = {"forget": FORGET, "push": PUSH}


# ------------------------------------------------------------------ the slot


def _end(pending: dict, decision: str, via: str, status: str, reason: str) -> None:
    """The `decision` event and the run's end, both under the proposing turn's run (no text). The
    approval kind rides as `workflow`: an event's `kind` is its envelope's."""
    with trace.run(pending["run_id"]):
        trace.event("decision", workflow=pending["kind"], decision=decision, via=via)
        trace.run_event(pending["kind"], "end", status, reason)


def propose(sess, kind: str, target: str) -> dict:
    """Plan `kind` for `target` (raises ValueError / `pipeline.IngestError` before anything is opened),
    then open its run, issue the awaiting card (`llm`'s card store, so a reconnect replays it) and set
    `sess.pending` — an older pending is expired first. Returns the pending dict."""
    spec = KINDS[kind]
    expire(sess, "button")                 # the older one goes first: a failed plan leaves nothing pending
    proposal = spec.plan(target)
    run = runs.Run(kind)
    with trace.span("step", name="propose", proposal=spec.rows(proposal), detail=spec.detail(proposal),
                    count=len(spec.rows(proposal))):
        pass
    run.step("proposed", "ok", spec.detail(proposal))
    card = llm.issue_card({"id": llm.next_card_id(), "markdown": spec.body(proposal), "sources": [],
                           "title": spec.title(proposal), **run.card([APPROVE, REJECT, runs.OPEN]),
                           "status": "awaiting"})
    pending = {"run_id": run.run_id, "kind": kind, "proposal": proposal, "hash": digest(proposal),
               "created": time.time(), "card_id": card["id"], "steps": list(run.steps)}
    with _SLOT:
        older, sess.pending = sess.pending, pending
    if older is not None:                  # one slipped in since the expire above
        _expired(older, "button")
    return pending


def expire(sess, via: str) -> dict | None:
    """The one-turn pop's silent branch: drop the pending approval (a `decision expired`); the old
    card keeps its buttons, and a later click on them is "that request expired"."""
    with _SLOT:
        pending, sess.pending = getattr(sess, "pending", None), None
    if pending is not None:
        _expired(pending, via)
    return pending


def _expired(pending: dict, via: str) -> None:
    """End an emptied slot's run as expired and fold its card (a replay shows it folded)."""
    _end(pending, "expired", via, "expired", "expired")
    card = llm.get_card(pending.get("card_id"))
    if card is not None and card.get("status") == "awaiting":
        card.update(status="expired", actions=[runs.OPEN])


class Outcome(NamedTuple):
    decision: str            # approve / reject / changed / expired
    pending: dict | None     # the approval decided (None when the run_id was not the pending one)
    error: str = ""          # the `error {scope: turn}` text, if any
    line: str = ""           # the one spoken line
    result: dict | None = None


def resolve_pending(sess, decision: str, via: str, run_id: str | None = None) -> Outcome:
    """Decide the pending approval. `run_id` given (a card button) and not the pending one, or nothing
    pending → "that request expired", `sess.pending` untouched. Else the slot is emptied; reject →
    "Left it."; approve recomputes the plan, a different hash (or a target gone) → "changed", nothing
    written; else the write (its one commit) in an `apply` step under the proposing run."""
    with _SLOT:
        pending = sess.pending
        if pending is None or (run_id is not None and run_id != pending["run_id"]):
            return Outcome("expired", None, error=EXPIRED)
        sess.pending = None
    spec = KINDS[pending["kind"]]
    if decision != "approve":
        _end(pending, "reject", via, "rejected", "rejected")
        return Outcome("reject", pending, line=REJECTED_LINE)
    try:
        fresh = spec.plan(pending["proposal"][spec.key])
    except (ValueError, pipeline.IngestError):
        fresh = None
    if fresh is None or digest(fresh) != pending["hash"]:
        _end(pending, "changed", via, "failed", "changed")
        return Outcome("changed", pending, error=CHANGED)
    with trace.run(pending["run_id"]):
        try:
            with trace.span("step", name="apply") as step:
                result = spec.apply(fresh)
                if result.get("commit"):
                    step.update(commit=result["commit"],
                                pages=[{**row, "commit": result["commit"]} for row in spec.rows(fresh) if row.get("path")])
        except pipeline.IngestError as e:
            if e.status != pipeline.CHANGED_STATUS:
                _end(pending, "approve", via, "failed", runs.failure_reason(e))
                raise
            _end(pending, "changed", via, "failed", "changed")          # changed under the lock: nothing written
            return Outcome("changed", pending, error=CHANGED)
        except Exception as e:
            _end(pending, "approve", via, "failed", runs.failure_reason(e))   # the decision stands, the write failed
            raise
    _end(pending, "approve", via, "done", "ok")
    return Outcome("approve", pending, line=spec.done, result=result)


# ------------------------------------------------------------------ spoken targets


def _norm(text: str) -> str:
    words = _words(text)
    return " ".join(words[1:] if words[:1] == ["the"] else words)


def _queries(title: str) -> list:
    """The spoken title, then without its kind word ("the Falcon status notebook" → "falcon status")."""
    query = _norm(title)
    words = query.split()
    kinds = commands.SOURCE_KINDS.split("|")
    tail = len(words) > 1 and any(re.fullmatch(k, words[-1]) for k in kinds)
    return [query, " ".join(words[:-1])] if tail else [query]


def forget_target(title: str) -> tuple:
    """``(sha, None)`` for the one source whose title the spoken `title` names, else ``(None, line)``:
    "Which one: A or B?" for several, "I have no source called <title>." for none. An exact title
    wins over the titles that merely contain every word."""
    rows = [s for s in pipeline.list_sources() if s.get("sha")]
    hits = []
    for query in _queries(title):
        exact = [s for s in rows if _norm(s["title"]) == query]
        hits = exact or [s for s in rows if query and set(query.split()) <= set(_norm(s["title"]).split())]
        if hits:
            break
    if len(hits) == 1:
        return hits[0]["sha"], None
    if not hits:
        return None, f"I have no source called {title}."
    names = [s["title"] for s in hits]
    return None, f"Which one: {', '.join(names[:-1])} or {names[-1]}?"
