"""Turns: respond routing, the trigger-tool bodies (briefing, write-up, research, save note, forget, push, new topic, open page), approvals, and the typed / spoken / greeting turn handlers (PRD 50 step 11, from chatbot.py)."""
from datetime import datetime, timedelta
from core import trace
import time
from agent import approvals
from agent import conversation as a_conversation
from agent import turn as a_turn
from agent import written as a_written
from agent import context as a_context
from agent import offers
from agent import recall
from memory import identity
from ingest import notebooks
from ingest import remarkable
from agent import research
from agent import runs
from agent import digest as a_digest
from vault import focus as v_focus
from vault import pages as v_pages
from vault import search as v_search
from vault import store as v_store
from ingest import pipeline
from core import settings
from server import services
from tools import commands, registry
from tools import time as t_time
from tools.registry import Ctx
from server import asr
from server.session import Session
from server import panels
from server.actions import sources as s_sources
from server import speech as s_speech
from core import util

log = trace.logger("App")


SAMPLERATE = 16000


# ---------------------------------------------------------------- turns


def _arm_playback(sess: Session, turn: int) -> None:
    """Start waiting for `turn`'s playback echo; call before the first speech is sent."""
    sess.playback_done.clear()
    sess.expected_turn = turn


def _wait_for_playback_then_idle(sess: Session, send_sync):
    """Common tail: wait for the browser to report playback_done (or time out), then go idle."""
    if sess.websocket is not None:
        # Fallback timeout: generous, since the browser normally reports playback_done.
        sess.playback_done.wait(timeout=120)
    send_sync({"type": "state", "value": "idle"})


# --- the trigger tools' bodies (PRD 50 step 3) ------------------------------------------
# Their regexes, order and flags live in tools/registry.py (+ tools/commands.py); the bodies drive TTS
# and turn state, so they stay here and are bound with `@registry.handler(name)`.

# --- briefing (docs/46-briefing.md) ---------------------------------------------------
BRIEFING_HOURS = 24          # the look-back of the very first briefing


@registry.handler("briefing")
def briefing(ctx: Ctx):
    """The briefing turn: "Nothing new since <when>." without the model, else one spoken recap;
    `briefing_at` moves forward only once the recap was spoken. A run (PRD 51): steps gather /
    skip / phrase, one run card (empty body) with the final status."""
    sess, turn, user_text, send_sync, origin = ctx.session, ctx.turn, ctx.text, ctx.send, ctx.origin
    run, stage = None, "gathered"
    try:
        now = util.stamp("%H:%M:%S")
        since = settings.load_settings().get("briefing_at") or \
            (datetime.fromisoformat(now) - timedelta(hours=BRIEFING_HOURS)).isoformat(timespec="seconds")
        run = runs.Run("briefing")
        digest = _gather(run, since)
        stage = "phrased"
        empty = a_digest.digest_empty(digest)
        if not empty and not services.llm_ok():
            _run_failed(run, "phrased", "failed-offline", send_sync)
            send_sync({"type": "error", "scope": "turn", "text": panels.llm_offline_text()})
            send_sync({"type": "state", "value": "idle"})
            return
        send_sync({"type": "state", "value": "thinking"})
        _arm_playback(sess, turn)
        if empty:
            say_canned(f"Nothing new since {a_digest.briefing_when(since)}.", send_sync)
            run.end("nothing", "nothing-to-do")
        else:
            t0 = time.monotonic()
            with trace.span("step", name="phrase"):
                reply, _ = a_written.briefing(digest, user_text, origin)
            run.step("phrased", "ok", f"{time.monotonic() - t0:.1f} s")
            log.debug(f"Bot: {reply}")
            trace.verbose("turn_text", user=user_text, reply=reply)
            say_canned(reply, send_sync)
            settings.save_setting("briefing_at", max(str(settings.load_settings().get("briefing_at") or ""), now))
            run.end("done", "ok")
        send_sync(_run_card(run))
        send_sync(context_msg())
        _wait_for_playback_then_idle(sess, send_sync)
    except Exception as e:
        _run_failed(run, stage, runs.failure_reason(e), send_sync)
        panels._turn_failed(send_sync, "briefing", e, "The briefing failed.", recheck=True)


def _gather(run, since: str) -> dict:
    """The briefing's `gather` step (+ a `skip` step for the housekeeping commits it left out):
    the digest, with its counts and the pages' commits on the event — never a title's text beyond that."""
    with trace.span("step", name="gather", detail=f"since {since[11:16]}") as step:
        digest = a_digest.changes_since(since)
        pages = len(digest.get("pages") or []) + len(digest.get("chat_pages") or [])
        n_sources = len(digest.get("sources") or [])
        mem = digest.get("memory") or {}
        n_memory = sum(int(mem.get(k) or 0) for k in ("added", "updated", "removed"))
        step.update(count=pages + n_sources + n_memory,
                    pages=[{"path": p.get("path"), "commit": p.get("commit"), "title": p.get("title")}
                           for p in digest.get("pages") or []
                           if isinstance(p, dict)],
                    memory={k: int(mem.get(k) or 0) for k in ("added", "updated", "removed")})
    run.step("gathered", "ok", f"since {a_digest.briefing_when(since)} — {_n(pages, 'page')}, {_n(n_sources, 'source')}, "
                               f"{_n(n_memory, 'memory change')}")
    skipped = int(digest.get("skipped") or 0)
    if skipped:
        detail = _n(skipped, "housekeeping commit")
        with trace.span("step", name="skip", count=skipped, detail=detail, reason="housekeeping") as step:
            step["status"] = "skipped"
        run.step("skipped", "skipped", detail)
    return digest


def _n(count: int, noun: str) -> str:
    """"1 page", "3 pages"."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _run_card(run) -> dict:
    """A run card with an empty body (the briefing): header, steps and Open run (PRD 51)."""
    return card_msg({"id": f"run-{run.run_id}", "markdown": "", "sources": [],
                     "title": runs.WORKFLOWS[run.workflow], **run.card([runs.OPEN])})


def _run_failed(run, step: str, reason: str, send_sync, card=None) -> None:
    """End a still-running `run` failed at `step`, then send its card: `card_update` on the card that
    already went out (it keeps the failed status for a replay), else a fresh run card. No-op otherwise."""
    if run is None or run.status != "running":
        return
    run.fail(step, reason)
    if card is None:
        send_sync(_run_card(run))
    else:
        card.update(run.card([runs.OPEN]))
        send_sync(card_update(card))


def context_msg() -> dict:
    return {"type": "context", **a_context.context_usage()}


# --- written answers (docs/43-written-answers.md) --------------------------------------
# "write it up" / `/write` turn the last answer into a card; "save that as a note" files the
# last card. Both are deterministic, matched on the whole turn, typed or spoken.
NOTHING_TO_SAVE = "nothing to save"
ALREADY_SAVED = "already saved"
DEFAULT_CARD_FOLDER = "notes"


#: a run card's optional fields (docs/51-run-cards.md); a written card without them renders as PRD 43
CARD_RUN_FIELDS = ("run_id", "workflow", "status", "steps", "actions", "run_url")


def card_msg(card: dict) -> dict:
    """`card` message: a card's body plus its run fields (if any)."""
    return {"type": "card", "id": card["id"], "markdown": card["markdown"], "sources": card["sources"],
            "title": card["title"], **{k: card[k] for k in CARD_RUN_FIELDS if k in card}}


def card_update(card: dict, **extra) -> dict:
    """`card_update`: a run card's current status, steps and actions (+ `markdown`, `title`, `sources`)."""
    return {"type": "card_update", "id": card["id"], "status": card["status"], "steps": card["steps"],
            "actions": card["actions"], **extra}


def card_path(card: dict, folder: str = "") -> str:
    """`notes/<date>-<slug(title)>` (or the folder given): the page a saved card becomes."""
    folder = str(folder or "").strip().strip("/") or DEFAULT_CARD_FOLDER
    return f"{folder}/{util.today_iso()}-{v_store.slugify(card['title'])[:60]}"


def save_card(card_id: str | None, folder: str = "") -> list:
    """File a card as a note and return the messages to send (`wiki_open`, `card_saved`, the
    refreshed `wiki` + `sources`); raises ValueError with the refusal reason (scope wiki)."""
    card = a_written.get_card(card_id) if card_id else a_conversation.last_card
    if not card or not card.get("markdown"):            # a running / failed run card has no body (PRD 51)
        raise ValueError(NOTHING_TO_SAVE)
    if card.get("saved"):
        raise ValueError(ALREADY_SAVED)
    markdown, run_id = card["markdown"], ""
    if card.get("workflow") == "research":            # PRD 54: the run in frontmatter, the dropped quotes stay out
        markdown, run_id = research.saved_markdown(markdown), card.get("run_id", "")
    path, reason = v_pages.save_card(card_path(card, folder), card["title"], markdown, card["sources"], run_id)
    if reason:
        raise ValueError(reason)
    card["saved"] = path
    a_conversation.refresh_context()
    recall.set_focus("wiki", path)
    return [{"type": "wiki_open", "path": path}, {"type": "card_saved", "id": card["id"], "path": path},
            panels._wiki_msg(), s_sources._sources_msg()]


@registry.handler("write_up")
def write_up(ctx: Ctx):
    """The write-up turn: one spoken line, then the card's body; the body is never spoken. A run
    (PRD 51): the card goes out `running` first (kept in `agent.written`'s card store, so a reconnect replays
    it), then `card_update` done (with the markdown, title and sources) or failed."""
    sess, turn, user_text, send_sync = ctx.session, ctx.turn, ctx.text, ctx.send
    run = card = None
    try:
        send_sync({"type": "state", "value": "thinking"})
        _arm_playback(sess, turn)
        researched = a_written.research_card()
        if researched is not None:                  # PRD 54: the research card is the write-up; no second card
            a_written.reuse_card(researched, user_text)
            send_sync(card_update(researched, markdown=researched["markdown"], title=researched["title"],
                                  sources=researched["sources"]))
            a_conversation.record_assistant(say_canned(a_written.WRITTEN_LINE, send_sync))
            send_sync(context_msg())
            _wait_for_playback_then_idle(sess, send_sync)
            return
        if not a_written.can_write_up():
            raise ValueError(a_written.NOTHING_TO_WRITE)
        run = runs.Run("write-up")
        card = a_written.issue_card({"id": a_written.next_card_id(), "markdown": "", "sources": [],
                               "title": runs.WORKFLOWS["write-up"],
                               **run.card([runs.OPEN], [{"label": "context", "status": "running"},
                                                        {"label": "write", "status": "pending"}])})
        send_sync(card_msg(card))
        parts = a_written.context_parts()
        with trace.span("step", name="context", count=len(parts), sources=parts):
            pass
        run.step("context", "ok", _n(len(parts), "source part"))
        t0 = time.monotonic()
        with trace.span("step", name="write"):
            a_written.write_up(user_text, card["id"])               # fills `card` in place
        run.step("write", "ok", f"{time.monotonic() - t0:.1f} s")
        say_canned(a_written.WRITTEN_LINE, send_sync)
        run.end("done", "ok")
        card.update(run.card([runs.OPEN, runs.COPY]))          # a reconnect replays the last status
        send_sync(card_update(card, markdown=card["markdown"], title=card["title"], sources=card["sources"]))
        send_sync(context_msg())
        _wait_for_playback_then_idle(sess, send_sync)
    except ValueError as e:
        _run_failed(run, "write", "failed-model", send_sync, card)
        send_sync({"type": "error", "scope": "turn", "text": str(e)})
        send_sync({"type": "state", "value": "idle"})
    except Exception as e:
        _run_failed(run, "write", runs.failure_reason(e), send_sync, card)
        panels._turn_failed(send_sync, "write_up", e, "The write-up failed.", recheck=True)


@registry.handler("research")
def research_turn(ctx: Ctx):
    """The research turn (docs/54-vault-map-research.md), typed or spoken, never confirmed: one short
    spoken line, then the run card — `running` at once, a `card_update` per finished step (state stays
    `thinking`), and the last one with the markdown, title and sources. The body is never spoken."""
    sess, turn, send_sync = ctx.session, ctx.turn, ctx.send
    try:
        send_sync({"type": "state", "value": "thinking"})
        _arm_playback(sess, turn)
        say_canned(research.LINE, send_sync)
        send_sync({"type": "state", "value": "thinking"})

        def notify(card, new=False):
            send_sync(card_msg(card) if new else card_update(card))

        card = research.run(commands.research_command(ctx.text) or ctx.text, ctx.origin, notify, ctx.text)
        send_sync(card_update(card, markdown=card["markdown"], title=card["title"], sources=card["sources"]))
        send_sync(context_msg())
        _wait_for_playback_then_idle(sess, send_sync)
    except Exception as e:
        panels._turn_failed(send_sync, "research", e, "The research failed.", recheck=True)


@registry.handler("save_note")
def save_note(ctx: Ctx):
    """The spoken "save that as a note" turn: the last card becomes a note, the page opens."""
    sess, turn, send_sync = ctx.session, ctx.turn, ctx.send
    try:
        send_sync({"type": "state", "value": "thinking"})
        _arm_playback(sess, turn)
        for msg in save_card(None):
            send_sync(msg)
        say_canned(f"[happy] Saved as {a_conversation.last_card['saved']}.", send_sync)
        _wait_for_playback_then_idle(sess, send_sync)
    except ValueError as e:
        send_sync({"type": "error", "scope": "wiki", "text": str(e)})
        send_sync({"type": "state", "value": "idle"})
    except Exception as e:
        panels._turn_failed(send_sync, "save_note", e, "Saving the note failed.", scope="wiki")


# --- approvals (docs/52-approvals.md) ---------------------------------------------------
# "forget <title>" / the Sources view's Forget and "send that to my reMarkable" propose a write: an
# awaiting card + one spoken question, `sess.pending` set. The next turn decides ("yes" / "no" /
# anything else, `_settle_pending`), or the card's Approve / Reject (`resolve`, through the same
# TurnController worker).


def _propose(ctx: Ctx, kind: str, target: str) -> None:
    """The proposal turn: `approvals.propose`, the awaiting card, the spoken question, idle."""
    sess, turn, send_sync = ctx.session, ctx.turn, ctx.send
    try:
        send_sync({"type": "state", "value": "thinking"})
        _arm_playback(sess, turn)
        if ctx.text:
            a_conversation.record_user(ctx.text)
        pending = approvals.propose(sess, kind, target)
        send_sync(card_msg(a_written.get_card(pending["card_id"])))
        a_conversation.record_assistant(say_canned(approvals.KINDS[kind].spoken(pending["proposal"]), send_sync))
        _wait_for_playback_then_idle(sess, send_sync)
    except (ValueError, pipeline.IngestError) as e:
        send_sync({"type": "error", "scope": "turn", "text": getattr(e, "message", None) or str(e)})
        send_sync({"type": "state", "value": "idle"})
    except Exception as e:
        panels._turn_failed(send_sync, "approval", e, "The proposal failed.")


def _say_line(sess: Session, turn, send_sync, line: str) -> None:
    """One spoken (canned) line as the whole reply, recorded in the conversation, then idle."""
    send_sync({"type": "state", "value": "thinking"})
    _arm_playback(sess, turn)
    a_conversation.record_assistant(say_canned(line, send_sync))
    _wait_for_playback_then_idle(sess, send_sync)


@registry.handler("forget")
def forget(ctx: Ctx):
    """"forget the Morrowind guide": one source by title → its proposal; none or several → one line."""
    try:
        sha, line = approvals.forget_target(commands.forget_command(ctx.text) or "")
        if sha is None:
            a_conversation.record_user(ctx.text)
            _say_line(ctx.session, ctx.turn, ctx.send, line)
            return
    except Exception as e:
        panels._turn_failed(ctx.send, "approval", e, "I could not look that source up.")
        return
    _propose(ctx, "forget", sha)


@registry.handler("push")
def push(ctx: Ctx):
    """"send that to my reMarkable": the last card's push proposal; no card → "nothing to send"."""
    if not notebooks.push_available():          # PRD 47b not there: nothing to approve, no run
        try:
            a_conversation.record_user(ctx.text)
            _say_line(ctx.session, ctx.turn, ctx.send, approvals.NOT_AVAILABLE)
        except Exception as e:
            panels._turn_failed(ctx.send, "approval", e, "I could not say that.")
        return
    card = a_conversation.last_card
    if not card or not card.get("markdown"):
        ctx.send({"type": "error", "scope": "turn", "text": approvals.NOTHING_TO_SEND})
        ctx.send({"type": "state", "value": "idle"})
        return
    _propose(ctx, "push", card["id"])


def _settle_pending(ctx: Ctx) -> bool:
    """The one-turn pop: a pending approval is decided by this turn — an explicit yes approves, a
    short no rejects (both answer here: True), anything else drops it silently and the turn runs."""
    via = approvals.via_of(ctx.origin)
    if approvals.is_approval(ctx.text, ctx.session.pending):
        decision = "approve"
    elif approvals.is_rejection(ctx.text):
        decision = "reject"
    else:
        approvals.expire(ctx.session, via)
        return False
    offers.drop_offer()                          # the answer takes the turn, as any turn takes an open offer
    a_conversation.record_user(ctx.text)
    _resolve(ctx.session, ctx.turn, ctx.send, decision, via)
    return True


def _fold_card(send_sync, card: dict, status: str, steps: list | None = None, commit: str = "") -> None:
    """Fold an approval's card to `status` (no Approve / Reject left, Open run kept; `steps` when given)
    and send the `card_update`, with the `commit` when there is one."""
    card.update(status=status, actions=[runs.OPEN], **({"steps": steps} if steps is not None else {}))
    send_sync(card_update(card, **({"commit": commit} if commit else {})))


def _resolve(sess: Session, turn, send_sync, decision: str, via: str, run_id: str | None = None) -> None:
    """Decide the pending approval (voice, typed or a button) and tell the page: `card_update` (done
    + the commit, or rejected; no Approve / Reject left), the refreshed lists after a forget, one
    spoken line. An expired / changed request is `error {scope: turn}` and writes nothing."""
    taken = sess.pending
    try:
        out = approvals.resolve_pending(sess, decision, via, run_id)
        card = a_written.get_card(out.pending["card_id"]) if out.pending else None
        if out.error and out.pending is None and run_id:     # a stale click: that card stops asking too
            stale = next((c for c in a_written.cards_for_run(run_id) if c.get("actions")
                          and any(a.get("kind") in approvals.DECISIONS for a in c["actions"])
                          or c.get("status") == "expired"), None)
            if stale is not None:
                _fold_card(send_sync, stale, "expired" if stale["status"] == "awaiting" else stale["status"])
        if out.error:
            if card is not None:              # changed: the card stops asking
                _fold_card(send_sync, card, "failed", out.pending["steps"] + [
                    {"label": "not applied", "status": "error", "detail": runs.reason_sentence("changed")}])
            send_sync({"type": "error", "scope": "turn", "text": out.error})
            send_sync({"type": "state", "value": "idle"})
            return
        commit = (out.result or {}).get("commit") or ""
        if out.decision == "approve":
            facts = approvals.KINDS[out.pending["kind"]].applied(out.pending["proposal"], out.result or {})
            step = {"label": "applied", "status": "ok", "detail": f"{facts} · commit {commit}" if commit else facts}
            _fold_card(send_sync, card, "done", out.pending["steps"] + [step], commit)
            if out.pending["kind"] == "forget":
                a_conversation.refresh_context()
                send_sync(panels._wiki_msg())
                send_sync(s_sources._sources_msg())
        else:
            _fold_card(send_sync, card, "rejected", out.pending["steps"] + [{"label": "rejected", "status": "skipped"}])
        _say_line(sess, turn, send_sync, out.line)
    except Exception as e:
        one_line = isinstance(e, remarkable.RemarkableError)      # its message is already fit to show
        card = a_written.get_card(taken["card_id"]) if taken is not None and sess.pending is None else None
        if card is not None and card.get("status") == "awaiting":      # the write itself failed
            _fold_card(send_sync, card, "failed", taken["steps"] + [
                {"label": "applied", "status": "error",
                 "detail": str(e) if one_line else runs.reason_sentence(runs.failure_reason(e))}])
        panels._turn_failed(send_sync, "approval", e, str(e) if one_line else "That did not work.")


def resolve_turn(sess: Session, turn: int, run_id: str, decision: str, loop):
    """The card's Approve / Reject (`resolve`), queued like any turn so it can never race a "yes"."""
    _resolve(sess, turn, panels.make_sender(loop, turn), decision, "button", run_id)


def propose_turn(sess: Session, turn: int, kind: str, target: str, loop):
    """The Sources view's Forget (`forget_source`): the same proposal turn as "forget <title>"."""
    _propose(_ctx(sess, turn, "", panels.make_sender(loop, turn)), kind, target)


def _with_context_notice(events):
    """The turn's events, then the 90 % crossing sentence as one more spoken delta (PRD 39)."""
    yield from events
    notice = a_context.context_notice()
    if notice:
        yield ("delta", " " + notice)


@registry.handler("new_topic")
def new_topic(ctx: Ctx):
    """Restart on a summary, tell the page (`thread`), speak the ack, start the memory fold, idle."""
    sess, turn, send_sync = ctx.session, ctx.turn, ctx.send
    try:
        send_sync({"type": "state", "value": "thinking"})
        _arm_playback(sess, turn)
        summary = a_conversation.new_thread()
        send_sync({"type": "thread", "text": summary})
        say_canned(a_conversation.NEW_TOPIC_ACK, send_sync)
        panels.controller.save_now()      # the fold comes after the answer, on the hook thread (PRD 56)
        send_sync(context_msg())
        _wait_for_playback_then_idle(sess, send_sync)
    except Exception as e:
        panels._turn_failed(send_sync, "new_topic", e, "Starting a new topic failed.")


def _ctx(sess: Session, turn: int, user_text: str, send_sync, origin: str = "user") -> Ctx:
    return Ctx(text=user_text, origin=origin, session=sess, turn=turn, send=send_sync,
               push_panel=lambda: panels.push_memory(with_wiki=True))


def new_topic_turn(sess: Session, turn: int, loop):
    approvals.expire(sess, "button")          # the palette's New topic is a turn too (PRD 52)
    new_topic(_ctx(sess, turn, "", panels.make_sender(loop, turn)))


@registry.handler("open_page")
def open_page(ctx: Ctx) -> str:
    """Open the page a find/open/go-back turn asks for (docs/38-focus-context.md) and return the
    one-turn note for the model.

    Pushes `wiki_open {path}` and sets the focus; "" when the turn is not a page command.
    """
    send_sync = ctx.send
    cmd = commands.page_command(ctx.text)
    if cmd is None:
        return ""
    kind, query = cmd
    if kind == "back":
        path, query = recall.last_page, "the last page"
    else:
        hits = v_search.search(query, 1)
        path = hits[0]["path"] if hits else None
    if not path:
        return recall.NO_MATCH_NOTE.format(query=query)
    title = v_focus.page_title(path) or path
    send_sync({"type": "wiki_open", "path": path})
    recall.set_focus("sources" if path.startswith("sources/") else "wiki", path)
    return recall.OPENED_NOTE.format(title=title)


def _day_boundary(send_sync) -> None:
    """Turn start (PRD 56): the first turn after a night restarts the thread (the divider, no
    spoken ack, the old turns folded); a stale date otherwise (a chat across midnight, or no turn since
    a new topic) only refreshes the prompt."""
    if a_conversation.day_rollover(t_time.now()):
        send_sync({"type": "state", "value": "thinking"})
        summary = a_conversation.new_thread()
        send_sync({"type": "thread", "text": summary})
        panels.controller.save_now()
    elif a_conversation.date_changed():
        a_conversation.refresh_context()


def respond(sess: Session, turn: int, user_text: str, send_sync, origin: str = "user"):
    """One turn between its `turn_start` and `turn_end` events (docs/49-event-trace.md); see `_respond`."""
    with trace.turn(turn):
        tool = registry.match(user_text)
        label = ("briefing" if tool and tool.name == "briefing" else "card" if tool and tool.name == "write_up"
                 else "typed" if origin == "typed" else "spoken")
        trace.event("turn_start", origin=label, text_len=len(user_text), words=len(user_text.split()))
        t0, end = time.monotonic(), {"outcome": "ok", "reply_len": 0, "tools": [], "band": None}
        try:
            if tool is None or tool.name != "new_topic":
                _day_boundary(send_sync)
            _respond(sess, turn, user_text, send_sync, origin, end, tool)
            if end["outcome"] == "ok" and (tool is None or tool.name != "new_topic"):
                a_conversation.stamp_turn()
        except BaseException:
            end["outcome"] = "error"
            raise
        finally:
            if trace.turn_failed():
                end["outcome"] = "error"
            trace.event("turn_end", ms=trace.ms_since(t0), ctx_pct=a_context.context_usage()["pct"], **end)


def _respond(sess: Session, turn: int, user_text: str, send_sync, origin: str, end: dict, tool=None):
    """LLM → sentence-streamed TTS → wait for browser playback → idle.

    The one reply path: a typed turn and a transcribed one differ only in `origin`, which
    tags the trace's `turn_start` ("user" spoken, "typed" from the composer); the conversation,
    tools and memory see them identically. Always ends in the idle state, errors included.
    A find/open/go-back turn first opens the page (PRD 38), then reaches the model with a note.
    "new topic" / `/new` never reaches the model: it is the new-topic turn (PRD 39); "what's new?" /
    `/brief` is the briefing turn, offline-safe when nothing changed (PRD 46). Every
    completed turn ends with a `context` message; the 90 % crossing adds one spoken sentence.
    """
    ctx = _ctx(sess, turn, user_text, send_sync, origin)
    if getattr(sess, "pending", None) is not None and _settle_pending(ctx):   # PRD 52: the turn answers it
        return
    if tool is not None and not tool.needs_llm:
        tool.run(ctx)
        return
    if not services.llm_ok():
        # PRD 41: no thinking state, no timeout — the composer line says what to run
        send_sync({"type": "error", "scope": "turn", "text": panels.llm_offline_text()})
        send_sync({"type": "state", "value": "idle"})
        return
    if tool is not None and tool.takeover:
        tool.run(ctx)
        return
    try:
        send_sync({"type": "state", "value": "thinking"})
        _arm_playback(sess, turn)
        note = tool.run(ctx) if tool is not None else ""
        with registry.bind(ctx):     # a tool round's file_source refreshes the panel through it
            events = _with_context_notice(a_turn.ask_events(user_text, origin=origin, note=note))
            reply = s_speech.speak_stream(events, send_sync)
        end.update(reply_len=len(reply), tools=[tc["name"] for tc in a_conversation.get_last_tool_calls()], band=a_conversation.last_band)
        log.debug(f"Bot: {reply}")
        trace.verbose("turn_text", user=user_text, reply=reply)
        send_sync({"type": "transcript", "role": "assistant", "text": reply})
        send_sync(context_msg())
        _wait_for_playback_then_idle(sess, send_sync)
    except Exception as e:
        end["outcome"] = "error"
        panels._turn_failed(send_sync, "respond", e, "The reply failed.", recheck=True)


def say_canned(text: str, send_sync, tts=None):
    """Speak a canned (non-LLM) line: same state/speech/speech_end shape as speak_stream,
    plus a transcript message. Returns the clean (tag-stripped) text.
    """
    reply = s_speech.speak_stream([("delta", text)], send_sync, tts=tts)
    send_sync({"type": "transcript", "role": "assistant", "text": reply})
    return reply


def handle_text(sess: Session, turn: int, text: str, loop):
    """One typed turn (PRD 24): the transcript path without the ASR.

    The browser already rendered the user's line, so no user transcript is echoed back.
    """
    log.debug(f"You (typed): {text}")
    respond(sess, turn, text, panels.make_sender(loop, turn), origin="typed")


def handle_utterance(sess: Session, turn: int, audio, loop):
    """Transcribe one captured utterance (hands-free) and respond.

    Idle is sent here when the transcript is empty or the ASR raised; once there is text,
    `respond` owns the rest of the turn (and its own idle).
    """
    send_sync = panels.make_sender(loop, turn)
    try:
        send_sync({"type": "state", "value": "processing"})
        text = asr.transcribe(audio, prompt=identity.asr_prompt())
        if getattr(asr, "last_error", None):
            services.recheck()   # the remote ASR failed: re-probe now
    except Exception as e:
        approvals.expire(sess, "voice")           # a turn started: a pending approval is over (PRD 52)
        with trace.turn(turn):
            panels._turn_failed(send_sync, "asr", e, "I could not transcribe that.")
        return
    if not text:
        approvals.expire(sess, "voice")
        send_sync({"type": "state", "value": "idle"})
        return
    log.debug(f"You: {text}")
    send_sync({"type": "transcript", "role": "user", "text": text})
    respond(sess, turn, text, send_sync)


def canned_turn(sess: Session, turn: int, loop, text: str, thinking: bool):
    """Speak a canned line as a full turn: optional thinking state, TTS, record, idle."""
    send_sync = panels.make_sender(loop, turn)
    try:
        _day_boundary(send_sync)
        if thinking:
            send_sync({"type": "state", "value": "thinking"})
        _arm_playback(sess, turn)
        a_conversation.record_assistant(say_canned(text, send_sync))
        a_conversation.stamp_turn()
        _wait_for_playback_then_idle(sess, send_sync)
    except Exception as e:
        with trace.turn(turn):
            panels._turn_failed(send_sync, "canned_turn", e, "I could not say that.")


def greet(sess: Session, turn: int, loop):
    """Speak the avatar's greeting on the first page connect."""
    canned_turn(sess, turn, loop, s_speech.AVATAR["greeting"], thinking=True)
