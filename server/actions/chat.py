"""WebSocket actions of the chat panel: typed turns, focus, playback, approvals, new topic and saving a card (PRD 50 step 11, from chatbot.py)."""
import asyncio
from agent import approvals
from agent import recall
from server import panels
from server import turns as s_turns



async def _action_text_turn(sess, msg, loop):
    """Composer input (PRD 24): queued exactly like a finished utterance."""
    text = (msg.get("text") or "").strip()
    if text:
        panels.submit_turn(s_turns.handle_text, text, loop)


async def _action_set_focus(sess, msg, loop):
    """`{view, path?}` from the client whenever the view or the open page changes (PRD 38)."""
    recall.set_focus(msg.get("view"), msg.get("path"))


async def _action_playback_done(sess, msg, loop):
    """The browser is done with a turn's audio: it finished playing it, or (as
    `playback_blocked`) could not autoplay it and will play it after the next gesture.
    Either way we stop waiting and go idle. Stale (mismatched) turns are ignored.
    """
    if panels._turn_matches(sess, msg.get("turn")):
        sess.playback_done.set()


#: same handling: both mean "stop waiting for this turn's playback"
_action_playback_blocked = _action_playback_done


async def _action_resolve(sess, msg, loop):
    """`{run_id, decision: approve|reject}` → the approval decided on the turn worker (PRD 52)."""
    decision = msg.get("decision")
    if decision not in approvals.DECISIONS:
        await panels._error("turn", f"unknown decision {str(decision)[:20]!r}")
        return
    panels.submit_turn(s_turns.resolve_turn, str(msg.get("run_id") or ""), decision, loop)


async def _action_save_card(sess, msg, loop):
    """The card's Save to wiki: `{id, folder?}` → `wiki_open` + `card_saved` + `wiki` + `sources`,
    or `error` (wiki): "nothing to save" / "already saved" / a bad folder (PRD 43)."""
    try:
        replies = await asyncio.to_thread(s_turns.save_card, msg.get("id"), str(msg.get("folder") or ""))
    except ValueError as e:
        await panels._error("wiki", str(e))
        return
    for reply in replies:
        await panels.send(reply)


async def _action_new_topic(sess, msg, loop):
    """The palette's "New topic": queued like a turn (PRD 39)."""
    panels.submit_turn(s_turns.new_topic_turn, loop)
