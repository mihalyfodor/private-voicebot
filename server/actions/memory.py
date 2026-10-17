"""WebSocket actions of the memory panel: the profile, its editor ops, saving the session and the memory / wiki bridge (PRD 50 step 11, from chatbot.py)."""
import asyncio
from agent import conversation as a_conversation
from memory import ops as m_ops
from memory import store as m_store
from vault import pages as v_pages
from server import panels
from server.actions import wiki as s_wiki



def _ingest_now() -> None:
    """Re-ingest the session on the worker/executor: the same save path a turn would take."""
    try:
        a_conversation.save_memory()
    except Exception as e:
        panels._memory_failed(e)


async def _save_session_memory() -> None:
    """Fold the session into memory.json off the loop, then push the fresh snapshot.

    The Wiki tab is only refreshed when the save actually filed pages — same rule as the
    periodic save in `_periodic_memory`.
    """
    filed = v_pages.applied_count()
    await asyncio.to_thread(_ingest_now)
    await panels.send(panels._memory_msg(await asyncio.to_thread(m_store.load_profile)))
    if v_pages.applied_count() != filed:
        await s_wiki._send_wiki()


async def _action_get_memory(sess, msg, loop):
    """The memory panel asking for the current profile (on open, and after every reconnect).

    Opening the tab is also a save trigger: the current snapshot goes out first so the panel
    never waits on the LLM, then — if this session has turns no save has folded in yet and no
    turn is running — the save runs and a second, fresh snapshot follows. While a turn is in
    flight nothing is saved here; the periodic save catches up.
    """
    await panels.send(panels._memory_msg(await asyncio.to_thread(m_store.load_profile)))
    if a_conversation.has_unsaved_turns() and not panels.controller.busy:
        await _save_session_memory()


async def _action_save_memory(sess, msg, loop):
    """The panel's "Save now": the same save as the tab-open trigger, but unconditional."""
    if panels.controller.busy:
        await panels._error("memory", "Wait until the reply finishes.")
        return
    await panels.send(panels._memory_msg(await asyncio.to_thread(m_store.load_profile)))
    await _save_session_memory()


async def _send_memory_error(path, reason) -> None:
    """The one shape a rejected memory/wiki edit comes back in; the panel puts it on the row."""
    await panels.send({"type": "memory_error", "path": path, "reason": reason})


async def _action_memory_op(sess, msg, loop):
    """One hand edit from the memory panel: {op, path, value, reason?} → memory | memory_error.

    `memory.edit` does file IO under the memory lock (and may wait for a background save), so it
    runs off the event loop.
    """
    op = {"op": msg.get("op"), "path": msg.get("path"),
          "value": msg.get("value"), "reason": msg.get("reason")}
    profile, reason = await asyncio.to_thread(m_ops.edit, op)
    if reason:
        await _send_memory_error(op["path"], reason)
        return
    await asyncio.to_thread(a_conversation.refresh_context)
    await panels.send(panels._memory_msg(profile))


async def _action_memory_ops(sess, msg, loop):
    """A batch of hand edits from the memory panel: {ops: [{op, path, value, reason?}, ...]}.

    All-or-nothing (the drag-and-drop moves are one ADD + one DELETE): on the first bad op
    nothing is written and the panel gets a `memory_error` naming that op's path.
    """
    ops = msg.get("ops")
    profile, error = await asyncio.to_thread(m_ops.edit_many, ops)
    if error:
        await _send_memory_error(error.get("path"), error.get("reason"))
        return
    await asyncio.to_thread(a_conversation.refresh_context)
    await panels.send(panels._memory_msg(profile))


async def _action_memory_reset(sess, msg, loop):
    """The panel's "Forget everything": the profile is emptied and the reset is audited."""
    profile = await asyncio.to_thread(m_ops.reset)
    await asyncio.to_thread(a_conversation.refresh_context)
    await panels.send(panels._memory_msg(profile))


async def _bridge(move, *args):
    """One memory <-> wiki move (PRD 19), then refresh both views — or the error on its row."""
    profile, error = await asyncio.to_thread(move, *args)
    if error:
        await _send_memory_error(error.get("path"), error.get("reason"))
        return
    await asyncio.to_thread(a_conversation.refresh_context)
    await panels.send(panels._memory_msg(profile))
    await s_wiki._send_wiki()


async def _action_archive_memory(sess, msg, loop):
    """Drag a memory row onto the Wiki tab: file the entry, then mark it stale."""
    await _bridge(m_ops.archive, str(msg.get("section") or ""), msg.get("index"))


async def _action_pull_wiki(sess, msg, loop):
    """Drag a wiki page onto a memory section: recreate a live entry from the page."""
    await _bridge(m_ops.pull, str(msg.get("path") or ""), str(msg.get("section") or ""))
