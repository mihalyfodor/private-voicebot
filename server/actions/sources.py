"""WebSocket actions of the sources panel: the list, re-ingest and forget (PRD 50 step 11, from chatbot.py)."""
import asyncio
from vault import manifest
from ingest import pipeline
from server import panels
from server import turns
from server.actions import wiki as s_wiki



def _sources_msg() -> dict:
    """The `sources` message the Sources view (and its rail badge) re-renders from (PRD 25)."""
    return {"type": "sources", "sources": pipeline.list_sources()}


async def _action_get_sources(sess, msg, loop):
    """The Sources view asking for the inbox list (on open, and after every reconnect)."""
    await panels.send(await asyncio.to_thread(_sources_msg))


async def _source_op(fn, *args):
    """One inbox action: run it off the loop, then refresh the list — or report the refusal."""
    try:
        await asyncio.to_thread(fn, *args)
    except pipeline.IngestError as exc:
        await panels._error("sources", exc.message)
        return
    await s_wiki._send_wiki()


async def _action_reingest_source(sess, msg, loop):
    """`{sha}` → the source page rewritten from its raw file; the open page is re-read via `wiki`."""
    await _source_op(pipeline.reingest, str(msg.get("sha") or ""))


async def _action_forget_source(sess, msg, loop):
    """`{sha}` → the forget proposal turn (PRD 52): the awaiting card in the transcript, one spoken
    question, nothing deleted until it is approved. An unknown sha is `error` (sources)."""
    sha = str(msg.get("sha") or "")
    if manifest.manifest_find(sha) is None:
        await panels._error("sources", f"no filed source with sha {sha[:12]}")
        return
    panels.submit_turn(turns.propose_turn, "forget", sha, loop)
