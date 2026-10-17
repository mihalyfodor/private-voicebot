"""WebSocket actions of the wiki panel: pages, notes, the editor writes, search and ingest (PRD 50 step 11, from chatbot.py)."""
import asyncio
from agent import conversation as a_conversation
from memory import store as m_store
from vault import pages as v_pages
from vault import search as v_search
from server.actions import memory as s_memory
from server import panels
from server.actions import sources as s_sources
from core import util



async def _send_wiki() -> None:
    """Push a fresh `wiki` message (page list + lint); the Wiki tab re-renders from it.

    A `sources` message follows: every path that changes the wiki may have filed a source, and
    the rail badge counts what arrived while the user was elsewhere.
    """
    await asyncio.to_thread(a_conversation.refresh_context)   # the next turn sees the new/changed pages
    await panels.send(await asyncio.to_thread(panels._wiki_msg))
    await panels.send(await asyncio.to_thread(s_sources._sources_msg))


async def _action_get_wiki(sess, msg, loop):
    """The Wiki tab asking for the page list (on open, and after every reconnect)."""
    await _send_wiki()


async def _action_read_wiki_page(sess, msg, loop):
    """One page for the right-hand pane, with its wikilink targets for the clickable links."""
    await panels.send(await asyncio.to_thread(panels._wiki_page_msg, str(msg.get("path") or "")))


async def _action_wiki_note(sess, msg, loop):
    """The "add note" box: append one dated bullet under `## Notes`, then refresh both panes."""
    path = str(msg.get("path") or "")
    op = {"op": "append_note", "path": path, "text": msg.get("text"),
          "source": f"user note {util._today_iso()}"}
    applied, rejected = await asyncio.to_thread(v_pages.apply_page_ops, [op], op["source"])
    if not applied:
        reason = rejected[0]["reason"] if rejected else "nothing to add"
        await panels._error("wiki", reason)
        return
    await panels.send(await asyncio.to_thread(panels._wiki_page_msg, path))
    await _send_wiki()


async def _wiki_user_op(fn, *args, reply_path=None):
    """One editor write (PRD 28): refresh the page (when `reply_path`) and the list, or `error`."""
    result, reason = await asyncio.to_thread(fn, *args)
    if reason:
        await panels._error("wiki", reason)
        return
    if reply_path is not None:
        await panels.send(await asyncio.to_thread(panels._wiki_page_msg, reply_path(result)))
    await _send_wiki()


async def _action_wiki_save(sess, msg, loop):
    """⌘S: `{path, text, etag}` → `wiki_page` (new etag) + `wiki`, or `wiki_conflict` on a stale etag."""
    path = str(msg.get("path") or "")
    result, reason = await asyncio.to_thread(v_pages.save_page, path, msg.get("text"), msg.get("etag"))
    if reason:
        await panels._error("wiki", reason)
    elif result.get("conflict"):
        await panels.send({"type": "wiki_conflict", "path": path, "text": result["text"], "etag": result["etag"]})
    else:
        await panels.send(await asyncio.to_thread(panels._wiki_page_msg, path))
        await _send_wiki()


async def _action_wiki_create(sess, msg, loop):
    """A missing link, "Today" or "New page": `{path}` → the new page + list."""
    await _wiki_user_op(v_pages.create_page, str(msg.get("path") or ""), reply_path=lambda p: p)


async def _action_wiki_rename(sess, msg, loop):
    """`{path, new_path}` → the moved page (links rewritten) + list."""
    await _wiki_user_op(v_pages.rename_page, str(msg.get("path") or ""), str(msg.get("new_path") or ""),
                        reply_path=lambda r: r["path"])


async def _action_wiki_delete(sess, msg, loop):
    """`{path}` → the list without it (lint names the links that broke)."""
    await _wiki_user_op(v_pages.delete_page, str(msg.get("path") or ""))


async def _action_wiki_search(sess, msg, loop):
    """The list's search box: `{query}` → `{type: wiki_search, query, hits}` (BM25, same as the tool)."""
    query = str(msg.get("query") or "")
    hits = await asyncio.to_thread(v_search.search, query, 20) if query.strip() else []
    await panels.send({"type": "wiki_search", "query": query, "hits": hits})


async def _action_wiki_ingest(sess, msg, loop):
    """"Re-ingest last session": run the memory save (which includes wiki ingest) on demand."""
    await asyncio.to_thread(s_memory._ingest_now)
    await panels.send(panels._memory_msg(m_store.load_profile()))
    await _send_wiki()
