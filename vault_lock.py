"""The one lock over everything on disk that memory, wiki and sources share.

`memory.save`, every wiki write and every source ingest take it, so a hand edit from the UI
waits for a running save instead of being overwritten by a profile that was loaded before it.
Re-entrant: `memory.save` files wiki pages while already holding it.
"""
import threading

LOCK = threading.RLock()
