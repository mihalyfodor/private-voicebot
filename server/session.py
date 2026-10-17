"""Per-connection state (`Session`) and the single-worker turn queue (`TurnController`).

See docs/12-turn-controller.md. A *turn* is one complete user-visible exchange:
transcribe → LLM → TTS → wait for browser playback → idle. Turns never overlap:
they all run on the controller's one worker thread, in submit order. The periodic
memory save is not a turn: it runs on a side thread so the next utterance is not held up.
"""
import contextlib
import os
import queue
import threading

from core import trace
log = trace.logger("Turn")

#: how many completed turns between periodic hook calls (the session-memory save).
#: Override with MEMORY_SAVE_EVERY=<n> in the environment.
MEMORY_SAVE_EVERY = int(os.getenv("MEMORY_SAVE_EVERY", "5"))


class Session:
    """All state belonging to one websocket connection.

    Created in `websocket_endpoint`; dropped when that socket goes away, so a
    reload can never leak hearing/playback state into the next connection.
    """

    def __init__(self, websocket=None):
        self.websocket = websocket
        self.endpointer = None      # vad.Endpointer, lazily built on the first hands-free frame
        self.hearing = False        # last "listening" value sent (only send on change)
        self.playback_done = threading.Event()
        self.turn_id = 0            # last allocated turn id
        self.expected_turn = None   # turn whose playback_done we are currently waiting for
        #: the one approval waiting for the next turn (docs/52-approvals.md): {run_id, kind, proposal,
        #: hash, created, card_id, steps} or None; the next turn (or a card button) decides it
        self.pending = None

    def next_turn(self) -> int:
        self.turn_id += 1
        return self.turn_id


class TurnController:
    """One worker thread + a queue. Everything a turn does goes through `submit`.

    `busy` is true from the moment a turn is submitted until its function returns,
    so the websocket handler can gate the VAD / refuse PTT synchronously, with no
    window between "submitted" and "started".
    """

    def __init__(self, on_turn_complete=None):
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0
        self._idle = threading.Event()
        self._idle.set()
        self._hook_idle = threading.Event()   # cleared while the periodic hook runs
        self._hook_idle.set()
        self.completed = 0
        self.on_turn_complete = on_turn_complete
        self._thread = threading.Thread(target=self._run, daemon=True, name="turn-worker")
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._pending > 0

    def submit(self, fn, *args, turn_id=None) -> None:
        """Queue `fn(*args)` for the worker thread and return immediately; with `turn_id` every
        event it writes carries that turn's ids (docs/49-event-trace.md)."""
        with self._lock:
            self._pending += 1
            self._idle.clear()
        self._queue.put((fn, args, turn_id))

    def join_idle(self, timeout: float = 5.0) -> bool:
        """Block until no turn is pending and no hook is running (test helper).

        True if both went idle in time.
        """
        return self._idle.wait(timeout) and self._hook_idle.wait(timeout)

    @property
    def hook_running(self) -> bool:
        return not self._hook_idle.is_set()

    def _start_hook(self, count: int) -> None:
        """Run the periodic hook on its own daemon thread so the next turn is not held up.

        Overlap guard: if the previous run is still going (a slow memory save), this one is
        skipped — the next multiple of MEMORY_SAVE_EVERY picks the turns up.
        """
        if not self._hook_idle.is_set():
            log.info("hook: previous run still going, skipping")
            return
        self._hook_idle.clear()

        def run():
            try:
                self.on_turn_complete(count)
            except Exception as e:
                log.warning(f"hook error: {e}")
            finally:
                self._hook_idle.set()

        threading.Thread(target=run, daemon=True, name="turn-hook").start()

    def _run(self) -> None:
        while True:
            fn, args, turn_id = self._queue.get()
            try:
                with trace.turn(turn_id) if turn_id is not None else contextlib.nullcontext():
                    fn(*args)
            except Exception as e:  # a broken turn must never kill the worker
                log.warning(f"error: {e}")
            finally:
                with self._lock:
                    self._pending -= 1
                    self.completed += 1
                    count = self.completed
                    if self._pending == 0:
                        self._idle.set()
                if self.on_turn_complete is not None and count % MEMORY_SAVE_EVERY == 0:
                    self._start_hook(count)
