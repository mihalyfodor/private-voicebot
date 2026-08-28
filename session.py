"""Per-connection state (`Session`) and the single-worker turn queue (`TurnController`).

See docs/12-turn-controller.md. A *turn* is one complete user-visible exchange:
transcribe → LLM → TTS → wait for browser playback → idle. Turns never overlap:
they all run on the controller's one worker thread, in submit order.
"""
import queue
import threading


class Session:
    """All state belonging to one websocket connection.

    Created in `websocket_endpoint`; dropped when that socket goes away, so a
    reload can never leak PTT/recorder/hearing state into the next connection.
    """

    def __init__(self, websocket=None):
        self.websocket = websocket
        self.endpointer = None      # vad.Endpointer, lazily built on the first hands-free frame
        self.recorder = None        # vad.Recorder, built on "ptt start"
        self.ptt_active = False
        self.hearing = False        # last "listening" value sent (only send on change)
        self.playback_done = threading.Event()
        self.turn_id = 0            # last allocated turn id
        self.expected_turn = None   # turn whose playback_done we are currently waiting for

    def next_turn(self) -> int:
        self.turn_id += 1
        return self.turn_id


class TurnController:
    """One worker thread + a queue. Everything a turn does goes through `submit`.

    `busy` is true from the moment a turn is submitted until its function returns,
    so the websocket handler can gate the VAD / refuse PTT synchronously, with no
    window between "submitted" and "started".
    """

    #: how many completed turns between periodic hook calls (session-memory save)
    PERIODIC_EVERY = 10

    def __init__(self, on_turn_complete=None):
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._pending = 0
        self._idle = threading.Event()
        self._idle.set()
        self.completed = 0
        self.on_turn_complete = on_turn_complete
        self._thread = threading.Thread(target=self._run, daemon=True, name="turn-worker")
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._pending > 0

    def submit(self, fn, *args) -> None:
        """Queue `fn(*args)` for the worker thread and return immediately."""
        with self._lock:
            self._pending += 1
            self._idle.clear()
        self._queue.put((fn, args))

    def join_idle(self, timeout: float = 5.0) -> bool:
        """Block until no turn is pending (test helper). True if it went idle in time."""
        return self._idle.wait(timeout)

    def _run(self) -> None:
        while True:
            fn, args = self._queue.get()
            try:
                fn(*args)
            except Exception as e:  # a broken turn must never kill the worker
                print(f"[Turn error] {e}")
            finally:
                with self._lock:
                    self._pending -= 1
                    self.completed += 1
                    count = self.completed
                    if self._pending == 0:
                        self._idle.set()
                if self.on_turn_complete is not None and count % self.PERIODIC_EVERY == 0:
                    try:
                        self.on_turn_complete(count)
                    except Exception as e:
                        print(f"[Turn hook error] {e}")
