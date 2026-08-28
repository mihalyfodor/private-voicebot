import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import chatbot


@pytest.fixture(autouse=True)
def reset_chatbot_session():
    """Per-connection state now lives on chatbot.session; make sure no test inherits one.

    Also waits for the turn worker to drain so a slow turn cannot bleed into the next test.
    """
    chatbot.session = None
    chatbot.greeted = True  # tests that want the greeting turn set this back to False
    yield
    chatbot.controller.join_idle(timeout=5)
    chatbot.session = None


def untagged(msg: dict) -> dict:
    """A received message without its turn id (for equality asserts that don't care)."""
    return {k: v for k, v in msg.items() if k != "turn"}


class Blocker:
    """A turn that parks the worker until released — makes `controller.busy` deterministic."""

    def __init__(self):
        import threading
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args):
        self.started.set()
        self.release.wait(timeout=10)

    def hold(self, controller):
        controller.submit(self)
        assert self.started.wait(timeout=5), "blocking turn never started"

    def free(self, controller):
        self.release.set()
        controller.join_idle(timeout=5)
