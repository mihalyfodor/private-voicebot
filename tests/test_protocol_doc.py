"""docs/protocol.md cannot drift: every action the server accepts and every message type it
sends (and the client handles) must be named there."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import re

import chatbot

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOC = open(os.path.join(ROOT, "docs/protocol.md")).read()
SERVER = open(os.path.join(ROOT, "chatbot.py")).read()
CLIENT = open(os.path.join(ROOT, "static/app.js")).read()


def documented(name: str) -> bool:
    return f"`{name}`" in DOC


def test_every_action_is_documented():
    missing = [a for a in chatbot.ACTIONS if not documented(a)]
    assert missing == [], f"actions missing from docs/protocol.md: {missing}"


def test_every_server_message_type_is_documented():
    sent = set(re.findall(r'"type": "([a-z_]+)"', SERVER))
    assert sent, "no message literals found in chatbot.py"
    missing = sorted(t for t in sent if not documented(t))
    assert missing == [], f"message types missing from docs/protocol.md: {missing}"


def test_every_client_handled_type_is_documented():
    handled = set(re.findall(r"msg\.type === '([a-z_]+)'", CLIENT))
    assert handled, "no ws.onmessage ladder found in static/app.js"
    missing = sorted(t for t in handled if not documented(t))
    assert missing == [], f"client-handled types missing from docs/protocol.md: {missing}"


def test_doc_names_only_real_actions_and_types():
    """The other direction: a table row for something that no longer exists is stale."""
    action_rows = re.findall(r"^\| `([a-z_]+)` \|", DOC.split("## Server")[0], re.M)
    assert set(action_rows) == set(chatbot.ACTIONS)
    type_rows = re.findall(r"^\| `([a-z_]+)` \|", DOC.split("## Server")[1], re.M)
    sent = set(re.findall(r'"type": "([a-z_]+)"', SERVER))
    assert set(type_rows) == sent
