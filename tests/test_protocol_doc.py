"""docs/protocol.md cannot drift: every action the server accepts and every message type it
sends (and the client handles) must be named there."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import re

from server import ws

ROOT = os.path.join(os.path.dirname(__file__), "..")
DOC = open(os.path.join(ROOT, "docs/protocol.md")).read()
SERVER = "".join(open(os.path.join(d, f)).read() for d, _, fs in sorted(os.walk(os.path.join(ROOT, "server")))
                 for f in sorted(fs) if f.endswith(".py"))
CLIENT = open(os.path.join(ROOT, "static/app.js")).read()


def documented(name: str) -> bool:
    return f"`{name}`" in DOC


def test_every_action_is_documented():
    missing = [a for a in ws.ACTIONS if not documented(a)]
    assert missing == [], f"actions missing from docs/protocol.md: {missing}"


def test_every_server_message_type_is_documented():
    sent = set(re.findall(r'"type": "([a-z_]+)"', SERVER))
    assert sent, "no message literals found in server/"
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
    assert set(action_rows) == set(ws.ACTIONS)
    type_rows = re.findall(r"^\| `([a-z_]+)` \|", DOC.split("## Server")[1], re.M)
    sent = set(re.findall(r'"type": "([a-z_]+)"', SERVER))
    assert set(type_rows) == sent


def _row(name: str) -> str:
    return next(line for line in DOC.splitlines() if line.startswith(f"| `{name}` |"))


def test_the_approval_rows_are_documented():
    """PRD 52 scenario 9: `resolve`, the rewritten `forget_source` (it proposes, no `confirm`) and the
    `card_update` fields an approval adds."""
    assert "`run_id`, `decision` approve/reject" in _row("resolve") and "that request expired" in _row("resolve")
    forget = _row("forget_source")
    assert "confirm" not in forget and "proposal" in forget
    assert "`commit?`" in _row("card_update") and "`rejected`" in _row("card_update")
    assert "awaiting/rejected" in _row("card")
