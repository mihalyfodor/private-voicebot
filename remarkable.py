"""Client of the host push service (docs/47b-remarkable-push.md): `push` sends a card to scripts/remarkable_server.py.

`URL` is `$REMARKABLE_PUSH_URL`; unset means the feature is off (no command, no status row). A root
module for now; it moves with approvals.py at PRD 50.
"""
import os

import requests

URL = os.getenv("REMARKABLE_PUSH_URL", "").rstrip("/")
NOT_RUNNING = "reMarkable push service not running"


class RemarkableError(Exception):
    """A push that did not happen; the message is one line fit for the card and the status line."""


def push(title: str, markdown: str) -> dict:
    """POST the card to the service -> its `{ok, pages, name}`; raises `RemarkableError` with the service's reason."""
    try:
        r = requests.post(f"{URL}/push", json={"title": title, "markdown": markdown}, timeout=120)
    except requests.RequestException as exc:
        raise RemarkableError(NOT_RUNNING) from exc
    try:
        reply = r.json()
    except ValueError:
        reply = {}
    if r.status_code >= 400 or not reply.get("ok"):
        raise RemarkableError(str(reply.get("error") or f"push failed (HTTP {r.status_code})"))
    return reply
