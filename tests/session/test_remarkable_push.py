"""PRD 47b scenarios 3-5 (chat half): "send this to my reMarkable" proposes, approval pushes once."""
import json
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import chatbot
import llm
import remarkable
from session import Session
from tests.conftest import _serve, button_turn, free_port, session_turn, spoken_lines, trace_lines
from tests.prod.test_remarkable_server import RENDER, serve_push

pytestmark = pytest.mark.usefixtures("fresh_llm")


def _of(sent, kind):
    return [m for m in sent if m["type"] == kind]


def _card() -> dict:
    card = {"id": llm.next_card_id(), "markdown": "# Plan\n\n" + "line\n" * 70, "sources": [], "title": "Plan"}
    llm._cards[card["id"]] = card
    llm.last_card = card
    return card


@pytest.fixture
def fake_push():
    """A stand-in push service: every POST /push body into `fake.requests`; `fake.status` / `fake.reply`."""
    fake = SimpleNamespace(requests=[], status=200, reply={"ok": True, "pages": 3, "name": "Plan.pdf"})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            fake.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps(fake.reply).encode()
            self.send_response(fake.status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    server, url = _serve(Handler)
    with patch.object(remarkable, "URL", url):
        yield fake
    server.shutdown()
    server.server_close()


def test_send_this_proposes_with_title_and_pages_and_calls_nothing(vault, fake_push):
    card = _card()
    sess = Session()
    sent = session_turn(sess, 1, "send this to my reMarkable")
    (proposal,) = _of(sent, "card")
    assert proposal["status"] == "awaiting" and proposal["workflow"] == "push"
    assert sess.pending["kind"] == "push"
    assert sess.pending["proposal"] == {"card_id": card["id"], "title": "Plan", "pages": 2}
    assert "Plan" in spoken_lines(sent)[0] and "2 pages" in spoken_lines(sent)[0]
    assert fake_push.requests == []


def test_nothing_to_send_leaves_nothing_pending(vault, fake_push):
    sess = Session()
    sent = session_turn(sess, 1, "send this to my reMarkable")
    assert _of(sent, "error")[0]["text"] == "nothing to send"
    assert sess.pending is None and fake_push.requests == []


def test_approve_pushes_the_card_once(vault, fake_push):
    card = _card()
    sess = Session()
    session_turn(sess, 1, "send this to my reMarkable")
    sent = session_turn(sess, 2, "yes")
    assert fake_push.requests == [{"title": "Plan", "markdown": card["markdown"]}]
    assert spoken_lines(sent) == ["Sent to your reMarkable."]
    (update,) = _of(sent, "card_update")
    assert update["status"] == "done" and update["steps"][-1]["detail"] == "uploaded 'Plan', 3 pages"


def test_reject_button_pushes_nothing_and_records_the_decision(vault, fake_push):
    _card()
    sess = Session()
    session_turn(sess, 1, "send this to my reMarkable")
    run_id = sess.pending["run_id"]
    button_turn(chatbot.resolve_turn, sess, 2, run_id, "reject")
    assert fake_push.requests == [] and sess.pending is None
    assert [r["decision"] for r in trace_lines(kind="decision")] == ["reject"]


def test_the_service_down_fails_the_run_card_with_its_reason(vault, fake_push):
    _card()
    fake_push.status, fake_push.reply = 503, {"ok": False, "error": "reMarkable not connected (10.11.99.1)"}
    sess = Session()
    session_turn(sess, 1, "send this to my reMarkable")
    sent = session_turn(sess, 2, "yes")
    (update,) = _of(sent, "card_update")
    assert update["status"] == "failed" and "not connected" in update["steps"][-1]["detail"]
    assert _of(sent, "error")[0]["text"].startswith("reMarkable not connected")
    assert sess.pending is None and "Sent to your reMarkable." not in spoken_lines(sent)


@RENDER
def test_device_down_end_to_end_is_not_connected(vault):
    proc, url = serve_push(f"http://127.0.0.1:{free_port()}")
    try:
        with patch.object(remarkable, "URL", url):
            _card()
            sess = Session()
            session_turn(sess, 1, "send this to my reMarkable")
            sent = session_turn(sess, 2, "yes")
    finally:
        proc.kill()
    assert _of(sent, "card_update")[0]["status"] == "failed"
    assert _of(sent, "error")[0]["text"].startswith("reMarkable not connected")
    assert sess.pending is None and "Sent to your reMarkable." not in spoken_lines(sent)
