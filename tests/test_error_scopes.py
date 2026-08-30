"""Scoped `error` messages: every refusal names the UI slot it belongs to (`scope`), and the
client routes on that field instead of on the view that happens to be open.

The server half drives one action per scope through the Panel fixture (which itself asserts a
scope on every `error` it sees); the client half asserts on tests/app_fixture.mjs.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import chatbot
from tests.conftest import Blocker


def test_wiki_scope(panel):
    reply = panel.send({"action": "wiki_create", "path": "raw/x"})
    assert reply == {"type": "error", "scope": "wiki", "text": "bad path 'raw/x'"}


def test_memory_scope(panel):
    blocker = Blocker()
    blocker.hold(chatbot.controller)
    try:
        reply = panel.send({"action": "save_memory"})
    finally:
        blocker.free(chatbot.controller)
    assert reply == {"type": "error", "scope": "memory", "text": "Wait until the reply finishes."}


def test_sources_scope(panel):
    reply = panel.send({"action": "forget_source", "sha": "abc"})
    assert (reply["scope"], reply["text"]) == ("sources", "forget needs an explicit confirm")


def test_settings_scope(panel):
    reply = panel.send({"action": "set_setting", "key": "vault_path", "value": 1})
    assert reply["scope"] == "settings" and "vault_path" in reply["text"]


def test_intake_scope(panel, monkeypatch):
    """An exception inside "Re-ingest last session" is the intake slot's, tagged with its action."""
    def boom():
        raise RuntimeError("oMLX is down")
    monkeypatch.setattr(chatbot, "_ingest_now", boom)
    reply = panel.send({"action": "wiki_ingest"})
    assert reply == {"type": "error", "scope": "intake", "text": "oMLX is down", "action": "wiki_ingest"}


def test_turn_scope_for_a_malformed_frame(panel):
    panel.ws.send_text("{not json")
    reply = panel.ws.receive_json()
    while reply["type"] != "error":   # the connect-time hands_free / state come first
        reply = panel.ws.receive_json()
    assert reply["scope"] == "turn" and "action" not in reply


def test_every_action_has_a_scope():
    assert {chatbot._scope_of(a) for a in chatbot.ACTIONS} <= set(chatbot.ERROR_SCOPES)
    assert chatbot._scope_of("memory_op") == "memory"
    assert chatbot._scope_of("reingest_source") == "sources"
    assert chatbot._scope_of("set_verbosity") == "settings"
    assert chatbot._scope_of("text_turn") == "turn"


def test_client_routes_on_scope_not_view(r):
    assert r["error_wiki_while_on_memory"] == {"memory": "", "wiki": "bad path raw/x", "status": ""}
    assert r["error_memory_slot"] == "Wait until the reply finishes."
    assert r["error_settings_status"] == {"status": "unknown setting", "wiki": "bad path raw/x"}
