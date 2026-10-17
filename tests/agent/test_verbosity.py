"""Tests for verbosity resolution, prompt rule text, and the WS set_verbosity action (scenario 4)."""

import pytest

from core import settings
from agent import conversation
from agent import prompt as a_prompt
from agent import prompt as a_prompt


@pytest.fixture(autouse=True)
def no_verbosity_env(monkeypatch):
    """No VERBOSITY env, and the level a test sets is put back so it cannot leak into later modules."""
    monkeypatch.delenv("VERBOSITY", raising=False)
    before = a_prompt._verbosity, a_prompt.SYSTEM_PROMPT
    yield
    a_prompt._verbosity, a_prompt.SYSTEM_PROMPT = before
    conversation.reset()


def _fake_avatar(verbosity=None):
    return {"name": "Test", "persona": "You are Test.", "verbosity": verbosity}


def test_resolve_verbosity_defaults_to_normal():
    assert a_prompt.resolve_verbosity(_fake_avatar()) == "normal"


def test_resolve_verbosity_card_beats_default():
    assert a_prompt.resolve_verbosity(_fake_avatar("short")) == "short"


def test_resolve_verbosity_env_beats_card(monkeypatch):
    monkeypatch.setenv("VERBOSITY", "long")
    assert a_prompt.resolve_verbosity(_fake_avatar("short")) == "long"


def test_resolve_verbosity_settings_beats_env(monkeypatch):
    monkeypatch.setenv("VERBOSITY", "long")
    settings.save_setting("verbosity", "short")
    assert a_prompt.resolve_verbosity(_fake_avatar("normal")) == "short"


def test_resolve_verbosity_invalid_saved_value_ignored(monkeypatch, capsys):
    settings.save_setting("verbosity", "loud")
    assert a_prompt.resolve_verbosity(_fake_avatar("short")) == "short"
    assert "warning" in capsys.readouterr().out.lower()


def test_build_system_prompt_contains_rule_text():
    prompt = a_prompt.build_system_prompt(_fake_avatar("short"))
    assert a_prompt.VERBOSITY_RULES["short"] in prompt

    prompt = a_prompt.build_system_prompt(_fake_avatar("long"))
    assert a_prompt.VERBOSITY_RULES["long"] in prompt


def test_build_system_prompt_lets_a_preferred_name_beat_the_persona():
    prompt = a_prompt.build_system_prompt(_fake_avatar())
    assert "preferred name or nickname" in prompt
    assert "overrides any habit from your character" in prompt


def test_rule_texts_match_prd():
    assert a_prompt.VERBOSITY_RULES["short"] == "at most two sentences"
    assert a_prompt.VERBOSITY_RULES["normal"] == (
        "two to four sentences; a little longer only when actually explaining something"
    )
    assert a_prompt.VERBOSITY_RULES["long"] == (
        "as long as the answer needs, but still spoken prose — no lists or headings"
    )


def test_set_verbosity_invalid_raises():
    with pytest.raises(ValueError, match="short"):
        a_prompt.set_verbosity("loud")


def test_set_verbosity_persists_and_rebuilds_prompt():
    conversation.reset()
    a_prompt.set_verbosity("long")
    assert settings.load_settings()["verbosity"] == "long"
    assert a_prompt.current_verbosity() == "long"
    assert a_prompt.VERBOSITY_RULES["long"] in conversation._conversation[0]["content"]


def test_ws_set_verbosity_broadcasts(monkeypatch):
    from fastapi.testclient import TestClient
    from server import app
    from server import panels
    panels.greeted = True
    client = TestClient(app.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # hands_free
        ws.receive_json()  # state idle
        ws.send_json({"action": "set_verbosity", "value": "long"})
        assert ws.receive_json() == {"type": "verbosity", "value": "long"}
        assert client.get("/api/config").json()["verbosity"] == "long"

        ws.send_json({"action": "set_verbosity", "value": "loud"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
