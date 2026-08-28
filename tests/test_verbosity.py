"""Tests for verbosity resolution, prompt rule text, and the WS set_verbosity action (scenario 4)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json

import pytest

import avatars
import llm


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(avatars, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(avatars, "_current", None)
    monkeypatch.delenv("AVATAR", raising=False)
    monkeypatch.delenv("VERBOSITY", raising=False)
    yield


def _fake_avatar(verbosity=None):
    return {"name": "Test", "persona": "You are Test.", "verbosity": verbosity}


def test_resolve_verbosity_defaults_to_normal():
    assert llm.resolve_verbosity(_fake_avatar()) == "normal"


def test_resolve_verbosity_card_beats_default():
    assert llm.resolve_verbosity(_fake_avatar("short")) == "short"


def test_resolve_verbosity_env_beats_card(monkeypatch):
    monkeypatch.setenv("VERBOSITY", "long")
    assert llm.resolve_verbosity(_fake_avatar("short")) == "long"


def test_resolve_verbosity_settings_beats_env(monkeypatch):
    monkeypatch.setenv("VERBOSITY", "long")
    avatars.save_setting("verbosity", "short")
    assert llm.resolve_verbosity(_fake_avatar("normal")) == "short"


def test_resolve_verbosity_invalid_saved_value_ignored(monkeypatch, capsys):
    avatars.save_setting("verbosity", "loud")
    assert llm.resolve_verbosity(_fake_avatar("short")) == "short"
    assert "warning" in capsys.readouterr().out.lower()


def test_build_system_prompt_contains_rule_text():
    prompt = llm.build_system_prompt(_fake_avatar("short"))
    assert llm.VERBOSITY_RULES["short"] in prompt

    prompt = llm.build_system_prompt(_fake_avatar("long"))
    assert llm.VERBOSITY_RULES["long"] in prompt


def test_rule_texts_match_prd():
    assert llm.VERBOSITY_RULES["short"] == "at most two sentences"
    assert llm.VERBOSITY_RULES["normal"] == (
        "two to four sentences; a little longer only when actually explaining something"
    )
    assert llm.VERBOSITY_RULES["long"] == (
        "as long as the answer needs, but still spoken prose — no lists or headings"
    )


def test_set_verbosity_invalid_raises():
    with pytest.raises(ValueError, match="short"):
        llm.set_verbosity("loud")


def test_set_verbosity_persists_and_rebuilds_prompt():
    llm.reset()
    llm.set_verbosity("long")
    assert avatars.load_settings()["verbosity"] == "long"
    assert llm.current_verbosity() == "long"
    assert llm.VERBOSITY_RULES["long"] in llm._conversation[0]["content"]


def test_ws_set_verbosity_broadcasts(monkeypatch):
    from fastapi.testclient import TestClient
    import chatbot
    chatbot.greeted = True
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # hands_free
        ws.receive_json()  # state idle
        ws.send_json({"action": "set_verbosity", "value": "long"})
        assert ws.receive_json() == {"type": "verbosity", "value": "long"}
        assert client.get("/api/config").json()["verbosity"] == "long"

        ws.send_json({"action": "set_verbosity", "value": "loud"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
