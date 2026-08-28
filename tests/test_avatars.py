import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import avatars


def test_default_is_wanko(monkeypatch):
    monkeypatch.delenv("AVATAR", raising=False)
    a = avatars.current()
    assert (a["key"], a["name"], a["voice"]) == ("wanko", "Wanko", "bm_lewis")


def test_haru_selectable(monkeypatch):
    monkeypatch.setenv("AVATAR", "haru")
    a = avatars.current()
    assert (a["name"], a["voice"]) == ("Haru", "af_sarah")


def test_unknown_avatar_raises(monkeypatch):
    monkeypatch.setenv("AVATAR", "orb")
    with pytest.raises(ValueError, match="wanko"):
        avatars.current()


def test_api_config_and_prompt(monkeypatch):
    monkeypatch.delenv("AVATAR", raising=False)
    from fastapi.testclient import TestClient
    import chatbot, llm
    assert TestClient(chatbot.app).get("/api/config").json() == {"avatar": "wanko", "name": "Wanko"}
    assert "Wanko" in llm.SYSTEM_PROMPT
