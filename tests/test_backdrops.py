import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import avatars, backdrops


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(avatars, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(avatars, "_current", None)
    monkeypatch.delenv("AVATAR", raising=False)
    yield


def test_validate_and_listing():
    assert backdrops.validate("none") == "none"
    with pytest.raises(ValueError, match="none"):
        backdrops.validate("mars")
    assert backdrops.listing()[0] == {"key": "none", "name": "None", "file": None, "credit": ""}


def test_ws_set_backdrop_persists_and_survives_avatar_switch():
    from fastapi.testclient import TestClient
    import chatbot
    chatbot.greeted = True
    client = TestClient(chatbot.app)
    key = list(backdrops.BACKDROPS)[-1]
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # hands_free
        ws.receive_json()  # state idle
        ws.send_json({"action": "set_backdrop", "key": key})
        assert ws.receive_json() == {"type": "backdrop", "key": key}
        ws.send_json({"action": "set_backdrop", "key": "mars"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"action": "set_avatar", "key": "haru"})
        ws.receive_json()
    assert avatars.load_settings() == {"backdrop": key, "avatar": "haru"}
    assert client.get("/api/config").json()["backdrop"] == key
