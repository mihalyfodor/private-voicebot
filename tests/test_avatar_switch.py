import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import pytest
import avatars


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(avatars, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(avatars, "_current", None)
    monkeypatch.delenv("AVATAR", raising=False)
    yield


def test_current_key_falls_back_on_invalid_persisted_key(capsys):
    with open(avatars.SETTINGS_PATH, "w") as f:
        json.dump({"avatar": "miku"}, f)
    assert avatars.current_key() == "wanko"
    assert "miku" in capsys.readouterr().out


def test_set_current_persists_and_validates():
    avatars.set_current("natori")
    assert avatars.current()["name"] == "Natori"
    assert json.load(open(avatars.SETTINGS_PATH)) == {"avatar": "natori", "version": 1}
    avatars._current = None  # simulate restart
    assert avatars.current_key() == "natori"
    with pytest.raises(ValueError, match="wanko"):
        avatars.set_current("nope")


def test_llm_set_avatar_keeps_history():
    import llm
    llm.reset()
    llm._conversation += [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "[happy] Hello."}]
    avatars.set_current("haru")
    llm.set_avatar()
    assert "Haru" in llm._conversation[0]["content"]
    assert "from now on you are Haru" in llm._conversation[0]["content"]
    assert llm._conversation[1:] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "[happy] Hello."}]


def test_ws_set_avatar_switches_and_is_blocked_while_busy(monkeypatch):
    from fastapi.testclient import TestClient
    import chatbot
    from tests.conftest import Blocker

    # The switch greeting is a real turn; neutralise its body so the controller frees up.
    monkeypatch.setattr(chatbot, "switch_greet", lambda sess, turn, loop: None)
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hands_free"
        assert ws.receive_json()["type"] == "state"
        ws.send_json({"action": "set_avatar", "key": "natori"})
        assert ws.receive_json() == {"type": "avatar", "key": "natori", "name": "Natori"}
        assert chatbot.KOKORO_VOICE == "am_michael"
        assert client.get("/api/config").json()["avatar"] == "natori"
        chatbot.controller.join_idle(timeout=5)

        blocker = Blocker()
        blocker.hold(chatbot.controller)
        ws.send_json({"action": "set_avatar", "key": "haru"})
        assert ws.receive_json()["type"] == "error"
        assert avatars.current_key() == "natori"
        blocker.free(chatbot.controller)


def test_switch_claims_the_controller_immediately(monkeypatch):
    """submit() marks the controller busy synchronously, so the next switch is refused."""
    from fastapi.testclient import TestClient
    import chatbot
    from tests.conftest import Blocker

    hold = Blocker()
    monkeypatch.setattr(chatbot, "switch_greet", lambda sess, turn, loop: hold(None))
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # hands_free
        ws.receive_json()  # state idle
        ws.send_json({"action": "set_avatar", "key": "haru"})
        assert ws.receive_json()["type"] == "avatar"
        ws.send_json({"action": "set_avatar", "key": "natori"})
        assert ws.receive_json()["type"] == "error"
        hold.free(chatbot.controller)
