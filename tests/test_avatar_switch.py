"""Server-side avatar state. Since PRD 23 Wanko is the only avatar, so these cover
persistence and the fallback rather than switching between characters."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import pytest
import avatars


def test_set_current_persists():
    avatars.set_current("wanko")
    assert avatars.current()["name"] == "Wanko"
    assert json.load(open(avatars.SETTINGS_PATH)) == {"avatar": "wanko", "version": 1}
    avatars._current = None  # simulate restart
    assert avatars.current_key() == "wanko"


def test_set_current_with_a_retired_key_resolves_to_wanko(capsys):
    """Scenario 1: set_current("haru") returns wanko instead of raising."""
    assert avatars.set_current("haru")["key"] == "wanko"
    assert "haru" in capsys.readouterr().out
    assert json.load(open(avatars.SETTINGS_PATH))["avatar"] == "wanko"


def test_llm_set_avatar_keeps_history():
    import llm
    llm.reset()
    llm._conversation += [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "[happy] Hello."}]
    avatars.set_current("wanko")
    llm.set_avatar()
    assert "Wanko" in llm._conversation[0]["content"]
    assert llm._conversation[1:] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "[happy] Hello."}]


def test_ws_set_avatar_is_blocked_while_busy(monkeypatch):
    """The action stays server-side (the UI no longer sends it) and still guards the controller."""
    from fastapi.testclient import TestClient
    import chatbot
    from tests.conftest import Blocker

    # The switch greeting is a real turn; neutralise its body so the controller frees up.
    monkeypatch.setattr(chatbot, "switch_greet", lambda sess, turn, loop: None)
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hands_free"
        assert ws.receive_json()["type"] == "state"
        ws.send_json({"action": "set_avatar", "key": "wanko"})
        assert ws.receive_json() == {"type": "avatar", "key": "wanko", "name": "Wanko"}
        assert chatbot.KOKORO_VOICE == "am_puck"
        assert client.get("/api/config").json()["avatar"] == "wanko"
        chatbot.controller.join_idle(timeout=5)

        blocker = Blocker()
        blocker.hold(chatbot.controller)
        ws.send_json({"action": "set_avatar", "key": "wanko"})
        assert ws.receive_json()["type"] == "error"
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
        ws.send_json({"action": "set_avatar", "key": "wanko"})
        assert ws.receive_json()["type"] == "avatar"
        ws.send_json({"action": "set_avatar", "key": "wanko"})
        assert ws.receive_json()["type"] == "error"
        hold.free(chatbot.controller)
