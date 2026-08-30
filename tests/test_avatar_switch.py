"""Server-side avatar state. Since PRD 23 Wanko is the only avatar, so these cover
persistence and the fallback; the `set_avatar` action itself is gone."""
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


def test_set_avatar_action_no_longer_exists():
    """PRD 23 left `set_avatar` as a harmless leftover; it is gone now, as is the switch greeting."""
    import chatbot
    assert "set_avatar" not in chatbot.ACTIONS
    assert not hasattr(chatbot, "switch_greet")
    assert "switch_greeting" not in avatars.current()


def test_unknown_action_is_ignored_and_the_socket_stays_open():
    from fastapi.testclient import TestClient
    import chatbot
    import llm
    llm.reset()                                                  # nothing to replay on connect
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hands_free"
        assert ws.receive_json()["type"] == "state"
        ws.send_json({"action": "set_avatar", "key": "wanko"})
        ws.send_json({"action": "set_backdrop", "key": "none"})
        assert ws.receive_json() == {"type": "backdrop", "key": "none"}
    assert "avatar" not in avatars.load_settings()
