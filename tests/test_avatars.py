"""Server-side avatar state and backdrops. Since PRD 23 Wanko is the only avatar, so these
cover persistence and the fallback; the `set_avatar` action itself is gone."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
import pytest
from fastapi.testclient import TestClient

import avatars, backdrops
from tests.conftest import drain_initial


def test_default_is_wanko(monkeypatch):
    monkeypatch.delenv("AVATAR", raising=False)
    a = avatars.current()
    assert (a["key"], a["name"], a["voice"]) == ("wanko", "Wanko", "am_puck")


def test_wanko_is_the_only_avatar():
    """Scenario 1: Wanko is the only avatar (no listing any more — nothing to pick from)."""
    assert list(avatars.AVATARS) == ["wanko"]
    assert not hasattr(avatars, "listing")


def _via_env(monkeypatch, key):
    monkeypatch.setenv("AVATAR", key)
    return avatars.current()


def _via_settings(monkeypatch, key):
    with open(avatars.SETTINGS_PATH, "w") as f:
        json.dump({"avatar": key}, f)
    return avatars.current()


def _via_set_current(monkeypatch, key):
    return avatars.set_current(key)


@pytest.mark.parametrize("source, key", [
    (_via_env, "orb"),               # an old AVATAR env value
    (_via_settings, "natori"),       # a settings.json left over from the Natori era
    (_via_set_current, "haru"),      # set_current() with a retired key
], ids=["env", "settings", "set_current"])
def test_unknown_or_retired_avatar_key_resolves_to_wanko(monkeypatch, capsys, source, key):
    """Scenario 1: every way of naming an unknown avatar resolves to wanko with one log line
    and no error (all three go through `_resolve`)."""
    a = source(monkeypatch, key)
    assert (a["key"], a["name"]) == ("wanko", "Wanko")
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and key in out[0]
    if source is _via_set_current:
        assert json.load(open(avatars.SETTINGS_PATH))["avatar"] == "wanko"


def test_reload_keeps_working():
    avatars.set_current("wanko")
    assert avatars.reload()["key"] == "wanko"
    assert list(avatars.AVATARS) == ["wanko"]


def test_api_config_and_prompt(monkeypatch):
    monkeypatch.delenv("AVATAR", raising=False)
    import chatbot, llm
    chatbot.apply_avatar("wanko")
    cfg = TestClient(chatbot.app).get("/api/config").json()
    assert (cfg["avatar"], cfg["name"]) == ("wanko", "Wanko")
    assert "avatars" not in cfg
    assert "Wanko" in llm.SYSTEM_PROMPT


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


# --- set_current / settings.json


def test_set_current_persists():
    avatars.set_current("wanko")
    assert avatars.current()["name"] == "Wanko"
    assert json.load(open(avatars.SETTINGS_PATH)) == {"avatar": "wanko", "version": 1}
    avatars._current = None  # simulate restart
    assert avatars.current_key() == "wanko"


def test_save_setting_is_atomic_and_versioned(monkeypatch):
    """Scenario 9: settings are written via a temp file + os.replace, stamped with a version."""
    replaces = []
    real_replace = os.replace

    def spy(src, dst):
        replaces.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(avatars.os, "replace", spy)
    avatars.save_setting("hands_free", True)

    assert replaces == [(avatars.SETTINGS_PATH + ".tmp", avatars.SETTINGS_PATH)]
    assert not os.path.exists(avatars.SETTINGS_PATH + ".tmp")
    assert json.load(open(avatars.SETTINGS_PATH)) == {"hands_free": True, "version": 1}
    assert avatars.load_settings()["version"] == avatars.SETTINGS_VERSION


def test_truncated_settings_file_loads_as_empty_with_a_warning(capsys):
    with open(avatars.SETTINGS_PATH, "w") as f:
        f.write('{"avatar": "nat')
    assert avatars.load_settings() == {}
    assert "warning" in capsys.readouterr().out.lower()


def test_missing_settings_file_is_silently_empty(capsys):
    assert avatars.load_settings() == {}
    assert capsys.readouterr().out == ""


# --- backdrops


def test_validate_and_listing():
    assert backdrops.validate("none") == "none"
    with pytest.raises(ValueError, match="none"):
        backdrops.validate("mars")
    assert backdrops.listing()[0] == {"key": "none", "name": "None", "file": None, "credit": ""}


def test_ws_set_backdrop_persists_and_survives_avatar_switch():
    import chatbot
    import llm
    llm.reset()                                                  # nothing to replay on connect
    client = TestClient(chatbot.app)
    key = list(backdrops.BACKDROPS)[-1]
    with client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        ws.send_json({"action": "set_backdrop", "key": key})
        assert ws.receive_json() == {"type": "backdrop", "key": key}
        ws.send_json({"action": "set_backdrop", "key": "mars"})
        assert ws.receive_json()["type"] == "error"
    assert avatars.load_settings() == {"backdrop": key, "version": 1}
    assert client.get("/api/config").json()["backdrop"] == key


def test_unknown_action_is_ignored_and_the_socket_stays_open():
    import chatbot
    import llm
    llm.reset()                                                  # nothing to replay on connect
    client = TestClient(chatbot.app)
    with client.websocket_connect("/ws") as ws:
        assert [m["type"] for m in drain_initial(ws)] == ["hands_free", "state"]
        ws.send_json({"action": "set_avatar", "key": "wanko"})
        ws.send_json({"action": "set_backdrop", "key": "none"})
        assert ws.receive_json() == {"type": "backdrop", "key": "none"}
    assert "avatar" not in avatars.load_settings()
