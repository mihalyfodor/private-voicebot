import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import avatars


def test_default_is_wanko(monkeypatch):
    monkeypatch.delenv("AVATAR", raising=False)
    a = avatars.current()
    assert (a["key"], a["name"], a["voice"]) == ("wanko", "Wanko", "am_puck")


def test_listing_has_exactly_one_entry():
    """Scenario 1: Wanko is the only avatar."""
    listing = avatars.listing()
    assert [a["key"] for a in listing] == ["wanko"]


def test_unknown_avatar_falls_back_to_wanko(monkeypatch, capsys):
    """Scenario 1: an unknown key resolves to wanko, with one log line and no error."""
    monkeypatch.setenv("AVATAR", "orb")
    assert avatars.current()["key"] == "wanko"
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "orb" in out[0]


def test_settings_with_old_avatar_key_loads_as_wanko():
    """Scenario 1: a settings.json left over from the Natori era still boots."""
    import json
    with open(avatars.SETTINGS_PATH, "w") as f:
        json.dump({"avatar": "natori"}, f)
    assert avatars.current()["name"] == "Wanko"


def test_reload_keeps_working():
    avatars.set_current("wanko")
    assert avatars.reload()["key"] == "wanko"
    assert list(avatars.AVATARS) == ["wanko"]


def test_api_config_and_prompt(monkeypatch):
    monkeypatch.delenv("AVATAR", raising=False)
    from fastapi.testclient import TestClient
    import chatbot, llm
    chatbot.apply_avatar("wanko")
    cfg = TestClient(chatbot.app).get("/api/config").json()
    assert (cfg["avatar"], cfg["name"]) == ("wanko", "Wanko")
    assert [a["key"] for a in cfg["avatars"]] == ["wanko"]
    assert "Wanko" in llm.SYSTEM_PROMPT


def test_save_setting_is_atomic_and_versioned(monkeypatch):
    """Scenario 9: settings are written via a temp file + os.replace, stamped with a version."""
    import json, os as _os
    replaces = []
    real_replace = _os.replace

    def spy(src, dst):
        replaces.append((src, dst))
        real_replace(src, dst)

    monkeypatch.setattr(avatars.os, "replace", spy)
    avatars.save_setting("hands_free", True)

    assert replaces == [(avatars.SETTINGS_PATH + ".tmp", avatars.SETTINGS_PATH)]
    assert not _os.path.exists(avatars.SETTINGS_PATH + ".tmp")
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
