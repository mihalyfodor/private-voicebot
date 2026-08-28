import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import avatars


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(avatars, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(avatars, "_current", None)
    monkeypatch.delenv("AVATAR", raising=False)
    yield


def test_default_is_wanko(monkeypatch):
    monkeypatch.delenv("AVATAR", raising=False)
    a = avatars.current()
    assert (a["key"], a["name"], a["voice"]) == ("wanko", "Wanko", "am_puck")


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
    chatbot.apply_avatar("wanko")
    cfg = TestClient(chatbot.app).get("/api/config").json()
    assert (cfg["avatar"], cfg["name"]) == ("wanko", "Wanko")
    assert [a["key"] for a in cfg["avatars"]] == ["wanko", "haru", "natori"]
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
