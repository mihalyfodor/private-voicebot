import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
import yaml

import characters
import avatars


def test_load_all_loads_three_shipped_cards():
    cards = characters.load_all()
    assert set(["wanko", "haru", "natori"]).issubset(cards.keys())
    assert cards["wanko"]["name"] == "Wanko"
    assert cards["haru"]["name"] == "Haru"
    assert cards["natori"]["name"] == "Natori"


def test_card_missing_name_raises_with_filename(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump({"voice": "am_puck"}))
    with pytest.raises(ValueError, match="broken.yaml"):
        characters.load_card(str(path))


def test_card_missing_voice_raises_with_filename(tmp_path):
    path = tmp_path / "broken2.yaml"
    path.write_text(yaml.safe_dump({"name": "Nameless"}))
    with pytest.raises(ValueError, match="broken2.yaml"):
        characters.load_card(str(path))


def test_build_persona_contains_all_sections():
    card = characters.load_card(os.path.join(characters.CHARACTERS_DIR, "wanko.yaml"))
    persona = characters.build_persona(card)

    assert "You are Wanko." in persona
    assert card["description"].strip() in persona
    for trait in card["personality"]:
        assert trait in persona
    assert "Speaking style:" in persona
    assert card["speaking_style"].strip() in persona
    assert "Scenario:" in persona
    assert card["scenario"].strip() in persona
    assert "Examples of your tone (do not repeat these lines verbatim):" in persona
    for ex in card["example_dialogue"]:
        assert f"User: {ex['user']}" in persona
        assert f"Wanko: {ex['assistant']}" in persona


def test_avatars_listing_uses_tagline():
    listing = avatars.listing()
    by_key = {a["key"]: a for a in listing}
    assert by_key["wanko"]["description"] == "Dog mascot, upbeat"
    assert by_key["haru"]["description"] == "Office assistant, calm"
    assert by_key["natori"]["description"] == "Office assistant, easygoing"


def test_persona_over_cap_prints_warning(tmp_path, capsys, monkeypatch):
    long_style = " ".join(["word"] * 300)
    card_dict = {
        "name": "Longwinded",
        "voice": "am_puck",
        "tagline": "Talks a lot",
        "speaking_style": long_style,
    }
    (tmp_path / "longwinded.yaml").write_text(yaml.safe_dump(card_dict))

    monkeypatch.setattr(characters, "CHARACTERS_DIR", str(tmp_path))
    cards = characters.load_all()
    out = capsys.readouterr().out
    assert "longwinded" in out
    assert "persona is" in out
    assert "cap 250" in out
    assert "Longwinded" in cards["longwinded"]["name"]
