import os
import pytest
import yaml

import characters
import avatars


def test_load_all_loads_the_single_shipped_card():
    cards = characters.load_all()
    assert list(cards.keys()) == ["wanko"]
    assert cards["wanko"]["name"] == "Wanko"


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


def test_avatar_profile_carries_the_tagline():
    assert avatars.AVATARS["wanko"]["tagline"] == "Office assistant, upbeat"


# The word appears once in the card, in the rule that forbids it as a form of address.
_BOSS_PROHIBITION = 'never "boss" or pet names'


def test_system_prompt_never_tells_wanko_to_say_boss():
    """Scenario 2: the only mention of "boss" in the built prompt is the ban on using it."""
    import llm
    prompt = llm.build_system_prompt(avatars.current())
    assert _BOSS_PROHIBITION in prompt
    assert "boss" not in prompt.replace(_BOSS_PROHIBITION, "").lower()


def test_system_prompt_keeps_the_preferred_name_rule():
    """Scenario 2: the PRD 15 nickname rule still overrides the character's own habit."""
    import llm
    prompt = llm.build_system_prompt(avatars.current())
    assert "preferred name or nickname" in prompt
    assert "overrides any habit from your character" in prompt


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
