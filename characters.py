"""Character cards: load SillyTavern-style YAML cards and build persona prompt text."""
import os

import yaml

CHARACTERS_DIR = os.path.join(os.path.dirname(__file__), "characters")
PERSONA_WORD_CAP = 250
VALID_VERBOSITY = ("short", "normal", "long")

# Files loaded first, in this order, so downstream iteration order is deterministic;
# any additional cards follow alphabetically by key.
_PRIORITY_ORDER = ["wanko", "haru", "natori"]


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        # allow a plain string as a single-item/comma-free personality description
        return [value.strip()] if value.strip() else []
    return [value]


def load_card(path: str) -> dict:
    fname = os.path.basename(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    name = raw.get("name")
    voice = raw.get("voice")
    if not name:
        raise ValueError(f"{fname}: missing required field 'name'")
    if not voice:
        raise ValueError(f"{fname}: missing required field 'voice'")

    verbosity = raw.get("verbosity")
    if verbosity is not None and verbosity not in VALID_VERBOSITY:
        raise ValueError(
            f"{fname}: invalid verbosity {verbosity!r}; valid: {', '.join(VALID_VERBOSITY)}"
        )

    example_dialogue = []
    for item in raw.get("example_dialogue") or []:
        example_dialogue.append({
            "user": (item or {}).get("user", ""),
            "assistant": (item or {}).get("assistant", ""),
        })

    return {
        "name": name,
        "tagline": raw.get("tagline", ""),
        "voice": voice,
        "speed": raw.get("speed", 1.0),
        "description": raw.get("description", "") or "",
        "personality": _as_list(raw.get("personality")),
        "speaking_style": raw.get("speaking_style", "") or "",
        "scenario": raw.get("scenario", "") or "",
        "example_dialogue": example_dialogue,
        "greeting": raw.get("greeting", "") or "",
        "switch_greeting": raw.get("switch_greeting", "") or "",
        "verbosity": verbosity,
    }


def _sort_key(key: str):
    if key in _PRIORITY_ORDER:
        return (0, _PRIORITY_ORDER.index(key))
    return (1, key)


def load_all() -> dict:
    cards = {}
    if not os.path.isdir(CHARACTERS_DIR):
        return cards
    keys = []
    for fname in os.listdir(CHARACTERS_DIR):
        if not fname.endswith((".yaml", ".yml")):
            continue
        key = os.path.splitext(fname)[0]
        cards[key] = load_card(os.path.join(CHARACTERS_DIR, fname))
        keys.append(key)

    ordered = {}
    for key in sorted(keys, key=_sort_key):
        card = cards[key]
        ordered[key] = card
        persona = build_persona(card)
        word_count = len(persona.split())
        if word_count > PERSONA_WORD_CAP:
            print(f"[characters] {key}: persona is {word_count} words (cap {PERSONA_WORD_CAP})")
    return ordered


def build_persona(card: dict) -> str:
    name = card["name"]
    parts = [f"You are {name}."]

    description = (card.get("description") or "").strip()
    if description:
        parts.append(description)

    personality = card.get("personality") or []
    if personality:
        parts.append("Personality: " + ", ".join(str(p) for p in personality) + ".")

    speaking_style = (card.get("speaking_style") or "").strip()
    if speaking_style:
        parts.append(f"Speaking style: {speaking_style}")

    scenario = (card.get("scenario") or "").strip()
    if scenario:
        parts.append(f"Scenario: {scenario}")

    examples = card.get("example_dialogue") or []
    if examples:
        lines = ["Examples of your tone (do not repeat these lines verbatim):"]
        for ex in examples:
            lines.append(f"User: {ex.get('user', '')}")
            lines.append(f"{name}: {ex.get('assistant', '')}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)
