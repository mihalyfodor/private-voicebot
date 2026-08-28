Status: done

# 10 — Character cards (SillyTavern-style personalities)

## Overview

Give each avatar a proper personality via an editable **character card** (YAML), in the spirit of SillyTavern's cards: description, personality traits, speaking style, scenario, example dialogue and a first greeting. The system prompt is composed from the card; the shared behavioural rules (emotion tags, tool use, brevity) stay in code so cards can't break the pipeline.

## User stories

- As a user, I can open `characters/wanko.yaml`, change how Wanko talks, restart, and hear the difference.
- As a user, each character greets me in their own way when the app opens or when I switch to them.
- As a user, I can add a fourth character by dropping a new card + avatar profile, without touching Python.
- As a user, the menu shows a one-line tagline per character taken from the card.

## UI/UX

- Menu avatar rows use `tagline` from the card instead of the hard-coded description.
- Switching avatars mid-chat: the new character says its `switch_greeting` (short, in character) — instead of silently taking over.
- No in-app editor in this slice (edit the YAML; a "Reload characters" item in the menu re-reads cards without restart).

## Technical approach

`characters/<key>.yaml` (committed; these are ours):

```yaml
name: Wanko
tagline: Dog mascot, upbeat
voice: am_puck
speed: 1.05
description: >
  A small white dog who lives in a rice bowl and somehow got a job as the office assistant.
personality: [cheerful, loyal, easily excited, a little scatterbrained but tries hard]
speaking_style: >
  Short upbeat sentences. Occasionally a dog-ish aside ("ooh, is that lunch?"), never more than one per reply.
  Calls the user "boss".
scenario: >
  A quiet open-plan office, mid-morning. Wanko sits on the desk next to the user's monitor.
example_dialogue:
  - user: any news?
    assistant: "[happy] Ooh, let me sniff out the headlines, boss."
  - user: what's 15% of 80
    assistant: "[thinking] Twelve! I counted on my paws."
greeting: "[happy] Morning, boss! Wanko reporting for duty. What are we doing today?"
switch_greeting: "[happy] Wanko here! I've got it from here, boss."
```

- `characters.py`: `load_all()` → dict; validation (required: name, voice; lists/strings typed); `build_persona(card)` → text block (description, personality, speaking style, scenario, examples formatted as `User:`/`{name}:` lines).
- `avatars.py`: `AVATARS` becomes derived from cards (`key` = file stem); `name`, `voice`, `speed`, `description`(=tagline) read from the card. Client `PROFILES` unchanged (visual only).
- `llm.py`: `build_system_prompt(card)` = `build_persona(card)` + fixed rules (memory note, ≤2 sentences, plain text, emotion tags, tool rules). Keep `set_avatar()`.
- `chatbot.py`: greeting flow uses `card.greeting` verbatim (spoken via TTS, no LLM call → instant start); on `set_avatar` speak `switch_greeting` verbatim. `reload_characters` WS action.
- Examples go into the system prompt as text (not as fake turns) to keep history clean.

### Decisions

- YAML cards over JSON/TavernAI PNG cards: human-editable, diff-able; importing real Tavern cards is out of scope (could be a converter later).
- Persona in the prompt, rules in code: cards can't remove the emotion tag or tool rules.
- Greetings are canned strings, not generated: instant, in-character, and they cost nothing; the LLM takes over from the first real turn.
- Risk: long cards + history eat context/latency on the 26B; cap `build_persona` at ~250 words and warn on load if exceeded.
- Risk: Gemma parroting example lines; mitigate with "these are examples of tone, do not repeat them".

## Data model

`characters/*.yaml` as above; `avatars.AVATARS[key]` = card + key. No change to `settings.json` or the WS protocol except the new `reload_characters` action and the `switch_greeting` being sent as a normal `speech` chunk.

## Test scenarios

1. `characters.load_all()` loads the three shipped cards; a card missing `name` raises with the file name in the message.
2. `build_persona(card)` contains description, every personality trait, speaking style, scenario and the example lines formatted as `User:` / `Wanko:`.
3. `llm.build_system_prompt(card)` still contains the emotion-tag list, the tool rules and the ≤2 sentences rule regardless of card content.
4. `avatars.listing()` uses `tagline` from the cards.
5. Greeting: on first WS connect the server emits the card's `greeting` as a `speech` chunk (mocked TTS) without calling the LLM.
6. `set_avatar` emits `switch_greeting` of the new character after the `avatar` message.
7. Persona longer than the cap → warning printed on load (captured).

Manual: edit Wanko's `speaking_style`, use "Reload characters", ask something → tone changes; switch to Natori → hears his switch greeting.

## Out of scope

In-app card editor; importing TavernAI/Chub PNG cards; per-character memory files; lorebooks/world info; multiple characters in one scene.
