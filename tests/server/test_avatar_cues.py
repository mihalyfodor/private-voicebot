"""PRD 27 scenarios: Wanko motion groups, click-to-pet, state cues and the new emotions.

Scenario 1 (the generated `.app.model3.json`) is checked here directly; scenarios 2 and 4
run static/app.js in node against the DOM/PIXI shim in tests/app_fixture.mjs and this file
only asserts on the JSON it prints. Scenario 3 (the emotion whitelist) lives in
tests/server/test_splitter.py plus `test_new_emotions_reach_the_system_prompt` below.
"""
import sys, os
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import wanko_groups

from agent import prompt as m_prompt
from server import speech

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
MODEL_DIR = os.path.join(ROOT, "static", "models", "wanko")

GROUPS = wanko_groups.GROUPS


# ---------------------------------------------------------------- scenario 1: the generator


def test_generated_model3_has_the_seven_groups_and_keeps_the_rest():
    src = {
        "Version": 3,
        "FileReferences": {"Moc": "Wanko.moc3", "Textures": ["t.png"],
                           "Motions": {"Idle": [{"File": "motions/idle_01.motion3.json"}]}},
        "Groups": [{"Target": "Parameter", "Name": "EyeBlink", "Ids": ["PARAM_EYE_L_OPEN"]}],
        "HitAreas": [{"Id": "HitArea", "Name": "Body"}],
    }
    out = wanko_groups.build(src)
    motions = out["FileReferences"]["Motions"]
    assert set(GROUPS) == {"Idle", "Pet", "PetSmall", "Greet", "No", "Thinking", "Listening"}
    assert list(motions) == list(GROUPS)
    for group, clips in GROUPS.items():
        assert clips and [m["File"] for m in motions[group]] == [f"motions/{c}.motion3.json" for c in clips]
    assert out["FileReferences"]["Motions"]["Pet"] == [
        {"File": "motions/touch_01.motion3.json"},
        {"File": "motions/touch_02.motion3.json"},
        {"File": "motions/touch_06.motion3.json"},
    ]
    assert out["Groups"] == src["Groups"] and out["HitAreas"] == src["HitAreas"]
    assert out["FileReferences"]["Moc"] == "Wanko.moc3"
    assert src["FileReferences"]["Motions"] == {"Idle": [{"File": "motions/idle_01.motion3.json"}]}


def test_every_grouped_clip_is_one_of_the_twelve_fetched_motions():
    grouped = {c for clips in wanko_groups.GROUPS.values() for c in clips}
    assert grouped <= set(wanko_groups.MOTION_FILES)
    assert set(wanko_groups.MOTION_FILES) - grouped == set()   # all 12 are used
    fetch = open(os.path.join(ROOT, "scripts", "fetch_assets.sh")).read()
    assert "from wanko_groups import MOTION_FILES" in fetch   # the clip list is not duplicated
    assert "scripts/wanko_groups.py" in fetch


@pytest.mark.skipif(not os.path.isfile(os.path.join(MODEL_DIR, wanko_groups.SRC_NAME)),
                    reason="model not fetched (scripts/fetch_assets.sh)")
def test_generated_model3_only_references_files_that_exist(tmp_path):
    """Scenario 1: run the generator against the fetched model, every File must be on disk."""
    dst = wanko_groups.write(MODEL_DIR)
    assert dst.endswith("Wanko.app.model3.json")
    refs = json.load(open(dst))["FileReferences"]
    files = [refs["Moc"], refs.get("Physics"), refs.get("DisplayInfo"), *refs["Textures"]]
    for group in refs["Motions"].values():
        files += [m["File"] for m in group]
    for f in filter(None, files):
        assert os.path.isfile(os.path.join(MODEL_DIR, f)), f
    assert len(refs["Motions"]) == 7


# ---------------------------------------------------------------- scenario 3: emotions


def test_new_emotions_reach_the_system_prompt():
    prompt = m_prompt.build_system_prompt({"persona": "p", "name": "Wanko", "verbosity": "normal"})
    for e in ("proud", "sleepy", "shy"):
        assert e in speech.EMOTIONS
        assert f"[{e}]" in prompt, e
    card = open(os.path.join(ROOT, "characters", "wanko.yaml")).read()
    for e in ("proud", "sleepy", "shy"):
        assert f"[{e}]" in card, e


def test_every_emotion_tag_has_an_expression_in_app_js():
    app_js = open(os.path.join(ROOT, "static", "app.js")).read()
    for e in speech.EMOTIONS:
        if e == "neutral":
            continue
        assert f"      {e}:" in app_js, e


# ---------------------------------------------------------------- scenarios 2 & 4: the client


def test_app_js_loads_the_generated_model_and_sees_every_group(r):
    """Scenario 2: the groups app.js plays are exactly the ones the generator writes."""
    assert r["model_url"] == "/static/models/wanko/Wanko.app.model3.json"
    assert r["motion_groups"] == sorted(GROUPS)
    app_js = open(os.path.join(ROOT, "static", "app.js")).read()
    table = app_js.split("const MOTIONS = {")[1].split("\n};")[0]
    assert sorted(l.split(":")[0].strip() for l in table.strip().splitlines()) == sorted(GROUPS)


def test_tapping_wanko_pets_her_big_then_small(r):
    """Scenario 4: click → Pet, a click within 4 s → PetSmall, after the gap → Pet again."""
    assert r["tap_groups"] == ["Pet", "PetSmall"]      # the third tap missed the Body hit area
    assert r["tap_after_pause"] == "Pet"
    assert r["pet_emotion"] == "happy"                 # the 800 ms happy blend rides along


def test_idle_never_repeats_and_reaches_every_clip(r):
    assert r["idle_repeats"] == 0
    assert r["idle_indices"] == [0, 1, 2, 3]
    assert r["idle_priority"] == 1                     # MotionPriority.IDLE


def test_thinking_sets_the_lid_and_steam_until_the_first_speech(r):
    """Scenario 4: state thinking → lid + steam targets; the first speech clears them."""
    assert r["thinking_cue"]["target"] == {"PARAM_BOWL_LID": 40, "PARAM_EFFECT": 0.4}
    assert r["thinking_cue"]["pulse"] == ["PARAM_YUGE_01", "PARAM_YUGE_02"]
    assert r["thinking_motion"] == 1                   # the Thinking clip plays once
    # a tool filler sparkles on top, the real reply clears everything
    assert r["filler_cue"]["target"] == {"PARAM_BOWL_LID": 40, "PARAM_EFFECT": 1}   # the tool cue wins
    assert r["speaking_cue"] == {"target": {}, "pulse": [], "active": ["greet"]}


def test_greeting_fires_once_per_session(r):
    assert r["greet_calls"] == 1                       # on the first speech of the session
    assert r["greet_calls_after"] == 0                 # and never again


def test_hearing_plays_the_listening_clip_once_and_lifts_the_ears(r):
    assert r["listening_motions"] == ["Listening"]     # the second `hearing` is not a transition
    assert r["listening_cue"]["target"] == {"PARAM_EAR_L": 1, "PARAM_EAR_R": 1}
    assert r["listening_cue_after"]["target"] == {}


def test_unknown_emotion_tag_is_ignored(r):
    assert r["emotion_proud"] == {"name": "proud", "target": {
        "PARAM_MOUTH_FORM": 1, "PARAM_EFFECT": 1, "PARAM_ANGLE_Y": 6}}
    assert r["emotion_unknown"] == "proud"             # the unknown tag changed nothing


def test_shutdown_forces_the_slow_idle_and_closes_the_lid(r):
    assert r["shutdown_motion"] == [["Idle", 3, 3]]    # idle_04 at MotionPriority.FORCE
    assert r["shutdown_cue"]["target"] == {"PARAM_BOWL_LID": -10}   # -10 = lid on the bowl
