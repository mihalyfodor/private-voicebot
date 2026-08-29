"""Generate `Wanko.app.model3.json` — the upstream model with our motion groups (PRD 27).

The model folder is gitignored, so the groups can't live in a checked-in model3.json; they live
here and `scripts/fetch_assets.sh` regenerates the `.app.` file after every fetch. app.js loads
`Wanko.app.model3.json` when it is present and falls back to the upstream file otherwise.

    python3 scripts/wanko_groups.py static/models/wanko
"""
import json
import os
import sys

#: All 12 clips Wanko ships with; fetch_assets.sh downloads exactly this list.
MOTION_FILES = [
    "idle_01", "idle_02", "idle_03", "idle_04",
    "shake_01", "shake_02",
    "touch_01", "touch_02", "touch_03", "touch_04", "touch_05", "touch_06",
]

#: group → clips, see docs/prototypes/wanko-motions.md for what each clip does.
GROUPS = {
    "Idle": ["idle_01", "idle_02", "idle_03", "idle_04"],
    "Pet": ["touch_01", "touch_02", "touch_06"],        # big reactions
    "PetSmall": ["touch_05", "touch_04"],               # rapid re-tap
    "Greet": ["shake_02"],                              # startled gasp, paws up
    "No": ["shake_01"],                                 # head shake
    "Thinking": ["touch_03"],                           # quizzical sideways turn
    "Listening": ["touch_04"],                          # small nod
}

SRC_NAME = "Wanko.model3.json"
DST_NAME = "Wanko.app.model3.json"


def build(src: dict) -> dict:
    """Upstream model3 with `FileReferences.Motions` replaced by GROUPS."""
    out = json.loads(json.dumps(src))
    out["FileReferences"]["Motions"] = {
        group: [{"File": f"motions/{clip}.motion3.json"} for clip in clips]
        for group, clips in GROUPS.items()
    }
    return out


def write(model_dir: str) -> str:
    """Read `model_dir/Wanko.model3.json`, write the `.app.` variant next to it."""
    with open(os.path.join(model_dir, SRC_NAME)) as f:
        src = json.load(f)
    dst = os.path.join(model_dir, DST_NAME)
    with open(dst, "w") as f:
        json.dump(build(src), f, indent="\t")
        f.write("\n")
    return dst


if __name__ == "__main__":
    print("   ", write(sys.argv[1] if len(sys.argv) > 1 else "static/models/wanko"))
