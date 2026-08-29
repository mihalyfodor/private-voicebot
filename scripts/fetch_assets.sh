#!/usr/bin/env bash
# Downloads browser libs and Live2D sample models into static/ (gitignored).
# Cubism Core and the sample models are Live2D-licensed and must not be committed.
set -euo pipefail
cd "$(dirname "$0")/.."

VENDOR=static/vendor
mkdir -p "$VENDOR"

echo "→ pixi.js v6"
curl -sSL -o "$VENDOR/pixi.min.js" https://cdn.jsdelivr.net/npm/pixi.js@6.5.10/dist/browser/pixi.min.js
echo "→ pixi-live2d-display (cubism4)"
curl -sSL -o "$VENDOR/cubism4.min.js" https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/cubism4.min.js
echo "→ Live2D Cubism Core"
curl -sSL -o "$VENDOR/live2dcubismcore.min.js" https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js

# fetch_model <target dir> <base url> <model3.json name>
fetch_model() {
  local dir=$1 base=$2 model3=$3
  echo "→ model $model3"
  mkdir -p "$dir"
  curl -sSL -o "$dir/$model3" "$base/$model3"
  python3 - "$dir" "$base" "$model3" <<'PY'
import json, os, sys, urllib.request, urllib.error
d, base, model3 = sys.argv[1:4]
refs = json.load(open(os.path.join(d, model3)))["FileReferences"]
optional = {refs.get("DisplayInfo")}
files = [refs.get("Moc"), refs.get("Physics"), refs.get("Pose"), refs.get("DisplayInfo")]
files += refs.get("Textures", [])
files += [e["File"] for e in refs.get("Expressions", [])]
for group in refs.get("Motions", {}).values():
    files += [m["File"] for m in group]
for f in filter(None, files):
    dst = os.path.join(d, f)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        urllib.request.urlretrieve(f"{base}/{f}", dst)
        print("   ", f)
    except urllib.error.HTTPError:
        if f in optional:
            print("   ", f, "(optional, skipped)")
        else:
            raise
PY
}

WANKO=static/models/wanko
WANKO_BASE=https://raw.githubusercontent.com/Live2D/CubismWebSamples/develop/Samples/Resources/Wanko
fetch_model "$WANKO" "$WANKO_BASE" Wanko.model3.json

# The upstream model3 wires only 5 of the 12 clips; PRD 27 groups all of them, so fetch the rest.
# The clip list lives in scripts/wanko_groups.py (MOTION_FILES) next to the groups it feeds.
echo "→ all Wanko motions"
mkdir -p "$WANKO/motions"
for m in $(python3 -c 'import sys; sys.path.insert(0, "scripts"); from wanko_groups import MOTION_FILES; print(*MOTION_FILES)'); do
  curl -sSfL -o "$WANKO/motions/$m.motion3.json" "$WANKO_BASE/motions/$m.motion3.json"
  echo "    motions/$m.motion3.json"
done

echo "→ Wanko.app.model3.json (PRD 27 motion groups)"
python3 scripts/wanko_groups.py "$WANKO"
echo "→ backdrops"
mkdir -p static/backdrops
python3 - <<'PY'
import urllib.request, os, sys
sys.path.insert(0, ".")
from backdrops import BACKDROPS
for b in BACKDROPS.values():
    if not b["url"]:
        continue
    dst = os.path.join("static/backdrops", b["file"])
    req = urllib.request.Request(b["url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        f.write(r.read())
    print("   ", b["file"])
PY
echo "done."
