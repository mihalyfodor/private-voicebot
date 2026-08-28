#!/usr/bin/env bash
# Downloads browser libs and the Live2D "Haru" sample model into static/ (gitignored).
# Cubism Core and the sample model are Live2D-licensed and must not be committed.
set -euo pipefail
cd "$(dirname "$0")/.."

VENDOR=static/vendor
MODEL_DIR=static/models/haru
HARU_BASE="https://raw.githubusercontent.com/guansss/pixi-live2d-display/master/test/assets/haru"

mkdir -p "$VENDOR" "$MODEL_DIR"

echo "→ pixi.js v6"
curl -sSL -o "$VENDOR/pixi.min.js" https://cdn.jsdelivr.net/npm/pixi.js@6.5.10/dist/browser/pixi.min.js
echo "→ pixi-live2d-display (cubism4)"
curl -sSL -o "$VENDOR/cubism4.min.js" https://cdn.jsdelivr.net/npm/pixi-live2d-display@0.4.0/dist/cubism4.min.js
echo "→ Live2D Cubism Core"
curl -sSL -o "$VENDOR/live2dcubismcore.min.js" https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js

echo "→ Haru model"
curl -sSL -o "$MODEL_DIR/haru_greeter_t03.model3.json" "$HARU_BASE/haru_greeter_t03.model3.json"
python3 - "$MODEL_DIR" "$HARU_BASE" <<'PY'
import json, os, sys, urllib.request
d, base = sys.argv[1], sys.argv[2]
refs = json.load(open(os.path.join(d, "haru_greeter_t03.model3.json")))["FileReferences"]
files = [refs.get("Moc"), refs.get("Physics"), refs.get("Pose")]
optional = {refs.get("DisplayInfo")}
files.append(refs.get("DisplayInfo"))
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
    except urllib.error.HTTPError as e:
        if f in optional:
            print("   ", f, "(optional, skipped)")
        else:
            raise
PY
echo "done."
