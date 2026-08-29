# Wanko motion catalogue (2026-08-29)

Source: parsed `static/models/wanko/motions/*.motion3.json` curves + frames captured from the live app.
Only head/body angles, eyes, mouth-open, ears, breath (and paws once) are keyed by the clips. Dormant parameters
never touched by any clip: `BOWL_LID`, `YUGE_01/02` (steam), `EFFECT` (sparkle), `TERE` (blush), `FACE_01` (squint),
`MOUTH_FORM`, `BASE_X/Y`. `SWING`/`BOWL_SWING` are physics-driven. Upstream `Wanko.model3.json` wires only
`Idle: idle_01/03/04` and `TapBody: touch_02/04`; the other seven clips are orphaned until grouped (PRD 27).

| motion | dur | upstream group | top parameters | description | use |
|---|---|---|---|---|---|
| idle_01 | 6.0 s | Idle | ANGLE_Y −13→14, ANGLE_Z −10→13 | gentle bob, ear flick, one blink | idle |
| idle_02 | 6.9 s | — | ANGLE_Z 0→23, ANGLE_Y 0→15 | curious lean to one side, late ear twitch | idle |
| idle_03 | 6.7 s | Idle | ANGLE_Y 0→16, ANGLE_X −14→0, mouth 0→0.6 late | tilt down-left, straighten, tiny sniff/yawn | idle |
| idle_04 | 10.4 s | Idle | ANGLE_Y −23→14 late, BREATH 0→0.73 | calm, visible breathing, no blinks | idle / shutdown |
| shake_01 | 2.5 s | — | ANGLE_Z −25→30 at 0.27 s, ANGLE_X −18→20 | fast head shake, eyes pop open | "no" / dismiss |
| shake_02 | 2.5 s | — | ANGLE_Z −30→30 at 0.07 s, mouth 0→1, HAND_L/R 0→1 | startled gasp, both paws up | greeting |
| touch_01 | 2.0 s | — | ANGLE_Y 0→30, ANGLE_Z 0→30 hold | head up/back, big ear flick — "pet me" | click (big) |
| touch_02 | 2.0 s | TapBody | ANGLE_Y 0→25 | milder tilt-up, ear flick | click |
| touch_03 | 2.0 s | — | ANGLE_X −22→17, ANGLE_Y 0→17 | sideways turn — quizzical | thinking cue |
| touch_04 | 2.0 s | TapBody | ANGLE_Y −11→10 | small dip-then-rise nod | listening cue |
| touch_05 | 1.5 s | — | ANGLE_Y −11→10 | tiny bob, nothing else | rapid re-tap |
| touch_06 | 1.9 s | — | ANGLE_Y −30→18, ANGLE_Z −24→0 | biggest lean, ends off-centre — nuzzle/dodge | click (special) |
