Status: implementation

# 09 — Hands-free conversation mode

## Overview

Add a menu toggle **Hands-free** that lets the user talk without pressing anything: the app listens continuously, detects the end of each utterance with a VAD, transcribes, answers, and listens again. To make this work on one machine with speakers, microphone capture moves into the browser (Chrome's built-in echo cancellation), and push-to-talk uses the same path.

## User stories

- As a user, I switch on Hands-free in ☰ and just talk; the assistant replies after I stop speaking, without me touching the keyboard.
- As a user, the assistant does not reply to its own voice or to a cough/keyboard click.
- As a user, I can still use push-to-talk (Space) when Hands-free is off.
- As a user, the choice persists across restarts.

## UI/UX

- Menu: new section **Listening** with a toggle "Hands-free". Persisted in `settings.json`.
- Status line while hands-free: `listening…` (idle) → `hearing you…` (VAD active, mic button pulses softly) → `thinking…` → `speaking…` → `listening…`.
- First activation asks for microphone permission (browser prompt). If denied, the toggle reverts with an error status.
- Speak button remains: in hands-free it shows a mic icon and acts as a mute toggle.

## Technical approach

**Browser** (`static/mic-worklet.js`, `static/app.js`)
- `getUserMedia({audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true}})`, played TTS stays on the same `AudioContext` so Chrome's AEC has the far-end reference.
- `AudioWorklet` downsamples to 16 kHz Int16 and posts 512-sample frames (32 ms); `ws.send(ArrayBuffer)`.
- PTT: same stream, but frames are only sent while recording is toggled on (server treats as a single utterance on stop).
- Handles `{"type":"listening", value}` for status and (later) `stop_playback`.

**Server**
- `vad.py` — `Endpointer`: Silero VAD (ONNX, `silero-vad` package) + 300 ms pre-roll ring buffer + state machine:
  `IDLE` → speech prob > threshold for ≥ 2 frames → `LISTENING` → silence ≥ `end_silence_ms` → utterance if speech ≥ `min_speech_ms`, else discard → `IDLE`. Hard cap `max_utterance_s`.
  Defaults: `threshold 0.5`, `end_silence_ms 700`, `min_speech_ms 300`, `pre_roll_ms 300`, `max_utterance_s 30`.
- `asr.py` — `transcribe(audio: np.ndarray) -> str` via `mlx-whisper` (`mlx-community/whisper-small-mlx`), in-process, replaces the `whisper-cli` subprocess for both modes. Junk filter: drop empty, `[BLANK_AUDIO]`, or < 2 words with low avg logprob.
- `chatbot.py` — websocket accepts binary frames; hands-free feeds `Endpointer`, which is **gated while `processing`/`speaking`** (v1: no barge-in). Utterance → existing `respond()`. `set_hands_free` action persisted like avatar/backdrop. `sounddevice` capture removed.

### Decisions

- Browser mic over server mic: the only practical way to get echo cancellation, since playback happens in the browser; also gives noise suppression/AGC for free and unifies PTT and hands-free.
- Silero over webrtcvad: neural VAD is far more robust to keyboard/fan noise at negligible CPU; MIT.
- `mlx-whisper` over `whisper-cli`: accepts arrays (no temp wav / process spawn), ~0.2–0.5 s per utterance on Apple Silicon.
- No streaming ASR: post-utterance transcription is faster than the 700 ms silence window, so streaming adds complexity for no visible gain.
- v1 gates the mic while the assistant speaks; barge-in (interrupting) is a follow-up PRD once the AEC residual has been observed on the user's hardware.
- Assume Chrome; Safari's AEC/worklet quality is weaker. Bluetooth headsets may break AEC alignment — documented, not handled.
- Risk: whisper hallucinating on near-silent clips → junk filter + `min_speech_ms`.
- Risk: `respond()` blocks on `playback_done`; fine for v1 since VAD is gated during that time.

## Data model / state shape

`settings.json`: `{"hands_free": bool}`. WS client→server: binary PCM frames; `{"action":"set_hands_free","value":bool}`, `{"action":"toggle"}` (PTT, unchanged). Server→client: `{"type":"listening","value":"idle"|"hearing"}`, existing `state` messages.

Server per-connection: `Endpointer` instance, `hands_free: bool`, `mic_gated: bool` (= processing or speaking).

## Test scenarios

1. `Endpointer` fed 1 s silence → no utterance; 0.2 s speech-like signal then silence → discarded (below `min_speech_ms`).
2. `Endpointer` fed 1.5 s speech, 0.5 s pause, 1 s speech, 0.8 s silence → exactly one utterance ≈ 3 s long (mid-sentence pause tolerated), includes pre-roll.
3. `Endpointer` fed 40 s continuous speech → utterance emitted at `max_utterance_s`.
4. `Endpointer.gated = True` while fed speech → nothing emitted; ungated afterwards → works.
5. `asr.junk(text, logprob)` → `""`, `"[BLANK_AUDIO]"`, `"uh"` (low logprob) filtered; `"what time is it"` passes.
6. WS: `set_hands_free true` persisted in settings; binary frames accepted without error; reflected in `/api/config`.
7. WS: binary frames while `processing` → no `respond()` call (mocked).

Manual: hands-free on, ask two questions in a row without touching keys; assistant does not respond to its own reply; a cough alone produces no reply; PTT still works with hands-free off.

## Out of scope

Barge-in / interrupting; wake word; streaming ASR; Safari support; server-side mic capture (removed).
