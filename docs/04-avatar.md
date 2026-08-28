Status: exploration

# 04 — Animated Avatar (Live2D office assistant)

## Overview

Add an animated 2D avatar to the voicebot's browser UI: a Live2D character that lip-syncs to Kokoro speech, shows a work-appropriate expression per reply, and idles (blinks/breathes) otherwise. Move audio playback from the server (`sounddevice`) to the browser so the avatar can react to what is actually being heard. Switch the LLM backend from Ollama to oMLX (OpenAI-compatible, Gemma 4 26B).

## User stories

- As a user, I see an assistant character on screen that looks alive while idle (blinks, breathes, subtle sway).
- As a user, when the bot speaks, the character's mouth moves in time with the audio.
- As a user, the character's expression matches the tone of the reply (happy / thinking / surprised / apologetic / neutral) in a subdued, office-friendly way.
- As a user, the bot starts speaking after the first sentence instead of waiting for the full reply.
- As a user, everything still runs locally; existing tools, memory and push-to-talk keep working.

## UI/UX

Layout (dark theme kept from current `index.html`):

```
┌─────────────────────────────────────────────┐
│  shut down                                  │
│  ┌──────────────┐   ┌────────────────────┐  │
│  │              │   │ transcript         │  │
│  │   Live2D     │   │ You: ...           │  │
│  │   Haru       │   │ Bot: ...           │  │
│  │  (≈480px h)  │   │                    │  │
│  │              │   └────────────────────┘  │
│  └──────────────┘        ( Speak )          │
│                    press space to speak     │
└─────────────────────────────────────────────┘
```

Flow per turn (unchanged from user's view): Space/click → record → Space/click → transcript appears → bot speaks with animation → idle.

Expressions: fade in over ~200 ms at the start of a sentence, fade back to neutral ~500 ms after audio ends. Intensity capped (~0.7) so it reads as a colleague, not a cartoon.

Model: Live2D sample **Haru** (Cubism 4, business-casual). Cubism Core JS and the model are downloaded by a script, not committed (Live2D license).

## Technical approach

Stack: existing FastAPI + WebSocket server; `pixi.js` + `pixi-live2d-display` in the browser; `kokoro-onnx` unchanged; `whisper-cli` unchanged; LLM via `openai` Python client pointed at oMLX.

### Changes

1. **`llm.py`** — replace `requests` → Ollama with `openai.OpenAI(base_url=OMLX_BASE_URL, api_key=OMLX_API_KEY)`. Keep tool-calling (OpenAI `tools` format; convert `TOOLS` schema). New `ask_stream(text)` generator that yields text deltas; tool-call rounds run non-streamed as today, only the final answer streams. `ask()` stays as a thin wrapper (sum of stream) so `memory.py`, tests and workflows keep working.
2. **`splitter.py`** (new) — pure function: consumes text deltas, yields `(emotion, sentence)`. Strips a single leading `[tag]`; unknown/missing tag → `neutral`. Splits on `.?!` followed by space/end, flushes remainder on close.
3. **`chatbot.py`** — `speak()` no longer plays audio. For each sentence: Kokoro → 24 kHz 16-bit wav → send `{"type":"speech","emotion":..,"text":..,"wav":<base64>}` over the WS. Send `{"type":"speech_end"}` after the last one. `state: speaking` is set when the first chunk is sent; `idle` is sent by the server after the browser reports `{"action":"playback_done"}` (or after a timeout fallback of total audio duration + 2 s).
4. **`index.html`** → split into `static/index.html`, `static/app.js`, `static/style.css`. Add:
   - Audio queue: decode base64 → `AudioContext.decodeAudioData` → play sequentially through an `AnalyserNode`.
   - Lip-sync loop (`requestAnimationFrame`): RMS of analyser time-domain data → smoothed → `coreModel.setParameterValueById('ParamMouthOpenY', v)`.
   - Expression handling: emotion → Haru expression id (mapping in `config` section of `app.js`), fade in/out.
   - Model load, `Idle` motion group looping, scaled to fit the left column.
5. **`scripts/fetch_assets.sh`** — downloads `pixi.min.js`, `pixi-live2d-display` cubism4 bundle, `live2dcubismcore.min.js`, and Haru model into `static/vendor/` and `static/models/haru/` (both gitignored).
6. **System prompt** — persona: calm, friendly office assistant; ≤2 sentences; must begin with exactly one of `[neutral] [happy] [thinking] [surprised] [apologetic]`. Keep existing tool instructions.
7. **Config** — `OMLX_BASE_URL`, `OMLX_API_KEY`, `OMLX_MODEL`, `KOKORO_VOICE` (default `af_sarah`, speed 0.95) via `.env`.

### Decisions

- Chose Live2D over VRM because 2D matches "assistant widget" better and `pixi-live2d-display` gives lip-sync/expressions for free.
- Chose Haru over Hiyori/Mao because it is the only free sample with an office look.
- Chose browser-side playback over server `sounddevice` because lip-sync needs the audio signal where the avatar renders; side effect: works over the network too.
- Chose amplitude-based lip-sync over viseme (Rhubarb) because it's ~30 lines and good enough for a first slice; viseme is a follow-up PRD if it looks off.
- Chose sentence-level streaming to TTS over full-reply TTS because it cuts perceived latency by roughly the reply length.
- Chose OpenAI client for oMLX over raw `requests` because tool-calling format is standard and streaming is built in. Assume oMLX supports `tools`; if not, fall back to prompt-based tool triggering (revisit).
- Risk: Gemma forgets the `[tag]` prefix on some replies → splitter defaults to `neutral`, never breaks playback.
- Risk: Docker on macOS has no GPU → oMLX stays native; the app itself stays native for this slice (Docker is out of scope).

## Data model / state shape

Server (per session, in-memory):
```
recording: bool, processing: bool, speaking: bool
_conversation: list[OpenAI messages]   # unchanged shape except tool_call format
```

WS messages server → client:
```
{type:"state", value:"idle"|"recording"|"processing"|"thinking"|"speaking"}
{type:"transcript", role:"user"|"assistant", text}
{type:"speech", emotion, text, wav}     # wav = base64 16-bit PCM wav, 24 kHz mono
{type:"speech_end"}
```
Client → server:
```
{action:"toggle"} | {action:"shutdown"} | {action:"playback_done"}
```

Invariants: `speech` chunks for one reply arrive in order; exactly one `speech_end` per reply; `transcript(assistant)` carries the full reply with tags stripped.

## Test scenarios

Unit (pytest):
1. `splitter`: input deltas `["[hap","py] Sure thing.", " Anything else?"]` → yields `("happy","Sure thing.")`, `("happy","Anything else?")`.
2. `splitter`: input without a tag → emotion `neutral`, text unchanged.
3. `splitter`: unknown tag `[angry]` → emotion `neutral`, tag stripped from text.
4. `splitter`: text with no terminal punctuation → single sentence flushed on close.
5. `llm.ask()` (mocked client): returns concatenated stream, appends assistant turn to `_conversation` without the tag.
6. `llm.ask()` (mocked client): tool_call round → `run_tool` invoked → final answer returned (existing tests in `tests/test_llm.py` adapted to OpenAI message format and still passing).
7. `chatbot.speak_stream` (mocked Kokoro + WS): 2-sentence reply → exactly 2 `speech` messages then 1 `speech_end`, each `wav` decodes to a valid wav header.

Manual (end-to-end, user):
8. Open app → Haru visible, blinking, idle motion; state `idle`.
9. Speak "hello" → transcript shows both turns; Haru's mouth moves while audio plays; mouth still and expression back to neutral within 1 s after audio ends; state returns to `idle`.
10. Ask a question producing a multi-sentence reply → first audio starts before the transcript text is complete.
11. Ask for the weather → tool call still works via oMLX; reply spoken.
12. Reply tagged `[happy]` → visibly different expression than a `[neutral]` reply.

## Out of scope

- Docker packaging (follow-up PRD).
- Viseme-accurate lip-sync (Rhubarb).
- Interrupting the bot mid-speech.
- Custom/commissioned avatar model, mouse-follow gaze, gestures/motions triggered by emotion.
- Wake-word / continuous listening.
- Email dashboard changes (`workflows/`) — untouched.
