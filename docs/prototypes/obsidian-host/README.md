# Obsidian host spike (voicebot-spike)

Throwaway spike plugin for [docs/21-obsidian-host.md](../../21-obsidian-host.md). Not a product feature —
do not install into a real vault, do not commit vault state.

## Setup

1. Create a throwaway Obsidian vault (e.g. `~/Desktop/obsidian-spike-vault`) — do **not** use a real vault.
2. In that vault, create `.obsidian/plugins/voicebot-spike/` and copy `manifest.json` and `main.js` from this
   folder into it (do not copy this README, it's not needed by Obsidian).
3. Open the vault in Obsidian. Settings → Community plugins → turn off "Restricted mode" (enables community
   plugins) if prompted.
4. Settings → Community plugins → find "Voicebot Spike" in the installed list → enable it.
5. Settings → Voicebot Spike (the plugin's own settings tab) → confirm/adjust "Python path" and "Repo path"
   (they default to this repo's `.venv/bin/python` and repo root).
6. Open the command palette (Cmd-P) and run the four `Voicebot spike: ...` commands as described below.
7. Check the developer console (Cmd-Option-I → Console) for `[chatbot.py]` / `[voicebot-spike]` log lines.

## Checklist of observations to record

Run commands via Cmd-P → type "Voicebot spike".

1. **Spawn server** (`Voicebot spike: spawn server`)
   - Does a Notice show a spawned pid?
   - Does the console show `chatbot.py` stdout (server startup logs)?
   - Wait for the server to be ready (watch console output), then continue.

2. **Open panel** (`Voicebot spike: open panel`)
   - Does a new tab/leaf open with the iframe?
   - Does the app's own UI render inside the iframe (Live2D/WebGL avatar, status text)?
   - **WebGL avatar renders?** yes/no — note any console/WebGL errors in the iframe.
   - Find the app's own status text "click to start the microphone" inside the iframe.
     - Click it. **Does a mic permission prompt appear** (OS-level or Electron-level)?
     - **Does `getUserMedia` resolve inside the iframe?** After clicking, does the status text change to
       something indicating it's listening? (check the app's own UI state, not just no-error)
     - If it silently fails (no prompt, no state change, no console error) — note that explicitly; this is
       the "Electron permission plumbing may block silently" risk flagged in the PRD.
   - **TTS audio plays?** Trigger a response from the bot and confirm audio is audible from the iframe.
   - **Echo/AEC behaviour while speaking during playback**: while TTS audio is playing, speak into the mic.
     Does the mic pick up the bot's own TTS audio (echo), or does echoCancellation suppress it cleanly? Note
     any feedback loops, double-triggers, or garbled ASR caused by picking up its own voice.

3. **Mic test (plugin context)** (`Voicebot spike: mic test (plugin context)`)
   - This calls `getUserMedia` directly in the plugin (not inside the iframe) as a control comparison.
   - Record: does the Notice say "mic OK" or "mic FAILED"? What is the reported `echoCancellation` value in
     `getSettings()`? Compare this control result against the iframe's behavior in step 2 — if the plugin
     context works but the iframe doesn't (or vice versa), that's the key spike finding.

4. **Stop server** (`Voicebot spike: stop server`)
   - Does a Notice confirm SIGTERM was sent?
   - Does the console show `chatbot.py exited code=... sig=...`?
   - **Does the process really exit?** Check with `ps aux | grep chatbot.py` in a terminal — confirm no
     orphaned process remains after SIGTERM. Also test: disable the plugin (or reload Obsidian) while the
     server is running, and confirm `onunload` kills it too (same `ps` check).

## Decision rule (from docs/21-obsidian-host.md)

> If spike 1 passes → Route 1 (days of plumbing, zero behaviour risk). Else Route 2 incrementally: kokoro-js
> and vad-web first, oMLX stays, ASR replacement gated on a real benchmark.

Concretely: if the iframe's `getUserMedia` resolves cleanly (mic prompt appears, status text confirms
listening) and AEC holds up reasonably while TTS plays back in the same iframe, Route 1 (Obsidian hosts,
Python stays — spawn + iframe + settings, per this spike's plugin) is recommended. If the iframe blocks mic
access silently or AEC breaks down badly during playback, fall back to Route 2 (TS/ONNX rewrite), starting
with kokoro-js and vad-web while keeping oMLX for the LLM stage, with ASR replacement (whisper.cpp-node vs
sherpa-onnx vs transformers.js) gated on a real latency benchmark rather than assumed parity.

## Cleanup

Delete the throwaway vault when done. Do not leave `voicebot-spike` installed in any vault you use day-to-day.
