Status: exploration (research only — not being built; decided 2026-08-29)

# 21 — Obsidian as host: plugin wrapping the voicebot vs. TypeScript rewrite

## Question
Can the voicebot run *inside* Obsidian so it needs no separate boot, and should the pipeline be rewritten in TS
for that?

## Route 1 — Obsidian hosts, Python stays (recommended if the spike passes)
- Desktop-only plugin spawns `chatbot.py` via Node `child_process` on load, kills on unload. Precedent: Local
  Runner, Ghostty Terminal (spawns a Python PTY helper), opencode-obsidian. Interpreter/venv path as a plugin
  setting (precedent: obsidian-python-scripter, JupyMD auto-discovery).
- UI: `<webview>` is disabled in Obsidian's main window; the working pattern is a custom `ItemView` leaf with an
  `<iframe>` pointing at `127.0.0.1:8010` (HTML Preview, Local Web Page Viewer, Web View plugins).
- Mic: `getUserMedia` + AudioWorklet works in Obsidian's renderer (Live Scribe, Audio Notes, SystemSculpt
  transcription all record in-app). **Unconfirmed inside a localhost iframe** (needs `allow="microphone"`;
  Electron permission plumbing may block silently) — this is the spike.
- WebGL/PixiJS in a leaf: fine (3D graph plugins ship Three.js views).
- Echo cancellation while TTS plays in the same iframe: unconfirmed.

## Route 2 — TS/ONNX rewrite inside the plugin
| stage | JS option | parity on Apple Silicon |
|---|---|---|
| VAD | `@ricky0123/vad-web` (Silero, onnxruntime-web) | same model; re-tune thresholds |
| ASR | whisper.cpp node bindings (CoreML) / sherpa-onnx / transformers.js WebGPU | whisper.cpp closest to mlx-whisper; transformers.js inconsistent |
| TTS | `kokoro-js` (same ONNX weights as kokoro-onnx) | near-free win |
| LLM | keep oMLX over HTTP | MLX beats llama.cpp/Metal by ~15–30 %; nothing to port |
| Live2D | already JS | — |
Native `.node` addons (onnxruntime-node) risk Electron-ABI breakage on Obsidian updates; an all-WASM/WebGPU
stack avoids it. GB-scale model downloads are normal for AI plugins.

| | Route 1 | Route 2 |
|---|---|---|
| effort | low (spawn + iframe + settings) | high (3 of 4 stages) |
| latency risk | low (same MLX code) | medium-high, benchmark-first |
| fidelity | identical | ASR/VAD need re-tuning |
| maintenance | one codebase + thin shell | bundler/ONNX assets, ABI sensitivity |

## Spikes before any decision
Route 1: (1) `getUserMedia` + AEC inside the iframe leaf; (2) mic stays clean while Kokoro plays; (3) child
process lifecycle across reload/quit/crash. Route 2: (1) whisper.cpp-node vs mlx-whisper latency on this Mac;
(2) all-WASM stack feasible end-to-end; (3) vad-web parity with `vad.py` tuning.

## Recommendation
If spike 1 passes → Route 1 (days of plumbing, zero behaviour risk). Else Route 2 incrementally: kokoro-js and
vad-web first, oMLX stays, ASR replacement gated on a real benchmark.

Sources: community.obsidian.md plugin pages (Local Runner, Ghostty Terminal, Live Scribe, Local HTML Browser,
HTML Preview, Web View, Three D Graph View); electron webview-tag docs; ricky0123/vad docs; kokoro-js npm;
onnx-community/Kokoro-82M-v1.0-ONNX; ggml-org/whisper.cpp; transformers.js issue #894; sherpa-onnx;
docs.obsidian.md/Developer+policies; electron/rebuild issues; yage.ai MLX vs llama.cpp benchmark (2026-03).
