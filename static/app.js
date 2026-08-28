// ---------- config ----------
// Server picks the avatar (AVATAR in .env); the matching client profile is loaded from here.
const PROFILES = {
  haru: {
    modelUrl: '/static/models/haru/haru_greeter_t03.model3.json',
    avatarScale: 1.15, avatarTopCrop: 0.0, idle: 'Idle',
    mouthParam: 'ParamMouthOpenY',
    // Haru ships expression files (f00–f07); see docs/04-avatar.md for what each looks like.
    expressions: { happy: { expr: 'f04' }, surprised: { expr: 'f05' }, thinking: { expr: 'f07' }, apologetic: { expr: 'f06' } },
    credit: 'Haru © Live2D Inc.',
  },
  wanko: {
    modelUrl: '/static/models/wanko/Wanko.model3.json',
    avatarScale: 1.0, avatarTopCrop: 0.12, idle: 'Idle',
    mouthParam: 'PARAM_MOUTH_OPEN_Y',
    // Wanko uses legacy param ids, so the library's built-in mouse-follow focus doesn't
    // reach them; we replicate it manually in afterMotionUpdate using these ids.
    focusParams: { angleX: 'PARAM_ANGLE_X', angleY: 'PARAM_ANGLE_Y', angleZ: 'PARAM_ANGLE_Z', bodyX: 'PARAM_BODY_ANGLE_X' },
    // No expression files: hand-built parameter sets, applied each frame with a lerped weight.
    expressions: {
      happy:      { params: { PARAM_MOUTH_FORM: 1, PARAM_EAR_L: 1, PARAM_EAR_R: 1, PARAM_TERE: 0.5 } },
      surprised:  { params: { PARAM_EYE_L_OPEN: 1.3, PARAM_EYE_R_OPEN: 1.3, PARAM_EAR_L: 1, PARAM_EAR_R: 1, PARAM_BODY_ANGLE_Y: -6 } },
      thinking:   { params: { PARAM_FACE_01: 1, PARAM_EYE_L_OPEN: 0.75, PARAM_EYE_R_OPEN: 0.75, PARAM_ANGLE_Z: 8 } },
      apologetic: { params: { PARAM_TERE: 1, PARAM_MOUTH_FORM: -1, PARAM_EAR_L: -1, PARAM_EAR_R: -1, PARAM_ANGLE_Y: -10 } },
    },
    credit: 'Wanko © Live2D Inc.',
  },
  natori: {
    modelUrl: '/static/models/natori/Natori.model3.json',
    avatarScale: 1.15, avatarTopCrop: 0.0, idle: 'Idle',
    mouthParam: 'ParamMouthOpenY',
    expressions: { happy: { expr: 'Smile' }, surprised: { expr: 'Surprised' }, thinking: { expr: 'Normal' }, apologetic: { expr: 'Sad' } },
    credit: 'Natori © Live2D Inc.',
  },
};

const CONFIG = {
  expressionWeight: 0.7,
  expressionFade: 0.12,  // per-frame lerp toward target weight
  expressionHoldMs: 500,
  mouthGain: 6.0,        // RMS → mouth openness multiplier
  mouthSmooth: 0.35,     // 0..1, higher = snappier
};
let profile = PROFILES.haru;
let avatarName = 'Haru';

// ---------- DOM ----------
const btn = document.getElementById('btn');
const status = document.getElementById('status');
const transcript = document.getElementById('transcript');
const avatarNote = document.getElementById('avatar-note');
let shutdownRequested = false;
document.getElementById('shutdown').onclick = () => {
  shutdownRequested = true;
  ws.send(JSON.stringify({ action: 'shutdown' }));
  openDrawer(false);
  status.textContent = 'shutting down…';
  window.close(); // only works for script-opened tabs; otherwise the status line tells the story
};
document.getElementById('reload-characters').onclick = () => {
  ws.send(JSON.stringify({ action: 'reload_characters' }));
};

// ---------- menu drawer ----------
const drawer = document.getElementById('drawer');
const backdrop = document.getElementById('drawer-backdrop');
const avatarList = document.getElementById('avatar-list');
let avatarOptions = [];
let currentAvatarKey = null;
let backdropOptions = [];
let currentBackdropKey = 'none';
const backdropEl = document.getElementById('backdrop');
const backdropList = document.getElementById('backdrop-list');

function applyBackdrop(key) {
  currentBackdropKey = key;
  const b = backdropOptions.find(x => x.key === key);
  if (b && b.file) {
    backdropEl.style.backgroundImage = `url("${b.file}")`;
    backdropEl.classList.add('on');
  } else {
    backdropEl.classList.remove('on');
  }
  const credit = document.getElementById('credit');
  credit.textContent = [profile.credit, b && b.credit].filter(Boolean).join('  ·  ');
}

function renderBackdropList() {
  backdropList.replaceChildren(...backdropOptions.map(b => {
    const c = document.createElement('button');
    c.className = 'chip' + (b.key === currentBackdropKey ? ' active' : '');
    c.textContent = b.name;
    c.onclick = () => ws.send(JSON.stringify({ action: 'set_backdrop', key: b.key }));
    return c;
  }));
}

// ---------- reply length (verbosity) ----------
const verbosityLevels = ['short', 'normal', 'long'];
let currentVerbosity = 'normal';
const verbositySlider = document.getElementById('verbosity-slider');
const verbosityLabels = document.querySelectorAll('.verbosity-label');

function renderVerbosity() {
  const idx = Math.max(0, verbosityLevels.indexOf(currentVerbosity));
  verbositySlider.value = String(idx);
  verbosityLabels.forEach(l => l.classList.toggle('active', Number(l.dataset.value) === idx));
}

verbositySlider.addEventListener('change', () => {
  const idx = Number(verbositySlider.value);
  ws.send(JSON.stringify({ action: 'set_verbosity', value: verbosityLevels[idx] }));
});

function openDrawer(open) {
  drawer.classList.toggle('open', open);
  drawer.setAttribute('aria-hidden', String(!open));
  backdrop.hidden = !open;
  if (open) { renderAvatarList(); renderBackdropList(); }
}
document.getElementById('menu-btn').onclick = () => openDrawer(!drawer.classList.contains('open'));
backdrop.onclick = () => openDrawer(false);
document.addEventListener('keydown', (e) => { if (e.code === 'Escape') openDrawer(false); });

function renderAvatarList() {
  avatarList.replaceChildren(...avatarOptions.map(a => {
    const b = document.createElement('button');
    b.className = 'drawer-item' + (a.key === currentAvatarKey ? ' active' : '');
    b.disabled = state !== 'idle';
    const n = document.createElement('span'); n.className = 'name'; n.textContent = a.name;
    const d = document.createElement('span'); d.className = 'desc'; d.textContent = a.description || '';
    b.append(n, d);
    b.onclick = () => { ws.send(JSON.stringify({ action: 'set_avatar', key: a.key })); openDrawer(false); };
    return b;
  }));
}

const statusText = {
  idle: 'press space or click to speak',
  recording: 'recording — press space or click to stop',
  processing: 'processing...',
  thinking: 'thinking...',
  speaking: 'speaking...',
};
const btnText = { idle: 'Speak', recording: 'Stop', processing: '...', thinking: '...', speaking: '...' };

let ws;
let state = 'idle';
let handsFree = false;   // mirrors the server-confirmed hands_free value
let muted = false;       // hands-free only: user muted the mic
let pttActive = false;   // push-to-talk only: between ptt start/stop
let listening = 'idle';  // hands-free VAD activity: 'idle' | 'hearing'
let currentTurn = 0;     // latest turn id seen on `state`/`speech`
let audioBlocked = false;      // true while waiting on a user gesture to unlock audio
let blockedSentForTurn = null; // turn we last sent playback_blocked for (send once per turn)

// ---------- audio ----------
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
const analyser = audioCtx.createAnalyser();
analyser.fftSize = 1024;
analyser.connect(audioCtx.destination);
const timeData = new Float32Array(analyser.fftSize);

function unlockAudio() {
  if (audioCtx.state === 'suspended') audioCtx.resume();
}

// ---------- microphone capture ----------
// Uses the SAME audioCtx that plays TTS so Chrome's echo cancellation has the
// far-end reference signal. Streams 16 kHz mono Int16 frames over the WS
// while `mic.sending` is true (gated by PTT / hands-free state, see below).
const mic = {
  started: false,
  starting: null,
  sending: false,
  node: null,

  async start() {
    if (this.started) return true;
    if (this.starting) return this.starting;
    this.starting = (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
        });
        await audioCtx.audioWorklet.addModule('/static/mic-worklet.js');
        const source = audioCtx.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(audioCtx, 'mic-processor');
        node.port.onmessage = (e) => {
          if (this.sending && ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
        };
        source.connect(node); // not connected to destination: we never play the mic back
        this.node = node;
        this.started = true;
        return true;
      } catch (err) {
        console.warn('microphone access failed', err);
        status.textContent = 'microphone permission denied';
        return false;
      } finally {
        this.starting = null;
      }
    })();
    return this.starting;
  },
};

function updateSending() {
  mic.sending = (handsFree && !muted) || pttActive;
}

const queue = [];           // pending {emotion, buffer}
let playing = false;        // a source is currently playing
let replyEnded = false;     // speech_end received for the current reply
let mouth = 0;

function rms() {
  analyser.getFloatTimeDomainData(timeData);
  let s = 0;
  for (let i = 0; i < timeData.length; i++) s += timeData[i] * timeData[i];
  return Math.sqrt(s / timeData.length);
}

// Turn ids only move forward. Returns true when `turn` is a newer turn than the
// one we were tracking, i.e. the caller should discard anything from the old one.
function adoptTurn(turn) {
  if (typeof turn !== 'number' || turn <= currentTurn) return false;
  currentTurn = turn;
  return true;
}

function enqueueSpeech(msg) {
  if (adoptTurn(msg.turn)) {
    // A newer turn superseded whatever was queued (but not yet playing): drop it.
    while (queue.length && typeof queue[0].turn === 'number' && queue[0].turn < currentTurn) queue.shift();
  }
  const bytes = Uint8Array.from(atob(msg.wav), c => c.charCodeAt(0));
  // Push synchronously so message order is preserved even though decoding is async;
  // playNext() awaits `ready` before playing each item.
  const item = { emotion: msg.emotion, buffer: null, ready: null, turn: msg.turn };
  item.ready = audioCtx.decodeAudioData(bytes.buffer)
    .then(buffer => { item.buffer = buffer; })
    .catch(e => { console.warn('audio decode failed, skipping chunk', e); item.buffer = null; });
  queue.push(item);
  playNext();
}

async function playNext() {
  if (playing) return;
  const item = queue[0];
  if (!item) {
    if (replyEnded) finishReply();
    return;
  }
  playing = true;
  await item.ready;
  queue.shift();
  if (!item.buffer) {
    // Decode failed: skip this chunk without stalling the queue.
    playing = false;
    playNext();
    return;
  }
  if (audioCtx.state === 'suspended') {
    // Autoplay policy: no user gesture on this page yet. Ask for one and wait.
    await audioCtx.resume().catch(() => {});
    if (audioCtx.state !== 'running') {
      audioBlocked = true;
      if (blockedSentForTurn !== currentTurn && ws && ws.readyState === WebSocket.OPEN) {
        blockedSentForTurn = currentTurn;
        ws.send(JSON.stringify({ action: 'playback_blocked', turn: currentTurn }));
      }
      render();
      await new Promise(resolve => {
        const unlock = () => audioCtx.resume().then(() => {
          document.removeEventListener('click', unlock);
          document.removeEventListener('keydown', unlock);
          resolve();
        });
        document.addEventListener('click', unlock);
        document.addEventListener('keydown', unlock);
      });
      audioBlocked = false;
      render();
    }
  }
  avatar.setEmotion(item.emotion);
  const src = audioCtx.createBufferSource();
  src.buffer = item.buffer;
  src.connect(analyser);
  src.onended = () => { playing = false; playNext(); };
  src.start();
}

function finishReply() {
  replyEnded = false;
  setTimeout(() => avatar.setEmotion('neutral'), CONFIG.expressionHoldMs);
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'playback_done', turn: currentTurn }));
}

// ---------- avatar ----------
const avatar = {
  model: null,
  currentEmotion: 'neutral',
  paramTarget: {},   // param id → target value for the active params-expression
  paramWeight: 0,    // current lerped weight of paramTarget

  app: null,

  async init() {
    try {
      const cfg = await (await fetch('/api/config')).json();
      avatarOptions = cfg.avatars || [];
      currentAvatarKey = cfg.avatar;
      avatarName = cfg.name || avatarName;
      backdropOptions = cfg.backdrops || [];
      currentBackdropKey = cfg.backdrop || 'none';
      handsFree = !!cfg.hands_free;
      currentVerbosity = verbosityLevels.includes(cfg.verbosity) ? cfg.verbosity : 'normal';
    } catch (e) { console.warn('config fetch failed, using default profile', e); }
    applyBackdrop(currentBackdropKey);
    renderHandsFreeToggle();
    renderVerbosity();
    render();

    if (!window.PIXI || !PIXI.live2d) { avatarNote.hidden = false; return; }

    const canvas = document.getElementById('stage');
    this.app = new PIXI.Application({
      view: canvas, backgroundAlpha: 0, resizeTo: canvas.parentElement, antialias: true,
      resolution: window.devicePixelRatio || 1, autoDensity: true,
    });
    this.app.renderer.on('resize', () => this.fit());
    await this.load(currentAvatarKey);
  },

  async load(key) {
    profile = PROFILES[key] || profile;
    applyBackdrop(currentBackdropKey);
    if (!this.app) return; // PIXI/Live2D unavailable: nothing more to render
    if (this.model) { this.app.stage.removeChild(this.model); this.model.destroy(); this.model = null; }
    this.paramTarget = {}; this.paramWeight = 0; this.currentEmotion = 'neutral';

    let model;
    try {
      model = await PIXI.live2d.Live2DModel.from(profile.modelUrl);
    } catch (e) {
      console.warn('Live2D model failed to load', e);
      avatarNote.hidden = false;
      return;
    }
    avatarNote.hidden = true;
    this.model = model;
    this.app.stage.addChild(model);
    this.fit();

    const core = model.internalModel.coreModel;
    model.internalModel.on('afterMotionUpdate', () => {
      // lip-sync (skip the analyser read entirely when nothing is playing)
      const target = playing ? Math.min(1, rms() * CONFIG.mouthGain) : 0;
      mouth += (target - mouth) * CONFIG.mouthSmooth;
      core.setParameterValueById(profile.mouthParam, mouth);
      // mouse-follow focus for models with legacy param ids (the library's built-in
      // focus only writes to the standard Cubism 4 ids); mirrors the library's own
      // formulas/behavior (add on top of the motion's current value).
      if (profile.focusParams) {
        const fp = profile.focusParams;
        const fc = model.internalModel.focusController;
        if (fp.angleX) core.setParameterValueById(fp.angleX, core.getParameterValueById(fp.angleX) + 30 * fc.x);
        if (fp.angleY) core.setParameterValueById(fp.angleY, core.getParameterValueById(fp.angleY) + 30 * fc.y);
        if (fp.angleZ) core.setParameterValueById(fp.angleZ, core.getParameterValueById(fp.angleZ) + fc.x * fc.y * -30);
        if (fp.bodyX) core.setParameterValueById(fp.bodyX, core.getParameterValueById(fp.bodyX) + 10 * fc.x);
      }
      // parameter-based expression (blend toward target on top of the motion's values)
      const wantWeight = Object.keys(this.paramTarget).length ? CONFIG.expressionWeight : 0;
      this.paramWeight += (wantWeight - this.paramWeight) * CONFIG.expressionFade;
      if (this.paramWeight > 0.01) {
        for (const [id, v] of Object.entries(this.paramTarget)) {
          const cur = core.getParameterValueById(id);
          core.setParameterValueById(id, cur + (v - cur) * this.paramWeight);
        }
      }
    });
    model.motion(profile.idle);
  },

  fit() {
    // Model canvases have generous transparent margins; scale relative to viewport height.
    const app = this.app, m = this.model;
    if (!m) return;
    const natural = { w: m.width / m.scale.x, h: m.height / m.scale.y };
    const scale = (app.screen.height / natural.h) * profile.avatarScale;
    m.scale.set(scale);
    m.x = (app.screen.width - natural.w * scale) / 2;
    m.y = -natural.h * scale * profile.avatarTopCrop;
  },

  setEmotion(emotion) {
    if (!this.model || emotion === this.currentEmotion) return;
    this.currentEmotion = emotion;
    const def = profile.expressions[emotion];
    const em = this.model.internalModel.motionManager.expressionManager;

    if (def && def.params) {
      this.paramTarget = def.params;
      if (em) em.resetExpression();
      return;
    }
    this.paramTarget = {};
    if (!em) return;
    if (!def) { em.resetExpression(); return; }
    this.model.expression(def.expr).catch(e => console.warn('expression failed', e));
  },
};

// ---------- websocket / ui ----------
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'state') {
      adoptTurn(msg.turn);
      setState(msg.value);
    }
    else if (msg.type === 'transcript') addMessage(msg.role, msg.text);
    else if (msg.type === 'speech') enqueueSpeech(msg);
    else if (msg.type === 'speech_end') { replyEnded = true; playNext(); }
    else if (msg.type === 'avatar') {
      currentAvatarKey = msg.key; avatarName = msg.name;
      avatar.load(msg.key);
    }
    else if (msg.type === 'backdrop') { applyBackdrop(msg.key); renderBackdropList(); }
    else if (msg.type === 'characters_reloaded') {
      avatarOptions = msg.avatars || avatarOptions;
      currentAvatarKey = msg.avatar;
      avatarName = msg.name || avatarName;
      renderAvatarList();
      status.textContent = 'characters reloaded';
      setTimeout(() => { render(); }, 1500);
    }
    else if (msg.type === 'listening') { listening = msg.value; render(); }
    else if (msg.type === 'hands_free') {
      handsFree = msg.value;
      if (!handsFree) { muted = false; pttActive = false; listening = 'idle'; }
      updateSending();
      renderHandsFreeToggle();
      render();
    }
    else if (msg.type === 'verbosity') {
      currentVerbosity = verbosityLevels.includes(msg.value) ? msg.value : currentVerbosity;
      renderVerbosity();
    }
    else if (msg.type === 'error') { status.textContent = msg.text.toLowerCase(); renderVerbosity(); }
  };

  ws.onopen = () => render();
  ws.onclose = (e) => {
    if (shutdownRequested) { status.textContent = 'shut down — close this tab'; btn.disabled = true; return; }
    if (e.code === 4000) {
      status.textContent = 'opened in another tab — reload to use here';
      return; // do not auto-reconnect: the other tab now owns the connection
    }
    if (e.code === 4003) {
      status.textContent = 'connection refused (origin)';
      return; // do not auto-reconnect: this origin will never be accepted
    }
    status.textContent = 'disconnected — retrying...';
    setTimeout(connect, 2000);
  };
}

function setState(value) {
  state = value;
  render();
}

function renderHandsFreeToggle() {
  document.getElementById('hands-free-toggle').classList.toggle('active', handsFree);
}

// Recomputes the Speak button + status line from state/handsFree/muted/listening.
function render() {
  const busy = state === 'processing' || state === 'thinking' || state === 'speaking';
  if (handsFree && !busy) {
    // Idle hands-free: the button is a mute switch, the status shows VAD activity.
    const hearing = listening === 'hearing';
    btn.className = !muted && hearing ? 'hearing' : '';
    btn.textContent = muted ? 'MUTED' : 'MUTE';
    status.textContent = muted ? 'muted'
      : hearing ? 'hearing you…'
      : mic.started ? 'listening…'
      : 'click to start the microphone';
    btn.disabled = false;
  } else {
    btn.className = state === 'idle' ? '' : state;
    btn.textContent = btnText[state] || '...';
    status.textContent = statusText[state] || '';
    btn.disabled = !(state === 'idle' || state === 'recording');
  }
  // Re-assert the audio-unlock hint last so other branches above don't clobber it
  // while we're still waiting on a user gesture to resume the AudioContext.
  if (audioBlocked) status.textContent = 'click anywhere to enable audio';
}

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const r = document.createElement('span'); r.className = 'role'; r.textContent = role === 'user' ? 'You' : avatarName;
  const t = document.createElement('span'); t.className = 'text'; t.textContent = text;
  div.append(r, t);
  transcript.appendChild(div);
  transcript.scrollTop = transcript.scrollHeight;
}

// Push-to-talk: Space/click starts capture on press, stops it on the next press.
async function pttToggle() {
  unlockAudio();
  if (state === 'idle') {
    const ok = await mic.start();
    if (!ok) return;
    pttActive = true;
    updateSending();
    ws.send(JSON.stringify({ action: 'ptt', value: 'start' }));
  } else if (state === 'recording') {
    pttActive = false;
    updateSending();
    ws.send(JSON.stringify({ action: 'ptt', value: 'stop' }));
  }
}

// Hands-free: Space/click mutes/unmutes the always-on stream.
async function toggleMute() {
  if (!(state === 'idle' || state === 'recording' || listening === 'hearing')) return;
  if (muted) {
    const ok = await mic.start();
    if (!ok) return;
    muted = false;
  } else {
    muted = true;
  }
  updateSending();
  render();
}

function btnAction() {
  if (handsFree) toggleMute();
  else pttToggle();
}

btn.addEventListener('click', btnAction);
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); btnAction(); }
});
// Any click counts as the user gesture Chrome wants: unlock audio playback, and
// lazily (re)start the mic if hands-free was already on at load (e.g. after a
// reload) and hasn't captured a start gesture yet.
document.addEventListener('click', () => {
  unlockAudio();
  if (handsFree && !mic.started) mic.start();
});

document.getElementById('hands-free-toggle').onclick = async () => {
  unlockAudio();
  const desired = !handsFree;
  if (desired) {
    const ok = await mic.start();
    if (!ok) return; // permission denied: toggle stays off, status already set by mic.start()
  }
  ws.send(JSON.stringify({ action: 'set_hands_free', value: desired }));
};

avatar.init();
connect();
