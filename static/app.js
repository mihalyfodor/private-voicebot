// ---------- config ----------
// Wanko is the only avatar (PRD 23); the client half of its profile lives here.
const PROFILES = {
  wanko: {
    // scripts/fetch_assets.sh generates the `.app.` file (upstream model + PRD 27 motion
    // groups); modelUrl is the fallback for a model folder fetched before PRD 27.
    modelUrl: '/static/models/wanko/Wanko.app.model3.json',
    fallbackModelUrl: '/static/models/wanko/Wanko.model3.json',
    avatarScale: 1.0, avatarTopCrop: 0.12, idle: 'Idle',
    mouthParam: 'PARAM_MOUTH_OPEN_Y',
    // Wanko uses legacy param ids, so the library's built-in mouse-follow focus doesn't
    // reach them; we replicate it manually in afterMotionUpdate using these ids.
    focusParams: { angleX: 'PARAM_ANGLE_X', angleY: 'PARAM_ANGLE_Y', angleZ: 'PARAM_ANGLE_Z', bodyX: 'PARAM_BODY_ANGLE_X' },
    // No expression files: hand-built parameter sets, applied each frame with a lerped weight.
    expressions: {
      happy:      { params: { PARAM_MOUTH_FORM: 1, PARAM_EAR_L: 1, PARAM_EAR_R: 1, PARAM_TERE: 0.5, PARAM_EFFECT: 0.6 } },
      surprised:  { params: { PARAM_EYE_L_OPEN: 1.3, PARAM_EYE_R_OPEN: 1.3, PARAM_EAR_L: 1, PARAM_EAR_R: 1, PARAM_BODY_ANGLE_Y: -6 } },
      thinking:   { params: { PARAM_FACE_01: 1, PARAM_EYE_L_OPEN: 0.75, PARAM_EYE_R_OPEN: 0.75, PARAM_ANGLE_Z: 8 } },
      apologetic: { params: { PARAM_TERE: 1, PARAM_MOUTH_FORM: -1, PARAM_EAR_L: -1, PARAM_EAR_R: -1, PARAM_ANGLE_Y: -10 } },
      proud:      { params: { PARAM_MOUTH_FORM: 1, PARAM_EFFECT: 1, PARAM_ANGLE_Y: 6 } },
      sleepy:     { params: { PARAM_EYE_L_OPEN: 0.35, PARAM_EYE_R_OPEN: 0.35, PARAM_BOWL_LID: 40 } },
      shy:        { params: { PARAM_TERE: 1, PARAM_FACE_01: 0.5 } },
    },
    // Cues (PRD 27): motion group + a parameter blend layered on top of it, driven by the
    // state/listening transitions. `pulse` params oscillate (sin over time, amplitude 1).
    // The prop parameters below are never keyed by any clip, so the layering is conflict-free.
    cues: {
      listening: { group: 'Listening', params: { PARAM_EAR_L: 1, PARAM_EAR_R: 1 } },
      // PARAM_BOWL_LID runs -10 (lid resting on the bowl) … 50 (lid away, the default) and the
      // lid only enters the framed area below ~35, where it covers Wanko's face — so thinking
      // just eases it down to 40 and the steam carries the cue. The steam puffs are drawn only
      // while PARAM_EFFECT is up, hence the 0.4 riding along with the YUGE pulse.
      thinking:  { group: 'Thinking', params: { PARAM_BOWL_LID: 40, PARAM_EFFECT: 0.4 },
                   pulse: ['PARAM_YUGE_01', 'PARAM_YUGE_02'] },
      tool:      { params: { PARAM_EFFECT: 1 } },
      greet:     { group: 'Greet' },
      pet:       { group: 'Pet' },
      petSmall:  { group: 'PetSmall' },
      shutdown:  { group: 'Idle', index: 3, force: true, params: { PARAM_BOWL_LID: -10 } },
    },
    credit: 'Wanko © Live2D Inc.',
  },
};

const CONFIG = {
  expressionWeight: 0.7,
  expressionFade: 0.12,  // per-frame lerp toward target weight
  expressionHoldMs: 500,
  mouthGain: 6.0,        // RMS → mouth openness multiplier
  mouthSmooth: 0.35,     // 0..1, higher = snappier
  fitWidth: 0.8,         // share of the avatar column the drawn model may use…
  fitHeight: 0.7,        // …and of its height; the smaller of the two wins
  fitAnchorY: 0.62,      // where the art's vertical centre sits (0.5 = centred, larger = lower)
  cueFade: 0.1,          // per-frame lerp of a cue blend toward its target
  petGapMs: 4000,        // a tap this long after the last one is a big Pet, else PetSmall
  petHappyMs: 800,       // how long the happy blend rides on a tap
  steamHz: 0.8,          // thinking steam pulse rate
};

// PRD 27 motion groups: every group in scripts/wanko_groups.py, with the priority app.js
// plays it at and (Idle) the pick weights per index. Priorities are the library's
// MotionPriority: 1 IDLE (only when nothing else plays), 2 NORMAL, 3 FORCE (always).
const PRIORITY = { IDLE: 1, NORMAL: 2, FORCE: 3 };
const MOTIONS = {
  Idle:      { priority: PRIORITY.IDLE,   weights: [1, 1, 1, 0.5] },  // idle_04 is long and calm
  Pet:       { priority: PRIORITY.NORMAL },
  PetSmall:  { priority: PRIORITY.NORMAL },
  Greet:     { priority: PRIORITY.NORMAL },
  No:        { priority: PRIORITY.NORMAL },
  Thinking:  { priority: PRIORITY.NORMAL },
  Listening: { priority: PRIORITY.NORMAL },
};
let profile = PROFILES.wanko;
let avatarName = 'Wanko';

// ---------- DOM ----------
// The companion column (avatar + history + composer) is one element; setView moves it
// between the chat slot and the shell's left slot (PRD 24).
const status = document.getElementById('status');
const historyEl = document.getElementById('history');
const avatarNote = document.getElementById('avatar-note');
const companionEl = document.getElementById('companion');
const chatSlot = document.getElementById('chat-slot');
const leftSlot = document.getElementById('left-slot');
const composerInput = document.getElementById('composer-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const muteBtn = document.getElementById('mute-btn');
const listeningDot = document.getElementById('listening-dot');
const handsFreeToggle = document.getElementById('hands-free-toggle');   // Settings' copy of the mic switch
let shutdownRequested = false;
function showShutdownScreen() {
  avatar.onShutdown();   // PRD 27: settle into the slow breathing idle, lid closing
  for (const id of ['companion', 'workspace', 'rail']) {
    document.getElementById(id).classList.add('gone');
  }
  // The columns hold for 0.6 s and then fade over 0.6 s (.gone); hold the opaque shutdown
  // screen back for both, so Wanko's settle — idle_04 + the lid closing (PRD 27) — is seen.
  setTimeout(() => { document.getElementById('shutdown-screen').hidden = false; }, 1200);
}
function requestShutdown() {
  shutdownRequested = true;
  ws.send(JSON.stringify({ action: 'shutdown' }));
  status.textContent = 'shutting down…';
  window.close(); // only works for script-opened tabs; otherwise the status line tells the story
}
document.getElementById('shutdown').onclick = requestShutdown;
document.getElementById('reload-characters').onclick = () => {
  ws.send(JSON.stringify({ action: 'reload_characters' }));
};

// ---------- backdrop picker (Settings view) ----------
let currentAvatarKey = null;
let backdropOptions = [];
let currentBackdropKey = 'none';
const backdropEl = document.getElementById('backdrop');
const backdropList = document.getElementById('backdrop-list');
const backdropThumbs = document.getElementById('backdrop-thumbs');

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

function setBackdrop(key) {
  ws.send(JSON.stringify({ action: 'set_backdrop', key }));
}

function renderBackdropList() {
  backdropList.replaceChildren(...backdropOptions.map(b => {
    const c = document.createElement('button');
    c.className = 'chip' + (b.key === currentBackdropKey ? ' active' : '');
    c.textContent = b.name;
    c.onclick = () => setBackdrop(b.key);
    return c;
  }));
  renderBackdropThumbs();
}

// The same picker as the Settings chips, as ~28px thumbs over the avatar (Chat view only).
function renderBackdropThumbs() {
  backdropThumbs.replaceChildren(...backdropOptions.map(b => {
    const t = document.createElement('button');
    t.className = 'bd-thumb' + (b.file ? '' : ' none') + (b.key === currentBackdropKey ? ' active' : '');
    t.title = b.name;
    t.setAttribute('aria-label', b.name);
    if (b.file) t.style.backgroundImage = `url("${b.file}")`;
    t.onclick = () => setBackdrop(b.key);
    return t;
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

const statusText = {
  idle: 'type a message or talk',
  processing: 'processing...',
  thinking: 'thinking...',
  speaking: 'speaking...',
};

let ws;
let state = 'idle';
let handsFree = false;   // mirrors the server-confirmed hands_free value (voice on/off)
let memorySaving = false;  // a background memory save is running (server `memory_saving`)
let muted = false;       // output mute: replies arrive and are shown, but not played
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
// while `mic.sending` is true (renderComposer gates it on hands-free).
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
  avatar.onSpeech(msg);   // PRD 27: end the thinking cue, greet on the session's first speech
  // Muted (PRD 24): the reply still arrives and its transcript is shown, we just never
  // decode or play it. `speech_end` still runs finishReply(), so the server's turn state
  // advances on playback_done exactly as it does when the audio was heard.
  if (muted) { adoptTurn(msg.turn); return; }
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
      renderComposer();
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
      renderComposer();
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
// Lerp each param of `target` from its current (motion-driven) value toward the target by
// `weight`; the expression and cue blends both layer on the model this way. Returns whether
// the blend is active at all (weight above the fade-out floor).
function blendParams(core, target, weight) {
  if (weight <= 0.01) return false;
  for (const [id, v] of Object.entries(target)) {
    const cur = core.getParameterValueById(id);
    core.setParameterValueById(id, cur + (v - cur) * weight);
  }
  return true;
}

const avatar = {
  model: null,
  currentEmotion: 'neutral',
  paramTarget: {},   // param id → target value for the active params-expression
  paramWeight: 0,    // current lerped weight of paramTarget

  // cues (PRD 27): a set of active cue names, merged into one parameter blend each frame
  activeCues: new Set(),
  cueTarget: {},     // param id → target value, merged from the active cues
  cuePulse: [],      // param ids that oscillate while their cue is active
  cueWeight: 0,
  lastIndex: {},     // motion group → last index played (no immediate repeat)
  pendingGroup: null,   // a group the library refused (something was playing): play it next
  lastTapAt: 0,
  greeted: false,    // Greet fires once per session, on the first speech

  app: null,

  async init() {
    try {
      const cfg = await (await fetch('/api/config')).json();
      currentAvatarKey = cfg.avatar;
      avatarName = cfg.name || avatarName;
      backdropOptions = cfg.backdrops || [];
      currentBackdropKey = cfg.backdrop || 'none';
      handsFree = !!cfg.hands_free;
      currentVerbosity = verbosityLevels.includes(cfg.verbosity) ? cfg.verbosity : 'normal';
      wikiVault = cfg.vault || wikiVault;
      renderDecayWindows(cfg.stale_days);
      renderCalendarHorizon(cfg.calendar_horizon_days);
      renderRecallMinScore(cfg.recall_min_score);
      settingsScreenshotEl.textContent = cfg.screenshot_cmd || '';
    } catch (e) { console.warn('config fetch failed, using default profile', e); }
    applyBackdrop(currentBackdropKey);
    renderVerbosity();
    renderBackdropList();
    renderWikiPanel();
    renderComposer();

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
    this.activeCues.clear(); this.cueTarget = {}; this.cuePulse = []; this.cueWeight = 0;
    this.lastIndex = {}; this.pendingGroup = null; this.lastTapAt = 0;

    let model;
    try {
      model = await PIXI.live2d.Live2DModel.from(profile.modelUrl);
    } catch (e) {
      // A model folder fetched before PRD 27 has no `.app.` file: fall back to the upstream one.
      try {
        model = await PIXI.live2d.Live2DModel.from(profile.fallbackModelUrl);
      } catch (e2) {
        console.warn('Live2D model failed to load', e, e2);
        avatarNote.hidden = false;
        return;
      }
    }
    avatarNote.hidden = true;
    this.model = model;
    this.app.stage.addChild(model);
    this.fit();

    const core = model.internalModel.coreModel;
    this.fitted = false;
    model.internalModel.on('afterMotionUpdate', () => {
      if (!this.fitted && this.artBox()) { this.fitted = true; this.fit(); }
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
      blendParams(core, this.paramTarget, this.paramWeight);
      // cue blend (PRD 27): props no clip ever keys — lid, steam, sparkle, ears — so this
      // just lerps them toward the merged cue target and back to 0 when the cue ends.
      const cueOn = Object.keys(this.cueTarget).length || this.cuePulse.length ? 1 : 0;
      this.cueWeight += (cueOn - this.cueWeight) * CONFIG.cueFade;
      if (blendParams(core, this.cueTarget, this.cueWeight)) {
        const t = Date.now() / 1000 * CONFIG.steamHz * 2 * Math.PI;
        this.cuePulse.forEach((id, i) => {
          core.setParameterValueById(id, this.cueWeight * (Math.sin(t + i * Math.PI / 2) + 1) / 2);
        });
      }
    });

    // Our own idle scheduler: the library picks idles uniformly and we want weights + no
    // immediate repeat, so its auto-idle group is pointed at nothing and we start the next
    // idle ourselves whenever a motion (idle or not) finishes. The microtask lets the
    // library's MotionState.complete() run first, or the IDLE-priority start is refused.
    const mm = model.internalModel.motionManager;
    mm.groups.idle = '__app_idle_off__';
    mm.on('motionFinish', () => {
      Promise.resolve().then(() => {
        const pending = this.pendingGroup;
        this.pendingGroup = null;
        if (pending) this.playGroup(pending);
        else this.playGroup(profile.idle);
      });
    });

    // Click/tap on the body: a big Pet after a pause, a small one when spammed.
    model.on('hit', (areas) => { if (areas && areas.includes('Body')) this.pet(); });

    this.playGroup(profile.idle);
  },

  // Play one clip from a motion group. Random index, never the same index twice in a row,
  // weighted where MOTIONS says so. Returns the index played, or null if nothing could play.
  playGroup(group, { index, priority } = {}) {
    const def = MOTIONS[group];
    if (!this.model || !def) return null;
    const defs = (this.model.internalModel.motionManager.definitions || {})[group] || [];
    if (!defs.length) return null;
    const pick = index !== undefined ? index : this.pickIndex(group, defs.length, def.weights);
    this.lastIndex[group] = pick;
    const prio = priority !== undefined ? priority : def.priority;
    Promise.resolve(this.model.motion(group, pick, prio)).then((ok) => {
      // Refused because something else is playing (a tap during the speaking/greet motion):
      // queue it instead of interrupting — motionFinish plays it next.
      if (ok === false && prio === PRIORITY.NORMAL) this.pendingGroup = group;
    }).catch(() => {});
    return pick;
  },

  pickIndex(group, count, weights) {
    const last = this.lastIndex[group];
    const options = [];
    for (let i = 0; i < count; i++) {
      if (i === last && count > 1) continue;           // no immediate repeat
      const w = weights && weights[i] !== undefined ? weights[i] : 1;
      for (let n = 0; n < Math.round(w * 2); n++) options.push(i);
    }
    if (!options.length) return 0;
    return options[Math.floor(Math.random() * options.length)];
  },

  // ---- cues: named parameter blends (+ an optional one-shot motion), merged per frame ----
  setCue(name, on) {
    const def = profile.cues[name];
    if (!def) return;
    if (on && !this.activeCues.has(name)) {
      this.activeCues.add(name);
      if (def.group) {
        this.playGroup(def.group, {
          index: def.index,
          priority: def.force ? PRIORITY.FORCE : undefined,
        });
      }
    } else if (!on) {
      this.activeCues.delete(name);
    }
    this.recomputeCue();
  },

  recomputeCue() {
    const target = {}, pulse = [];
    for (const name of this.activeCues) {
      const def = profile.cues[name] || {};
      Object.assign(target, def.params || {});
      for (const id of def.pulse || []) pulse.push(id);
    }
    this.cueTarget = target;
    this.cuePulse = pulse;
  },

  // A tap on the body: big reaction after a pause, small one when spammed, plus a short
  // happy blend. Never interrupts a motion that is already playing (playGroup queues it).
  pet(now = Date.now()) {
    const big = now - this.lastTapAt > CONFIG.petGapMs;
    this.lastTapAt = now;
    const group = big ? 'Pet' : 'PetSmall';
    this.playGroup(group);
    if (!playing) {
      this.setEmotion('happy');
      setTimeout(() => { if (!playing && this.currentEmotion === 'happy') this.setEmotion('neutral'); },
                 CONFIG.petHappyMs);
    }
    return group;
  },

  // ---- the state/listening transitions the cues hang off (PRD 27) ----
  onListening(value) {
    this.setCue('listening', value === 'hearing');
  },

  onState(value) {
    this.setCue('thinking', value === 'thinking');
    if (value === 'idle') this.setCue('tool', false);
  },

  // First speech of a reply: the thinking cue ends, the session's first one also greets.
  // A tool filler (chatbot.py marks it) sparkles instead of ending the thinking cue.
  onSpeech(msg) {
    if (msg.filler) { this.setCue('tool', true); return; }
    this.setCue('thinking', false);
    this.setCue('tool', false);
    if (!this.greeted) { this.greeted = true; this.setCue('greet', true); }
  },

  onShutdown() {
    this.activeCues.clear();
    this.setCue('shutdown', true);
  },

  // Box the drawn art occupies inside the model's canvas, in canvas units. Live2D canvases
  // have generous transparent margins, so the model's own size is a poor thing to fit against.
  // Returns null until the first frame has run and the vertices are filled in.
  artBox() {
    const im = this.model.internalModel, core = im.coreModel;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < core.getDrawableCount(); i++) {
      if (!core.getDrawableDynamicFlagIsVisible(i) || core.getDrawableOpacity(i) < 0.01) continue;
      const v = im.getDrawableVertices(i);
      for (let j = 0; j < v.length; j += 2) {
        if (v[j] < minX) minX = v[j];
        if (v[j] > maxX) maxX = v[j];
        if (v[j + 1] < minY) minY = v[j + 1];
        if (v[j + 1] > maxY) maxY = v[j + 1];
      }
    }
    const w = maxX - minX, h = maxY - minY;
    if (!(w > 0 && h > 0)) return null;
    return { x: minX, y: minY, w, h };
  },

  fit() {
    // The canvas resizes with its column (PIXI resizeTo: canvas.parentElement): scale the drawn
    // art to CONFIG.fitWidth/fitHeight of the column, whichever limits, and anchor it at fitAnchorY.
    const app = this.app, m = this.model;
    if (!m) return;
    const canvas = { w: m.width / m.scale.x, h: m.height / m.scale.y };
    const art = this.artBox() || { x: 0, y: 0, w: canvas.w, h: canvas.h };
    const scale = profile.avatarScale * Math.min(
      CONFIG.fitWidth * app.screen.width / art.w,
      CONFIG.fitHeight * app.screen.height / art.h,
    );
    m.scale.set(scale);
    m.x = app.screen.width / 2 - (art.x + art.w / 2) * scale;
    m.y = app.screen.height * CONFIG.fitAnchorY - (art.y + art.h / 2) * scale
        - art.h * scale * profile.avatarTopCrop;
  },

  setEmotion(emotion) {
    if (!this.model || emotion === this.currentEmotion) return;
    // An emotion tag we have no expression for is ignored, not a reset: whatever is
    // showing keeps showing until a known tag (or 'neutral') replaces it.
    if (emotion !== 'neutral' && !profile.expressions[emotion]) return;
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

// ---------- rail (PRD 22, 22b) ----------
// One table drives the rail buttons, the keyboard digits and the workspace view each one shows.
// Sources is the wiki view filtered to `sources/`; there is no contextual panel (PRD 22b) —
// settings live in the Settings view, view-local actions in the views themselves.
const SVG_NS = 'http://www.w3.org/2000/svg';

// The inline icons are [tag, attrs][] tables (rail items, the typed glyph): one builder.
function svgIcon(parts, attrs = {}) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  for (const [k, v] of Object.entries(attrs)) svg.setAttribute(k, String(v));
  for (const [tag, a] of parts) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(a)) node.setAttribute(k, String(v));
    svg.append(node);
  }
  return svg;
}

const RAIL_ICONS = {
  chat: [['path', { d: 'M4 5h16v10H8l-4 4z' }]],
  memory: [['ellipse', { cx: 12, cy: 6, rx: 7, ry: 2.6 }],
           ['path', { d: 'M5 6v6c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6V6' }],
           ['path', { d: 'M5 12v6c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6v-6' }]],
  wiki: [['path', { d: 'M4 5.5A2.5 2.5 0 0 1 6.5 3H12v16H6.5A2.5 2.5 0 0 0 4 21z' }],
         ['path', { d: 'M20 5.5A2.5 2.5 0 0 0 17.5 3H12v16h5.5A2.5 2.5 0 0 1 20 21z' }]],
  sources: [['path', { d: 'M4 13h4l2 3h4l2-3h4' }], ['path', { d: 'M5 6h14l1 7v5H4v-5z' }]],
  settings: [['path', { d: 'M4 7h9m3 0h4M4 17h4m3 0h9M4 12h14m3 0h1' }],
             ['circle', { cx: 15, cy: 7, r: 2 }], ['circle', { cx: 9.5, cy: 17, r: 2 }],
             ['circle', { cx: 19.5, cy: 12, r: 2 }]],
};
const VIEWS = [
  { id: 'chat', label: 'Chat', key: '1', el: 'view-chat' },
  { id: 'memory', label: 'Memory', key: '2', el: 'view-memory' },
  { id: 'wiki', label: 'Wiki', key: '3', el: 'view-wiki' },
  { id: 'sources', label: 'Sources', key: '4', el: 'view-wiki' },
  { id: 'settings', label: 'Settings', key: '5', el: 'view-settings', bottom: true },
];
const VIEW_BY_ID = new Map(VIEWS.map(v => [v.id, v]));

const railEl = document.getElementById('rail');
const decayListEl = document.getElementById('decay-list');
const calendarHorizonEl = document.getElementById('calendar-horizon');
const recallMinScoreEl = document.getElementById('recall-min-score');

let currentView = 'chat';
let railItems = [];

function railItem(v) {
  const b = el('button', 'rail-item');
  b.dataset.view = v.id;
  b.title = v.label + (v.key ? ` (${v.key})` : '');
  b.setAttribute('aria-label', v.label);
  b.append(el('span', 'rail-dot'));
  b.append(svgIcon(RAIL_ICONS[v.id], { 'aria-hidden': 'true' }), el('span', 'rail-label', v.label));
  if (v.key) b.append(el('span', 'key', v.key));
  if (v.id === 'sources') { const badge = el('span', 'rail-badge'); badge.hidden = true; b.append(badge); }
  b.onclick = () => setView(v.id);
  return b;
}

function buildRail() {
  const nodes = VIEWS.filter(v => !v.bottom).map(railItem);
  nodes.push(el('div', 'rail-spacer'), el('div', 'rail-sep'));
  nodes.push(...VIEWS.filter(v => v.bottom).map(railItem));
  railEl.replaceChildren(...nodes);
  railItems = Array.from(railEl.querySelectorAll('.rail-item'));
}

function railItemFor(id) {
  return railItems.find(b => b.dataset.view === id);
}

// Switching views does three things: move the companion column into this view's slot
// (chat slot on Chat, the shell's left slice everywhere else — PIXI only resizes on window
// events, so the canvas is resized and refitted after the move), show the view element,
// and load whatever data that view needs.
function setView(name) {
  const v = VIEW_BY_ID.get(name);
  if (!v) return;
  currentView = name;

  const chat = name === 'chat';
  const slot = chat ? chatSlot : leftSlot;
  companionEl.className = chat ? 'chat' : 'side';
  if (companionEl.parentNode !== slot) slot.appendChild(companionEl);
  requestAnimationFrame(() => {
    if (avatar.app) avatar.app.resize();
    avatar.fit();
  });

  for (const other of VIEWS) document.getElementById(other.el).hidden = true;
  document.getElementById(v.el).hidden = false;
  railItems.forEach(b => b.classList.toggle('active', b.dataset.view === name));
  // Wiki and Sources share one view element; only their footer action differs.
  wikiIngestBtn.hidden = name === 'sources';
  sourcesFileBtn.hidden = name !== 'sources';
  if (name === 'memory' && !memoryProfile) requestMemory();
  if (name === 'sources') sourcesMarkSeen();   // opening the inbox clears its badge
  if (name === 'wiki' || name === 'sources') {
    if (!wikiPages) requestWiki(); else { renderWikiList(); renderWikiPage(); }   // per-view list + head
  }
}

function renderDecayWindows(staleDays) {
  decayListEl.replaceChildren(...Object.entries(staleDays || {}).map(([section, days]) => {
    const row = el('div', 'decay-row');
    row.append(el('span', 'what', section), el('span', 'days', `${days} d`));
    return row;
  }));
}

// PRD 31: how far a dropped recurring invite is expanded into meeting stubs (settings.json).
let calendarHorizonDays = 14;
function renderCalendarHorizon(days) {
  if (Number.isInteger(days) && days > 0) calendarHorizonDays = days;
  calendarHorizonEl.value = String(calendarHorizonDays);
}
calendarHorizonEl.onchange = () => {
  const days = parseInt(calendarHorizonEl.value, 10);
  if (!Number.isInteger(days) || days < 1) { renderCalendarHorizon(); return; }   // revert a bad entry
  sendAction({ action: 'set_setting', key: 'calendar_horizon_days', value: days });
};

// PRD 32: the BM25-per-word floor a source passage must clear to be quoted into a turn.
let recallMinScore = 4.5;
function renderRecallMinScore(score) {
  if (typeof score === 'number' && Number.isFinite(score) && score >= 0) recallMinScore = score;
  recallMinScoreEl.value = String(recallMinScore);
}
recallMinScoreEl.onchange = () => {
  const score = parseFloat(recallMinScoreEl.value);
  if (!Number.isFinite(score) || score < 0) { renderRecallMinScore(); return; }   // revert a bad entry
  sendAction({ action: 'set_setting', key: 'recall_min_score', value: score });
};

buildRail();

// ---------- settings view ----------
const settingsVaultEl = document.getElementById('settings-vault');
const settingsOpenVaultEl = document.getElementById('settings-open-vault');
const settingsScreenshotEl = document.getElementById('settings-screenshot');

// ---------- memory view ----------
// One spec table drives all six sections: 'scalar' sections are key/value maps
// (identity, preferences), 'list' sections are arrays of fixed-field dicts.
// Every edit is sent as a single apply_ops op; the server answers with a fresh
// `memory` message that we re-render from, so the UI never guesses at state.
const MEMORY_SPEC = [
  { section: 'identity', kind: 'scalar' },
  { section: 'preferences', kind: 'scalar' },
  { section: 'people', kind: 'list', key: 'name', fields: ['name', 'rel', 'note'], retirable: true, archivable: true },
  { section: 'projects', kind: 'list', key: 'name', fields: ['name', 'status', 'note'], retirable: true, archivable: true },
  { section: 'recurring', kind: 'list', key: 'what', fields: ['what', 'when'], retirable: true, archivable: true },
  { section: 'episodic', kind: 'list', key: 'text', fields: ['text', 'date', 'importance', 'ttl_days'], retirable: false },
];
const MEMORY_NUMERIC = new Set(['importance', 'ttl_days']);
const MEM_SPEC_BY_SECTION = new Map(MEMORY_SPEC.map(s => [s.section, s]));

const memSectionsEl = document.getElementById('memory-sections');
const memRenderedEl = document.getElementById('memory-rendered');
const memCountEl = document.getElementById('memory-count');
const memErrorEl = document.getElementById('memory-error');
const memModelToggle = document.getElementById('memory-model-toggle');
const memSaveBtn = document.getElementById('memory-save-now');
const memSaveLabel = document.getElementById('memory-save-now-label');

//: how long "Save now" stays disabled if the server never answers with a `memory` message
const MEMORY_SAVE_TIMEOUT_MS = 30000;
let memSaveTimer = null;

let memoryProfile = null;    // last `profile` from the server
let memoryRendered = '';     // last `rendered` block
let memoryCount = 0;
let memoryModelView = false;
const memoryErrorEls = new Map();  // op path prefix → the row's error line

// Drag-and-drop state: the row currently being dragged ({section, index, kind})
// and the drop strips that only appear while a compatible drag is in progress.
const MEM_DRAG_MIME = 'application/x-memory-row';
const WIKI_DRAG_MIME = 'application/x-wiki-page';
let memDrag = null;
let memArmable = [];   // [{node, accepts}] — stale drop strips
let memOpenMenu = null;

// Every memory action is fire-and-forget: the server answers with a fresh `memory` message.
function sendAction(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
}

function requestMemory() {
  sendAction({ action: 'get_memory' });
}

function sendMemoryOp(op, path, value, reason) {
  memErrorEl.textContent = '';
  sendAction({ action: 'memory_op', op, path, value, reason: reason || 'ui' });
}

// Several ops the server applies atomically (used by the episodic → recurring promote).
function sendMemoryOps(ops) {
  memErrorEl.textContent = '';
  sendAction({ action: 'memory_ops', ops: ops.map(o => ({ reason: 'ui', ...o })) });
}

function setMemory(msg) {
  memorySaveDone();
  memoryProfile = msg.profile || {};
  memoryRendered = msg.rendered || '';
  memoryCount = typeof msg.count === 'number' ? msg.count : 0;
  renderMemory();
}

// The server names the exact path it rejected; walk up to the row that owns it.
function showMemoryError(path, reason) {
  const text = reason || 'rejected';
  const p = String(path || '');
  const el = memoryErrorEls.get(p)
    || memoryErrorEls.get(p.replace(/\.[^.]+$/, ''))
    || memoryErrorEls.get(p.split(/[.[]/)[0]);
  if (el) el.textContent = text;
  // No memory row owns this path — it came from a wiki action (paths there are `people/anna`).
  else if (currentView === 'wiki') { setWikiError(text); wikiIngestDone(); }
  else memErrorEl.textContent = text;
}

// ---- small builders ----
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = String(text);
  return n;
}

// Click → input; Enter or blur saves (when changed), Esc cancels.
function editable(value, placeholder, onSave) {
  const current = value === undefined || value === null ? '' : String(value);
  const span = el('span', 'mem-field', current === '' ? placeholder : current);
  if (current === '') span.classList.add('empty');
  span.title = 'click to edit';
  span.onclick = () => {
    const input = el('input', 'mem-input');
    input.value = current;
    input.size = Math.max(8, current.length + 2);
    let done = false;
    const finish = (save) => {
      if (done) return;
      done = true;
      const next = input.value.trim();
      input.replaceWith(span);
      if (save && next !== '' && next !== current) onSave(next);
    };
    input.onkeydown = (e) => {
      e.stopPropagation();  // Esc/Space are global shortcuts otherwise
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
    };
    input.onblur = () => finish(true);
    span.replaceWith(input);
    input.focus();
    input.select();
  };
  return span;
}

function deleteButton(path) {
  const b = el('button', 'mem-del', '🗑');
  b.title = 'delete';
  b.onclick = () => sendMemoryOp('DELETE', path);
  return b;
}

// ---- decay: seen labels ----
const MEM_DAY_MS = 86400000;

// Whole days between an ISO date and today (null when the date is unusable).
function memDaysSince(iso) {
  const t = Date.parse(`${iso}T00:00:00`);
  if (Number.isNaN(t)) return null;
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return Math.max(0, Math.round((today.getTime() - t) / MEM_DAY_MS));
}

function memRelative(days) {
  if (days === 0) return 'today';
  if (days === 1) return 'yesterday';
  if (days < 30) return `${days} days ago`;
  const months = Math.round(days / 30);
  if (months < 12) return `${months} month${months === 1 ? '' : 's'} ago`;
  const years = Math.round(days / 365);
  return `${years} year${years === 1 ? '' : 's'} ago`;
}

// Muted right-aligned "3 months ago" with the ISO date in the tooltip.
function seenLabel(entry, days) {
  if (!entry.seen || days === null) return null;
  const node = el('span', 'mem-seen', memRelative(days));
  node.title = `last confirmed ${entry.seen}`;
  return node;
}

// Half the section's window is when a row starts looking tired.
function memIsAging(entry, days) {
  return days !== null && typeof entry.stale_after === 'number' && days > entry.stale_after / 2;
}

// ---- the two moves ----
function retireEntry(section, index, on) {
  sendMemoryOp('UPDATE', `${section}[${index}].stale`, !!on);
}

// PRD 19: hand the entry to the wiki; the server answers with fresh `memory` + `wiki` messages.
function archiveEntry(section, index) {
  memErrorEl.textContent = '';
  sendAction({ action: 'archive_memory', section, index });
}

function promoteEpisodic(entry, index) {
  const value = { what: entry.text || '' };
  if (entry.when) value.when = entry.when;
  sendMemoryOps([
    { op: 'ADD', path: 'recurring[]', value },
    { op: 'DELETE', path: `episodic[${index}]` },
  ]);
}

// The moves offered for one row. `to` names the drop zone that performs the move, so the same
// table drives the "→" menu and drag-and-drop (see memDropZone) — no second set of conditionals.
function rowMoves(spec, entry, index, stale) {
  const moves = [];
  if (stale) moves.push({ label: 'restore', to: `${spec.section}/section`, run: () => retireEntry(spec.section, index, false) });
  else if (spec.section === 'episodic') moves.push({ label: 'make recurring', to: 'recurring/section', run: () => promoteEpisodic(entry, index) });
  else if (spec.retirable) moves.push({ label: 'retire', to: `${spec.section}/stale`, run: () => retireEntry(spec.section, index, true) });
  // PRD 19: file the entry into its wiki page (and mark it stale); the drop zone is the Wiki rail item.
  if (spec.archivable) moves.push({ label: 'archive', to: 'wiki/rail', run: () => archiveEntry(spec.section, index) });
  return moves;
}

// One entry of a section by index — how a drag payload resolves back to its row.
function memEntryAt(section, index) {
  const list = (memoryProfile && memoryProfile[section]) || [];
  return list[index] || {};
}

// The moves for whatever is being dragged: a memory row ({section, index, kind}) or,
// from the wiki view, a page ({path, type}). One table, so memDropZone stays payload-agnostic.
function dragMoves(drag) {
  if (!drag) return [];
  if (drag.path) return wikiMoves(drag);
  const spec = MEM_SPEC_BY_SECTION.get(drag.section);
  if (!spec) return [];
  return rowMoves(spec, memEntryAt(drag.section, drag.index), drag.index, drag.kind === 'stale');
}

// ---- inline "→" menu ----
function closeMemMenu() {
  if (memOpenMenu) memOpenMenu.remove();
  memOpenMenu = null;
}
document.addEventListener('click', (e) => {
  if (memOpenMenu && !memOpenMenu.contains(e.target)) closeMemMenu();
});

function moveButton(moves) {
  const b = el('button', 'mem-move', '→');
  b.title = 'move';
  b.onclick = (e) => {
    e.stopPropagation();
    const open = memOpenMenu;
    closeMemMenu();
    if (open && open.previousSibling === b) return;   // second click closes
    const menu = el('div', 'mem-menu');
    for (const move of moves) {
      const item = el('button', null, move.label);
      item.onclick = (ev) => { ev.stopPropagation(); closeMemMenu(); move.run(); };
      menu.append(item);
    }
    b.after(menu);
    memOpenMenu = menu;
  };
  return b;
}

// ---- drag and drop ----
function memReadDrag(e) {
  // A drag of files from the desktop belongs to the intake popup, never to a row drop zone.
  if (e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files')) return null;
  if (memDrag) return memDrag;
  for (const mime of [MEM_DRAG_MIME, WIKI_DRAG_MIME]) {
    try {
      const raw = e.dataTransfer.getData(mime);
      if (raw) return JSON.parse(raw);
    } catch (_) { /* getData is not readable during dragover in some browsers */ }
  }
  return null;
}

// Show the stale drop strips that accept the row being dragged.
function armDropTargets(drag) {
  for (const t of memArmable) t.node.classList.toggle('armed', t.accepts(drag));
}
function clearDropTargets() {
  for (const t of memArmable) t.node.classList.remove('armed');
}

function makeDraggable(row, drag, mime) {
  row.draggable = true;
  row.ondragstart = (e) => {
    memDrag = drag;
    e.dataTransfer.setData(mime || MEM_DRAG_MIME, JSON.stringify(drag));
    e.dataTransfer.effectAllowed = 'move';
    row.classList.add('mem-dragging');
    armDropTargets(drag);
  };
  row.ondragend = () => {
    memDrag = null;
    row.classList.remove('mem-dragging');
    clearDropTargets();
  };
}

// `node` accepts a drag exactly when one of the dragged row's moves names `zone`, and dropping
// runs that move. Only compatible drags get preventDefault (and therefore a drop) plus the
// highlight; `armable` also shows the zone while such a drag is in progress.
function memDropZone(node, zone, armable) {
  const moveFor = (drag) => dragMoves(drag).find(m => m.to === zone);
  node.ondragover = (e) => {
    if (!moveFor(memReadDrag(e))) return;
    e.preventDefault();
    node.classList.add('drop-ok');
  };
  node.ondragleave = () => node.classList.remove('drop-ok');
  node.ondrop = (e) => {
    node.classList.remove('drop-ok');
    const move = moveFor(memReadDrag(e));
    if (!move) return;
    e.preventDefault();
    clearDropTargets();
    move.run();
  };
  if (armable) memArmable.push({ node, accepts: (drag) => !!moveFor(drag) });
}

// A row plus the error line the server's memory_error is routed into.
function memRow(path) {
  const row = el('div', 'mem-row');
  const err = el('div', 'mem-error');
  memoryErrorEls.set(path, err);
  row.append(err);
  return { row, err };
}

// `identity.aliases` is the one list-valued scalar key: shown (and edited) as a comma-separated
// string, which the server normalises back into a list. See docs/26-identity-aliases.md.
function scalarRow(section, key, value, superseded) {
  const path = `${section}.${key}`;
  const { row, err } = memRow(path);
  const shown = Array.isArray(value) ? value.join(', ') : value;
  row.prepend(el('span', 'mem-key', key), editable(shown, '(empty)', v => sendMemoryOp('UPDATE', path, v)), deleteButton(path));
  const old = superseded && superseded[key];
  if (Array.isArray(old) && old.length) {
    const prev = old[old.length - 1];
    const was = el('div', 'mem-was', `was: ${prev} `);
    const restore = el('button', 'mem-restore', 'restore');
    restore.onclick = () => sendMemoryOp('UPDATE', path, prev, 'ui:restore');
    was.append(restore);
    row.insertBefore(was, err);
  }
  return row;
}

function listRow(spec, entry, index, stale) {
  const path = `${spec.section}[${index}]`;
  const { row, err } = memRow(path);
  const parts = [];
  spec.fields.forEach((field, i) => {
    if (i) parts.push(el('span', 'mem-sep', '·'));
    parts.push(editable(entry[field], field, v => {
      sendMemoryOp('UPDATE', `${path}.${field}`, MEMORY_NUMERIC.has(field) ? Number(v) : v);
    }));
  });
  if (entry.expired) {
    row.classList.add('expired');
    parts.push(el('span', 'mem-tag', 'expired'));
  }
  const days = entry.seen ? memDaysSince(entry.seen) : null;
  const seen = seenLabel(entry, days);
  if (seen) parts.push(seen);
  if (stale) row.classList.add('mem-stale');
  else if (memIsAging(entry, days)) row.classList.add('aging');
  const moves = rowMoves(spec, entry, index, stale);
  if (stale && moves.length) {
    // A stale row's one move is "restore"; it gets a spelled-out button as well as the menu.
    const restore = el('button', 'mem-restore', 'restore');
    restore.onclick = moves[0].run;
    parts.push(restore);
  }
  if (moves.length) parts.push(moveButton(moves));
  parts.push(deleteButton(path));
  row.prepend(...parts);
  makeDraggable(row, { section: spec.section, index, kind: stale ? 'stale' : 'live' });
  return row;
}

// Bottom-of-section group holding the retired rows; its header doubles as the
// "drop here to retire" target (rendered even when empty, shown only mid-drag).
function staleGroup(spec, entries) {
  const group = el('div', 'mem-stale-group');
  const head = el('div', 'mem-stale-head');
  const count = entries.length;
  head.textContent = count ? `stale (${count})` : 'drop here to retire';
  if (!count) head.classList.add('empty');
  if (spec.retirable) memDropZone(head, `${spec.section}/stale`, true);
  group.append(head, ...entries.map(([entry, i]) => listRow(spec, entry, i, true)));
  group.hidden = !count && !spec.retirable;
  return group;
}

function addForm(spec) {
  const form = el('div', 'mem-add');
  const fields = spec.kind === 'scalar' ? ['key', 'value'] : spec.fields;
  const inputs = fields.map(f => {
    const i = el('input');
    i.placeholder = f;
    return i;
  });
  const submit = () => {
    const values = inputs.map(i => i.value.trim());
    if (!values[0]) { inputs[0].focus(); return; }
    if (spec.kind === 'scalar') {
      if (!values[1]) { inputs[1].focus(); return; }
      sendMemoryOp('UPDATE', `${spec.section}.${values[0]}`, values[1]);
    } else {
      const value = {};
      fields.forEach((f, i) => {
        if (!values[i]) return;
        value[f] = MEMORY_NUMERIC.has(f) ? Number(values[i]) : values[i];
      });
      sendMemoryOp('ADD', `${spec.section}[]`, value);
    }
    inputs.forEach(i => { i.value = ''; });
  };
  inputs.forEach(i => {
    i.onkeydown = (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') { e.preventDefault(); submit(); }
    };
  });
  const btnAdd = el('button', null, '+ add');
  btnAdd.onclick = submit;
  // The section itself owns errors from an ADD (path `people[]`, no row to blame).
  memoryErrorEls.set(spec.section, el('div', 'mem-error'));
  form.append(...inputs, btnAdd);
  return form;
}

function memorySection(spec) {
  const wrap = el('div', 'mem-section');
  const title = el('div', 'mem-section-title', spec.section);
  wrap.append(title);
  const data = (memoryProfile && memoryProfile[spec.section]) || (spec.kind === 'scalar' ? {} : []);
  let rows = [];
  let stale = [];
  if (spec.kind === 'scalar') {
    const superseded = data.superseded || null;
    rows = Object.keys(data)
      .filter(k => k !== 'superseded')
      .map(k => scalarRow(spec.section, k, data[k], superseded));
  } else {
    // the title accepts an episodic row (promote) or one of its own stale rows (restore)
    memDropZone(title, `${spec.section}/section`, false);
    const live = [];
    data.forEach((entry, i) => (entry && entry.stale ? stale : live).push([entry || {}, i]));
    rows = live.map(([entry, i]) => listRow(spec, entry, i, false));
  }
  if (rows.length) wrap.append(...rows);
  else wrap.append(el('div', 'mem-empty', 'nothing yet'));
  if (spec.kind === 'list') wrap.append(staleGroup(spec, stale));
  wrap.append(addForm(spec));
  wrap.append(memoryErrorEls.get(spec.section));
  return wrap;
}

function renderMemory() {
  memoryErrorEls.clear();
  memArmable = [];
  memDrag = null;
  closeMemMenu();
  memErrorEl.textContent = '';
  memCountEl.textContent = `Memory · ${memoryCount} ${memoryCount === 1 ? 'fact' : 'facts'}`;
  memModelToggle.classList.toggle('active', memoryModelView);
  memSectionsEl.hidden = memoryModelView;
  memRenderedEl.hidden = !memoryModelView;
  if (memoryModelView) {
    memRenderedEl.textContent = memoryRendered || '(nothing yet)';
    memSectionsEl.replaceChildren();
    return;
  }
  memSectionsEl.replaceChildren(...MEMORY_SPEC.map(memorySection));
}

function memorySaveDone() {
  if (memSaveTimer) { clearTimeout(memSaveTimer); memSaveTimer = null; }
  memSaveBtn.disabled = false;
  memSaveLabel.textContent = 'Save now';
}

// Fold the session into memory.json on demand; the server answers with a `memory` message
// (or an `error` while a reply is still running), and either re-enables the button.
memSaveBtn.onclick = () => {
  if (memSaveBtn.disabled) return;
  memErrorEl.textContent = '';
  memSaveBtn.disabled = true;
  memSaveLabel.textContent = 'saving…';
  memSaveTimer = setTimeout(memorySaveDone, MEMORY_SAVE_TIMEOUT_MS);
  sendAction({ action: 'save_memory' });
};

memModelToggle.onclick = () => { memoryModelView = !memoryModelView; renderMemory(); };
document.getElementById('memory-reset').onclick = () => {
  if (!confirm('Forget everything the assistant remembers about you? This cannot be undone.')) return;
  sendAction({ action: 'memory_reset' });
};

// ---------- wiki view ----------
// A light editor over ~/llm-wiki (PRD 28): the server sends the page index (`wiki`) and one
// page at a time (`wiki_page`, with its content-hash etag and backlinks). Writes from here are
// whole-page saves guarded by that etag, create/rename/delete, an appended note, an on-demand
// ingest and the two drag bridges to the memory view (archive / pull).
const WIKI_FOLDERS = ['people', 'projects', 'meetings', 'sources', 'notes'];
const WIKI_LINK_RE = /(!?)\[\[([^[\]]+)\]\]/g;
const WIKI_PLACEHOLDER_RE = /(\d+)/g;
const WIKI_INGEST_TIMEOUT_MS = 60000;
const WIKI_RECENT_MAX = 5;
// What a rendered page may contain: marked's output is parsed into an inert document and only
// these tags/attributes are copied out, so user text never reaches innerHTML.
const WIKI_SAFE_TAGS = new Set(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'blockquote',
  'pre', 'code', 'em', 'strong', 'del', 'a', 'br', 'hr', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
  'input', 'span', 'div', 'img']);
const WIKI_SAFE_ATTRS = { a: ['href', 'title'], input: ['type', 'checked', 'disabled'], img: ['alt', 'title'],
  th: ['align'], td: ['align'], code: ['class'] };

const wikiListEl = document.getElementById('wiki-list');
const wikiHeadEl = document.getElementById('wiki-page-head');
const wikiFmEl = document.getElementById('wiki-fm');
const wikiTextEl = document.getElementById('wiki-text');
const wikiEditorEl = document.getElementById('wiki-editor');
const wikiConflictEl = document.getElementById('wiki-conflict');
const wikiConflictReloadBtn = document.getElementById('wiki-conflict-reload');
const wikiConflictOverwriteBtn = document.getElementById('wiki-conflict-overwrite');
const wikiBacklinksEl = document.getElementById('wiki-backlinks');
const wikiNoteEl = document.getElementById('wiki-note');
const wikiLintEl = document.getElementById('wiki-lint');
const wikiVaultEl = document.getElementById('wiki-vault');
const wikiErrorEl = document.getElementById('wiki-error');
const wikiIngestBtn = document.getElementById('wiki-ingest');
const wikiIngestLabel = document.getElementById('wiki-ingest-label');
const wikiNoteInput = document.getElementById('wiki-note-input');
const wikiNoteBtn = document.getElementById('wiki-note-add');
const wikiSearchInput = document.getElementById('wiki-search');
const wikiTodayBtn = document.getElementById('wiki-today');
const wikiNewBtn = document.getElementById('wiki-new');
const sourcesFileBtn = document.getElementById('sources-file');
const sourcesInputEl = document.getElementById('sources-input');

let wikiPages = null;      // last `wiki` page list (null = never loaded)
let wikiLint = [];
let wikiVault = '';
let wikiSelected = null;   // path of the open page, kept across refreshes
let wikiPage = null;       // last `wiki_page` for wikiSelected
let wikiIngestTimer = null;
let wikiMode = 'view';     // 'view' (rendered) | 'edit' (textarea)
let wikiDraft = null;      // unsaved editor text (null = nothing typed since the last load)
let wikiConflict = null;   // the server's {text, etag} after a stale save
let wikiTag = null;        // the Tags-group filter
let wikiSearch = null;     // last `wiki_search` hits (null = no query)
const wikiRecent = [];     // MRU paths, client-side only

function requestWiki() {
  sendAction({ action: 'get_wiki' });
}

function setWikiError(text) {
  wikiErrorEl.textContent = text || '';
}

// A wiki page can be pulled back into the matching profile section; the Memory tab is a plain
// "take me there" target so a mis-aimed drop is not a dead end.
function wikiMoves(drag) {
  const moves = [];
  const section = drag.type === 'person' ? 'people' : drag.type === 'project' ? 'projects' : null;
  if (section) {
    moves.push({
      label: 'pull into memory',
      to: `${section}/section`,
      run: () => { setWikiError(''); sendAction({ action: 'pull_wiki', path: drag.path, section }); },
    });
  }
  moves.push({ label: 'open memory', to: 'memory/rail', run: () => setView('memory') });
  return moves;
}

function wikiFolder(path) {
  const i = String(path).indexOf('/');
  return i === -1 ? 'root' : path.slice(0, i);
}

// Known folders first in a fixed order, then anything else alphabetically, root last.
function wikiGroups(pages) {
  const groups = new Map();
  for (const page of pages) {
    const folder = wikiFolder(page.path);
    if (!groups.has(folder)) groups.set(folder, []);
    groups.get(folder).push(page);
  }
  const rank = (f) => (f === 'root' ? 2 : WIKI_FOLDERS.includes(f) ? 0 : 1);
  return Array.from(groups.entries()).sort((a, b) => {
    const ra = rank(a[0]);
    const rb = rank(b[0]);
    if (ra !== rb) return ra - rb;
    if (ra === 0) return WIKI_FOLDERS.indexOf(a[0]) - WIKI_FOLDERS.indexOf(b[0]);
    return a[0].localeCompare(b[0]);
  });
}

function wikiKnown(path) {
  return (wikiPages || []).some(p => p.path === path);
}

// Make `path` (or nothing) the open page: whatever was loaded, typed or in conflict is gone.
function wikiOpen(path) {
  wikiSelected = path;
  wikiPage = null;
  wikiMode = 'view';
  wikiDraft = null;
  wikiConflict = null;
}

// Paths are suffix-less everywhere (`people/anna`) — in the page list, in wikilinks and in
// every action sent back; the server is what normalises them onto files.
function selectWikiPage(path) {
  path = String(path || '').trim();
  if (!path) return;
  if (path !== wikiSelected && wikiDraft !== null && !wikiDiscard()) return;
  wikiOpen(path);
  const at = wikiRecent.indexOf(path);
  if (at !== -1) wikiRecent.splice(at, 1);
  wikiRecent.unshift(path);
  wikiRecent.length = Math.min(wikiRecent.length, WIKI_RECENT_MAX);
  setWikiError('');
  sendAction({ action: 'read_wiki_page', path });
  renderWikiList();
  renderWikiPage();
}

// Create a page the server does not have yet and open it when the `wiki_page` comes back.
function createWikiPage(path) {
  path = String(path || '').trim().replace(/\.md$/, '');
  if (!path) return;
  if (wikiDraft !== null && !wikiDiscard()) return;
  wikiOpen(path);
  setWikiError('');
  sendAction({ action: 'wiki_create', path });
  renderWikiPage();
}

function wikiRow(page) {
  const row = el('div', 'wiki-row');
  if (page.path === wikiSelected) row.classList.add('selected');
  row.append(el('span', 'wiki-row-title', page.title || page.path));
  if (page.type) row.append(el('span', 'wiki-row-type', page.type));
  row.onclick = () => selectWikiPage(page.path);
  // Only the two page kinds a profile section can hold are draggable.
  if (page.type === 'person' || page.type === 'project') {
    makeDraggable(row, { path: page.path, type: page.type }, WIKI_DRAG_MIME);
  }
  return row;
}

function wikiTagRow(tag, count) {
  const row = el('div', 'wiki-row wiki-tag-row');
  if (tag === wikiTag) row.classList.add('selected');
  row.append(el('span', 'wiki-row-title', `#${tag}`), el('span', 'wiki-row-type', String(count)));
  row.onclick = () => { wikiTag = wikiTag === tag ? null : tag; renderWikiList(); };
  return row;
}

function wikiHitRow(hit) {
  const meta = (wikiPages || []).find(p => p.path === hit.path);
  const row = el('div', 'wiki-row wiki-hit');
  if (hit.path === wikiSelected) row.classList.add('selected');
  const text = el('div', 'wiki-hit-text');
  text.append(el('div', 'wiki-row-title', (meta && meta.title) || hit.path));
  if (hit.snippet) text.append(el('div', 'wiki-hit-snippet', hit.snippet));
  row.append(text);
  row.onclick = () => selectWikiPage(hit.path);
  return row;
}

function wikiTagCounts(pages) {
  const counts = new Map();
  for (const page of pages) for (const tag of page.tags || []) counts.set(tag, (counts.get(tag) || 0) + 1);
  return Array.from(counts.entries()).sort((a, b) => a[0].localeCompare(b[0]));
}

// ---- Sources view (PRD 25): the manifest as an inbox, newest first, in the same list element ----
let sourcesList = null;      // last `sources` message (null = never loaded)
let sourcesSeen = null;      // shas known when the view was last opened (null = baseline pending)
let sourcesKind = null;      // the kind-chip filter
let sourcesBusy = false;     // a re-ingest / forget is in flight (buttons disabled until `sources`)

function requestSources() {
  sendAction({ action: 'get_sources' });
}

// The badge is session-local: new shas since the view was last opened (or since connect).
function renderSourceBadge() {
  const item = railItemFor('sources');
  const badge = item && item.querySelector('.rail-badge');
  if (!badge) return;
  const fresh = sourcesSeen ? (sourcesList || []).filter(s => !sourcesSeen.has(s.sha)).length : 0;
  badge.textContent = String(fresh);
  badge.hidden = !fresh || currentView === 'sources';
}

function sourcesMarkSeen() {
  sourcesSeen = new Set((sourcesList || []).map(s => s.sha));
  renderSourceBadge();
}

function setSources(msg) {
  sourcesList = Array.isArray(msg.sources) ? msg.sources : [];
  sourcesBusy = false;
  if (sourcesSeen === null || currentView === 'sources') sourcesMarkSeen(); else renderSourceBadge();
  if (sourcesKind && !sourcesList.some(s => s.kind === sourcesKind)) sourcesKind = null;
  if (currentView === 'sources') { renderWikiList(); renderWikiPage(); }
}

// Re-ingest / Forget: mark the head busy until the next `sources` (or `error`) message.
function sourceAction(payload) {
  sourcesBusy = true;
  setWikiError('');
  sendAction(payload);
  renderWikiPage();
}

function sourceRow(s) {
  const row = el('div', 'wiki-row source-row');
  row.dataset.sha = s.sha;
  if (s.path === wikiSelected) row.classList.add('selected');
  row.append(el('span', 'source-kind', s.kind || '?'));
  const text = el('div', 'source-text');
  text.append(el('div', 'wiki-row-title', s.title || s.path));
  const fed = s.fed.length === 1 ? '1 page fed' : `${s.fed.length} pages fed`;
  text.append(el('div', 'source-meta', `${s.when} · ${fed}`));
  row.append(text);
  row.onclick = () => selectWikiPage(s.path);
  return row;
}

function renderSourcesList() {
  if (!sourcesList) { wikiListEl.replaceChildren(el('div', 'mem-empty', 'loading…')); return; }
  if (!sourcesList.length) { wikiListEl.replaceChildren(el('div', 'mem-empty', 'nothing filed yet')); return; }
  const kinds = Array.from(new Set(sourcesList.map(s => s.kind).filter(Boolean))).sort();
  const chips = el('div', 'chip-grid source-kinds');
  for (const kind of kinds) {
    const chip = el('button', 'chip', kind);
    if (kind === sourcesKind) chip.classList.add('active');
    chip.onclick = () => { sourcesKind = sourcesKind === kind ? null : kind; renderWikiList(); };
    chips.append(chip);
  }
  const rows = sourcesList.filter(s => !sourcesKind || s.kind === sourcesKind).map(sourceRow);
  wikiListEl.replaceChildren(chips, ...(rows.length ? rows : [el('div', 'mem-empty', 'no sources of that kind')]));
}

function sourceSelected() {
  return (sourcesList || []).find(s => s.path === wikiSelected) || null;
}

// The three per-source actions on the page head (only in the Sources view).
function sourceHeadButtons(s) {
  const raw = el('a', 'wiki-obsidian source-open-raw', 'Open raw');
  raw.href = `${vaultHref()}&file=${encodeURIComponent(s.raw)}`;
  const reingest = wikiHeadButton('source-reingest', sourcesBusy ? 'working…' : 'Re-ingest',
    () => sourceAction({ action: 'reingest_source', sha: s.sha }), sourcesBusy);
  const forget = wikiHeadButton('source-forget', 'Forget', () => {
    if (!confirm(`Forget "${s.title}"? Its page, raw file and the notes it added are removed (one git commit).`)) return;
    sourceAction({ action: 'forget_source', sha: s.sha, confirm: true });
  }, sourcesBusy);
  const buttons = [raw, reingest, forget];
  // A transcript / calendar drop names the meeting page it produced (PRD 31).
  if (s.meeting) {
    buttons.unshift(wikiHeadButton('source-meeting', 'Meeting', () => { setView('wiki'); selectWikiPage(s.meeting); }));
  }
  return buttons;
}

function renderWikiList() {
  // The Sources view is the inbox: manifest rows, newest first, no folders/tags/search.
  if (currentView === 'sources') { renderSourcesList(); return; }
  const all = wikiPages || [];
  if (!all.length) {
    wikiListEl.replaceChildren(el('div', 'mem-empty', !wikiPages ? 'loading…' : 'no pages yet'));
    return;
  }
  const nodes = [];
  if (wikiSearch) {
    nodes.push(el('div', 'wiki-group-title', 'results'));
    if (!wikiSearch.length) nodes.push(el('div', 'mem-empty', 'no matches'));
    nodes.push(...wikiSearch.map(wikiHitRow));
    wikiListEl.replaceChildren(...nodes);
    return;
  }
  const pages = wikiTag ? all.filter(p => (p.tags || []).includes(wikiTag)) : all;
  const recent = wikiRecent.map(path => all.find(p => p.path === path)).filter(Boolean);
  if (!wikiTag && recent.length) {
    nodes.push(el('div', 'wiki-group-title', 'recent'));
    nodes.push(...recent.map(wikiRow));
  }
  for (const [folder, group] of wikiGroups(pages)) {
    nodes.push(el('div', 'wiki-group-title', folder));
    group.sort((a, b) => String(a.title || a.path).localeCompare(String(b.title || b.path)));
    nodes.push(...group.map(wikiRow));
  }
  const tagCounts = wikiTagCounts(all);
  if (tagCounts.length) {
    nodes.push(el('div', 'wiki-group-title', 'tags'));
    nodes.push(...tagCounts.map(([tag, count]) => wikiTagRow(tag, count)));
  }
  wikiListEl.replaceChildren(...nodes);
}

// ---- rendering: frontmatter strip + body (marked when present, plain text otherwise) ----

// Frontmatter keys worth a chip at a glance; everything else (hashes, paths, counters) sits
// behind a "details" toggle so a source page's head stays one line.
const WIKI_FM_SHOWN = ['tags', 'kind', 'source', 'updated', 'date', 'time', 'attendees', 'meeting'];
let wikiFmOpen = false;

function wikiFmChips(fm) {
  const key = line => line.split(':')[0].trim();
  const shown = fm.filter(line => WIKI_FM_SHOWN.includes(key(line)));
  const rest = fm.filter(line => !WIKI_FM_SHOWN.includes(key(line)));
  const chips = shown.map(line => el('span', 'wiki-fm-item', line));
  if (rest.length) {
    const toggle = el('button', 'wiki-fm-toggle', wikiFmOpen ? 'less' : `details (${rest.length})`);
    toggle.type = 'button';
    toggle.onclick = () => { wikiFmOpen = !wikiFmOpen; renderWikiPage(); };
    chips.push(toggle);
    if (wikiFmOpen) chips.push(...rest.map(line => el('span', 'wiki-fm-item wiki-fm-more', line)));
  }
  return chips;
}

function wikiSplitFrontmatter(text) {
  text = String(text || '');
  if (!text.startsWith('---')) return { fm: [], body: text };
  const lines = text.split('\n');
  const end = lines.findIndex((l, i) => i > 0 && l.trim() === '---');
  if (end === -1) return { fm: [], body: text };
  return { fm: lines.slice(1, end).filter(l => l.trim()), body: lines.slice(end + 1).join('\n') };
}

// `[[page|alias]]` → { target: 'page', label: 'alias' }; `[[page#h]]` → target 'page', label 'page#h'.
function wikiLinkParts(inner, embed) {
  const alias = inner.indexOf('|');
  const label = alias === -1 ? inner : inner.slice(alias + 1);
  const target = (alias === -1 ? inner : inner.slice(0, alias)).split('#')[0].trim();
  return { target, label: label.trim() || target, embed: !!embed };
}

function wikiLinkSpan(link) {
  const span = el('span', 'wiki-link', link.embed ? `![[${link.label}]]` : link.label);
  span.dataset.target = link.target;
  if (link.embed) span.classList.add('embed');
  if (!wikiKnown(link.target)) {
    span.classList.add('missing');
    span.title = 'no such page (yet) — click to create it';
    span.onclick = () => { if (confirm(`Create ${link.target}?`)) createWikiPage(link.target); };
  } else {
    span.onclick = () => selectWikiPage(link.target);
  }
  return span;
}

// Copy `node` out of the inert parsed document, keeping only whitelisted tags and attributes.
function wikiImportSafe(node) {
  if (node.nodeType === 3) return document.createTextNode(node.textContent);
  if (node.nodeType !== 1) return null;
  const tag = node.tagName.toLowerCase();
  const out = WIKI_SAFE_TAGS.has(tag) ? document.createElement(tag) : document.createElement('span');
  for (const name of WIKI_SAFE_ATTRS[tag] || []) {
    const value = node.getAttribute(name);
    if (value === null) continue;
    if (name === 'href' && !/^(https?:|mailto:)/i.test(value.trim())) continue;
    out.setAttribute(name, value);
  }
  if (tag === 'a') { out.setAttribute('target', '_blank'); out.setAttribute('rel', 'noopener'); }
  for (const child of Array.from(node.childNodes)) {
    const imported = wikiImportSafe(child);
    if (imported) out.append(imported);
  }
  return out;
}

// Replace every placeholder text run inside `root` with its link span (recursively).
function wikiSwapPlaceholders(root, links) {
  const kids = Array.from(root.childNodes || root.children || []);
  const next = [];
  for (const kid of kids) {
    if (kid.nodeType === 3) {
      const parts = String(kid.textContent).split(WIKI_PLACEHOLDER_RE);
      parts.forEach((part, i) => {
        if (i % 2 === 0) { if (part) next.push(document.createTextNode(part)); }
        else next.push(wikiLinkSpan(links[Number(part)]));
      });
    } else {
      if (kid.childNodes || kid.children) wikiSwapPlaceholders(kid, links);
      next.push(kid);
    }
  }
  root.replaceChildren(...next);
}

function wikiBodyNodes(body) {
  const links = [];
  const marked = String(body).replace(WIKI_LINK_RE, (_, bang, inner) => {
    links.push(wikiLinkParts(inner, bang));
    return `${links.length - 1}`;
  });
  let container;
  if (typeof window.marked !== 'undefined' && typeof DOMParser !== 'undefined') {
    const parsed = new DOMParser().parseFromString(window.marked.parse(marked), 'text/html');
    container = el('div', 'wiki-rendered');
    for (const child of Array.from(parsed.body.childNodes)) {
      const imported = wikiImportSafe(child);
      if (imported) container.append(imported);
    }
  } else {
    container = el('pre', 'wiki-plain');
    container.append(document.createTextNode(marked));
  }
  wikiSwapPlaceholders(container, links);
  return container;
}

function wikiDirty() {
  return wikiDraft !== null && !!wikiPage && wikiDraft !== wikiPage.text;
}

// Esc / navigating away with a dirty draft: ask, then drop it.
function wikiDiscard() {
  if (wikiDirty() && !confirm('Discard unsaved changes?')) return false;
  wikiDraft = null;
  wikiConflict = null;
  return true;
}

// Preview keeps the draft (Save stays enabled); only Esc discards.
function wikiSetMode(mode) {
  wikiMode = mode;
  renderWikiPage();
  if (mode === 'edit') wikiEditorEl.focus();
}

function saveWikiPage(etag) {
  if (!wikiPage || !wikiSelected || wikiIngestTimer) return;
  const text = wikiDraft === null ? wikiPage.text : wikiDraft;
  if (text === wikiPage.text && !etag) return;
  setWikiError('');
  sendAction({ action: 'wiki_save', path: wikiSelected, text, etag: etag || wikiPage.etag });
}

function wikiHeadButton(cls, label, onclick, disabled) {
  const b = el('button', cls, label);
  b.onclick = onclick;
  b.disabled = !!disabled;
  return b;
}

function renameWikiPage() {
  const n = (wikiPage && wikiPage.backlinks || []).length;
  const target = prompt(`Rename/move to (folder/name) — ${n} page${n === 1 ? '' : 's'} link here, links will be rewritten:`, wikiSelected);
  if (!target || target.trim() === wikiSelected) return;
  const from = wikiSelected;
  wikiOpen(target.trim().replace(/\.md$/, ''));
  setWikiError('');
  sendAction({ action: 'wiki_rename', path: from, new_path: wikiSelected });
  renderWikiPage();
}

function deleteWikiPage() {
  const n = (wikiPage && wikiPage.backlinks || []).length;
  if (!confirm(`Delete ${wikiSelected}? Links from ${n} page${n === 1 ? '' : 's'} will break.`)) return;
  const path = wikiSelected;
  wikiOpen(null);
  setWikiError('');
  sendAction({ action: 'wiki_delete', path });
  renderWikiPage();
}

function renderWikiPage() {
  const meta = (wikiPages || []).find(p => p.path === wikiSelected);
  if (!wikiSelected) {
    wikiHeadEl.replaceChildren(el('div', 'wiki-muted', 'pick a page'));
    wikiTextEl.replaceChildren();
    wikiFmEl.hidden = wikiTextEl.hidden = wikiEditorEl.hidden = wikiConflictEl.hidden = true;
    wikiBacklinksEl.hidden = wikiNoteEl.hidden = true;
    return;
  }
  const editing = wikiMode === 'edit';
  const head = [];
  const title = el('div', 'wiki-title-row');
  title.append(el('div', 'wiki-title', (meta && meta.title) || wikiSelected));
  const crumb = wikiSelected.includes('/') ? wikiSelected.split('/') : [wikiSelected];
  title.append(el('span', 'wiki-crumb', crumb.join(' › ')));
  head.push(title);
  const sub = el('div', 'wiki-sub');
  sub.append(el('span', 'wiki-path', wikiSelected));
  if (meta && meta.updated) sub.append(el('span', 'wiki-muted', `updated ${meta.updated}`));
  if (wikiPage) {
    sub.append(wikiHeadButton('wiki-edit', editing ? 'Preview' : 'Edit',
      () => wikiSetMode(editing ? 'view' : 'edit')));
    sub.append(wikiHeadButton('wiki-save', 'Save', () => saveWikiPage(), !wikiDirty() || !!wikiIngestTimer));
  }
  if (wikiVault) {
    const a = el('a', 'wiki-obsidian', 'Open in Obsidian');
    a.href = `${vaultHref()}&file=${encodeURIComponent(wikiSelected)}`;
    sub.append(a);
  }
  const source = currentView === 'sources' && wikiPage ? sourceSelected() : null;
  if (source) sub.append(...sourceHeadButtons(source));
  // Fallback for the drag bridge: the same move, as a button.
  const pull = meta && wikiMoves({ path: wikiSelected, type: meta.type }).find(m => m.to.endsWith('/section'));
  if (pull) {
    const b = el('button', 'wiki-pull', `→ ${pull.label}`);
    b.onclick = pull.run;
    sub.append(b);
  }
  if (wikiPage && wikiPage.etag) {
    const menu = el('span', 'wiki-menu');
    const more = wikiHeadButton('wiki-more', '⋯', () => menu.classList.toggle('open'));
    more.title = 'rename / delete';
    menu.append(more,
      wikiHeadButton('wiki-menu-item', 'Rename/move', renameWikiPage),
      wikiHeadButton('wiki-menu-item', 'Delete', deleteWikiPage));
    sub.append(menu);
  }
  head.push(sub);
  wikiHeadEl.replaceChildren(...head);

  const text = wikiDraft === null ? (wikiPage ? wikiPage.text : '') : wikiDraft;
  const { fm, body } = wikiSplitFrontmatter(text);
  wikiFmEl.hidden = editing || !fm.length;
  wikiFmEl.replaceChildren(...wikiFmChips(fm));
  wikiTextEl.hidden = editing;
  wikiEditorEl.hidden = !editing;
  if (editing) {
    if (wikiEditorEl.value !== text) wikiEditorEl.value = text;
    wikiTextEl.replaceChildren();
  } else {
    wikiTextEl.replaceChildren(wikiPage ? wikiBodyNodes(body) : document.createTextNode('loading…'));
  }
  wikiConflictEl.hidden = !wikiConflict;
  const backlinks = (wikiPage && wikiPage.backlinks) || [];
  wikiBacklinksEl.hidden = !backlinks.length;
  wikiBacklinksEl.replaceChildren(el('div', 'wiki-group-title', 'backlinks'), ...backlinks.map(path => {
    const row = el('div', 'wiki-backlink', path);
    row.onclick = () => selectWikiPage(path);
    return row;
  }));
  wikiNoteEl.hidden = editing;
}

// `obsidian://open?vault=<the vault folder's name>` — the same link the page head uses.
function vaultHref() {
  if (!wikiVault) return '#';
  return `obsidian://open?vault=${encodeURIComponent(wikiVault.replace(/\/+$/, '').split('/').pop())}`;
}

// The wiki footer (lint, vault) and the vault line in the Settings view.
function renderWikiPanel() {
  wikiVaultEl.textContent = wikiVault || '(not created yet)';
  wikiLintEl.hidden = !wikiLint.length;
  wikiLintEl.replaceChildren(...wikiLint.map(line => el('div', 'wiki-lint-line', line)));
  settingsVaultEl.textContent = wikiVault || '(not created yet)';
  settingsOpenVaultEl.href = vaultHref();
}

function setWiki(msg) {
  wikiPages = Array.isArray(msg.pages) ? msg.pages : [];
  wikiLint = Array.isArray(msg.lint) ? msg.lint : [];
  wikiVault = msg.vault || wikiVault;
  wikiIngestDone();
  if (wikiTag && !wikiPages.some(p => (p.tags || []).includes(wikiTag))) wikiTag = null;
  // Keep the open page open across refreshes and re-read it (an ingest may have changed it) —
  // unless it is being edited: the next save then reports the conflict instead.
  if (wikiSelected && !wikiKnown(wikiSelected)) wikiOpen(null);
  else if (wikiSelected && wikiDraft === null && wikiMode === 'view') {
    sendAction({ action: 'read_wiki_page', path: wikiSelected });
  }
  renderWikiList();
  renderWikiPage();
  renderWikiPanel();
  if (!intakeEl.hidden) intakeSyncPages();   // the popup's picker is fed by this list
}

function setWikiPage(msg) {
  if (msg.path !== wikiSelected) return;   // a stale answer for a page we navigated away from
  wikiPage = msg;
  // A fresh page after our own save: the draft is on disk now.
  if (wikiDraft !== null && wikiConflict === null && msg.etag && wikiDraftSaved(msg)) { wikiDraft = null; wikiMode = 'view'; }
  renderWikiPage();
}

// The reply to a save carries our text (plus a bumped `updated:` line).
function wikiDraftSaved(msg) {
  const same = (a, b) => a.replace(/^updated:.*$/m, '') === b.replace(/^updated:.*$/m, '');
  return same(String(msg.text || ''), String(wikiDraft || ''));
}

function setWikiConflict(msg) {
  if (msg.path !== wikiSelected) return;
  wikiConflict = { text: msg.text, etag: msg.etag };
  wikiMode = 'edit';
  renderWikiPage();
}

function setWikiSearch(msg) {
  if (String(msg.query || '') !== wikiSearchInput.value.trim()) return;   // a stale answer
  wikiSearch = Array.isArray(msg.hits) ? msg.hits : [];
  renderWikiList();
}

wikiConflictReloadBtn.onclick = () => {
  if (!wikiConflict) return;
  wikiPage = { ...wikiPage, text: wikiConflict.text, etag: wikiConflict.etag };
  wikiDraft = null;
  wikiConflict = null;
  renderWikiPage();
};
wikiConflictOverwriteBtn.onclick = () => {
  if (!wikiConflict) return;
  const etag = wikiConflict.etag;
  wikiConflict = null;
  wikiPage = { ...wikiPage, etag };
  saveWikiPage(etag);
  renderWikiPage();
};

wikiEditorEl.oninput = () => {
  wikiDraft = wikiEditorEl.value;
  const save = wikiHeadEl.querySelector('.wiki-save');
  if (save) save.disabled = !wikiDirty() || !!wikiIngestTimer;
};
wikiEditorEl.onkeydown = (e) => {
  e.stopPropagation();   // the global shortcuts stay out of the editor
  if ((e.metaKey || e.ctrlKey) && e.key === 's') { e.preventDefault(); saveWikiPage(); return; }
  if (e.key === 'Escape') {
    e.preventDefault();
    if (wikiDiscard()) { wikiMode = 'view'; renderWikiPage(); }
    return;
  }
  if (e.key === 'Tab' && typeof wikiEditorEl.selectionStart === 'number') {
    e.preventDefault();
    const { selectionStart: s, selectionEnd: t, value } = wikiEditorEl;
    wikiEditorEl.value = value.slice(0, s) + '  ' + value.slice(t);
    wikiEditorEl.selectionStart = wikiEditorEl.selectionEnd = s + 2;
    wikiEditorEl.oninput();
  }
};

wikiSearchInput.oninput = () => {
  const query = wikiSearchInput.value.trim();
  if (!query) { wikiSearch = null; renderWikiList(); return; }
  sendAction({ action: 'wiki_search', query });
};
wikiSearchInput.onkeydown = (e) => {
  e.stopPropagation();
  if (e.key === 'Escape') { wikiSearchInput.value = ''; wikiSearchInput.oninput(); }
};

function wikiToday() {
  const d = new Date();
  const iso = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  return `meetings/${iso}`;
}
wikiTodayBtn.onclick = () => { const p = wikiToday(); if (wikiKnown(p)) selectWikiPage(p); else createWikiPage(p); };
wikiNewBtn.onclick = () => { const p = prompt('New page (folder/name):', 'notes/'); if (p) createWikiPage(p); };

function wikiIngestDone() {
  if (wikiIngestTimer) { clearTimeout(wikiIngestTimer); wikiIngestTimer = null; }
  wikiIngestBtn.disabled = false;
  wikiIngestLabel.textContent = 'Re-ingest last session';
}

wikiIngestBtn.onclick = () => {
  if (wikiIngestBtn.disabled) return;
  setWikiError('');
  wikiIngestBtn.disabled = true;
  wikiIngestLabel.textContent = 'ingesting…';
  // Re-enabled by the next `wiki` message, or by this timeout if the ingest never answers.
  wikiIngestTimer = setTimeout(wikiIngestDone, WIKI_INGEST_TIMEOUT_MS);
  sendAction({ action: 'wiki_ingest' });
};

function addWikiNote() {
  const text = wikiNoteInput.value.trim();
  if (!wikiSelected || !text) return;
  setWikiError('');
  sendAction({ action: 'wiki_note', path: wikiSelected, text });
  wikiNoteInput.value = '';
}
wikiNoteBtn.onclick = addWikiNote;
wikiNoteInput.onkeydown = (e) => {
  e.stopPropagation();   // Esc/Space are global shortcuts otherwise
  if (e.key === 'Enter') { e.preventDefault(); addWikiNote(); }
};

// The two cross-view drop zones, wired with the same table-driven helper as the memory rows.
// They used to be the top-right tabs; they are the rail items now.
memDropZone(railItemFor('wiki'), 'wiki/rail', false);
memDropZone(railItemFor('memory'), 'memory/rail', false);
renderWikiList();
renderWikiPage();

// Sources view: the file picker is the click-driven twin of the drop zone — same popup,
// same review step, nothing is filed until File is pressed.
sourcesFileBtn.onclick = () => sourcesInputEl.click();
sourcesInputEl.onchange = () => {
  if (sourcesInputEl.files && sourcesInputEl.files.length) intakeFromFiles(sourcesInputEl.files);
  sourcesInputEl.value = '';   // so picking the same file twice fires change again
};

// ---------- intake: drop / paste a source (PRD 20) ----------
// Nothing reaches the vault automatically: a file drag over the workspace shows an overlay, the
// drop (or a ⌘V with an image/text on the clipboard) opens the modal below, and only the File
// button posts to /ingest. Esc/Cancel closes it and sends nothing.
const INTAKE_MAX_IMAGE = 4 * 1024 * 1024;
const INTAKE_MAX_FILE = 2 * 1024 * 1024;       // pdf/eml/msg/ics
const INTAKE_MAX_TEXT = 50 * 1024 * 1024;      // text kinds: the server only shows the model a sample (PRD 31)
const INTAKE_IMAGE_EXT = ['png', 'jpg', 'jpeg'];
const INTAKE_TEXT_EXT = ['md', 'txt', 'docx', 'pptx', 'xlsx', 'vtt', 'epub'];
const INTAKE_EXT = [...INTAKE_TEXT_EXT, 'pdf', 'eml', 'msg', 'ics', ...INTAKE_IMAGE_EXT];
// Names that say nothing about the content: the title field stays empty and the model's title wins.
const INTAKE_GENERIC = /^(note|untitled|pasted text|drop|document|new document)?[\s\d:._-]*$/;
const INTAKE_MIN_TEXT = 20;
const INTAKE_PREVIEW_LINES = 3;
const INTAKE_TOAST_MS = 4000;

const intakeEl = document.getElementById('intake');
const intakeItemsEl = document.getElementById('intake-items');
const intakeTitleField = document.getElementById('intake-title-field');
const intakeTitleEl = document.getElementById('intake-title');
const intakePageEl = document.getElementById('intake-page');
const intakeHintEl = document.getElementById('intake-hint');
const intakeStatusEl = document.getElementById('intake-status');
const intakeFileBtn = document.getElementById('intake-file');
const intakeCancelBtn = document.getElementById('intake-cancel');
const intakeToastEl = document.getElementById('intake-toast');
const dropOverlayEl = document.getElementById('drop-overlay');
const workspaceEl = document.getElementById('workspace');

let intakeKind = null;        // 'file' | 'paste_image' | 'paste_text'
let intakeItems = [];         // [{name, size, file?, text?, ok, reason, url?}]
let intakeBusy = false;
let intakeToastTimer = null;
let dragDepth = 0;

// A file drag carries the `Files` type; memory/wiki row drags never do. Everything in this
// section is gated on it so the two drag systems can't be confused for one another.
function isFileDrag(e) {
  const types = e.dataTransfer && e.dataTransfer.types;
  return !!types && Array.from(types).includes('Files');
}

function intakeExt(name) {
  const m = /\.([A-Za-z0-9]+)$/.exec(String(name || ''));
  return m ? m[1].toLowerCase() : '';
}

// Refuse oversized/unsupported items here rather than at the server: the row stays in the list,
// struck through with the reason, and is left out of the upload. Mirrors sources.check_upload.
function intakeCheck(name, size) {
  const ext = intakeExt(name);
  if (!INTAKE_EXT.includes(ext)) return `unsupported (.${ext || '?'})`;
  const cap = INTAKE_IMAGE_EXT.includes(ext) ? INTAKE_MAX_IMAGE : INTAKE_TEXT_EXT.includes(ext) ? INTAKE_MAX_TEXT : INTAKE_MAX_FILE;
  return tooLarge(size, cap);
}

// '' when it fits — the one wording for the size refusal, on files, images and pasted text.
function tooLarge(size, cap) {
  return size > cap ? `too large (${intakeSize(size)} > ${intakeSize(cap)})` : '';
}

function intakeSize(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} kB`;
  return `${bytes} B`;
}

function intakeStamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function firstLines(text, n) {
  return String(text).split('\n').slice(0, n).join('\n');
}

// The default title of a single dropped file: its name, unless that name is generic.
function intakeTitleFor(name) {
  const stem = String(name || '').replace(/\.[A-Za-z0-9]+$/, '');
  return INTAKE_GENERIC.test(stem.toLowerCase()) ? '' : stem;
}

// The picker lists "auto" plus every page the last `wiki` message named; the open wiki page is
// pre-selected so "file under what I'm looking at" is one drop.
function intakeSyncPages() {
  const want = intakePageEl.value
    || ((currentView === 'wiki' || currentView === 'sources') && wikiSelected) || '';
  const opts = [el('option', null, 'auto')];
  opts[0].value = '';
  for (const page of (wikiPages || [])) {
    const o = el('option', null, page.title ? `${page.title} — ${page.path}` : page.path);
    o.value = page.path;
    opts.push(o);
  }
  intakePageEl.replaceChildren(...opts);
  intakePageEl.value = (wikiPages || []).some(p => p.path === want) ? want : '';
}

function intakeClearItems() {
  for (const it of intakeItems) if (it.url) URL.revokeObjectURL(it.url);
  intakeItems = [];
}

function renderIntakeItems() {
  const rows = intakeItems.map((it) => {
    const row = el('div', 'intake-item' + (it.ok ? '' : ' bad'));
    if (it.url) {
      const img = el('img', 'intake-thumb');
      img.src = it.url;
      img.alt = '';
      row.append(img);
    }
    const body = el('div', 'intake-item-body');
    body.append(el('div', 'intake-item-name', it.name));
    const meta = el('div', 'intake-item-meta', it.ok ? intakeSize(it.size) : `${intakeSize(it.size)} — ${it.reason}`);
    body.append(meta);
    if (it.preview) body.append(el('pre', 'intake-preview', it.preview));
    row.append(body);
    return row;
  });
  // Pasted text is editable in place — it is the payload, not a file on disk.
  if (intakeKind === 'paste_text') {
    const ta = el('textarea', 'intake-textarea');
    ta.value = intakeItems[0] ? intakeItems[0].text : '';
    ta.oninput = () => { if (intakeItems[0]) intakeItems[0].text = ta.value; };
    ta.onkeydown = (e) => { if (e.key !== 'Escape') e.stopPropagation(); };
    intakeItemsEl.replaceChildren(ta);
    return;
  }
  intakeItemsEl.replaceChildren(...rows);
}

function setIntakeStatus(text, cls) {
  intakeStatusEl.className = cls || '';
  intakeStatusEl.textContent = text || '';
}

function openIntake(kind, items, defaultTitle) {
  intakeClearItems();
  intakeKind = kind;
  intakeItems = items;
  intakeBusy = false;
  intakeEl.hidden = false;
  intakeTitleEl.value = defaultTitle || '';
  // With several files the server names each page after its own file; one title would lie.
  const multi = items.length > 1;
  intakeTitleField.hidden = multi;
  intakeHintEl.value = '';
  intakeFileBtn.disabled = false;
  if (!wikiPages) requestWiki();
  intakePageEl.value = '';
  intakeSyncPages();
  renderIntakeItems();
  const bad = items.filter(i => !i.ok).length;
  setIntakeStatus(bad ? `${bad} item${bad > 1 ? 's' : ''} skipped` : '', bad ? 'warn' : '');
  (multi ? intakeHintEl : intakeTitleEl).focus();
}

function closeIntake() {
  if (intakeBusy) return;
  // Focus would otherwise stay on the now-hidden title input and swallow the next paste.
  if (document.activeElement && intakeEl.contains(document.activeElement)) document.activeElement.blur();
  intakeEl.hidden = true;
  intakeClearItems();
  intakeKind = null;
  intakeTitleEl.value = '';
  intakeHintEl.value = '';
  setIntakeStatus('');
}

function intakeToast(text) {
  intakeToastEl.textContent = text;
  intakeToastEl.hidden = false;
  if (intakeToastTimer) clearTimeout(intakeToastTimer);
  intakeToastTimer = setTimeout(() => { intakeToastEl.hidden = true; }, INTAKE_TOAST_MS);
}

// ---- building the popup's items from a drop / a paste ----
function intakeFromFiles(files) {
  const list = Array.from(files);
  if (!list.length) return;
  const items = list.map((file) => {
    const reason = intakeCheck(file.name, file.size);
    const item = { name: file.name, size: file.size, file, ok: !reason, reason };
    if (item.ok && INTAKE_IMAGE_EXT.includes(intakeExt(file.name))) item.url = URL.createObjectURL(file);
    return item;
  });
  openIntake('file', items, list.length === 1 ? intakeTitleFor(list[0].name) : '');
  // Text files get their first lines as a preview once they are read.
  for (const item of items) {
    if (!item.ok || !['md', 'txt'].includes(intakeExt(item.name))) continue;
    item.file.slice(0, 4096).text().then((t) => {
      item.preview = firstLines(t, INTAKE_PREVIEW_LINES);
      if (!intakeEl.hidden) renderIntakeItems();
    }).catch(() => {});
  }
}

function intakeFromImage(blob) {
  const name = 'screenshot.png';
  const reason = tooLarge(blob.size, INTAKE_MAX_IMAGE);
  const item = { name, size: blob.size, file: new File([blob], name, { type: blob.type || 'image/png' }), ok: !reason, reason };
  if (item.ok) item.url = URL.createObjectURL(blob);
  openIntake('paste_image', [item], `screenshot ${intakeStamp()}`);
}

function intakeFromText(text) {
  const size = new Blob([text]).size;
  const reason = tooLarge(size, INTAKE_MAX_TEXT);   // pasted text is sampled server-side (PRD 31)
  // No title from the first line: the summary's title names the page (PRD 31).
  openIntake('paste_text', [{ name: 'pasted text', size, text, ok: !reason, reason }], '');
}

// ---- the one HTTP call in the app ----
async function submitIntake() {
  if (intakeBusy) return;
  const usable = intakeItems.filter(i => i.ok);
  if (!usable.length) { setIntakeStatus('nothing to file', 'err'); return; }
  const fd = new FormData();
  fd.append('confirm', 'true');
  fd.append('kind', intakeKind);
  fd.append('title', intakeTitleEl.value.trim());
  fd.append('hint', intakeHintEl.value.trim());
  fd.append('page', intakePageEl.value || '');
  for (const item of usable) {
    if (item.file) fd.append('file', item.file, item.name);
    else fd.append('text', item.text);
  }
  intakeBusy = true;
  intakeFileBtn.disabled = true;
  setIntakeStatus('uploading…');
  let data = null;
  try {
    const res = await fetch('/ingest', { method: 'POST', body: fd });
    const text = await res.text();
    try { data = JSON.parse(text); } catch (_) { data = null; }
    if (!res.ok) {
      const why = (data && (data.detail || (data.errors || []).join(', '))) || text.slice(0, 120);
      throw new Error(`${res.status} — ${why || 'upload failed'}`);
    }
  } catch (e) {
    intakeBusy = false;
    intakeFileBtn.disabled = false;
    setIntakeStatus(String((e && e.message) || e), 'err');
    return;
  }
  intakeBusy = false;
  intakeFileBtn.disabled = false;
  const filed = (data && data.filed) || [];
  const errors = (data && data.errors) || [];
  if (!filed.length) { setIntakeStatus(errors.join(', ') || 'nothing was filed', 'err'); return; }
  const first = filed[0];
  const verb = first.already ? 'already filed' : 'filed';
  const label = (first.path ? `${verb} as ${first.path}` : `${verb} (${first.raw || 'no page'})`)
    + (filed.length > 1 ? ` (+${filed.length - 1})` : '');
  closeIntake();
  intakeToast(errors.length ? `${label} — ${errors.join(', ')}` : label);
  setView('wiki');
  if (first.path) selectWikiPage(first.path);
}

intakeFileBtn.onclick = submitIntake;
intakeCancelBtn.onclick = closeIntake;
// Clicking the dimmed area outside the card cancels, like Esc.
intakeEl.onclick = (e) => { if (e.target === intakeEl) closeIntake(); };
for (const input of [intakeTitleEl, intakeHintEl]) {
  input.onkeydown = (e) => {
    if (e.key === 'Escape') return;      // let the global handler close the popup
    e.stopPropagation();                 // Space is a global shortcut otherwise
    if (e.key === 'Enter') { e.preventDefault(); submitIntake(); }
  };
}

// ---- workspace drop zone ----
function showDropOverlay(on) {
  if (!on) dragDepth = 0;
  dropOverlayEl.hidden = !on;
}
document.addEventListener('dragenter', (e) => {
  if (!isFileDrag(e)) return;
  dragDepth += 1;
  showDropOverlay(true);
});
document.addEventListener('dragover', (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();                          // without this the browser opens the file
  e.dataTransfer.dropEffect = 'copy';
  if (dropOverlayEl.hidden) showDropOverlay(true);
});
document.addEventListener('dragleave', (e) => {
  if (!isFileDrag(e)) return;
  dragDepth -= 1;
  if (dragDepth <= 0) showDropOverlay(false);
});
document.addEventListener('drop', (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  showDropOverlay(false);
  intakeFromFiles(e.dataTransfer.files);
});

// ---- paste ----
document.addEventListener('paste', (e) => {
  if (!intakeEl.hidden) return;                                  // the popup owns its own inputs
  if (isTyping(e.target) || isTyping(document.activeElement)) return;
  const cd = e.clipboardData;
  if (!cd) return;
  const image = Array.from(cd.items || []).find(i => i.kind === 'file' && i.type.startsWith('image/'));
  if (image) {
    const blob = image.getAsFile();
    if (blob) { e.preventDefault(); intakeFromImage(blob); }
    return;
  }
  const text = (cd.getData('text/plain') || '').trim();
  if (text.length >= INTAKE_MIN_TEXT) { e.preventDefault(); intakeFromText(text); }
});

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
    else if (msg.type === 'backdrop') { applyBackdrop(msg.key); renderBackdropList(); }
    else if (msg.type === 'characters_reloaded') {
      currentAvatarKey = msg.avatar;
      avatarName = msg.name || avatarName;
      status.textContent = 'characters reloaded';
      setTimeout(renderComposer, 1500);
    }
    else if (msg.type === 'listening') { setListening(msg.value); }
    else if (msg.type === 'memory_saving') { memorySaving = !!msg.value; renderComposer(); }
    else if (msg.type === 'hands_free') {
      handsFree = msg.value;
      if (!handsFree) setListening('idle');
      renderComposer();
    }
    else if (msg.type === 'verbosity') {
      currentVerbosity = verbosityLevels.includes(msg.value) ? msg.value : currentVerbosity;
      renderVerbosity();
    }
    else if (msg.type === 'memory') setMemory(msg);
    else if (msg.type === 'wiki') { setWiki(msg); if (palette.open) renderPalette(); }
    else if (msg.type === 'wiki_page') setWikiPage(msg);
    else if (msg.type === 'wiki_conflict') setWikiConflict(msg);
    else if (msg.type === 'wiki_search') setWikiSearch(msg);
    else if (msg.type === 'sources') setSources(msg);
    else if (msg.type === 'setting') {
      if (msg.key === 'calendar_horizon_days') renderCalendarHorizon(msg.value);
      else if (msg.key === 'recall_min_score') renderRecallMinScore(msg.value);
    }
    else if (msg.type === 'memory_error') showMemoryError(msg.path, msg.reason);
    else if (msg.type === 'error') {
      status.textContent = msg.text.toLowerCase();
      if (currentView === 'wiki' || currentView === 'sources') setWikiError(msg.text);   // e.g. a rejected `wiki_note`
      if (currentView === 'sources' && sourcesBusy) { sourcesBusy = false; renderWikiPage(); }
      if (currentView === 'memory') { memErrorEl.textContent = msg.text; memorySaveDone(); }
      renderVerbosity();
    }
  };

  ws.onopen = () => {
    renderComposer();
    // A reconnect (or a first connect while the tab is already on Memory) needs fresh data.
    if (currentView === 'memory') requestMemory();
    else if (currentView === 'wiki' || currentView === 'sources') requestWiki();
    // The badge baseline: what is filed now is "seen"; anything pushed later counts.
    sourcesSeen = null;
    requestSources();
  };
  ws.onclose = (e) => {
    if (shutdownRequested) { showShutdownScreen(); return; }
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
  avatar.onState(value);          // PRD 27 cues: thinking lid/steam, tool sparkle off on idle
  renderComposer();
}

// VAD activity: the composer's hearing state and Wanko's listening cue (PRD 27).
function setListening(value) {
  listening = value;
  avatar.onListening(value);
  renderComposer();
}

// The one place the voice/mute UI is computed from (state, handsFree, muted, listening):
// both mic buttons, the listening dot, the status line, and the mic's own sending gate.
// Everything that changes one of those four variables just calls this.
function renderComposer() {
  const busy = state === 'processing' || state === 'thinking' || state === 'speaking';
  const hearing = handsFree && listening === 'hearing';

  mic.sending = handsFree;
  handsFreeToggle.classList.toggle('active', handsFree);   // the Settings switch, same state
  micBtn.className = 'composer-btn' + (handsFree ? ' on' : '') + (hearing ? ' hearing' : '');
  micBtn.title = handsFree ? 'Voice on — click to stop listening' : 'Voice off — click to listen';
  muteBtn.className = 'composer-btn' + (muted ? ' muted' : state === 'speaking' ? ' speaking' : '');
  muteBtn.title = muted ? 'Replies are muted (space)' : 'Mute replies (space)';
  listeningDot.className = hearing ? 'hearing' : handsFree && mic.started ? 'on' : '';

  if (handsFree && !busy) {
    status.textContent = hearing ? 'hearing you…'
      : mic.started ? 'listening…'
      : 'click to start the microphone';
  } else {
    status.textContent = statusText[state] || '';
  }
  if (muted && !busy) status.textContent += status.textContent ? ' · muted' : 'muted';
  if (memorySaving && !busy) status.textContent += status.textContent ? ' · saving memory…' : 'saving memory…';
  // Re-assert the audio-unlock hint last so other branches above don't clobber it
  // while we're still waiting on a user gesture to resume the AudioContext.
  if (audioBlocked) status.textContent = 'click anywhere to enable audio';
}

const KEYBOARD_GLYPH = [['rect', { x: 2, y: 6, width: 20, height: 12, rx: 2 }],
                        ['path', { d: 'M7 10h.01M11 10h.01M15 10h.01M8 14h8' }]];

// One history line. `typed` adds the keyboard glyph next to the role label (PRD 24).
function addMessage(role, text, { typed = false } = {}) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const r = document.createElement('span'); r.className = 'role';
  r.textContent = role === 'user' ? 'You' : avatarName;
  if (typed) r.append(svgIcon(KEYBOARD_GLYPH, { class: 'typed-glyph', 'aria-label': 'typed' }));
  const t = document.createElement('span'); t.className = 'text'; t.textContent = text;
  div.append(r, t);
  historyEl.appendChild(div);
  historyEl.scrollTop = historyEl.scrollHeight;
}

// ---------- composer (PRD 24) ----------
// Typed turns take the same server path as a final transcript; the server does not echo the
// user line back (we render it here immediately), the reply arrives as `speech`/`transcript`.
function sendText() {
  const text = composerInput.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  unlockAudio();
  addMessage('user', text, { typed: true });
  ws.send(JSON.stringify({ action: 'text_turn', text }));
  composerInput.value = '';
  growComposer();
}

const COMPOSER_MAX_LINES = 4;
function growComposer() {
  const cs = getComputedStyle(composerInput);
  const line = parseFloat(cs.lineHeight) || 20;
  const pad = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom) + 2;
  composerInput.style.height = 'auto';
  composerInput.style.height = Math.min(composerInput.scrollHeight, line * COMPOSER_MAX_LINES + pad) + 'px';
}

composerInput.addEventListener('input', growComposer);
composerInput.addEventListener('keydown', (e) => {
  e.stopPropagation();                 // Space / digits are global shortcuts otherwise
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendText(); }
});
sendBtn.addEventListener('click', sendText);

// Output mute: client-side only, so nothing about the turn/interrupt protocol changes.
function toggleMute() {
  muted = !muted;
  renderComposer();
}
muteBtn.addEventListener('click', toggleMute);

// The mic button is voice on/off — the same `set_hands_free` action the Settings switch sends.
async function setHandsFree(desired) {
  unlockAudio();
  if (desired) {
    const ok = await mic.start();
    if (!ok) return; // permission denied: stays off, status already set by mic.start()
  }
  ws.send(JSON.stringify({ action: 'set_hands_free', value: desired }));
}
micBtn.addEventListener('click', () => setHandsFree(!handsFree));

// ---------- command palette (PRD 30) ----------
// ⌘K opens an overlay over the workspace; every area contributes a source function returning
// its entries `{id, label, keywords, group, detail?, run}`. Sources run on every render, so
// dynamic sets (wiki pages, backdrops, the on/off labels) are read live, nothing is cached.
const PALETTE_MAX = 12;
const paletteEl = document.getElementById('palette');
const paletteInput = document.getElementById('palette-input');
const paletteListEl = document.getElementById('palette-list');
const palette = { open: false, query: '', index: 0, items: [] };
const PALETTE_SOURCES = [];

PALETTE_SOURCES.push(() => VIEWS.map(v => ({
  id: `view:${v.id}`, label: `Go to ${v.label}`, keywords: ['view', v.label, v.id],
  group: 'view', detail: v.key, run: () => setView(v.id),
})));
PALETTE_SOURCES.push(() => [
  { id: 'wiki:today', label: 'Today\'s note', keywords: ['wiki', 'meeting', 'daily'], group: 'wiki',
    detail: wikiToday(), run: () => { setView('wiki'); wikiTodayBtn.onclick(); } },
  { id: 'wiki:new', label: 'New page…', keywords: ['wiki', 'create'], group: 'wiki',
    run: () => { setView('wiki'); wikiNewBtn.onclick(); } },
]);
PALETTE_SOURCES.push(() => [
  { id: 'voice', label: handsFree ? 'Voice: turn off' : 'Voice: turn on', group: 'avatar',
    keywords: ['hands-free', 'continuous', 'listen', 'mic', 'microphone'], run: () => setHandsFree(!handsFree) },
  { id: 'mute', label: muted ? 'Unmute replies' : 'Mute replies', group: 'avatar',
    keywords: ['mute', 'sound', 'audio', 'quiet'], run: toggleMute },
  ...verbosityLevels.map(v => ({
    id: `verbosity:${v}`, label: `Reply length: ${v}`, group: 'avatar', keywords: ['verbosity', 'reply', v],
    run: () => ws.send(JSON.stringify({ action: 'set_verbosity', value: v })),
  })),
  ...backdropOptions.map(b => ({
    id: `backdrop:${b.key}`, label: `Backdrop: ${b.name}`, group: 'avatar', keywords: ['backdrop', 'scene', b.key],
    run: () => setBackdrop(b.key),
  })),
  { id: 'reload-characters', label: 'Reload characters', group: 'avatar', keywords: ['character', 'yaml', 'persona'],
    run: () => ws.send(JSON.stringify({ action: 'reload_characters' })) },
]);
PALETTE_SOURCES.push(() => [
  { id: 'sources:file', label: 'File a source…', group: 'sources', keywords: ['ingest', 'upload', 'drop', 'pdf', 'email'],
    run: () => { setView('sources'); sourcesInputEl.click(); } },
]);
PALETTE_SOURCES.push(() => [
  { id: 'app:ingest', label: 'Re-ingest last session', group: 'app', keywords: ['wiki', 'session', 'memory'],
    run: () => { setView('wiki'); wikiIngestBtn.onclick(); } },
  { id: 'app:shutdown', label: 'Shut down', group: 'app', keywords: ['quit', 'exit', 'stop', 'server'],
    run: () => { if (confirm('Shut down the server? Session memory is saved first.')) requestShutdown(); } },
]);
// Pages last: with an empty query the fixed commands come first, a query ranks them all together.
PALETTE_SOURCES.push(() => (wikiPages || []).map(p => ({
  id: `page:${p.path}`, label: p.title || p.path, keywords: [p.path, ...(p.tags || [])], group: 'wiki',
  detail: p.path, run: () => { setView('wiki'); selectWikiPage(p.path); },
})));

function paletteSearchEntry(q) {
  return {
    id: 'wiki:search', label: `Search wiki for "${q}"`, keywords: [], group: 'wiki',
    run: () => { setView('wiki'); wikiSearchInput.value = q; wikiSearchInput.oninput(); },
  };
}

function isSubsequence(needle, hay) {
  let i = 0;
  for (const ch of hay) if (ch === needle[i]) i++;
  return i === needle.length;
}

// 0 = no match. Label prefix > label word prefix > label substring > keyword substring > subsequence.
function paletteScore(entry, q) {
  const label = entry.label.toLowerCase();
  if (label.startsWith(q)) return 4;
  if (label.split(/[^a-z0-9]+/).some(w => w.startsWith(q))) return 3;
  if (label.includes(q)) return 2;
  if (entry.keywords.some(k => String(k).toLowerCase().includes(q))) return 1;
  if (isSubsequence(q, label)) return 0.5;
  return 0;
}

function paletteItems(query) {
  const q = query.trim().toLowerCase();
  const all = PALETTE_SOURCES.flatMap(src => src());
  if (!q) return all;   // the whole list, fixed commands first, then the pages (scrolls)
  const items = all.map((e, i) => ({ e, i, s: paletteScore(e, q) })).filter(x => x.s > 0)
    .sort((a, b) => b.s - a.s || a.i - b.i).map(x => x.e).slice(0, PALETTE_MAX);
  items.push(paletteSearchEntry(query.trim()));
  return items;
}

function paletteRow(entry, i) {
  const row = el('div', 'palette-row' + (i === palette.index ? ' active' : ''));
  row.setAttribute('role', 'option');
  row.append(el('span', 'palette-label', entry.label));
  row.append(el('span', 'palette-detail', entry.detail || ''));
  row.append(el('span', 'palette-group', entry.group));
  row.onmouseenter = () => { palette.index = i; highlightPalette(); };
  row.onclick = () => runPalette(i);
  return row;
}

// Rebuild the rows from the sources; the highlight alone moves without a rebuild.
function renderPalette() {
  palette.items = paletteItems(palette.query);
  palette.index = Math.min(palette.index, Math.max(0, palette.items.length - 1));
  const rows = palette.items.map(paletteRow);
  paletteListEl.replaceChildren(...(rows.length ? rows : [el('div', 'palette-empty', 'nothing here yet')]));
  highlightPalette();
}

function highlightPalette() {
  Array.from(paletteListEl.children).forEach((row, i) => row.classList.toggle('active', i === palette.index));
  const active = paletteListEl.children[palette.index];
  if (palette.items.length && active && active.scrollIntoView) active.scrollIntoView({ block: 'nearest' });
}

function openPalette() {
  if (!intakeEl.hidden) return;
  palette.open = true;
  palette.query = '';
  palette.index = 0;
  paletteInput.value = '';
  paletteEl.hidden = false;
  if (!wikiPages) requestWiki();   // the page rows fill in when `wiki` answers
  renderPalette();
  paletteInput.focus();
}

function closePalette() {
  palette.open = false;
  paletteEl.hidden = true;
  paletteInput.blur();
}

function togglePalette() {
  if (palette.open) closePalette(); else openPalette();
}

function movePalette(delta) {
  const n = palette.items.length;
  if (!n) return;
  palette.index = (palette.index + delta + n) % n;
  highlightPalette();
}

function runPalette(i) {
  const entry = palette.items[i];
  if (!entry) return;
  closePalette();
  entry.run();
}

paletteInput.oninput = () => { palette.query = paletteInput.value; palette.index = 0; renderPalette(); };
paletteInput.onkeydown = (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') return;   // bubbles to the document handler: toggles closed
  e.stopPropagation();   // the other global shortcuts stay out of the palette
  if (e.key === 'ArrowDown') { e.preventDefault(); movePalette(1); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); movePalette(-1); }
  else if (e.key === 'Enter') { e.preventDefault(); runPalette(palette.index); }
  else if (e.key === 'Escape') { e.preventDefault(); closePalette(); }
};
paletteEl.onclick = (e) => { if (e.target === paletteEl) closePalette(); };

// ---------- keyboard ----------
// The one keydown handler: ⌘K toggles the palette (even while typing), Esc closes the intake
// popup or the palette, Space toggles the output mute and 1–5 switch views. Everything below
// the typing guard is off while the keyboard is being used for typing or a popup is up.
function isTyping(node) {
  return !!(node && node.closest && node.closest('input, textarea, [contenteditable=""], [contenteditable="true"]'));
}
function isInteractive(node) {
  return !!(node && node.closest && node.closest('button, a, select, [role="button"]'));
}
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); togglePalette(); return; }
  if (e.key === 'Escape' && !intakeEl.hidden) { closeIntake(); return; }
  if (e.key === 'Escape' && palette.open) { closePalette(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (isTyping(e.target) || isTyping(document.activeElement)) return;
  if (!intakeEl.hidden || palette.open) return;
  if (e.code === 'Space') {
    if (isInteractive(e.target) || isInteractive(document.activeElement)) return;  // let the button take it
    e.preventDefault(); toggleMute(); return;
  }
  const v = VIEWS.find(x => x.key === e.key);
  if (v) { e.preventDefault(); setView(v.id); }
});
// Any click counts as the user gesture Chrome wants: unlock audio playback, and
// lazily (re)start the mic if hands-free was already on at load (e.g. after a
// reload) and hasn't captured a start gesture yet.
document.addEventListener('click', () => {
  unlockAudio();
  if (handsFree && !mic.started) mic.start();
});

// The Settings "Listening" switch and the composer mic button drive the same state.
handsFreeToggle.onclick = () => setHandsFree(!handsFree);

setView('chat');
avatar.init();
connect();
