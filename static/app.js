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
  fitWidth: 0.8,         // share of the avatar column the drawn model may use…
  fitHeight: 0.7,        // …and of its height; the smaller of the two wins
};
let profile = PROFILES.haru;
let avatarName = 'Haru';

// ---------- DOM ----------
const btn = document.getElementById('btn');
const status = document.getElementById('status');
const transcript = document.getElementById('transcript');
const avatarNote = document.getElementById('avatar-note');
let shutdownRequested = false;
function showShutdownScreen() {
  document.getElementById('avatar').classList.add('gone');
  document.getElementById('workspace').classList.add('gone');
  document.getElementById('menu-btn').classList.add('gone');
  document.getElementById('shutdown-screen').hidden = false;
}
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
      if (this.paramWeight > 0.01) {
        for (const [id, v] of Object.entries(this.paramTarget)) {
          const cur = core.getParameterValueById(id);
          core.setParameterValueById(id, cur + (v - cur) * this.paramWeight);
        }
      }
    });
    model.motion(profile.idle);
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
    // art to CONFIG.fitWidth/fitHeight of the column, whichever limits, and centre it there.
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
    m.y = app.screen.height / 2 - (art.y + art.h / 2) * scale
        - art.h * scale * profile.avatarTopCrop;
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

// ---------- workspace views ----------
const viewEls = {
  chat: document.getElementById('view-chat'),
  memory: document.getElementById('view-memory'),
  wiki: document.getElementById('view-wiki'),
};
const viewTabs = Array.from(document.querySelectorAll('.view-tab'));
let currentView = 'chat';

function setView(name) {
  if (!viewEls[name]) return;
  currentView = name;
  for (const [key, el] of Object.entries(viewEls)) el.hidden = key !== name;
  viewTabs.forEach(t => t.classList.toggle('active', t.dataset.view === name));
  if (name === 'memory' && !memoryProfile) requestMemory();
  if (name === 'wiki' && !wikiPages) requestWiki();
}
viewTabs.forEach(t => { t.onclick = () => setView(t.dataset.view); });
document.getElementById('open-memory').onclick = () => { setView('memory'); openDrawer(false); };
document.getElementById('open-wiki').onclick = () => { setView('wiki'); openDrawer(false); };

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
  // PRD 19: file the entry into its wiki page (and mark it stale); the drop zone is the Wiki tab.
  if (spec.archivable) moves.push({ label: 'archive', to: 'wiki/tab', run: () => archiveEntry(spec.section, index) });
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

function scalarRow(section, key, value, superseded) {
  const path = `${section}.${key}`;
  const { row, err } = memRow(path);
  row.prepend(el('span', 'mem-key', key), editable(value, '(empty)', v => sendMemoryOp('UPDATE', path, v)), deleteButton(path));
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

memModelToggle.onclick = () => { memoryModelView = !memoryModelView; renderMemory(); };
document.getElementById('memory-reset').onclick = () => {
  if (!confirm('Forget everything the assistant remembers about you? This cannot be undone.')) return;
  sendAction({ action: 'memory_reset' });
};

// ---------- wiki view ----------
// Browse-only mirror of ~/llm-wiki: the server sends the page index (`wiki`) and one page at a
// time (`wiki_page`); the only writes from here are an appended note, an on-demand ingest and
// the two drag bridges to the memory view (archive / pull).
const WIKI_FOLDERS = ['people', 'projects', 'meetings', 'sources'];
const WIKI_LINK_RE = /\[\[([^[\]]+)\]\]/g;
const WIKI_INGEST_TIMEOUT_MS = 60000;

const wikiListEl = document.getElementById('wiki-list');
const wikiHeadEl = document.getElementById('wiki-page-head');
const wikiTextEl = document.getElementById('wiki-text');
const wikiNoteEl = document.getElementById('wiki-note');
const wikiLintEl = document.getElementById('wiki-lint');
const wikiVaultEl = document.getElementById('wiki-vault');
const wikiErrorEl = document.getElementById('wiki-error');
const wikiIngestBtn = document.getElementById('wiki-ingest');
const wikiNoteInput = document.getElementById('wiki-note-input');
const wikiNoteBtn = document.getElementById('wiki-note-add');
const wikiTabBtn = viewTabs.find(t => t.dataset.view === 'wiki');
const memTabBtn = viewTabs.find(t => t.dataset.view === 'memory');

let wikiPages = null;      // last `wiki` page list (null = never loaded)
let wikiLint = [];
let wikiVault = '';
let wikiSelected = null;   // path of the open page, kept across refreshes
let wikiPage = null;       // last `wiki_page` for wikiSelected
let wikiIngestTimer = null;

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
  moves.push({ label: 'open memory', to: 'memory/tab', run: () => setView('memory') });
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

// Paths are suffix-less everywhere (`people/anna`) — in the page list, in wikilinks and in
// every action sent back; the server is what normalises them onto files.
function selectWikiPage(path) {
  path = String(path || '').trim();
  if (!path) return;
  wikiSelected = path;
  wikiPage = null;
  setWikiError('');
  sendAction({ action: 'read_wiki_page', path });
  renderWikiList();
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

function renderWikiList() {
  const pages = wikiPages || [];
  if (!pages.length) {
    wikiListEl.replaceChildren(el('div', 'mem-empty', wikiPages ? 'no pages yet' : 'loading…'));
    return;
  }
  const nodes = [];
  for (const [folder, group] of wikiGroups(pages)) {
    nodes.push(el('div', 'wiki-group-title', folder));
    group.sort((a, b) => String(a.title || a.path).localeCompare(String(b.title || b.path)));
    nodes.push(...group.map(wikiRow));
  }
  wikiListEl.replaceChildren(...nodes);
}

// The page text, with every [[link]] turned into a span that opens that page. Built as DOM
// nodes from a split on the wikilink pattern — never innerHTML.
function wikiTextNodes(text) {
  const known = new Set((wikiPages || []).map(p => p.path));
  return String(text).split(WIKI_LINK_RE).map((part, i) => {
    if (i % 2 === 0) return document.createTextNode(part);
    const target = part.trim();
    const link = el('span', 'wiki-link', `[[${part}]]`);
    if (!known.has(target)) { link.classList.add('missing'); link.title = 'no such page (yet)'; }
    link.onclick = () => selectWikiPage(target);
    return link;
  });
}

function renderWikiPage() {
  const meta = (wikiPages || []).find(p => p.path === wikiSelected);
  const head = [];
  if (!wikiSelected) {
    wikiHeadEl.replaceChildren(el('div', 'wiki-muted', 'pick a page'));
    wikiTextEl.replaceChildren();
    wikiTextEl.hidden = true;
    wikiNoteEl.hidden = true;
    return;
  }
  wikiTextEl.hidden = false;
  wikiNoteEl.hidden = false;
  head.push(el('div', 'wiki-title', (meta && meta.title) || wikiSelected));
  const sub = el('div', 'wiki-sub');
  sub.append(el('span', 'wiki-path', wikiSelected));
  if (meta && meta.updated) sub.append(el('span', 'wiki-muted', `updated ${meta.updated}`));
  if (wikiVault) {
    const vaultName = wikiVault.replace(/\/+$/, '').split('/').pop();
    const a = el('a', 'wiki-obsidian', 'Open in Obsidian');
    a.href = `obsidian://open?vault=${encodeURIComponent(vaultName)}&file=${encodeURIComponent(wikiSelected)}`;
    sub.append(a);
  }
  // Fallback for the drag bridge: the same move, as a button.
  const pull = meta && wikiMoves({ path: wikiSelected, type: meta.type }).find(m => m.to.endsWith('/section'));
  if (pull) {
    const b = el('button', 'wiki-pull', `→ ${pull.label}`);
    b.onclick = pull.run;
    sub.append(b);
  }
  head.push(sub);
  wikiHeadEl.replaceChildren(...head);
  wikiTextEl.replaceChildren(...(wikiPage ? wikiTextNodes(wikiPage.text || '') : [document.createTextNode('loading…')]));
}

function renderWikiFoot() {
  wikiVaultEl.textContent = wikiVault || '';
  wikiLintEl.hidden = !wikiLint.length;
  wikiLintEl.replaceChildren(...wikiLint.map(line => el('div', 'wiki-lint-line', line)));
}

function setWiki(msg) {
  wikiPages = Array.isArray(msg.pages) ? msg.pages : [];
  wikiLint = Array.isArray(msg.lint) ? msg.lint : [];
  wikiVault = msg.vault || wikiVault;
  wikiIngestDone();
  // Keep the open page open across refreshes, and re-read it (an ingest may have changed it).
  if (wikiSelected && wikiPages.some(p => p.path === wikiSelected)) selectWikiPage(wikiSelected);
  else {
    if (wikiSelected) { wikiSelected = null; wikiPage = null; }
    renderWikiList();
    renderWikiPage();
  }
  renderWikiFoot();
  if (!intakeEl.hidden) intakeSyncPages();   // the popup's picker is fed by this list
}

function setWikiPage(msg) {
  if (msg.path !== wikiSelected) return;   // a stale answer for a page we navigated away from
  wikiPage = msg;
  renderWikiPage();
}

function wikiIngestDone() {
  if (wikiIngestTimer) { clearTimeout(wikiIngestTimer); wikiIngestTimer = null; }
  wikiIngestBtn.disabled = false;
  wikiIngestBtn.textContent = 'Re-ingest last session';
}

wikiIngestBtn.onclick = () => {
  if (wikiIngestBtn.disabled) return;
  setWikiError('');
  wikiIngestBtn.disabled = true;
  wikiIngestBtn.textContent = 'ingesting…';
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
memDropZone(wikiTabBtn, 'wiki/tab', false);
memDropZone(memTabBtn, 'memory/tab', false);
renderWikiList();
renderWikiPage();

// ---------- intake: drop / paste a source (PRD 20) ----------
// Nothing reaches the vault automatically: a file drag over the workspace shows an overlay, the
// drop (or a ⌘V with an image/text on the clipboard) opens the modal below, and only the File
// button posts to /ingest. Esc/Cancel closes it and sends nothing.
const INTAKE_MAX_IMAGE = 4 * 1024 * 1024;
const INTAKE_MAX_FILE = 2 * 1024 * 1024;
const INTAKE_IMAGE_EXT = ['png', 'jpg', 'jpeg'];
const INTAKE_EXT = ['md', 'txt', 'pdf', 'eml', 'msg', ...INTAKE_IMAGE_EXT];
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
  return tooLarge(size, INTAKE_IMAGE_EXT.includes(ext) ? INTAKE_MAX_IMAGE : INTAKE_MAX_FILE);
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

// The picker lists "auto" plus every page the last `wiki` message named; the open wiki page is
// pre-selected so "file under what I'm looking at" is one drop.
function intakeSyncPages() {
  const want = intakePageEl.value || (currentView === 'wiki' && wikiSelected) || '';
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
  openIntake('file', items, list.length === 1 ? list[0].name : '');
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
  const reason = tooLarge(size, INTAKE_MAX_FILE);
  openIntake('paste_text', [{ name: 'pasted text', size, text, ok: !reason, reason }],
    firstLines(text, 1).slice(0, 80).trim() || `note ${intakeStamp()}`);
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
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !intakeEl.hidden) closeIntake(); });

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
function inTextEntry(node) {
  const t = node && node.closest ? node.closest('input, textarea, [contenteditable=""], [contenteditable="true"]') : null;
  return !!t;
}
document.addEventListener('paste', (e) => {
  if (!intakeEl.hidden) return;                                  // the popup owns its own inputs
  if (inTextEntry(e.target) || inTextEntry(document.activeElement)) return;
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
    else if (msg.type === 'memory') setMemory(msg);
    else if (msg.type === 'wiki') setWiki(msg);
    else if (msg.type === 'wiki_page') setWikiPage(msg);
    else if (msg.type === 'memory_error') showMemoryError(msg.path, msg.reason);
    else if (msg.type === 'error') {
      status.textContent = msg.text.toLowerCase();
      if (currentView === 'wiki') setWikiError(msg.text);   // e.g. a rejected `wiki_note`
      renderVerbosity();
    }
  };

  ws.onopen = () => {
    render();
    // A reconnect (or a first connect while the tab is already on Memory) needs fresh data.
    if (currentView === 'memory') requestMemory();
    else if (currentView === 'wiki') requestWiki();
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
