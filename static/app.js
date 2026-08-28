// ---------- config ----------
const CONFIG = {
  modelUrl: '/static/models/haru/haru_greeter_t03.model3.json',
  // emotion → Haru expression name (see expressions in the model3.json). null = neutral/reset.
  expressions: { neutral: null, happy: 'f00', surprised: 'f02', thinking: 'f04', apologetic: 'f05' },
  expressionWeight: 0.7,
  expressionHoldMs: 500,
  mouthGain: 6.0,      // RMS → mouth openness multiplier
  mouthSmooth: 0.35,   // 0..1, higher = snappier
};

// ---------- DOM ----------
const btn = document.getElementById('btn');
const status = document.getElementById('status');
const transcript = document.getElementById('transcript');
const avatarNote = document.getElementById('avatar-note');
document.getElementById('shutdown').onclick = () => {
  ws.send(JSON.stringify({ action: 'shutdown' }));
  window.close();
};

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

// ---------- audio ----------
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
const analyser = audioCtx.createAnalyser();
analyser.fftSize = 1024;
analyser.connect(audioCtx.destination);
const timeData = new Float32Array(analyser.fftSize);

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

async function enqueueSpeech(msg) {
  const bytes = Uint8Array.from(atob(msg.wav), c => c.charCodeAt(0));
  const buffer = await audioCtx.decodeAudioData(bytes.buffer);
  queue.push({ emotion: msg.emotion, buffer });
  playNext();
}

function playNext() {
  if (playing) return;
  const item = queue.shift();
  if (!item) {
    if (replyEnded) finishReply();
    return;
  }
  playing = true;
  if (audioCtx.state === 'suspended') audioCtx.resume();
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
  ws.send(JSON.stringify({ action: 'playback_done' }));
}

// ---------- avatar ----------
const avatar = {
  model: null,
  currentExpr: undefined,

  async init() {
    if (!window.PIXI || !PIXI.live2d) { avatarNote.hidden = false; return; }
    const canvas = document.getElementById('stage');
    const app = new PIXI.Application({
      view: canvas, backgroundAlpha: 0, resizeTo: canvas.parentElement, antialias: true,
    });
    let model;
    try {
      model = await PIXI.live2d.Live2DModel.from(CONFIG.modelUrl);
    } catch (e) {
      console.warn('Live2D model failed to load', e);
      avatarNote.hidden = false;
      return;
    }
    this.model = model;
    app.stage.addChild(model);
    this.fit(app);
    app.renderer.on('resize', () => this.fit(app));

    model.internalModel.on('afterMotionUpdate', () => {
      const target = Math.min(1, rms() * CONFIG.mouthGain);
      mouth += (target - mouth) * CONFIG.mouthSmooth;
      model.internalModel.coreModel.setParameterValueById('ParamMouthOpenY', playing ? mouth : 0);
    });
    model.motion('Idle');
  },

  fit(app) {
    const m = this.model;
    const scale = Math.min(app.screen.width / m.width, app.screen.height / m.height) * 1.15;
    m.scale.set(scale);
    m.x = (app.screen.width - m.width) / 2;
    m.y = app.screen.height * 0.02;
  },

  setEmotion(emotion) {
    if (!this.model) return;
    const name = CONFIG.expressions[emotion] ?? null;
    if (name === this.currentExpr) return;
    this.currentExpr = name;
    const em = this.model.internalModel.motionManager.expressionManager;
    if (!em) return;
    if (name === null) { em.resetExpression(); return; }
    this.model.expression(name).then(() => {
      const expr = em.expressions?.[em.expressionIndex ?? -1];
      if (expr && typeof expr.setWeight === 'function') expr.setWeight(CONFIG.expressionWeight);
    }).catch(e => console.warn('expression failed', e));
  },
};

// ---------- websocket / ui ----------
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'state') setState(msg.value);
    else if (msg.type === 'transcript') addMessage(msg.role, msg.text);
    else if (msg.type === 'speech') enqueueSpeech(msg);
    else if (msg.type === 'speech_end') { replyEnded = true; playNext(); }
  };

  ws.onopen = () => status.textContent = statusText.idle;
  ws.onclose = () => {
    status.textContent = 'disconnected — retrying...';
    setTimeout(connect, 2000);
  };
}

function setState(value) {
  state = value;
  btn.className = state === 'idle' ? '' : state;
  btn.textContent = btnText[state] || '...';
  status.textContent = statusText[state] || '';
  btn.disabled = !(state === 'idle' || state === 'recording');
}

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  const r = document.createElement('span'); r.className = 'role'; r.textContent = role === 'user' ? 'You' : 'Haru';
  const t = document.createElement('span'); t.className = 'text'; t.textContent = text;
  div.append(r, t);
  transcript.appendChild(div);
  transcript.scrollTop = transcript.scrollHeight;
}

function toggle() {
  if (audioCtx.state === 'suspended') audioCtx.resume();
  if (state === 'idle' || state === 'recording') ws.send(JSON.stringify({ action: 'toggle' }));
}

btn.addEventListener('click', toggle);
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && e.target === document.body) { e.preventDefault(); toggle(); }
});
document.addEventListener('click', () => { if (audioCtx.state === 'suspended') audioCtx.resume(); }, { once: true });

avatar.init();
connect();
