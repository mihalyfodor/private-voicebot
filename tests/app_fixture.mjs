// A tiny DOM/WS/audio shim that boots static/app.js in node, so the client half of a PRD
// can be asserted from pytest without a browser (first used for PRD 24).
//
//   node tests/app_fixture.mjs      → prints ONE JSON object of results on stdout
//
// The python side (tests/test_text_input.py) runs that command once per module and asserts
// on the parsed JSON; a non-zero exit (any uncaught error in app.js) fails the test.
//
// ---------------------------------------------------------------------------- what it is
// Deliberately minimal: enough of the browser for app.js to boot, no more.
//   * DOM      — `El`, a stub element with classList / textContent / append / replaceChildren /
//                setAttribute / addEventListener + `dispatch(type, event)` / matches / closest /
//                querySelectorAll (tag, #id and .class selectors only). Every id="…" in
//                static/index.html gets one, so a typo in a getElementById is a null crash
//                here exactly as it would be in the browser.
//   * document — getElementById / createElement(NS) / addEventListener / dispatch / activeElement.
//   * stubs    — AudioContext (counts decodes/plays in `audio`), WebSocket (records every
//                frame in `socket.sent`), fetch (always answers CONFIG_JSON, i.e. /api/config),
//                requestAnimationFrame, getComputedStyle, setTimeout (a no-op: nothing timed runs).
//   * PIXI    — a stub renderer + Live2D model (PRD 27): `model.motion()` only records its
//                arguments in `motionCalls`, the core model is a plain param map, and the
//                'hit' / 'afterMotionUpdate' handlers app.js registers are exposed as hooks.
//
// ------------------------------------------------------------------------ the test hooks
// HOOKS below is appended to app.js inside the same VM context, so it closes over app.js's
// module-level `const`s, and exposes them as `globalThis.__t`:
//   setView(name) / toggleMute() / sendText() / addMessage(role, text, opts)  — app.js functions
//   recv(msg)        push a server message into ws.onmessage
//   sent()           every payload the client has sent (also `socket.sent`, which tests clear)
//   muted            the client-side output mute flag
//   companionParent()/companionClass()   where the companion column currently lives
//   historyLines()   [{ role, text, typed }] rendered in #history
//   input            the composer textarea (set .value, then dispatch a keydown)
//   thumbs()/clickThumb(i)               the backdrop thumbnails over the avatar
//   keydown(code)    a document-level keydown (the global shortcuts)
//   avatar           the avatar object itself (PRD 27; `pet(now)` takes an explicit clock)
//   motions()        [group, index, priority] of every model.motion() call so far
//   cue()            { target, pulse, active } of the current cue blend
//   emotion()        the emotion the expression blend is showing
//   setState(v) / showShutdownScreen()   the two transitions the cues hang off
//
// ------------------------------------------------------------- adding a scenario (next PRD)
// 1. add whatever the scenario needs to HOOKS (keep it a thin accessor, no logic);
// 2. drive it in the "scenarios" section at the bottom and store the observable value in
//    `results` under a snake_case key;
// 3. assert on that key from python. Keep results JSON-serialisable and order-independent
//    where possible — the whole file runs once per pytest module.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const html = fs.readFileSync(path.join(ROOT, 'static/index.html'), 'utf8');
const appSrc = fs.readFileSync(path.join(ROOT, 'static/app.js'), 'utf8');

// ---------------------------------------------------------------- minimal DOM
class El {
  constructor(tag = 'div', id = '') {
    this.tagName = tag.toUpperCase();
    this.id = id;
    this.children = [];
    this.parentNode = null;
    this.style = {};
    this.dataset = {};
    this.attrs = {};
    this.hidden = false;
    this.value = '';
    this.disabled = false;
    this.scrollTop = 0;
    this.scrollHeight = 0;
    this._text = '';
    this._cls = new Set();
    this._on = {};
  }
  get className() { return [...this._cls].join(' '); }
  set className(v) { this._cls = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get classList() {
    const c = this._cls;
    return {
      add: (...n) => n.forEach(x => c.add(x)),
      remove: (...n) => n.forEach(x => c.delete(x)),
      contains: n => c.has(n),
      toggle: (n, force) => (force === undefined ? (c.has(n) ? c.delete(n) : c.add(n))
                                                 : (force ? c.add(n) : c.delete(n))),
    };
  }
  get textContent() { return this._text; }
  set textContent(v) { this._text = String(v); this.children = []; }
  append(...nodes) {
    for (const n of nodes) {
      if (n && typeof n === 'object') {
        if (n.parentNode) n.parentNode.children = n.parentNode.children.filter(c => c !== n);
        n.parentNode = this;
      }
      this.children.push(n);
    }
  }
  appendChild(n) { this.append(n); return n; }
  replaceChildren(...nodes) {
    for (const c of this.children) if (c && c.parentNode === this) c.parentNode = null;
    this.children = [];
    this.append(...nodes);
  }
  remove() {
    if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(c => c !== this);
    this.parentNode = null;
  }
  setAttribute(k, v) { this.attrs[k] = String(v); if (k === 'class') this.className = v; }
  getAttribute(k) { return this.attrs[k]; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(type, fn) { (this._on[type] ||= []).push(fn); }
  removeEventListener(type, fn) { this._on[type] = (this._on[type] || []).filter(f => f !== fn); }
  dispatch(type, event = {}) {
    const e = { type, target: this, preventDefault() {}, stopPropagation() {}, ...event };
    for (const fn of this._on[type] || []) fn(e);
    const prop = this['on' + type];
    if (prop) prop(e);
    return e;
  }
  focus() { doc.activeElement = this; }
  matches(sel) {
    return sel.split(',').map(s => s.trim()).some(s =>
      (s.startsWith('.') && this._cls.has(s.slice(1))) ||
      (s.startsWith('#') && this.id === s.slice(1)) ||
      (/^[a-z]+$/.test(s) && this.tagName === s.toUpperCase()));
  }
  closest(sel) {
    let n = this;
    while (n) { if (n.matches && n.matches(sel)) return n; n = n.parentNode; }
    return null;
  }
  querySelectorAll(sel) {
    const out = [];
    const walk = n => { for (const c of n.children) { if (c && c.matches && c.matches(sel)) out.push(c); if (c && c.children) walk(c); } };
    walk(this);
    return out;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}

const byId = new Map();
for (const m of html.matchAll(/<(\w+)([^>]*\sid="([^"]+)"[^>]*)>/g)) {
  const el = new El(m[1], m[3]);
  el.hidden = /\shidden(\s|$|=)/.test(m[2]);   // e.g. #intake starts hidden, as in the browser
  byId.set(m[3], el);
}

const doc = {
  body: new El('body'),
  createElement: t => new El(t),
  createElementNS: (_ns, t) => new El(t),
  createTextNode: t => ({ textContent: t, nodeType: 3 }),
  getElementById: id => byId.get(id) || null,
  querySelectorAll: () => [],
  querySelector: () => null,
  addEventListener: (t, fn) => ((doc._on[t] ||= []).push(fn)),
  removeEventListener: () => {},
  _on: {},
  dispatch(type, event = {}) {
    const e = { type, target: doc.body, preventDefault() {}, stopPropagation() {}, ...event };
    for (const fn of doc._on[type] || []) fn(e);
    return e;
  },
};
doc.activeElement = doc.body;

// ---------------------------------------------------------------- audio / net stubs
const audio = { decodes: 0, plays: 0 };
class AudioContextStub {
  constructor() { this.state = 'running'; this.destination = {}; this.audioWorklet = { addModule: async () => {} }; }
  createAnalyser() { return { fftSize: 0, connect() {}, getFloatTimeDomainData() {} }; }
  createBufferSource() { audio.plays += 1; return { connect() {}, start() {}, onended: null }; }
  createMediaStreamSource() { return { connect() {} }; }
  decodeAudioData() { audio.decodes += 1; return Promise.resolve({}); }
  resume() { return Promise.resolve(); }
}

// ---------------------------------------------------------------- PIXI / Live2D stub
// Motion groups come from scripts/wanko_groups.py so the fixture and the generated
// Wanko.app.model3.json can never drift apart.
function motionGroups() {
  const py = fs.readFileSync(path.join(ROOT, 'scripts/wanko_groups.py'), 'utf8');
  const body = py.match(/GROUPS = \{([\s\S]*?)\n\}/)[1].replace(/#[^\n]*/g, '');
  return JSON.parse(('{' + body + '}').replace(/,(\s*[}\]])/g, '$1'));
}

const motionCalls = [];          // [group, index, priority] for every model.motion() call
const live2d = { hit: null, frame: null, url: null };
const coreModel = {
  params: {},
  getDrawableCount: () => 0,
  getDrawableDynamicFlagIsVisible: () => false,
  getDrawableOpacity: () => 0,
  getParameterValueById(id) { return this.params[id] || 0; },
  setParameterValueById(id, v) { this.params[id] = v; },
};
const motionManager = {
  definitions: Object.fromEntries(Object.entries(motionGroups())
    .map(([g, clips]) => [g, clips.map(c => ({ File: `motions/${c}.motion3.json` }))])),
  groups: { idle: 'Idle' },
  expressionManager: null,
  _finish: [],
  on(type, fn) { if (type === 'motionFinish') this._finish.push(fn); },
  emit(type) { if (type === 'motionFinish') this._finish.forEach(f => f()); },
};
const modelStub = {
  width: 200, height: 200, x: 0, y: 0,
  scale: { x: 1, y: 1, set(v) { this.x = v; this.y = v; } },
  internalModel: {
    coreModel, motionManager,
    focusController: { x: 0, y: 0 },
    getDrawableVertices: () => [],
    on(type, fn) { if (type === 'afterMotionUpdate') live2d.frame = fn; },
  },
  motion(group, index, priority) { motionCalls.push([group, index, priority]); return Promise.resolve(true); },
  on(type, fn) { if (type === 'hit') live2d.hit = fn; },
  expression: () => Promise.resolve(),
  destroy() {},
};
const PIXIStub = {
  Application: class {
    constructor() {
      this.stage = { addChild() {}, removeChild() {} };
      this.renderer = { on() {} };
      this.screen = { width: 400, height: 400 };
    }
    resize() {}
  },
  live2d: { Live2DModel: { from: async (url) => { live2d.url = url; return modelStub; } } },
};

let socket = null;
class WebSocketStub {
  constructor() { this.readyState = 1; this.sent = []; socket = this; }
  send(data) { this.sent.push(typeof data === 'string' ? JSON.parse(data) : data); }
  close() {}
}
WebSocketStub.OPEN = 1;

const CONFIG_JSON = {
  avatar: 'wanko', name: 'Wanko', backdrops: [
    { key: 'none', name: 'None', file: null, credit: '' },
    { key: 'office', name: 'Home office', file: '/static/backdrops/office.jpg', credit: 'c' },
  ],
  backdrop: 'none', hands_free: false, verbosity: 'normal', vault: '/tmp/v', stale_days: {},
  screenshot_cmd: 'x',
};

const sandbox = {
  document: doc,
  navigator: { mediaDevices: { getUserMedia: async () => ({}) } },
  location: { host: '127.0.0.1:8010' },
  console,
  atob, btoa, Uint8Array, Promise, JSON, Math, Object, Array, Number, String, Set, Map, Error, Date,
  setTimeout: () => 0,
  clearTimeout: () => {},
  requestAnimationFrame: fn => { fn(); return 0; },
  getComputedStyle: () => ({ lineHeight: '20px', paddingTop: '9px', paddingBottom: '9px' }),
  fetch: async () => ({ json: async () => CONFIG_JSON, ok: true }),
  AudioContext: AudioContextStub,
  WebSocket: WebSocketStub,
  FormData: class {},
  URL: { createObjectURL: () => 'blob:x', revokeObjectURL() {} },
  PIXI: PIXIStub,
  performance: { now: () => Date.now() },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

// The hooks the tests drive: appended to app.js so they close over its `const` state.
const HOOKS = `
globalThis.__t = {
  setView, toggleMute, sendText, addMessage,
  recv: (m) => ws.onmessage({ data: JSON.stringify(m) }),
  sent: () => ws.sent,
  get muted() { return muted; },
  companionParent: () => companionEl.parentNode && companionEl.parentNode.id,
  companionClass: () => companionEl.className,
  historyLines: () => historyEl.children.map(c => ({
    role: c.className, text: c.children[1].textContent,
    typed: c.children[0].children.length > 0,
  })),
  input: composerInput,
  thumbs: () => backdropThumbs.children.map(t => ({ cls: t.className, bg: t.style.backgroundImage })),
  clickThumb: (i) => backdropThumbs.children[i].dispatch('click'),
  keydown: (code) => document.dispatch('keydown', { code, key: code === 'Space' ? ' ' : code }),
  // ---- avatar (PRD 27)
  avatar,
  motions: () => globalThis.__motionCalls,
  cue: () => ({ target: { ...avatar.cueTarget }, pulse: [...avatar.cuePulse],
                active: [...avatar.activeCues] }),
  emotion: () => avatar.currentEmotion,
  motionGroups: () => Object.keys(avatar.model.internalModel.motionManager.definitions),
  setState,
  showShutdownScreen,
};
`;

sandbox.__motionCalls = motionCalls;
vm.createContext(sandbox);
vm.runInContext(appSrc + HOOKS, sandbox);
const t = sandbox.__t;
await new Promise(r => setImmediate(r));   // let avatar.init()'s awaits settle

// ---------------------------------------------------------------- scenarios (PRD 24)
const results = {};

// ---- scenario 3: setView re-parents the one companion element
results.parent_chat = t.companionParent();
results.class_chat = t.companionClass();
t.setView('memory');
results.parent_memory = t.companionParent();
results.class_memory = t.companionClass();
t.setView('chat');
results.parent_back = t.companionParent();
results.class_back = t.companionClass();

// ---- scenario 1: typed turn renders locally and sends text_turn
t.input.value = '  hello there  ';
t.input.dispatch('keydown', { key: 'Enter', shiftKey: false });
results.typed_sent = socket.sent.filter(m => m.action === 'text_turn');
results.after_send_value = t.input.value;

// ---- scenario 2: muted client shows the transcript but plays nothing, and still
//      answers playback_done so the server's turn state advances
const speech = { type: 'speech', emotion: 'happy', turn: 1, wav: btoa('RIFFfake') };
t.keydown('Space');                    // no input focused → mute
results.muted_after_space = t.muted;
t.recv({ type: 'transcript', role: 'assistant', text: 'muted reply' });
t.recv(speech);
results.decodes_while_muted = audio.decodes;
socket.sent.length = 0;
t.recv({ type: 'speech_end', turn: 1 });
results.playback_done_while_muted = socket.sent.filter(m => m.action === 'playback_done');

t.keydown('Space');                    // unmute
results.muted_after_second_space = t.muted;
t.recv({ type: 'speech', emotion: 'happy', turn: 2, wav: btoa('RIFFfake') });
await new Promise(r => setImmediate(r));
results.decodes_after_unmute = audio.decodes;

// typing must not steal the space bar
t.input.value = 'x';
doc.activeElement = t.input;
t.keydown('Space');
results.muted_after_space_while_typing = t.muted;
doc.activeElement = doc.body;

results.history = t.historyLines();

// ---------------------------------------------------------------- scenarios (PRD 27)
// The greeting already fired above (the first `speech` of the "session"), so record that
// first, then clear the motion log for the interactivity scenarios.
results.greet_calls = t.motions().filter(m => m[0] === 'Greet').length;
results.model_url = live2d.url;
results.motion_groups = t.motionGroups().sort();
motionCalls.length = 0;

// ---- click → Pet, a second click within 4 s → PetSmall, after a pause → Pet again
live2d.hit(['Body']);
live2d.hit(['Body']);
live2d.hit(['Ear']);                                   // not the body hit area: ignored
results.tap_groups = t.motions().map(m => m[0]);
results.tap_after_pause = t.avatar.pet(Date.now() + 5000);
results.pet_emotion = t.emotion();

// ---- idle picks: never the same index twice in a row, all four clips reachable
motionCalls.length = 0;
const idlePicks = [];
for (let i = 0; i < 40; i++) idlePicks.push(t.avatar.playGroup('Idle'));
results.idle_repeats = idlePicks.filter((v, i) => i > 0 && v === idlePicks[i - 1]).length;
results.idle_indices = [...new Set(idlePicks)].sort();
results.idle_priority = t.motions()[0][2];

// ---- state thinking sets the lid/steam targets; the first speech clears them
t.setState('thinking');
results.thinking_cue = t.cue();
results.thinking_motion = t.motions().filter(m => m[0] === 'Thinking').length;
t.recv({ type: 'speech', emotion: 'happy', turn: 3, wav: btoa('RIFFfake'), filler: true });
results.filler_cue = t.cue();
t.recv({ type: 'speech', emotion: 'happy', turn: 3, wav: btoa('RIFFfake') });
results.speaking_cue = t.cue();
results.greet_calls_after = t.motions().filter(m => m[0] === 'Greet').length;   // still once

// ---- listening: the Listening clip once, ears up while hearing
motionCalls.length = 0;
t.recv({ type: 'listening', value: 'hearing' });
t.recv({ type: 'listening', value: 'hearing' });
results.listening_cue = t.cue();
results.listening_motions = t.motions().map(m => m[0]);
t.recv({ type: 'listening', value: 'idle' });
results.listening_cue_after = t.cue();

// ---- an emotion tag with no expression is ignored, a known one is applied
t.avatar.setEmotion('neutral');
t.avatar.setEmotion('proud');
results.emotion_proud = { name: t.emotion(), target: { ...t.avatar.paramTarget } };
t.avatar.setEmotion('bewildered');
results.emotion_unknown = t.emotion();

// ---- shutdown: forced idle_04 + the lid closing
motionCalls.length = 0;
t.showShutdownScreen();
results.shutdown_motion = t.motions();
results.shutdown_cue = t.cue();

// ---- scenario 4: backdrop thumbs send the same payload as the settings chips
socket.sent.length = 0;
results.thumbs = t.thumbs();
t.clickThumb(1);
results.thumb_payload = socket.sent;

console.log(JSON.stringify(results, null, 1));
