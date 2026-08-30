// A tiny DOM/WS/audio shim that boots static/app.js in node, so the client half of a PRD
// can be asserted from pytest without a browser (first used for PRD 24).
//
//   node tests/app_fixture.mjs      → prints ONE JSON object of results on stdout
//
// The python side (tests/session/test_text_input.py) runs that command once per module and asserts
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
//   palette / paletteInput / paletteList / view()   the ⌘K palette state, its input, its rows (PRD 30)
//   cmdK()           a document-level ⌘K keydown
//   cards()          the `.msg.card` elements in #history (PRD 43)
//   submitIntake() / intakeStatus / intakeCancel   the intake batch loop, its status line and
//                    its Cancel/Close button (PRD 36; `fetch` is re-scripted per scenario)
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
    this.nodeType = 1;
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
  prepend(...nodes) {
    for (const n of nodes) if (n && typeof n === 'object' && n.parentNode) n.parentNode.children = n.parentNode.children.filter(c => c !== n);
    for (const n of nodes) if (n && typeof n === 'object') n.parentNode = this;
    this.children = [...nodes, ...this.children];
  }
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
  blur() { if (doc.activeElement === this) doc.activeElement = doc.body; }
  contains(node) { for (let n = node; n; n = n.parentNode) if (n === this) return true; return false; }
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

const clipboardLog = [];         // every navigator.clipboard.writeText payload (PRD 41)
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
  navigator: {
    mediaDevices: { getUserMedia: async () => ({}) },
    // PRD 41: the copy button's payload lands here
    clipboard: { writeText: (t) => { clipboardLog.push(t); return Promise.resolve(); } },
    // PRD 35: a stubbed position, handed over synchronously on request
    geolocation: { getCurrentPosition: (ok) => ok({ coords: { latitude: 46.7712, longitude: 23.6236 } }) },
  },
  Intl: { DateTimeFormat: () => ({ resolvedOptions: () => ({ timeZone: 'Europe/Bucharest' }) }) },
  location: { host: '127.0.0.1:8010' },
  console,
  atob, btoa, Uint8Array, Promise, JSON, Math, Object, Array, Number, String, Set, Map, Error, Date,
  setTimeout: () => 0,
  clearTimeout: () => {},
  setInterval: () => 0,        // the intake batch's elapsed-time tick (PRD 36): nothing timed runs
  clearInterval: () => {},
  requestAnimationFrame: fn => { fn(); return 0; },
  getComputedStyle: () => ({ lineHeight: '20px', paddingTop: '9px', paddingBottom: '9px' }),
  fetch: async () => ({ json: async () => CONFIG_JSON, ok: true }),
  AudioContext: AudioContextStub,
  WebSocket: WebSocketStub,
  FormData: class { constructor() { this.fields = []; } append(...a) { this.fields.push(a); } },
  URL: { createObjectURL: () => 'blob:x', revokeObjectURL() {} },
  // PRD 31 intake: enough of Blob/File for size checks and the (unread) text preview.
  Blob: class { constructor(parts) { this.size = (parts || []).join('').length; } },
  File: class {
    constructor(parts, name, opts) { this.size = (parts || []).join('').length; this.name = name; this.type = (opts || {}).type || ''; }
    slice() { return { text: async () => '' }; }
  },
  PIXI: PIXIStub,
  performance: { now: () => Date.now() },
  close() {},   // window.close(), called by the shutdown path
  // PRD 28 dialogs: confirm always agrees, prompt answers with `dialogs.answer` and both log.
  confirm: (msg) => { dialogs.log.push(['confirm', msg]); return true; },
  prompt: (msg, def) => { dialogs.log.push(['prompt', msg]); return dialogs.answer ?? def; },
};
const dialogs = { log: [], answer: null };
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
  // ---- wiki editor (PRD 28)
  wikiList: wikiListEl, wikiHead: wikiHeadEl, wikiText: wikiTextEl, wikiEditor: wikiEditorEl,
  wikiConflict: wikiConflictEl, wikiBacklinks: wikiBacklinksEl, wikiSearch: wikiSearchInput,
  wikiToday: wikiTodayBtn, wikiNew: wikiNewBtn,
  selectWikiPage,
  // ---- command palette (PRD 30)
  palette, paletteInput, paletteList: paletteListEl,
  view: () => currentView,
  cmdK: () => document.dispatch('keydown', { key: 'k', metaKey: true }),
  // ---- office formats (PRD 31)
  intakeFromText, intakeFromFiles, closeIntake,
  intakeTitle: intakeTitleEl, intake: intakeEl, intakeItems: intakeItemsEl,
  // ---- intake progress (PRD 36)
  submitIntake, intakeStatus: intakeStatusEl, intakeCancel: intakeCancelBtn,
  calendarHorizon: calendarHorizonEl,
  // ---- source recall (PRD 32)
  recallMinScore: recallMinScoreEl,
  // ---- memory provenance (PRD 33)
  memSections: memSectionsEl,
  // ---- scoped errors
  status, memError: memErrorEl, wikiError: wikiErrorEl,
  // ---- service health (PRD 41)
  serviceBanner, serviceBannerText, serviceCopyBtn, settingsServices: settingsServicesEl,
  adminPassword: adminPasswordEl, adminPasswordState: adminPasswordStateEl,
  closeSocket: (code) => { shutdownRequested = false; ws.onclose({ code }); },   // the palette's Shut down above set it
  // ---- context meter (PRD 39)
  chatDot: () => railItems.find(b => b.dataset.view === 'chat').querySelector('.rail-dot').className,
  // ---- written cards (PRD 43)
  cards: () => historyEl.children.filter(c => c.classList.contains('card')),
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
results.thumb_payload = [...socket.sent];

// ---------------------------------------------------------------- scenarios (PRD 28)
// The list rows as the user sees them: group titles and row titles, in order.
const listRows = () => t.wikiList.children.map(c =>
  c.className.includes('wiki-group-title') ? `# ${c.textContent}` : c.querySelector('.wiki-row-title').textContent);
const linkSpans = () => t.wikiText.querySelectorAll('.wiki-link').map(s =>
  ({ text: s.textContent, cls: s.className, target: s.dataset.target }));
const headButton = (cls) => t.wikiHead.querySelector('.' + cls);
const actions = () => socket.sent.map(m => m.action);

t.setView('wiki');
t.recv({ type: 'wiki', vault: '/tmp/v', lint: [], pages: [
  { path: 'people/anna', title: 'Anna', type: 'person', updated: '2026-08-29', tags: ['team'] },
  { path: 'projects/atlas', title: 'Atlas', type: 'project', updated: '2026-08-29', tags: ['team', 'infra'] },
  { path: 'notes/todo', title: 'todo', type: 'note', updated: '2026-08-29', tags: [] },
] });
// ---- scenario 7: folder groups + the virtual Tags group; clicking a tag filters
results.wiki_list = listRows();
t.wikiList.children.find(c => c.children[0] && c.children[0].textContent === '#infra').dispatch('click');
results.wiki_list_infra = listRows();
t.wikiList.children.find(c => c.children[0] && c.children[0].textContent === '#infra').dispatch('click');   // toggle off
results.wiki_list_after = listRows();

// ---- scenario 8: alias link renders its alias and navigates to the page; a red link creates
socket.sent.length = 0;
t.selectWikiPage('people/anna');
const anna = '---\ntype: person\nname: Anna\ntags: [team]\nsource: s\nupdated: 2026-08-29\nseen: 2026-08-29\n---\n\n# Anna\n\nWorks on [[projects/atlas|Atlas project]] and [[projects/atlas#Summary]]; see [[people/missing]] and ![[notes/todo]].\n';
t.recv({ type: 'wiki_page', path: 'people/anna', text: anna, links: [], etag: 'e1', backlinks: ['projects/atlas'] });
results.wiki_links = linkSpans();
results.wiki_backlinks = t.wikiBacklinks.children.slice(1).map(c => c.textContent);
results.wiki_fm_hidden = doc.getElementById('wiki-fm').hidden;
// PRD 28 polish: only the at-a-glance keys get chips; the rest sit behind a details toggle
const fmEl = doc.getElementById('wiki-fm');
results.wiki_fm_chips = fmEl.children.filter(c => c.className === 'wiki-fm-item').map(c => c.textContent.split(':')[0]);
results.wiki_fm_toggle = fmEl.children.filter(c => c.className === 'wiki-fm-toggle').map(c => c.textContent);
fmEl.children.find(c => c.className === 'wiki-fm-toggle').dispatch('click');
results.wiki_fm_chips_open = doc.getElementById('wiki-fm').children.filter(c => c.className.includes('wiki-fm-item')).map(c => c.textContent.split(':')[0]);
t.wikiText.querySelectorAll('.wiki-link')[0].dispatch('click');          // the alias link
results.wiki_alias_nav = socket.sent.filter(m => m.action === 'read_wiki_page').map(m => m.path);
t.selectWikiPage('people/anna');
t.recv({ type: 'wiki_page', path: 'people/anna', text: anna, links: [], etag: 'e1', backlinks: ['projects/atlas'] });
socket.sent.length = 0;
t.wikiText.querySelectorAll('.wiki-link').find(s => s.className.includes('missing')).dispatch('click');  // the red link
results.wiki_create_sent = [...socket.sent];
results.wiki_create_confirm = dialogs.log.filter(d => d[0] === 'confirm').length;

// ---- editor: Edit → type → ⌘S sends wiki_save with the etag; a conflict shows the dialog,
//      overwrite re-sends with the server's etag; Esc discards
t.selectWikiPage('people/anna');
t.recv({ type: 'wiki_page', path: 'people/anna', text: anna, links: [], etag: 'e1', backlinks: [] });
results.wiki_save_disabled_clean = headButton('wiki-save').disabled;
headButton('wiki-edit').dispatch('click');
results.wiki_editor_shown = !t.wikiEditor.hidden && t.wikiText.hidden;
t.wikiEditor.value = anna + 'more\n';
t.wikiEditor.dispatch('input');
results.wiki_save_disabled_dirty = headButton('wiki-save').disabled;
socket.sent.length = 0;
t.wikiEditor.dispatch('keydown', { key: 's', metaKey: true });
results.wiki_save_sent = [...socket.sent];
t.recv({ type: 'wiki_conflict', path: 'people/anna', text: anna + 'theirs\n', etag: 'e2' });
results.wiki_conflict_shown = !t.wikiConflict.hidden;
socket.sent.length = 0;
doc.getElementById('wiki-conflict-overwrite').dispatch('click');
results.wiki_overwrite_sent = [...socket.sent];
results.wiki_conflict_hidden_after = t.wikiConflict.hidden;
t.recv({ type: 'wiki_page', path: 'people/anna', text: anna + 'more\n', links: [], etag: 'e3', backlinks: [] });
results.wiki_mode_after_save = t.wikiEditor.hidden ? 'view' : 'edit';
headButton('wiki-edit').dispatch('click');
t.wikiEditor.value = anna + 'scratch\n';
t.wikiEditor.dispatch('input');
t.wikiEditor.dispatch('keydown', { key: 'Escape' });
results.wiki_escape = { confirmed: dialogs.log.at(-1), editor_hidden: t.wikiEditor.hidden };

// ---- Today / New page / search
socket.sent.length = 0;
t.wikiToday.dispatch('click');
dialogs.answer = 'notes/scratch';
t.wikiNew.dispatch('click');
t.wikiSearch.value = 'atlas';
t.wikiSearch.dispatch('input');
results.wiki_tools_sent = [...socket.sent];
t.recv({ type: 'wiki_search', query: 'atlas', hits: [{ path: 'projects/atlas', snippet: 'The rewrite.', score: 1 }] });
results.wiki_search_rows = listRows();

// ---------------------------------------------------------------- scenarios (PRD 30)
const rows = () => t.paletteList.children.map(c => c.children[0].textContent);
const activeRow = () => t.paletteList.children.findIndex(c => c.className.includes('active'));
const type = (q) => { t.paletteInput.value = q; t.paletteInput.dispatch('input'); };
const key = (k) => t.paletteInput.dispatch('keydown', { key: k });
const palEl = doc.getElementById('palette');

// ---- scenario 1: ⌘K opens (focused, empty, fixed commands then pages), toggles, Esc closes,
//      the view digits are off while it is open
t.setView('chat');
t.wikiSearch.value = ''; t.wikiSearch.dispatch('input');
t.cmdK();
results.pal_open = { hidden: palEl.hidden, focused: doc.activeElement === t.paletteInput,
                     value: t.paletteInput.value, rows: rows() };
t.keydown('3');
results.pal_view_while_open = t.view();
t.cmdK();
results.pal_hidden_after_toggle = palEl.hidden;
t.cmdK(); key('Escape');
results.pal_hidden_after_escape = palEl.hidden;

// ---- scenario 2: ranking
const firstFor = (q) => { t.cmdK(); type(q); const r = rows(); t.cmdK(); return r; };
results.pal_rank_atl = firstFor('atl');
results.pal_rank_set = firstFor('set');
results.pal_rank_wk = firstFor('wk');
results.pal_rank_zzz = firstFor('zzz');

// ---- scenario 3: ↓ ↑ wrap back to 0; Enter opens the page in the Wiki view
t.cmdK(); type('atl');
key('ArrowDown'); const afterDown = activeRow(); key('ArrowUp');
results.pal_nav = { after_down: afterDown, after_up: activeRow() };
socket.sent.length = 0;
key('Enter');
results.pal_enter = { view: t.view(), hidden: palEl.hidden, sent: [...socket.sent] };

// ---- scenario 4: the search fallback
t.setView('chat');
t.cmdK(); type('zzz'); socket.sent.length = 0; key('Enter');
results.pal_fallback = { view: t.view(), search: t.wikiSearch.value, sent: [...socket.sent] };

// ---- scenario 5: payloads equal the existing controls
const runLabel = (label) => {
  t.cmdK(); type(label);
  const i = rows().indexOf(label);
  socket.sent.length = 0;
  t.paletteList.children[i].dispatch('click');
  return [...socket.sent];
};
results.pal_backdrop = runLabel('Backdrop: Home office');
results.pal_verbosity = runLabel('Reply length: short');
t.recv({ type: 'hands_free', value: true });   // 'turn on' would start the mic; assert the off direction
results.pal_voice = runLabel('Voice: turn off');
t.recv({ type: 'hands_free', value: false });
const mutedBefore = t.muted; runLabel('Mute replies');
results.pal_mute = { before: mutedBefore, after: t.muted };
results.pal_ingest = runLabel('Re-ingest last session');
sandbox.confirm = (msg) => { dialogs.log.push(['confirm', msg]); return false; };
results.pal_shutdown_declined = runLabel('Shut down');
results.pal_shutdown_asked = dialogs.log.at(-1);
sandbox.confirm = (msg) => { dialogs.log.push(['confirm', msg]); return true; };
results.pal_shutdown_agreed = runLabel('Shut down');

// ---- scenario 6: clicking the second row runs it (Go to Memory)
t.setView('chat');
t.cmdK();
t.paletteList.children[1].dispatch('click');
results.pal_click = { view: t.view(), hidden: palEl.hidden };

// ---------------------------------------------------------------- scenarios (PRD 31)
// ---- scenario 8: the Sources view shows the new kind chips and a Meeting button on a
//      transcript/calendar source that names its meeting page
t.setView('sources');
t.recv({ type: 'sources', sources: [
  { sha: 's1', path: 'sources/falcon-sync', raw: 'raw/inbox/2026-08-30-sync.vtt', kind: 'transcript', title: 'Falcon sync',
    when: '2026-08-30 09:00:00', fed: ['meetings/2026-08-30-falcon-sync'], meeting: 'meetings/2026-08-30-falcon-sync' },
  { sha: 's2', path: 'sources/deck', raw: 'raw/inbox/2026-08-30-deck.pptx', kind: 'pptx', title: 'Deck', when: '2026-08-30 09:01:00', fed: [], meeting: '' },
  { sha: 's3', path: 'sources/plan', raw: 'raw/inbox/2026-08-30-plan.docx', kind: 'docx', title: 'Plan', when: '2026-08-30 09:02:00', fed: [], meeting: '' },
  { sha: 's4', path: 'sources/week', raw: 'raw/inbox/2026-08-30-week.ics', kind: 'calendar', title: 'Week', when: '2026-08-30 09:03:00', fed: [], meeting: '' },
  { sha: 's5', path: 'sources/sheet', raw: 'raw/inbox/2026-08-30-sheet.xlsx', kind: 'xlsx', title: 'Sheet', when: '2026-08-30 09:04:00', fed: [], meeting: '' },
] });
results.src_chips = t.wikiList.querySelectorAll('.chip').map(c => c.textContent);
t.selectWikiPage('sources/falcon-sync');
t.recv({ type: 'wiki_page', path: 'sources/falcon-sync', text: '---\ntype: source\n---\n\n# Falcon sync\n', links: [], etag: 'e9', backlinks: [] });
results.src_meeting_button = !!headButton('source-meeting');
t.selectWikiPage('sources/deck');
t.recv({ type: 'wiki_page', path: 'sources/deck', text: '---\ntype: source\n---\n\n# Deck\n', links: [], etag: 'e9', backlinks: [] });
results.src_meeting_button_plain = !!headButton('source-meeting');
t.selectWikiPage('sources/falcon-sync');
t.recv({ type: 'wiki_page', path: 'sources/falcon-sync', text: '---\ntype: source\n---\n\n# Falcon sync\n', links: [], etag: 'e9', backlinks: [] });
socket.sent.length = 0;
headButton('source-meeting').dispatch('click');
results.src_meeting_click = { view: t.view(), sent: socket.sent.filter(m => m.action === 'read_wiki_page').map(m => m.path) };

// ---- scenario 10: a paste leaves the title empty (the model picks one); a dropped file keeps
//      its name; a generic file name (note.txt) is left empty too; the new kinds are accepted
t.intakeFromText('Frequently Asked Questions about the Falcon rocket, version 3, last updated in August\nmore text here');
results.intake_paste_title = { value: t.intakeTitle.value };
t.closeIntake();
t.intakeFromFiles([new sandbox.File(['x'], 'falcon-status.docx')]);
results.intake_file_title = t.intakeTitle.value;
results.intake_docx_ok = !t.intakeItems.children[0].className.includes('bad');
t.closeIntake();
t.intakeFromFiles([new sandbox.File(['x'], 'note.txt')]);
results.intake_generic_title = t.intakeTitle.value;
t.closeIntake();
t.intakeFromFiles([new sandbox.File(['x'], 'legacy.doc')]);
results.intake_doc_ok = !t.intakeItems.children[0].className.includes('bad');
t.closeIntake();

// ---- scenario 6: the calendar horizon field round-trips through set_setting
t.recv({ type: 'setting', key: 'calendar_horizon_days', value: 28 });
results.horizon_shown = t.calendarHorizon.value;
socket.sent.length = 0;
t.calendarHorizon.value = '21';
t.calendarHorizon.dispatch('change');
results.horizon_sent = [...socket.sent];

// ---- PRD 35: the page offers its position once per connect; the Settings row shows what is in use
if (socket.onopen) socket.onopen();                       // the stub never fires it on its own
results.location_sent = socket.sent.filter(m => m.action === 'set_location');
socket.sent.length = 0;
t.recv({ type: 'setting', key: 'location', value: { lat: 46.771, lon: 23.624, tz: 'Europe/Bucharest', source: 'browser', name: 'Cluj-Napoca, Romania' } });
results.location_shown = doc.getElementById('settings-location').textContent;

// ---- PRD 32: the recall threshold field round-trips through set_setting
t.recv({ type: 'setting', key: 'recall_min_score', value: 3.5 });
results.recall_shown = t.recallMinScore.value;
socket.sent.length = 0;
t.recallMinScore.value = '1.5';
t.recallMinScore.dispatch('change');
results.recall_sent = [...socket.sent];

results.verbosity_note = (html.match(/id="verbosity-desc"[^>]*>([^<]*)</) || [])[1] || '';
// ---- PRD 36: a three-file batch is posted one file at a time, rows show their state, Cancel
//      stops after the file in flight
const tick = () => new Promise(r => setImmediate(r));
const flat = (n) => (typeof n === 'string' ? n : (n.children.length ? n.children.map(flat).join('') : n.textContent));
const rowStates = () => t.intakeItems.children.map(row => {
  const state = row.querySelector('.intake-item-state');
  return { cls: row.className, state: state ? flat(state) : '' };
});
const posts = [];                    // every /ingest call: {fields, resolve}
sandbox.fetch = (url, opts) => new Promise((resolve) => posts.push({
  url, fields: opts.body.fields,
  resolve: (status, body) => resolve({ ok: status === 200, status, text: async () => JSON.stringify(body) }),
}));
t.intakeFromFiles([new sandbox.File(['1'], 'one.md'), new sandbox.File(['2'], 'two.md'), new sandbox.File(['3'], 'three.md')]);
results.batch_before = rowStates().map(r => r.state);
const batch = t.submitIntake();
await tick();
results.batch_started = { rows: rowStates(), posts: posts.length, status: t.intakeStatus.textContent,
                          first_post_files: posts[0].fields.filter(f => f[0] === 'file').map(f => f[2]) };
t.recv({ type: 'ingest_progress', filename: 'one.md', stage: 'summarising', i: 1, n: 1 });
results.batch_summarising = rowStates()[0];
posts[0].resolve(200, { filed: [{ path: 'sources/falcon-status', raw: 'raw/inbox/2026-08-30-one.md', already: false }], errors: [] });
await tick(); await tick();
t.recv({ type: 'wiki', pages: [{ path: 'sources/falcon-status', title: 'Falcon status', type: 'source', tags: [] }], lint: [], vault: '/tmp/v' });
results.batch_one_filed = { rows: rowStates(), posts: posts.length, status: t.intakeStatus.textContent,
                            link: !!t.intakeItems.children[0].querySelector('.intake-link') };
t.intakeCancel.dispatch('click');
results.batch_cancel_status = t.intakeStatus.textContent;
posts[1].resolve(200, { filed: [{ path: 'sources/two', raw: 'raw/inbox/two.md', already: true }], errors: [] });
await batch;
results.batch_after_cancel = { rows: rowStates(), posts: posts.length, status: t.intakeStatus.textContent,
                               button: t.intakeCancel.textContent, open: !t.intake.hidden };
t.intakeCancel.dispatch('click');
results.batch_closed = t.intake.hidden;

// a failed request marks its row and the batch goes on
t.intakeFromFiles([new sandbox.File(['a'], 'bad.md'), new sandbox.File(['b'], 'good.md')]);
const batch2 = t.submitIntake();
await tick();
posts[2].resolve(413, { detail: 'the text is larger than 50 MB' });
await tick(); await tick();
posts[3].resolve(200, { filed: [{ path: 'sources/good', raw: 'raw/inbox/good.md', already: false }], errors: [] });
await batch2;
results.batch_error = { rows: rowStates(), status: t.intakeStatus.textContent, open: !t.intake.hidden };
t.intakeCancel.dispatch('click');

// ---------------------------------------------------------------- scenarios (PRD 33)
// ---- scenario 3: memory entries carry a `source` tag; `identity.sources` renders the same
// tag on the scalar row instead of a `sources` fact row, and never as an editable field.
t.recv({ type: 'memory', profile: {
  identity: { name: 'Misi', sources: { name: 'Falcon plan' } },
  preferences: {},
  people: [{ name: 'Anna', rel: 'colleague', note: '', seen: '2026-08-30', stale: false, stale_after: 90, source: 'Falcon plan' }],
  projects: [], recurring: [], episodic: [],
}, rendered: '', count: 2 });
results.memory_source_tags = t.memSections.querySelectorAll('.mem-source').map(n => n.textContent);
results.memory_scalar_keys = t.memSections.querySelectorAll('.mem-key').map(n => n.textContent);

// ---------------------------------------------------------------- scoped errors
// An `error` lands in the slot its `scope` names, not in whichever view is open: a wiki-scoped
// error arriving while on Memory leaves the memory slot alone; a memory-scoped one fills it.
t.setView('memory');
t.memError.textContent = ''; t.wikiError.textContent = ''; t.status.textContent = '';
t.recv({ type: 'error', scope: 'wiki', text: 'bad path raw/x' });
results.error_wiki_while_on_memory = {
  memory: t.memError.textContent, wiki: t.wikiError.textContent, status: t.status.textContent,
};
t.recv({ type: 'error', scope: 'memory', text: 'Wait until the reply finishes.' });
results.error_memory_slot = t.memError.textContent;
t.setView('wiki');
t.recv({ type: 'error', scope: 'settings', text: 'Unknown setting' });
results.error_settings_status = { status: t.status.textContent, wiki: t.wikiError.textContent };
t.setView('chat');

// ---------------------------------------------------------------- scenarios (PRD 38)
// ---- scenario 7: opening a page sends set_focus; Chat sends set_focus {view: chat}; a server
// `wiki_open` selects the page and switches to Wiki without echoing set_focus; the composer
// placeholder names the focused page.
const focusSent = () => socket.sent.filter(m => m.action === 'set_focus');
t.setView('wiki');
t.recv({ type: 'wiki', vault: '/tmp/v', lint: [], pages: [
  { path: 'people/anna', title: 'Anna', type: 'person', updated: '2026-08-29', tags: [] },
  { path: 'projects/atlas', title: 'Atlas', type: 'project', updated: '2026-08-29', tags: [] },
] });
socket.sent.length = 0;
t.selectWikiPage('people/anna');
results.focus_select = { sent: focusSent(), placeholder: t.input.placeholder };
socket.sent.length = 0;
t.setView('chat');
results.focus_chat = { sent: focusSent(), placeholder: t.input.placeholder };
socket.sent.length = 0;
t.recv({ type: 'wiki_open', path: 'projects/atlas' });
results.focus_wiki_open = {
  view: t.view(), sent: focusSent(),
  read: socket.sent.filter(m => m.action === 'read_wiki_page').map(m => m.path),
  placeholder: t.input.placeholder,
};
t.setView('chat');

// ---------------------------------------------------------------- scenarios (PRD 39)
// ---- scenario 2: `context` → "ctx N %" on the status line, the Chat rail dot amber ≥ 75 / red ≥ 90,
// "forgot n old turns" when trimmed; scenario 5: `thread` inserts a divider with the summary and
// the palette's "New topic" sends the action.
t.setState('idle');
t.recv({ type: 'context', used: 9920, budget: 16000, pct: 62, trimmed: 0 });
results.ctx_62 = { status: t.status.textContent, dot: t.chatDot() };
t.recv({ type: 'context', used: 12000, budget: 16000, pct: 75, trimmed: 0 });
results.ctx_75 = { status: t.status.textContent, dot: t.chatDot() };
t.recv({ type: 'context', used: 14400, budget: 16000, pct: 90, trimmed: 3 });
results.ctx_90 = { status: t.status.textContent, dot: t.chatDot() };
t.recv({ type: 'context', used: 3000, budget: 16000, pct: 19, trimmed: 0 });
results.ctx_19 = { status: t.status.textContent, dot: t.chatDot() };
t.recv({ type: 'thread', text: 'We planned the Falcon sync; open: the budget.' });
results.thread_divider = t.historyLines().at(-1);
results.pal_new_topic = runLabel('New topic');

// ---------------------------------------------------------------- scenarios (PRD 41)
// ---- scenario 4: `services` with asr down → the amber banner + a copy button whose payload is
// the hint; both ok → hidden; close code 1006 → the red backend banner; Settings rows show
// dots + times.
const svcRows = () => t.settingsServices.children.map(r => ({
  name: r.querySelector('.name').textContent,
  dot: r.querySelector('.svc-dot').className,
  value: r.querySelector('.set-value').textContent,
  copy: r.querySelectorAll('.svc-copy').map(b => b.dataset.copy),
}));
const banner = () => ({ hidden: t.serviceBanner.hidden, cls: t.serviceBanner.className,
                        text: t.serviceBannerText.textContent, copy: t.serviceCopyBtn.dataset.copy });
results.svc_initial = banner();
t.recv({ type: 'services', hint: 'scripts/voicebot start',
         llm: { ok: true, label: 'oMLX :8000', detail: 'model g loaded', checked: '2026-08-30T10:11:12' },
         asr: { ok: false, label: 'ASR :8020', detail: 'not answering (ConnectionError)', checked: '2026-08-30T10:11:13' } });
results.svc_asr_down = banner();
clipboardLog.length = 0;
t.serviceCopyBtn.dispatch('click');
results.svc_copied = [...clipboardLog];
results.svc_rows_down = svcRows();
t.recv({ type: 'services', hint: 'scripts/voicebot start',
         llm: { ok: true, label: 'oMLX :8000', detail: 'model g loaded', checked: '2026-08-30T10:11:42' },
         asr: { ok: true, label: 'ASR :8020', detail: 'answering', checked: '2026-08-30T10:11:43' } });
results.svc_both_ok = banner();
results.svc_rows_ok = svcRows();
t.closeSocket(1006);
results.svc_backend_down = banner();
t.closeSocket(4000);                                   // a takeover is not "backend down"
results.svc_after_4000 = t.status.textContent;

// ---- addendum scenario 9: the admin password row + the Load / Unload model buttons
results.pw_initial = t.adminPasswordState.textContent;
t.recv({ type: 'setting', key: 'omlx_admin_password', omlx_admin_password_set: true, omlx_admin_password_source: 'env' });
results.pw_env = t.adminPasswordState.textContent;
t.recv({ type: 'setting', key: 'omlx_admin_password', omlx_admin_password_set: true, omlx_admin_password_source: 'settings' });
results.pw_settings = t.adminPasswordState.textContent;
socket.sent.length = 0;
t.adminPassword.value = 'hunter2';
doc.getElementById('admin-password-save').dispatch('click');
results.pw_save_sent = [...socket.sent];
results.pw_input_after_save = t.adminPassword.value;
socket.sent.length = 0;
doc.getElementById('admin-password-clear').dispatch('click');
results.pw_clear_sent = [...socket.sent];
t.recv({ type: 'setting', key: 'omlx_admin_password', omlx_admin_password_set: false, omlx_admin_password_source: null });
results.pw_cleared = t.adminPasswordState.textContent;
socket.sent.length = 0;
doc.getElementById('model-load').dispatch('click');
doc.getElementById('model-unload').dispatch('click');
results.model_sent = socket.sent.map(m => m.action);

// ---------------------------------------------------------------- scenarios (PRD 43)
// ---- scenario 3: a `card` renders collapsed with "show all"; Copy puts the markdown on the
// clipboard; Save opens the folder picker (notes/) and sends save_card; `card_saved` relabels
// the button; nothing about the card is decoded as audio.
t.setView('chat');
const cardText = (n) => (n.children && n.children.length ? n.children.map(cardText).join("") : String(n.textContent || ""));
const CARD_MD = '# Bleak Falls Barrow\n\n## Route\n- Arvel\n\n## Puzzle\nSnake, snake, whale.\n\nSources: Skyrim guide · part 12/300';
const decodesBefore = audio.decodes;
t.recv({ type: 'card', id: 'card1', markdown: CARD_MD, sources: ['Skyrim guide · part 12/300'], title: 'Bleak Falls Barrow' });
const cardEl = t.cards().at(-1);
const cardBody = cardEl.querySelector('.card-body');
const cardMore = cardEl.querySelector('.card-more');
results.card_rendered = { role: cardEl.children[0].textContent, cls: cardBody.className, more: cardMore.textContent,
                          text: cardText(cardBody), decodes: audio.decodes - decodesBefore,
                          history_role: t.historyLines().at(-1).role };
cardMore.dispatch('click');
results.card_expanded = { cls: cardBody.className, more: cardMore.textContent };
clipboardLog.length = 0;
cardEl.querySelector('.card-copy').dispatch('click');
await tick();
results.card_copied = [...clipboardLog];
const cardSave = cardEl.querySelector('.card-save');
const cardPicker = cardEl.querySelector('.card-picker');
results.card_picker_before = cardPicker.hidden;
socket.sent.length = 0;
cardSave.dispatch('click');
results.card_picker = { hidden: cardPicker.hidden, folder: cardEl.querySelector('.card-folder').value, sent: [...socket.sent] };
cardEl.querySelector('.card-save-ok').dispatch('click');
results.card_save_sent = [...socket.sent];
results.card_saving = { label: cardSave.textContent, disabled: cardSave.disabled };
t.recv({ type: 'error', scope: 'wiki', text: 'already saved' });
results.card_save_failed = { label: cardSave.textContent, disabled: cardSave.disabled, note: cardEl.querySelector('.card-note').textContent };
cardSave.dispatch('click');
cardEl.querySelector('.card-folder').value = 'drafts/';
cardEl.querySelector('.card-save-ok').dispatch('click');
results.card_save_folder = socket.sent.at(-1);
t.recv({ type: 'card_saved', id: 'card1', path: 'notes/2026-08-30-bleak-falls-barrow' });
results.card_saved = { label: cardSave.textContent, disabled: cardSave.disabled, picker_hidden: cardPicker.hidden };
t.recv({ type: 'card', id: 'card2', markdown: '# Two\n\n## A\nx', sources: [], title: 'Two', saved: 'notes/2026-08-30-two' });
results.card_replayed_saved = t.cards().at(-1).querySelector('.card-save').textContent;

console.log(JSON.stringify(results, null, 1));
