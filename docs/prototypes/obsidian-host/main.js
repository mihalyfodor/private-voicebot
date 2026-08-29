// Voicebot spike plugin — answers docs/21-obsidian-host.md spike questions.
// Plain CommonJS, no build step: Obsidian loads main.js directly via its
// own require() (Node/Electron renderer context), so `require('obsidian')`
// and `require('child_process')` both work unbundled.
'use strict';

const { Plugin, ItemView, Notice, PluginSettingTab, Setting } = require('obsidian');
const { spawn } = require('child_process');
const path = require('path');

const VIEW_TYPE = 'voicebot-spike-view';
const VOICEBOT_URL = 'http://127.0.0.1:8010/';

// Default settings point at this repo's own venv/python so the spike can be
// run with zero configuration when the plugin folder is copied verbatim.
const DEFAULT_SETTINGS = {
  pythonPath: '/Users/fomi/projects/personal/private-voicebot/.venv/bin/python',
  repoPath: '/Users/fomi/projects/personal/private-voicebot',
};

// ItemView hosting the iframe. Obsidian's <webview> tag is disabled in the
// main window (see README/PRD sources), so this uses a plain <iframe> —
// the pattern used by Local Web Page Viewer / HTML Preview / Web View.
class VoicebotView extends ItemView {
  getViewType() { return VIEW_TYPE; }
  getDisplayText() { return 'Voicebot spike'; }
  getIcon() { return 'audio-lines'; }

  async onOpen() {
    const container = this.containerEl.children[1];
    container.empty();
    container.style.padding = '0';
    const iframe = container.createEl('iframe');
    iframe.src = VOICEBOT_URL;
    // "microphone" must be explicitly allowed on the iframe for getUserMedia
    // to have any chance of succeeding inside it; autoplay is needed for TTS.
    iframe.setAttribute('allow', 'microphone; autoplay');
    iframe.style.width = '100%';
    iframe.style.height = '100%';
    iframe.style.border = 'none';
  }

  async onClose() {}
}

module.exports = class VoicebotSpikePlugin extends Plugin {
  async onload() {
    await this.loadSettings();
    this.serverProcess = null;

    this.registerView(VIEW_TYPE, (leaf) => new VoicebotView(leaf));
    this.addSettingTab(new VoicebotSettingTab(this.app, this));

    this.addCommand({ id: 'open-panel', name: 'Voicebot spike: open panel', callback: () => this.openPanel() });
    this.addCommand({ id: 'mic-test', name: 'Voicebot spike: mic test (plugin context)', callback: () => this.micTest() });
    this.addCommand({ id: 'spawn-server', name: 'Voicebot spike: spawn server', callback: () => this.spawnServer() });
    this.addCommand({ id: 'stop-server', name: 'Voicebot spike: stop server', callback: () => this.stopServer() });
  }

  onunload() {
    // Mirror what a real Route-1 plugin must do: kill the child on unload
    // (plugin disable/reload), not just on app quit.
    this.stopServer();
  }

  async openPanel() {
    const { workspace } = this.app;
    let leaf = workspace.getLeavesOfType(VIEW_TYPE)[0];
    if (!leaf) {
      leaf = workspace.getLeaf('tab');
      await leaf.setViewState({ type: VIEW_TYPE, active: true });
    }
    workspace.revealLeaf(leaf);
  }

  // Runs getUserMedia in the *plugin's* renderer context (not inside the
  // iframe) so we have a control data point for comparison against the
  // iframe's own mic behaviour (see README checklist).
  async micTest() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true },
      });
      const track = stream.getAudioTracks()[0];
      const settings = track.getSettings();
      new Notice(
        `mic OK (plugin ctx). echoCancellation=${settings.echoCancellation}`,
        8000
      );
      console.log('[voicebot-spike] getUserMedia settings:', settings);
      stream.getTracks().forEach((t) => t.stop());
    } catch (err) {
      new Notice(`mic FAILED (plugin ctx): ${err.message}`, 8000);
      console.error('[voicebot-spike] getUserMedia error:', err);
    }
  }

  spawnServer() {
    if (this.serverProcess) {
      new Notice('voicebot-spike: server already running');
      return;
    }
    const { pythonPath, repoPath } = this.settings;
    const script = path.join(repoPath, 'chatbot.py');
    this.serverProcess = spawn(pythonPath, [script], { cwd: repoPath });
    this.serverProcess.stdout.on('data', (d) => console.log('[chatbot.py]', d.toString()));
    this.serverProcess.stderr.on('data', (d) => console.error('[chatbot.py]', d.toString()));
    this.serverProcess.on('exit', (code, sig) => {
      console.log(`[voicebot-spike] chatbot.py exited code=${code} sig=${sig}`);
      this.serverProcess = null;
    });
    new Notice(`voicebot-spike: spawned pid ${this.serverProcess.pid}`);
  }

  stopServer() {
    if (!this.serverProcess) return;
    this.serverProcess.kill('SIGTERM');
    new Notice('voicebot-spike: sent SIGTERM');
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }
};

class VoicebotSettingTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    containerEl.empty();

    new Setting(containerEl)
      .setName('Python path')
      .setDesc('Path to the python interpreter (venv) used to run chatbot.py')
      .addText((text) => text.setValue(this.plugin.settings.pythonPath).onChange(async (value) => {
        this.plugin.settings.pythonPath = value.trim();
        await this.plugin.saveSettings();
      }));

    new Setting(containerEl)
      .setName('Repo path')
      .setDesc('Path to the private-voicebot repo (cwd for chatbot.py)')
      .addText((text) => text.setValue(this.plugin.settings.repoPath).onChange(async (value) => {
        this.plugin.settings.repoPath = value.trim();
        await this.plugin.saveSettings();
      }));
  }
}
