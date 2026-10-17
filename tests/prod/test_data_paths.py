"""PRD 40 scenario 1: every state path follows its env knob; nothing is written next to the code."""
import sys, os
import json
import subprocess
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import app
from server import speech
from agent import conversation
from agent import fold
from core import llm as core_llm
from vault import pages
from vault import store
from tests.conftest import ROOT, FakeClient, commits, drain_initial, fake_tts

MODULE_PATHS = """
import json
from core import settings, trace
from memory import store as m_store
from tools import gmail
from vault import store
print(json.dumps([settings.SETTINGS_PATH, trace.TRACE_DIR, m_store.MEMORY_PATH, m_store.OPS_LOG_PATH,
                  m_store.SHORTMEM_PATH, store.WIKI_PATH, gmail.GMAIL_CREDENTIALS_PATH, gmail.GMAIL_TOKEN_PATH]))
"""


def test_every_state_path_follows_its_env_knob(tmp_path):
    env = dict(os.environ,
               MEMORY_DIR=str(tmp_path / "memory"),
               SETTINGS_PATH=str(tmp_path / "settings.json"),
               TRACE_DIR=str(tmp_path / "logs"),
               GMAIL_DIR=str(tmp_path / "secrets"),
               WIKI_PATH=str(tmp_path / "llm-wiki"),
               GIT_CONFIG_GLOBAL=os.devnull)
    out = subprocess.run([sys.executable, "-c", MODULE_PATHS], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    paths = json.loads(out.stdout)
    assert len(paths) == 8
    for p in paths:
        assert p.startswith(str(tmp_path)), p


def test_a_turn_a_memory_save_and_a_wiki_op_write_only_under_the_sandbox(tmp_path, monkeypatch):
    root_before = set(os.listdir(ROOT))
    conversation.reset()
    monkeypatch.setattr(speech, "synthesize", fake_tts)
    fake = FakeClient("[happy] Sure thing.")

    client = TestClient(app.app)
    with patch.object(core_llm, "_client", fake), client.websocket_connect("/ws") as ws:
        drain_initial(ws)
        ws.send_json({"action": "text_turn", "text": "remind me to call Anna"})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "speech_end":
                ws.send_json({"action": "playback_done", "turn": msg["turn"]})
            if msg["type"] == "state" and msg["value"] == "idle":
                break

    monkeypatch.setattr(core_llm, "_client",
                        FakeClient(json.dumps([{"op": "ADD", "path": "people[]", "value": {"name": "Anna"}}])))
    fold.save([{"role": "user", "content": "remind me to call Anna"},
                 {"role": "assistant", "content": "Sure thing."}])

    store.ensure_vault()
    pages.apply_page_ops([{"op": "new_page", "path": "people/anna", "type": "person", "name": "Anna",
                          "summary": "Colleague."}], "test")

    assert os.path.exists(tmp_path / "events.jsonl")
    assert "remind me to call Anna" not in open(tmp_path / "events.jsonl").read()
    assert os.path.exists(tmp_path / "memory.json")
    assert os.path.exists(tmp_path / "wiki" / "people" / "anna.md")
    assert os.path.isdir(tmp_path / "wiki" / ".git")
    assert commits()
    assert set(os.listdir(ROOT)) == root_before


DEFAULT_PATHS = """
import json
from core import config
from vault import store
from vault.__main__ import cli_vault
print(json.dumps([config.WIKI_PATH, store.WIKI_PATH, cli_vault("/nonexistent/.env")]))
"""


def test_default_vault_is_never_the_prod_vault(tmp_path):
    """Two environments: with no WIKI_PATH anywhere, every default lands on the dev vault, never ~/llm-wiki
    (the 2026-10-16 incident: a bare import built an index inside the real vault)."""
    env = {k: v for k, v in os.environ.items() if k not in ("WIKI_PATH", "DATA")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["PYTHONPATH"] = ROOT
    out = subprocess.run([sys.executable, "-c", DEFAULT_PATHS], cwd=tmp_path, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    paths = json.loads(out.stdout)
    prod = os.path.expanduser("~/llm-wiki")
    assert all(p == os.path.expanduser("~/llm-wiki-dev") for p in paths), paths
    assert prod not in paths
