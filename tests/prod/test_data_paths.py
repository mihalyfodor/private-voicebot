"""PRD 40 scenario 1: every state path follows its env knob; nothing is written next to the code."""
import sys, os
import json
import subprocess
from unittest.mock import patch

from fastapi.testclient import TestClient

import chatbot
import llm
import memory
from core import llm as core_llm
import wiki
from tests.conftest import ROOT, FakeClient, StreamClient, commits, drain_initial, fake_tts

MODULE_PATHS = """
import json, memory, wiki
from core import settings, trace
from tools import gmail
print(json.dumps([settings.SETTINGS_PATH, trace.TRACE_DIR, memory.MEMORY_PATH, memory.OPS_LOG_PATH,
                  memory.SHORTMEM_PATH, wiki.WIKI_PATH, gmail.GMAIL_CREDENTIALS_PATH, gmail.GMAIL_TOKEN_PATH]))
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
    llm.reset()
    monkeypatch.setattr(chatbot, "synthesize", fake_tts)
    fake = StreamClient("[happy] Sure thing.")

    client = TestClient(chatbot.app)
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
    memory.save([{"role": "user", "content": "remind me to call Anna"},
                 {"role": "assistant", "content": "Sure thing."}])

    wiki.ensure_vault()
    wiki.apply_page_ops([{"op": "new_page", "path": "people/anna", "type": "person", "name": "Anna",
                          "summary": "Colleague."}], "test")

    assert os.path.exists(tmp_path / "events.jsonl")
    assert "remind me to call Anna" not in open(tmp_path / "events.jsonl").read()
    assert os.path.exists(tmp_path / "memory.json")
    assert os.path.exists(tmp_path / "wiki" / "people" / "anna.md")
    assert os.path.isdir(tmp_path / "wiki" / ".git")
    assert commits()
    assert set(os.listdir(ROOT)) == root_before
