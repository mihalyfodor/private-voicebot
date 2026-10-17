"""PRD 49 scenario 7: the prod / dev profiles and the startup refusal (docs/49-event-trace.md)."""
import os
import subprocess
import sys
from unittest.mock import patch

from server import turns
from server import speech
from agent import conversation
from core import trace
from server.session import Session
from tests.conftest import ROOT, fake_tts, run_turn, trace_lines, FakeClient
from core import llm as core_llm


def test_prod_turn_never_creates_verbose_and_starts_with_service_start(vault, prod_profile):
    conversation.reset()
    with patch.object(core_llm, "_client", FakeClient("[neutral] Sure.")), patch.object(speech, "synthesize", fake_tts):
        run_turn(turns.handle_text, Session(), 1, "remind me to call Anna tomorrow")
    rows = trace_lines()
    assert rows[0]["kind"] == "service" and rows[0]["event"] == "start" and rows[0]["profile"] == "prod"
    assert {r["profile"] for r in rows} == {"prod"}
    assert any(r["kind"] == "turn_end" for r in rows)
    assert not os.path.exists(trace.paths()["verbose"])


def test_dev_against_prod_data_refuses_to_start(tmp_path):
    home = tmp_path / "home"
    wiki_path = home / "docker-data" / "voicebot" / "llm-wiki"
    env = dict(os.environ, HOME=str(home), TRACE_PROFILE="dev", WIKI_PATH=str(wiki_path),
               TRACE_DIR=str(tmp_path / "logs"), MEMORY_DIR=str(tmp_path / "memory"),
               SETTINGS_PATH=str(tmp_path / "settings.json"), OPEN_BROWSER="0", PORT="0")
    out = subprocess.run([sys.executable, "chatbot.py"], cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 2
    lines = [line for line in out.stderr.splitlines() if line.strip()]
    assert len(lines) == 1 and str(wiki_path) in lines[0] and "refusing" in lines[0]
    assert not (tmp_path / "logs").exists()


def test_a_host_script_against_prod_data_refuses_too(tmp_path):
    home = tmp_path / "home"
    vault = home / "docker-data" / "voicebot" / "llm-wiki"
    vault.mkdir(parents=True)
    (tmp_path / "export").mkdir()
    env = dict(os.environ, HOME=str(home), TRACE_PROFILE="dev", MEMORY_DIR=str(tmp_path / "memory"), PORT="1")
    env.pop("TRACE_DIR", None)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ingest_batch.py"), str(tmp_path / "export"),
                          "--vault", str(vault), "--dry-run"], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=60)
    assert out.returncode != 0
    lines = [line for line in out.stderr.splitlines() if line.strip()]
    assert len(lines) == 1 and str(vault) in lines[0] and "refusing" in lines[0]


def test_a_host_script_writes_to_trace_dir_never_beside_the_vault(tmp_path):
    data = tmp_path / "data"
    (data / "logs").mkdir(parents=True)
    vault = data / "llm-wiki"
    vault.mkdir()
    (tmp_path / "export").mkdir()
    env = dict(os.environ, TRACE_PROFILE="prod", TRACE_DIR=str(tmp_path / "trace"), MEMORY_DIR=str(tmp_path / "memory"),
               PORT="1", PROD_PORT="1", DEV_PORT="1")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "ingest_batch.py"), str(tmp_path / "export"),
                          "--vault", str(vault), "--dry-run"], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    first = open(tmp_path / "trace" / "events.jsonl").readline()
    assert '"event": "start"' in first and '"profile": "prod"' in first
    assert os.listdir(data / "logs") == [] and not (tmp_path / "trace" / "verbose.jsonl").exists()


def test_dev_defaults_to_a_repo_local_logs_folder():
    import subprocess as sp
    out = sp.run([sys.executable, "-c", "from core import trace; print(trace.TRACE_DIR)"], cwd=ROOT, capture_output=True,
                 text=True, env={k: v for k, v in os.environ.items() if k != "TRACE_DIR"})
    assert out.stdout.strip() == os.path.join(ROOT, "logs")


def test_dev_inside_a_container_is_refused(tmp_path, monkeypatch):
    marker = tmp_path / ".dockerenv"
    marker.write_text("")
    monkeypatch.setattr(trace, "DOCKERENV", str(marker))
    assert "container" in trace.decide({"TRACE_PROFILE": "dev"})[1]
    assert trace.decide({"TRACE_PROFILE": "prod"}) == ("prod", "")
    assert "unknown" in trace.decide({"TRACE_PROFILE": "debug"})[1]


def test_an_unknown_profile_fails_closed_to_prod(monkeypatch):
    monkeypatch.setenv("TRACE_PROFILE", "verbose-please")
    trace.reset()
    trace.verbose("prompt", messages=["x"])
    assert trace.profile() == "prod" and not os.path.exists(trace.paths()["verbose"])


def test_compose_forces_prod_and_caps_the_container_log():
    compose = open(os.path.join(ROOT, "docker-compose.yml")).read()
    assert "TRACE_PROFILE: prod" in compose and "TRACE_DIR: /data/logs" in compose
    assert 'max-size: "10m"' in compose and 'max-file: "5"' in compose
    assert "SESSION_LOG" not in compose
