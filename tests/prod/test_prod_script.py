"""PRD 40 scenario 5: prod.sh migrate gathers state into DATA once; compose file validates."""
import os
import shutil
import subprocess

import pytest

from tests.conftest import ROOT

PROD_SH = os.path.join(ROOT, "scripts", "prod.sh")
GIT_ID = ["-c", "user.name=t", "-c", "user.email=t@t"]


def test_migrate_moves_state_into_data_and_refuses_twice(tmp_path):
    home = tmp_path / "home"
    vault = home / "llm-wiki"
    vault.mkdir(parents=True)
    (vault / "index.md").write_text("# index\n")
    subprocess.run(["git", "init", "-q"], cwd=vault, check=True)
    subprocess.run(["git", *GIT_ID, "add", "."], cwd=vault, check=True)
    subprocess.run(["git", *GIT_ID, "commit", "-q", "-m", "seed"], cwd=vault, check=True)

    src = tmp_path / "checkout"
    src.mkdir()
    state = ["memory.json", "memory_ops.jsonl", "shortmem.txt", "settings.json", ".env",
             "credentials.json", "token.json", "session.log"]
    for name in state:
        (src / name).write_text(name)

    data = home / "docker-data" / "voicebot"
    env = dict(os.environ, HOME=str(home), SRC=str(src), DATA=str(data), GIT_CONFIG_GLOBAL=os.devnull)
    out = subprocess.run(["bash", PROD_SH, "migrate"], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr

    assert (data / "llm-wiki" / ".git").is_dir()
    assert (data / "llm-wiki" / "index.md").exists()
    for rel in ["memory/memory.json", "memory/memory_ops.jsonl", "memory/shortmem.txt", "settings.json",
                ".env", "secrets/credentials.json", "secrets/token.json", "logs/session.log"]:
        assert (data / rel).read_text() == os.path.basename(rel), rel
    assert (src / ".env").exists()                          # copied, not moved
    assert not vault.exists()
    for name in state:
        if name != ".env":
            assert not (src / name).exists(), name
    assert f"moved {vault} -> {data / 'llm-wiki'}" in out.stdout
    assert f"moved {src / 'settings.json'} -> {data / 'settings.json'}" in out.stdout
    assert "copied" in out.stdout

    again = subprocess.run(["bash", PROD_SH, "migrate"], env=env, capture_output=True, text=True)
    assert again.returncode != 0
    assert "refusing" in again.stderr


def test_prod_script_parses():
    assert subprocess.run(["bash", "-n", PROD_SH], capture_output=True).returncode == 0


def test_compose_file_validates(tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("docker is not installed")
    data = tmp_path / "data"
    data.mkdir()
    (data / ".env").write_text("")
    env = dict(os.environ, DATA=str(data))
    out = subprocess.run(["docker", "compose", "-f", os.path.join(ROOT, "docker-compose.yml"),
                          "config", "--quiet"], env=env, cwd=ROOT, capture_output=True, text=True)
    if "Cannot connect" in out.stderr or "is not a docker command" in out.stderr:
        pytest.skip(out.stderr.strip())
    assert out.returncode == 0, out.stderr
