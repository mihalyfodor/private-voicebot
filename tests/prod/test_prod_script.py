"""PRD 40 scenario 5: prod.sh migrate gathers state into DATA once; compose file validates."""
import os
import glob
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


# --- backup chore (docs/roadmap.md section 0) ---
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(vault, *args):
    subprocess.run(["git", *GIT_ID, *args], cwd=vault, check=True, capture_output=True)


def _commit(vault, text):
    (vault / "index.md").write_text(text)
    _git(vault, "add", ".")
    _git(vault, "commit", "-q", "-m", text)


def _seed(tmp_path, **extra):
    data, backups = tmp_path / "data", tmp_path / "backups"
    vault = data / "llm-wiki"
    vault.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=vault, check=True)
    _commit(vault, "one")
    (data / "settings.json").write_text("{}")
    env = dict(os.environ, HOME=str(tmp_path), DATA=str(data), BACKUP_DIR=str(backups),
               GIT_CONFIG_GLOBAL=os.devnull, **extra)
    return env, data, backups, vault


def _backup(env):
    return subprocess.run(["bash", PROD_SH, "backup"], env=env, capture_output=True, text=True)


def _files(backups, pat):
    return sorted(glob.glob(str(backups / pat)))


def _log(data):
    return (data / "logs" / "backup.log").read_text().splitlines()


@needs_git
def test_backup_first_run_writes_archives_marker_and_log(tmp_path):
    env, data, backups, _ = _seed(tmp_path)
    out = _backup(env)
    assert out.returncode == 0, out.stderr
    assert len(_files(backups, "voicebot-data-*.tgz")) == 1
    assert len(_files(backups, "llm-wiki-*.bundle")) == 1
    assert (backups / ".last-head").read_text().strip()
    assert " | ok | " in _log(data)[-1]
    assert not list(backups.glob("*.tmp"))


@needs_git
def test_backup_rotation_keeps_newest_seven(tmp_path):
    env, _, backups, _ = _seed(tmp_path, FORCE="1")
    backups.mkdir()
    for i in range(1, 10):
        (backups / f"voicebot-data-2020010{i}-000000.tgz").touch()
        (backups / f"llm-wiki-2020010{i}-000000.bundle").touch()
    assert _backup(env).returncode == 0
    for pat in ("voicebot-data-*.tgz", "llm-wiki-*.bundle"):
        files = _files(backups, pat)
        assert len(files) == 7
        assert "2020" not in os.path.basename(files[-1])        # newest real one survives
        assert "20200101" not in "".join(files)


@needs_git
def test_backup_refuses_low_free_space(tmp_path):
    env, data, backups, _ = _seed(tmp_path, MIN_FREE_MB="999999999")
    out = _backup(env)
    assert out.returncode == 1
    assert "free" in out.stderr
    assert not _files(backups, "*.tgz") and not _files(backups, "*.bundle")
    assert " | fail | " in _log(data)[-1]


@needs_git
def test_backup_refuses_over_total_cap(tmp_path):
    env, data, backups, _ = _seed(tmp_path, MAX_TOTAL_MB="0")
    out = _backup(env)
    assert out.returncode == 1
    assert "total" in out.stderr
    assert not _files(backups, "*.tgz") and not _files(backups, "*.bundle")
    assert " | fail | " in _log(data)[-1]


@needs_git
def test_backup_held_lock_exits_and_keeps_lock(tmp_path):
    env, _, backups, _ = _seed(tmp_path)
    (backups / ".lock").mkdir(parents=True)
    out = _backup(env)
    assert out.returncode == 1
    assert "lock" in out.stderr
    assert (backups / ".lock").is_dir()
    assert not _files(backups, "*.tgz")


@needs_git
def test_backup_skips_unchanged_head_then_runs_after_new_commit(tmp_path):
    env, _, backups, vault = _seed(tmp_path)
    assert _backup(env).returncode == 0
    again = _backup(env)
    assert again.returncode == 0
    assert "skipping" in again.stdout
    assert len(_files(backups, "*.tgz")) == 1
    _commit(vault, "two")
    subprocess.run(["sleep", "1"])                              # names carry second-resolution stamps
    assert _backup(env).returncode == 0
    assert len(_files(backups, "*.tgz")) == 2
    assert not (backups / ".lock").exists()


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is not installed")
@pytest.mark.parametrize("name", ["com.voicebot.asr.plist", "com.voicebot.backup.plist",
                                  "com.voicebot.remarkable.plist"])
def test_launchd_plists_lint(name):
    path = os.path.join(ROOT, "scripts", "launchd", name)
    assert subprocess.run(["plutil", "-lint", path], capture_output=True).returncode == 0


def test_remarkable_plist_runs_the_push_service_on_loopback():
    text = open(os.path.join(ROOT, "scripts", "launchd", "com.voicebot.remarkable.plist")).read()
    assert "scripts/remarkable_server.py" in text and "8021" in text and "127.0.0.1" in text
    assert "<key>KeepAlive</key><true/>" in text


@needs_git
def test_backup_log_truncated_to_200_lines(tmp_path):
    env, data, _, _ = _seed(tmp_path)
    (data / "logs").mkdir()
    (data / "logs" / "backup.log").write_text("".join(f"old {i}\n" for i in range(250)))
    assert _backup(env).returncode == 0
    lines = _log(data)
    assert len(lines) == 200
    assert " | ok | " in lines[-1]


def test_backup_removes_stale_tmp_from_a_killed_run(tmp_path):
    env, _data, backups, _vault = _seed(tmp_path)
    backups.mkdir()
    (backups / "voicebot-data-20200101-000000.tgz.tmp").write_text("half")
    out = _backup(env)
    assert out.returncode == 0, out.stderr
    assert not list(backups.glob("*.tmp"))
