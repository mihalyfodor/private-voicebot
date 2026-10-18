"""PRD 40 scenario 5: prod.sh migrate gathers state into DATA once; compose file validates."""
import os
import glob
import shutil
import socket
import subprocess

import pytest

from tests.conftest import ROOT
from tests.prod.test_remarkable_server import push_service  # noqa: F401  (the fixture)

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

    data = home / ".voicebot" / "data"
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
    assert (backups / ".last-state").read_text().strip()
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
def test_backup_skips_when_nothing_changed_then_runs_after_new_commit(tmp_path):
    env, _, backups, vault = _seed(tmp_path)
    assert _backup(env).returncode == 0
    again = _backup(env)
    assert again.returncode == 0
    assert "skipping" in again.stdout
    assert "state unchanged" in _log(env_data(env))[-1]
    assert len(_files(backups, "*.tgz")) == 1
    _commit(vault, "two")
    subprocess.run(["sleep", "1"])                              # names carry second-resolution stamps
    assert _backup(env).returncode == 0
    assert len(_files(backups, "*.tgz")) == 2
    assert not (backups / ".lock").exists()


def env_data(env):
    return __import__("pathlib").Path(env["DATA"])


def _shim(tmp_path, **scripts):
    """A PATH-first dir of fake executables: _shim(tmp_path, docker="exit 0")."""
    d = tmp_path / "shims"
    d.mkdir(exist_ok=True)
    for name, body in scripts.items():
        (d / name).write_text("#!/bin/bash\n" + body + "\n")
        (d / name).chmod(0o755)
    return str(d) + os.pathsep + os.environ["PATH"]


@needs_git
def test_backup_runs_when_memory_changed_with_same_head(tmp_path):
    env, data, backups, _ = _seed(tmp_path)
    (data / "memory").mkdir()
    (data / "memory" / "memory.json").write_text("{}")
    assert _backup(env).returncode == 0
    (data / "memory" / "memory.json").write_text('{"a": 1}')
    subprocess.run(["sleep", "1"])
    assert _backup(env).returncode == 0
    assert len(_files(backups, "*.tgz")) == 2


@needs_git
def test_backup_skips_when_nothing_changed(tmp_path):
    env, data, backups, _ = _seed(tmp_path)
    (data / "memory").mkdir()
    (data / "memory" / "memory.json").write_text("{}")
    assert _backup(env).returncode == 0
    again = _backup(env)
    assert again.returncode == 0 and "skipping" in again.stdout
    assert len(_files(backups, "*.tgz")) == 1
    assert "state unchanged" in _log(data)[-1]


@needs_git
def test_backup_warns_when_agent_not_loaded(tmp_path):
    env, _, backups, _ = _seed(tmp_path)
    env = dict(env, PATH=_shim(tmp_path, launchctl="exit 1"))
    env.pop("XPC_SERVICE_NAME", None)
    out = _backup(env)
    assert out.returncode == 0, out.stderr
    assert "backup agent not loaded" in out.stderr
    assert len(_files(backups, "*.tgz")) == 1


@needs_git
def test_backup_quiet_under_launchd(tmp_path):
    env, *_ = _seed(tmp_path)
    env = dict(env, PATH=_shim(tmp_path, launchctl="exit 1"), XPC_SERVICE_NAME="com.voicebot.backup")
    out = _backup(env)
    assert out.returncode == 0, out.stderr
    assert "warning" not in out.stderr


def _deploy_env(tmp_path, status="", describe="abc123", branch="main", config=None):
    env, data, backups, _ = _seed(tmp_path)
    real_git = shutil.which("git")
    git = (f'case "$*" in *"status --porcelain"*) printf %s "{status}";; *describe*) echo {describe};;'
           f' *--abbrev-ref*) echo {branch};; *) exec {real_git} "$@";; esac')
    curl = f"echo '{config}'" if config else "exit 22"
    path = _shim(tmp_path, git=git, docker="exit 0", curl=curl, launchctl="exit 0")
    return dict(env, PATH=path), data


def _deploy(env):
    return subprocess.run(["bash", PROD_SH, "deploy"], env=env, capture_output=True, text=True)


@needs_git
def test_deploy_refuses_dirty_tree_and_dirty_tag(tmp_path):
    env, _ = _deploy_env(tmp_path, status=" M file")
    out = _deploy(env)
    assert out.returncode == 1 and "deploy:" in out.stderr and "dirty" in out.stderr
    (tmp_path / "second").mkdir()
    env, _ = _deploy_env(tmp_path / "second", describe="abc-dirty")
    out = _deploy(env)
    assert out.returncode == 1 and "deploy:" in out.stderr and "dirty tag" in out.stderr


@needs_git
def test_deploy_refuses_when_busy(tmp_path):
    env, _ = _deploy_env(tmp_path, config='{"busy": true, "version": "x"}')
    out = _deploy(env)
    assert out.returncode == 1
    assert "idle check" in out.stderr


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is not installed")
@pytest.mark.parametrize("name", ["com.voicebot.asr.plist", "com.voicebot.backup.plist",
                                  "com.voicebot.remarkable.plist"])
def test_launchd_plists_lint(name):
    path = os.path.join(ROOT, "scripts", "launchd", name)
    assert subprocess.run(["plutil", "-lint", path], capture_output=True).returncode == 0


def test_remarkable_plist_runs_the_push_service_on_loopback():
    text = open(os.path.join(ROOT, "scripts", "launchd", "com.voicebot.remarkable.plist")).read()
    assert "scripts/remarkable_server.py" in text and "8021" in text
    assert "<key>KeepAlive</key><true/>" in text


def _lan_address():
    """The machine's first non-loopback IPv4 address, or None."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))          # no packet is sent; picks the outgoing interface
            return None if s.getsockname()[0].startswith("127.") else s.getsockname()[0]
    except OSError:
        return None


def test_remarkable_service_refuses_non_loopback_connections(push_service):
    lan = _lan_address()
    if lan is None:
        pytest.skip("no non-loopback IPv4 address")
    port = int(push_service.rsplit(":", 1)[1])
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)
        assert s.connect_ex((lan, port)) != 0


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


# --- prod knows its commit; shutdown gets time; the live search index stays out of backups ---

def test_build_bakes_the_tag_in_and_compose_does_not_override_it():
    script = open(PROD_SH).read()
    assert '--build-arg VOICEBOT_VERSION="$tag"' in script
    dockerfile = open(os.path.join(ROOT, "Dockerfile")).read()
    assert "ARG VOICEBOT_VERSION" in dockerfile and "ENV VOICEBOT_VERSION=$VOICEBOT_VERSION" in dockerfile
    compose = open(os.path.join(ROOT, "docker-compose.yml")).read()
    assert "VOICEBOT_VERSION" not in compose
    assert "stop_grace_period: 90s" in compose


@needs_git
def test_backup_excludes_both_search_indexes(tmp_path):
    env, data, backups, vault = _seed(tmp_path)
    for rel in ("raw/.index", ".index"):
        (vault / rel).mkdir(parents=True)
        (vault / rel / "cache").write_text("x")
    assert _backup(env).returncode == 0
    names = subprocess.run(["tar", "tzf", _files(backups, "voicebot-data-*.tgz")[0]],
                           capture_output=True, text=True).stdout
    assert "settings.json" in names
    assert ".index" not in names


@pytest.mark.parametrize("agent", ["asr", "backup", "remarkable"])
def test_install_launchd_renders_the_checkout_paths_and_no_placeholder_survives(agent, tmp_path):
    env = {**os.environ, "HOME": str(tmp_path)}
    out = subprocess.run([os.path.join(ROOT, "scripts", "install_launchd.sh"), agent, "--dry-run"],
                         capture_output=True, text=True, env=env, check=True).stdout
    assert "__ROOT__" not in out and "__HOME__" not in out and "/Users/" not in out.replace(str(ROOT), "").replace(str(tmp_path), "")
    assert f"<string>{ROOT}</string>" in out                  # WorkingDirectory is this checkout
    if agent == "backup":
        assert f"<string>{tmp_path}</string>" in out          # HOME for launchd's bare environment
    assert not (tmp_path / "Library").exists()                 # --dry-run writes nothing
