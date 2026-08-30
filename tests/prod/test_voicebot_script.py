"""PRD 41 scenario 5: scripts/voicebot start/stop check what's down and fix only that."""
import os
import re
import stat
import subprocess

from tests.conftest import ROOT

VOICEBOT = os.path.join(ROOT, "scripts", "voicebot")

CURL_STUB = r"""#!/usr/bin/env bash
echo "curl $*" >> "$STUB_LOG"
url="${@: -1}"
case "$url" in
  */health)
    if [[ "$url" == *:8000/health* ]]; then
      [ -f "$STUB_HOME/omlx-up" ] && exit 0
      [ "$STUB_OMLX" = up ] && exit 0 || exit 7
    elif [[ "$url" == *:8020/health* ]]; then
      [ "$STUB_ASR" = up ] && exit 0 || exit 7
    else
      exit 0
    fi
    ;;
  */v1/models)
    if [ -f "$STUB_HOME/model-loaded" ] || [ "$STUB_MODEL" = loaded ]; then
      echo "{\"data\":[{\"id\":\"$OMLX_MODEL\"}]}"
    else
      echo "{\"data\":[]}"
    fi
    ;;
  */admin/api/models/*/load|*/admin/api/models/*/unload)
    exit 0
    ;;
  */api/config)
    if [ -f "$STUB_HOME/app-up" ] || [ "$STUB_APP" = up ]; then
      exit 0
    else
      exit 7
    fi
    ;;
  *)
    exit 0
    ;;
esac
"""

COLIMA_STUB = r"""#!/usr/bin/env bash
echo "colima $*" >> "$STUB_LOG"
case "$1" in
  status) [ "$STUB_COLIMA" = running ] && exit 0 || exit 1 ;;
  start) exit 0 ;;
  stop) exit 0 ;;
  *) exit 0 ;;
esac
"""

OMLX_STUB = r"""#!/usr/bin/env bash
echo "omlx $*" >> "$STUB_LOG"
case "$1" in
  start) touch "$STUB_HOME/omlx-up"; exit 0 ;;
  stop) exit 0 ;;
  *) exit 0 ;;
esac
"""

LAUNCHCTL_STUB = r"""#!/usr/bin/env bash
echo "launchctl $*" >> "$STUB_LOG"
case "$1" in
  list) [ "$STUB_ASR" = up ] && echo "$ASR_LABEL" ; exit 0 ;;
  *) exit 0 ;;
esac
"""

DOCKER_STUB = r"""#!/usr/bin/env bash
echo "docker $*" >> "$STUB_LOG"
if [ "$1" = compose ]; then
  shift
  case "$*" in
    *"up -d"*) touch "$STUB_HOME/app-up"; exit 0 ;;
    *down*) exit 0 ;;
    *ps*) exit 0 ;;
    *) exit 0 ;;
  esac
fi
exit 0
"""

STUBS = {"curl": CURL_STUB, "colima": COLIMA_STUB, "omlx": OMLX_STUB,
         "launchctl": LAUNCHCTL_STUB, "docker": DOCKER_STUB}


def _write_stub(path, content):
    path.write_text(content)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def run(tmp_path, *args, **stub_env):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name, content in STUBS.items():
        _write_stub(bindir / name, content)

    home = tmp_path / "home"
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    (home / "Library" / "LaunchAgents" / "com.voicebot.asr.plist").write_text("stub")
    data = tmp_path / "data"
    data.mkdir()
    (data / ".env").write_text("")
    log = tmp_path / "stub.log"
    log.write_text("")

    env = dict(os.environ)
    env.update({
        "PATH": f"{bindir}:{env['PATH']}",
        "HOME": str(home),
        "OMLX_BIN": str(bindir / "omlx"),
        "ASR_PLIST": str(home / "Library" / "LaunchAgents" / "com.voicebot.asr.plist"),
        "ASR_LABEL": "com.voicebot.asr",
        "DATA": str(data),
        "OMLX_MODEL": "test-model",
        "OMLX_API_KEY": "k",
        "STUB_LOG": str(log),
        "STUB_HOME": str(home),
        "VOICEBOT_ENV": str(tmp_path / "no-such-env"),
        "STUB_COLIMA": stub_env.get("colima", "running"),
        "STUB_OMLX": stub_env.get("omlx", "up"),
        "STUB_MODEL": stub_env.get("model", "loaded"),
        "STUB_ASR": stub_env.get("asr", "up"),
        "STUB_APP": stub_env.get("app", "up"),
    })
    result = subprocess.run([VOICEBOT, *args], cwd=str(tmp_path), env=env,
                             capture_output=True, text=True, timeout=60)
    lines = log.read_text().splitlines()
    return result, lines


def test_script_parses():
    assert subprocess.run(["bash", "-n", VOICEBOT], capture_output=True).returncode == 0


def test_start_with_everything_up_starts_nothing(tmp_path):
    result, lines = run(tmp_path, "start")
    assert result.returncode == 0, result.stderr
    log = "\n".join(lines)
    assert "colima start" not in log
    assert "omlx start" not in log
    assert "/load" not in log
    for line in lines:
        assert not line.startswith("launchctl load")
        assert not line.startswith("launchctl kickstart")
    assert "up -d" not in log
    assert "colima     running" in result.stdout
    assert "app" in result.stdout and "up" in result.stdout


def test_start_brings_up_only_whats_down(tmp_path):
    result, lines = run(tmp_path, "start", omlx="down", app="down", model="loaded")
    assert result.returncode == 0, result.stderr
    log = "\n".join(lines)
    assert "omlx start" in log
    assert re.search(r"docker compose .* up -d", log)
    assert "colima start" not in log
    assert not any(line.startswith("launchctl load") or line.startswith("launchctl kickstart")
                   for line in lines)
    assert "/load" not in log


def test_stop_order(tmp_path):
    result, lines = run(tmp_path, "stop")
    assert result.returncode == 0, result.stderr
    log_str = "\n".join(lines)
    down_idx = next(i for i, l in enumerate(lines) if re.search(r"docker compose .* down", l))
    unload_idx = next(i for i, l in enumerate(lines) if l.startswith("launchctl unload"))
    curl_unload_idx = next(i for i, l in enumerate(lines)
                            if "/admin/api/models/test-model/unload" in l)
    assert down_idx < unload_idx < curl_unload_idx
    assert "omlx stop" not in log_str
    assert "colima stop" not in log_str


def test_stop_all_also_stops_omlx_and_colima(tmp_path):
    result, lines = run(tmp_path, "stop", "--all")
    assert result.returncode == 0, result.stderr
    curl_unload_idx = next(i for i, l in enumerate(lines)
                            if "/admin/api/models/test-model/unload" in l)
    omlx_stop_idx = next(i for i, l in enumerate(lines) if l == "omlx stop")
    colima_stop_idx = next(i for i, l in enumerate(lines) if l == "colima stop")
    assert curl_unload_idx < omlx_stop_idx < colima_stop_idx


def test_status_dev_starts_nothing(tmp_path):
    result, lines = run(tmp_path, "status", "--dev")
    assert result.returncode == 0, result.stderr
    assert "skipped" in result.stdout
    for line in lines:
        assert "start" not in line
        assert not line.startswith("launchctl load")
        assert "up -d" not in line
