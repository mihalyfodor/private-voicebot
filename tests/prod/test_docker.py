"""PRD 40 scenarios 6–8: the prod image for real (marker `docker`, skipped without a daemon).

One image build per session (`voicebot:test-<sha>`, removed afterwards unless DOCKER_KEEP_IMAGE=1),
one throwaway data dir under $HOME (colima only maps bind-mount ownership for paths under the home
directory) and a throwaway host port. Scenario 7 also needs oMLX on :8000 and the host ASR service
on :8020 (started here from the checkout's venv when nothing answers there).

    pytest -m docker tests/prod/test_docker.py -s
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROD = str(ROOT / "scripts" / "prod.sh")
PORT = int(os.environ.get("DOCKER_TEST_PORT", "8013"))
RESTORE_PORT = 8012
ASR_URL = "http://127.0.0.1:8020"
HEALTHY_WITHIN_S = 90

pytestmark = pytest.mark.docker


def sh(*args, env=None, check=True, timeout=600, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(list(args), capture_output=True, text=True, check=check, timeout=timeout,
                          env={**os.environ, **(env or {})}, **kw)


def daemon_up() -> bool:
    return shutil.which("docker") is not None and sh("docker", "info", check=False, timeout=30).returncode == 0


def get(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def wait_healthy(container: str, limit: float) -> float:
    """Seconds until the compose healthcheck says healthy; fails past `limit`."""
    start = time.time()
    while time.time() - start < limit:
        status = sh("docker", "inspect", "--format", "{{.State.Health.Status}}", container).stdout.strip()
        if status == "healthy":
            return round(time.time() - start, 1)
        if status not in ("starting", ""):
            logs = sh("docker", "logs", "--tail", "40", container, check=False)
            pytest.fail(f"container went {status}:\n{logs.stdout}{logs.stderr}")
        time.sleep(2)
    logs = sh("docker", "logs", "--tail", "40", container, check=False)
    pytest.fail(f"not healthy within {limit}s:\n{logs.stdout}{logs.stderr}")


class Prod:
    """One `prod.sh` deployment against a throwaway data dir, port and compose project."""

    def __init__(self, tag: str):
        self.data = pathlib.Path(tempfile.mkdtemp(prefix="voicebot-docker-", dir=os.path.expanduser("~/.cache")))
        self.project = f"voicebot-test-{os.getpid()}"
        self.env = {"DATA": str(self.data), "TAG": tag, "PROD_PORT": str(PORT), "PROJECT": self.project}
        self.base = f"http://127.0.0.1:{PORT}"
        for sub in ("memory", "secrets", "logs"):
            (self.data / sub).mkdir()
        self.data.joinpath(".env").write_text(self._env_file())

    @staticmethod
    def _env_file() -> str:
        """Only the oMLX settings the live turn needs, from the environment or the checkout's .env."""
        from dotenv import dotenv_values
        values = {**dotenv_values(ROOT / ".env"), **{k: v for k, v in os.environ.items() if k.startswith("OMLX_")}}
        return "".join(f"{k}={v}\n" for k, v in values.items() if k.startswith("OMLX_") and v)

    def prod(self, *args, check=True, env=None) -> subprocess.CompletedProcess:
        return sh("bash", PROD, *args, env={**self.env, **(env or {})}, check=check)

    def compose(self, *args) -> subprocess.CompletedProcess:
        return sh("docker", "compose", "-p", self.project, "-f", str(ROOT / "docker-compose.yml"), *args, env=self.env)

    @property
    def container(self) -> str:
        return self.compose("ps", "-q").stdout.strip()

    def exec(self, *args, check=False) -> subprocess.CompletedProcess:
        return sh("docker", "exec", self.container, *args, check=check)

    def vault_head(self) -> str:
        """`git log -1 --oneline` of the vault on the host ("" before its first commit)."""
        return sh("git", "-C", str(self.data / "llm-wiki"), "log", "-1", "--oneline", check=False).stdout.strip()

    def down(self):
        self.compose("down")
        shutil.rmtree(self.data, ignore_errors=True)


@pytest.fixture(scope="session")
def image() -> str:
    if not daemon_up():
        pytest.skip("no Docker daemon")
    sha = sh("git", "-C", str(ROOT), "rev-parse", "--short", "HEAD").stdout.strip()
    tag = f"test-{sha}"
    if sh("docker", "image", "inspect", f"voicebot:{tag}", check=False).returncode != 0:
        print(f"\nbuilding voicebot:{tag} (several minutes) ...")
        sh("bash", PROD, "build", env={"TAG": tag}, timeout=1800)
    yield tag
    if not os.environ.get("DOCKER_KEEP_IMAGE"):
        sh("docker", "rmi", f"voicebot:{tag}", check=False)


@pytest.fixture(scope="session")
def prod(image):
    p = Prod(image)
    try:
        p.prod("up")
        p.healthy_after = wait_healthy(p.container, HEALTHY_WITHIN_S)
        yield p
    finally:
        p.down()


# ---------------------------------------------------------------- scenario 6: hardened container


def test_scenario_6_container_is_healthy_unprivileged_and_read_only(prod):
    print(f"\nhealthy after {prod.healthy_after}s")
    assert prod.healthy_after <= HEALTHY_WITHIN_S
    uid = prod.exec("id", "-u", check=True).stdout.strip()
    print(f"uid inside the container: {uid}")
    assert uid != "0"
    touch = prod.exec("touch", "/app/should-fail")
    assert touch.returncode != 0, "the app directory must be read-only"
    config = json.loads(get(prod.base + "/api/config"))
    print(f"/api/config vault: {config['vault']}")
    assert config["vault"] == "/data/llm-wiki"


# ---------------------------------------------------------------- scenario 7: live turns


class Socket:
    """A bare websocket client for the app: drains, sends, answers playback_done like the browser."""

    def __init__(self, base: str):
        from websockets.sync.client import connect
        self.ws = connect(base.replace("http", "ws") + "/ws", max_size=None)
        self.wait_idle()

    def recv(self, timeout=120) -> dict:
        while True:
            raw = self.ws.recv(timeout=timeout)
            if isinstance(raw, bytes):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "speech_end":
                self.send({"action": "playback_done", "turn": msg.get("turn")})
            return msg

    def send(self, msg: dict):
        self.ws.send(json.dumps(msg))

    def wait_idle(self) -> list:
        seen = []
        while True:
            msg = self.recv()
            seen.append(msg)
            if msg.get("type") == "state" and msg.get("value") == "idle":
                return seen

    def until(self, wanted: str) -> dict:
        while True:
            msg = self.recv()
            if msg.get("type") == wanted:
                return msg
            if msg.get("type") in ("error", "memory_error"):
                pytest.fail(f"unexpected {msg}")

    def close(self):
        self.ws.close()


def omlx_up() -> bool:
    import urllib.error
    try:
        get("http://127.0.0.1:8000/v1/models", timeout=3)
        return True
    except urllib.error.HTTPError:
        return True          # 401 without the key is still an answer
    except Exception:
        return False


def asr_up() -> bool:
    try:
        get(ASR_URL + "/health", timeout=2)
        return True
    except Exception:
        return False


def start_asr() -> subprocess.Popen:
    """scripts/asr_server.py from this venv on :8020, returned once /health answers."""
    proc = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "asr_server.py")], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        if asr_up():
            return proc
        time.sleep(1)
    proc.terminate()
    pytest.fail("asr_server.py did not come up on :8020")


class AsrService:
    """The host ASR service on :8020: `proc` is ours (restartable), None when something else answers there."""

    def __init__(self):
        self.proc = None if asr_up() else start_asr()

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)

    def restart(self):
        self.proc = start_asr()


@pytest.fixture(scope="session")
def asr_service():
    """The host ASR service on :8020 — whatever is there, or one started from this venv."""
    svc = AsrService()
    yield svc
    if svc.proc is not None:
        svc.proc.terminate()


def spoken_pcm(text: str) -> bytes:
    """`text` as Kokoro speech at 16 kHz int16, with a second of silence to close the utterance."""
    from kokoro_onnx import Kokoro
    model = os.path.expanduser(os.getenv("KOKORO_MODEL", "~/models/kokoro/kokoro-v1.0.onnx"))
    voices = os.path.expanduser(os.getenv("KOKORO_VOICES", "~/models/kokoro/voices-v1.0.bin"))
    if not os.path.exists(model):
        pytest.skip("no Kokoro model on the host")
    samples, rate = Kokoro(model, voices).create(text, voice="af_bella", speed=1.0)
    n = int(len(samples) * 16000 / rate)
    audio = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples)
    audio = np.concatenate([np.zeros(8000), audio, np.zeros(24000)])
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


def test_scenario_7_typed_turn_wiki_commit_memory_restart_and_spoken_turn(prod, asr_service):
    if not omlx_up():
        pytest.skip("oMLX is not answering on :8000")
    ws = Socket(prod.base)

    # a typed turn answered by the host oMLX
    ws.send({"action": "text_turn", "text": "Reply with the single word pong and nothing else."})
    reply = ws.until("transcript")
    assert reply["role"] == "assistant"
    print(f"\ntyped turn reply: {reply['text']!r}")
    assert reply["text"].strip()
    ws.wait_idle()

    # a wiki save from the UI commits into $DATA/llm-wiki
    ws.send({"action": "wiki_create", "path": "notes/docker-check"})
    ws.until("wiki")
    head = prod.vault_head()
    print(f"vault commit on the host: {head}")
    assert head, "the vault has no commit on the host"
    assert (prod.data / "llm-wiki" / "notes" / "docker-check.md").exists()

    # a memory entry survives a restart
    ws.send({"action": "memory_op", "op": "UPDATE", "path": "identity.location", "value": "Mahe"})
    mem = ws.until("memory")
    assert mem["profile"]["identity"]["location"] == "Mahe"
    ws.close()
    prod.compose("restart")
    wait_healthy(prod.container, HEALTHY_WITHIN_S)
    ws = Socket(prod.base)
    ws.send({"action": "get_memory"})
    assert ws.until("memory")["profile"]["identity"]["location"] == "Mahe"
    print("memory entry survived the restart")

    # a spoken turn transcribed by the host ASR service
    ws.send({"action": "set_hands_free", "value": True})
    ws.until("hands_free")
    pcm = spoken_pcm("What is two plus two?")
    for i in range(0, len(pcm), 1024):
        ws.ws.send(pcm[i:i + 1024])
        time.sleep(0.01)
    heard = ws.until("transcript")
    print(f"spoken turn transcript: {heard['text']!r}")
    assert heard["role"] == "user" and "plus" in heard["text"].lower()   # whisper writes "two" or "2"
    ws.wait_idle()
    ws.close()


# ---------------------------------------------------------------- PRD 41 scenario 6: the banner


def services_of(prod) -> dict:
    return json.loads(get(prod.base + "/api/config"))["services"]


def wait_asr_state(prod, ok: bool, limit: float = 60) -> float:
    """Seconds until /api/config.services.asr.ok == ok (what the banner renders); fails past `limit`."""
    start = time.time()
    while time.time() - start < limit:
        asr = services_of(prod).get("asr")
        if asr is not None and asr["ok"] is ok:
            return round(time.time() - start, 1)
        time.sleep(1)
    pytest.fail(f"asr.ok never became {ok} within {limit}s: {services_of(prod)}")


def test_prd41_scenario_6_service_banner_while_the_asr_service_is_stopped(prod, asr_service):
    if asr_service.proc is None:
        pytest.skip("the ASR service on :8020 is not ours to stop")
    wait_asr_state(prod, True)
    ws = Socket(prod.base)
    asr_service.stop()
    shown = wait_asr_state(prod, False)
    push = ws.until("services")                      # the page gets it pushed, not only via /api/config
    assert push["asr"]["ok"] is False and push["asr"]["label"] == "ASR :8020"
    assert push["hint"] == "scripts/voicebot start"
    print(f"\nbanner shown {shown}s after the ASR service stopped: {push['asr']['detail']}")
    assert shown <= 60
    asr_service.restart()
    gone = wait_asr_state(prod, True)
    assert ws.until("services")["asr"]["ok"] is True
    print(f"banner gone {gone}s after the ASR service came back")
    assert gone <= 60
    ws.close()


# ---------------------------------------------------------------- scenario 8: backup / restore


def test_scenario_8_backup_restore_round_trip(prod):
    if not (prod.data / "llm-wiki" / ".git").is_dir():
        # scenario 7 may have been skipped: give the vault a commit to round-trip
        Socket(prod.base).send({"action": "wiki_create", "path": "notes/backup-check"})
        for _ in range(20):
            if prod.vault_head():
                break
            time.sleep(0.5)
    head = prod.vault_head()
    backup_dir = prod.data.parent / (prod.data.name + "-backups")
    restore_dir = prod.data.parent / (prod.data.name + "-restore")
    try:
        out = prod.prod("backup", env={"BACKUP_DIR": str(backup_dir)}).stdout.split()
        archive, bundle = out[-2], out[-1]
        assert archive.endswith(".tgz") and bundle.endswith(".bundle")
        restored = prod.prod("restore", archive, bundle, str(restore_dir),
                             env={"RESTORE_PORT": str(RESTORE_PORT)})
        print("\n" + restored.stdout.strip())
        assert "restore ok" in restored.stdout
        assert f"restored vault head: {head}" in restored.stdout
        assert '"vault":"/data/llm-wiki"' in restored.stdout       # the curl of /api/config on :8012
        assert sh("git", "-C", str(restore_dir / "llm-wiki"), "log", "-1", "--oneline").stdout.strip() == head
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)
        shutil.rmtree(restore_dir, ignore_errors=True)
