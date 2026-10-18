"""PRD 58: scripts/turn.py against a scripted websocket server (no app, no model)."""
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from websockets.sync.server import serve

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def server():
    """`run(script)`: serve one connection with script(ws, seen); returns the port and the frames seen."""
    servers = []

    def start(script):
        seen = []

        def handler(ws):
            script(ws, seen)

        srv = serve(handler, "127.0.0.1", 0)
        servers.append(srv)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv.socket.getsockname()[1], seen

    yield start
    for s in servers:
        s.shutdown()


def send(ws, **msg):
    ws.send(json.dumps(msg))


def run_turn(port, *extra):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "turn.py"), "--port", str(port), *extra],
                          capture_output=True, text=True, timeout=30)


def test_reply_is_printed_and_playback_acknowledged(server):
    def script(ws, seen):
        send(ws, type="state", value="idle")
        json.loads(ws.recv())                       # text_turn
        send(ws, type="state", value="thinking")
        send(ws, type="speech_end", turn=1)
        seen.append(json.loads(ws.recv()))          # playback_done
        send(ws, type="transcript", role="assistant", text="It is noon.")
        send(ws, type="state", value="idle")
    port, seen = server(script)
    out = run_turn(port)
    assert out.returncode == 0, out.stderr
    assert "It is noon." in out.stdout
    assert seen == [{"action": "playback_done", "turn": 1}]


def test_error_message_exits_1(server):
    def script(ws, seen):
        send(ws, type="state", value="idle")
        json.loads(ws.recv())
        send(ws, type="error", scope="turn", text="LLM is down")
    port, _ = server(script)
    out = run_turn(port)
    assert out.returncode == 1 and "LLM is down" in out.stderr


def test_timeout_exits_1(server):
    def script(ws, seen):
        send(ws, type="state", value="idle")
        json.loads(ws.recv())
        send(ws, type="state", value="thinking")
        threading.Event().wait(5)
    port, _ = server(script)
    out = run_turn(port, "--timeout", "2")
    assert out.returncode == 1 and "turn did not complete" in out.stderr


def test_first_connect_greet_is_not_taken_for_the_reply(server):
    def script(ws, seen):
        send(ws, type="state", value="thinking")
        send(ws, type="speech_end", turn=0)
        seen.append(json.loads(ws.recv()))          # playback_done for the greet
        send(ws, type="transcript", role="assistant", text="Hello")
        send(ws, type="state", value="idle")
        seen.append(json.loads(ws.recv()))          # text_turn
        send(ws, type="state", value="thinking")
        send(ws, type="transcript", role="assistant", text="It is noon.")
        send(ws, type="state", value="idle")
    port, seen = server(script)
    out = run_turn(port)
    assert out.returncode == 0, out.stderr
    assert "It is noon." in out.stdout and "Hello" not in out.stdout
    assert seen[0] == {"action": "playback_done", "turn": 0}
    assert seen[1]["action"] == "text_turn"


def test_stale_replayed_reply_is_not_printed(server):
    def script(ws, seen):
        send(ws, type="transcript", role="assistant", text="OLD REPLY")
        send(ws, type="state", value="idle")
        seen.append(json.loads(ws.recv()))          # text_turn
        send(ws, type="state", value="thinking")
        send(ws, type="state", value="idle")
    port, seen = server(script)
    out = run_turn(port)
    assert out.returncode == 0, out.stderr
    assert "OLD REPLY" not in out.stdout
    assert seen[0]["action"] == "text_turn"


def test_empty_text_is_a_usage_error(server):
    out = run_turn(1, "  ")
    assert out.returncode == 2 and "empty" in out.stderr
