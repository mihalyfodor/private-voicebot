"""PRD 58: one typed turn against a running app, headless. Prints the assistant reply.

    .venv/bin/python scripts/turn.py [--port 8010] [--timeout 180] ["what time is it"]

Frames before our text_turn (connect replay, first-connect greet) are drained and ignored.
Answers every speech_end with playback_done (without it the server never goes idle again).
Exit 0 with the reply on stdout; exit 1 on an error message or when the turn does not finish in time.
"""
import argparse
import json
import sys
import time

from websockets.exceptions import ConnectionClosed, InvalidHandshake
from websockets.sync.client import connect


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("text", nargs="?", default="what time is it")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--timeout", type=float, default=180)
    args = ap.parse_args(argv)
    if not args.text.strip():
        ap.error("text must not be empty")
    deadline = time.monotonic() + args.timeout
    reply, thinking, sent = "", False, False
    try:
        with connect(f"ws://{args.host}:{args.port}/ws", max_size=None) as ws:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise TimeoutError
                raw = ws.recv(timeout=left)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "speech_end":
                    ws.send(json.dumps({"action": "playback_done", "turn": msg.get("turn")}))
                elif kind == "transcript" and msg.get("role") == "assistant" and sent:
                    reply = msg.get("text", "")
                elif kind == "error":
                    print(msg.get("text", "error"), file=sys.stderr)
                    return 1
                elif kind == "state" and msg.get("value") == "thinking" and sent:
                    thinking = True
                elif kind == "state" and msg.get("value") == "idle":
                    if thinking:
                        print(reply)
                        return 0
                    if not sent:
                        ws.send(json.dumps({"action": "text_turn", "text": args.text}))
                        sent, reply = True, ""
    except (TimeoutError, OSError) as e:
        print(f"turn did not complete ({type(e).__name__})", file=sys.stderr)
        return 1
    except (ConnectionClosed, InvalidHandshake) as e:
        print(f"turn failed: {type(e).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
