"""Headless smoke pass: stack, e2e, one live turn, one battery section, view screenshots (dev only).

    .venv/bin/python scripts/smoke.py [--out DIR] [--steps 1,2,3,4,5] [--no-start] [--section A] [--answer-timeout 300]

Never touches ~/llm-wiki: every vault is a throwaway under the out folder. Results: RESULT.json + RESULT.txt.
"""
import argparse
import functools
import json
import pathlib
import re
import shutil
import subprocess
import sys
import time

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
from tests.qa import stress  # noqa: E402

print = functools.partial(print, flush=True)
PY = sys.executable
E2E_SHOTS = pathlib.Path("/tmp/e2e")
QA_OUT = pathlib.Path("/tmp/qa")
VIEWS = [("chat", None), ("wiki", "wiki"), ("sources", "sources"), ("memory", "memory"), ("settings", "settings")]
MIN_PNG = 10 * 1024


def run(cmd, log=None, timeout=1800, **kw):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, **kw)
    out = p.stdout + p.stderr
    if log:
        pathlib.Path(log).write_text(out)
    return p.returncode, out


def copy_pngs(src: pathlib.Path, dst: pathlib.Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.png"):
        shutil.copy(f, dst / f.name)


def summary_line(out: str) -> str:
    lines = [l.strip(" =") for l in out.strip().splitlines() if l.strip()]
    return lines[-1] if lines else "(no output)"


def step_stack(out, no_start):
    if not no_start:
        run(["scripts/voicebot", "start", "--dev"], log=out / "stack.log", timeout=900)
    _, text = run(["scripts/voicebot", "status", "--dev"])
    up = re.search(r"^omlx\s+up", text, re.M) and re.search(r"^model\s+loaded", text, re.M)
    return ("PASS" if up else "FAIL"), ("omlx up, model loaded" if up else "omlx down or model not loaded")


def step_e2e(out):
    shutil.rmtree(E2E_SHOTS, ignore_errors=True)
    code, text = run([PY, "-m", "pytest", "-m", "e2e", "tests/e2e", "-q", "-p", "no:cacheprovider"])
    (out / "e2e.log").write_text(text)
    copy_pngs(E2E_SHOTS, out / "e2e")
    return ("PASS" if code == 0 else "FAIL"), summary_line(text)


def step_live(out):
    shutil.rmtree(E2E_SHOTS, ignore_errors=True)
    code, text = run([PY, "-m", "pytest", "-m", "e2e and live", "tests/e2e/test_app.py", "-k", "live_text_turn",
                      "-q", "-p", "no:cacheprovider"])
    (out / "live.log").write_text(text)
    copy_pngs(E2E_SHOTS, out / "live")
    last = summary_line(text)
    return ("PASS" if code == 0 and "skipped" not in last else "FAIL"), last


def step_battery(out, section, timeout):
    if not stress.INBOX.exists():
        return "SKIP", f"{stress.INBOX} missing"
    shutil.rmtree(QA_OUT, ignore_errors=True)
    qa = out / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    port = stress.free_port(8014)
    try:
        code, text = run([PY, "tests/qa/stress.py", "both", str(qa / "vault"), "--port", str(port), "--only", section],
                         log=qa / "stress.log", timeout=timeout * 40)
    except subprocess.TimeoutExpired:
        return "FAIL", "battery timed out"
    for name in ("results.json", "transcript.md"):
        if (QA_OUT / name).exists():
            shutil.copy(QA_OUT / name, qa / name)
    copy_pngs(QA_OUT, qa)
    try:
        rows = json.loads((qa / "results.json").read_text())["results"]
    except Exception:
        return "FAIL", f"exit {code}, no results.json"
    replied = sum(1 for r in rows if str(r.get("reply", "")).strip())
    return ("PASS" if code == 0 else "FAIL"), f"section {section}: {len(rows)} steps, {replied} with a reply"


def step_views(out):
    root = out / "views"
    vault = root / "vault"
    for rel, text in stress.PAGES.items():   # static seed only: this step never needs the model
        (vault / rel).parent.mkdir(parents=True, exist_ok=True)
        (vault / rel).write_text(text)
    (root / "memory").mkdir(parents=True, exist_ok=True)
    proc, url = stress.start_server(vault, stress.free_port(8015))
    try:
        for name, view in VIEWS:
            cmd = [PY, "scripts/screenshot.py", str(root / f"{name}.png"), "--url", url]
            run(cmd + (["--view", view] if view else []), log=root / f"{name}.log")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    bad = [n for n, _ in VIEWS if not (root / f"{n}.png").exists() or (root / f"{n}.png").stat().st_size <= MIN_PNG]
    return ("FAIL", "missing or tiny: " + ", ".join(bad)) if bad else ("PASS", f"{len(VIEWS)} screenshots")


def summarize(results):
    count = lambda s: sum(1 for r in results if r["status"] == s)  # noqa: E731
    lines = [f"{r['step']:<8}{r['status']:<5} {r['detail']} ({r['seconds']}s)" for r in results]
    lines.append(f"smoke: {count('PASS')} pass, {count('FAIL')} fail, {count('SKIP')} skip")
    return "\n".join(lines), 1 if count("FAIL") else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", default="1,2,3,4,5")
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--section", default="A")
    ap.add_argument("--answer-timeout", type=int, default=300)
    a = ap.parse_args()
    out = pathlib.Path(a.out or f"/tmp/smoke/{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    wanted = {int(s) for s in a.steps.split(",")}
    model_up = [None]

    def llm(fn, *args):
        return ("SKIP", "model down") if model_up[0] is False else fn(*args)

    def stack():
        res = step_stack(out, a.no_start)
        model_up[0] = res[0] == "PASS"
        return res

    plan = [(1, "stack", stack), (2, "e2e", lambda: step_e2e(out)), (3, "live", lambda: llm(step_live, out)),
            (4, "battery", lambda: llm(step_battery, out, a.section, a.answer_timeout)),
            (5, "views", lambda: step_views(out))]
    if 1 not in wanted:   # LLM steps without a stack check: probe status only
        model_up[0] = step_stack(out, True)[0] == "PASS" if wanted & {3, 4} else None
    results = []
    for n, name, fn in plan:
        if n not in wanted:
            continue
        print(f"[smoke] {name} ...")
        t0 = time.time()
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = "FAIL", f"{type(e).__name__}: {e}"[:200]
        results.append({"step": name, "status": status, "detail": detail, "seconds": round(time.time() - t0)})
        print(f"[smoke] {name}: {status} {detail}")
    text, code = summarize(results)
    (out / "RESULT.json").write_text(json.dumps(results, indent=1))
    (out / "RESULT.txt").write_text(text + "\n")
    print(text)
    print(f"out: {out}")
    sys.exit(code)


if __name__ == "__main__":
    main()
