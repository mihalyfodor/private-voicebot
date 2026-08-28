"""Text-only evaluation harness for the LLM + memory pipeline.

Runs scripted, multi-session conversations against llm.ask()/llm.save_memory() with
shortmem.txt and settings.json redirected to a sandbox, then scores "probe" turns by
keyword hit and reports what memory retained.

Usage:
    python -m eval.harness eval/scripts/basic.yaml --out eval/results/run.json
    python -m eval.harness eval/scripts/basic.yaml --rounds 2 --keep-memory /tmp/mem.txt
    python -m eval.harness --chat
"""
import argparse
import difflib
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import avatars  # noqa: E402
import llm  # noqa: E402
import memory  # noqa: E402
from splitter import strip_tag  # noqa: E402

CATEGORIES = (
    "single-hop",
    "multi-session",
    "update",
    "temporal",
    "abstention",
    "preference",
)

DUP_RATIO = 0.85


# --------------------------------------------------------------------------- sandbox


class Sandbox:
    """Redirect memory/settings/log to throwaway paths so evals never touch the real files.

    ``memory_path`` (optional) persists memory across runs; otherwise a temp file is used
    and discarded on exit. Tools are disabled by default (``llm.TOOLS = []``) so probes
    never trigger weather/news calls.
    """

    def __init__(self, memory_path: str | None = None, tools: bool = False):
        self.memory_path = memory_path
        self.tools = tools
        self._tmpdir: str | None = None
        self._saved: dict = {}

    def __enter__(self) -> "Sandbox":
        self._tmpdir = tempfile.mkdtemp(prefix="memeval-")
        self.log_path = os.path.join(self._tmpdir, "session.log")
        self.settings_path = os.path.join(self._tmpdir, "settings.json")
        if self.memory_path is None:
            self.memory_path = os.path.join(self._tmpdir, "shortmem.txt")
        else:
            self.memory_path = os.path.abspath(self.memory_path)
            os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)

        self._saved = {
            "shortmem": memory.SHORTMEM_PATH,
            "settings": avatars.SETTINGS_PATH,
            "tools": llm.TOOLS,
            "log": llm._log,
            "log_path": llm.LOG_PATH,
            "avatar": avatars._current,
            "system_prompt": llm.SYSTEM_PROMPT,
        }

        memory.SHORTMEM_PATH = self.memory_path
        avatars.SETTINGS_PATH = self.settings_path
        avatars._current = None
        if not self.tools:
            llm.TOOLS = []
        llm.LOG_PATH = self.log_path
        llm._log = self._quiet_log
        llm.SYSTEM_PROMPT = llm.build_system_prompt(avatars.current())
        return self

    def _quiet_log(self, user_text: str, tool_calls: list, reply: str) -> None:
        line = "%s | user: %r | tools: %s | reply_len: %d\n" % (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            user_text,
            ",".join(tc["name"] for tc in tool_calls) if tool_calls else "none",
            len(reply),
        )
        with open(self.log_path, "a") as f:
            f.write(line)

    def read_memory(self) -> str:
        if not os.path.exists(self.memory_path):
            return ""
        with open(self.memory_path) as f:
            return f.read()

    def __exit__(self, *exc) -> bool:
        memory.SHORTMEM_PATH = self._saved["shortmem"]
        avatars.SETTINGS_PATH = self._saved["settings"]
        llm.TOOLS = self._saved["tools"]
        llm._log = self._saved["log"]
        llm.LOG_PATH = self._saved["log_path"]
        avatars._current = self._saved["avatar"]
        llm.SYSTEM_PROMPT = self._saved["system_prompt"]
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
        return False


# --------------------------------------------------------------------------- session


class Session:
    """One simulated conversation: start() -> say()* -> end()."""

    def __init__(self, sandbox: Sandbox | None = None):
        self.sandbox = sandbox
        self.turns: list = []

    def _memory_text(self) -> str:
        if self.sandbox is not None:
            return self.sandbox.read_memory()
        if not os.path.exists(memory.SHORTMEM_PATH):
            return ""
        with open(memory.SHORTMEM_PATH) as f:
            return f.read()

    def start(self) -> "Session":
        llm.reset()
        self.turns = []
        return self

    def say(self, text: str) -> str:
        reply = strip_tag(llm.ask(text) or "").strip()
        self.turns.append({"user": text, "reply": reply})
        return reply

    def end(self) -> str:
        """save_memory(), returning whatever was appended to shortmem.txt."""
        before = self._memory_text()
        llm.save_memory()
        after = self._memory_text()
        return _appended(before, after)


def _appended(before: str, after: str) -> str:
    """The text added to `before` to reach `after` (falls back to a line diff)."""
    if after.startswith(before):
        return after[len(before):]
    added = [
        line[2:]
        for line in difflib.ndiff(before.splitlines(), after.splitlines())
        if line.startswith("+ ")
    ]
    return "\n".join(added)


# --------------------------------------------------------------------------- scoring


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())


def _groups(expect) -> list:
    """Normalise `expect` into a list of any-of groups.

    A flat list of strings is one any-of group; a list of lists is several groups that
    must all hit. A bare string is a single required keyword.
    """
    if expect is None:
        return []
    if isinstance(expect, str):
        return [[expect]]
    if all(isinstance(e, str) for e in expect):
        return [list(expect)]
    out = []
    for e in expect:
        out.append([e] if isinstance(e, str) else list(e))
    return out


def _hit(reply: str, keyword: str) -> bool:
    return _norm(keyword).strip() in _norm(reply)


def score_probe(reply: str, expect=None, expect_not=None) -> dict:
    """Keyword scoring: every `expect` group must hit, no `expect_not` keyword may."""
    groups = _groups(expect)
    missing, hits = [], []
    for group in groups:
        matched = [k for k in group if _hit(reply, k)]
        if matched:
            hits.extend(matched)
        else:
            missing.append(group)
    forbidden = [k for k in (expect_not or []) if _hit(reply, k)]
    return {
        "passed": not missing and not forbidden,
        "hits": hits,
        "missing": missing,
        "forbidden_hits": forbidden,
    }


def category_rates(probes: list) -> dict:
    rates = {}
    for p in probes:
        cat = p.get("category") or "uncategorised"
        entry = rates.setdefault(cat, {"passed": 0, "total": 0})
        entry["total"] += 1
        entry["passed"] += 1 if p["passed"] else 0
    for entry in rates.values():
        entry["rate"] = round(entry["passed"] / entry["total"], 3) if entry["total"] else 0.0
    return rates


# --------------------------------------------------------------------------- memory stats


def _fact_lines(text: str) -> list:
    """Memory lines that carry facts (drops blanks and `--- timestamp ---` headers)."""
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or re.fullmatch(r"-{2,}.*-{2,}", s):
            continue
        out.append(s)
    return out


def duplicate_lines(text: str, ratio: float = DUP_RATIO) -> list:
    """Lines that are exact or near duplicates (difflib ratio > `ratio`) of an earlier line."""
    lines = _fact_lines(text)
    dups = []
    for i, line in enumerate(lines):
        a = _norm(line).strip()
        if not a:
            continue
        for j in range(i):
            b = _norm(lines[j]).strip()
            if not b:
                continue
            if a == b or difflib.SequenceMatcher(None, a, b).ratio() > ratio:
                dups.append({"line": line, "duplicate_of": lines[j]})
                break
    return dups


def memory_stats(text: str) -> dict:
    facts = _fact_lines(text)
    dups = duplicate_lines(text)
    return {
        "content": text,
        "lines": len(text.splitlines()),
        "fact_lines": len(facts),
        "tokens_est": len(text) // 4,
        "duplicates": len(dups),
        "duplicate_pairs": dups,
    }


# --------------------------------------------------------------------------- runner


def load_script(path: str) -> list:
    with open(path) as f:
        raw = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)
    if isinstance(data, dict):
        data = data.get("sessions", [])
    return data


def run_script(script, memory_path: str | None = None, rounds: int = 1,
               tools: bool = False, sandbox: Sandbox | None = None,
               verbose: bool = True) -> dict:
    """Replay `script` (list of sessions of turns) `rounds` times against llm + memory."""
    if isinstance(script, (str, os.PathLike)):
        script = load_script(str(script))

    owns_sandbox = sandbox is None
    box = sandbox or Sandbox(memory_path=memory_path, tools=tools)
    if owns_sandbox:
        box.__enter__()
    try:
        results = {"rounds": [], "memory": {}}
        for r in range(1, rounds + 1):
            round_result = {"round": r, "probes": [], "sessions": []}
            for s_index, session_turns in enumerate(script, start=1):
                sess = Session(box).start()
                record = {"session": s_index, "turns": [], "memory_delta": ""}
                for turn in session_turns or []:
                    if "probe" in turn:
                        text = turn["probe"]
                        reply = sess.say(text)
                        score = score_probe(reply, turn.get("expect"), turn.get("expect_not"))
                        probe = {
                            "round": r,
                            "session": s_index,
                            "probe": text,
                            "category": turn.get("category", "uncategorised"),
                            "reply": reply,
                            **score,
                        }
                        round_result["probes"].append(probe)
                        record["turns"].append(probe)
                        if verbose:
                            mark = "PASS" if score["passed"] else "FAIL"
                            print(f"  [{mark}] ({probe['category']}) {text}")
                            print(f"         -> {reply[:160]}")
                    else:
                        text = turn.get("user", "")
                        reply = sess.say(text)
                        record["turns"].append({"user": text, "reply": reply})
                        if verbose:
                            print(f"  user: {text}")
                            print(f"         -> {reply[:160]}")
                record["memory_delta"] = sess.end()
                if verbose:
                    print(f"  [session {s_index} memory delta]\n{record['memory_delta'].strip()}\n")
                round_result["sessions"].append(record)
            probes = round_result["probes"]
            round_result["categories"] = category_rates(probes)
            passed = sum(1 for p in probes if p["passed"])
            round_result["overall"] = {
                "passed": passed,
                "total": len(probes),
                "rate": round(passed / len(probes), 3) if probes else 0.0,
            }
            round_result["memory"] = memory_stats(box.read_memory())
            results["rounds"].append(round_result)
        results["memory"] = memory_stats(box.read_memory())
        results["memory_path"] = box.memory_path
        return results
    finally:
        if owns_sandbox:
            box.__exit__(None, None, None)


# --------------------------------------------------------------------------- reporting


def format_report(results: dict) -> str:
    lines = []
    for rnd in results["rounds"]:
        o = rnd["overall"]
        lines.append(f"round {rnd['round']}: {o['passed']}/{o['total']} probes passed ({o['rate']:.0%})")
        lines.append(f"  {'category':<16}{'pass':>6}{'total':>7}{'rate':>8}")
        for cat, e in sorted(rnd["categories"].items()):
            lines.append(f"  {cat:<16}{e['passed']:>6}{e['total']:>7}{e['rate']:>8.0%}")
        fails = [p for p in rnd["probes"] if not p["passed"]]
        for p in fails:
            reason = []
            if p["missing"]:
                reason.append("missing " + "; ".join("|".join(g) for g in p["missing"]))
            if p["forbidden_hits"]:
                reason.append("stale " + ", ".join(p["forbidden_hits"]))
            lines.append(f"  FAIL ({p['category']}) {p['probe']} -- {', '.join(reason)}")
            lines.append(f"       {p['reply'][:200]}")
        m = rnd["memory"]
        lines.append(
            f"  memory: {m['lines']} lines ({m['fact_lines']} facts), "
            f"~{m['tokens_est']} tokens, {m['duplicates']} duplicate lines"
        )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- interactive


def chat(memory_path: str | None = None, tools: bool = False) -> None:
    with Sandbox(memory_path=memory_path, tools=tools) as box:
        print(f"[chat] memory: {box.memory_path}  tools: {'on' if tools else 'off'}")
        print("[chat] /end saves the session, /mem prints memory, /quit exits.")
        sess = Session(box).start()
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text == "/quit":
                break
            if text == "/mem":
                print(box.read_memory() or "[empty]")
                continue
            if text == "/end":
                delta = sess.end()
                print("[memory delta]\n" + (delta.strip() or "[nothing new]"))
                sess = Session(box).start()
                continue
            print(sess.say(text))


# --------------------------------------------------------------------------- cli


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="eval.harness", description=__doc__)
    ap.add_argument("script", nargs="?", help="scenario YAML/JSON")
    ap.add_argument("--out", help="write full results JSON here")
    ap.add_argument("--keep-memory", help="memory file to reuse across runs")
    ap.add_argument("--rounds", type=int, default=1, help="replay the script N times (memory persists)")
    ap.add_argument("--tools", action="store_true", help="leave tools enabled (default: disabled)")
    ap.add_argument("--chat", action="store_true", help="interactive REPL instead of a script")
    ap.add_argument("--quiet", action="store_true", help="do not stream turns while running")
    args = ap.parse_args(argv)

    if args.chat:
        chat(memory_path=args.keep_memory, tools=args.tools)
        return 0
    if not args.script:
        ap.error("a script path is required (or use --chat)")

    results = run_script(
        args.script,
        memory_path=args.keep_memory,
        rounds=args.rounds,
        tools=args.tools,
        verbose=not args.quiet,
    )
    results["script"] = args.script
    print(format_report(results))
    print("final memory:\n" + (results["memory"]["content"].strip() or "[empty]"))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[wrote {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
