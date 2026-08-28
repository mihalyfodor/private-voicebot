"""Text-only evaluation harness for the LLM + memory pipeline.

Runs scripted, multi-session conversations against llm.ask()/llm.save_memory() with
memory.json, memory_ops.jsonl and settings.json redirected to a sandbox, then scores
"probe" turns by keyword hit and reports what memory retained.

Usage:
    python -m eval.harness eval/scripts/basic.yaml --out eval/results/run.json
    python -m eval.harness eval/scripts/basic.yaml --rounds 2 --keep-memory /tmp/memory.json
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

    ``memory_path`` (optional) is the ``memory.json`` to persist across runs; the ops log and
    the v1 ``shortmem.txt`` are placed beside it. Otherwise everything lives in a temp dir and
    is discarded on exit. Tools are disabled by default (``llm.TOOLS = []``) so probes never
    trigger weather/news calls.
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
            self.memory_path = os.path.join(self._tmpdir, "memory.json")
        else:
            self.memory_path = os.path.abspath(self.memory_path)
            os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)
        mem_dir = os.path.dirname(self.memory_path) or "."
        self.ops_path = os.path.join(mem_dir, "memory_ops.jsonl")
        self.shortmem_path = os.path.join(mem_dir, "shortmem.txt")

        self._saved = {
            "memory": memory.MEMORY_PATH,
            "ops": memory.OPS_LOG_PATH,
            "shortmem": memory.SHORTMEM_PATH,
            "settings": avatars.SETTINGS_PATH,
            "tools": llm.TOOLS,
            "log": llm._log,
            "log_path": llm.LOG_PATH,
            "avatar": avatars._current,
            "system_prompt": llm.SYSTEM_PROMPT,
        }

        memory.MEMORY_PATH = self.memory_path
        memory.OPS_LOG_PATH = self.ops_path
        memory.SHORTMEM_PATH = self.shortmem_path
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
        """The memory.json text, falling back to the v1 shortmem.txt when there is no profile."""
        return _read(self.memory_path) or _read(self.shortmem_path)

    def read_ops(self) -> list:
        return _read_ops(_read(self.ops_path))

    def __exit__(self, *exc) -> bool:
        memory.MEMORY_PATH = self._saved["memory"]
        memory.OPS_LOG_PATH = self._saved["ops"]
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
        self.last_ops: list = []

    def start(self) -> "Session":
        llm.reset()
        self.turns = []
        return self

    def say(self, text: str) -> str:
        reply = strip_tag(llm.ask(text) or "").strip()
        self.turns.append({"user": text, "reply": reply})
        return reply

    def end(self) -> str:
        """save_memory(); returns the ops applied this session (v1: the shortmem delta).

        The structured ops also land in ``self.last_ops``.
        """
        before_ops = _read(memory.OPS_LOG_PATH)
        before_text = _read(memory.SHORTMEM_PATH)
        llm.save_memory()
        self.last_ops = _read_ops(_appended(before_ops, _read(memory.OPS_LOG_PATH)))
        if self.last_ops:
            return "\n".join(format_op(op) for op in self.last_ops)
        return _appended(before_text, _read(memory.SHORTMEM_PATH))


def _read(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read()


def _read_ops(text: str) -> list:
    ops = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ops.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return ops


def format_op(op: dict) -> str:
    value = op.get("value")
    value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    reason = f"  ({op['reason']})" if op.get("reason") else ""
    return f"{op.get('op')} {op.get('path')} = {value}{reason}"


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


def as_profile(text: str):
    """The v2 profile parsed out of `text`, or None when it is a v1 shortmem dump."""
    stripped = (text or "").strip()
    if not stripped.startswith("{"):
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and data.get("version") == memory.VERSION else None


def duplicate_entries(profile: dict, ratio: float = DUP_RATIO) -> list:
    """List entries whose identifying key near-duplicates an earlier one in the same list."""
    dups = []
    for section, key_field in memory.LIST_KEY.items():
        keys = [str(e.get(key_field, "")) for e in profile.get(section, []) if isinstance(e, dict)]
        for i, key in enumerate(keys):
            a = _norm(key).strip()
            if not a:
                continue
            for j in range(i):
                b = _norm(keys[j]).strip()
                if b and (a == b or difflib.SequenceMatcher(None, a, b).ratio() > ratio):
                    dups.append({"section": section, "line": key, "duplicate_of": keys[j]})
                    break
    return dups


def profile_stats(profile: dict) -> dict:
    """Section sizes, episodic count, render token estimate and duplicate entries."""
    rendered = memory.render(profile, budget_tokens=10 ** 6)
    sections = {
        section: len([k for k in profile.get(section, {}) if k != "superseded"])
        for section in memory.SCALAR_SECTIONS
    }
    sections.update({section: len(profile.get(section, [])) for section in memory.LIST_SECTIONS})
    dups = duplicate_entries(profile)
    return {
        "version": memory.VERSION,
        "render": rendered,
        "sections": sections,
        "episodic": sections["episodic"],
        "episodic_live": len(memory.live_episodics(profile)),
        "lines": len(rendered.splitlines()),
        "fact_lines": sum(sections.values()),
        "tokens_est": memory.tokens_est(rendered),
        "duplicates": len(dups),
        "duplicate_pairs": dups,
    }


def memory_stats(text: str) -> dict:
    """Stats for whichever memory format `text` holds (v2 profile JSON or v1 shortmem)."""
    profile = as_profile(text)
    if profile is not None:
        return {"content": text, **profile_stats(profile)}
    facts = _fact_lines(text)
    dups = duplicate_lines(text)
    return {
        "content": text,
        "version": 1,
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
                record["memory_ops"] = sess.last_ops
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
        if m.get("version") == 2:
            sizes = ", ".join(f"{k} {v}" for k, v in m["sections"].items() if v)
            lines.append(
                f"  memory: {m['fact_lines']} entries ({sizes or 'empty'}), "
                f"~{m['tokens_est']} tokens rendered, {m['duplicates']} duplicate entries"
            )
        else:
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
                stats = memory_stats(box.read_memory())
                print((stats.get("render") or stats["content"]).strip() or "[empty]")
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
    ap.add_argument("--keep-memory", help="memory.json to reuse across runs")
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
    final = results["memory"]
    print("final memory:\n" + ((final.get("render") or final["content"]).strip() or "[empty]"))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[wrote {args.out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
