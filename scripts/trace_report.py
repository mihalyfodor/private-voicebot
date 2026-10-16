#!/usr/bin/env python3
"""Read events.jsonl (docs/49-event-trace.md): one row per turn and ingest, then the errors.

    scripts/trace_report.py <events.jsonl> [--since 24h] [--turn ID|pid:ID] [--ingest ID] [--errors]

Reads the rotated `.5` … `.1` files too. A turn is keyed `(pid, turn_id)` (ids restart per process);
`--turn` / `--ingest` print that unit's events one per line, ordered by `start` (spans) or `ts`;
`--errors` prints only the error list and exits 1 when it is non-empty. Stdlib + the repo's
`runs.read` (the same reader the run pages use).
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from runs import read  # noqa: E402 — the events reader lives with the run pages (docs/51-run-cards.md)

UNITS = {"m": "minutes", "h": "hours", "d": "days"}
DASH = "–"


def since(rows: list, spec: str) -> list:
    m = re.fullmatch(r"(\d+)([mhd])", spec or "")
    if not m:
        sys.exit(f"--since wants <n>m|h|d, not {spec!r}")
    cutoff = datetime.now().astimezone() - timedelta(**{UNITS[m.group(2)]: int(m.group(1))})
    return [r for r in rows if _when(r.get("ts")) and _when(r["ts"]) >= cutoff]


def _when(ts):
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def order(row: dict) -> str:
    return row.get("start") or row.get("ts") or ""


def units(rows: list) -> tuple:
    """``(turns, ingests)``: lists of ``(key, [events])``; a second `turn_start` on the same key
    (a reconnect restarted the ids) opens a new unit."""
    turns, ingests, open_turns, open_ingests = [], [], {}, {}
    for r in rows:
        if r.get("turn_id") is not None and not r.get("ingest_id"):
            key = f"{r.get('pid')}:{r['turn_id']}"
            if r["kind"] == "turn_start" or key not in open_turns:
                open_turns[key] = []
                turns.append((key, open_turns[key]))
            open_turns[key].append(r)
        if r.get("ingest_id"):
            key = r["ingest_id"]
            if key not in open_ingests:
                open_ingests[key] = []
                ingests.append((key, open_ingests[key]))
            open_ingests[key].append(r)
    return turns, ingests


def _first(events: list, kind: str) -> dict:
    return next((e for e in events if e["kind"] == kind), {})


def _sum(values: list):
    values = [v for v in values if isinstance(v, (int, float))]
    return sum(values) if values else DASH


def turn_row(key: str, events: list) -> list:
    start, end = _first(events, "turn_start"), _first(events, "turn_end")
    calls = sorted((e for e in events if e["kind"] == "llm_call"), key=order)
    tools = end.get("tools") or [e.get("name") for e in events if e["kind"] == "tool_call"]
    ttft = calls[0].get("ttft_ms") if calls else None
    err = any(e["kind"] == "error" for e in events) or end.get("outcome") == "error"
    return [key, (start or events[0]).get("ts", "")[:19], start.get("origin", DASH), end.get("band") or DASH,
            ",".join(tools) or DASH, f"{_sum([c.get('ms') for c in calls])} / {DASH if ttft is None else ttft}",
            f"{_sum([c.get('tokens_in') for c in calls])}/{_sum([c.get('tokens_out') for c in calls])}",
            str(end.get("reply_len", DASH)), "error" if err else ""]


def ingest_row(key: str, events: list) -> list:
    start, end = _first(events, "ingest_start"), _first(events, "ingest_end")
    rejected = end.get("ops_rejected") or []         # [{kind, reason_code}]
    return [key, start.get("ingest_kind", DASH), end.get("title") or end.get("path") or DASH,
            str(end.get("ms", DASH)), f"{end.get('ops_applied', 0)}/{len(rejected)}",
            "degraded" if end.get("degraded") else ""]


def error_row(e: dict) -> list:
    frames = [line.strip() for line in (e.get("traceback") or "").splitlines() if line.strip().startswith("File ")]
    where = e.get("where") or DASH
    return [e.get("ts", "")[:19], e.get("scope", DASH), where, e.get("exc_type", DASH), frames[-1] if frames else DASH]


def table(head: list, rows: list) -> str:
    widths = [max(len(str(c)) for c in col) for col in zip(head, *rows)]
    return "\n".join("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip() for row in [head] + rows)


def error_table(errors: list) -> str:
    if not errors:
        return "no errors"
    return table(["ts", "scope", "where", "exc", "traceback"], [error_row(e) for e in errors])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--since")
    ap.add_argument("--turn")
    ap.add_argument("--ingest")
    ap.add_argument("--errors", action="store_true")
    a = ap.parse_args(argv)
    rows = read(a.path)
    if a.since:
        rows = since(rows, a.since)
    turns, ingests = units(rows)
    errors = [r for r in rows if r.get("kind") == "error"]

    if a.turn or a.ingest:
        if a.turn:
            want = a.turn if ":" in a.turn else None
            picked = [ev for key, ev in turns if key == want or (want is None and key.split(":")[1] == a.turn)]
        else:
            picked = [ev for key, ev in ingests if key == a.ingest]
        for events in picked:
            for e in sorted(events, key=order):
                print(json.dumps(e, ensure_ascii=False))
        return 0 if picked else 1
    if a.errors:
        print(error_table(errors))
        return 1 if errors else 0

    print(table(["turn", "ts", "origin", "band", "tools", "llm ms / ttft", "tokens", "reply", "error"],
                [turn_row(k, ev) for k, ev in turns]))
    print()
    print(table(["ingest", "kind", "title", "ms", "ops", "degraded"], [ingest_row(k, ev) for k, ev in ingests]))
    print()
    print(error_table(errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
