#!/usr/bin/env python3
"""File a folder of PDF/DOCX files (a OneNote export) through the same ingest a drop uses (PRD 48).

    scripts/ingest_batch.py <folder> [--vault PATH] [--kinds pdf,docx] [--resume] [--dry-run] [--limit N] [--sleep S]

Run it with the app stopped (`scripts/voicebot stop`). The vault is `--vault`, else WIKI_PATH like the app
(dev ~/llm-wiki-dev, prod ~/docker-data/voicebot/llm-wiki); MEMORY_DIR / SETTINGS_PATH come from the same env.
Idempotent (the manifest sha guard); `<folder>/batch.log` remembers skips and failures for --resume.
Exit 0 nothing failed, 1 something failed, 2 refused start (bad args, dirty vault, app running, a dev
TRACE_PROFILE against ~/docker-data — docs/49-event-trace.md).
"""
import argparse
import os
import pathlib
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="File a folder of PDF/DOCX through the drop ingest.")
    ap.add_argument("folder")
    ap.add_argument("--vault", help="vault directory (sets WIKI_PATH); default the env")
    ap.add_argument("--kinds", default="pdf,docx", help="extensions to file (default pdf,docx)")
    ap.add_argument("--resume", action="store_true", help="skip files whose last logged state was a skip or failure")
    ap.add_argument("--dry-run", action="store_true", help="list the state of every file; write nothing, call no model")
    ap.add_argument("--limit", type=int, default=0, help="stop after N files that reached the model")
    ap.add_argument("--sleep", type=float, default=0, help="seconds to wait after each such file")
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    if args.vault:
        os.environ["WIKI_PATH"] = args.vault        # before wiki / sources are imported
    import tracing
    tracing.configure(out=sys.stderr)                # PRD 49: the app's profile rule
    import batch_import

    res = batch_import.ingest_batch(os.path.expanduser(args.folder), kinds=args.kinds.split(","),
                                    resume=args.resume, dry_run=args.dry_run, limit=args.limit,
                                    sleep=args.sleep)
    return res["exit"]


if __name__ == "__main__":
    sys.exit(main())
