"""`python -m vault reindex`: rebuild the vault's search index (PRD 53; was `python -m vault reindex`).

Refused (exit 2) while the app runs, because swapping the file under a running app would orphan its open handle.
"""
import os
import sys

from dotenv import dotenv_values, load_dotenv

from core import config
from core import util
from vault import search, store


def cli_vault(env_file: str) -> str:
    """The vault this CLI works on: WIKI_PATH from the environment, else from `env_file`
    (the checkout's `.env`, so a dev checkout targets its dev vault), else the default."""
    value = os.environ.get("WIKI_PATH") or dotenv_values(env_file).get("WIKI_PATH") or "~/llm-wiki-dev"
    return os.path.expanduser(value)


def main(argv) -> int:
    if list(argv) != ["reindex"]:
        print("usage: python -m vault reindex", file=sys.stderr)
        return 2
    running = util.running_app()
    if running:
        print(f"refusing to reindex: {running}", file=sys.stderr)
        return 2
    print(f"reindexed {store.WIKI_PATH} -> {search.reindex()}")
    return 0


if __name__ == "__main__":
    _env = os.path.join(config.ROOT, ".env")
    load_dotenv(_env)         # the app's ports for the running-app guard
    store.WIKI_PATH = cli_vault(_env)
    sys.exit(main(sys.argv[1:]))
