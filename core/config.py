"""The data paths and model endpoint every layer shares, read from env once (docs/50-modular-layout.md).
Module-specific settings (ASR, TTS, Gmail, ports) stay next to their module. Modules keep their own names, copied from here."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
WIKI_PATH = os.path.expanduser(os.getenv("WIKI_PATH", "~/llm-wiki"))
MEMORY_DIR = os.getenv("MEMORY_DIR") or ROOT        # MEMORY_DIR env: e2e tests use a temp dir
# SETTINGS_PATH env overrides the location (the e2e tests point it at a temp dir)
SETTINGS_PATH = os.getenv("SETTINGS_PATH") or os.path.join(ROOT, "settings.json")
TRACE_DIR = os.path.expanduser(os.getenv("TRACE_DIR") or os.path.join(ROOT, "logs"))     # dev: repo-local, gitignored
OMLX_BASE_URL = os.getenv("OMLX_BASE_URL", "http://localhost:8000/v1")
OMLX_API_KEY = os.getenv("OMLX_API_KEY", "omlx")
OMLX_MODEL = os.getenv("OMLX_MODEL", "gemma-4-26b")
