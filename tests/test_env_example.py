"""`.env.example` documents every environment variable the code reads, and nothing stale.

Forward check: every key read via os.getenv("X") / os.environ.get("X") / os.environ["X"] in a *.py
outside tests/, docs/ and the venvs must appear in .env.example as `X=` or `# X=`.
Reverse check: every key in .env.example must be read: matched by READ_RE in some *.py (tests included, so the
"test knobs" block at the end stays valid), or used as `${X` / `$X` / `X=` / `X:` in scripts/*.sh, scripts/voicebot,
docker-compose.yml, the Dockerfile or a launchd plist. Mentions in comments or docstrings do not count.
Shell scripts are only used for the reverse check (their ${X:-...} locals are mostly internal).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", "venv", "node_modules", "tests", "docs", ".git", "__pycache__"}

#: Read by the code but deliberately not documented: the OS search path.
IGNORED = {"PATH"}

#: os.getenv("X") / os.environ.get("X") / os.environ["X"], plus reads through helpers: any call whose
#: name contains "env" (`_env_float("X"`, `env_flag("X"`, `env.get("X"`) with the key as first argument.
READ_RE = re.compile(
    r"""os\.(?:getenv\(|environ\.get\(|environ\[)\s*["']([A-Z][A-Z0-9_]*)["']"""
    r"""|\b\w*(?i:env)\w*(?:\.get)?\(\s*["']([A-Z][A-Z0-9_]*)["']""",
)
KEY_RE = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=", re.M)


def _sources(suffixes, skip=SKIP_DIRS):
    for p in ROOT.rglob("*"):
        if p.suffix in suffixes and not (set(p.relative_to(ROOT).parts) & skip):
            yield p


def _documented():
    return set(KEY_RE.findall((ROOT / ".env.example").read_text()))


def test_every_env_read_is_documented():
    read = set()
    for p in _sources({".py"}):
        read |= {k for m in READ_RE.findall(p.read_text()) for k in m if k}
    missing = sorted(read - IGNORED - _documented())
    assert not missing, f"read by the code but missing from .env.example: {missing}"


def _read_sites():
    """Text of the non-Python files that may read a key, plus the keys READ_RE finds in any .py (tests included)."""
    read = set()
    for p in _sources({".py"}, SKIP_DIRS - {"tests"}):
        read |= {k for m in READ_RE.findall(p.read_text()) for k in m if k}
    files = [*(ROOT / "scripts").glob("*.sh"), ROOT / "scripts" / "voicebot", ROOT / "docker-compose.yml",
             ROOT / "Dockerfile", *(ROOT / "scripts" / "launchd").glob("*.plist")]
    return read, "\n".join(f.read_text() for f in files if f.is_file())


def test_no_stale_keys_in_env_example():
    read, text = _read_sites()
    stale = sorted(k for k in _documented()
                   if k not in read and not re.search(rf"\$\{{?{k}\b|\b{k}[=:]|<string>{k}</string>", text))
    assert not stale, f"in .env.example but never read: {stale}"
