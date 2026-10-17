"""PRD 50 scenario 1: the layer ratchet. App code imports only downward; `ALLOWED` (importer, imported module) may only shrink."""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = ("core", "vault", "memory", "ingest", "tools", "agent", "server")   # bottom -> top; "root" sits above all
SHIMS = {"tracing", "untrusted", "vault_lock", "sources", "notebooks", "batch_import", "index_db", "wiki", "llm", "chatbot", "runs", "approvals", "research", "remarkable", "avatars", "characters", "session", "services", "asr", "vad", "backdrops", "splitter", "fillers"}    # root modules that only re-export a new home; app code may not import them
SKIP_DIRS = {"tests", "scripts", "eval", ".venv", "docs", "static", ".claude", "__pycache__", "logs", "characters"}

# (importer path, imported top-level module) pairs still pointing upward. Keyed on the module, not its layer, so a
# new upward import in the same file fails. May only shrink (docs/50-modular-layout.md).
ALLOWED = {
    # vault/ modules still import memory (a layer above them: tokens_est, _today_iso, user-page guards); step 12 moves
    # those helpers down
    ("vault/store.py", "memory"),
    ("vault/index.py", "memory"),
    ("vault/pages.py", "memory"),
    ("vault/search.py", "memory"),
    ("vault/digest.py", "memory"),
    ("vault/ingest.py", "memory"),
    ("vault/bridge.py", "memory"),
    ("vault/focus.py", "agent"),     # step 12: `tools_registered` reads the live agent.prompt.TOOLS list (the eval harness empties it)
    # step 7: vault code reading ingest's source pages / the batch guard (the PRD's expected vault -> ingest pairs)
    ("vault/chunks.py", "ingest"),
    ("vault/digest.py", "ingest"),
    ("vault/bridge.py", "ingest"),
}


def _rank(layer: str) -> int:
    return LAYERS.index(layer) if layer in LAYERS else len(LAYERS)


def _modules(root: str, shims) -> list:
    """(relpath, layer) of every app module: root-level *.py (minus shims) and everything under a layer folder."""
    found = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if name.endswith(".py") and os.path.isfile(path) and name[:-3] not in shims:
            found.append((name, "root"))
    for layer in LAYERS:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, layer)):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if fn.endswith(".py"):
                    found.append((os.path.relpath(os.path.join(dirpath, fn), root), layer))
    return found


def check(root: str, allowed: set, shims=SHIMS) -> list:
    problems, observed = [], set()
    for rel, layer in _modules(root, shims):
        with open(os.path.join(root, rel)) as f:
            tree = ast.parse(f.read(), rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in shims:
                    problems.append(f"{rel}:{node.lineno}: imports shim {top}")
                    continue
                if top in LAYERS and os.path.isdir(os.path.join(root, top)):
                    target = top
                elif os.path.isfile(os.path.join(root, top + ".py")):
                    target = "root"
                else:
                    continue
                if _rank(target) > _rank(layer):
                    if (rel, top) in allowed:
                        observed.add((rel, top))
                    else:
                        problems.append(f"{rel}:{node.lineno}: {layer} imports {target} ({name})")
    problems += [f"stale allow-list entry: {pair}" for pair in sorted(allowed - observed)]
    return sorted(problems)


def _tree(tmp_path, files: dict) -> str:
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return str(tmp_path)


def test_real_tree_is_clean():
    problems = check(ROOT, ALLOWED)
    assert problems == [], "\n".join(problems)


def test_function_local_upward_import_is_named_with_line(tmp_path):
    root = _tree(tmp_path, {"core/__init__.py": "", "vault/__init__.py": "", "vault/store.py": "def f():\n    import server\n",
                            "server/__init__.py": "", "server/app.py": ""})
    found = check(root, set())
    assert len(found) == 1 and found[0].startswith("vault/store.py:2:")


def test_stale_allow_list_entry_fails():
    assert any("stale allow-list entry" in p for p in check(ROOT, ALLOWED | {("core/config.py", "chatbot")}))


def test_shim_import_fails(tmp_path):
    root = _tree(tmp_path, {"tracing.py": "", "foo.py": "import tracing\n"})
    found = check(root, set())
    assert len(found) == 1 and "shim tracing" in found[0]
