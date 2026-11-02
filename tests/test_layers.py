"""PRD 50 scenarios 1 and 10: the layer ratchet. App code imports only downward and never a deleted root module; `ALLOWED` is empty."""
import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = ("core", "vault", "memory", "ingest", "tools", "agent", "server")   # bottom -> top; "root" sits above all
SHIMS: set = set()     # root modules that only re-export a new home (none left after step 12); app code may not import them
# The root modules PRD 50 moved or deleted: importing one of these names (any layer) fails, whether or not a file is left.
DELETED = {"tracing", "untrusted", "vault_lock", "sources", "notebooks", "batch_import", "index_db", "wiki", "llm", "chatbot",
           "runs", "approvals", "research", "remarkable", "avatars", "characters", "session", "services", "asr", "vad",
           "backdrops", "splitter", "fillers"}
SKIP_DIRS = {"tests", "scripts", "eval", ".venv", "docs", "static", ".claude", "__pycache__", "logs", "characters"}

# (importer path, imported top-level module) pairs still pointing upward. Keyed on the module, not its layer, so a
# new upward import in the same file fails. Empty since PRD 50 step 12; a new entry needs a reason in the PRD.
ALLOWED: set = set()


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


def _relative_names(rel: str, node: ast.ImportFrom) -> list:
    """Absolute module names a relative import points at. Level 1 is the module's own package; each extra level goes one up."""
    base = rel.split("/")[:-1]
    base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
    if node.level - 1 > len(rel.split("/")[:-1]):
        return []   # climbs out of the repo: not an app import
    if node.module:
        return [".".join(base + [node.module])]
    return [".".join(base + [a.name]) for a in node.names]


def _literal(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value else None


def _dynamic_target(node: ast.Call):
    """The literal module name of importlib.import_module("x") / import_module(name="x") / __import__("x"), else None
    (a non-literal can't be known). A relative literal (".x", "..x") resolves against a literal package (second
    positional or package=): level 1 is the package itself, each extra dot one level up. Without a package it is None."""
    f = node.func
    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
    if name not in ("import_module", "__import__"):
        return None
    kw = {k.arg: k.value for k in node.keywords}
    target = _literal(node.args[0] if node.args else kw.get("name"))
    if target is None or not target.startswith("."):
        return target
    package = _literal(node.args[1] if len(node.args) > 1 else kw.get("package"))
    if name != "import_module" or not package:
        return None
    dots = len(target) - len(target.lstrip("."))
    parts = package.split(".")
    if dots - 1 >= len(parts):
        return None
    return ".".join(parts[:len(parts) - (dots - 1)] + [target[dots:]])


def check(root: str, allowed: set, shims=SHIMS, deleted=DELETED) -> list:
    problems, observed = [], set()
    for rel, layer in _modules(root, shims):
        with open(os.path.join(root, rel)) as f:
            tree = ast.parse(f.read(), rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif isinstance(node, ast.ImportFrom) and node.level > 0:
                names = _relative_names(rel, node)
            elif isinstance(node, ast.Call) and (dyn := _dynamic_target(node)):
                names = [dyn]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in shims:
                    problems.append(f"{rel}:{node.lineno}: imports shim {top}")
                    continue
                if top in deleted:
                    problems.append(f"{rel}:{node.lineno}: imports deleted module {top}")
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


def test_relative_upward_import_is_named_with_line(tmp_path):
    root = _tree(tmp_path, {"vault/__init__.py": "", "vault/store.py": "x = 1\nfrom ..server import app\n",
                            "vault/other.py": "from .. import server\n", "server/__init__.py": "", "server/app.py": ""})
    found = check(root, set())
    assert len(found) == 2 and found[0].startswith("vault/other.py:1:") and found[1].startswith("vault/store.py:2:")


def test_relative_sibling_import_is_allowed(tmp_path):
    root = _tree(tmp_path, {"vault/__init__.py": "from . import store\nfrom .store import f\n", "vault/store.py": "from . import other\n",
                            "vault/other.py": "", "server/__init__.py": "", "server/app.py": "from ..vault import store\n"})
    assert check(root, set()) == []


def test_importlib_literal_upward_import_is_named_with_line(tmp_path):
    root = _tree(tmp_path, {"vault/__init__.py": "", "server/__init__.py": "", "server/app.py": "",
                            "vault/store.py": "import importlib\n\ndef f(n):\n    importlib.import_module(n)\n    return importlib.import_module(\"server.app\")\n"})
    found = check(root, set())
    assert len(found) == 1 and found[0].startswith("vault/store.py:5:")


def test_dunder_import_literal_upward_import_is_named_with_line(tmp_path):
    root = _tree(tmp_path, {"vault/__init__.py": "", "server/__init__.py": "", "server/app.py": "",
                            "vault/store.py": "def f(n):\n    __import__(n)\n    return __import__(\"server\")\n"})
    found = check(root, set())
    assert len(found) == 1 and found[0].startswith("vault/store.py:3:")


def test_import_module_keyword_name_upward_import_is_named_with_line(tmp_path):
    root = _tree(tmp_path, {"vault/__init__.py": "", "server/__init__.py": "", "server/app.py": "",
                            "vault/store.py": "import importlib\n\ndef f():\n    return importlib.import_module(name=\"server\")\n"})
    found = check(root, set())
    assert len(found) == 1 and found[0].startswith("vault/store.py:4:")


def test_import_module_relative_literal_with_package_upward_import_is_named(tmp_path):
    root = _tree(tmp_path, {"vault/__init__.py": "", "server/__init__.py": "", "server/app.py": "",
                            "vault/store.py": "import importlib\n\ndef f():\n    return importlib.import_module(\".app\", \"server\")\n",
                            "vault/other.py": "import importlib\nimportlib.import_module(\".app\", package=\"server\")\n"})
    found = check(root, set())
    assert len(found) == 2 and found[0].startswith("vault/other.py:2:") and found[1].startswith("vault/store.py:4:")


def test_stale_allow_list_entry_fails():
    assert any("stale allow-list entry" in p for p in check(ROOT, ALLOWED | {("core/config.py", "chatbot")}))


def test_shim_import_fails(tmp_path):
    root = _tree(tmp_path, {"tracing.py": "", "foo.py": "import tracing\n"})
    found = check(root, set(), shims={"tracing"}, deleted=set())
    assert len(found) == 1 and "shim tracing" in found[0]


def test_allow_list_and_shims_are_empty():
    assert ALLOWED == set() and SHIMS == set()


def test_importing_a_deleted_root_module_fails(tmp_path):
    """PRD 50 scenario 10: llm / wiki / sources / chatbot (and the rest) are gone; importing one names the line."""
    root = _tree(tmp_path, {"server/__init__.py": "", "server/app.py": "import llm\nfrom wiki import search\n\ndef f():\n    import sources\n",
                            "agent/__init__.py": "", "agent/x.py": "import chatbot\n"})
    found = check(root, set())
    assert [p.split(": ", 1)[1] for p in found] == ["imports deleted module chatbot", "imports deleted module llm",
                                                      "imports deleted module wiki", "imports deleted module sources"]
    assert found[1].startswith("server/app.py:1:") and found[3].startswith("server/app.py:5:")


def _script_files():
    files = [os.path.join(ROOT, n) for n in ("Dockerfile", "docker-compose.yml")]
    for d, _, names in os.walk(os.path.join(ROOT, "scripts")):
        files += [os.path.join(d, n) for n in names if n.endswith((".py", ".sh"))]
    return [f for f in files if os.path.exists(f)]


def test_scripts_do_not_reference_deleted_modules():
    """scripts/ may import anything that exists, but not a deleted root module (a heredoc in fetch_assets.sh did)."""
    import re
    names = "|".join(sorted(DELETED))
    pat = re.compile(rf"^\s*(?:import\s+(?:{names})\b|from\s+(?:{names})\s+import\b)|python3?\s+-m\s+(?:{names})\b", re.M)
    bad = []
    for f in _script_files():
        for m in pat.finditer(open(f, encoding="utf-8", errors="replace").read()):
            bad.append(f"{os.path.relpath(f, ROOT)}: {m.group(0).strip()}")
    assert not bad, bad


def _app_modules():
    mods = []
    for layer in LAYERS:
        for n in sorted(os.listdir(os.path.join(ROOT, layer))):
            if n.endswith(".py") and n != "__init__.py":
                mods.append(f"{layer}.{n[:-3]}" if n != "__main__.py" else None)
            elif os.path.isfile(os.path.join(ROOT, layer, n, "__init__.py")):
                mods.append(f"{layer}.{n}")
    return [m for m in mods if m]


def test_every_module_imports_fresh():
    """Each app module imports alone in a new interpreter (import cycles that only bite on a given entry order)."""
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    def run(mod):
        r = subprocess.run([sys.executable, "-c", f"import {mod}"], cwd=ROOT, capture_output=True, text=True)
        return mod, r.returncode, r.stderr.strip().splitlines()[-1:] 

    with ThreadPoolExecutor(8) as ex:
        failed = [(m, e) for m, rc, e in ex.map(run, _app_modules()) if rc]
    assert not failed, failed
