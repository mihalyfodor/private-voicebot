"""Shim (PRD 50): the wiki layer lives in vault/ (store, index, pages, search, chunks, focus, digest, lint, ingest, bridge).

Names stay readable here (``wiki.search(...)``) for tests; assigning one raises, because a patch on this copy would not
reach the module that defines it: patch ``vault.<module>.<name>``. `python -m wiki reindex` still works from here.
"""
import os
import sys as _sys
import types as _types

from vault import bridge as _bridge, chunks as _chunks, digest as _digest, focus as _focus, index as _index
from vault import ingest as _ingest, lint as _lint, pages as _pages, search as _search, store as _store

_MODULES = (_store, _index, _pages, _search, _chunks, _focus, _digest, _lint, _ingest, _bridge)


def _build():
    owners = {}
    for module in _MODULES:
        for name, obj in vars(module).items():
            if not isinstance(obj, _types.ModuleType) and (not callable(obj) or getattr(obj, "__module__", module.__name__) == module.__name__):
                owners.setdefault(name, module)
    return owners


_OWNERS = _build()      # fixed at import, so a patched value (a lambda from a test) still forwards


def _owner(name):
    return _OWNERS.get(name)


def __getattr__(name):
    module = _owner(name)
    if module is None:
        raise AttributeError(f"module 'wiki' has no attribute {name!r}")
    return getattr(module, name)


class _Forwarding(_types.ModuleType):
    def __setattr__(self, name, value):
        if name not in ("__class__",) and name not in vars(self) and _owner(name) is not None:
            raise AttributeError(f"wiki.{name} moved to vault/: patch the module that defines it")
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Forwarding


if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    import wiki as _wiki      # the imported module, not this __main__ copy: the index handle lives in vault/

    _env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(_env)         # the app's ports for the running-app guard
    _store.WIKI_PATH = _wiki.cli_vault(_env)
    sys.exit(_wiki.main(sys.argv[1:]))
