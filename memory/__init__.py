"""Memory v2 — a structured user profile with LLM-proposed ops applied in Python (docs/15-memory-v2.md).

PRD 50 step 8: the old memory.py is the package `memory` (store, render, identity, ops) plus `agent/fold.py`
(propose / reflect / save). This file keeps `memory.<name>` readable for tests; assigning a moved name raises, because
a patch here would not reach the module that defines it: patch `memory.<module>.<name>`. `memory.render` is the
submodule, so call `memory.render.render(profile)`. Deleted with the other shims at step 12.
"""
import importlib as _importlib
import sys as _sys
import types as _types

_MODULES = ("memory.store", "memory.render", "memory.identity", "memory.ops", "agent.fold")
_OWNERS = {}


def _owners():
    if not _OWNERS:
        found = {}
        for path in _MODULES:      # agent.fold imports lazily: memory itself may not import an upper layer at load
            module = _importlib.import_module(path)
            for name, obj in vars(module).items():
                if not isinstance(obj, _types.ModuleType) and (not callable(obj) or getattr(obj, "__module__", path) == path):
                    found.setdefault(name, module)
        _OWNERS.update(found)
    return _OWNERS


def __getattr__(name):
    if name.startswith("__") or name in ("store", "render", "identity", "ops"):
        raise AttributeError(name)          # the import system probing for a submodule
    module = _owners().get(name)
    if module is None:
        raise AttributeError(f"module 'memory' has no attribute {name!r}")
    return getattr(module, name)


class _Forwarding(_types.ModuleType):
    def __setattr__(self, name, value):
        if not isinstance(value, _types.ModuleType) and name not in vars(self) and name in _owners():
            raise AttributeError(f"memory.{name} moved: patch the module that defines it")
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Forwarding
