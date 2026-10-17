"""Shim (PRD 50): the chat agent lives in agent/ (prompt, context, recall, offers, turn, written, conversation).

Names stay readable here (``llm.ask(...)``) for tests; assigning one raises, because a patch on this copy would not
reach the module that defines it: patch ``agent.<module>.<name>``. Deleted with the other shims at step 12.
"""
import sys as _sys
import types as _types

from agent import conversation as _conversation_mod      # first: it loads the others in a safe order, then resets
from agent import context as _context, offers as _offers, prompt as _prompt
from agent import recall as _recall, turn as _turn, written as _written

_MODULES = (_prompt, _context, _recall, _offers, _turn, _written, _conversation_mod)


def _build():
    owners = {}
    for module in _MODULES:
        for name, obj in vars(module).items():
            if not isinstance(obj, _types.ModuleType) and (not callable(obj) or getattr(obj, "__module__", module.__name__) == module.__name__):
                owners.setdefault(name, module)
    return owners


_OWNERS = _build()      # fixed at import, so a patched value still forwards


def __getattr__(name):
    module = _OWNERS.get(name)
    if module is None:
        raise AttributeError(f"module 'llm' has no attribute {name!r}")
    return getattr(module, name)


class _Forwarding(_types.ModuleType):
    def __setattr__(self, name, value):
        if name not in vars(self) and name in _OWNERS:
            raise AttributeError(f"llm.{name} moved to agent/: patch the module that defines it")
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Forwarding
