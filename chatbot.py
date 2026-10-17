"""Entry point (PRD 50): `python chatbot.py` starts the voicebot; the code lives in server/.

For tests, names stay readable here (`chatbot.app`, `chatbot.handle_text`); assigning one raises, because a patch on this
copy would not reach the module that defines it: patch `server.<module>.<name>`. Deleted with the other shims at step 12.
"""
import sys as _sys
import types as _types

from dotenv import load_dotenv
load_dotenv()
from core import trace  # noqa: E402
if __name__ == "__main__":
    trace.configure()      # PRD 49: a dev process never starts against prod data; before anything loads

from server import app as _app, ingest as _ingest, panels as _panels, speech as _speech, turns as _turns, ws as _ws  # noqa: E402
from server.actions import chat as _chat, memory as _memory, settings as _settings, sources as _sources  # noqa: E402
from server.actions import wiki as _wiki  # noqa: E402

_MODULES = (_app, _panels, _speech, _turns, _ingest, _ws, _chat, _settings, _memory, _sources, _wiki)


def _build():
    owners = {}
    for module in _MODULES:
        for name, obj in vars(module).items():
            if isinstance(obj, _types.ModuleType):
                continue
            if isinstance(obj, (_types.FunctionType, type)) and getattr(obj, "__module__", None) != module.__name__:
                continue
            owners.setdefault(name, module)
    return owners


_OWNERS = _build()      # fixed at import, so a patched value still forwards


def __getattr__(name):
    module = _OWNERS.get(name)
    if module is None:
        raise AttributeError(f"module 'chatbot' has no attribute {name!r}")
    return getattr(module, name)


class _Forwarding(_types.ModuleType):
    def __setattr__(self, name, value):
        if name not in vars(self) and name in _OWNERS:
            raise AttributeError(f"chatbot.{name} moved to server/: patch the module that defines it")
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Forwarding

if __name__ == "__main__":
    _app.main()
