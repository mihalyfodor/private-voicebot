"""Shim (PRD 50): the module moved to server/session.py; this name is the same module object, so patches reach it."""
import sys

from server import session as _real

sys.modules[__name__] = _real
