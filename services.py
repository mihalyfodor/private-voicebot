"""Shim (PRD 50): the module moved to server/services.py; this name is the same module object, so patches reach it."""
import sys

from server import services as _real

sys.modules[__name__] = _real
