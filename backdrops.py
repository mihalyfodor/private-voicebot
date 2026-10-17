"""Shim (PRD 50): the module moved to server/backdrops.py; this name is the same module object, so patches reach it."""
import sys

from server import backdrops as _real

sys.modules[__name__] = _real
