"""Shim (PRD 50): the module moved to core/avatars.py; this name is the same module object, so patches reach it."""
import sys

from core import avatars as _real

sys.modules[__name__] = _real
