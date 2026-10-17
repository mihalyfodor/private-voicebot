"""Shim (PRD 50): the module moved to agent/research.py; this name is the same module object, so patches reach it."""
import sys

from agent import research as _real

sys.modules[__name__] = _real
