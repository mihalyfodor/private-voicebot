"""Shim (PRD 50): the module moved to vault/index_db.py; this name is the same module object, so patches reach it."""
import sys

from vault import index_db as _real

sys.modules[__name__] = _real
