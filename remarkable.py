"""Shim (PRD 50): the module moved to ingest/remarkable.py; this name is the same module object, so patches reach it."""
import sys

from ingest import remarkable as _real

sys.modules[__name__] = _real
