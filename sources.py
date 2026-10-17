"""Shim (PRD 50): the module split into core/formats.py, vault/manifest.py and ingest/pipeline.py. Tests only."""
from core import formats as _formats
from vault import manifest as _manifest
from ingest import pipeline as _pipeline
from core.formats import *  # noqa: F401,F403
from vault.manifest import *  # noqa: F401,F403
from ingest.pipeline import *  # noqa: F401,F403

# `import *` skips underscore names; tests read some of them through the shim.
for _mod in (_formats, _manifest, _pipeline):
    globals().update({k: v for k, v in vars(_mod).items() if k.startswith("_") and not k.startswith("__")})
