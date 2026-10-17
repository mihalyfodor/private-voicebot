"""Memory v2 — a structured user profile with LLM-proposed ops applied in Python (docs/15-memory-v2.md).

Modules: `store` (files, schema), `render` (decay, the prompt block), `identity` (who the user is), `ops` (validate and
apply ops, the editor), `bridge` (profile entry <-> vault page), `wiki_ingest` (session ingest into the vault).
The model-calling half (propose / reflect / save) is `agent/fold.py`.
"""
