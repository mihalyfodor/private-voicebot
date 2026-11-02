"""Shared fixtures and constants for the ingest tests."""
import json

import pytest

from core import llm as core_llm
from tests.conftest import FakeClient


@pytest.fixture
def fake(monkeypatch):
    """A FakeClient wired in as the module's client: summary call first, page ops second."""
    client = FakeClient(SUMMARY, OPS)
    monkeypatch.setattr(core_llm, "_client", client)
    return client


SUMMARY = json.dumps({
    "title": "Falcon status",
    "summary": "Project Falcon is amber; the load test showed 940 ms p99.",
    "entities": [{"name": "Anna", "type": "person"}, {"name": "Falcon", "type": "project"}],
    "unreadable": False,
})

OPS = json.dumps([
    {"op": "new_page", "path": "projects/falcon", "type": "project", "name": "Falcon",
     "summary": "Platform project, amber."},
])


def prompts(client: FakeClient) -> list:
    """The user content of every completion the client was asked for."""
    return [call["messages"][1]["content"] for call in client.calls]
