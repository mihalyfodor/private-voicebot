import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from types import SimpleNamespace

import pytest

import memory


def _client_with_content(content):
    def create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_save_raises_when_content_is_none(monkeypatch, tmp_path):
    monkeypatch.setattr(memory, "SHORTMEM_PATH", str(tmp_path / "shortmem.txt"))
    client = _client_with_content(None)
    with pytest.raises(RuntimeError):
        memory.save([{"role": "user", "content": "hi"}], client, "some-model")


def test_save_nothing_new_is_still_a_noop(monkeypatch, tmp_path, capsys):
    path = tmp_path / "shortmem.txt"
    monkeypatch.setattr(memory, "SHORTMEM_PATH", str(path))
    client = _client_with_content("NOTHING")
    memory.save([{"role": "user", "content": "hi"}], client, "some-model")
    assert not path.exists()


def test_shutdown_hook_saves_memory(monkeypatch):
    from fastapi.testclient import TestClient
    import chatbot, llm
    calls = []
    monkeypatch.setattr(llm, "save_memory", lambda: calls.append(1))
    with TestClient(chatbot.app):
        pass  # exiting the context runs the shutdown event
    assert calls == [1]


def test_save_memory_is_idempotent(monkeypatch):
    import llm, memory
    saved = []
    monkeypatch.setattr(memory, "save", lambda turns, client, model: saved.append(list(turns)))
    monkeypatch.setattr(llm, "client", lambda: object())
    llm.reset()
    llm._session_turns.append({"role": "user", "content": "hi"})
    llm.save_memory(); llm.save_memory()
    assert saved == [[{"role": "user", "content": "hi"}]]
