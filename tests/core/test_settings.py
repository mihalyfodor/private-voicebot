"""settings.json saves from several threads must not lose keys."""
import threading

from core import settings


def test_concurrent_save_setting_keeps_every_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    keys = [f"k{i}" for i in range(40)]

    def save(chunk):
        for k in chunk:
            settings.save_setting(k, k)

    threads = [threading.Thread(target=save, args=(keys[i::4],)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    data = settings.load_settings()
    assert all(data.get(k) == k for k in keys)
