"""Shared helpers and the push-service fixture for the prod tests."""
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import pytest

from tests.conftest import ROOT, free_port

SERVER = os.path.join(ROOT, "scripts", "remarkable_server.py")
RENDER = pytest.mark.skipif(shutil.which("cupsfilter") is None or shutil.which("pdftotext") is None,
                            reason="cupsfilter / pdftotext are not installed")


def serve_push(device: str):
    """The push service on a free loopback port, its tablet at `device` -> ``(process, base url)``."""
    port = free_port()
    proc = subprocess.Popen([sys.executable, SERVER], env={**os.environ, "REMARKABLE_PORT": str(port),
                                                           "REMARKABLE_URL": device},
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            return proc, url
        except OSError:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("push service did not start")


@pytest.fixture
def push_service(fake_remarkable):
    proc, url = serve_push(fake_remarkable.url)
    yield url
    proc.kill()
