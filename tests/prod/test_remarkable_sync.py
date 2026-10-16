"""PRD 47 scenarios 1-4: scripts/remarkable_sync.py pull against the fake device and a fake /ingest."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

from tests.conftest import ROOT, free_port

SYNC = os.path.join(ROOT, "scripts", "remarkable_sync.py")

pytestmark = pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="pdftoppm (poppler) is not installed")


def pull(tmp_path, device: str, app: str, **env) -> subprocess.CompletedProcess:
    run_env = {**os.environ, "REMARKABLE_OCR_BIN": str(tmp_path / "no-such-ocr"), "TRACE_DIR": str(tmp_path), **env}
    return subprocess.run([sys.executable, SYNC, "pull", "--device", device, "--app", app,
                           "--state", str(tmp_path / "remarkable.json")],
                          capture_output=True, text=True, env=run_env, timeout=60)


def state(tmp_path) -> dict:
    path = tmp_path / "remarkable.json"
    return json.loads(path.read_text()) if path.exists() else {}


def lines(out: subprocess.CompletedProcess) -> list:
    return [line for line in out.stdout.splitlines() if line.strip()]


def ocr_sent(fake_ingest) -> list:
    """The `ocr` field of every request, decoded."""
    return [json.loads(r["fields"]["ocr"]) for r in fake_ingest.requests]


def stub(tmp_path, body: str) -> str:
    path = tmp_path / "vision_ocr"
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    return str(path)


# --------------------------------------------------------------------------- scenario 1


def test_pull_files_each_new_notebook_in_the_folder_once(tmp_path, fake_remarkable, fake_ingest):
    out = pull(tmp_path, fake_remarkable.url, fake_ingest.url)

    assert out.returncode == 0, out.stderr
    reqs = fake_ingest.requests
    assert sorted(r["fields"]["title"] for r in reqs) == ["Ideas", "Standup"]   # not "Paper", not "Outside"
    for r in reqs:
        assert r["path"] == "/ingest"
        assert r["fields"]["kind"] == "notebook" and r["fields"]["confirm"] == "true"
        assert len(r["files"]) == 2 and all(data.startswith(b"\x89PNG") for _n, data in r["files"])
        assert [n for n, _d in r["files"]] == ["page-01.png", "page-02.png"]
        assert "replaces" not in r["fields"] or r["fields"]["replaces"] == ""
    assert len({r["fields"]["run_id"] for r in reqs}) == 1                     # one pull, one run (PRD 49)
    assert re.fullmatch(r"rm-\d{8}-\d{6}", reqs[0]["fields"]["run_id"])          # its run page id (PRD 51)
    saved = state(tmp_path)
    assert set(saved) == {"n1", "n2"}
    assert saved["n1"]["pages"] == 2 and saved["n1"]["name"] == "Standup"
    assert saved["n1"]["modified"] == "2026-10-14T09:00:00.000Z"
    first = next(r for r in reqs if r["fields"]["title"] == "Standup")
    assert saved["n1"]["sha"] == hashlib.sha256(b"".join(d for _n, d in first["files"])).hexdigest()
    assert sorted(lines(out)) == ['filed "Ideas" (2 pages) -> sources/ideas',
                                  'filed "Standup" (2 pages) -> sources/standup']
    assert fake_remarkable.downloads.count("p1") == 0 and "n9" not in fake_remarkable.downloads


# --------------------------------------------------------------------------- scenario 2


def test_an_unchanged_notebook_is_skipped_and_an_edited_one_replaces(tmp_path, fake_remarkable, fake_ingest):
    pull(tmp_path, fake_remarkable.url, fake_ingest.url)
    old_sha = state(tmp_path)["n1"]["sha"]
    fake_ingest.requests.clear()

    again = pull(tmp_path, fake_remarkable.url, fake_ingest.url)
    assert again.returncode == 0 and fake_ingest.requests == []
    assert lines(again) == ["nothing new"]

    fake_remarkable.docs["n1"]["ModifiedClient"] = "2026-10-15T08:00:00.000Z"
    edited = pull(tmp_path, fake_remarkable.url, fake_ingest.url)
    assert edited.returncode == 0
    assert [r["fields"]["title"] for r in fake_ingest.requests] == ["Standup"]
    assert fake_ingest.requests[0]["fields"]["replaces"] == old_sha
    assert state(tmp_path)["n1"]["modified"] == "2026-10-15T08:00:00.000Z"


# --------------------------------------------------------------------------- scenario 3


def test_device_down_is_one_line_and_exit_0(tmp_path, fake_ingest):
    (tmp_path / "remarkable.json").write_text('{"n1": {"modified": "x"}}')
    out = pull(tmp_path, f"http://127.0.0.1:{free_port()}", fake_ingest.url)

    assert out.returncode == 0
    assert len(lines(out)) == 1 and lines(out)[0].startswith("reMarkable not connected")
    assert (tmp_path / "remarkable.json").read_text() == '{"n1": {"modified": "x"}}'
    assert fake_ingest.requests == []


def test_app_down_exits_1_and_leaves_the_state(tmp_path, fake_remarkable):
    out = pull(tmp_path, fake_remarkable.url, f"http://127.0.0.1:{free_port()}")

    assert out.returncode == 1
    assert not (tmp_path / "remarkable.json").exists()
    assert "not reachable" in out.stdout


def test_one_failing_notebook_does_not_stop_the_other(tmp_path, fake_remarkable, fake_ingest):
    fake_ingest.fail.add("Ideas")
    out = pull(tmp_path, fake_remarkable.url, fake_ingest.url)

    assert out.returncode == 1
    assert set(state(tmp_path)) == {"n1"}
    assert 'filed "Standup" (2 pages) -> sources/standup' in lines(out)
    assert any(line.startswith('failed "Ideas"') for line in lines(out))


def test_a_garbled_state_file_is_treated_as_empty(tmp_path, fake_remarkable, fake_ingest):
    (tmp_path / "remarkable.json").write_text("{not json")
    out = pull(tmp_path, fake_remarkable.url, fake_ingest.url)

    assert out.returncode == 0 and set(state(tmp_path)) == {"n1", "n2"}


# --------------------------------------------------------------------------- scenario 4


def test_no_ocr_helper_sends_empty_strings(tmp_path, fake_remarkable, fake_ingest):
    out = pull(tmp_path, fake_remarkable.url, fake_ingest.url)

    assert out.returncode == 0
    assert ocr_sent(fake_ingest) == [["", ""]] * len(fake_ingest.requests)


def test_ocr_disabled_skips_a_present_helper(tmp_path, fake_remarkable, fake_ingest):
    out = pull(tmp_path, fake_remarkable.url, fake_ingest.url,
               REMARKABLE_OCR_BIN=stub(tmp_path, "echo seen"), REMARKABLE_OCR="0")

    assert out.returncode == 0
    assert ocr_sent(fake_ingest) == [["", ""]] * len(fake_ingest.requests)


def test_an_ocr_helper_text_arrives_per_page(tmp_path, fake_remarkable, fake_ingest):
    out = pull(tmp_path, fake_remarkable.url, fake_ingest.url,
               REMARKABLE_OCR_BIN=stub(tmp_path, 'echo "standup notes"'))

    assert out.returncode == 0
    assert ocr_sent(fake_ingest) == [["standup notes", "standup notes"]] * len(fake_ingest.requests)


def test_a_failing_ocr_helper_is_an_empty_string(tmp_path, fake_remarkable, fake_ingest):
    out = pull(tmp_path, fake_remarkable.url, fake_ingest.url,
               REMARKABLE_OCR_BIN=stub(tmp_path, "echo half; exit 1"))

    assert out.returncode == 0 and len(fake_ingest.requests) == 2
    assert ocr_sent(fake_ingest) == [["", ""]] * len(fake_ingest.requests)
