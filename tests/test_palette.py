"""PRD 30 — the ⌘K command palette. Scenario numbers are the PRD's.

Everything here is client-side: static/app.js runs in node against the DOM shim in tests/app_fixture.mjs
and this file asserts on the JSON it prints (scenario 7 is the browser e2e in tests/e2e/test_palette.py).
"""
import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

FIXED = ["Go to Chat", "Go to Memory", "Go to Wiki", "Go to Sources", "Go to Settings",
         "Today's note", "New page…", "Voice: turn on", "Mute replies",
         "Reply length: short", "Reply length: normal", "Reply length: long",
         "Backdrop: None", "Backdrop: Home office", "Reload characters",
         "File a source…", "New topic", "Re-ingest last session", "Shut down"]
PAGES = ["Anna", "Atlas", "todo"]


@pytest.fixture(scope="module")
def r():
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    out = subprocess.run(["node", "tests/app_fixture.mjs"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


# --------------------------------------------------------------- 1. open / close

def test_cmd_k_opens_focused_and_empty_with_commands_then_pages(r):
    assert r["pal_open"] == {"hidden": False, "focused": True, "value": "", "rows": FIXED + PAGES}


def test_view_keys_are_off_while_open_and_toggle_or_escape_closes(r):
    assert r["pal_view_while_open"] == "chat"
    assert r["pal_hidden_after_toggle"] is True
    assert r["pal_hidden_after_escape"] is True


# --------------------------------------------------------------- 2. ranking

def test_ranking_prefers_label_prefix_then_substring_then_subsequence(r):
    assert r["pal_rank_atl"] == ["Atlas", 'Search wiki for "atl"']
    assert r["pal_rank_set"][0] == "Go to Settings"
    assert r["pal_rank_wk"] == ["Go to Wiki", 'Search wiki for "wk"']       # subsequence only
    assert r["pal_rank_zzz"] == ['Search wiki for "zzz"']


# --------------------------------------------------------------- 3. keyboard

def test_arrows_move_and_enter_opens_the_page_in_the_wiki_view(r):
    assert r["pal_nav"] == {"after_down": 1, "after_up": 0}
    sent = [m for m in r["pal_enter"]["sent"] if m["action"] != "set_focus"]   # PRD 38 rides along
    assert {**r["pal_enter"], "sent": sent} == {"view": "wiki", "hidden": True,
                                                "sent": [{"action": "read_wiki_page", "path": "projects/atlas"}]}


# --------------------------------------------------------------- 4. search fallback

def test_unknown_query_falls_back_to_a_wiki_search(r):
    sent = [m for m in r["pal_fallback"]["sent"] if m["action"] != "set_focus"]
    assert {**r["pal_fallback"], "sent": sent} == {"view": "wiki", "search": "zzz",
                                                   "sent": [{"action": "wiki_search", "query": "zzz"}]}


# --------------------------------------------------------------- 5. payloads

def test_commands_send_the_same_payloads_as_the_existing_controls(r):
    assert r["pal_backdrop"] == [{"action": "set_backdrop", "key": "office"}]
    assert r["pal_verbosity"] == [{"action": "set_verbosity", "value": "short"}]
    assert r["pal_voice"] == [{"action": "set_hands_free", "value": False}]
    assert r["pal_mute"] == {"before": False, "after": True}
    assert r["pal_ingest"] == [{"action": "wiki_ingest"}]


def test_shutdown_asks_first_and_only_sends_when_agreed(r):
    assert r["pal_shutdown_declined"] == []
    assert r["pal_shutdown_asked"][0] == "confirm" and "Shut down" in r["pal_shutdown_asked"][1]
    assert r["pal_shutdown_agreed"] == [{"action": "shutdown"}]


# --------------------------------------------------------------- 6. click

def test_clicking_a_row_runs_it(r):
    assert r["pal_click"] == {"view": "memory", "hidden": True}
