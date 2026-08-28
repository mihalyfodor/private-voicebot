import re
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import run_tool

pytestmark = pytest.mark.live  # needs a running oMLX server


def test_get_time():
    result = run_tool("get_time", {})
    assert re.match(r"\d{2}:\d{2}:\d{2}", result), f"Unexpected time format: {result}"


def test_get_weather():
    result = run_tool("get_weather", {})
    assert "degrees" in result, f"Missing 'degrees' in weather result: {result}"


def test_get_news():
    result = run_tool("get_news", {})
    assert "1." in result, f"Missing numbered headlines in news result: {result}"


def test_get_emails():
    from tools import gmail
    if not gmail.is_configured():
        pytest.skip("Gmail not configured (credentials.json missing)")
    result = run_tool("get_emails", {"max_results": 1})
    assert "From" in result or "empty" in result, f"Unexpected email result: {result}"


def test_unknown_tool():
    result = run_tool("nonexistent", {})
    assert result == "unknown tool"


def test_gmail_unconfigured_returns_message(monkeypatch):
    from tools import gmail
    monkeypatch.setattr(gmail, "is_configured", lambda: False)
    assert "not set up" in gmail.run({})
