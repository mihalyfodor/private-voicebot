import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llm


def setup_function():
    llm.reset()


def test_time_triggers_tool():
    llm.ask("what time is it?")
    assert any(tc["name"] == "get_time" for tc in llm.get_last_tool_calls()), \
        f"Expected get_time tool call, got: {llm.get_last_tool_calls()}"


def test_weather_triggers_tool():
    llm.ask("what's the weather like?")
    assert any(tc["name"] == "get_weather" for tc in llm.get_last_tool_calls()), \
        f"Expected get_weather tool call, got: {llm.get_last_tool_calls()}"


def test_news_triggers_tool():
    llm.ask("what's in the news today?")
    assert any(tc["name"] == "get_news" for tc in llm.get_last_tool_calls()), \
        f"Expected get_news tool call, got: {llm.get_last_tool_calls()}"


def test_email_triggers_tool():
    llm.ask("check my emails")
    assert any(tc["name"] == "get_emails" for tc in llm.get_last_tool_calls()), \
        f"Expected get_emails tool call, got: {llm.get_last_tool_calls()}"


def test_conversational_no_tool():
    llm.ask("how are you doing?")
    assert llm.get_last_tool_calls() == [], \
        f"Expected no tool calls, got: {llm.get_last_tool_calls()}"
