import pytest
import llm

THRESHOLD = 0.70

WEATHER_VARIANTS = [
    "what's the weather like?",
    "is it going to rain today?",
    "should I bring a jacket?",
    "how's it looking outside?",
    "tell me about the weather",
    "what's the temperature right now?",
]

TIME_VARIANTS = [
    "what time is it?",
    "what's the time?",
    "do you know what time it is?",
    "can you tell me the time?",
    "what time do we have?",
    "give me the current time",
]

NEWS_VARIANTS = [
    "what's in the news today?",
    "any news?",
    "what's happening in the world?",
    "catch me up on current events",
    "what are the headlines?",
    "anything interesting going on?",
]

EMAIL_VARIANTS = [
    "check my emails",
    "do I have any new emails?",
    "what's in my inbox?",
    "any messages for me?",
    "check my inbox",
    "read my email",
]

NO_TOOL_VARIANTS = [
    "how are you doing?",
    "tell me a joke",
    "what do you think about AI?",
    "say something interesting",
    "I'm bored",
    "thanks",
]


def setup_function():
    llm.reset()


def _called(tool_name):
    return any(tc["name"] == tool_name for tc in llm.get_last_tool_calls())


# --- weather ---

@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("phrase", WEATHER_VARIANTS)
def test_weather_variant(phrase):
    llm.reset()
    llm.ask(phrase)
    assert _called("get_weather"), f"get_weather not triggered for: {phrase!r} | calls: {llm.get_last_tool_calls()}"


def test_weather_success_rate():
    hits = 0
    for phrase in WEATHER_VARIANTS:
        llm.reset()
        llm.ask(phrase)
        if _called("get_weather"):
            hits += 1
    rate = hits / len(WEATHER_VARIANTS)
    assert rate >= THRESHOLD, f"weather trigger rate {hits}/{len(WEATHER_VARIANTS)} ({rate:.0%}) below {THRESHOLD:.0%}"


# --- time ---

@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("phrase", TIME_VARIANTS)
def test_time_variant(phrase):
    llm.reset()
    llm.ask(phrase)
    assert _called("get_time"), f"get_time not triggered for: {phrase!r} | calls: {llm.get_last_tool_calls()}"


def test_time_success_rate():
    hits = 0
    for phrase in TIME_VARIANTS:
        llm.reset()
        llm.ask(phrase)
        if _called("get_time"):
            hits += 1
    rate = hits / len(TIME_VARIANTS)
    assert rate >= THRESHOLD, f"time trigger rate {hits}/{len(TIME_VARIANTS)} ({rate:.0%}) below {THRESHOLD:.0%}"


# --- news ---

@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("phrase", NEWS_VARIANTS)
def test_news_variant(phrase):
    llm.reset()
    llm.ask(phrase)
    assert _called("get_news"), f"get_news not triggered for: {phrase!r} | calls: {llm.get_last_tool_calls()}"


def test_news_success_rate():
    hits = 0
    for phrase in NEWS_VARIANTS:
        llm.reset()
        llm.ask(phrase)
        if _called("get_news"):
            hits += 1
    rate = hits / len(NEWS_VARIANTS)
    assert rate >= THRESHOLD, f"news trigger rate {hits}/{len(NEWS_VARIANTS)} ({rate:.0%}) below {THRESHOLD:.0%}"


# --- email ---

@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("phrase", EMAIL_VARIANTS)
def test_email_variant(phrase):
    llm.reset()
    llm.ask(phrase)
    assert _called("get_emails"), f"get_emails not triggered for: {phrase!r} | calls: {llm.get_last_tool_calls()}"


def test_email_success_rate():
    hits = 0
    for phrase in EMAIL_VARIANTS:
        llm.reset()
        llm.ask(phrase)
        if _called("get_emails"):
            hits += 1
    rate = hits / len(EMAIL_VARIANTS)
    assert rate >= THRESHOLD, f"email trigger rate {hits}/{len(EMAIL_VARIANTS)} ({rate:.0%}) below {THRESHOLD:.0%}"


# --- no-tool (conversational) ---

@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("phrase", NO_TOOL_VARIANTS)
def test_no_tool_variant(phrase):
    llm.reset()
    llm.ask(phrase)
    assert llm.get_last_tool_calls() == [], f"unexpected tool call for: {phrase!r} | calls: {llm.get_last_tool_calls()}"


def test_no_tool_success_rate():
    hits = 0
    for phrase in NO_TOOL_VARIANTS:
        llm.reset()
        llm.ask(phrase)
        if llm.get_last_tool_calls() == []:
            hits += 1
    rate = hits / len(NO_TOOL_VARIANTS)
    assert rate >= THRESHOLD, f"no-tool rate {hits}/{len(NO_TOOL_VARIANTS)} ({rate:.0%}) below {THRESHOLD:.0%}"
