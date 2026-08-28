import pytest
import llm

THRESHOLD = 0.70
SAMPLES = 3  # runs per phrase; a phrase "hits" on a majority (>= 2/3)


def _omlx_available() -> bool:
    try:
        llm.client().with_options(timeout=3.0).models.list()
        return True
    except Exception:
        return False


if not _omlx_available():
    pytest.skip("oMLX server not reachable at OMLX_BASE_URL", allow_module_level=True)


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
    "what's going on in the world today?",
]

EMAIL_VARIANTS = [
    "check my emails",
    "do I have any new emails?",
    "what's in my inbox?",
    "did I get any new emails?",
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


def _majority_hit(phrase, tool_name):
    """Run `phrase` SAMPLES times; True if the expected behaviour wins the majority.

    `tool_name=None` means "no tool call expected". Generation is sampled, so a single run
    is noisy — the majority vote is what the rate assertions are built on.
    """
    hits = 0
    for _ in range(SAMPLES):
        llm.reset()
        llm.ask(phrase)
        if _called(tool_name) if tool_name else (llm.get_last_tool_calls() == []):
            hits += 1
    return hits * 2 >= SAMPLES


def _assert_rate(label, variants, tool_name):
    hits = sum(1 for phrase in variants if _majority_hit(phrase, tool_name))
    rate = hits / len(variants)
    assert rate >= THRESHOLD, \
        f"{label} rate {hits}/{len(variants)} ({rate:.0%}) below {THRESHOLD:.0%}"


# --- weather ---

@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("phrase", WEATHER_VARIANTS)
def test_weather_variant(phrase):
    llm.reset()
    llm.ask(phrase)
    assert _called("get_weather"), f"get_weather not triggered for: {phrase!r} | calls: {llm.get_last_tool_calls()}"


def test_weather_success_rate():
    _assert_rate("weather trigger", WEATHER_VARIANTS, "get_weather")


# --- time ---

@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("phrase", TIME_VARIANTS)
def test_time_variant(phrase):
    llm.reset()
    llm.ask(phrase)
    assert _called("get_time"), f"get_time not triggered for: {phrase!r} | calls: {llm.get_last_tool_calls()}"


def test_time_success_rate():
    _assert_rate("time trigger", TIME_VARIANTS, "get_time")


# --- news ---

@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("phrase", NEWS_VARIANTS)
def test_news_variant(phrase):
    llm.reset()
    llm.ask(phrase)
    assert _called("get_news"), f"get_news not triggered for: {phrase!r} | calls: {llm.get_last_tool_calls()}"


def test_news_success_rate():
    _assert_rate("news trigger", NEWS_VARIANTS, "get_news")


# --- email ---

@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("phrase", EMAIL_VARIANTS)
def test_email_variant(phrase):
    llm.reset()
    llm.ask(phrase)
    assert _called("get_emails"), f"get_emails not triggered for: {phrase!r} | calls: {llm.get_last_tool_calls()}"


def test_email_success_rate():
    _assert_rate("email trigger", EMAIL_VARIANTS, "get_emails")


# --- no-tool (conversational) ---

@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("phrase", NO_TOOL_VARIANTS)
def test_no_tool_variant(phrase):
    llm.reset()
    llm.ask(phrase)
    assert llm.get_last_tool_calls() == [], f"unexpected tool call for: {phrase!r} | calls: {llm.get_last_tool_calls()}"


def test_no_tool_success_rate():
    _assert_rate("no-tool", NO_TOOL_VARIANTS, None)
