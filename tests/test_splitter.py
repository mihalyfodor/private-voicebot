import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from splitter import SentenceSplitter, split_stream, strip_tag


def test_tag_split_across_deltas_and_two_sentences():
    out = list(split_stream(["[hap", "py] Sure thing.", " Anything else?"]))
    assert out == [("happy", "Sure thing."), ("happy", "Anything else?")]


def test_no_tag_defaults_to_neutral():
    out = list(split_stream(["Sure thing. ", "Anything else?"]))
    assert out == [("neutral", "Sure thing."), ("neutral", "Anything else?")]


def test_unknown_tag_is_stripped_and_neutral():
    out = list(split_stream(["[angry] Not today."]))
    assert out == [("neutral", "Not today.")]


def test_no_terminal_punctuation_flushed_on_close():
    out = list(split_stream(["[thinking] let me see"]))
    assert out == [("thinking", "let me see")]


def test_first_sentence_emitted_before_stream_ends():
    sp = SentenceSplitter()
    assert list(sp.feed("[happy] First one. Second")) == [("happy", "First one.")]
    assert list(sp.close()) == [("happy", "Second")]


def test_strip_tag():
    assert strip_tag("[happy] Hi there.") == "Hi there."
    assert strip_tag("Hi there.") == "Hi there."


def test_leaked_thought_line_is_dropped():
    out = list(split_stream(["thought\n\n[neutral] It is noon."]))
    assert out == [("neutral", "It is noon.")]
