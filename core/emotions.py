"""The emotion vocabulary and the leading `[tag]` of a reply: what the prompt offers and the speech splitter and the
written answers read, so it sits below all of them."""
import re

EMOTIONS = ("neutral", "happy", "thinking", "surprised", "apologetic", "proud", "sleepy", "shy")
# Optional leaked "thought" line seen from Gemma 4 after tool rounds, then the [tag].
_TAG_RE = re.compile(r"^\s*(?:thought\s*\n)?\s*\[([a-zA-Z]+)\]\s*")


def strip_tag(text: str) -> str:
    return _TAG_RE.sub("", text, count=1)
