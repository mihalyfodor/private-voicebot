"""Turn a stream of text deltas into (emotion, sentence) pairs.

A single leading ``[tag]`` selects the emotion for the whole reply. Unknown or
missing tags fall back to ``neutral``. Sentences split on ``.``, ``?``, ``!``
followed by whitespace or end of text; any remainder is flushed on ``close()``.
"""
import re
from typing import Iterator, Iterable

EMOTIONS = ("neutral", "happy", "thinking", "surprised", "apologetic")
# Optional leaked "thought" line seen from Gemma 4 after tool rounds, then the [tag].
_TAG_RE = re.compile(r"^\s*(?:thought\s*\n)?\s*\[([a-zA-Z]+)\]\s*")
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


class SentenceSplitter:
    def __init__(self):
        self._buf = ""
        self._emotion: str | None = None  # None until tag is resolved

    @property
    def emotion(self) -> str:
        return self._emotion or "neutral"

    def _resolve_tag(self) -> bool:
        """Return True once the tag question is settled (present or absent)."""
        if self._emotion is not None:
            return True
        m = _TAG_RE.match(self._buf)
        if m:
            tag = m.group(1).lower()
            self._emotion = tag if tag in EMOTIONS else "neutral"
            self._buf = self._buf[m.end():]
            return True
        stripped = self._buf.lstrip()
        if stripped.startswith("[") and "]" not in stripped:
            return False  # tag may still be arriving
        # Leaked "thought" line may still be followed by a [tag] in a later
        # delta; keep waiting while the buffer is only a (partial) prefix of
        # "thought", or "thought" plus trailing whitespace.
        lowered = stripped.lower()
        if lowered and "thought".startswith(lowered):
            return False
        if lowered.startswith("thought") and lowered[len("thought"):].strip() == "":
            return False
        if stripped:  # text present and it is not a tag
            self._emotion = "neutral"
            return True
        return False

    def feed(self, delta: str) -> Iterator[tuple[str, str]]:
        self._buf += delta
        if not self._resolve_tag():
            return
        parts = _SENTENCE_END.split(self._buf)
        self._buf = parts.pop()
        for p in parts:
            p = p.strip()
            if p:
                yield (self.emotion, p)

    def close(self) -> Iterator[tuple[str, str]]:
        self._resolve_tag()
        rest = _TAG_RE.sub("", self._buf).strip() if self._emotion is None else self._buf.strip()
        self._buf = ""
        if rest:
            yield (self.emotion, rest)


def split_stream(deltas: Iterable[str]) -> Iterator[tuple[str, str]]:
    s = SentenceSplitter()
    for d in deltas:
        yield from s.feed(d)
    yield from s.close()


def strip_tag(text: str) -> str:
    return _TAG_RE.sub("", text, count=1)
