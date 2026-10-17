"""Speech: the sentence splitter, the spoken fillers, Kokoro TTS and the avatar voice (PRD 50 step 11, from splitter.py, fillers.py and chatbot.py).

Splitter: turn a stream of text deltas into (emotion, sentence) pairs.

A single leading ``[tag]`` selects the emotion for the whole reply. Unknown or
missing tags fall back to ``neutral``. Sentences split on ``.``, ``?``, ``!``
followed by whitespace or end of text; any remainder is flushed on ``close()``.
"""
import random
import re
from typing import Iterator, Iterable

from core.emotions import EMOTIONS, _TAG_RE, strip_tag  # noqa: F401  (re-exported: tests read them here)
import os
import io
import base64
from core import trace
from agent import conversation as a_conversation
from core import avatars
import soundfile as sf

log = trace.logger("App")

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


# --- fillers: short spoken lines played while a tool call is in flight (from fillers.py) -----------
FILLERS = {
    "get_time": ["One moment.", "Let me check the clock.", "Just a second."],
    "get_weather": ["Let me check outside.", "One sec, checking the weather.", "Let me have a look at the forecast."],
    "get_news": ["Pulling up the headlines.", "Let me see what's going on.", "One moment, checking the news."],
    "get_news_detail": ["Let me read that one.", "One sec, opening it up."],
    "get_emails": ["Let me check your inbox.", "One sec, looking at your mail.", "Checking your emails now."],
    "search_wiki": ["Let me look that up in my notes.", "One sec, checking what we wrote down.", "Digging through the wiki."],
    "read_page": ["Let me open that page.", "One sec, reading it up.", "Pulling up the notes."],
    "dig_deeper": ["Let me read a bit further.", "One sec, going deeper into it.", "Let me pull up the rest."],
    "file_source": ["Filing that away.", "One sec, writing it down.", "Let me put that in the wiki."],
    "get_wikipedia": ["Let me check Wikipedia.", "One sec, looking it up on Wikipedia.", "Checking Wikipedia now."],
    "read_wikipedia": ["Let me read that section.", "One sec, reading further on Wikipedia."],
    "default": ["One moment.", "Let me check.", "Just a second."],
}

_last: dict[str, str] = {}


def pick(tool_name: str) -> str:
    options = FILLERS.get(tool_name) or FILLERS["default"]
    key = tool_name if tool_name in FILLERS else "default"
    choices = [p for p in options if p != _last.get(key)] or options
    phrase = random.choice(choices)
    _last[key] = phrase
    return phrase


# --- Kokoro TTS and the avatar voice (from chatbot.py) ---------------------------------------------

KOKORO_MODEL = os.path.expanduser(os.getenv("KOKORO_MODEL", "~/models/kokoro/kokoro-v1.0.onnx"))
KOKORO_VOICES = os.path.expanduser(os.getenv("KOKORO_VOICES", "~/models/kokoro/voices-v1.0.bin"))
AVATAR = avatars.current()
KOKORO_VOICE = os.getenv("KOKORO_VOICE", AVATAR["voice"])
KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", AVATAR["speed"]))


def apply_avatar(key: str) -> dict:
    """(Re)apply the avatar profile: persona, voice, filler cache. Returns the profile."""
    global AVATAR, KOKORO_VOICE, KOKORO_SPEED
    AVATAR = avatars.set_current(key)
    KOKORO_VOICE = os.getenv("KOKORO_VOICE", AVATAR["voice"])
    KOKORO_SPEED = float(os.getenv("KOKORO_SPEED", AVATAR["speed"]))
    a_conversation.set_avatar()
    _filler_wavs.clear()
    return AVATAR

kokoro = None  # loaded in main()


def synthesize(text: str) -> bytes:
    """Text → 16-bit PCM wav bytes (Kokoro's native 24 kHz)."""
    samples, sample_rate = kokoro.create(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


_filler_wavs: dict[str, bytes] = {}


def speak_stream(events, send_sync, tts=None):
    """Consume LLM events, TTS each sentence and push it over the socket.

    `events` yields ("tool_calls", [names]) and ("delta", text) — see agent.turn.ask_events.
    A cached filler is spoken as soon as a tool call is detected.
    Returns the full reply text (tag stripped, fillers excluded).
    `send_sync(dict)` must be thread-safe; `tts(text) -> wav bytes` defaults to Kokoro.
    """
    tts = tts or synthesize
    sp = SentenceSplitter()
    sentences = []
    started = False

    def emit(emotion, text, wav, filler=False):
        nonlocal started
        if not started:
            started = True
            send_sync({"type": "state", "value": "speaking"})
        msg = {
            "type": "speech",
            "emotion": emotion,
            "text": text,
            "wav": base64.b64encode(wav).decode("ascii"),
        }
        if filler:
            msg["filler"] = True   # the browser sparkles while a tool runs (PRD 27)
        send_sync(msg)

    def say(emotion, sentence):
        sentences.append(sentence)
        emit(emotion, sentence, tts(sentence))

    def filler(tool_names):
        phrase = pick(tool_names[0] if tool_names else "default")
        if phrase not in _filler_wavs:
            _filler_wavs[phrase] = tts(phrase)
        emit("thinking", phrase, _filler_wavs[phrase], filler=True)

    try:
        for kind, payload in events:
            if kind == "tool_calls":
                filler(payload)
            elif kind == "delta":
                for emotion, sentence in sp.feed(payload):
                    say(emotion, sentence)
        for emotion, sentence in sp.close():
            say(emotion, sentence)
    finally:
        if started:
            send_sync({"type": "speech_end"})

    return " ".join(sentences)
