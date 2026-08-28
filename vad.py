"""Voice activity detection: silence-terminated utterance endpointing.

`Endpointer` wraps Silero VAD (ONNX) with a pre-roll ring buffer and a small
state machine (see docs/09-continuous-mode.md) to turn a continuous stream of
PCM audio into discrete utterances. `Recorder` is the trivial push-to-talk
counterpart with no VAD at all.
"""

import os

import numpy as np

FRAME_SAMPLES = 512  # Silero VAD frame size at 16 kHz


def _env_float(name, default):
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return type(default)(val)
    except (TypeError, ValueError):
        return default


DEFAULTS = {
    "sample_rate": 16000,
    "threshold": _env_float("VAD_THRESHOLD", 0.5),
    "end_silence_ms": _env_float("VAD_END_SILENCE_MS", 700),
    "min_speech_ms": _env_float("VAD_MIN_SPEECH_MS", 300),
    "pre_roll_ms": _env_float("VAD_PRE_ROLL_MS", 300),
    "max_utterance_s": _env_float("VAD_MAX_UTTERANCE_S", 30),
    "speech_start_frames": 2,
}

_silero_model = None


def _get_silero_model():
    """Lazily load the process-wide Silero VAD ONNX model (singleton)."""
    global _silero_model
    if _silero_model is None:
        from silero_vad import load_silero_vad

        _silero_model = load_silero_vad(onnx=True)
    return _silero_model


def _to_float32(pcm):
    """Convert int16 or float32 PCM of any length to float32 in [-1, 1]."""
    arr = np.asarray(pcm)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


class Recorder:
    """Push-to-talk capture: no VAD, just accumulate until stop()."""

    def __init__(self):
        self._chunks = []

    def feed(self, pcm):
        self._chunks.append(_to_float32(pcm))

    def stop(self):
        if not self._chunks:
            audio = np.zeros(0, dtype=np.float32)
        else:
            audio = np.concatenate(self._chunks)
        self._chunks = []
        return audio


class Endpointer:
    def __init__(
        self,
        sample_rate=16000,
        threshold=None,
        end_silence_ms=None,
        min_speech_ms=None,
        pre_roll_ms=None,
        max_utterance_s=None,
        speech_start_frames=None,
        prob_fn=None,
    ):
        self.sample_rate = sample_rate
        self.threshold = DEFAULTS["threshold"] if threshold is None else threshold
        self.end_silence_ms = (
            DEFAULTS["end_silence_ms"] if end_silence_ms is None else end_silence_ms
        )
        self.min_speech_ms = (
            DEFAULTS["min_speech_ms"] if min_speech_ms is None else min_speech_ms
        )
        self.pre_roll_ms = DEFAULTS["pre_roll_ms"] if pre_roll_ms is None else pre_roll_ms
        self.max_utterance_s = (
            DEFAULTS["max_utterance_s"] if max_utterance_s is None else max_utterance_s
        )
        self.speech_start_frames = (
            DEFAULTS["speech_start_frames"]
            if speech_start_frames is None
            else speech_start_frames
        )
        self._prob_fn = prob_fn

        self._frame_ms = FRAME_SAMPLES / sample_rate * 1000.0
        self._pre_roll_frames = max(0, round(self.pre_roll_ms / self._frame_ms))
        self._end_silence_frames = max(1, round(self.end_silence_ms / self._frame_ms))
        self._min_speech_frames = max(0, round(self.min_speech_ms / self._frame_ms))
        self._max_utterance_frames = max(1, round(self.max_utterance_s * 1000.0 / self._frame_ms))

        self.gated = False
        self._buffer = np.zeros(0, dtype=np.float32)  # unframed leftover samples
        self._pre_roll = []  # list[np.ndarray] ring buffer of pre-speech frames

        self._reset_utterance_state()

    # -- state -----------------------------------------------------------

    def _reset_utterance_state(self):
        self._in_speech = False
        self._speech_frame_count = 0  # consecutive speech frames (onset detector)
        self._silence_run = 0  # consecutive silent frames since last speech frame
        self._utterance_frames = []  # list[np.ndarray] collected frames of current utterance
        self._utterance_speech_frames = 0  # frames counted as speech within utterance
        self._pre_roll = []

    def reset(self):
        self._buffer = np.zeros(0, dtype=np.float32)
        self._reset_utterance_state()

    @property
    def hearing(self):
        return self._in_speech

    # -- model -------------------------------------------------------------

    def _prob(self, frame):
        if self._prob_fn is not None:
            return self._prob_fn(frame)
        import torch

        model = _get_silero_model()
        return model(torch.from_numpy(frame), self.sample_rate).item()

    def _model_reset(self):
        if self._prob_fn is None:
            model = _get_silero_model()
            model.reset_states()

    # -- public API ----------------------------------------------------

    def feed(self, pcm):
        audio = _to_float32(pcm)
        if self.gated:
            # Consume but ignore; drop any in-progress utterance.
            self._reset_utterance_state()
            self._buffer = np.zeros(0, dtype=np.float32)
            return []

        self._buffer = np.concatenate([self._buffer, audio])
        results = []

        while len(self._buffer) >= FRAME_SAMPLES:
            frame = self._buffer[:FRAME_SAMPLES]
            self._buffer = self._buffer[FRAME_SAMPLES:]
            utterance = self._process_frame(frame)
            if utterance is not None:
                results.append(utterance)

        return results

    def _process_frame(self, frame):
        prob = self._prob(frame)
        is_speech = prob >= self.threshold

        if not self._in_speech:
            if is_speech:
                self._speech_frame_count += 1
            else:
                self._speech_frame_count = 0
                # maintain pre-roll ring buffer while idle
                self._pre_roll.append(frame)
                if len(self._pre_roll) > self._pre_roll_frames:
                    self._pre_roll.pop(0)
                return None

            if self._speech_frame_count >= self.speech_start_frames:
                # Speech onset: enter LISTENING, seed with pre-roll + the
                # frames that triggered onset detection.
                self._in_speech = True
                self._silence_run = 0
                self._utterance_frames = list(self._pre_roll)
                self._utterance_speech_frames = 0
                self._pre_roll = []
                # The frames that contributed to onset detection beyond
                # pre-roll: we only have the current frame here since
                # earlier onset frames were pushed to pre-roll above.
                self._utterance_frames.append(frame)
                self._utterance_speech_frames += 1
            else:
                # Not yet confirmed speech onset; keep buffering as pre-roll.
                self._pre_roll.append(frame)
                if len(self._pre_roll) > self._pre_roll_frames:
                    self._pre_roll.pop(0)
            return None

        # In speech.
        self._utterance_frames.append(frame)
        if is_speech:
            self._silence_run = 0
            self._utterance_speech_frames += 1
        else:
            self._silence_run += 1

        if self._silence_run >= self._end_silence_frames:
            return self._finish_utterance()

        if len(self._utterance_frames) >= self._max_utterance_frames:
            return self._finish_utterance(continue_speech=True)

        return None

    def _finish_utterance(self, continue_speech=False):
        frames = self._utterance_frames
        speech_frames = self._utterance_speech_frames
        silence_run = self._silence_run

        if not continue_speech and silence_run > 0:
            # Trim the trailing end-silence window used only to detect the
            # endpoint; it isn't part of the spoken utterance.
            frames = frames[: len(frames) - silence_run] or frames

        self._in_speech = False
        self._speech_frame_count = 0
        self._silence_run = 0
        self._utterance_frames = []
        self._utterance_speech_frames = 0
        self._pre_roll = []
        self._model_reset()

        if continue_speech:
            # Hard cap hit while still speaking: keep listening immediately,
            # no pre-roll needed since audio is continuous.
            self._in_speech = True
            self._speech_frame_count = self.speech_start_frames

        speech_ms = speech_frames * self._frame_ms
        if speech_ms < self.min_speech_ms or not frames:
            return None

        return np.concatenate(frames).astype(np.float32)

    def flush(self):
        if not self._in_speech:
            return None
        result = self._finish_utterance()
        return result

    @property
    def gated(self):
        return self._gated

    @gated.setter
    def gated(self, value):
        self._gated = value
        if value:
            self._reset_utterance_state()
            self._buffer = np.zeros(0, dtype=np.float32)
