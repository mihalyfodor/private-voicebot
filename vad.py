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
        audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
        self._chunks = []
        return audio


class Endpointer:
    """Streaming VAD endpointer: PCM in, whole utterances out."""

    def __init__(
        self,
        sample_rate=DEFAULTS["sample_rate"],
        threshold=DEFAULTS["threshold"],
        end_silence_ms=DEFAULTS["end_silence_ms"],
        min_speech_ms=DEFAULTS["min_speech_ms"],
        pre_roll_ms=DEFAULTS["pre_roll_ms"],
        max_utterance_s=DEFAULTS["max_utterance_s"],
        speech_start_frames=DEFAULTS["speech_start_frames"],
        prob_fn=None,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.end_silence_ms = end_silence_ms
        self.min_speech_ms = min_speech_ms
        self.pre_roll_ms = pre_roll_ms
        self.max_utterance_s = max_utterance_s
        self.speech_start_frames = speech_start_frames
        self._prob_fn = prob_fn

        self._frame_ms = FRAME_SAMPLES / sample_rate * 1000.0
        self._pre_roll_frames = max(0, round(self.pre_roll_ms / self._frame_ms))
        self._end_silence_frames = max(1, round(self.end_silence_ms / self._frame_ms))
        self._max_utterance_frames = max(1, round(self.max_utterance_s * 1000.0 / self._frame_ms))

        self._gated = False
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
    def gated(self):
        return self._gated

    @gated.setter
    def gated(self, value):
        """Ignore incoming audio (while the bot speaks), dropping any utterance in progress."""
        self._gated = value
        if value:
            self.reset()

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
        """Feed PCM (any length, int16 or float32); returns the utterances it completed."""
        if self.gated:
            self.reset()  # consume but ignore; drop any in-progress utterance
            return []

        self._buffer = np.concatenate([self._buffer, _to_float32(pcm)])
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
            self._speech_frame_count = self._speech_frame_count + 1 if is_speech else 0
            if self._speech_frame_count >= self.speech_start_frames:
                # Speech onset: enter LISTENING, seeded with the pre-roll (which
                # already holds the earlier onset frames) plus this frame.
                self._in_speech = True
                self._silence_run = 0
                self._utterance_frames = self._pre_roll + [frame]
                self._utterance_speech_frames = 1
                self._pre_roll = []
            else:
                # Not speech, or onset not confirmed yet: keep it as pre-roll.
                self._pre_roll.append(frame)
                del self._pre_roll[: len(self._pre_roll) - self._pre_roll_frames]
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

        self._reset_utterance_state()
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
        """End the utterance in progress, if any, and return it."""
        if not self._in_speech:
            return None
        return self._finish_utterance()

