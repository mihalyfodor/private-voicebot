import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

import vad


SR = 16000


def tone_int16(duration_s, freq=440, amplitude=0.3, sample_rate=SR):
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * freq * t)
    return (signal * 32767).astype(np.int16)


def silence_int16(duration_s, sample_rate=SR):
    n = int(duration_s * sample_rate)
    return np.zeros(n, dtype=np.int16)


def rms_prob_fn(frame_f32):
    """Fake VAD: RMS above a small floor => 'speech' (0.9), else 'silence' (0.0)."""
    rms = float(np.sqrt(np.mean(frame_f32.astype(np.float64) ** 2)))
    return 0.9 if rms > 0.05 else 0.0


def make_endpointer(**kwargs):
    kwargs.setdefault("prob_fn", rms_prob_fn)
    return vad.Endpointer(**kwargs)


def feed_all(ep, pcm, chunk_size=512):
    """Feed pcm to ep in chunks, collecting all emitted utterances."""
    out = []
    for i in range(0, len(pcm), chunk_size):
        out.extend(ep.feed(pcm[i : i + chunk_size]))
    return out


# Scenario 1a: 1s silence -> no utterance.
def test_silence_only_yields_nothing():
    ep = make_endpointer()
    result = feed_all(ep, silence_int16(1.0))
    assert result == []


# Scenario 1b: 0.2s tone then 1s silence -> discarded (below min_speech_ms).
def test_short_speech_below_min_is_discarded():
    ep = make_endpointer()
    pcm = np.concatenate([tone_int16(0.2), silence_int16(1.0)])
    result = feed_all(ep, pcm)
    assert result == []


# Scenario 2: 1.5s tone, 0.5s pause, 1s tone, 0.8s silence -> one utterance ~3s.
def test_mid_utterance_pause_is_tolerated():
    ep = make_endpointer()
    pcm = np.concatenate(
        [
            tone_int16(1.5),
            silence_int16(0.5),
            tone_int16(1.0),
            silence_int16(0.8),
        ]
    )
    result = feed_all(ep, pcm)
    assert len(result) == 1
    duration = len(result[0]) / SR
    assert 2.9 <= duration <= 3.6
    assert result[0].dtype == np.float32


# Scenario 3: 40s continuous tone -> first utterance emitted at ~max_utterance_s.
def test_long_utterance_emitted_at_max_cap():
    ep = make_endpointer(max_utterance_s=30)
    pcm = tone_int16(40.0)
    result = feed_all(ep, pcm)
    assert len(result) >= 1
    duration = len(result[0]) / SR
    assert 29.0 <= duration <= 31.0


# Scenario 4: gated -> nothing emitted; ungated afterwards -> works.
def test_gated_drops_audio_then_recovers():
    ep = make_endpointer()
    ep.gated = True
    pcm = np.concatenate([tone_int16(1.5), silence_int16(0.8)])
    result = feed_all(ep, pcm)
    assert result == []

    ep.gated = False
    result = feed_all(ep, pcm)
    assert len(result) == 1


def test_hearing_property_tracks_state():
    ep = make_endpointer()
    assert ep.hearing is False
    ep.feed(tone_int16(1.0))
    assert ep.hearing is True
    ep.feed(silence_int16(1.0))
    assert ep.hearing is False


def test_flush_mid_utterance_returns_audio():
    ep = make_endpointer()
    ep.feed(tone_int16(1.0))
    assert ep.hearing is True
    audio = ep.flush()
    assert audio is not None
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) > 0
    assert ep.hearing is False


def test_flush_when_idle_returns_none():
    ep = make_endpointer()
    assert ep.flush() is None


def test_odd_sized_chunks_behave_like_512():
    pcm = np.concatenate(
        [
            tone_int16(1.5),
            silence_int16(0.5),
            tone_int16(1.0),
            silence_int16(0.8),
        ]
    )
    ep_512 = make_endpointer()
    result_512 = feed_all(ep_512, pcm, chunk_size=512)

    ep_odd = make_endpointer()
    result_odd = feed_all(ep_odd, pcm, chunk_size=333)

    assert len(result_512) == len(result_odd) == 1
    assert abs(len(result_512[0]) - len(result_odd[0])) <= 512


def test_recorder_accumulates_and_stops():
    rec = vad.Recorder()
    rec.feed(tone_int16(0.5))
    rec.feed(silence_int16(0.5))
    audio = rec.stop()
    assert audio.dtype == np.float32
    assert len(audio) == int(1.0 * SR)
    # stop() again after no more feeding returns empty
    assert len(rec.stop()) == 0


def test_defaults_dict_present():
    for key in (
        "threshold",
        "end_silence_ms",
        "min_speech_ms",
        "pre_roll_ms",
        "max_utterance_s",
    ):
        assert key in vad.DEFAULTS
