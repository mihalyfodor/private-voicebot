import sys, os, io, base64
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import soundfile as sf
import numpy as np

import chatbot


def fake_tts(text):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(2400, dtype=np.float32), 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_speak_stream_emits_speech_per_sentence_then_end():
    sent = []
    reply = chatbot.speak_stream(["[happy] Sure thing.", " Anything else?"], sent.append, tts=fake_tts)

    assert reply == "Sure thing. Anything else?"
    types = [m["type"] for m in sent]
    assert types == ["state", "speech", "speech", "speech_end"]
    assert sent[0]["value"] == "speaking"
    assert [m["text"] for m in sent[1:3]] == ["Sure thing.", "Anything else?"]
    assert all(m["emotion"] == "happy" for m in sent[1:3])
    for m in sent[1:3]:
        data, sr = sf.read(io.BytesIO(base64.b64decode(m["wav"])))
        assert sr == 24000 and len(data) == 2400
