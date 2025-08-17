import sys
import types
import numpy as np


# Stub heavy dependencies before importing extract
class _DummyTensor:
    def __init__(self, array):
        self.array = array

    def unsqueeze(self, _):
        return self


torch_stub = types.SimpleNamespace(
    backends=types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    ),
    device=lambda *args, **kwargs: None,
    cuda=types.SimpleNamespace(is_available=lambda: False),
    from_numpy=lambda arr: _DummyTensor(arr),
)
sys.modules["torch"] = torch_stub


class _DummyPipeline:
    @staticmethod
    def from_pretrained(_):
        return _DummyPipeline()

    def to(self, _):
        return self


pyannote_audio = types.SimpleNamespace(Pipeline=_DummyPipeline)
sys.modules["pyannote"] = types.ModuleType("pyannote")
sys.modules["pyannote.audio"] = pyannote_audio


class _DummyWhisperModel:
    def transcribe(self, *args, **kwargs):
        return {"segments": []}


whisper_stub = types.SimpleNamespace(load_model=lambda *a, **k: _DummyWhisperModel())
sys.modules["whisper"] = whisper_stub


# librosa stub for spectral centroid / RMS
librosa_stub = types.ModuleType("librosa")
librosa_stub.feature = types.SimpleNamespace(rms=lambda y: np.array([[0.0]]))
sys.modules["librosa"] = librosa_stub


from extract import (
    choose_window,
    spectral_centroid,
    pitch_variation,
    snr,
    STRIDE_SEC,
    SAMPLE_RATE,
    MAX_SPEECH_RATE,
)


def choose_window_original(
    pcm_array,
    speaker_words,
    duration,
    max_speech_rate,
    max_centroid,
    min_centroid,
):
    best_score, best_word_count, best_window_start = -1.0, -1, 0.0
    track_length = len(pcm_array) / SAMPLE_RATE
    position = 0.0
    while position + duration <= track_length:
        chunk = pcm_array[int(position * SAMPLE_RATE):int((position + duration) * SAMPLE_RATE)]
        centroid = spectral_centroid(chunk)
        if centroid > max_centroid or centroid < min_centroid:
            position += STRIDE_SEC
            continue
        words_in_window = [w for w in speaker_words if position <= w["start"] < position + duration]
        word_count = len(words_in_window)
        if word_count == 0:
            position += STRIDE_SEC
            continue
        if word_count / duration > max_speech_rate:
            position += STRIDE_SEC
            continue
        average_confidence = sum(
            w["probability"] for w in speaker_words if position <= w["start"] < position + duration
        ) / word_count
        variation = pitch_variation(chunk)
        quality_score = average_confidence * snr(chunk) * (1.0 + variation)
        score = word_count * quality_score
        if score > best_score:
            best_score, best_word_count, best_window_start = score, word_count, position
        position += STRIDE_SEC
    if best_score < 0:
        raise RuntimeError("no suitable window found")
    return best_window_start


def test_choose_window_identical():
    np.random.seed(0)
    pcm = np.random.randint(-2000, 2000, SAMPLE_RATE * 6, dtype=np.int16)
    speaker_words = [
        {"start": 0.1, "probability": 0.9},
        {"start": 1.2, "probability": 0.8},
        {"start": 2.5, "probability": 0.95},
        {"start": 3.3, "probability": 0.85},
        {"start": 4.1, "probability": 0.9},
    ]
    duration = 2.0
    min_centroid = 0.0
    max_centroid = 10000.0
    expected = choose_window_original(
        pcm, speaker_words, duration, MAX_SPEECH_RATE, max_centroid, min_centroid
    )
    result = choose_window(
        pcm,
        speaker_words,
        duration,
        max_speech_rate=MAX_SPEECH_RATE,
        max_centroid=max_centroid,
        min_centroid=min_centroid,
    )
    assert result == expected
