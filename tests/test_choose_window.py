import sys
import types
import unittest
import numpy as np
import importlib
import unittest
from unittest.mock import patch


class ChooseWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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

        class _DummyPipeline:
            @staticmethod
            def from_pretrained(_):
                return _DummyPipeline()

            def to(self, _):
                return self

        pyannote_audio = types.SimpleNamespace(Pipeline=_DummyPipeline)

        class _DummyWhisperModel:
            def transcribe(self, *args, **kwargs):
                return {"segments": []}

        whisper_stub = types.SimpleNamespace(load_model=lambda *a, **k: _DummyWhisperModel())

        librosa_stub = types.ModuleType("librosa")
        librosa_stub.feature = types.SimpleNamespace(rms=lambda y: np.array([[0.0]]))

        cls.patcher = patch.dict(
            sys.modules,
            {
                "torch": torch_stub,
                "pyannote": types.ModuleType("pyannote"),
                "pyannote.audio": pyannote_audio,
                "whisper": whisper_stub,
                "librosa": librosa_stub,
                "ffmpeg": types.ModuleType("ffmpeg"),
            },
        )
        cls.patcher.start()
        import extract as _extract
        cls.extract = importlib.reload(_extract)
        cls.choose_window = staticmethod(cls.extract.choose_window)
        cls.spectral_centroid = cls.extract.spectral_centroid
        cls.pitch_variation = cls.extract.pitch_variation
        cls.snr = cls.extract.snr
        cls.STRIDE_SEC = cls.extract.STRIDE_SEC
        cls.SAMPLE_RATE = cls.extract.SAMPLE_RATE
        cls.MAX_SPEECH_RATE = cls.extract.MAX_SPEECH_RATE

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    @classmethod
    def choose_window_original(
        cls,
        pcm_array,
        speaker_words,
        duration,
        max_speech_rate,
        max_centroid,
        min_centroid,
    ):
        best_score, best_word_count, best_window_start = -1.0, -1, 0.0
        track_length = len(pcm_array) / cls.SAMPLE_RATE
        position = 0.0
        while position + duration <= track_length:
            chunk = pcm_array[
                int(position * cls.SAMPLE_RATE) : int((position + duration) * cls.SAMPLE_RATE)
            ]
            centroid = cls.spectral_centroid(chunk)
            if centroid > max_centroid or centroid < min_centroid:
                position += cls.STRIDE_SEC
                continue
            words_in_window = [
                w for w in speaker_words if position <= w["start"] < position + duration
            ]
            word_count = len(words_in_window)
            if word_count == 0:
                position += cls.STRIDE_SEC
                continue
            if word_count / duration > max_speech_rate:
                position += cls.STRIDE_SEC
                continue
            average_confidence = (
                sum(
                    w["probability"]
                    for w in speaker_words
                    if position <= w["start"] < position + duration
                )
                / word_count
            )
            variation = cls.pitch_variation(chunk)
            quality_score = average_confidence * cls.snr(chunk) * (1.0 + variation)
            score = word_count * quality_score
            if score > best_score:
                best_score, best_word_count, best_window_start = (
                    score,
                    word_count,
                    position,
                )
            position += cls.STRIDE_SEC
        if best_score < 0:
            raise RuntimeError("no suitable window found")
        return best_window_start

    def test_choose_window_identical(self):
        np.random.seed(0)
        pcm = np.random.randint(-2000, 2000, self.SAMPLE_RATE * 6, dtype=np.int16)
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
        expected = self.choose_window_original(
            pcm, speaker_words, duration, self.MAX_SPEECH_RATE, max_centroid, min_centroid
        )
        result = self.choose_window(
            pcm,
            speaker_words,
            duration,
            max_speech_rate=self.MAX_SPEECH_RATE,
            max_centroid=max_centroid,
            min_centroid=min_centroid,
        )
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

