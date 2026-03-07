from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))

from dictator.transcription.models import TranscriptionResult, WordSegment
from dictator.transcription.service import (
    TranscriptionService,
    _coerce_audio_input,
    load_whisper_model,
    serialise_word_segments,
    transcribe,
    transcribe_text,
    transcribe_word_segments,
    transcribe_words,
)


class _FakeModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return self.result


class TranscriptionServiceCoverageTests(unittest.TestCase):
    def test_load_whisper_model_uses_device_and_cache_dir(self):
        whisper_module = types.SimpleNamespace(load_model=lambda *args, **kwargs: (args, kwargs))
        with (
            patch.dict(sys.modules, {"whisper": whisper_module}),
            patch.object(__import__("dictator.transcription.service", fromlist=["torch"]).torch, "cuda", types.SimpleNamespace(is_available=lambda: True), create=True),
        ):
            args, kwargs = load_whisper_model("small", cache_dir=Path("/tmp/cache"))
        self.assertEqual(args, ("small",))
        self.assertEqual(kwargs["device"], "cuda")
        self.assertEqual(kwargs["download_root"], "/tmp/cache")

    def test_coerce_audio_input_handles_path_array_and_empty_array(self):
        self.assertEqual(_coerce_audio_input(Path("sample.wav")), "sample.wav")
        audio = np.array([0, 32767], dtype=np.int16)
        coerced = _coerce_audio_input(audio)
        self.assertEqual(coerced.dtype, np.float32)
        self.assertAlmostEqual(coerced[0], 0.0)
        self.assertAlmostEqual(coerced[1], 32767.0 / 32768.0)
        with self.assertRaisesRegex(ValueError, "audio array is empty"):
            _coerce_audio_input(np.array([], dtype=np.int16))

    def test_transcribe_detects_language_invokes_progress_and_overrides_language(self):
        result = {
            "language": "fr",
            "segments": [
                {
                    "end": 1.0,
                    "words": [
                        {"word": " Hello ", "start": 0.0, "end": 0.4},
                        {"word": "world", "start": 0.4, "end": 0.8},
                    ],
                }
            ],
        }
        model = _FakeModel(result)
        progress = []
        transcription = transcribe(Path("sample.wav"), model=model, progress_cb=progress.append)
        self.assertEqual(transcription.language, "fr")
        self.assertEqual(transcription.text, "Hello world")
        self.assertEqual(progress, [1.0])
        self.assertEqual(model.calls[0][1]["word_timestamps"], True)
        self.assertNotIn("language", model.calls[0][1])

        model = _FakeModel(result)
        transcription = transcribe(Path("sample.wav"), language="en", model=model)
        self.assertEqual(transcription.language, "en")
        self.assertEqual(model.calls[0][1]["language"], "en")

    def test_transcribe_loads_default_model_when_not_provided(self):
        model = _FakeModel({"segments": [], "language": "en"})
        with patch("dictator.transcription.service.load_whisper_model", return_value=model) as loader_mock:
            transcription = transcribe(Path("sample.wav"))
        loader_mock.assert_called_once_with("base")
        self.assertEqual(transcription.language, "en")

    def test_transcription_service_uses_loader_and_helpers(self):
        loader_calls = []
        model = _FakeModel({"segments": [], "language": "en"})

        def loader(model_size):
            loader_calls.append(model_size)
            return model

        service = TranscriptionService(model_loader=loader)
        result = service.transcribe(Path("audio.wav"), model_size="tiny")
        self.assertEqual(result, TranscriptionResult(language="en", words=()))
        words = service.transcribe_word_segments(Path("audio.wav"), model_size="tiny")
        self.assertEqual(words, [])
        self.assertEqual(service.transcribe_words(Path("audio.wav"), model_size="tiny"), [])
        self.assertEqual(service.transcribe_text(Path("audio.wav"), model_size="tiny"), "")
        self.assertEqual(loader_calls, ["tiny", "tiny", "tiny", "tiny"])

    def test_top_level_wrappers_and_serialization(self):
        words = [WordSegment("hello", 0.0, 0.4), WordSegment("world", None, None)]
        self.assertEqual(
            serialise_word_segments(words),
            [
                {"content": "hello", "start": 0.0, "end": 0.4},
                {"content": "world", "start": None, "end": None},
            ],
        )
        with patch("dictator.transcription.service.transcribe", return_value=TranscriptionResult("en", tuple(words))):
            self.assertEqual(transcribe_word_segments(Path("a.wav")), words)
            self.assertEqual(transcribe_words(Path("a.wav"))[0]["content"], "hello")
            self.assertEqual(transcribe_text(Path("a.wav")), "hello world")


if __name__ == "__main__":
    unittest.main()
