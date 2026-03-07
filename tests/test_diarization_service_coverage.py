from __future__ import annotations

import builtins
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())

from dictator.diarization.models import (
    DiarizeAudioRequest,
    DiarizedUtterance,
    DiarizedWord,
    SpeakerSegment,
)
from dictator.diarization.service import (
    DiarizationService,
    _best_speaker_segment,
    _coerce_word_bounds,
    _speaker_distance,
    build_utterances,
    dominant_speaker_label,
    run_diarization,
)
from dictator.runtime import DependencyError, ProcessingError, ValidationError
from dictator.transcription.models import TranscriptionResult, WordSegment


class _DummyTensor:
    def __init__(self, array):
        self.array = array
        self.unsqueeze_calls = []

    def unsqueeze(self, dim):
        self.unsqueeze_calls.append(dim)
        return self


class _Turn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _DiarizationResult:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        for item in self._tracks:
            yield item


class _FakeTranscriptionService:
    def __init__(self, result: TranscriptionResult):
        self.result = result
        self.calls = []

    def transcribe(self, audio, language=None, model_size="base", model=None):
        self.calls.append((audio, language, model_size, model))
        return self.result


class DiarizationServiceCoverageTests(unittest.TestCase):
    def test_word_bounds_and_speaker_selection_helpers(self):
        self.assertEqual(_coerce_word_bounds({"start": 1.0}), (1.0, 1.0))
        self.assertEqual(_coerce_word_bounds({"end": 2.0}), (2.0, 2.0))
        self.assertEqual(_coerce_word_bounds({"start": 3.0, "end": 2.0}), (3.0, 3.0))
        with self.assertRaisesRegex(ProcessingError, "timestamps are required"):
            _coerce_word_bounds({})

        segment = SpeakerSegment("S1", 0.0, 1.0)
        self.assertEqual(_speaker_distance(0.5, segment), 0.0)
        self.assertEqual(_speaker_distance(2.0, segment), 1.0)

        with self.assertRaisesRegex(ProcessingError, "no speakers detected"):
            _best_speaker_segment(0.0, 0.5, ())

        chosen = _best_speaker_segment(
            0.4,
            0.8,
            (
                SpeakerSegment("S1", 0.0, 0.5),
                SpeakerSegment("S2", 0.5, 1.5),
            ),
        )
        self.assertEqual(chosen.speaker, "S2")

    def test_build_utterances_and_dominant_speaker_edge_cases(self):
        with self.assertRaisesRegex(ValidationError, "non-negative"):
            build_utterances((), utterance_gap_seconds=-1.0)
        self.assertEqual(build_utterances(()), ())

        words = (
            DiarizedWord("hello", 0.0, 0.2, "S1"),
            DiarizedWord("world", 0.1, 0.4, "S1"),
        )
        utterances = build_utterances(words)
        self.assertEqual(utterances[0].text, "hello world")
        self.assertEqual(utterances[0].to_json_dict(include_words=False), {
            "speaker": "S1",
            "start": 0.0,
            "end": 0.4,
            "text": "hello world",
        })
        self.assertEqual(words[0].to_legacy_dict()["content"], "hello")

        with self.assertRaisesRegex(ProcessingError, "no speakers detected"):
            dominant_speaker_label(())

    def test_run_diarization_handles_dependency_missing_empty_and_success(self):
        original_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("missing numpy")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            with self.assertRaisesRegex(DependencyError, "required for diarization"):
                run_diarization(object(), Path("sample.wav"))

        fake_torch = types.SimpleNamespace(from_numpy=lambda array: _DummyTensor(array))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("dictator.audio.ffmpeg_ops.decode_pcm", return_value=np.array([0, 1], dtype=np.int16)):
                empty_pipeline = lambda payload: _DiarizationResult([])
                with self.assertRaisesRegex(ProcessingError, "no speakers detected"):
                    run_diarization(empty_pipeline, Path("sample.wav"))

                seen = {}

                def pipeline(payload):
                    seen.update(payload)
                    return _DiarizationResult(
                        [
                            (_Turn(0.0, 1.0), None, "speaker_a"),
                            (_Turn(1.0, 2.0), None, "speaker_b"),
                        ]
                    )

                segments = run_diarization(pipeline, Path("sample.wav"))

        self.assertEqual([segment.speaker for segment in segments], ["S1", "S2"])
        self.assertEqual(seen["uri"], "sample")
        self.assertEqual(seen["sample_rate"], 16000)
        self.assertIsInstance(seen["waveform"], _DummyTensor)
        self.assertEqual(seen["waveform"].unsqueeze_calls, [0])

    def test_diarization_service_default_constructor_loader_and_success(self):
        fake_transcription_result = TranscriptionResult(
            language="en",
            words=(WordSegment("hello", 0.0, 0.4), WordSegment("world", 0.5, 0.9)),
        )
        fake_transcription = _FakeTranscriptionService(fake_transcription_result)
        service = DiarizationService(
            transcription_service=fake_transcription,
            diarization_pipeline_loader=lambda: "loaded-pipeline",
        )

        with patch("dictator.diarization.service.run_diarization", return_value=(SpeakerSegment("S1", 0.0, 1.0, "a"),)):
            result = service.diarize(
                DiarizeAudioRequest(
                    input_path=Path("sample.wav"),
                    language="en",
                    utterance_gap_seconds=0.2,
                ),
                model="model",
            )

        self.assertEqual(result.text, "hello world")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.words[0].speaker, "S1")
        self.assertEqual(fake_transcription.calls[0], (Path("sample.wav"), "en", "base", "model"))
        self.assertIn("speakers", result.to_json_dict(include_speaker_segments=True))

        with self.assertRaisesRegex(ValidationError, "non-negative"):
            service.diarize(DiarizeAudioRequest(input_path=Path("sample.wav"), utterance_gap_seconds=-1.0))

        no_loader_service = DiarizationService(transcription_service=fake_transcription)
        with self.assertRaisesRegex(DependencyError, "pipeline loader is required"):
            no_loader_service._load_pipeline()

        with patch("dictator.transcription.service.TranscriptionService", return_value=fake_transcription):
            default_service = DiarizationService()
        self.assertIs(default_service._transcription_service, fake_transcription)


if __name__ == "__main__":
    unittest.main()
