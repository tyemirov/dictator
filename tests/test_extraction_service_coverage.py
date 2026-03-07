from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())
sys.modules.setdefault("librosa", types.SimpleNamespace(feature=types.SimpleNamespace(rms=lambda y: np.array([[0.0]]))))

from dictator.diarization.models import DiarizedWord, SpeakerSegment
from dictator.extraction.models import ReferenceExtractionRequest
from dictator.extraction import service as extraction_service


class _FakePipeline:
    @classmethod
    def from_pretrained(cls, model_id):
        pipeline = cls()
        pipeline.model_id = model_id
        pipeline.device = None
        return pipeline

    def to(self, device):
        self.device = device
        return self


class ExtractionServiceCoverageTests(unittest.TestCase):
    def test_timed_and_configure_torch_runtime(self):
        with patch.object(extraction_service.logging, "info") as log_info, patch.object(extraction_service.time, "perf_counter", side_effect=[1.0, 3.5]):
            with extraction_service.timed("demo"):
                pass
        self.assertEqual(log_info.call_args_list[0].args, ("START %s", "demo"))
        self.assertEqual(log_info.call_args_list[1].args[0], "DONE  %s  (delta = %.1fs)")

        fake_torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
                cudnn=types.SimpleNamespace(allow_tf32=False),
            )
        )
        with patch.object(extraction_service, "torch", fake_torch), patch.object(extraction_service.warnings, "filterwarnings") as filterwarnings:
            extraction_service.configure_torch_runtime()
        self.assertTrue(fake_torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(fake_torch.backends.cudnn.allow_tf32)
        self.assertEqual(filterwarnings.call_count, 3)

    def test_signal_quality_helpers_and_pipeline_loading(self):
        self.assertEqual(extraction_service.spectral_centroid(np.zeros(8, dtype=np.int16)), 0.0)
        self.assertGreaterEqual(extraction_service.snr(np.array([0, 1, -1, 2], dtype=np.int16)), 0.0)

        with patch.object(extraction_service.librosa.feature, "rms", return_value=np.array([[0.5, 1.5]])):
            self.assertAlmostEqual(extraction_service.pitch_variation(np.array([0, 1], dtype=np.int16)), 0.5)

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            device=lambda name: f"device:{name}",
        )
        with (
            patch.object(extraction_service, "torch", fake_torch),
            patch.dict(sys.modules, {"pyannote.audio": types.SimpleNamespace(Pipeline=_FakePipeline)}),
            patch.object(extraction_service, "configure_torch_runtime") as configure,
        ):
            pipeline = extraction_service.load_diarization_pipeline()
        configure.assert_called_once()
        self.assertEqual(pipeline.model_id, extraction_service.DIARIZATION_MODEL)
        self.assertEqual(pipeline.device, "device:cpu")

    def test_apply_filter_choose_window_and_trim_bounds(self):
        segments = (SpeakerSegment("S1", 0.0, 2.0, "a"), SpeakerSegment("S2", 2.0, 3.0, "b"))
        with (
            patch.object(extraction_service, "run_diarization", return_value=segments),
            patch.object(
                extraction_service,
                "assign_words_to_speakers",
                return_value=(
                    DiarizedWord("hello", 0.1, 0.3, "S1"),
                    DiarizedWord("world", 2.1, 2.4, "S2"),
                ),
            ),
            patch.object(extraction_service.logging, "info") as log_info,
        ):
            filtered = extraction_service.apply_diarization_filter([{"content": "x"}], object(), Path("sample.wav"))
        self.assertEqual([word["content"] for word in filtered], ["hello"])
        self.assertEqual(log_info.call_args.args, ("dominant speaker: %s", "S1"))

        with (
            patch.object(extraction_service, "run_diarization", return_value=segments),
            patch.object(extraction_service, "assign_words_to_speakers", return_value=(DiarizedWord("world", 2.1, 2.4, "S2"),)),
        ):
            with self.assertRaisesRegex(RuntimeError, "dominant speaker"):
                extraction_service.apply_diarization_filter([{"content": "x"}], object(), Path("sample.wav"))

        pcm = np.zeros(extraction_service.SAMPLE_RATE * 3, dtype=np.int16)
        with self.assertRaisesRegex(RuntimeError, "exceeds track length"):
            extraction_service.choose_window(pcm, [], 10.0)

        with (
            patch.object(extraction_service, "spectral_centroid", return_value=1000.0),
            patch.object(extraction_service, "pitch_variation", return_value=0.5),
            patch.object(extraction_service, "snr", return_value=2.0),
            patch.object(extraction_service.logging, "info") as log_info,
        ):
            window_start = extraction_service.choose_window(
                pcm,
                [{"start": 0.1}, {"start": 1.1}, {"start": 1.9}],
                1.0,
                max_speech_rate=10.0,
                max_centroid=2000.0,
                min_centroid=10.0,
            )
        self.assertEqual(window_start, 1.0)
        self.assertEqual(log_info.call_args.args[0], "chosen window: %d words, score %.2f")

        with (
            patch.object(extraction_service, "spectral_centroid", return_value=0.0),
            patch.object(extraction_service, "pitch_variation", return_value=0.0),
            patch.object(extraction_service, "snr", return_value=0.0),
        ):
            with self.assertRaisesRegex(RuntimeError, "no suitable window"):
                extraction_service.choose_window(pcm, [{"start": 0.1}], 1.0)

        with self.assertRaisesRegex(RuntimeError, "no words found"):
            extraction_service.compute_trim_bounds(5.0, [])
        self.assertEqual(
            extraction_service.compute_trim_bounds(
                5.0,
                [{"start": 0.1, "end": 0.5}, {"start": 1.0, "end": 4.9}],
            ),
            (0.0, 5.0),
        )

    def test_reference_extraction_service_paths(self):
        service = extraction_service.ReferenceExtractionService()
        pcm = np.zeros(extraction_service.SAMPLE_RATE * 5, dtype=np.int16)
        request = ReferenceExtractionRequest(
            input_path=Path("sample.wav"),
            output_path=None,
            duration_seconds=1.0,
        )

        with (
            patch.object(extraction_service, "configure_torch_runtime"),
            patch.object(extraction_service, "decode_pcm", return_value=pcm),
            patch.object(extraction_service, "load_diarization_pipeline", return_value="pipeline"),
            patch.object(extraction_service, "load_whisper_model", return_value="model"),
            patch.object(extraction_service, "transcribe_words", return_value=[]),
        ):
            with self.assertRaisesRegex(RuntimeError, "no words transcribed"):
                service.extract(request)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "reference.wav"
            request = ReferenceExtractionRequest(
                input_path=Path("sample.wav"),
                output_path=output,
                duration_seconds=1.0,
            )
            raw_words = [{"content": "hello", "start": 0.2, "end": 0.6}]
            dominant = [{"content": "hello", "start": 0.2, "end": 0.6}]
            with (
                patch.object(extraction_service, "configure_torch_runtime"),
                patch.object(extraction_service, "decode_pcm", return_value=pcm),
                patch.object(extraction_service, "transcribe_words", return_value=raw_words),
                patch.object(extraction_service, "apply_diarization_filter", return_value=dominant),
                patch.object(extraction_service, "choose_window", return_value=0.0),
                patch.object(extraction_service, "compute_trim_bounds", return_value=(0.0, 1.0)),
                patch.object(extraction_service, "trim_and_normalise") as trim_and_normalise,
            ):
                result = service.extract(request, model="model", diarization_pipeline="pipeline")

        trim_and_normalise.assert_called_once_with(Path("sample.wav"), output, 0.0, 1.0)
        self.assertEqual(result.window_end_seconds, 1.0)
        self.assertEqual(result.trim_end_seconds, 1.0)
        self.assertEqual(result.raw_words, tuple(raw_words))


if __name__ == "__main__":
    unittest.main()
