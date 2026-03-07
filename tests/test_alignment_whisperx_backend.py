from __future__ import annotations

import importlib
from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from dictator.alignment import whisperx_backend as backend
from dictator.runtime import DependencyError, ProcessingError, ValidationError


class FakeAlignmentModule:
    def __init__(self):
        self.load_calls = []
        self.align_calls = []
        self.load_results = []

    def load_align_model(self, language_code, device):
        self.load_calls.append((language_code, device))
        if self.load_results:
            result = self.load_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return (f"model:{language_code}:{device}", {"language": language_code})

    def align(self, segments, align_model, metadata, audio, resolved_device, return_char_alignments=False):
        self.align_calls.append((segments, align_model, metadata, audio, resolved_device, return_char_alignments))
        return {"segments": [{"words": [{"word": "hello", "start": 0.0, "end": 0.4}]}]}


class AlignmentWhisperXBackendTests(unittest.TestCase):
    def setUp(self):
        backend.clear_alignment_model_cache()

    def tearDown(self):
        backend.clear_alignment_model_cache()

    def test_version_and_dependency_helpers(self):
        self.assertEqual(backend.parse_torch_version("2.6.1"), (2, 6))
        self.assertIsNone(backend.parse_torch_version("2"))
        self.assertIsNone(backend.parse_torch_version("x.y"))

        good_torch = types.SimpleNamespace(__version__="2.6.0")
        backend.ensure_torch_version(good_torch, "ja")
        with self.assertRaisesRegex(DependencyError, "invalid"):
            backend.ensure_torch_version(types.SimpleNamespace(__version__="bad"), "ja")
        with self.assertRaisesRegex(DependencyError, ">= 2.6"):
            backend.ensure_torch_version(types.SimpleNamespace(__version__="2.5.0"), "ja")

        with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
            self.assertEqual(backend.load_torch_module().__name__, "torch")

        original_import = __import__
        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(DependencyError, "torch is unavailable"):
                backend.load_torch_module()

    def test_ensure_torchaudio_metadata_and_module_loading(self):
        torchaudio = types.ModuleType("torchaudio")
        torchaudio.AudioMetaData = object
        with patch.dict(sys.modules, {"torchaudio": torchaudio}):
            backend.ensure_torchaudio_metadata()

        original_import = __import__
        def fake_import(name, *args, **kwargs):
            if name == "torchaudio":
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(DependencyError, "torchaudio is unavailable"):
                backend.ensure_torchaudio_metadata()

        torchaudio = types.ModuleType("torchaudio")
        with (
            patch.dict(sys.modules, {"torchaudio": torchaudio}),
            patch("dictator.alignment.whisperx_backend.importlib.import_module") as import_module_mock,
        ):
            import_module_mock.side_effect = [types.SimpleNamespace(AudioMetaData="meta")]
            backend.ensure_torchaudio_metadata()
        self.assertEqual(torchaudio.AudioMetaData, "meta")

        torchaudio = types.ModuleType("torchaudio")
        with (
            patch.dict(sys.modules, {"torchaudio": torchaudio}),
            patch("dictator.alignment.whisperx_backend.importlib.import_module", side_effect=ImportError("nope")),
        ):
            with self.assertRaisesRegex(DependencyError, "missing AudioMetaData"):
                backend.ensure_torchaudio_metadata()

        with patch("dictator.alignment.whisperx_backend.ensure_torchaudio_metadata"):
            with patch("dictator.alignment.whisperx_backend.importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(DependencyError, "whisperx is unavailable"):
                    backend.load_whisperx_alignment_modules()
            spec = ModuleSpec("whisperx", loader=None, is_package=True)
            spec.submodule_search_locations = ["/tmp"]
            with patch("dictator.alignment.whisperx_backend.importlib.util.find_spec", return_value=spec):
                with patch("dictator.alignment.whisperx_backend.importlib.import_module", side_effect=[types.SimpleNamespace(), types.SimpleNamespace(SAMPLE_RATE=16000)]):
                    alignment_module, audio_module = backend.load_whisperx_alignment_modules()
                self.assertEqual(audio_module.SAMPLE_RATE, 16000)
            with patch("dictator.alignment.whisperx_backend.importlib.util.find_spec", return_value=spec):
                with patch("dictator.alignment.whisperx_backend.importlib.import_module", side_effect=RuntimeError("broken")):
                    with self.assertRaisesRegex(DependencyError, "alignment import failed"):
                        backend.load_whisperx_alignment_modules()

    def test_torch_cache_directory_helpers(self):
        self.assertTrue(backend.is_corrupted_torch_archive_error("failed finding central directory"))
        self.assertFalse(backend.is_corrupted_torch_archive_error("other"))

        with patch.dict("os.environ", {backend.TORCH_HOME_ENV: "/tmp/torch-home"}, clear=True):
            self.assertEqual(backend.resolve_torch_hub_checkpoints_dir(), Path("/tmp/torch-home") / "hub" / "checkpoints")

        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("pathlib.Path.home", return_value=home),
            ):
                self.assertEqual(backend.clear_torch_hub_checkpoints(), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir)
            checkpoints = home / ".cache" / "torch" / "hub" / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "a.bin").write_text("x", encoding="utf-8")
            (checkpoints / "subdir").mkdir()
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("pathlib.Path.home", return_value=home),
            ):
                self.assertEqual(backend.clear_torch_hub_checkpoints(), 1)
                self.assertEqual(backend.clear_torch_hub_checkpoints(), 0)

    def test_model_key_and_model_loading_helpers(self):
        self.assertEqual(backend.normalize_alignment_model_key(" EN ", " CPU "), ("en", "cpu"))
        with self.assertRaises(ValidationError):
            backend.normalize_alignment_model_key("", "cpu")
        with self.assertRaises(ValidationError):
            backend.normalize_alignment_model_key("xx", "cpu")
        with self.assertRaises(ValidationError):
            backend.normalize_alignment_model_key("en", "")

        alignment_module = FakeAlignmentModule()
        result = backend.load_alignment_model("en", "cpu", alignment_module)
        self.assertEqual(result[0], "model:en:cpu")

        alignment_module = FakeAlignmentModule()
        alignment_module.load_results = [RuntimeError("PytorchStreamReader failed reading zip archive"), ("model", {})]
        with (
            patch("dictator.alignment.whisperx_backend.clear_torch_hub_checkpoints", return_value=2) as clear_mock,
            patch(
                "dictator.alignment.whisperx_backend.load_torch_module",
                return_value=types.SimpleNamespace(__version__="2.6.0"),
            ),
        ):
            result = backend.load_alignment_model("ja", "cpu", alignment_module)
        clear_mock.assert_called_once()
        self.assertEqual(result, ("model", {}))

        alignment_module = FakeAlignmentModule()
        alignment_module.load_results = [RuntimeError("failed finding central directory"), RuntimeError("still bad")]
        with (
            patch("dictator.alignment.whisperx_backend.clear_torch_hub_checkpoints", return_value=1),
            patch(
                "dictator.alignment.whisperx_backend.load_torch_module",
                return_value=types.SimpleNamespace(__version__="2.6.0"),
            ),
        ):
            with self.assertRaisesRegex(ProcessingError, "after checkpoint cache reset"):
                backend.load_alignment_model("ja", "cpu", alignment_module)

        alignment_module = FakeAlignmentModule()
        alignment_module.load_results = [RuntimeError("boom")]
        with patch(
            "dictator.alignment.whisperx_backend.load_torch_module",
            return_value=types.SimpleNamespace(__version__="2.6.0"),
        ):
            with self.assertRaisesRegex(ProcessingError, "align model load failed: boom"):
                backend.load_alignment_model("ja", "cpu", alignment_module)

    def test_cached_loading_device_and_preload(self):
        alignment_module = FakeAlignmentModule()
        first = backend.load_cached_alignment_model("en", "cpu", alignment_module)
        second = backend.load_cached_alignment_model("en", "cpu", alignment_module)
        self.assertIs(first, second)
        self.assertEqual(alignment_module.load_calls, [("en", "cpu")])
        backend.clear_alignment_model_cache()

        sentinel = object()
        key = ("en", "cpu")

        class InjectingRegistryLock:
            def __init__(self):
                self.enter_count = 0

            def __enter__(self):
                self.enter_count += 1
                if self.enter_count == 2:
                    backend._alignment_model_cache[key] = sentinel
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with patch("dictator.alignment.whisperx_backend._alignment_model_registry_lock", InjectingRegistryLock()):
            cached = backend.load_cached_alignment_model("en", "cpu", alignment_module)
        self.assertIs(cached, sentinel)

        torch_module = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
        with patch("dictator.alignment.whisperx_backend.load_torch_module", return_value=torch_module):
            self.assertEqual(backend.resolve_device("auto"), "cuda")
        with patch("dictator.alignment.whisperx_backend.load_torch_module", return_value=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))):
            self.assertEqual(backend.resolve_device("auto"), "cpu")
        self.assertEqual(backend.resolve_device("CPU"), "cpu")

        with (
            patch("dictator.alignment.whisperx_backend.load_whisperx_alignment_modules", return_value=(alignment_module, object())),
            patch("dictator.alignment.whisperx_backend.resolve_device", return_value="cpu") as resolve_mock,
            patch("dictator.alignment.whisperx_backend.load_cached_alignment_model") as load_cached_mock,
        ):
            backend.preload_alignment_model("en")
        resolve_mock.assert_called_once_with("auto")
        load_cached_mock.assert_called_once_with("en", "cpu", alignment_module)

    def test_timestamp_and_bounds_helpers(self):
        self.assertIsNone(backend.coerce_timestamp(True))
        self.assertEqual(backend.coerce_timestamp(1), 1.0)
        self.assertIsNone(backend.coerce_timestamp(float("inf")))
        self.assertEqual(backend.default_segment_bounds(2, None), (0.0, 0.5))
        self.assertEqual(backend.default_segment_bounds(2, 1.0), (1.0, 1.5))
        self.assertEqual(backend.segment_bounds({"start": 1.0, "end": 2.0}, (0.0, 1.0)), (1.0, 2.0))
        self.assertEqual(backend.segment_bounds({"start": None, "end": None}, (0.0, 1.0)), (0.0, 1.0))
        self.assertEqual(
            backend.segment_bounds_from_tokens([
                {"start": 0.0, "end": 0.2},
                {"start": 0.2, "end": 0.5},
            ]),
            (0.0, 0.5),
        )
        self.assertIsNone(backend.segment_bounds_from_tokens([{"start": None, "end": None}]))
        self.assertEqual(backend.token_weight(" a b "), 2)

    def test_infer_missing_timings(self):
        tokens = [
            {"text": "a", "start": None, "end": None},
            {"text": "bbb", "start": None, "end": None},
        ]
        backend.infer_missing_timings(tokens, 0.0, 2.0)
        self.assertEqual(tokens[0]["start"], 0.0)
        self.assertLess(tokens[0]["end"], tokens[1]["end"])
        self.assertEqual(tokens[1]["end"], 2.0)

        tokens = [{"text": "a", "start": 0.0, "end": 0.5}]
        backend.infer_missing_timings(tokens, 0.0, 0.5)
        self.assertEqual(tokens[0]["end"], 0.5)

        tokens = [
            {"text": "lead", "start": 0.0, "end": 0.2},
            {"text": "gap", "start": None, "end": None},
            {"text": "tail", "start": 0.8, "end": 1.0},
        ]
        backend.infer_missing_timings(tokens, 0.0, 1.0)
        self.assertEqual((tokens[1]["start"], tokens[1]["end"]), (0.2, 0.8))

        tokens = [
            {"text": "lead", "start": 0.0, "end": 0.6},
            {"text": "gap", "start": None, "end": None},
            {"text": "tail", "start": 0.5, "end": 0.7},
        ]
        backend.infer_missing_timings(tokens, 0.0, 1.0)
        self.assertGreater(tokens[1]["end"], tokens[1]["start"])

    def test_extract_aligned_words_branches(self):
        with self.assertRaisesRegex(ProcessingError, "segment payload must be an object"):
            backend.extract_aligned_words(["bad"])
        with self.assertRaisesRegex(ProcessingError, "words must be a list"):
            backend.extract_aligned_words([{"words": "bad"}])
        with self.assertRaisesRegex(ProcessingError, "word payload must be an object"):
            backend.extract_aligned_words([{"words": ["bad"]}])

        with self.assertRaisesRegex(ProcessingError, "empty after punctuation removal"):
            backend.extract_aligned_words(
                [{"words": [{"word": "", "start": None, "end": None}]}],
                remove_punctuation=True,
            )

        words = backend.extract_aligned_words(
            [
                {
                    "words": [
                        {"word": "(", "start": None, "end": None},
                        {"word": "hello", "start": 0.0, "end": 0.5},
                        {"word": ")", "start": None, "end": None},
                    ]
                },
                {
                    "words": [
                        {"word": "world", "start": None, "end": None},
                    ]
                },
            ]
        )
        self.assertEqual(words[0].text, "( hello )")
        self.assertEqual(words[1].text, "world")
        self.assertGreater(words[1].end_seconds, words[1].start_seconds)

        words = backend.extract_aligned_words(
            [
                {"start": 0.0, "end": 1.0, "words": [{"word": "hi!", "start": 0.0, "end": 0.5}]},
                {"words": [{"word": ",", "start": None, "end": None}]},
            ],
            remove_punctuation=True,
        )
        self.assertEqual(words[0].text, "hi")

        words = backend.extract_aligned_words(
            [
                {"words": [{"word": "hello", "start": 0.0, "end": 0.5}]},
                {"words": [{"word": "!", "start": None, "end": None}]},
            ]
        )
        self.assertEqual(words[0].text, "hello !")

        with self.assertRaisesRegex(ProcessingError, "produced no words"):
            backend.extract_aligned_words([{"words": []}])

    def test_backend_align_success_and_error_mapping(self):
        alignment_module = FakeAlignmentModule()
        audio_module = types.SimpleNamespace(load_audio=lambda path: [0, 1, 2, 3], SAMPLE_RATE=2)
        with (
            patch("dictator.alignment.whisperx_backend.load_whisperx_alignment_modules", return_value=(alignment_module, audio_module)),
            patch("dictator.alignment.whisperx_backend.resolve_device", return_value="cpu"),
            patch("dictator.alignment.whisperx_backend.load_cached_alignment_model", return_value=("model", {"meta": True})),
        ):
            words = backend.WhisperXAlignmentBackend().align(Path("audio.wav"), "hello", "en")
        self.assertEqual(words[0].text, "hello")
        self.assertEqual(alignment_module.align_calls[0][0][0]["end"], 2.0)

        broken_alignment_module = FakeAlignmentModule()
        def raise_missing(*args, **kwargs):
            raise RuntimeError("missing timestamps in alignment")
        broken_alignment_module.align = raise_missing
        with (
            patch("dictator.alignment.whisperx_backend.load_whisperx_alignment_modules", return_value=(broken_alignment_module, audio_module)),
            patch("dictator.alignment.whisperx_backend.resolve_device", return_value="cpu"),
            patch("dictator.alignment.whisperx_backend.load_cached_alignment_model", return_value=("model", {})),
        ):
            with self.assertRaisesRegex(ProcessingError, "missing timestamps"):
                backend.WhisperXAlignmentBackend().align(Path("audio.wav"), "hello", "en")

        def raise_other(*args, **kwargs):
            raise RuntimeError("boom")
        broken_alignment_module.align = raise_other
        with (
            patch("dictator.alignment.whisperx_backend.load_whisperx_alignment_modules", return_value=(broken_alignment_module, audio_module)),
            patch("dictator.alignment.whisperx_backend.resolve_device", return_value="cpu"),
            patch("dictator.alignment.whisperx_backend.load_cached_alignment_model", return_value=("model", {})),
        ):
            with self.assertRaisesRegex(ProcessingError, "alignment failed: boom"):
                backend.WhisperXAlignmentBackend().align(Path("audio.wav"), "hello", "en")


if __name__ == "__main__":
    unittest.main()
