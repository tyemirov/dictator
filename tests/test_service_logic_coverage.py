from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from dictator.alignment.models import AlignedWord
from dictator.diarization.models import (
    DiarizeAudioRequest,
    DiarizedWord,
    SpeakerSegment,
)
from dictator.runtime import DependencyError, ProcessingError, ValidationError
from dictator.subtitles.models import RenderSubtitlesRequest, TimedWord
from dictator.synthesis.models import SynthesisEngine, SynthesisRequest
from dictator.subtitles.service import (
    SubtitleService,
    _coerce_word_bounds as subtitle_coerce_word_bounds,
    grouped_cues,
    render_srt,
    sentence_units,
    words_from_alignment,
    words_from_transcription,
)
from dictator.transcription.models import TranscriptionResult, WordSegment


class FakeTranscriptionService:
    def __init__(self, result: TranscriptionResult | None = None):
        self.result = result or TranscriptionResult(language="en", words=())
        self.calls = []

    def transcribe(self, audio, language=None, model_size="base", model=None, progress_cb=None):
        self.calls.append((audio, language, model_size, model))
        if language is None:
            return self.result
        return TranscriptionResult(language=language, words=self.result.words)


class FakeAlignmentService:
    def __init__(self, words=()):
        self.words = words
        self.calls = []

    def align(self, request):
        self.calls.append(request)
        return types.SimpleNamespace(language=request.language or "en", words=self.words)


class ServiceLogicCoverageTests(unittest.TestCase):
    def test_subtitle_helper_branches(self):
        with self.assertRaisesRegex(ProcessingError, "include text"):
            subtitle_coerce_word_bounds(" ", 0.0, 1.0)
        with self.assertRaisesRegex(ProcessingError, "require timestamps"):
            subtitle_coerce_word_bounds("hello", None, None)
        word = subtitle_coerce_word_bounds("hello", 1.0, 0.5)
        self.assertEqual((word.start_seconds, word.end_seconds), (1.0, 1.0))

        transcription_words = words_from_transcription(
            [WordSegment("", 0.0, 0.1), WordSegment("hello", None, 1.0)]
        )
        self.assertEqual(transcription_words[0].start_seconds, 1.0)
        alignment_words = words_from_alignment([types.SimpleNamespace(text="", start_seconds=0.0, end_seconds=0.1)])
        self.assertEqual(alignment_words, ())

        self.assertEqual(sentence_units(()), ())
        sentences = sentence_units(
            (
                TimedWord("Hello", 0.0, 0.2),
                TimedWord("world.", 0.2, 0.5),
                TimedWord("Tail", 0.6, 0.8),
            )
        )
        self.assertEqual([item.text for item in sentences], ["Hello world.", "Tail"])
        with self.assertRaisesRegex(ValidationError, "group_size"):
            grouped_cues(sentences, 0)
        self.assertEqual(render_srt(()), "")

    def test_subtitle_service_invalid_modes_and_default_dependencies(self):
        with (
            patch("dictator.subtitles.service.TranscriptionService", return_value="tx") as tx_mock,
            patch("dictator.subtitles.service.AlignmentService", return_value="ax") as ax_mock,
        ):
            service = SubtitleService()
        self.assertEqual(service.transcription_service, "tx")
        self.assertEqual(service.alignment_service, "ax")
        tx_mock.assert_called_once()
        ax_mock.assert_called_once()

        service = SubtitleService(
            transcription_service=FakeTranscriptionService(),
            alignment_service=FakeAlignmentService(),
        )
        with self.assertRaisesRegex(ValidationError, "only SRT"):
            service.render(RenderSubtitlesRequest(audio_path=Path("a.wav"), output_format="vtt"))
        with self.assertRaisesRegex(ValidationError, "granularity"):
            service.render(RenderSubtitlesRequest(audio_path=Path("a.wav"), granularity="chars"))

    def test_synthesis_service_and_backend_branches(self):
        backend_module = importlib.import_module("dictator.synthesis.service")

        class FakeTTS:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                self.device = None
                self.calls = []

            def to(self, device):
                self.device = device
                return self

            def tts_to_file(self, **kwargs):
                self.calls.append(kwargs)
                Path(kwargs["file_path"]).write_bytes(b"wav")

        remote_tts_factory = MagicMock(return_value=FakeTTS("model"))
        with (
            patch.dict(
                sys.modules,
                {
                    "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
                    "TTS": types.ModuleType("TTS"),
                    "TTS.api": types.SimpleNamespace(TTS=remote_tts_factory),
                },
            ),
        ):
            backend = backend_module.XTTSBackend(model_id="model")
            fake_tts = backend.load()
            self.assertIs(backend.load(), fake_tts)
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "out.wav"
                backend.synthesise_to_file("hello", Path("speaker.wav"), "en", output_path)
            remote_tts_factory.assert_called_once_with("model")
            self.assertEqual(fake_tts.device, "cpu")
            self.assertEqual(fake_tts.calls[0]["language"], "en")
            self.assertEqual(
                [chunk.text for chunk in backend_module.XTTSSynthesisSession(fake_tts, speaker_wav=Path("speaker.wav"), language_code="en").build_chunks("One.")],
                ["One."],
            )

        local_tts_factory = MagicMock(return_value=FakeTTS())
        with tempfile.TemporaryDirectory() as tmpdir:
            local_model_dir = Path(tmpdir) / "xtts"
            local_model_dir.mkdir()
            (local_model_dir / "config.json").write_text("{}", encoding="utf-8")
            with (
                patch.dict(
                    sys.modules,
                    {
                        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
                        "TTS": types.ModuleType("TTS"),
                        "TTS.api": types.SimpleNamespace(TTS=local_tts_factory),
                    },
                ),
            ):
                backend = backend_module.XTTSBackend(model_id=str(local_model_dir))
                fake_local_tts = backend.load()
            local_tts_factory.assert_called_once_with(
                model_path=str(local_model_dir),
                config_path=str(local_model_dir / "config.json"),
                progress_bar=False,
            )
            self.assertEqual(fake_local_tts.device, "cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            missing_config_model_dir = Path(tmpdir) / "xtts-missing-config"
            missing_config_model_dir.mkdir()
            with patch.dict(
                sys.modules,
                {
                    "TTS": types.ModuleType("TTS"),
                    "TTS.api": types.SimpleNamespace(TTS=local_tts_factory),
                },
            ):
                backend = backend_module.XTTSBackend(model_id=str(missing_config_model_dir))
                with self.assertRaisesRegex(DependencyError, "XTTS config.json was not found"):
                    backend._load_local_model(missing_config_model_dir, device="cpu")

        fake_qwen_model = types.SimpleNamespace(
            prompt_calls=[],
            generate_calls=[],
        )

        def create_voice_clone_prompt(**kwargs):
            fake_qwen_model.prompt_calls.append(kwargs)
            return "voice-clone-prompt"

        def generate_voice_clone(**kwargs):
            fake_qwen_model.generate_calls.append(kwargs)
            return ([np.array([0.1, -0.1], dtype=np.float32)], 24000)

        fake_qwen_model.create_voice_clone_prompt = create_voice_clone_prompt
        fake_qwen_model.generate_voice_clone = generate_voice_clone
        fake_qwen_model._build_assistant_text = lambda text: f"assistant::{text}"
        fake_qwen_model._tokenize_texts = lambda texts: [np.zeros((1, len(text.split())), dtype=np.int64) for text in texts]
        fake_qwen_factory = MagicMock(return_value=fake_qwen_model)
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            bfloat16="bfloat16",
            float16="float16",
            float32="float32",
        )
        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=fake_qwen_factory)),
                "soundfile": types.SimpleNamespace(write=MagicMock()),
            },
        ):
            qwen_backend = backend_module.Qwen3TTSBackend(
                model_id="qwen-model",
                dtype="float32",
                text_token_budget=1,
            )
            self.assertIs(qwen_backend.load(), fake_qwen_model)
            self.assertIs(qwen_backend.load(), fake_qwen_model)
            qwen_request = SynthesisRequest(
                engine=SynthesisEngine.QWEN3,
                speaker_wav=Path("speaker.wav"),
                text="Hello. Again?",
                language_code="ru-RU",
                cap_seconds=None,
                speaker_artifact_id="speaker-1",
                speaker_transcript_text="sample transcript",
            )
            qwen_session = qwen_backend.open_session(qwen_request)
            cached_qwen_session = qwen_backend.open_session(qwen_request)
            self.assertEqual(
                [chunk.text for chunk in qwen_session.build_chunks("Hello. Again?")],
                ["Hello.", "Again?"],
            )
            self.assertEqual(
                [chunk.text for chunk in qwen_session.refine_chunk(backend_module.SynthesisChunk.from_units(("Hello.", "Again?")))],
                ["Hello.", "Again?"],
            )
            self.assertEqual(
                [chunk.text for chunk in qwen_session.refine_chunk(backend_module.SynthesisChunk.from_text("Hello."))],
                ["Hello."],
            )
            self.assertEqual(
                [chunk.text for chunk in backend_module.XTTSSynthesisSession(fake_tts, speaker_wav=Path("speaker.wav"), language_code="en").refine_chunk(backend_module.SynthesisChunk.from_text("One."))],
                ["One."],
            )
            generated_chunk = qwen_session.synthesise_chunk("Hello.")
            self.assertGreater(generated_chunk.duration_seconds, 0.0)
            qwen_session.synthesise_to_file("Hello.", Path("out.wav"))
            self.assertIsNotNone(cached_qwen_session)
            uncached_request = SynthesisRequest(
                engine=SynthesisEngine.QWEN3,
                speaker_wav=Path("speaker.wav"),
                text="Hello.",
                language_code="ru-RU",
                cap_seconds=None,
                speaker_transcript_text="sample transcript",
            )
            qwen_backend.open_session(uncached_request)
        fake_qwen_factory.assert_called_once_with(
            "qwen-model",
            device_map="cpu",
            dtype="float32",
            attn_implementation="flash_attention_2",
        )
        self.assertEqual(len(fake_qwen_model.prompt_calls), 2)
        self.assertEqual(fake_qwen_model.prompt_calls[0]["ref_audio"], "speaker.wav")
        self.assertEqual(fake_qwen_model.prompt_calls[0]["ref_text"], "sample transcript")
        self.assertFalse(fake_qwen_model.prompt_calls[0]["x_vector_only_mode"])
        self.assertEqual(fake_qwen_model.generate_calls[0]["language"], "Russian")
        self.assertEqual(fake_qwen_model.generate_calls[0]["voice_clone_prompt"], "voice-clone-prompt")

        fake_cosyvoice_model = types.SimpleNamespace(
            sample_rate=24000,
            zero_shot_calls=[],
        )

        class FakeTensor:
            def __init__(self, values):
                self._values = np.array(values, dtype=np.float32)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._values

        def inference_zero_shot(text, prompt_text, prompt_wav, stream=False):
            fake_cosyvoice_model.zero_shot_calls.append(
                {
                    "text": text,
                    "prompt_text": prompt_text,
                    "prompt_wav": prompt_wav,
                    "stream": stream,
                }
            )
            return iter(
                [
                    {"tts_speech": None},
                    {"tts_speech": np.array([[0.1, -0.1]], dtype=np.float32)},
                    {"tts_speech": FakeTensor([[0.2, -0.2]])},
                ]
            )

        fake_cosyvoice_model.inference_zero_shot = inference_zero_shot
        fake_cosyvoice_factory = MagicMock(return_value=fake_cosyvoice_model)
        model_ref = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
        with patch.dict(
            sys.modules,
            {
                "soundfile": types.SimpleNamespace(write=MagicMock()),
                "cosyvoice": types.ModuleType("cosyvoice"),
                "cosyvoice.cli": types.ModuleType("cosyvoice.cli"),
                "cosyvoice.cli.cosyvoice": types.SimpleNamespace(AutoModel=fake_cosyvoice_factory),
            },
        ):
            cosy_backend = backend_module.CosyVoice3Backend(model_dir=model_ref)
            self.assertIs(cosy_backend.load(), fake_cosyvoice_model)
            cosy_request = SynthesisRequest(
                engine=SynthesisEngine.COSYVOICE3,
                speaker_wav=Path("speaker.wav"),
                text="Hello. Again?",
                language_code="en",
                cap_seconds=None,
                speaker_transcript_text="sample transcript",
            )
            cosy_session = cosy_backend.open_session(cosy_request)
            self.assertEqual(
                [chunk.text for chunk in cosy_session.build_chunks("Hello. Again?")],
                ["Hello.", "Again?"],
            )
            self.assertEqual(
                [chunk.text for chunk in cosy_session.refine_chunk(backend_module.SynthesisChunk.from_text("Hello."))],
                ["Hello."],
            )
            cosy_chunk = cosy_session.synthesise_chunk("Hello.")
            self.assertGreater(cosy_chunk.duration_seconds, 0.0)
            cosy_session.synthesise_to_file("Hello.", Path("out.wav"))
        fake_cosyvoice_factory.assert_called_once_with(model_dir=model_ref)
        self.assertEqual(
            fake_cosyvoice_model.zero_shot_calls[0]["prompt_text"],
            "You are a helpful assistant.<|endofprompt|>sample transcript",
        )
        self.assertEqual(fake_cosyvoice_model.zero_shot_calls[0]["prompt_wav"], "speaker.wav")
        self.assertFalse(fake_cosyvoice_model.zero_shot_calls[0]["stream"])

        with self.assertRaisesRegex(ValidationError, "speaker_transcript_text"):
            backend_module.Qwen3TTSBackend(model_id="qwen-model").open_session(
                SynthesisRequest(
                    engine=SynthesisEngine.QWEN3,
                    speaker_wav=Path("speaker.wav"),
                    text="Hello",
                    language_code="ru",
                    cap_seconds=None,
                )
            )
        with self.assertRaisesRegex(ValidationError, "speaker_transcript_text"):
            backend_module.CosyVoice3Backend(model_dir="missing-model").open_session(
                SynthesisRequest(
                    engine=SynthesisEngine.COSYVOICE3,
                    speaker_wav=Path("speaker.wav"),
                    text="Hello",
                    language_code="en",
                    cap_seconds=None,
                )
            )
        with self.assertRaisesRegex(ValidationError, "does not support"):
            backend_module._qwen3_language_name("sv")
        self.assertEqual(
            backend_module.Qwen3TTSBackend(model_id="qwen-model", dtype="auto")._resolve_dtype(fake_torch),
            "float32",
        )
        with self.assertRaisesRegex(ValueError, "unsupported qwen3 dtype"):
            backend_module.Qwen3TTSBackend(model_id="qwen-model", dtype="float8")._resolve_dtype(fake_torch)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            backend_module.SynthesisChunk.from_units(("", " "))
        with self.assertRaisesRegex(DependencyError, "cannot be empty"):
            backend_module.CosyVoice3Backend(model_dir=" ").load()

        with self.assertRaisesRegex(DependencyError, "CosyVoice3 is unavailable"):
            backend_module.CosyVoice3Backend(model_dir="FunAudioLLM/Fun-CosyVoice3-0.5B-2512").load()

        with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(write=MagicMock())}):
            with self.assertRaisesRegex(ValueError, "returned no audio"):
                backend_module.Qwen3SynthesisSession(
                    types.SimpleNamespace(generate_voice_clone=lambda **kwargs: ([], 24000)),
                    voice_clone_prompt="prompt",
                    language_name="Russian",
                    text_token_budget=16,
                ).synthesise_to_file("Hello", Path("out.wav"))
            with self.assertRaisesRegex(ValueError, "returned no audio"):
                backend_module.CosyVoice3SynthesisSession(
                    types.SimpleNamespace(sample_rate=24000, inference_zero_shot=lambda *args, **kwargs: iter([])),
                    speaker_wav=Path("speaker.wav"),
                    speaker_transcript_text="sample transcript",
                ).synthesise_to_file("Hello", Path("out.wav"))

        class FakeBackend:
            def synthesise_to_file(self, text, speaker_wav, language_code, output_path):
                output_path.write_bytes(text.encode("utf-8"))

        with self.assertRaisesRegex(ValueError, "No text chunks"):
            backend_module.SpeechSynthesisService(backend=FakeBackend()).synthesise(Path("speaker.wav"), [], None, "en")

        soundfile_module = types.SimpleNamespace(info=MagicMock(side_effect=[types.SimpleNamespace(frames=24000, samplerate=24000)]))
        with patch.dict(sys.modules, {"soundfile": soundfile_module}):
            result = backend_module.SpeechSynthesisService(backend=FakeBackend()).synthesise(
                Path("speaker.wav"), ["one", "two"], 1.0, "en"
            )
        self.assertEqual(len(result.wav_paths), 1)

        soundfile_module = types.SimpleNamespace(info=MagicMock(return_value=types.SimpleNamespace(frames=24000, samplerate=24000)))
        with patch.dict(sys.modules, {"soundfile": soundfile_module}):
            with self.assertRaisesRegex(ValueError, "No chunks fit"):
                backend_module.SpeechSynthesisService(backend=FakeBackend()).synthesise(
                    Path("speaker.wav"), ["one"], 0.5, "en"
                )

        class FakeSession:
            def __init__(self):
                self.calls = []

            def build_chunks(self, text):
                self.calls.append(("build_chunks", text))
                return (
                    backend_module.SynthesisChunk.from_text("One."),
                    backend_module.SynthesisChunk.from_text("Two."),
                )

            def refine_chunk(self, chunk):
                self.calls.append(("refine_chunk", chunk.text))
                return (chunk,)

            def synthesise_to_file(self, text, output_path):
                self.calls.append(("synthesise_to_file", text))
                output_path.write_bytes(text.encode("utf-8"))

        class FakeInMemorySession(FakeSession):
            def synthesise_chunk(self, text):
                self.calls.append(("synthesise_chunk", text))
                return backend_module.SynthesisedAudioChunk(
                    samples=np.array([0.1, -0.1], dtype=np.float32),
                    sample_rate=2,
                    duration_seconds=1.0,
                )

        class RefiningInMemorySession(FakeInMemorySession):
            def build_chunks(self, text):
                self.calls.append(("build_chunks", text))
                return (
                    backend_module.SynthesisChunk.from_units(("One.", "Two.")),
                )

            def refine_chunk(self, chunk):
                self.calls.append(("refine_chunk", chunk.text))
                if len(chunk.units) <= 1:
                    return (chunk,)
                return (
                    backend_module.SynthesisChunk.from_text("One."),
                    backend_module.SynthesisChunk.from_text("Two."),
                )

            def synthesise_chunk(self, text):
                self.calls.append(("synthesise_chunk", text))
                duration = 1.0 if text == "One. Two." else 0.4
                return backend_module.SynthesisedAudioChunk(
                    samples=np.array([0.1, -0.1], dtype=np.float32),
                    sample_rate=2,
                    duration_seconds=duration,
                )

        class FakeSessionBackend:
            engine = SynthesisEngine.QWEN3

            def __init__(self, session=None):
                self.requests = []
                self.session = session or FakeSession()

            def open_session(self, request):
                self.requests.append(request)
                return self.session

        soundfile_module = types.SimpleNamespace(
            info=MagicMock(
                side_effect=[
                    types.SimpleNamespace(frames=24000, samplerate=24000),
                    types.SimpleNamespace(frames=24000, samplerate=24000),
                ]
            )
        )
        fake_session_backend = FakeSessionBackend()
        with patch.dict(sys.modules, {"soundfile": soundfile_module}):
            result = backend_module.SpeechSynthesisService(
                backends={SynthesisEngine.QWEN3: fake_session_backend}
            ).synthesise_text(
                SynthesisRequest(
                    engine=SynthesisEngine.QWEN3,
                    speaker_wav=Path("speaker.wav"),
                    text=" One.\nTwo. ",
                    language_code="ru",
                    cap_seconds=None,
                    speaker_transcript_text="sample transcript",
                )
            )
        self.assertEqual(fake_session_backend.requests[0].text, "One.Two.")
        self.assertEqual(result.segments[1].text, "Two.")
        self.assertEqual(
            [chunk.text for chunk in backend_module.LegacySynthesisSession(FakeBackend(), speaker_wav=Path("speaker.wav"), language_code="en").build_chunks("One.")],
            ["One."],
        )

        in_memory_backend = FakeSessionBackend(session=FakeInMemorySession())
        soundfile_module = types.SimpleNamespace(write=MagicMock())
        with patch.dict(sys.modules, {"soundfile": soundfile_module}):
            in_memory_result = backend_module.SpeechSynthesisService(
                backends={SynthesisEngine.QWEN3: in_memory_backend}
            ).synthesise_text(
                SynthesisRequest(
                    engine=SynthesisEngine.QWEN3,
                    speaker_wav=Path("speaker.wav"),
                    text="One.",
                    language_code="ru",
                    cap_seconds=None,
                    speaker_artifact_id="speaker-1",
                    speaker_transcript_text="sample transcript",
                )
            )
        self.assertEqual(in_memory_result.segments[0].end_seconds, 1.0)
        self.assertIn(("synthesise_chunk", "One."), in_memory_backend.session.calls)
        soundfile_module.write.assert_called()

        with patch.dict(sys.modules, {"soundfile": soundfile_module}):
            with self.assertRaisesRegex(ValueError, "No chunks fit"):
                backend_module.SpeechSynthesisService(
                    backends={SynthesisEngine.QWEN3: FakeSessionBackend(session=FakeInMemorySession())}
                ).synthesise_text(
                    SynthesisRequest(
                        engine=SynthesisEngine.QWEN3,
                        speaker_wav=Path("speaker.wav"),
                        text="One.",
                        language_code="ru",
                        cap_seconds=0.5,
                        speaker_artifact_id="speaker-1",
                        speaker_transcript_text="sample transcript",
                    )
                )

        refining_backend = FakeSessionBackend(session=RefiningInMemorySession())
        with patch.dict(sys.modules, {"soundfile": soundfile_module}):
            refined_result = backend_module.SpeechSynthesisService(
                backends={SynthesisEngine.QWEN3: refining_backend}
            ).synthesise_text(
                SynthesisRequest(
                    engine=SynthesisEngine.QWEN3,
                    speaker_wav=Path("speaker.wav"),
                    text="One. Two.",
                    language_code="ru",
                    cap_seconds=0.8,
                    speaker_artifact_id="speaker-1",
                    speaker_transcript_text="sample transcript",
                )
            )
        self.assertEqual([segment.text for segment in refined_result.segments], ["One.", "Two."])
        self.assertIn(("refine_chunk", "One. Two."), refining_backend.session.calls)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "tts"
            temp_dir.mkdir()
            backend_module.cleanup_synthesis_result(
                types.SimpleNamespace(temp_dir=temp_dir)
            )
            self.assertFalse(temp_dir.exists())

        with self.assertRaisesRegex(ValueError, "set backend or backends"):
            backend_module.SpeechSynthesisService(backend=FakeBackend(), backends={SynthesisEngine.XTTS: FakeBackend()})
        with self.assertRaisesRegex(ValidationError, "unsupported synthesis engine"):
            backend_module.SpeechSynthesisService(backends={})._resolve_backend(SynthesisEngine.XTTS)
        with self.assertRaisesRegex(ValueError, "does not support synthesis sessions"):
            backend_module.SpeechSynthesisService(backends={SynthesisEngine.XTTS: object()})._open_session(
                object(),
                SynthesisRequest(
                    engine=SynthesisEngine.XTTS,
                    speaker_wav=Path("speaker.wav"),
                    text="Hello",
                    language_code="en",
                    cap_seconds=None,
                ),
            )

        fake_result = types.SimpleNamespace(
            wav_paths=(Path("a.wav"),),
            segments=(types.SimpleNamespace(to_legacy_dict=lambda: {"content": "one", "start": 0.0, "end": 1.0}),),
        )
        with patch.object(backend_module.SpeechSynthesisService, "synthesise", return_value=fake_result):
            paths, timeline = backend_module.synthesise(Path("speaker.wav"), ["one"], None, "en")
        self.assertEqual(paths, [Path("a.wav")])
        self.assertEqual(timeline[0]["content"], "one")

    def test_diarization_service_branches(self):
        diarization_module = importlib.import_module("dictator.diarization.service")
        with self.assertRaisesRegex(ProcessingError, "timestamps are required"):
            diarization_module._coerce_word_bounds({"content": "hello"})
        self.assertEqual(
            diarization_module._coerce_word_bounds({"start": 2.0, "end": 1.0}),
            (2.0, 2.0),
        )
        with self.assertRaisesRegex(ProcessingError, "no speakers detected"):
            diarization_module._best_speaker_segment(0.0, 1.0, ())
        with self.assertRaisesRegex(ValidationError, "non-negative"):
            diarization_module.build_utterances((), utterance_gap_seconds=-0.1)
        self.assertEqual(diarization_module.build_utterances(()), ())
        with self.assertRaisesRegex(ProcessingError, "no speakers detected"):
            diarization_module.dominant_speaker_label(())

        original_import = __import__
        def fake_import(name, *args, **kwargs):
            if name in {"numpy", "torch"}:
                raise ImportError("missing")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(DependencyError, "numpy and torch"):
                diarization_module.run_diarization(object(), Path("audio.wav"))

        class FakeTensor:
            def __init__(self, array):
                self.array = array
            def unsqueeze(self, _):
                return self

        class FakeTurn:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        class FakeResult:
            def __init__(self, tracks):
                self._tracks = tracks
            def itertracks(self, yield_label=True):
                return iter(self._tracks)

        fake_ffmpeg_module = types.ModuleType("dictator.audio.ffmpeg_ops")
        fake_ffmpeg_module.decode_pcm = lambda path: np.array([0, 1], dtype=np.int16)
        fake_torch = types.SimpleNamespace(from_numpy=lambda array: FakeTensor(array))
        with patch.dict(sys.modules, {"dictator.audio.ffmpeg_ops": fake_ffmpeg_module, "torch": fake_torch}):
            with self.assertRaisesRegex(ProcessingError, "no speakers detected"):
                diarization_module.run_diarization(lambda payload: FakeResult([]), Path("audio.wav"))
            segments = diarization_module.run_diarization(
                lambda payload: FakeResult([(FakeTurn(0.0, 1.0), None, "speaker_a")]),
                Path("audio.wav"),
            )
        self.assertEqual(segments[0].speaker, "S1")

        fake_transcription_module = types.ModuleType("dictator.transcription.service")
        fake_transcription_module.TranscriptionService = lambda: "tx"
        with patch.dict(sys.modules, {"dictator.transcription.service": fake_transcription_module}):
            service = diarization_module.DiarizationService()
        self.assertEqual(service._transcription_service, "tx")
        with self.assertRaisesRegex(DependencyError, "pipeline loader"):
            diarization_module.DiarizationService(transcription_service=FakeTranscriptionService())._load_pipeline()
        with self.assertRaisesRegex(ValidationError, "non-negative"):
            diarization_module.DiarizationService(transcription_service=FakeTranscriptionService()).diarize(
                DiarizeAudioRequest(input_path=Path("audio.wav"), utterance_gap_seconds=-1.0),
                diarization_pipeline=object(),
            )

        fake_transcription = FakeTranscriptionService(
            TranscriptionResult(language="en", words=(WordSegment("hello", 0.0, 0.4),))
        )
        with patch("dictator.diarization.service.run_diarization", return_value=(SpeakerSegment("S1", 0.0, 1.0),)):
            result = diarization_module.DiarizationService(transcription_service=fake_transcription).diarize(
                DiarizeAudioRequest(input_path=Path("audio.wav")),
                diarization_pipeline=object(),
            )
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.speakers[0].speaker, "S1")

    def test_extraction_service_branches(self):
        torch_stub = types.SimpleNamespace(
            backends=types.SimpleNamespace(
                cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
                cudnn=types.SimpleNamespace(allow_tf32=False),
            ),
            cuda=types.SimpleNamespace(is_available=lambda: False),
            device=lambda value: f"device:{value}",
        )
        librosa_stub = types.SimpleNamespace(feature=types.SimpleNamespace(rms=lambda y: np.array([[0.0, 1.0]])))
        with patch.dict(sys.modules, {"torch": torch_stub, "librosa": librosa_stub, "ffmpeg": types.ModuleType("ffmpeg")}):
            extraction_module = importlib.import_module("dictator.extraction.service")
            extraction_module = importlib.reload(extraction_module)

        extraction_module.configure_torch_runtime()
        self.assertTrue(torch_stub.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch_stub.backends.cudnn.allow_tf32)
        self.assertEqual(extraction_module.spectral_centroid(np.array([0, 0], dtype=np.int16)), 0.0)
        self.assertGreaterEqual(extraction_module.snr(np.array([1, 2, 3], dtype=np.int16)), 0.0)
        self.assertGreaterEqual(extraction_module.pitch_variation(np.array([1, 2, 3], dtype=np.int16)), 0.0)

        fake_pipeline = types.SimpleNamespace(to=MagicMock())
        from_pretrained = MagicMock(return_value=fake_pipeline)
        fake_pyannote = types.SimpleNamespace(Pipeline=types.SimpleNamespace(from_pretrained=from_pretrained))
        with (
            patch.dict(sys.modules, {"pyannote": types.ModuleType("pyannote"), "pyannote.audio": fake_pyannote}),
            patch.object(extraction_module, "require_diarization_token", return_value="hf-token"),
        ):
            loaded = extraction_module.load_diarization_pipeline()
        self.assertIs(loaded, fake_pipeline)
        from_pretrained.assert_called_once_with(
            extraction_module.DIARIZATION_MODEL,
            use_auth_token="hf-token",
        )
        fake_pipeline.to.assert_called_once_with("device:cpu")

        with patch.object(extraction_module.os, "getenv", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                extraction_module.require_diarization_token()

        with patch.object(extraction_module, "run_diarization", return_value=(SpeakerSegment("S1", 0.0, 1.0),)):
            with patch.object(extraction_module, "assign_words_to_speakers", return_value=(DiarizedWord("hello", 0.0, 0.4, "S2"),)):
                with self.assertRaisesRegex(RuntimeError, "dominant speaker"):
                    extraction_module.apply_diarization_filter([{"content": "hello", "start": 0.0, "end": 0.4}], object(), Path("audio.wav"))

        with self.assertRaisesRegex(RuntimeError, "exceeds track length"):
            extraction_module.choose_window(np.zeros(10, dtype=np.int16), [], 1.0)
        with patch.object(extraction_module, "spectral_centroid", return_value=100.0):
            with self.assertRaisesRegex(RuntimeError, "no suitable window"):
                extraction_module.choose_window(
                    np.zeros(extraction_module.SAMPLE_RATE * 2, dtype=np.int16),
                    [{"start": 0.1, "end": 0.2}],
                    1.0,
                    min_centroid=200.0,
                )
        with (
            patch.object(extraction_module, "spectral_centroid", return_value=1000.0),
            patch.object(extraction_module, "pitch_variation", return_value=0.0),
            patch.object(extraction_module, "snr", return_value=1.0),
        ):
            window = extraction_module.choose_window(
                np.ones(extraction_module.SAMPLE_RATE * 3, dtype=np.int16),
                [{"start": 0.1, "end": 0.2}, {"start": 1.1, "end": 1.2}],
                1.0,
                )
        self.assertEqual(window, 0.0)

        with patch.object(extraction_module, "spectral_centroid", return_value=1000.0):
            with self.assertRaisesRegex(RuntimeError, "no suitable window"):
                extraction_module.choose_window(
                    np.ones(extraction_module.SAMPLE_RATE * 2, dtype=np.int16),
                    [
                        {"start": 0.0, "end": 0.1},
                        {"start": 0.1, "end": 0.2},
                        {"start": 0.2, "end": 0.3},
                        {"start": 0.3, "end": 0.4},
                        {"start": 0.4, "end": 0.5},
                    ],
                    1.0,
                )

        with self.assertRaisesRegex(RuntimeError, "no words found"):
            extraction_module.compute_trim_bounds(10.0, [])
        self.assertEqual(
            extraction_module.compute_trim_bounds(1.0, [{"start": 0.1, "end": 0.95}]),
            (0.0, 1.0),
        )

        request = types.SimpleNamespace(
            input_path=Path("audio.wav"),
            output_path=None,
            model_size="base",
            duration_seconds=1.0,
            language="en",
            max_speech_rate=4.0,
            min_centroid_hz=500.0,
            max_centroid_hz=4000.0,
        )
        with (
            patch.object(extraction_module, "configure_torch_runtime"),
            patch.object(extraction_module, "decode_pcm", return_value=np.ones(extraction_module.SAMPLE_RATE * 2, dtype=np.int16)),
            patch.object(extraction_module, "load_diarization_pipeline", return_value="pipeline"),
            patch.object(extraction_module, "load_whisper_model", return_value="model"),
            patch.object(extraction_module, "transcribe_words", return_value=[]),
        ):
            with self.assertRaisesRegex(RuntimeError, "no words transcribed"):
                extraction_module.ReferenceExtractionService().extract(request)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.wav"
            request = types.SimpleNamespace(
                input_path=Path("audio.wav"),
                output_path=output_path,
                model_size="base",
                duration_seconds=1.0,
                language="en",
                max_speech_rate=4.0,
                min_centroid_hz=500.0,
                max_centroid_hz=4000.0,
            )
            with (
                patch.object(extraction_module, "configure_torch_runtime"),
                patch.object(extraction_module, "decode_pcm", return_value=np.ones(extraction_module.SAMPLE_RATE * 2, dtype=np.int16)),
                patch.object(extraction_module, "load_diarization_pipeline", return_value="pipeline"),
                patch.object(extraction_module, "load_whisper_model", return_value="model"),
                patch.object(extraction_module, "transcribe_words", return_value=[{"content": "hello", "start": 0.1, "end": 0.4}]),
                patch.object(extraction_module, "apply_diarization_filter", return_value=[{"content": "hello", "start": 0.1, "end": 0.4}]),
                patch.object(extraction_module, "choose_window", return_value=0.0),
                patch.object(extraction_module, "compute_trim_bounds", return_value=(0.0, 0.5)),
                patch.object(extraction_module, "trim_and_normalise") as trim_mock,
            ):
                result = extraction_module.ReferenceExtractionService().extract(request)
        trim_mock.assert_called_once()
        self.assertEqual(result.output_path, output_path)
        self.assertEqual(result.window_end_seconds, 1.0)


if __name__ == "__main__":
    unittest.main()
