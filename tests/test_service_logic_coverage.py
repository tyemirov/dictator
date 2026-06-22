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
from dictator.synthesis.models import SynthesisAudioFormat, SynthesisEngine, SynthesisRequest, SynthesisTextFormat
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
        fake_qwen_model._tokenize_texts = lambda texts: [
            np.zeros((1, max(1, len(text.split()))), dtype=np.int64) for text in texts
        ]
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
            generated_chunk = qwen_session.synthesise_chunk("Hello.")
            self.assertGreater(generated_chunk.duration_seconds, 0.0)
            self.assertIsNotNone(cached_qwen_session)
            with self.assertRaisesRegex(ValidationError, "SSML"):
                qwen_backend.open_session(
                    SynthesisRequest(
                        engine=SynthesisEngine.QWEN3,
                        speaker_wav=Path("speaker.wav"),
                        text="<speak>Hello.</speak>",
                        language_code="en",
                        cap_seconds=None,
                        speaker_transcript_text="sample transcript",
                        text_format=SynthesisTextFormat.SSML,
                    )
                )
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
        )
        fake_qwen_factory.reset_mock()
        fake_gpu_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True),
            bfloat16="bfloat16",
            float16="float16",
            float32="float32",
        )
        with patch.dict(
            sys.modules,
            {
                "torch": fake_gpu_torch,
                "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=fake_qwen_factory)),
            },
            clear=False,
        ):
            with self.assertRaisesRegex(DependencyError, "flash-attn is required"):
                backend_module.Qwen3TTSBackend(model_id="qwen-model", dtype="auto").load()
        fake_qwen_factory.reset_mock()
        with patch.dict(
            sys.modules,
            {
                "torch": fake_gpu_torch,
                "flash_attn": types.ModuleType("flash_attn"),
                "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=fake_qwen_factory)),
            },
        ):
            backend_module.Qwen3TTSBackend(model_id="qwen-model", dtype="auto").load()
        fake_qwen_factory.assert_called_once_with(
            "qwen-model",
            device_map="cuda:0",
            dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )
        self.assertEqual(len(fake_qwen_model.prompt_calls), 2)
        self.assertEqual(fake_qwen_model.prompt_calls[0]["ref_audio"], "speaker.wav")
        self.assertEqual(fake_qwen_model.prompt_calls[0]["ref_text"], "sample transcript")
        self.assertFalse(fake_qwen_model.prompt_calls[0]["x_vector_only_mode"])
        self.assertEqual(fake_qwen_model.generate_calls[0]["language"], "Russian")
        self.assertEqual(fake_qwen_model.generate_calls[0]["voice_clone_prompt"], "voice-clone-prompt")

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

        with self.assertRaisesRegex(ValueError, "returned no audio"):
            backend_module.Qwen3SynthesisSession(
                types.SimpleNamespace(generate_voice_clone=lambda **kwargs: ([], 24000)),
                voice_clone_prompt="prompt",
                language_name="Russian",
                text_token_budget=16,
            ).synthesise_chunk("Hello")

        class FakeTensor:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.array([0.1, -0.1], dtype=np.float32)

        silero_load_order = []
        fake_silero_package = types.SimpleNamespace(q_model_unpacked=False)
        fake_silero_package.unpack_q_model = MagicMock(
            side_effect=lambda: (
                silero_load_order.append("unpack"),
                setattr(fake_silero_package, "q_model_unpacked", True),
            )
        )
        fake_silero_model = types.SimpleNamespace(
            packages=[fake_silero_package],
            to=MagicMock(side_effect=lambda device: silero_load_order.append(f"to:{device}")),
            apply_tts=MagicMock(return_value=FakeTensor()),
        )

        class FakePackageImporter:
            def __init__(self, model_path):
                self.model_path = model_path

            def load_pickle(self, package_name, object_name):
                self.package_name = package_name
                self.object_name = object_name
                return fake_silero_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "v5_5_ru.pt"
            model_path.write_bytes(b"model")
            model_digest = "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4"
            downloaded_path = Path(tmpdir) / "downloaded" / "v5_5_ru.pt"
            fake_silero_torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: False),
                device=lambda name: name,
                package=types.SimpleNamespace(PackageImporter=FakePackageImporter),
                hub=types.SimpleNamespace(
                    download_url_to_file=MagicMock(side_effect=lambda url, dst: Path(dst).write_bytes(b"model"))
                ),
            )
            with patch.dict(sys.modules, {"torch": fake_silero_torch}):
                silero_backend = backend_module.SileroRuTTSBackend(
                    model_path=str(model_path),
                    model_sha256=model_digest,
                    default_speaker="baya",
                    sample_rate=48000,
                    text_char_budget=10,
                )
                self.assertIs(silero_backend.load(), fake_silero_model)
                self.assertIs(silero_backend.load(), fake_silero_model)
                self.assertEqual(silero_load_order, ["unpack", "to:cpu"])
                already_unpacked_package = types.SimpleNamespace(
                    q_model_unpacked=True,
                    unpack_q_model=MagicMock(),
                )
                silero_backend._unpack_quantized_accentor_before_device_move(
                    types.SimpleNamespace(packages=[already_unpacked_package])
                )
                already_unpacked_package.unpack_q_model.assert_not_called()

                class ReadOnlyUnpackedPackage:
                    def __init__(self):
                        self.unpack_q_model = MagicMock()

                    @property
                    def q_model_unpacked(self):
                        return False

                    @q_model_unpacked.setter
                    def q_model_unpacked(self, _value):
                        raise RuntimeError("read-only")

                read_only_package = ReadOnlyUnpackedPackage()
                silero_backend._unpack_quantized_accentor_before_device_move(
                    types.SimpleNamespace(packages=[read_only_package])
                )
                read_only_package.unpack_q_model.assert_called_once_with()
                silero_session = silero_backend.open_session(
                    SynthesisRequest(
                        engine=SynthesisEngine.SILERO_RU,
                        speaker_wav=None,
                        text="Привет. Еще?",
                        language_code="ru-RU",
                        cap_seconds=None,
                        preset_speaker="xenia",
                        audio_format=SynthesisAudioFormat("wav", "pcm_s16le", 24000, 1, 16),
                    )
                )
                self.assertEqual([chunk.text for chunk in silero_session.build_chunks("Привет. Еще?")], ["Привет.", "Еще?"])
                self.assertEqual(
                    [chunk.text for chunk in silero_session.refine_chunk(backend_module.SynthesisChunk.from_units(("Раз.", "Два.")))],
                    ["Раз.", "Два."],
                )
                self.assertEqual(
                    [chunk.text for chunk in silero_session.refine_chunk(backend_module.SynthesisChunk.from_text("Раз."))],
                    ["Раз."],
                )
                silero_chunk = silero_session.synthesise_chunk("Привет.")
                self.assertEqual(silero_chunk.sample_rate, 24000)
                self.assertGreater(silero_chunk.duration_seconds, 0.0)
                self.assertEqual(fake_silero_model.apply_tts.call_args.kwargs["speaker"], "xenia")
                self.assertTrue(fake_silero_model.apply_tts.call_args.kwargs["put_accent"])
                self.assertTrue(fake_silero_model.apply_tts.call_args.kwargs["put_yo"])

                self.assertEqual(silero_backend._ensure_model_path(fake_silero_torch), model_path)
                configured_download = backend_module.SileroRuTTSBackend(
                    model_path=str(downloaded_path),
                    model_sha256=model_digest,
                )
                self.assertEqual(configured_download._ensure_model_path(fake_silero_torch), downloaded_path)
                with self.assertRaisesRegex(DependencyError, "model digest mismatch"):
                    backend_module.SileroRuTTSBackend(
                        model_path=str(model_path),
                        model_sha256="0" * 64,
                    )._ensure_model_path(fake_silero_torch)
                bad_download_path = Path(tmpdir) / "bad-download" / "v5_5_ru.pt"
                with self.assertRaisesRegex(DependencyError, "model digest mismatch"):
                    backend_module.SileroRuTTSBackend(
                        model_path=str(bad_download_path),
                        model_sha256="0" * 64,
                    )._ensure_model_path(fake_silero_torch)
                self.assertFalse(bad_download_path.exists())
                unchecked_path = Path(tmpdir) / "unchecked" / "v5_5_ru.pt"
                unchecked_path.parent.mkdir(parents=True, exist_ok=True)
                unchecked_path.write_bytes(b"unchecked")
                unchecked_backend = backend_module.SileroRuTTSBackend(
                    model_path=str(unchecked_path),
                    model_sha256="",
                )
                self.assertEqual(unchecked_backend._ensure_model_path(fake_silero_torch), unchecked_path)
                with patch.object(Path, "home", return_value=Path(tmpdir)):
                    cached_download = backend_module.SileroRuTTSBackend(
                        model_path="",
                        model_sha256=model_digest,
                    )
                    self.assertEqual(cached_download._ensure_model_path(fake_silero_torch).name, "v5_5_ru.pt")
                    self.assertEqual(cached_download._ensure_model_path(fake_silero_torch).name, "v5_5_ru.pt")
                    self.assertEqual(cached_download._model_cache_path().name, "v5_5_ru.pt")
                repair_home = Path(tmpdir) / "repair-home"
                with patch.object(Path, "home", return_value=repair_home):
                    repaired_cache = backend_module.SileroRuTTSBackend(
                        model_path="",
                        model_sha256=model_digest,
                    )
                    repaired_cache_path = repaired_cache._model_cache_path()
                    repaired_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    repaired_cache_path.write_bytes(b"stale")
                    self.assertEqual(repaired_cache._ensure_model_path(fake_silero_torch), repaired_cache_path)
                    self.assertEqual(repaired_cache_path.read_bytes(), b"model")
                bad_cache_home = Path(tmpdir) / "bad-cache-home"
                with patch.object(Path, "home", return_value=bad_cache_home):
                    bad_cache = backend_module.SileroRuTTSBackend(
                        model_path="",
                        model_sha256="0" * 64,
                    )
                    bad_cache_path = bad_cache._model_cache_path()
                    bad_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    bad_cache_path.write_bytes(b"stale")
                    with self.assertRaisesRegex(DependencyError, "model digest mismatch"):
                        bad_cache._ensure_model_path(fake_silero_torch)
                    self.assertFalse(bad_cache_path.exists())

            with self.assertRaisesRegex(ValidationError, "language_code"):
                silero_backend.open_session(
                    SynthesisRequest(
                        engine=SynthesisEngine.SILERO_RU,
                        speaker_wav=None,
                        text="Hello",
                        language_code="en",
                        cap_seconds=None,
                    )
                )
            with self.assertRaisesRegex(ValidationError, "speaker"):
                silero_backend._resolve_speaker("unknown")
            self.assertEqual(
                silero_backend._resolve_sample_rate(
                    SynthesisRequest(
                        engine=SynthesisEngine.SILERO_RU,
                        speaker_wav=None,
                        text="Привет",
                        language_code="ru",
                        cap_seconds=None,
                        audio_format=SynthesisAudioFormat("wav", "pcm_s16le", 16000, 1, 16),
                    )
                ),
                48000,
            )
            self.assertEqual(
                silero_backend._resolve_sample_rate(
                    SynthesisRequest(
                        engine=SynthesisEngine.SILERO_RU,
                        speaker_wav=None,
                        text="Привет",
                        language_code="ru",
                        cap_seconds=None,
                    )
                ),
                48000,
            )

        class FallbackSileroModel:
            def __init__(self):
                self.calls = []

            def apply_tts(self, **kwargs):
                self.calls.append(kwargs)
                if "put_accent" in kwargs:
                    raise TypeError("old silero signature")
                return np.array([0.1, -0.1, 0.2], dtype=np.float32)

        fallback_model = FallbackSileroModel()
        fallback_chunk = backend_module.SileroRuSynthesisSession(
            fallback_model,
            speaker="baya",
            sample_rate=3,
            text_char_budget=16,
        ).synthesise_chunk("Привет.")
        self.assertEqual(fallback_chunk.duration_seconds, 1.0)
        self.assertNotIn("put_accent", fallback_model.calls[-1])

        ssml_model = types.SimpleNamespace(apply_tts=MagicMock(return_value=FakeTensor()))
        ssml_session = backend_module.SileroRuSynthesisSession(
            ssml_model,
            speaker="xenia",
            sample_rate=2,
            text_char_budget=64,
            text_format=SynthesisTextFormat.SSML,
        )
        ssml_text = '<speak><prosody rate="slow">Стоит в поле терем+ок.</prosody><break time="500ms"/></speak>'
        ssml_chunks = ssml_session.build_chunks(ssml_text)
        self.assertEqual(len(ssml_chunks), 1)
        self.assertEqual(ssml_chunks[0].timeline_text, "Стоит в поле теремок.")
        ssml_chunk = ssml_session.synthesise_chunk(ssml_text)
        self.assertEqual(ssml_chunk.duration_seconds, 1.0)
        self.assertEqual(ssml_model.apply_tts.call_args.kwargs["ssml_text"], ssml_text)
        self.assertNotIn("text", ssml_model.apply_tts.call_args.kwargs)
        self.assertNotIn("put_accent", ssml_model.apply_tts.call_args.kwargs)
        auto_ssml_session = backend_module.SileroRuSynthesisSession(
            ssml_model,
            speaker="xenia",
            sample_rate=2,
            text_char_budget=16,
        )
        self.assertEqual(auto_ssml_session.build_chunks("<speak>Авто.</speak>")[0].timeline_text, "Авто.")
        plain_ssml_session = backend_module.SileroRuSynthesisSession(
            ssml_model,
            speaker="xenia",
            sample_rate=2,
            text_char_budget=16,
            text_format=SynthesisTextFormat.PLAIN_TEXT,
        )
        self.assertEqual(plain_ssml_session.build_chunks("<speak>Не SSML.</speak>")[0].text, "<speak>Не SSML.</speak>")
        with self.assertRaisesRegex(ValidationError, "SSML"):
            ssml_session.build_chunks("<speak><bad>")
        with self.assertRaisesRegex(ValidationError, "root"):
            ssml_session.build_chunks("<p>Привет.</p>")
        with self.assertRaisesRegex(ValidationError, "unsupported"):
            ssml_session.build_chunks("<speak><emphasis>Привет.</emphasis></speak>")
        with self.assertRaisesRegex(ValidationError, "attributes"):
            ssml_session.build_chunks('<speak><prosody volume="loud">Привет.</prosody></speak>')
        with self.assertRaisesRegex(ValidationError, "speakable"):
            ssml_session.build_chunks('<speak><break time="500ms"/></speak>')
        small_ssml_session = backend_module.SileroRuSynthesisSession(
            ssml_model,
            speaker="xenia",
            sample_rate=2,
            text_char_budget=16,
            text_format=SynthesisTextFormat.SSML,
        )
        with self.assertRaisesRegex(ValidationError, "char budget"):
            small_ssml_session.build_chunks("<speak>Очень длинная фраза.</speak>")
        typeerror_ssml_model = types.SimpleNamespace(apply_tts=MagicMock(side_effect=TypeError("no ssml_text")))
        with self.assertRaisesRegex(DependencyError, "ssml_text"):
            backend_module.SileroRuSynthesisSession(
                typeerror_ssml_model,
                speaker="xenia",
                sample_rate=2,
                text_char_budget=16,
                text_format=SynthesisTextFormat.SSML,
            ).synthesise_chunk("<speak>Привет.</speak>")

        fake_config = types.SimpleNamespace(
            qwen3_model_id="default-model",
            qwen3_dtype="float16",
            qwen3_text_token_budget=128,
            silero_ru_model_path="/models/silero/v5_5_ru.pt",
            silero_ru_model_url="https://example.invalid/v5_5_ru.pt",
            silero_ru_model_sha256="abc123",
            silero_ru_default_speaker="xenia",
            silero_ru_sample_rate=48000,
            silero_ru_text_char_budget=777,
        )
        with (
            patch.object(backend_module.SynthesisConfig, "from_env", return_value=fake_config),
            patch.object(backend_module, "Qwen3TTSBackend", return_value="default-backend") as backend_ctor,
            patch.object(backend_module, "SileroRuTTSBackend", return_value="silero-backend") as silero_ctor,
        ):
            default_service = backend_module.SpeechSynthesisService()
        self.assertEqual(default_service.backends[SynthesisEngine.QWEN3], "default-backend")
        self.assertEqual(default_service.backends[SynthesisEngine.SILERO_RU], "silero-backend")
        backend_ctor.assert_called_once_with(
            model_id="default-model",
            dtype="float16",
            text_token_budget=128,
        )
        silero_ctor.assert_called_once_with(
            model_path="/models/silero/v5_5_ru.pt",
            model_url="https://example.invalid/v5_5_ru.pt",
            model_sha256="abc123",
            default_speaker="xenia",
            sample_rate=48000,
            text_char_budget=777,
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

            def synthesise_chunk(self, text):
                self.calls.append(("synthesise_chunk", text))
                return backend_module.SynthesisedAudioChunk(
                    samples=np.array([0.1, -0.1], dtype=np.float32),
                    sample_rate=2,
                    duration_seconds=1.0,
                )

        class RefiningInMemorySession(FakeSession):
            def build_chunks(self, text):
                self.calls.append(("build_chunks", text))
                return (backend_module.SynthesisChunk.from_units(("One.", "Two.")),)

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
                duration = 1.0 if text in {"One. Two.", "One.\n\nTwo."} else 0.4
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

        fake_session_backend = FakeSessionBackend()
        soundfile_module = types.SimpleNamespace(write=MagicMock())
        progress_updates = []
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
                ),
                progress_callback=lambda completed, total: progress_updates.append((completed, total)),
            )
        self.assertEqual(fake_session_backend.requests[0].text, "One. Two.")
        self.assertEqual(result.segments[1].text, "Two.")
        self.assertEqual(progress_updates, [(0, 2), (1, 2), (2, 2)])
        soundfile_module.write.assert_called()

        with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(write=MagicMock())}):
            with self.assertRaisesRegex(ValueError, "No chunks fit"):
                backend_module.SpeechSynthesisService(
                    backends={SynthesisEngine.QWEN3: FakeSessionBackend(session=FakeSession())}
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
        refined_progress_updates = []
        with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(write=MagicMock())}):
            refined_result = backend_module.SpeechSynthesisService(
                backends={SynthesisEngine.QWEN3: refining_backend}
            ).synthesise_text(
                SynthesisRequest(
                    engine=SynthesisEngine.QWEN3,
                    speaker_wav=Path("speaker.wav"),
                    text="One. Two.",
                    language_code="ru",
                    cap_seconds=0.99,
                    speaker_artifact_id="speaker-1",
                    speaker_transcript_text="sample transcript",
                )
                ,
                progress_callback=lambda completed, total: refined_progress_updates.append((completed, total)),
            )
        self.assertEqual([segment.text for segment in refined_result.segments], ["One.", "Two."])
        self.assertIn(("refine_chunk", "One. Two."), refining_backend.session.calls)
        self.assertEqual(refined_progress_updates, [(0, 1), (0, 2), (1, 2), (2, 2)])

        class SmallChunkSession(FakeSession):
            def synthesise_chunk(self, text):
                self.calls.append(("synthesise_chunk", text))
                return backend_module.SynthesisedAudioChunk(
                    samples=np.array([0.1, -0.1], dtype=np.float32),
                    sample_rate=2,
                    duration_seconds=0.4,
                )

        small_chunk_backend = FakeSessionBackend(session=SmallChunkSession())
        with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(write=MagicMock())}):
            capped_result = backend_module.SpeechSynthesisService(
                backends={SynthesisEngine.QWEN3: small_chunk_backend}
            ).synthesise_text(
                SynthesisRequest(
                    engine=SynthesisEngine.QWEN3,
                    speaker_wav=Path("speaker.wav"),
                    text="One. Two.",
                    language_code="ru",
                    cap_seconds=0.4,
                    speaker_transcript_text="sample transcript",
                )
            )
        self.assertEqual([segment.text for segment in capped_result.segments], ["One."])

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir) / "tts"
            temp_dir.mkdir()
            backend_module.cleanup_synthesis_result(types.SimpleNamespace(temp_dir=temp_dir))
            self.assertFalse(temp_dir.exists())

        with self.assertRaisesRegex(ValidationError, "unsupported synthesis engine"):
            backend_module.SpeechSynthesisService(backends={})._resolve_backend(SynthesisEngine.QWEN3)

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
