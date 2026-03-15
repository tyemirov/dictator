from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())
sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))

from dictator.client import DiarizationClient, DictationClient, SubtitleClient, SubtitleResult
from dictator.diarization.models import DiarizeAudioResult, DiarizedUtterance, DiarizedWord, SpeakerSegment, SpeakerSummary
from dictator.runtime import ProcessingError, ValidationError
from dictator.speech.v1 import subtitle_pb2, transcription_pb2
from dictator.subtitles.models import RenderSubtitlesRequest, TimedWord
from dictator.subtitles.service import (
    SubtitleService,
    _coerce_word_bounds,
    grouped_cues,
    render_srt,
    sentence_units,
    words_from_alignment,
    words_from_transcription,
)
from dictator.synthesis.models import SpeechSegment, SynthesisEngine, SynthesisRequest, SynthesisResult
from dictator.synthesis.service import Qwen3TTSBackend, cleanup_synthesis_result
from dictator.transcription.models import TranscriptionResult, WordSegment


class _ArtifactStub:
    pass


class _TranscriptionStub:
    def __init__(self, transcribe_response=None, diarize_response=None):
        self.transcribe_response = transcribe_response
        self.diarize_response = diarize_response
        self.calls = []

    def SubmitTranscribeJob(self, request, metadata=()):
        self.calls.append((request, metadata))
        return types.SimpleNamespace(
            job_id="tx-1",
            state=transcription_pb2.TRANSCRIPTION_JOB_STATE_QUEUED,
        )

    def GetTranscribeJob(self, request, metadata=()):
        return self.transcribe_response

    def SubmitDiarizeAudioJob(self, request, metadata=()):
        self.calls.append((request, metadata))
        return types.SimpleNamespace(
            job_id="dia-1",
            state=transcription_pb2.DIARIZATION_JOB_STATE_QUEUED,
        )

    def GetDiarizeAudioJob(self, request, metadata=()):
        return self.diarize_response


class _SubtitleStub:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def SubmitRenderSubtitlesJob(self, request, metadata=()):
        self.calls.append((request, metadata))
        return types.SimpleNamespace(
            job_id="sub-1",
            state=subtitle_pb2.SUBTITLE_JOB_STATE_QUEUED,
        )

    def GetRenderSubtitlesJob(self, request, metadata=()):
        return self.response


class _FakeTranscriptionService:
    def __init__(self, result):
        self.result = result

    def transcribe(self, audio, language=None, model_size="base", model=None):
        return self.result if language is None else TranscriptionResult(language, self.result.words)


class _FakeAlignmentService:
    def __init__(self, words):
        self.words = words

    def align(self, request):
        return types.SimpleNamespace(words=self.words, language=request.language or "en")


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def synthesise_to_file(self, text, speaker_wav, language_code, output_path):
        self.calls.append((text, speaker_wav, language_code, output_path))
        output_path.write_bytes(b"wav")


class ClientsSubtitlesSynthesisCoverageTests(unittest.TestCase):
    def test_client_helpers_cover_file_path_and_edge_flags(self):
        dictation_response = types.SimpleNamespace(
            job_id="tx-1",
            state=transcription_pb2.TRANSCRIPTION_JOB_STATE_SUCCEEDED,
            error_code="",
            error_message="",
            text="hello",
            language_code="en",
            words=[types.SimpleNamespace(content="hello", start_seconds=0.0, end_seconds=0.4)],
            created_at_unix_seconds=1.0,
            started_at_unix_seconds=2.0,
            finished_at_unix_seconds=3.0,
        )
        dictation_stub = _TranscriptionStub(transcribe_response=dictation_response)
        with patch("dictator.client.dictation.artifacts_pb2_grpc.ArtifactServiceStub", return_value=_ArtifactStub()), patch(
            "dictator.client.dictation.transcription_pb2_grpc.TranscriptionServiceStub",
            return_value=dictation_stub,
        ), patch(
            "dictator.client.dictation.upload_audio_artifact",
            return_value=types.SimpleNamespace(artifact_id="artifact-1"),
        ):
            client = DictationClient(channel=object())
            with tempfile.TemporaryDirectory() as tmpdir:
                audio = Path(tmpdir) / "audio.wav"
                audio.write_bytes(b"audio")
                result = client.dictate_file(audio, language_code="en")
        self.assertEqual(result.artifact_id, "artifact-1")
        self.assertEqual(dictation_stub.calls[0][0].audio_artifact_id, "artifact-1")
        with self.assertRaisesRegex(ValueError, "cannot both be set"):
            DictationClient._resolve_autodetect(language_code="en", autodetect_language=True)

        diarization_struct = types.SimpleNamespace()
        diarization_response = types.SimpleNamespace(
            job_id="dia-1",
            state=transcription_pb2.DIARIZATION_JOB_STATE_SUCCEEDED,
            error_code="",
            error_message="",
            text="hello",
            language_code="en",
            diarization=diarization_struct,
            diarization_artifact_id="json-1",
            created_at_unix_seconds=1.0,
            started_at_unix_seconds=2.0,
            finished_at_unix_seconds=3.0,
        )
        diarization_stub = _TranscriptionStub(diarize_response=diarization_response)
        with patch("dictator.client.diarization.artifacts_pb2_grpc.ArtifactServiceStub", return_value=_ArtifactStub()), patch(
            "dictator.client.diarization.transcription_pb2_grpc.TranscriptionServiceStub",
            return_value=diarization_stub,
        ), patch(
            "dictator.client.diarization.upload_audio_artifact",
            return_value=types.SimpleNamespace(artifact_id="artifact-2"),
        ), patch("dictator.client.diarization.MessageToDict", return_value={"text": "hello"}):
            client = DiarizationClient(channel=object())
            with tempfile.TemporaryDirectory() as tmpdir:
                audio = Path(tmpdir) / "audio.wav"
                audio.write_bytes(b"audio")
                result = client.diarize_file(audio, autodetect_language=True, utterance_gap_seconds=0.25)
        self.assertEqual(result.source_artifact_id, "artifact-2")
        self.assertEqual(diarization_stub.calls[0][0].utterance_gap_seconds, 0.25)
        self.assertEqual(result.diarization_artifact_id, "json-1")

        subtitle_response = types.SimpleNamespace(
            job_id="sub-1",
            state=subtitle_pb2.SUBTITLE_JOB_STATE_SUCCEEDED,
            error_code="",
            error_message="",
            language_code="en",
            mode=subtitle_pb2.SUBTITLE_MODE_FORCED_ALIGNMENT,
            granularity=subtitle_pb2.SUBTITLE_GRANULARITY_SENTENCES,
            group_size=2,
            srt_artifact_id="srt-1",
            srt_text="1\n00:00:00,000 --> 00:00:00,400\nhello world\n",
            cues=[types.SimpleNamespace(content="hello world", start_seconds=0.0, end_seconds=0.4, item_count=2)],
            created_at_unix_seconds=1.0,
            started_at_unix_seconds=2.0,
            finished_at_unix_seconds=3.0,
        )
        subtitle_stub = _SubtitleStub(subtitle_response)
        with (
            patch("dictator.client.subtitles.artifacts_pb2_grpc.ArtifactServiceStub", return_value=_ArtifactStub()),
            patch("dictator.client.subtitles.subtitle_pb2_grpc.SubtitleServiceStub", return_value=subtitle_stub),
            patch(
                "dictator.client.subtitles.upload_audio_artifact",
                return_value=types.SimpleNamespace(artifact_id="artifact-3"),
            ),
        ):
            client = SubtitleClient(channel=object())
            with tempfile.TemporaryDirectory() as tmpdir:
                audio = Path(tmpdir) / "audio.wav"
                transcript = Path(tmpdir) / "transcript.txt"
                audio.write_bytes(b"audio")
                transcript.write_text("hello world", encoding="utf-8")
                result = client.render_file(
                    audio,
                    autodetect_language=True,
                    granularity="sentences",
                    group_size=2,
                    source_text_file=transcript,
                )
        self.assertEqual(
            result,
            SubtitleResult(
                language_code="en",
                mode="forced_alignment",
                granularity="sentences",
                group_size=2,
                source_artifact_id="artifact-3",
                srt_artifact_id="srt-1",
                srt_text="1\n00:00:00,000 --> 00:00:00,400\nhello world\n",
                cues=(
                    {
                        "content": "hello world",
                        "start": 0.0,
                        "end": 0.4,
                        "itemCount": 2,
                    },
                ),
            ),
        )
        request = subtitle_stub.calls[0][0]
        self.assertEqual(request.granularity, subtitle_pb2.SUBTITLE_GRANULARITY_SENTENCES)
        self.assertEqual(request.source_text, "hello world")
        self.assertEqual(request.source_text_name, "transcript.txt")

        subtitle_response.mode = subtitle_pb2.SUBTITLE_MODE_TRANSCRIPTION
        with (
            patch("dictator.client.subtitles.artifacts_pb2_grpc.ArtifactServiceStub", return_value=_ArtifactStub()),
            patch("dictator.client.subtitles.subtitle_pb2_grpc.SubtitleServiceStub", return_value=subtitle_stub),
            patch(
                "dictator.client.subtitles.upload_audio_artifact",
                return_value=types.SimpleNamespace(artifact_id="artifact-4"),
            ),
        ):
            client = SubtitleClient(channel=object())
            result = client.render_bytes(
                b"audio",
                language_code="en",
                autodetect_language=False,
                source_text="inline",
                source_text_name="inline.txt",
                include_srt_text=False,
            )
        self.assertEqual(result.mode, "transcription")
        request = subtitle_stub.calls[-1][0]
        self.assertEqual(request.granularity, subtitle_pb2.SUBTITLE_GRANULARITY_WORDS)
        self.assertEqual(request.source_text_name, "inline.txt")
        self.assertFalse(request.include_srt_text)

        self.assertEqual(
            SubtitleClient._resolve_source_text(
                source_text="inline",
                source_text_file=None,
                source_text_name="",
            ),
            ("inline", "transcript.txt"),
        )
        self.assertEqual(
            SubtitleClient._resolve_granularity("words"),
            subtitle_pb2.SUBTITLE_GRANULARITY_WORDS,
        )
        self.assertEqual(
            SubtitleClient._resolve_mode(subtitle_pb2.SUBTITLE_MODE_FORCED_ALIGNMENT),
            "forced_alignment",
        )
        with self.assertRaisesRegex(ValueError, "granularity"):
            SubtitleClient._resolve_granularity("paragraphs")
        with self.assertRaisesRegex(ValueError, "cannot both be set"):
            SubtitleClient._resolve_source_text(
                source_text="inline",
                source_text_file=Path("transcript.txt"),
                source_text_name="",
            )

    def test_diarization_models_subtitle_helpers_and_render_errors(self):
        word = DiarizedWord("hello", 0.0, 0.4, "S1")
        utterance = DiarizedUtterance("S1", 0.0, 0.4, "hello", (word,))
        result = DiarizeAudioResult(
            language="en",
            text="hello",
            words=(word,),
            utterances=(utterance,),
            speakers=(SpeakerSummary("S1", 1, 1, 1.0),),
            speaker_segments=(SpeakerSegment("S1", 0.0, 1.0),),
        )
        self.assertEqual(word.to_legacy_dict()["speaker"], "S1")
        self.assertIn("words", utterance.to_json_dict())
        self.assertNotIn("words", utterance.to_json_dict(include_words=False))
        self.assertIn("speakerSegments", result.to_json_dict(include_speaker_segments=True))

        with self.assertRaisesRegex(ProcessingError, "include text"):
            _coerce_word_bounds("  ", 0.0, 0.1)
        with self.assertRaisesRegex(ProcessingError, "require timestamps"):
            _coerce_word_bounds("hello", None, None)
        self.assertEqual(_coerce_word_bounds("hello", 1.0, 0.5).end_seconds, 1.0)
        self.assertEqual(words_from_transcription([WordSegment(" ", 0.0, 0.1)]), ())
        self.assertEqual(words_from_alignment([types.SimpleNamespace(text="", start_seconds=0.0, end_seconds=0.1)]), ())
        self.assertEqual(sentence_units(()), ())
        units = (TimedWord("Hello.", 0.0, 0.4), TimedWord("Again", 0.5, 0.8))
        self.assertEqual(sentence_units(units)[1].text, "Again")
        with self.assertRaisesRegex(ValidationError, "group_size"):
            grouped_cues((), 0)
        self.assertEqual(render_srt(()), "")

        service = SubtitleService(
            transcription_service=_FakeTranscriptionService(
                TranscriptionResult(language="en", words=(WordSegment("hello", 0.0, 0.4),))
            ),
            alignment_service=_FakeAlignmentService((types.SimpleNamespace(text="hello", start_seconds=0.0, end_seconds=0.4),)),
        )
        with self.assertRaisesRegex(ValidationError, "supported"):
            service.render(RenderSubtitlesRequest(audio_path=Path("sample.wav"), output_format="vtt"))
        with self.assertRaisesRegex(ValidationError, "granularity"):
            service.render(RenderSubtitlesRequest(audio_path=Path("sample.wav"), granularity="letters"))

    def test_synthesis_backend_service_and_cleanup_helpers(self):
        fake_model = Mock()
        fake_model.create_voice_clone_prompt.return_value = "prompt"
        fake_model.generate_voice_clone.return_value = ([__import__("numpy").array([0.1, -0.1], dtype=__import__("numpy").float32)], 24000)
        fake_model._build_assistant_text = lambda text: text
        fake_model._tokenize_texts = lambda texts: [__import__("numpy").zeros((1, max(1, len(texts[0].split()))), dtype=__import__("numpy").int64)]
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
                "qwen_tts": types.SimpleNamespace(Qwen3TTSModel=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: fake_model)),
            },
        ):
            backend = Qwen3TTSBackend(model_id="model-id", dtype="float32", text_token_budget=32)
            loaded = backend.load()
            session = backend.open_session(
                __import__("dictator.synthesis.models", fromlist=["SynthesisRequest"]).SynthesisRequest(
                    engine=__import__("dictator.synthesis.models", fromlist=["SynthesisEngine"]).SynthesisEngine.QWEN3,
                    speaker_wav=Path("speaker.wav"),
                    text="hello",
                    language_code="en",
                    cap_seconds=None,
                    speaker_transcript_text="reference transcript",
                )
            )
            chunk = session.synthesise_chunk("hello")
        self.assertIs(loaded, fake_model)
        self.assertGreater(chunk.duration_seconds, 0.0)

        backend_module = __import__("dictator.synthesis.service", fromlist=["SpeechSynthesisService", "SynthesisChunk", "SynthesisedAudioChunk"])

        class FakeSession:
            def build_chunks(self, text):
                if not text:
                    return ()
                return (backend_module.SynthesisChunk.from_text(text),)

            def refine_chunk(self, chunk):
                return (chunk,)

            def synthesise_chunk(self, text):
                return backend_module.SynthesisedAudioChunk(
                    samples=__import__("numpy").array([0.1, -0.1], dtype=__import__("numpy").float32),
                    sample_rate=2,
                    duration_seconds=1.0,
                )

        class FakeBackend:
            engine = SynthesisEngine.QWEN3

            def open_session(self, request):
                return FakeSession()

        service = backend_module.SpeechSynthesisService(backends={SynthesisEngine.QWEN3: FakeBackend()})
        with self.assertRaisesRegex(ValueError, "No text chunks"):
            service.synthesise_text(
                SynthesisRequest(
                    engine=SynthesisEngine.QWEN3,
                    speaker_wav=Path("speaker.wav"),
                    text="",
                    language_code="en",
                    cap_seconds=None,
                    speaker_transcript_text="reference transcript",
                )
            )

        with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(write=Mock())}):
            result = service.synthesise_text(
                SynthesisRequest(
                    engine=SynthesisEngine.QWEN3,
                    speaker_wav=Path("speaker.wav"),
                    text="hello",
                    language_code="en",
                    cap_seconds=None,
                    speaker_transcript_text="reference transcript",
                )
            )
        self.assertEqual(result.segments[0].end_seconds, 1.0)

        with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(write=Mock())}):
            with self.assertRaisesRegex(ValueError, "fit within the length cap"):
                service.synthesise_text(
                    SynthesisRequest(
                        engine=SynthesisEngine.QWEN3,
                        speaker_wav=Path("speaker.wav"),
                        text="hello",
                        language_code="en",
                        cap_seconds=0.5,
                        speaker_transcript_text="reference transcript",
                    )
                )

        wrapped = SynthesisResult(result.temp_dir, result.wav_paths, result.segments)
        cleanup_synthesis_result(wrapped)
        self.assertFalse(wrapped.temp_dir.exists())


if __name__ == "__main__":
    unittest.main()
