from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.modules.setdefault("ffmpeg", types.SimpleNamespace())
sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))

from dictator.client.diarization import DiarizationClient
from dictator.client.dictation import DictationClient
from dictator.diarization.models import DiarizeAudioResult, DiarizedUtterance, DiarizedWord, SpeakerSegment, SpeakerSummary
from dictator.runtime import ProcessingError, ValidationError
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
from dictator.synthesis.models import SpeechSegment, SynthesisResult
from dictator.synthesis.service import XTTSBackend, cleanup_synthesis_result, synthesise
from dictator.transcription.models import TranscriptionResult, WordSegment


class _ArtifactStub:
    pass


class _TranscriptionStub:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def Transcribe(self, request, metadata=()):
        self.calls.append((request, metadata))
        return self.response

    def DiarizeAudio(self, request, metadata=()):
        self.calls.append((request, metadata))
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
            text="hello",
            language_code="en",
            words=[types.SimpleNamespace(content="hello", start_seconds=0.0, end_seconds=0.4)],
        )
        dictation_stub = _TranscriptionStub(dictation_response)
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
            text="hello",
            language_code="en",
            diarization=diarization_struct,
            diarization_artifact_id="json-1",
        )
        diarization_stub = _TranscriptionStub(diarization_response)
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
        fake_tts = Mock()
        fake_tts.to.return_value = fake_tts
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "TTS": types.SimpleNamespace(api=types.SimpleNamespace(TTS=lambda model_id: fake_tts)),
                "TTS.api": types.SimpleNamespace(TTS=lambda model_id: fake_tts),
            },
        ):
            backend = XTTSBackend("model-id")
            loaded = backend.load()
            backend.synthesise_to_file("hello", Path("speaker.wav"), "en", Path("out.wav"))
        self.assertIs(loaded, fake_tts)
        fake_tts.tts_to_file.assert_called_once()

        backend = _FakeBackend()
        service = __import__("dictator.synthesis.service", fromlist=["SpeechSynthesisService"]).SpeechSynthesisService(backend=backend)
        with self.assertRaisesRegex(ValueError, "No text chunks"):
            service.synthesise(Path("speaker.wav"), [], None, "en")

        with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(info=lambda path: types.SimpleNamespace(frames=24000, samplerate=24000))}):
            result = service.synthesise(Path("speaker.wav"), ["hello"], None, "en")
        self.assertEqual(result.segments[0].end_seconds, 1.0)

        with patch.dict(sys.modules, {"soundfile": types.SimpleNamespace(info=lambda path: types.SimpleNamespace(frames=48000, samplerate=24000))}):
            with self.assertRaisesRegex(ValueError, "fit within the length cap"):
                service.synthesise(Path("speaker.wav"), ["hello"], 1.0, "en")

        wrapped = SynthesisResult(result.temp_dir, result.wav_paths, result.segments)
        cleanup_synthesis_result(wrapped)
        self.assertFalse(wrapped.temp_dir.exists())

        with patch("dictator.synthesis.service.SpeechSynthesisService.synthesise", return_value=SynthesisResult(Path("/tmp"), (), ())) as synth_mock:
            self.assertEqual(synthesise(Path("speaker.wav"), ["hello"], None, "en"), ([], []))
        synth_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
