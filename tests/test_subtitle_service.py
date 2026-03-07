import tempfile
from pathlib import Path
import unittest

from dictator.alignment.models import AlignTranscriptResult, AlignedWord
from dictator.runtime import ValidationError
from dictator.subtitles.models import RenderSubtitlesRequest
from dictator.subtitles.service import SubtitleService
from dictator.transcription.models import TranscriptionResult, WordSegment


class FakeTranscriptionService:
    def __init__(self, result: TranscriptionResult) -> None:
        self.result = result
        self.calls = []

    def transcribe(self, audio, language=None, model_size="base", model=None, progress_cb=None):
        self.calls.append(
            {
                "audio": audio,
                "language": language,
                "model_size": model_size,
                "model": model,
            }
        )
        return self.result if language is None else TranscriptionResult(language=language, words=self.result.words)


class FakeAlignmentService:
    def __init__(self, words: tuple[AlignedWord, ...]) -> None:
        self.words = words
        self.calls = []

    def align(self, request):
        self.calls.append(request)
        return AlignTranscriptResult(
            audio_path=request.audio_path,
            language=request.language,
            words=self.words,
            srt_text="",
            output_srt_path=request.output_srt_path,
        )


class SubtitleServiceTests(unittest.TestCase):
    def test_render_groups_words_by_size(self):
        transcription = FakeTranscriptionService(
            TranscriptionResult(
                language="en",
                words=(
                    WordSegment("hello", 0.0, 0.4),
                    WordSegment("world", 0.4, 0.9),
                    WordSegment("again", 1.0, 1.4),
                ),
            )
        )
        service = SubtitleService(
            transcription_service=transcription,
            alignment_service=FakeAlignmentService(()),
        )

        result = service.render(
            RenderSubtitlesRequest(
                audio_path=Path("sample.wav"),
                granularity="words",
                group_size=2,
            )
        )

        self.assertEqual(result.mode, "transcription")
        self.assertEqual(result.language, "en")
        self.assertEqual(len(result.cues), 2)
        self.assertEqual(result.cues[0].text, "hello world")
        self.assertEqual(result.cues[0].start_seconds, 0.0)
        self.assertEqual(result.cues[0].end_seconds, 0.9)
        self.assertEqual(result.cues[0].item_count, 2)
        self.assertEqual(result.cues[1].text, "again")
        self.assertIn("hello world", result.srt_text)

    def test_render_groups_sentences_by_size(self):
        transcription = FakeTranscriptionService(
            TranscriptionResult(
                language="en",
                words=(
                    WordSegment("Hello", 0.0, 0.2),
                    WordSegment("world.", 0.2, 0.5),
                    WordSegment("How", 0.6, 0.7),
                    WordSegment("are", 0.7, 0.8),
                    WordSegment("you?", 0.8, 1.0),
                    WordSegment("Thanks.", 1.1, 1.3),
                ),
            )
        )
        service = SubtitleService(
            transcription_service=transcription,
            alignment_service=FakeAlignmentService(()),
        )

        result = service.render(
            RenderSubtitlesRequest(
                audio_path=Path("sample.wav"),
                granularity="sentences",
                group_size=2,
            )
        )

        self.assertEqual(len(result.cues), 2)
        self.assertEqual(result.cues[0].text, "Hello world. How are you?")
        self.assertEqual(result.cues[0].item_count, 2)
        self.assertEqual(result.cues[1].text, "Thanks.")
        self.assertEqual(result.cues[1].item_count, 1)

    def test_render_with_source_text_uses_alignment_and_detected_language(self):
        transcription = FakeTranscriptionService(
            TranscriptionResult(
                language="fr",
                words=(WordSegment("bonjour", 0.0, 0.4),),
            )
        )
        alignment = FakeAlignmentService(
            (
                AlignedWord("bonjour", 0.0, 0.4),
                AlignedWord("monde", 0.4, 0.9),
            )
        )
        service = SubtitleService(
            transcription_service=transcription,
            alignment_service=alignment,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.srt"
            result = service.render(
                RenderSubtitlesRequest(
                    audio_path=Path("sample.wav"),
                    granularity="words",
                    group_size=1,
                    source_text="bonjour monde",
                    source_text_name="source.txt",
                    output_srt_path=output_path,
                ),
                model=object(),
            )
            persisted = output_path.read_text(encoding="utf-8")

        self.assertEqual(result.mode, "forced_alignment")
        self.assertEqual(result.language, "fr")
        self.assertEqual(alignment.calls[0].language, "fr")
        self.assertTrue(transcription.calls)
        self.assertEqual(persisted, result.srt_text)

    def test_render_rejects_invalid_group_size(self):
        service = SubtitleService(
            transcription_service=FakeTranscriptionService(
                TranscriptionResult(language="en", words=())
            ),
            alignment_service=FakeAlignmentService(()),
        )

        with self.assertRaisesRegex(ValidationError, "group_size"):
            service.render(
                RenderSubtitlesRequest(
                    audio_path=Path("sample.wav"),
                    group_size=0,
                )
            )


if __name__ == "__main__":
    unittest.main()
