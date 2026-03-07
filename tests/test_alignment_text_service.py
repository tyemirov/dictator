import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from dictator.alignment.models import AlignedWord, AlignTranscriptRequest
from dictator.alignment.service import AlignmentService, align_transcript
from dictator.alignment import text as alignment_text
from dictator.runtime import ValidationError


class FakeAlignmentBackend:
    def __init__(self, words):
        self.words = words
        self.calls = []

    def align(self, **kwargs):
        self.calls.append(kwargs)
        return self.words


class AlignmentTextServiceTests(unittest.TestCase):
    def test_aligned_word_validates_bounds_and_serializes(self):
        word = AlignedWord("hello", 0.1, 0.2)
        self.assertEqual(word.to_legacy_dict(), {"content": "hello", "start": 0.1, "end": 0.2})

        with self.assertRaisesRegex(ValueError, "text is empty"):
            AlignedWord("", 0.1, 0.2)
        with self.assertRaisesRegex(ValueError, "start is negative"):
            AlignedWord("hello", -0.1, 0.2)
        with self.assertRaisesRegex(ValueError, "end is not after start"):
            AlignedWord("hello", 0.2, 0.2)

    def test_alignment_service_uses_default_backend_when_omitted(self):
        backend = object()
        with patch("dictator.alignment.service.WhisperXAlignmentBackend", return_value=backend):
            service = AlignmentService()
        self.assertIs(service.backend, backend)

    def test_alignment_service_aligns_writes_srt_and_normalizes_language(self):
        words = (AlignedWord("hello", 0.0, 0.4), AlignedWord("world", 0.4, 0.8))
        backend = FakeAlignmentBackend(words)
        service = AlignmentService(backend=backend)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.srt"
            result = service.align(
                AlignTranscriptRequest(
                    audio_path=Path("sample.wav"),
                    transcript_text="Hello world",
                    language="",
                    transcript_source_name="transcript.txt",
                    output_srt_path=output_path,
                )
            )
            persisted = output_path.read_text(encoding="utf-8")

        self.assertEqual(result.language, "en")
        self.assertEqual(result.words, words)
        self.assertEqual(backend.calls[0]["language"], "en")
        self.assertIn("00:00:00,000 --> 00:00:00,400", result.srt_text)
        self.assertEqual(persisted, result.srt_text)

    def test_align_transcript_convenience_wrapper(self):
        request = AlignTranscriptRequest(audio_path=Path("sample.wav"), transcript_text="hello")
        sentinel = object()
        with patch.object(AlignmentService, "align", return_value=sentinel) as align_mock:
            result = align_transcript(request)
        self.assertIs(result, sentinel)
        align_mock.assert_called_once_with(request)

    def test_is_srt_text_recognizes_extension_and_timestamp_lines(self):
        srt_text = "1\n00:00:00,000 --> 00:00:01,000\nHello\n"
        self.assertTrue(alignment_text.is_srt_text("captions.srt", "Hello"))
        self.assertTrue(alignment_text.is_srt_text("captions.txt", srt_text))
        self.assertFalse(alignment_text.is_srt_text("captions.txt", "Hello world"))

    def test_sanitize_and_normalize_transcript(self):
        srt_text = "\ufeff1\n00:00:00,000 --> 00:00:01,000\nHello\n\n2\n00:00:01,000 --> 00:00:02,000\nworld\n"
        self.assertEqual(alignment_text.sanitize_srt_text(srt_text), "Hello world")
        self.assertEqual(alignment_text.normalize_transcript("  hello\n world  "), "hello world")
        self.assertEqual(alignment_text.normalize_transcript(srt_text, "captions.srt"), "Hello world")
        with self.assertRaisesRegex(ValidationError, "contains no words"):
            alignment_text.normalize_transcript("  \n  ")

    def test_normalize_transcript_for_alignment_handles_punctuation_removal(self):
        self.assertEqual(
            alignment_text.normalize_transcript_for_alignment(
                "Hello, world!", "transcript.txt", False
            ),
            "Hello, world!",
        )
        self.assertEqual(
            alignment_text.normalize_transcript_for_alignment(
                "Hello, world!", "transcript.txt", True
            ),
            "Hello world",
        )
        with self.assertRaisesRegex(ValidationError, "after punctuation removal"):
            alignment_text.normalize_transcript_for_alignment("!!!", "transcript.txt", True)

    def test_language_and_token_helpers(self):
        self.assertEqual(alignment_text.detect_default_language("привет"), "ru")
        self.assertEqual(alignment_text.detect_default_language("hello"), "en")
        self.assertEqual(alignment_text.normalize_language_value("", "fr"), "fr")
        self.assertEqual(alignment_text.normalize_language_value("EN", "fr"), "en")
        with self.assertRaisesRegex(ValidationError, "unsupported language"):
            alignment_text.normalize_language_value("xx", "en")

        self.assertTrue(alignment_text.is_punctuation_token("..."))
        self.assertFalse(alignment_text.is_punctuation_token("hi!"))
        self.assertEqual(alignment_text.remove_punctuation_from_transcript("a,b!"), "a b ")
        self.assertEqual(alignment_text.strip_punctuation_from_token("(hi!)"), "hi")


if __name__ == "__main__":
    unittest.main()
