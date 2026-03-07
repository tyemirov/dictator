import unittest

from dictator.diarization.models import DiarizedWord
from dictator.diarization.service import (
    assign_words_to_speakers,
    build_speaker_summaries,
    build_speaker_segments,
    build_utterances,
    dominant_speaker_label,
)


class DiarizationServiceTests(unittest.TestCase):
    def test_assign_words_to_speakers_uses_overlap_and_nearest_segment(self):
        speaker_segments = build_speaker_segments(
            [
                (0.0, 1.0, "speaker_a"),
                (1.0, 2.0, "speaker_b"),
            ]
        )
        words = [
            {"content": "hello", "start": 0.1, "end": 0.3},
            {"content": "world", "start": 1.1, "end": 1.4},
            {"content": "tail", "start": 2.1, "end": 2.2},
        ]

        diarized_words = assign_words_to_speakers(words, speaker_segments)

        self.assertEqual([word.speaker for word in diarized_words], ["S1", "S2", "S2"])
        self.assertEqual(dominant_speaker_label(speaker_segments), "S1")

    def test_build_utterances_splits_on_speaker_change_and_gap(self):
        words = (
            DiarizedWord("hello", 0.0, 0.2, "S1"),
            DiarizedWord("again", 0.25, 0.45, "S1"),
            DiarizedWord("world", 1.4, 1.7, "S1"),
            DiarizedWord("switch", 1.8, 2.0, "S2"),
        )

        utterances = build_utterances(words, utterance_gap_seconds=0.5)

        self.assertEqual(len(utterances), 3)
        self.assertEqual(utterances[0].text, "hello again")
        self.assertEqual(utterances[1].text, "world")
        self.assertEqual(utterances[2].speaker, "S2")

    def test_build_speaker_segments_assigns_request_local_labels(self):
        speaker_segments = build_speaker_segments(
            [
                (5.0, 6.0, "speaker_b"),
                (0.0, 1.0, "speaker_a"),
                (1.0, 2.0, "speaker_b"),
            ]
        )

        self.assertEqual([segment.speaker for segment in speaker_segments], ["S1", "S2", "S2"])
        self.assertEqual(speaker_segments[0].raw_label, "speaker_a")

    def test_build_speaker_summaries_aggregates_word_utterance_and_duration_counts(self):
        speaker_segments = build_speaker_segments(
            [
                (0.0, 1.0, "speaker_a"),
                (1.0, 3.0, "speaker_b"),
            ]
        )
        words = (
            DiarizedWord("hello", 0.0, 0.2, "S1"),
            DiarizedWord("world", 1.2, 1.5, "S2"),
            DiarizedWord("again", 1.6, 1.9, "S2"),
        )
        utterances = build_utterances(words, utterance_gap_seconds=0.5)

        speakers = build_speaker_summaries(words, utterances, speaker_segments)

        self.assertEqual(len(speakers), 2)
        self.assertEqual(speakers[0].speaker, "S1")
        self.assertEqual(speakers[0].word_count, 1)
        self.assertEqual(speakers[1].utterance_count, 1)
        self.assertEqual(speakers[1].total_duration_seconds, 2.0)


if __name__ == "__main__":
    unittest.main()
