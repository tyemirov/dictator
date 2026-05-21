import sys
import tempfile
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from dictator.storage import ArtifactAudioMetadata, LocalArtifactStore
from dictator.storage.artifact_store import _normalise_container, _optional_float, _resolve_bit_depth


class LocalArtifactStoreTests(unittest.TestCase):
    def test_write_and_read_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(Path(tmpdir))
            record = store.write_artifact(
                [b"hello", b" ", b"world"],
                filename="greeting.txt",
                media_type="text/plain",
            )

            self.assertEqual(record.filename, "greeting.txt")
            self.assertEqual(record.media_type, "text/plain")
            self.assertEqual(record.size_bytes, 11)
            self.assertEqual(store.read_text(record.artifact_id), "hello world")

    def test_write_and_read_audio_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(Path(tmpdir))
            reservation = store.reserve_artifact("sample.wav", media_type="audio/wav")
            reservation.path.write_bytes(b"not really wav")
            record = store.finalize_artifact(
                reservation,
                audio_metadata=ArtifactAudioMetadata(
                    container="wav",
                    codec="pcm_s16le",
                    sample_rate_hz=24000,
                    channel_count=1,
                    bit_depth=16,
                    duration_seconds=1.25,
                ),
            )

            loaded = store.get_artifact(record.artifact_id)
            self.assertEqual(loaded.audio_metadata.container, "wav")
            self.assertEqual(loaded.audio_metadata.codec, "pcm_s16le")
            self.assertEqual(loaded.audio_metadata.sample_rate_hz, 24000)
            self.assertEqual(loaded.audio_metadata.duration_seconds, 1.25)

    def test_probe_audio_metadata_from_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(Path(tmpdir))
            fake_ffmpeg = types.SimpleNamespace(
                probe=lambda _path: {
                    "streams": [
                        {
                            "codec_type": "audio",
                            "codec_name": "pcm_s16le",
                            "sample_rate": "24000",
                            "channels": 1,
                            "bits_per_sample": 16,
                            "duration": "0.75",
                        }
                    ],
                    "format": {"format_name": "wav", "duration": "0.80"},
                }
            )
            with patch.dict(sys.modules, {"ffmpeg": fake_ffmpeg}):
                record = store.write_artifact([b"RIFF"], filename="sample.wav", media_type="audio/wav")

            self.assertEqual(record.audio_metadata.container, "wav")
            self.assertEqual(record.audio_metadata.codec, "pcm_s16le")
            self.assertEqual(record.audio_metadata.sample_rate_hz, 24000)
            self.assertEqual(record.audio_metadata.channel_count, 1)
            self.assertEqual(record.audio_metadata.bit_depth, 16)
            self.assertEqual(record.audio_metadata.duration_seconds, 0.75)

    def test_probe_audio_metadata_ignores_probe_without_audio_stream(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(Path(tmpdir))
            fake_ffmpeg = types.SimpleNamespace(
                probe=lambda _path: {
                    "streams": [{"codec_type": "video"}],
                    "format": {"format_name": "wav"},
                }
            )
            with patch.dict(sys.modules, {"ffmpeg": fake_ffmpeg}):
                record = store.write_artifact([b"RIFF"], filename="sample.wav", media_type="audio/wav")

            self.assertIsNone(record.audio_metadata)

    def test_audio_metadata_parse_helpers(self):
        self.assertEqual(_normalise_container("mp3,mp2"), "mp3")
        self.assertEqual(_normalise_container("matroska,webm"), "matroska")
        self.assertEqual(_resolve_bit_depth({"codec_name": "pcm_s16le"}), 16)
        self.assertEqual(_resolve_bit_depth({"codec_name": "pcm_s24le"}), 24)
        self.assertEqual(_resolve_bit_depth({"codec_name": "pcm_s32le"}), 32)
        self.assertEqual(_resolve_bit_depth({"codec_name": "opus"}), 0)
        self.assertIsNone(_optional_float(""))

    def test_iter_artifact_chunks_includes_offsets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalArtifactStore(Path(tmpdir))
            record = store.write_artifact(
                [b"abcdef"],
                filename="sample.bin",
                media_type="application/octet-stream",
            )

            chunks = list(store.iter_artifact_chunks(record.artifact_id, chunk_size=2))
            self.assertEqual([payload for _, _, payload, _ in chunks], [b"ab", b"cd", b"ef"])
            self.assertEqual([offset for _, offset, _, _ in chunks], [0, 2, 4])
            self.assertTrue(chunks[-1][-1])


if __name__ == "__main__":
    unittest.main()
