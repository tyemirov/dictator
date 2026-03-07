import tempfile
from pathlib import Path
import unittest

from dictator.storage import LocalArtifactStore


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
