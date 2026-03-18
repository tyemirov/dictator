
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import tempfile
import time
import sys

class RuntimeResilienceCoverageTests(unittest.TestCase):
    def test_read_record_retry_failure(self):
        from dictator.runtime.jobs import LocalSynthesisJobStore
        # Test that it eventually raises after 3 attempts
        store = LocalSynthesisJobStore(Path("/tmp/fake-jobs"))
        with patch.object(Path, "read_text") as mock_read:
            mock_read.side_effect = FileNotFoundError("transient")
            with self.assertRaises(FileNotFoundError):
                store._read_record_with_retry(Path("/tmp/fake-jobs/job.json"))
            self.assertEqual(mock_read.call_count, 3)

    def test_read_record_retry_success(self):
        from dictator.runtime.jobs import LocalSynthesisJobStore
        # Test that it succeeds on second attempt
        store = LocalSynthesisJobStore(Path("/tmp/fake-jobs"))
        with patch.object(Path, "read_text") as mock_read:
            mock_read.side_effect = [FileNotFoundError("transient"), '{"job_id": "123", "state": "queued", "engine": "qwen3", "language_code": "en", "include_timeline": false, "speaker_artifact_id": "spk1", "created_at_unix_seconds": 123.0, "estimated_total_chunks": 0, "completed_chunks": 0}']
            record = store._read_record_with_retry(Path("/tmp/fake-jobs/job.json"))
            self.assertEqual(record.job_id, "123")
            self.assertEqual(mock_read.call_count, 2)

    def test_execute_synthesis_request_cleanup_on_failure(self):
        # Mock the ffmpeg_ops module to avoid import error
        mock_ffmpeg = MagicMock()
        sys.modules['dictator.audio.ffmpeg_ops'] = mock_ffmpeg
        
        from dictator.synthesis.workflow import execute_synthesis_request, PreparedSynthesisRequest
        from dictator.synthesis.models import SynthesisRequest, SynthesisEngine, SynthesisResult
        from dictator.storage import LocalArtifactStore, ArtifactRecord

        artifact_store = MagicMock(spec=LocalArtifactStore)
        execution_runtime = MagicMock()
        synthesis_service = execution_runtime.get_synthesis_service.return_value
        
        # Setup result
        result = MagicMock(spec=SynthesisResult)
        result.wav_paths = [Path("chunk1.wav")]
        result.temp_dir = Path("/tmp/fake-synthesis")
        synthesis_service.synthesise_text.return_value = result
        
        speaker_record = MagicMock(spec=ArtifactRecord)
        speaker_record.filename = "speaker.wav"
        speaker_record.artifact_id = "spk1"
        speaker_record.path = Path("speaker.wav")
        
        synthesis_request = SynthesisRequest(
            engine=SynthesisEngine.QWEN3,
            speaker_wav=Path("speaker.wav"),
            text="hello",
            language_code="en",
            cap_seconds=10.0,
            speaker_artifact_id="spk1"
        )
        
        prepared = PreparedSynthesisRequest(
            speaker_record=speaker_record,
            synthesis_request=synthesis_request,
            include_timeline=False
        )
        
        reservation = MagicMock()
        artifact_store.reserve_artifact.return_value = reservation
        
        # Make concat_normalise fail
        mock_ffmpeg.concat_normalise.side_effect = RuntimeError("ffmpeg failed")
        
        with self.assertRaises(RuntimeError):
            execute_synthesis_request(
                artifact_store=artifact_store,
                execution_runtime=execution_runtime,
                prepared=prepared
            )
        
        # Verify discard_reservation was called
        artifact_store.discard_reservation.assert_called_once_with(reservation)

if __name__ == "__main__":
    unittest.main()
