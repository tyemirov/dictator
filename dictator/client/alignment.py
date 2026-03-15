"""Alignment gRPC client helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import grpc

from dictator.speech.v1 import alignment_pb2, alignment_pb2_grpc, artifacts_pb2_grpc

from ._jobs import wait_for_job
from ._uploads import DEFAULT_CHUNK_BYTES, DEFAULT_MEDIA_TYPE, upload_audio_artifact


@dataclass(frozen=True)
class AlignmentResult:
    language_code: str
    source_artifact_id: str
    srt_artifact_id: str
    srt_text: str
    words: tuple[dict[str, float | str], ...]


@dataclass(frozen=True)
class AlignmentJob:
    job_id: str
    state: str
    source_artifact_id: str = ""
    error_code: str = ""
    error_message: str = ""
    result: AlignmentResult | None = None
    created_at_unix_seconds: float = 0.0
    started_at_unix_seconds: float = 0.0
    finished_at_unix_seconds: float = 0.0


class AlignmentClient:
    """Upload audio and call AlignmentService in one step."""

    def __init__(
        self,
        channel: grpc.Channel,
        metadata: Sequence[tuple[str, str]] = (),
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        self._artifact_stub = artifacts_pb2_grpc.ArtifactServiceStub(channel)
        self._alignment_stub = alignment_pb2_grpc.AlignmentServiceStub(channel)
        self._metadata = tuple(metadata)
        self._chunk_bytes = chunk_bytes

    def align_file(
        self,
        audio_path: Path,
        *,
        transcript_text: str | None = None,
        transcript_file: Path | None = None,
        transcript_artifact_id: str = "",
        language_code: str = "",
        remove_punctuation: bool = False,
        include_srt_text: bool = True,
        media_type: str | None = None,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> AlignmentResult:
        resolved_text, resolved_artifact_id = self._resolve_transcript_source(
            transcript_text=transcript_text,
            transcript_file=transcript_file,
            transcript_artifact_id=transcript_artifact_id,
        )
        return self.align_bytes(
            audio_path.read_bytes(),
            filename=audio_path.name,
            media_type=media_type,
            transcript_text=resolved_text,
            transcript_artifact_id=resolved_artifact_id,
            language_code=language_code,
            remove_punctuation=remove_punctuation,
            include_srt_text=include_srt_text,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def align_bytes(
        self,
        payload: bytes,
        *,
        filename: str = "audio.webm",
        media_type: str | None = None,
        transcript_text: str | None = None,
        transcript_artifact_id: str = "",
        language_code: str = "",
        remove_punctuation: bool = False,
        include_srt_text: bool = True,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> AlignmentResult:
        try:
            submitted = self.submit_align_bytes_job(
                payload,
                filename=filename,
                media_type=media_type,
                transcript_text=transcript_text,
                transcript_artifact_id=transcript_artifact_id,
                language_code=language_code,
                remove_punctuation=remove_punctuation,
                include_srt_text=include_srt_text,
            )
            finished = self.wait_for_alignment_job(
                submitted.job_id,
                timeout_seconds=timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            if finished.result is None:
                raise RuntimeError("alignment job succeeded without a result payload")
            return AlignmentResult(
                language_code=finished.result.language_code,
                source_artifact_id=submitted.source_artifact_id,
                srt_artifact_id=finished.result.srt_artifact_id,
                srt_text=finished.result.srt_text,
                words=finished.result.words,
            )
        except grpc.RpcError as error:
            if not self._should_fallback_to_sync(error):
                raise

        resolved_text, resolved_artifact_id = self._resolve_transcript_source(
            transcript_text=transcript_text,
            transcript_file=None,
            transcript_artifact_id=transcript_artifact_id,
        )
        artifact = upload_audio_artifact(
            self._artifact_stub,
            metadata=self._metadata,
            chunk_bytes=self._chunk_bytes,
            payload=payload,
            filename=filename,
            media_type=media_type or DEFAULT_MEDIA_TYPE,
        )
        response = self._alignment_stub.AlignTranscript(
            self._build_request(
                audio_artifact_id=artifact.artifact_id,
                transcript_text=resolved_text,
                transcript_artifact_id=resolved_artifact_id,
                language_code=language_code,
                remove_punctuation=remove_punctuation,
                include_srt_text=include_srt_text,
            ),
            metadata=self._metadata,
        )
        return AlignmentResult(
            language_code=response.language_code,
            source_artifact_id=artifact.artifact_id,
            srt_artifact_id=response.srt_artifact_id,
            srt_text=response.srt_text,
            words=self._words_from_segments(response.words),
        )

    def submit_align_file_job(
        self,
        audio_path: Path,
        *,
        transcript_text: str | None = None,
        transcript_file: Path | None = None,
        transcript_artifact_id: str = "",
        language_code: str = "",
        remove_punctuation: bool = False,
        include_srt_text: bool = True,
        media_type: str | None = None,
    ) -> AlignmentJob:
        resolved_text, resolved_artifact_id = self._resolve_transcript_source(
            transcript_text=transcript_text,
            transcript_file=transcript_file,
            transcript_artifact_id=transcript_artifact_id,
        )
        return self.submit_align_bytes_job(
            audio_path.read_bytes(),
            filename=audio_path.name,
            media_type=media_type,
            transcript_text=resolved_text,
            transcript_artifact_id=resolved_artifact_id,
            language_code=language_code,
            remove_punctuation=remove_punctuation,
            include_srt_text=include_srt_text,
        )

    def submit_align_bytes_job(
        self,
        payload: bytes,
        *,
        filename: str = "audio.webm",
        media_type: str | None = None,
        transcript_text: str | None = None,
        transcript_artifact_id: str = "",
        language_code: str = "",
        remove_punctuation: bool = False,
        include_srt_text: bool = True,
    ) -> AlignmentJob:
        resolved_text, resolved_artifact_id = self._resolve_transcript_source(
            transcript_text=transcript_text,
            transcript_file=None,
            transcript_artifact_id=transcript_artifact_id,
        )
        artifact = upload_audio_artifact(
            self._artifact_stub,
            metadata=self._metadata,
            chunk_bytes=self._chunk_bytes,
            payload=payload,
            filename=filename,
            media_type=media_type or DEFAULT_MEDIA_TYPE,
        )
        response = self._alignment_stub.SubmitAlignTranscriptJob(
            self._build_request(
                audio_artifact_id=artifact.artifact_id,
                transcript_text=resolved_text,
                transcript_artifact_id=resolved_artifact_id,
                language_code=language_code,
                remove_punctuation=remove_punctuation,
                include_srt_text=include_srt_text,
            ),
            metadata=self._metadata,
        )
        return AlignmentJob(
            job_id=response.job_id,
            state=alignment_pb2.AlignmentJobState.Name(response.state),
            source_artifact_id=artifact.artifact_id,
        )

    def get_alignment_job(self, job_id: str) -> AlignmentJob:
        response = self._alignment_stub.GetAlignTranscriptJob(
            alignment_pb2.GetAlignTranscriptJobRequest(job_id=job_id),
            metadata=self._metadata,
        )
        result = None
        if response.state == alignment_pb2.ALIGNMENT_JOB_STATE_SUCCEEDED:
            result = AlignmentResult(
                language_code=response.language_code,
                source_artifact_id="",
                srt_artifact_id=response.srt_artifact_id,
                srt_text=response.srt_text,
                words=self._words_from_segments(response.words),
            )
        return AlignmentJob(
            job_id=response.job_id,
            state=alignment_pb2.AlignmentJobState.Name(response.state),
            error_code=response.error_code,
            error_message=response.error_message,
            result=result,
            created_at_unix_seconds=response.created_at_unix_seconds,
            started_at_unix_seconds=response.started_at_unix_seconds,
            finished_at_unix_seconds=response.finished_at_unix_seconds,
        )

    def wait_for_alignment_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float | None = 300.0,
        poll_interval_seconds: float = 1.0,
    ) -> AlignmentJob:
        return wait_for_job(
            lambda: self.get_alignment_job(job_id),
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    @staticmethod
    def _build_request(
        *,
        audio_artifact_id: str,
        transcript_text: str | None,
        transcript_artifact_id: str,
        language_code: str,
        remove_punctuation: bool,
        include_srt_text: bool,
    ) -> alignment_pb2.AlignTranscriptRequest:
        request = alignment_pb2.AlignTranscriptRequest(
            audio_artifact_id=audio_artifact_id,
            language_code=language_code,
            remove_punctuation=remove_punctuation,
            include_srt_text=include_srt_text,
        )
        if transcript_text is not None:
            request.transcript_text = transcript_text
        else:
            request.transcript_artifact_id = transcript_artifact_id
        return request

    @staticmethod
    def _resolve_transcript_source(
        *,
        transcript_text: str | None,
        transcript_file: Path | None,
        transcript_artifact_id: str,
    ) -> tuple[str | None, str]:
        normalized_artifact_id = transcript_artifact_id.strip()
        variants = sum(
            1
            for candidate in (
                transcript_text is not None,
                transcript_file is not None,
                bool(normalized_artifact_id),
            )
            if candidate
        )
        if variants != 1:
            raise ValueError(
                "exactly one of transcript_text, transcript_file, or transcript_artifact_id must be set"
            )
        if transcript_file is not None:
            return transcript_file.read_text(encoding="utf-8"), ""
        return transcript_text, normalized_artifact_id

    @staticmethod
    def _should_fallback_to_sync(error: grpc.RpcError) -> bool:
        if error.code() == grpc.StatusCode.UNIMPLEMENTED:
            return True
        if error.code() != grpc.StatusCode.INVALID_ARGUMENT:
            return False
        return (error.details() or "").endswith("job manager is not configured")

    @staticmethod
    def _words_from_segments(words) -> tuple[dict[str, float | str], ...]:
        return tuple(
            {
                "content": word.content,
                "start": word.start_seconds,
                "end": word.end_seconds,
            }
            for word in words
        )
