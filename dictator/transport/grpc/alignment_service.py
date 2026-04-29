"""Forced-alignment gRPC servicer."""

from __future__ import annotations

from pathlib import Path

from dictator.runtime import ValidationError
from dictator.runtime.jobs import (
    AlignmentJobRecord,
    AlignmentJobState,
    PreparedAlignmentJob,
    validate_alignment_job_id,
)
from dictator.speech.v1 import alignment_pb2, alignment_pb2_grpc, common_pb2

from .base import BaseServicer


class AlignmentServiceServicer(BaseServicer, alignment_pb2_grpc.AlignmentServiceServicer):
    def _resolve_prepared_alignment_job(self, request) -> PreparedAlignmentJob:
        audio = self.service_context.artifact_store.get_artifact(request.audio_artifact_id)
        transcript_text = request.transcript_text
        if request.transcript_artifact_id:
            transcript_text = self.service_context.artifact_store.read_text(request.transcript_artifact_id)
        if not transcript_text.strip():
            raise ValidationError(
                "dictator.grpc.alignment.missing_transcript",
                "transcript_text or transcript_artifact_id is required",
            )
        return PreparedAlignmentJob(
            audio_record=audio,
            transcript_text=transcript_text,
            language_code=request.language_code,
            remove_punctuation=request.remove_punctuation,
            include_srt_text=request.include_srt_text,
        )

    def _job_state_value(self, state: AlignmentJobState) -> int:
        mapping = {
            AlignmentJobState.QUEUED: alignment_pb2.ALIGNMENT_JOB_STATE_QUEUED,
            AlignmentJobState.RUNNING: alignment_pb2.ALIGNMENT_JOB_STATE_RUNNING,
            AlignmentJobState.SUCCEEDED: alignment_pb2.ALIGNMENT_JOB_STATE_SUCCEEDED,
            AlignmentJobState.FAILED: alignment_pb2.ALIGNMENT_JOB_STATE_FAILED,
            AlignmentJobState.CANCELED: alignment_pb2.ALIGNMENT_JOB_STATE_CANCELED,
        }
        return mapping[state]

    def _job_response(self, record: AlignmentJobRecord):
        response = alignment_pb2.GetAlignTranscriptJobResponse(
            job_id=record.job_id,
            state=self._job_state_value(record.state),
            error_code=record.error_code or "",
            error_message=record.error_message or "",
            language_code=record.language_code or "",
            srt_text=record.srt_text or "",
            srt_artifact_id=record.srt_artifact_id or "",
            created_at_unix_seconds=record.created_at_unix_seconds,
            started_at_unix_seconds=record.started_at_unix_seconds or 0.0,
            finished_at_unix_seconds=record.finished_at_unix_seconds or 0.0,
            source_artifact_id=record.audio_artifact_id,
        )
        response.words.extend(
            common_pb2.WordSegment(
                content=word.text,
                start_seconds=word.start_seconds,
                end_seconds=word.end_seconds,
            )
            for word in record.words
        )
        return response

    def AlignTranscript(self, request, context):
        with self._request_scope(context):
            prepared = self._resolve_prepared_alignment_job(request)
            service = self.service_context.execution_runtime.get_alignment_service()
            stem = Path(prepared.audio_record.filename).stem
            srt_reservation = self.service_context.artifact_store.reserve_artifact(
                f"{stem}.srt",
                media_type="application/x-subrip",
                fallback_suffix=".srt",
            )
            from dictator.alignment.models import AlignTranscriptRequest

            result = service.align(
                AlignTranscriptRequest(
                    audio_path=prepared.audio_record.path,
                    transcript_text=prepared.transcript_text,
                    language=prepared.language_code,
                    remove_punctuation=prepared.remove_punctuation,
                    transcript_source_name="transcript.txt",
                    output_srt_path=srt_reservation.path,
                )
            )
            srt_record = self.service_context.artifact_store.finalize_artifact(srt_reservation)
            response = alignment_pb2.AlignTranscriptResponse(
                language_code=result.language,
                srt_artifact_id=srt_record.artifact_id,
            )
            response.words.extend(
                common_pb2.WordSegment(
                    content=word.text,
                    start_seconds=word.start_seconds,
                    end_seconds=word.end_seconds,
                )
                for word in result.words
            )
            if prepared.include_srt_text:
                response.srt_text = result.srt_text
            return response

    def SubmitAlignTranscriptJob(self, request, context):
        with self._request_scope(context):
            if self.service_context.alignment_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.alignment.jobs_unavailable",
                    "alignment job manager is not configured",
                )
            prepared = self._resolve_prepared_alignment_job(request)
            record = self.service_context.alignment_job_manager.submit(prepared)
            return alignment_pb2.SubmitAlignTranscriptJobResponse(
                job_id=record.job_id,
                state=self._job_state_value(record.state),
            )

    def GetAlignTranscriptJob(self, request, context):
        with self._request_scope(context, is_inquiry=True):
            if self.service_context.alignment_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.alignment.jobs_unavailable",
                    "alignment job manager is not configured",
                )
            if not request.job_id.strip():
                raise ValidationError(
                    "dictator.grpc.alignment.job_id_required",
                    "job_id is required",
                )
            job_id = validate_alignment_job_id(request.job_id)
            return self._job_response(self.service_context.alignment_job_manager.get(job_id))

    def CancelAlignTranscriptJob(self, request, context):
        with self._request_scope(context, is_inquiry=True):
            if self.service_context.alignment_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.alignment.jobs_unavailable",
                    "alignment job manager is not configured",
                )
            if not request.job_id.strip():
                raise ValidationError(
                    "dictator.grpc.alignment.job_id_required",
                    "job_id is required",
                )
            job_id = validate_alignment_job_id(request.job_id)
            record = self.service_context.alignment_job_manager.cancel(job_id)
            return alignment_pb2.CancelAlignTranscriptJobResponse(
                job_id=record.job_id,
                state=self._job_state_value(record.state),
            )
