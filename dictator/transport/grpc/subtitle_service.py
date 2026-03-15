"""Grouped subtitle gRPC servicer."""

from __future__ import annotations

from pathlib import Path

from dictator.runtime import ValidationError
from dictator.runtime.jobs import (
    PreparedSubtitleJob,
    SubtitleJobRecord,
    SubtitleJobState,
    validate_subtitle_job_id,
)
from dictator.speech.v1 import subtitle_pb2, subtitle_pb2_grpc

from .base import BaseServicer, DEFAULT_MODEL_SIZE


class SubtitleServiceServicer(BaseServicer, subtitle_pb2_grpc.SubtitleServiceServicer):
    def _resolve_prepared_subtitle_job(self, request) -> PreparedSubtitleJob:
        if request.output_format not in (
            subtitle_pb2.SUBTITLE_FORMAT_SRT,
            subtitle_pb2.SUBTITLE_FORMAT_UNSPECIFIED,
        ):
            raise ValidationError(
                "dictator.grpc.subtitles.unsupported_format",
                "only SUBTITLE_FORMAT_SRT is supported",
            )
        if request.granularity == subtitle_pb2.SUBTITLE_GRANULARITY_WORDS:
            granularity = "words"
        elif request.granularity == subtitle_pb2.SUBTITLE_GRANULARITY_SENTENCES:
            granularity = "sentences"
        else:
            raise ValidationError(
                "dictator.grpc.subtitles.invalid_granularity",
                "granularity must be WORDS or SENTENCES",
            )
        if request.group_size <= 0:
            raise ValidationError(
                "dictator.grpc.subtitles.invalid_group_size",
                "group_size must be positive",
            )
        audio = self.service_context.artifact_store.get_artifact(request.audio_artifact_id)
        language = self._resolve_language_request(
            language_code=request.language_code,
            autodetect_language=request.autodetect_language,
            error_scope="dictator.grpc.subtitles",
        )
        source_text = None
        source_text_name = request.source_text_name or "transcript.txt"
        if request.source_text_artifact_id:
            source_record = self.service_context.artifact_store.get_artifact(request.source_text_artifact_id)
            source_text = self.service_context.artifact_store.read_text(request.source_text_artifact_id)
            source_text_name = request.source_text_name or source_record.filename
        elif request.source_text:
            source_text = request.source_text
        return PreparedSubtitleJob(
            audio_record=audio,
            language_code=language,
            model_size=request.model_size or DEFAULT_MODEL_SIZE,
            granularity=granularity,
            group_size=request.group_size,
            source_text=source_text,
            source_text_name=source_text_name,
            include_srt_text=request.include_srt_text,
        )

    def _subtitle_job_state_value(self, state: SubtitleJobState) -> int:
        mapping = {
            SubtitleJobState.QUEUED: subtitle_pb2.SUBTITLE_JOB_STATE_QUEUED,
            SubtitleJobState.RUNNING: subtitle_pb2.SUBTITLE_JOB_STATE_RUNNING,
            SubtitleJobState.SUCCEEDED: subtitle_pb2.SUBTITLE_JOB_STATE_SUCCEEDED,
            SubtitleJobState.FAILED: subtitle_pb2.SUBTITLE_JOB_STATE_FAILED,
        }
        return mapping[state]

    def _subtitle_job_response(self, record: SubtitleJobRecord):
        mode = subtitle_pb2.SUBTITLE_MODE_UNSPECIFIED
        if record.mode == "transcription":
            mode = subtitle_pb2.SUBTITLE_MODE_TRANSCRIPTION
        elif record.mode == "forced_alignment":
            mode = subtitle_pb2.SUBTITLE_MODE_FORCED_ALIGNMENT
        granularity = subtitle_pb2.SUBTITLE_GRANULARITY_UNSPECIFIED
        if record.granularity == "words":
            granularity = subtitle_pb2.SUBTITLE_GRANULARITY_WORDS
        elif record.granularity == "sentences":
            granularity = subtitle_pb2.SUBTITLE_GRANULARITY_SENTENCES
        output_format = subtitle_pb2.SUBTITLE_FORMAT_UNSPECIFIED
        if record.output_format == "srt":
            output_format = subtitle_pb2.SUBTITLE_FORMAT_SRT
        response = subtitle_pb2.GetRenderSubtitlesJobResponse(
            job_id=record.job_id,
            state=self._subtitle_job_state_value(record.state),
            error_code=record.error_code or "",
            error_message=record.error_message or "",
            language_code=record.language_code or "",
            mode=mode,
            output_format=output_format,
            granularity=granularity,
            group_size=record.group_size,
            srt_text=record.srt_text or "",
            srt_artifact_id=record.srt_artifact_id or "",
            created_at_unix_seconds=record.created_at_unix_seconds,
            started_at_unix_seconds=record.started_at_unix_seconds or 0.0,
            finished_at_unix_seconds=record.finished_at_unix_seconds or 0.0,
        )
        response.cues.extend(self._subtitle_cue(cue) for cue in record.cues)
        return response

    def RenderSubtitles(self, request, context):
        with self._request_scope(context):
            prepared = self._resolve_prepared_subtitle_job(request)
            stem = Path(prepared.audio_record.filename).stem
            srt_reservation = self.service_context.artifact_store.reserve_artifact(
                f"{stem}.grouped.srt",
                media_type="application/x-subrip",
                fallback_suffix=".srt",
            )
            from dictator.subtitles.models import RenderSubtitlesRequest

            subtitle_service = self.service_context.execution_runtime.get_subtitle_service()
            needs_whisper_model = prepared.source_text is None or prepared.language_code is None
            whisper_model = None
            if needs_whisper_model:
                whisper_model = self.service_context.execution_runtime.get_whisper_model(
                    prepared.model_size
                )
            result = subtitle_service.render(
                RenderSubtitlesRequest(
                    audio_path=prepared.audio_record.path,
                    language=prepared.language_code,
                    model_size=prepared.model_size,
                    output_format="srt",
                    granularity=prepared.granularity,
                    group_size=prepared.group_size,
                    source_text=prepared.source_text,
                    source_text_name=prepared.source_text_name,
                    output_srt_path=srt_reservation.path,
                ),
                model=whisper_model,
            )
            srt_record = self.service_context.artifact_store.finalize_artifact(srt_reservation)
            mode = subtitle_pb2.SUBTITLE_MODE_TRANSCRIPTION
            if result.mode == "forced_alignment":
                mode = subtitle_pb2.SUBTITLE_MODE_FORCED_ALIGNMENT
            response = subtitle_pb2.RenderSubtitlesResponse(
                language_code=result.language,
                mode=mode,
                output_format=subtitle_pb2.SUBTITLE_FORMAT_SRT,
                granularity=request.granularity,
                group_size=result.group_size,
                srt_artifact_id=srt_record.artifact_id,
            )
            response.cues.extend(self._subtitle_cue(cue) for cue in result.cues)
            if prepared.include_srt_text:
                response.srt_text = result.srt_text
            return response

    def SubmitRenderSubtitlesJob(self, request, context):
        with self._request_scope(context):
            if self.service_context.subtitle_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.subtitles.jobs_unavailable",
                    "subtitle job manager is not configured",
                )
            prepared = self._resolve_prepared_subtitle_job(request)
            record = self.service_context.subtitle_job_manager.submit(prepared)
            return subtitle_pb2.SubmitRenderSubtitlesJobResponse(
                job_id=record.job_id,
                state=self._subtitle_job_state_value(record.state),
            )

    def GetRenderSubtitlesJob(self, request, context):
        with self._request_scope(context):
            if self.service_context.subtitle_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.subtitles.jobs_unavailable",
                    "subtitle job manager is not configured",
                )
            if not request.job_id.strip():
                raise ValidationError(
                    "dictator.grpc.subtitles.job_id_required",
                    "job_id is required",
                )
            job_id = validate_subtitle_job_id(request.job_id)
            return self._subtitle_job_response(
                self.service_context.subtitle_job_manager.get(job_id)
            )
