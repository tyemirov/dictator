"""Voice extraction and synthesis gRPC servicer."""

from __future__ import annotations

from pathlib import Path

from dictator.runtime import ValidationError
from dictator.runtime.jobs import SynthesisJobRecord, SynthesisJobState, validate_synthesis_job_id
from dictator.speech.v1 import voice_pb2, voice_pb2_grpc
from dictator.synthesis.models import SynthesisEngine, SynthesisRequest
from dictator.synthesis.workflow import (
    execute_synthesis_request,
    prepare_synthesis_request,
)

from .base import BaseServicer


class VoiceServiceServicer(BaseServicer, voice_pb2_grpc.VoiceServiceServicer):
    def _resolve_synthesis_engine(self, engine_value: int) -> SynthesisEngine:
        if engine_value == voice_pb2.SYNTHESIS_ENGINE_QWEN3:
            return SynthesisEngine.QWEN3
        raise ValidationError(
            "dictator.grpc.voice.synthesis_engine_required",
            "synthesis_engine must be set to QWEN3",
        )

    def _resolve_speaker_transcript_text(self, request) -> str | None:
        return request.speaker_transcript_text or None

    def _resolve_prepared_synthesis_request(self, request):
        return prepare_synthesis_request(
            self.service_context.artifact_store,
            speaker_artifact_id=request.speaker_artifact_id,
            text=request.text,
            text_artifact_id=request.text_artifact_id,
            language_code=request.language_code,
            max_duration_seconds=request.max_duration_seconds,
            include_timeline=request.include_timeline,
            engine=self._resolve_synthesis_engine(request.synthesis_engine),
            speaker_transcript_text=self._resolve_speaker_transcript_text(request),
        )

    def _job_state_value(self, state: SynthesisJobState) -> int:
        mapping = {
            SynthesisJobState.QUEUED: voice_pb2.SYNTHESIS_JOB_STATE_QUEUED,
            SynthesisJobState.RUNNING: voice_pb2.SYNTHESIS_JOB_STATE_RUNNING,
            SynthesisJobState.SUCCEEDED: voice_pb2.SYNTHESIS_JOB_STATE_SUCCEEDED,
            SynthesisJobState.FAILED: voice_pb2.SYNTHESIS_JOB_STATE_FAILED,
        }
        return mapping[state]

    def _job_response(self, record: SynthesisJobRecord):
        response = voice_pb2.GetSynthesizeSpeechJobResponse(
            job_id=record.job_id,
            state=self._job_state_value(record.state),
            error_code=record.error_code or "",
            error_message=record.error_message or "",
            audio_duration_seconds=record.audio_duration_seconds or 0.0,
            timeline_artifact_id=record.timeline_artifact_id or "",
            chunk_count=record.chunk_count or 0,
            created_at_unix_seconds=record.created_at_unix_seconds,
            started_at_unix_seconds=record.started_at_unix_seconds or 0.0,
            finished_at_unix_seconds=record.finished_at_unix_seconds or 0.0,
        )
        if record.audio_artifact_id:
            response.audio_artifact.CopyFrom(
                self._artifact_ref(self.service_context.artifact_store.get_artifact(record.audio_artifact_id))
            )
        return response

    def ExtractReferenceSample(self, request, context):
        with self._request_scope(context):
            source = self.service_context.artifact_store.get_artifact(request.source_artifact_id)
            reservation = self.service_context.artifact_store.reserve_artifact(
                f"{Path(source.filename).stem}_reference.wav",
                media_type="audio/wav",
                fallback_suffix=".wav",
            )
            from dictator.extraction.models import ReferenceExtractionRequest

            extraction_service = self.service_context.execution_runtime.get_reference_extraction_service()
            result = extraction_service.extract(
                ReferenceExtractionRequest(
                    input_path=source.path,
                    output_path=reservation.path,
                    model_size=request.model_size or "medium",
                    language=request.language_code or None,
                    duration_seconds=request.duration_seconds or 20.0,
                    max_speech_rate=request.max_speech_rate or 4.0,
                    min_centroid_hz=request.min_centroid_hz or 500.0,
                    max_centroid_hz=request.max_centroid_hz or 4000.0,
                ),
                model=self.service_context.execution_runtime.get_whisper_model(request.model_size or "medium"),
                diarization_pipeline=self.service_context.execution_runtime.get_diarization_pipeline(),
            )
            sample_record = self.service_context.artifact_store.finalize_artifact(reservation)
            return voice_pb2.ExtractReferenceSampleResponse(
                sample_artifact=self._artifact_ref(sample_record),
                trim_start_seconds=result.trim_start_seconds,
                trim_end_seconds=result.trim_end_seconds,
                window_start_seconds=result.window_start_seconds,
                window_end_seconds=result.window_end_seconds,
                dominant_speaker_word_count=len(result.dominant_speaker_words),
            )

    def SynthesizeSpeech(self, request, context):
        with self._request_scope(context):
            prepared = self._resolve_prepared_synthesis_request(request)
            outcome = execute_synthesis_request(
                artifact_store=self.service_context.artifact_store,
                execution_runtime=self.service_context.execution_runtime,
                prepared=prepared,
            )
            response = voice_pb2.SynthesizeSpeechResponse(
                audio_artifact=self._artifact_ref(outcome.audio_record),
                audio_duration_seconds=outcome.audio_duration_seconds,
                chunk_count=outcome.chunk_count,
            )
            if request.include_timeline:
                response.timeline.extend(
                    self._timeline_segment(segment)
                    for segment in outcome.timeline_segments
                )
                response.timeline_artifact_id = outcome.timeline_artifact_id or ""
            return response

    def SubmitSynthesizeSpeechJob(self, request, context):
        with self._request_scope(context):
            if self.service_context.synthesis_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.voice.jobs_unavailable",
                    "synthesis job manager is not configured",
                )
            prepared = self._resolve_prepared_synthesis_request(request)
            record = self.service_context.synthesis_job_manager.submit(prepared)
            return voice_pb2.SubmitSynthesizeSpeechJobResponse(
                job_id=record.job_id,
                state=self._job_state_value(record.state),
            )

    def GetSynthesizeSpeechJob(self, request, context):
        with self._request_scope(context):
            if self.service_context.synthesis_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.voice.jobs_unavailable",
                    "synthesis job manager is not configured",
                )
            if not request.job_id.strip():
                raise ValidationError(
                    "dictator.grpc.voice.job_id_required",
                    "job_id is required",
                )
            job_id = validate_synthesis_job_id(request.job_id)
            return self._job_response(self.service_context.synthesis_job_manager.get(job_id))
