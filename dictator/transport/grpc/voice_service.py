"""Voice extraction and synthesis gRPC servicer."""

from __future__ import annotations

from pathlib import Path

from dictator.runtime import ValidationError
from dictator.runtime.jobs import SynthesisJobRecord, SynthesisJobState, validate_synthesis_job_id
from dictator.runtime.jobs import (
    ExtractReferenceSampleJobRecord,
    ExtractReferenceSampleJobState,
    PreparedExtractReferenceSampleJob,
    validate_extract_reference_sample_job_id,
)
from dictator.speech.v1 import common_pb2, voice_pb2, voice_pb2_grpc
from dictator.storage import ArtifactAudioMetadata
from dictator.synthesis.models import (
    DEFAULT_SYNTHESIS_AUDIO_FORMAT,
    SILERO_RU_NATIVE_SAMPLE_RATES,
    SILERO_RU_SUPPORTED_SPEAKERS,
    SILERO_RU_SYNTHESIS_AUDIO_FORMAT,
    SynthesisAudioFormat,
    SynthesisEngine,
    SynthesisTextFormat,
)
from dictator.synthesis.config import DEFAULT_SILERO_RU_DEFAULT_SPEAKER
from dictator.synthesis.workflow import (
    execute_synthesis_request,
    prepare_synthesis_request,
)

from .base import BaseServicer


SILERO_RU_VOICE_DISPLAY_NAMES = {
    "baya": "Baya",
    "xenia": "Xenia",
}


class VoiceServiceServicer(BaseServicer, voice_pb2_grpc.VoiceServiceServicer):
    def _synthesis_audio_format_pb(self, audio_format: SynthesisAudioFormat) -> common_pb2.AudioFormat:
        container = {
            "wav": common_pb2.AUDIO_CONTAINER_WAV,
        }.get(audio_format.container, common_pb2.AUDIO_CONTAINER_UNSPECIFIED)
        codec = {
            "pcm_s16le": common_pb2.AUDIO_CODEC_PCM_S16LE,
        }.get(audio_format.codec, common_pb2.AUDIO_CODEC_UNSPECIFIED)
        return common_pb2.AudioFormat(
            container=container,
            codec=codec,
            sample_rate_hz=audio_format.sample_rate_hz,
            channel_count=audio_format.channel_count,
            bit_depth=audio_format.bit_depth,
        )

    def _resolve_synthesis_audio_format(self, request, engine: SynthesisEngine) -> SynthesisAudioFormat:
        default_format = (
            SILERO_RU_SYNTHESIS_AUDIO_FORMAT
            if engine == SynthesisEngine.SILERO_RU
            else DEFAULT_SYNTHESIS_AUDIO_FORMAT
        )
        if not request.HasField("audio_format"):
            return default_format

        requested = request.audio_format
        resolved_container = requested.container or common_pb2.AUDIO_CONTAINER_WAV
        resolved_codec = requested.codec or common_pb2.AUDIO_CODEC_PCM_S16LE
        resolved_sample_rate_hz = requested.sample_rate_hz or default_format.sample_rate_hz
        resolved_channel_count = requested.channel_count or default_format.channel_count
        resolved_bit_depth = requested.bit_depth or default_format.bit_depth

        if (
            resolved_container != common_pb2.AUDIO_CONTAINER_WAV
            or resolved_codec != common_pb2.AUDIO_CODEC_PCM_S16LE
            or resolved_channel_count != default_format.channel_count
            or resolved_bit_depth != default_format.bit_depth
        ):
            raise ValidationError(
                "dictator.grpc.voice.unsupported_audio_format",
                "unsupported synthesis audio_format; supported container/codec/channels/bit depth is WAV pcm_s16le mono 16-bit",
            )
        if resolved_sample_rate_hz <= 0:
            raise ValidationError(
                "dictator.grpc.voice.unsupported_audio_format",
                "unsupported synthesis sample_rate_hz; value must be a positive integer",
            )
        return SynthesisAudioFormat(
            container="wav",
            codec="pcm_s16le",
            sample_rate_hz=resolved_sample_rate_hz,
            channel_count=resolved_channel_count,
            bit_depth=resolved_bit_depth,
        )

    def _resolve_synthesis_engine(
        self,
        engine_value: int,
        language_code: str = "",
        *,
        speaker_artifact_id: str = "",
        speaker_transcript_text: str = "",
    ) -> SynthesisEngine:
        if engine_value == voice_pb2.SYNTHESIS_ENGINE_QWEN3:
            return SynthesisEngine.QWEN3
        if engine_value == voice_pb2.SYNTHESIS_ENGINE_SILERO_RU:
            return SynthesisEngine.SILERO_RU
        if engine_value == voice_pb2.SYNTHESIS_ENGINE_UNSPECIFIED:
            if speaker_artifact_id.strip() or speaker_transcript_text.strip():
                return SynthesisEngine.QWEN3
            base_language = (language_code or "").strip().lower().replace("_", "-").split("-", 1)[0]
            return SynthesisEngine.SILERO_RU if base_language == "ru" else SynthesisEngine.QWEN3
        raise ValidationError(
            "dictator.grpc.voice.synthesis_engine_required",
            "synthesis_engine must be QWEN3, SILERO_RU, or unspecified for language-based defaulting",
        )

    def _resolve_synthesis_text_format(self, request) -> SynthesisTextFormat:
        if request.text_format == voice_pb2.SYNTHESIS_TEXT_FORMAT_UNSPECIFIED:
            return SynthesisTextFormat.AUTO
        if request.text_format == voice_pb2.SYNTHESIS_TEXT_FORMAT_PLAIN_TEXT:
            return SynthesisTextFormat.PLAIN_TEXT
        if request.text_format == voice_pb2.SYNTHESIS_TEXT_FORMAT_SSML:
            return SynthesisTextFormat.SSML
        raise ValidationError(
            "dictator.grpc.voice.text_format_unsupported",
            "text_format must be PLAIN_TEXT, SSML, or unspecified for auto-detection",
        )

    def _resolve_speaker_transcript_text(self, request) -> str | None:
        return request.speaker_transcript_text or None

    def _voice_discovery_default_speaker(self) -> str:
        synthesis_config = getattr(self.service_context.execution_runtime, "_synthesis_config", None)
        configured = getattr(synthesis_config, "silero_ru_default_speaker", "") if synthesis_config is not None else ""
        return (configured or DEFAULT_SILERO_RU_DEFAULT_SPEAKER).strip().lower()

    def _matches_voice_discovery_filters(self, request, language_code: str) -> bool:
        if request.synthesis_engine == voice_pb2.SYNTHESIS_ENGINE_QWEN3:
            return False
        if request.synthesis_engine not in (
            voice_pb2.SYNTHESIS_ENGINE_UNSPECIFIED,
            voice_pb2.SYNTHESIS_ENGINE_QWEN3,
            voice_pb2.SYNTHESIS_ENGINE_SILERO_RU,
        ):
            raise ValidationError(
                "dictator.grpc.voice.synthesis_engine_required",
                "synthesis_engine must be QWEN3, SILERO_RU, or unspecified for voice discovery",
            )
        base_language = (request.language_code or "").strip().lower().replace("_", "-").split("-", 1)[0]
        return not base_language or base_language == language_code

    def _resolve_prepared_synthesis_request(self, request):
        speaker_transcript_text = self._resolve_speaker_transcript_text(request)
        engine = self._resolve_synthesis_engine(
            request.synthesis_engine,
            request.language_code,
            speaker_artifact_id=request.speaker_artifact_id,
            speaker_transcript_text=speaker_transcript_text or "",
        )
        return prepare_synthesis_request(
            self.service_context.artifact_store,
            speaker_artifact_id=request.speaker_artifact_id,
            text=request.text,
            text_artifact_id=request.text_artifact_id,
            language_code=request.language_code,
            max_duration_seconds=request.max_duration_seconds,
            include_timeline=request.include_timeline,
            engine=engine,
            speaker_transcript_text=speaker_transcript_text,
            preset_speaker=request.preset_speaker or None,
            audio_format=self._resolve_synthesis_audio_format(request, engine),
            text_format=self._resolve_synthesis_text_format(request),
        )

    def ListSynthesisVoices(self, request, context):
        with self._request_scope(context, is_inquiry=True):
            if not self._matches_voice_discovery_filters(request, "ru"):
                return voice_pb2.ListSynthesisVoicesResponse()
            default_speaker = self._voice_discovery_default_speaker()
            return voice_pb2.ListSynthesisVoicesResponse(
                voices=[
                    voice_pb2.SynthesisVoice(
                        synthesis_engine=voice_pb2.SYNTHESIS_ENGINE_SILERO_RU,
                        language_code="ru",
                        voice_id=speaker,
                        display_name=SILERO_RU_VOICE_DISPLAY_NAMES.get(speaker, speaker),
                        is_default=speaker == default_speaker,
                        native_sample_rate_hz=SILERO_RU_NATIVE_SAMPLE_RATES,
                        default_sample_rate_hz=SILERO_RU_SYNTHESIS_AUDIO_FORMAT.sample_rate_hz,
                        requires_reference_audio=False,
                    )
                    for speaker in SILERO_RU_SUPPORTED_SPEAKERS
                ],
            )

    def _job_state_value(self, state: SynthesisJobState) -> int:
        mapping = {
            SynthesisJobState.QUEUED: voice_pb2.SYNTHESIS_JOB_STATE_QUEUED,
            SynthesisJobState.RUNNING: voice_pb2.SYNTHESIS_JOB_STATE_RUNNING,
            SynthesisJobState.SUCCEEDED: voice_pb2.SYNTHESIS_JOB_STATE_SUCCEEDED,
            SynthesisJobState.FAILED: voice_pb2.SYNTHESIS_JOB_STATE_FAILED,
            SynthesisJobState.CANCELED: voice_pb2.SYNTHESIS_JOB_STATE_CANCELED,
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
            estimated_total_chunks=record.estimated_total_chunks or 0,
            completed_chunks=record.completed_chunks or 0,
        )
        if record.audio_artifact_id:
            response.audio_artifact.CopyFrom(
                self._artifact_ref(self.service_context.artifact_store.get_artifact(record.audio_artifact_id))
            )
        audio_format = record.audio_format or DEFAULT_SYNTHESIS_AUDIO_FORMAT
        response.resolved_audio_format.CopyFrom(self._synthesis_audio_format_pb(audio_format))
        return response

    def _resolve_prepared_reference_extraction_job(self, request) -> PreparedExtractReferenceSampleJob:
        source = self.service_context.artifact_store.get_artifact(request.source_artifact_id)
        return PreparedExtractReferenceSampleJob(
            source_record=source,
            model_size=request.model_size or "medium",
            language_code=request.language_code or None,
            duration_seconds=request.duration_seconds or 20.0,
            max_speech_rate=request.max_speech_rate or 4.0,
            min_centroid_hz=request.min_centroid_hz or 500.0,
            max_centroid_hz=request.max_centroid_hz or 4000.0,
        )

    def _reference_extraction_job_state_value(self, state: ExtractReferenceSampleJobState) -> int:
        mapping = {
            ExtractReferenceSampleJobState.QUEUED: voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_QUEUED,
            ExtractReferenceSampleJobState.RUNNING: voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_RUNNING,
            ExtractReferenceSampleJobState.SUCCEEDED: voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_SUCCEEDED,
            ExtractReferenceSampleJobState.FAILED: voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_FAILED,
            ExtractReferenceSampleJobState.CANCELED: voice_pb2.EXTRACT_REFERENCE_SAMPLE_JOB_STATE_CANCELED,
        }
        return mapping[state]

    def _reference_extraction_job_response(self, record: ExtractReferenceSampleJobRecord):
        response = voice_pb2.GetExtractReferenceSampleJobResponse(
            job_id=record.job_id,
            state=self._reference_extraction_job_state_value(record.state),
            error_code=record.error_code or "",
            error_message=record.error_message or "",
            trim_start_seconds=record.trim_start_seconds or 0.0,
            trim_end_seconds=record.trim_end_seconds or 0.0,
            window_start_seconds=record.window_start_seconds or 0.0,
            window_end_seconds=record.window_end_seconds or 0.0,
            dominant_speaker_word_count=record.dominant_speaker_word_count or 0,
            created_at_unix_seconds=record.created_at_unix_seconds,
            started_at_unix_seconds=record.started_at_unix_seconds or 0.0,
            finished_at_unix_seconds=record.finished_at_unix_seconds or 0.0,
            source_artifact_id=record.source_artifact_id,
        )
        if record.sample_artifact_id:
            response.sample_artifact.CopyFrom(
                self._artifact_ref(self.service_context.artifact_store.get_artifact(record.sample_artifact_id))
            )
        return response

    def ExtractReferenceSample(self, request, context):
        with self._request_scope(context):
            prepared = self._resolve_prepared_reference_extraction_job(request)
            reservation = self.service_context.artifact_store.reserve_artifact(
                f"{Path(prepared.source_record.filename).stem}_reference.wav",
                media_type="audio/wav",
                fallback_suffix=".wav",
            )
            from dictator.extraction.models import ReferenceExtractionRequest

            extraction_service = self.service_context.execution_runtime.get_reference_extraction_service()
            result = extraction_service.extract(
                ReferenceExtractionRequest(
                    input_path=prepared.source_record.path,
                    output_path=reservation.path,
                    model_size=prepared.model_size,
                    language=prepared.language_code,
                    duration_seconds=prepared.duration_seconds,
                    max_speech_rate=prepared.max_speech_rate,
                    min_centroid_hz=prepared.min_centroid_hz,
                    max_centroid_hz=prepared.max_centroid_hz,
                ),
                model=self.service_context.execution_runtime.get_whisper_model(prepared.model_size),
                diarization_pipeline=self.service_context.execution_runtime.get_diarization_pipeline(),
            )
            sample_record = self.service_context.artifact_store.finalize_artifact(
                reservation,
                audio_metadata=ArtifactAudioMetadata(
                    container="wav",
                    codec="pcm_s16le",
                    sample_rate_hz=24_000,
                    channel_count=1,
                    bit_depth=16,
                    duration_seconds=max(0.0, result.trim_end_seconds - result.trim_start_seconds),
                ),
            )
            return voice_pb2.ExtractReferenceSampleResponse(
                sample_artifact=self._artifact_ref(sample_record),
                trim_start_seconds=result.trim_start_seconds,
                trim_end_seconds=result.trim_end_seconds,
                window_start_seconds=result.window_start_seconds,
                window_end_seconds=result.window_end_seconds,
                dominant_speaker_word_count=len(result.dominant_speaker_words),
            )

    def SubmitExtractReferenceSampleJob(self, request, context):
        with self._request_scope(context):
            if self.service_context.reference_extraction_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.voice.reference_jobs_unavailable",
                    "reference extraction job manager is not configured",
                )
            prepared = self._resolve_prepared_reference_extraction_job(request)
            record = self.service_context.reference_extraction_job_manager.submit(prepared)
            return voice_pb2.SubmitExtractReferenceSampleJobResponse(
                job_id=record.job_id,
                state=self._reference_extraction_job_state_value(record.state),
            )

    def GetExtractReferenceSampleJob(self, request, context):
        with self._request_scope(context, is_inquiry=True):
            if self.service_context.reference_extraction_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.voice.reference_jobs_unavailable",
                    "reference extraction job manager is not configured",
                )
            if not request.job_id.strip():
                raise ValidationError(
                    "dictator.grpc.voice.reference_job_id_required",
                    "job_id is required",
                )
            job_id = validate_extract_reference_sample_job_id(request.job_id)
            return self._reference_extraction_job_response(
                self.service_context.reference_extraction_job_manager.get(job_id)
            )

    def CancelExtractReferenceSampleJob(self, request, context):
        with self._request_scope(context, is_inquiry=True):
            if self.service_context.reference_extraction_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.voice.reference_jobs_unavailable",
                    "reference extraction job manager is not configured",
                )
            if not request.job_id.strip():
                raise ValidationError(
                    "dictator.grpc.voice.reference_job_id_required",
                    "job_id is required",
                )
            job_id = validate_extract_reference_sample_job_id(request.job_id)
            record = self.service_context.reference_extraction_job_manager.cancel(job_id)
            return voice_pb2.CancelExtractReferenceSampleJobResponse(
                job_id=record.job_id,
                state=self._reference_extraction_job_state_value(record.state),
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
                resolved_audio_format=self._synthesis_audio_format_pb(outcome.audio_format),
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
        with self._request_scope(context, is_inquiry=True):
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

    def CancelSynthesizeSpeechJob(self, request, context):
        with self._request_scope(context, is_inquiry=True):
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
            record = self.service_context.synthesis_job_manager.cancel(job_id)
            return voice_pb2.CancelSynthesizeSpeechJobResponse(
                job_id=record.job_id,
                state=self._job_state_value(record.state),
            )
