"""Transcription and diarization gRPC servicer."""

from __future__ import annotations

import json
from pathlib import Path

from google.protobuf import json_format, struct_pb2

from dictator.diarization.models import DiarizeAudioRequest
from dictator.runtime import ValidationError
from dictator.runtime.jobs import (
    DiarizationJobRecord,
    DiarizationJobState,
    PreparedDiarizationJob,
    PreparedTranscriptionJob,
    TranscriptionJobRecord,
    TranscriptionJobState,
    validate_diarization_job_id,
    validate_transcription_job_id,
)
from dictator.speech.v1 import common_pb2, transcription_pb2, transcription_pb2_grpc

from .base import BaseServicer, DEFAULT_MODEL_SIZE


class TranscriptionServiceServicer(BaseServicer, transcription_pb2_grpc.TranscriptionServiceServicer):
    def _resolve_prepared_transcription_job(self, request) -> PreparedTranscriptionJob:
        audio = self.service_context.artifact_store.get_artifact(request.audio_artifact_id)
        language = self._resolve_language_request(
            language_code=request.language_code,
            autodetect_language=request.autodetect_language,
            error_scope="dictator.grpc.transcription",
        )
        return PreparedTranscriptionJob(
            audio_record=audio,
            language_code=language,
            model_size=request.model_size or DEFAULT_MODEL_SIZE,
            include_word_segments=request.include_word_segments,
        )

    def _resolve_prepared_diarization_job(self, request) -> PreparedDiarizationJob:
        audio = self.service_context.artifact_store.get_artifact(request.audio_artifact_id)
        language = self._resolve_language_request(
            language_code=request.language_code,
            autodetect_language=request.autodetect_language,
            error_scope="dictator.grpc.diarization",
        )
        include_words = request.include_words
        include_utterances = request.include_utterances
        include_speakers = request.include_speakers
        include_speaker_segments = request.include_speaker_segments
        if not any((include_words, include_utterances, include_speakers, include_speaker_segments)):
            include_words = True
            include_utterances = True
            include_speakers = True
        utterance_gap_seconds = 0.75
        if request.HasField("utterance_gap_seconds"):
            utterance_gap_seconds = request.utterance_gap_seconds
        return PreparedDiarizationJob(
            audio_record=audio,
            language_code=language,
            model_size=request.model_size or DEFAULT_MODEL_SIZE,
            include_words=include_words,
            include_utterances=include_utterances,
            include_speakers=include_speakers,
            include_speaker_segments=include_speaker_segments,
            utterance_gap_seconds=utterance_gap_seconds,
            persist_json_artifact=request.persist_json_artifact,
        )

    def _transcription_job_state_value(self, state: TranscriptionJobState) -> int:
        mapping = {
            TranscriptionJobState.QUEUED: transcription_pb2.TRANSCRIPTION_JOB_STATE_QUEUED,
            TranscriptionJobState.RUNNING: transcription_pb2.TRANSCRIPTION_JOB_STATE_RUNNING,
            TranscriptionJobState.SUCCEEDED: transcription_pb2.TRANSCRIPTION_JOB_STATE_SUCCEEDED,
            TranscriptionJobState.FAILED: transcription_pb2.TRANSCRIPTION_JOB_STATE_FAILED,
        }
        return mapping[state]

    def _diarization_job_state_value(self, state: DiarizationJobState) -> int:
        mapping = {
            DiarizationJobState.QUEUED: transcription_pb2.DIARIZATION_JOB_STATE_QUEUED,
            DiarizationJobState.RUNNING: transcription_pb2.DIARIZATION_JOB_STATE_RUNNING,
            DiarizationJobState.SUCCEEDED: transcription_pb2.DIARIZATION_JOB_STATE_SUCCEEDED,
            DiarizationJobState.FAILED: transcription_pb2.DIARIZATION_JOB_STATE_FAILED,
        }
        return mapping[state]

    def _transcription_job_response(self, record: TranscriptionJobRecord):
        response = transcription_pb2.GetTranscribeJobResponse(
            job_id=record.job_id,
            state=self._transcription_job_state_value(record.state),
            error_code=record.error_code or "",
            error_message=record.error_message or "",
            text=record.text or "",
            language_code=record.language_code or "",
            created_at_unix_seconds=record.created_at_unix_seconds,
            started_at_unix_seconds=record.started_at_unix_seconds or 0.0,
            finished_at_unix_seconds=record.finished_at_unix_seconds or 0.0,
        )
        response.words.extend(
            common_pb2.WordSegment(
                content=word.text,
                start_seconds=float(word.start_seconds or 0.0),
                end_seconds=float(word.end_seconds or 0.0),
            )
            for word in record.words
        )
        return response

    def _diarization_job_response(self, record: DiarizationJobRecord):
        response = transcription_pb2.GetDiarizeAudioJobResponse(
            job_id=record.job_id,
            state=self._diarization_job_state_value(record.state),
            error_code=record.error_code or "",
            error_message=record.error_message or "",
            text=record.text or "",
            language_code=record.language_code or "",
            diarization_artifact_id=record.diarization_artifact_id or "",
            created_at_unix_seconds=record.created_at_unix_seconds,
            started_at_unix_seconds=record.started_at_unix_seconds or 0.0,
            finished_at_unix_seconds=record.finished_at_unix_seconds or 0.0,
        )
        if record.diarization is not None:
            response.diarization.CopyFrom(
                json_format.ParseDict(record.diarization, struct_pb2.Struct())
            )
        return response

    def Transcribe(self, request, context):
        with self._request_scope(context):
            prepared = self._resolve_prepared_transcription_job(request)
            transcription_service = self.service_context.execution_runtime.get_transcription_service()
            transcription = transcription_service.transcribe(
                prepared.audio_record.path,
                language=prepared.language_code,
                model_size=prepared.model_size,
            )
            response = transcription_pb2.TranscribeResponse(
                text=transcription.text,
                language_code=transcription.language or "",
            )
            if prepared.include_word_segments:
                response.words.extend(
                    common_pb2.WordSegment(
                        content=word.text,
                        start_seconds=float(word.start_seconds or 0.0),
                        end_seconds=float(word.end_seconds or 0.0),
                    )
                    for word in transcription.words
                )
            return response

    def DiarizeAudio(self, request, context):
        with self._request_scope(context):
            prepared = self._resolve_prepared_diarization_job(request)
            diarization_service = self.service_context.execution_runtime.get_diarization_service()
            result = diarization_service.diarize(
                DiarizeAudioRequest(
                    input_path=prepared.audio_record.path,
                    language=prepared.language_code,
                    model_size=prepared.model_size,
                    include_words=prepared.include_words,
                    include_utterances=prepared.include_utterances,
                    include_speakers=prepared.include_speakers,
                    include_speaker_segments=prepared.include_speaker_segments,
                    utterance_gap_seconds=prepared.utterance_gap_seconds,
                ),
                model=self.service_context.execution_runtime.get_whisper_model(
                    prepared.model_size
                ),
                diarization_pipeline=self.service_context.execution_runtime.get_diarization_pipeline(),
            )
            payload = result.to_json_dict(
                include_words=prepared.include_words,
                include_utterances=prepared.include_utterances,
                include_speakers=prepared.include_speakers,
                include_speaker_segments=prepared.include_speaker_segments,
            )
            response = transcription_pb2.DiarizeAudioResponse(
                text=result.text,
                language_code=result.language or "",
            )
            response.diarization.CopyFrom(
                json_format.ParseDict(payload, struct_pb2.Struct())
            )
            if prepared.persist_json_artifact:
                json_record = self.service_context.artifact_store.write_artifact(
                    [json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")],
                    filename=f"{Path(prepared.audio_record.filename).stem}.diarization.json",
                    media_type="application/json",
                    fallback_suffix=".json",
                )
                response.diarization_artifact_id = json_record.artifact_id
            return response

    def SubmitTranscribeJob(self, request, context):
        with self._request_scope(context):
            if self.service_context.transcription_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.transcription.jobs_unavailable",
                    "transcription job manager is not configured",
                )
            prepared = self._resolve_prepared_transcription_job(request)
            record = self.service_context.transcription_job_manager.submit(prepared)
            return transcription_pb2.SubmitTranscribeJobResponse(
                job_id=record.job_id,
                state=self._transcription_job_state_value(record.state),
            )

    def GetTranscribeJob(self, request, context):
        with self._request_scope(context):
            if self.service_context.transcription_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.transcription.jobs_unavailable",
                    "transcription job manager is not configured",
                )
            if not request.job_id.strip():
                raise ValidationError(
                    "dictator.grpc.transcription.job_id_required",
                    "job_id is required",
                )
            job_id = validate_transcription_job_id(request.job_id)
            return self._transcription_job_response(
                self.service_context.transcription_job_manager.get(job_id)
            )

    def SubmitDiarizeAudioJob(self, request, context):
        with self._request_scope(context):
            if self.service_context.diarization_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.diarization.jobs_unavailable",
                    "diarization job manager is not configured",
                )
            prepared = self._resolve_prepared_diarization_job(request)
            record = self.service_context.diarization_job_manager.submit(prepared)
            return transcription_pb2.SubmitDiarizeAudioJobResponse(
                job_id=record.job_id,
                state=self._diarization_job_state_value(record.state),
            )

    def GetDiarizeAudioJob(self, request, context):
        with self._request_scope(context):
            if self.service_context.diarization_job_manager is None:
                raise ValidationError(
                    "dictator.grpc.diarization.jobs_unavailable",
                    "diarization job manager is not configured",
                )
            if not request.job_id.strip():
                raise ValidationError(
                    "dictator.grpc.diarization.job_id_required",
                    "job_id is required",
                )
            job_id = validate_diarization_job_id(request.job_id)
            return self._diarization_job_response(
                self.service_context.diarization_job_manager.get(job_id)
            )
