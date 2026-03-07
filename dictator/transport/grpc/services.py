"""gRPC servicers over the packaged Dictator services."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterator

import grpc

from dictator.runtime import (
    DependencyError,
    InflightLimiter,
    MetricsRegistry,
    ProcessingError,
    ServiceRequestError,
    SpeechExecutionRuntime,
    ValidationError,
)
from dictator.storage import ArtifactRecord, LocalArtifactStore
from dictator.speech.v1 import (
    alignment_pb2,
    alignment_pb2_grpc,
    artifacts_pb2,
    artifacts_pb2_grpc,
    common_pb2,
    runtime_pb2,
    runtime_pb2_grpc,
    transcription_pb2,
    transcription_pb2_grpc,
    voice_pb2,
    voice_pb2_grpc,
)

_ERROR_CODE_METADATA = "x-dictator-error-code"
_AUTH_HEADER = "authorization"
_TOKEN_HEADER = "x-dictator-token"
_DEFAULT_MODEL_SIZE = "base"


@dataclass(frozen=True)
class ServiceContext:
    artifact_store: LocalArtifactStore
    execution_runtime: SpeechExecutionRuntime
    metrics: MetricsRegistry
    limiter: InflightLimiter
    auth_token: str | None
    download_chunk_bytes: int


class BaseServicer:
    def __init__(self, service_context: ServiceContext) -> None:
        self.service_context = service_context

    def _require_auth(self, context: grpc.ServicerContext) -> None:
        expected = self.service_context.auth_token
        if not expected:
            return
        metadata = {key.lower(): value for key, value in context.invocation_metadata()}
        presented = metadata.get(_TOKEN_HEADER)
        authorization = metadata.get(_AUTH_HEADER, "")
        if not presented and authorization.lower().startswith("bearer "):
            presented = authorization[7:]
        if presented != expected:
            self._abort(
                context,
                grpc.StatusCode.UNAUTHENTICATED,
                "dictator.grpc.auth.required",
                "missing or invalid auth token",
            )

    def _abort(
        self,
        context: grpc.ServicerContext,
        status: grpc.StatusCode,
        code: str,
        message: str,
    ) -> None:
        context.set_trailing_metadata(((_ERROR_CODE_METADATA, code),))
        context.abort(status, message)

    @contextmanager
    def _request_scope(
        self,
        context: grpc.ServicerContext,
        bytes_received: int = 0,
    ) -> Iterator[None]:
        started_at = time.monotonic()
        success = False
        self.service_context.metrics.record_start()
        if bytes_received:
            self.service_context.metrics.record_bytes(bytes_received)
        try:
            with self.service_context.limiter.acquire():
                try:
                    self._require_auth(context)
                    yield
                    success = True
                except ValidationError as exc:
                    self._abort(context, grpc.StatusCode.INVALID_ARGUMENT, exc.code, str(exc))
                except DependencyError as exc:
                    self._abort(context, grpc.StatusCode.FAILED_PRECONDITION, exc.code, str(exc))
                except ProcessingError as exc:
                    self._abort(context, grpc.StatusCode.INTERNAL, exc.code, str(exc))
                except FileNotFoundError as exc:
                    self._abort(
                        context,
                        grpc.StatusCode.NOT_FOUND,
                        "dictator.artifact.not_found",
                        str(exc),
                    )
                except ValueError as exc:
                    self._abort(
                        context,
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "dictator.input.invalid",
                        str(exc),
                    )
        except ServiceRequestError as exc:
            self._abort(context, grpc.StatusCode.RESOURCE_EXHAUSTED, exc.code, str(exc))
        finally:
            self.service_context.metrics.record_finish(success, time.monotonic() - started_at)

    def _artifact_ref(self, record: ArtifactRecord) -> common_pb2.ArtifactRef:
        return common_pb2.ArtifactRef(
            artifact_id=record.artifact_id,
            filename=record.filename,
            media_type=record.media_type,
            size_bytes=record.size_bytes,
            sha256=record.sha256,
        )

    def _word_segment(self, payload: dict[str, object]) -> common_pb2.WordSegment:
        return common_pb2.WordSegment(
            content=str(payload.get("content", "")),
            start_seconds=float(payload.get("start") or 0.0),
            end_seconds=float(payload.get("end") or 0.0),
        )

    def _timeline_segment(self, payload: dict[str, object]) -> common_pb2.TimelineSegment:
        return common_pb2.TimelineSegment(
            content=str(payload.get("content", "")),
            start_seconds=float(payload.get("start") or 0.0),
            end_seconds=float(payload.get("end") or 0.0),
        )


class ArtifactServiceServicer(BaseServicer, artifacts_pb2_grpc.ArtifactServiceServicer):
    def UploadArtifact(self, request_iterator, context):
        with self._request_scope(context):
            payload_size = 0
            iterator = iter(request_iterator)
            try:
                first_chunk = next(iterator)
            except StopIteration as exc:
                raise ValidationError(
                    "dictator.grpc.artifact.empty_upload",
                    "upload stream is empty",
                ) from exc
            if first_chunk.WhichOneof("payload") != "metadata":
                raise ValidationError(
                    "dictator.grpc.artifact.missing_metadata",
                    "first upload chunk must contain metadata",
                )
            metadata_message = first_chunk.metadata
            chunks: list[bytes] = []
            for chunk in iterator:
                payload_type = chunk.WhichOneof("payload")
                if payload_type != "content":
                    raise ValidationError(
                        "dictator.grpc.artifact.invalid_chunk",
                        "upload content chunks cannot contain metadata",
                    )
                chunks.append(chunk.content)
                payload_size += len(chunk.content)
            if payload_size:
                self.service_context.metrics.record_bytes(payload_size)
            record = self.service_context.artifact_store.write_artifact(
                chunks,
                filename=metadata_message.filename,
                media_type=metadata_message.media_type,
            )
            return artifacts_pb2.UploadArtifactResponse(artifact=self._artifact_ref(record))

    def DownloadArtifact(self, request, context):
        chunk_size = request.chunk_size or self.service_context.download_chunk_bytes
        with self._request_scope(context):
            for record, offset, payload, eof in self.service_context.artifact_store.iter_artifact_chunks(
                request.artifact_id,
                chunk_size=chunk_size,
            ):
                yield artifacts_pb2.DownloadArtifactChunk(
                    artifact=self._artifact_ref(record),
                    content=payload,
                    offset=offset,
                    eof=eof,
                )


class TranscriptionServiceServicer(BaseServicer, transcription_pb2_grpc.TranscriptionServiceServicer):
    def Transcribe(self, request, context):
        with self._request_scope(context):
            audio = self.service_context.artifact_store.get_artifact(request.audio_artifact_id)
            from dictator.transcription.service import transcribe_word_segments

            model_size = request.model_size or _DEFAULT_MODEL_SIZE
            model = self.service_context.execution_runtime.get_whisper_model(model_size)
            words = transcribe_word_segments(
                audio.path,
                language=request.language_code or None,
                model=model,
            )
            text = " ".join(word.text for word in words if word.text)
            response = transcription_pb2.TranscribeResponse(
                text=text,
                language_code=request.language_code,
            )
            if request.include_word_segments:
                response.words.extend(
                    common_pb2.WordSegment(
                        content=word.text,
                        start_seconds=float(word.start_seconds or 0.0),
                        end_seconds=float(word.end_seconds or 0.0),
                    )
                    for word in words
                )
            return response


class AlignmentServiceServicer(BaseServicer, alignment_pb2_grpc.AlignmentServiceServicer):
    def AlignTranscript(self, request, context):
        with self._request_scope(context):
            audio = self.service_context.artifact_store.get_artifact(request.audio_artifact_id)
            transcript_text = request.transcript_text
            if request.transcript_artifact_id:
                transcript_text = self.service_context.artifact_store.read_text(request.transcript_artifact_id)
            if not transcript_text.strip():
                raise ValidationError(
                    "dictator.grpc.alignment.missing_transcript",
                    "transcript_text or transcript_artifact_id is required",
                )
            service = self.service_context.execution_runtime.get_alignment_service()
            stem = Path(audio.filename).stem
            srt_reservation = self.service_context.artifact_store.reserve_artifact(
                f"{stem}.srt",
                media_type="application/x-subrip",
                fallback_suffix=".srt",
            )
            from dictator.alignment.models import AlignTranscriptRequest

            result = service.align(
                AlignTranscriptRequest(
                    audio_path=audio.path,
                    transcript_text=transcript_text,
                    language=request.language_code,
                    remove_punctuation=request.remove_punctuation,
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
            if request.include_srt_text:
                response.srt_text = result.srt_text
            return response


class VoiceServiceServicer(BaseServicer, voice_pb2_grpc.VoiceServiceServicer):
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
                    duration_seconds=request.duration_seconds or 20.0,
                    language=request.language_code or None,
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
            speaker = self.service_context.artifact_store.get_artifact(request.speaker_artifact_id)
            text = request.text
            if request.text_artifact_id:
                text = self.service_context.artifact_store.read_text(request.text_artifact_id)
            if not text.strip():
                raise ValidationError(
                    "dictator.grpc.voice.missing_text",
                    "text or text_artifact_id is required",
                )
            from dictator.audio.ffmpeg_ops import concat_normalise
            from dictator.synthesis.service import cleanup_synthesis_result
            from dictator.synthesis.text import build_chunks, clean

            synthesis_service = self.service_context.execution_runtime.get_synthesis_service()
            cap_seconds = request.max_duration_seconds or None
            result = None
            try:
                result = synthesis_service.synthesise(
                    speaker_wav=speaker.path,
                    chunks=build_chunks(clean(text)),
                    cap_seconds=cap_seconds,
                    language_code=request.language_code or "en",
                )
                audio_reservation = self.service_context.artifact_store.reserve_artifact(
                    f"{Path(speaker.filename).stem}_synth.wav",
                    media_type="audio/wav",
                    fallback_suffix=".wav",
                )
                concat_normalise(result.wav_paths, audio_reservation.path, cap_seconds)
                audio_record = self.service_context.artifact_store.finalize_artifact(audio_reservation)
                response = voice_pb2.SynthesizeSpeechResponse(
                    audio_artifact=self._artifact_ref(audio_record),
                    audio_duration_seconds=result.segments[-1].end_seconds if result.segments else 0.0,
                    chunk_count=len(result.wav_paths),
                )
                if request.include_timeline:
                    timeline_payload = {
                        "textSegments": [segment.to_legacy_dict() for segment in result.segments],
                        "imageCues": [],
                        "voices": [
                            {
                                "id": speaker.artifact_id,
                                "label": Path(speaker.filename).stem,
                                "file": str(speaker.path),
                            }
                        ],
                    }
                    timeline_record = self.service_context.artifact_store.write_artifact(
                        [json.dumps(timeline_payload, ensure_ascii=False, indent=2).encode("utf-8")],
                        filename=f"{Path(audio_record.filename).stem}.timeline.json",
                        media_type="application/json",
                        fallback_suffix=".json",
                    )
                    response.timeline.extend(
                        self._timeline_segment(segment.to_legacy_dict())
                        for segment in result.segments
                    )
                    response.timeline_artifact_id = timeline_record.artifact_id
                return response
            finally:
                if result is not None:
                    cleanup_synthesis_result(result)


class RuntimeServiceServicer(BaseServicer, runtime_pb2_grpc.RuntimeServiceServicer):
    def GetMetrics(self, request, context):
        with self._request_scope(context):
            snapshot = self.service_context.metrics.snapshot()
            return runtime_pb2.GetMetricsResponse(
                requests_total=snapshot.requests_total,
                requests_succeeded=snapshot.requests_succeeded,
                requests_failed=snapshot.requests_failed,
                inflight=snapshot.inflight,
                bytes_received=snapshot.bytes_received,
                uptime_seconds=snapshot.uptime_seconds,
                average_latency_seconds=snapshot.average_latency_seconds,
                max_latency_seconds=snapshot.max_latency_seconds,
            )


def register_services(server: grpc.Server, service_context: ServiceContext) -> None:
    artifacts_pb2_grpc.add_ArtifactServiceServicer_to_server(
        ArtifactServiceServicer(service_context),
        server,
    )
    transcription_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServiceServicer(service_context),
        server,
    )
    alignment_pb2_grpc.add_AlignmentServiceServicer_to_server(
        AlignmentServiceServicer(service_context),
        server,
    )
    voice_pb2_grpc.add_VoiceServiceServicer_to_server(
        VoiceServiceServicer(service_context),
        server,
    )
    runtime_pb2_grpc.add_RuntimeServiceServicer_to_server(
        RuntimeServiceServicer(service_context),
        server,
    )
