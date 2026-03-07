"""Grouped subtitle gRPC servicer."""

from __future__ import annotations

from pathlib import Path

from dictator.runtime import ValidationError
from dictator.speech.v1 import subtitle_pb2, subtitle_pb2_grpc

from .base import BaseServicer, DEFAULT_MODEL_SIZE


class SubtitleServiceServicer(BaseServicer, subtitle_pb2_grpc.SubtitleServiceServicer):
    def RenderSubtitles(self, request, context):
        with self._request_scope(context):
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

            stem = Path(audio.filename).stem
            srt_reservation = self.service_context.artifact_store.reserve_artifact(
                f"{stem}.grouped.srt",
                media_type="application/x-subrip",
                fallback_suffix=".srt",
            )
            from dictator.subtitles.models import RenderSubtitlesRequest

            subtitle_service = self.service_context.execution_runtime.get_subtitle_service()
            needs_whisper_model = source_text is None or language is None
            whisper_model = None
            if needs_whisper_model:
                whisper_model = self.service_context.execution_runtime.get_whisper_model(
                    request.model_size or DEFAULT_MODEL_SIZE
                )
            result = subtitle_service.render(
                RenderSubtitlesRequest(
                    audio_path=audio.path,
                    language=language,
                    model_size=request.model_size or DEFAULT_MODEL_SIZE,
                    output_format="srt",
                    granularity=granularity,
                    group_size=request.group_size,
                    source_text=source_text,
                    source_text_name=source_text_name,
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
            if request.include_srt_text:
                response.srt_text = result.srt_text
            return response
