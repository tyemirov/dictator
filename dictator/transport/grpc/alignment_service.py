"""Forced-alignment gRPC servicer."""

from __future__ import annotations

from pathlib import Path

from dictator.runtime import ValidationError
from dictator.speech.v1 import alignment_pb2, alignment_pb2_grpc, common_pb2

from .base import BaseServicer


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
