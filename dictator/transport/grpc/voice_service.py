"""Voice extraction and synthesis gRPC servicer."""

from __future__ import annotations

import json
from pathlib import Path

from dictator.runtime import ValidationError
from dictator.speech.v1 import voice_pb2, voice_pb2_grpc
from dictator.synthesis.models import SynthesisEngine, SynthesisRequest

from .base import BaseServicer


class VoiceServiceServicer(BaseServicer, voice_pb2_grpc.VoiceServiceServicer):
    def _resolve_synthesis_engine(self, engine_value: int) -> SynthesisEngine:
        if engine_value == voice_pb2.SYNTHESIS_ENGINE_XTTS:
            return SynthesisEngine.XTTS
        if engine_value == voice_pb2.SYNTHESIS_ENGINE_QWEN3:
            return SynthesisEngine.QWEN3
        raise ValidationError(
            "dictator.grpc.voice.synthesis_engine_required",
            "synthesis_engine must be set to XTTS or QWEN3",
        )

    def _resolve_speaker_transcript_text(self, request) -> str | None:
        if request.speaker_transcript_artifact_id:
            return self.service_context.artifact_store.read_text(request.speaker_transcript_artifact_id)
        if request.speaker_transcript_text:
            return request.speaker_transcript_text
        return None

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

            synthesis_engine = self._resolve_synthesis_engine(request.synthesis_engine)
            synthesis_service = self.service_context.execution_runtime.get_synthesis_service()
            cap_seconds = request.max_duration_seconds or None
            speaker_transcript_text = self._resolve_speaker_transcript_text(request)
            result = None
            try:
                result = synthesis_service.synthesise_text(
                    SynthesisRequest(
                        engine=synthesis_engine,
                        speaker_wav=speaker.path,
                        text=text,
                        language_code=request.language_code or "en",
                        cap_seconds=cap_seconds,
                        speaker_artifact_id=request.speaker_artifact_id,
                        speaker_transcript_text=speaker_transcript_text,
                    )
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
                                "engine": synthesis_engine.value,
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
