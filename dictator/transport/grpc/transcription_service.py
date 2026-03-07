"""Transcription and diarization gRPC servicer."""

from __future__ import annotations

import json
from pathlib import Path

from google.protobuf import json_format, struct_pb2

from dictator.speech.v1 import common_pb2, transcription_pb2, transcription_pb2_grpc

from .base import BaseServicer, DEFAULT_MODEL_SIZE


class TranscriptionServiceServicer(BaseServicer, transcription_pb2_grpc.TranscriptionServiceServicer):
    def Transcribe(self, request, context):
        with self._request_scope(context):
            audio = self.service_context.artifact_store.get_artifact(request.audio_artifact_id)

            model_size = request.model_size or DEFAULT_MODEL_SIZE
            language = self._resolve_language_request(
                language_code=request.language_code,
                autodetect_language=request.autodetect_language,
                error_scope="dictator.grpc.transcription",
            )
            transcription_service = self.service_context.execution_runtime.get_transcription_service()
            transcription = transcription_service.transcribe(
                audio.path,
                language=language,
                model_size=model_size,
            )
            response = transcription_pb2.TranscribeResponse(
                text=transcription.text,
                language_code=transcription.language or "",
            )
            if request.include_word_segments:
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

            from dictator.diarization.models import DiarizeAudioRequest

            diarization_service = self.service_context.execution_runtime.get_diarization_service()
            result = diarization_service.diarize(
                DiarizeAudioRequest(
                    input_path=audio.path,
                    language=language,
                    model_size=request.model_size or DEFAULT_MODEL_SIZE,
                    include_words=include_words,
                    include_utterances=include_utterances,
                    include_speakers=include_speakers,
                    include_speaker_segments=include_speaker_segments,
                    utterance_gap_seconds=utterance_gap_seconds,
                ),
                model=self.service_context.execution_runtime.get_whisper_model(
                    request.model_size or DEFAULT_MODEL_SIZE
                ),
                diarization_pipeline=self.service_context.execution_runtime.get_diarization_pipeline(),
            )
            payload = result.to_json_dict(
                include_words=include_words,
                include_utterances=include_utterances,
                include_speakers=include_speakers,
                include_speaker_segments=include_speaker_segments,
            )
            response = transcription_pb2.DiarizeAudioResponse(
                text=result.text,
                language_code=result.language or "",
            )
            response.diarization.CopyFrom(
                json_format.ParseDict(payload, struct_pb2.Struct())
            )
            if request.persist_json_artifact:
                json_record = self.service_context.artifact_store.write_artifact(
                    [json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")],
                    filename=f"{Path(audio.filename).stem}.diarization.json",
                    media_type="application/json",
                    fallback_suffix=".json",
                )
                response.diarization_artifact_id = json_record.artifact_id
            return response
