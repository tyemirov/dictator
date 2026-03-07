"""Long-lived model/runtime registry for speech execution services."""

from __future__ import annotations

import threading


class SpeechExecutionRuntime:
    """Caches heavyweight models and adapters across requests."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._whisper_models: dict[str, object] = {}
        self._transcription_service: object | None = None
        self._diarization_pipeline: object | None = None
        self._tts_backend: object | None = None
        self._alignment_backend: object | None = None

    def get_whisper_model(self, model_size: str) -> object:
        with self._lock:
            cached = self._whisper_models.get(model_size)
            if cached is not None:
                return cached
        from dictator.transcription.service import load_whisper_model

        loaded = load_whisper_model(model_size)
        with self._lock:
            self._whisper_models.setdefault(model_size, loaded)
            return self._whisper_models[model_size]


    def get_transcription_service(self):
        from dictator.transcription.service import TranscriptionService

        with self._lock:
            if self._transcription_service is None:
                self._transcription_service = TranscriptionService(
                    model_loader=self.get_whisper_model,
                )
            return self._transcription_service

    def get_diarization_pipeline(self) -> object:
        with self._lock:
            if self._diarization_pipeline is not None:
                return self._diarization_pipeline
        from dictator.extraction.service import load_diarization_pipeline

        loaded = load_diarization_pipeline()
        with self._lock:
            if self._diarization_pipeline is None:
                self._diarization_pipeline = loaded
            return self._diarization_pipeline

    def get_synthesis_service(self):
        from dictator.synthesis.service import SpeechSynthesisService, XTTSBackend

        with self._lock:
            if self._tts_backend is None:
                self._tts_backend = XTTSBackend()
            backend = self._tts_backend
        return SpeechSynthesisService(backend=backend)

    def get_alignment_service(self):
        from dictator.alignment.service import AlignmentService
        from dictator.alignment.whisperx_backend import WhisperXAlignmentBackend

        with self._lock:
            if self._alignment_backend is None:
                self._alignment_backend = WhisperXAlignmentBackend()
            backend = self._alignment_backend
        return AlignmentService(backend=backend)

    def get_reference_extraction_service(self):
        from dictator.extraction.service import ReferenceExtractionService

        return ReferenceExtractionService()
