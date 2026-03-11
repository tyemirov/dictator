"""Long-lived model/runtime registry for speech execution services."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading

from dictator.synthesis.config import SynthesisConfig
from dictator.synthesis.models import SynthesisEngine


@dataclass(frozen=True)
class ReadinessComponent:
    name: str
    ready: bool
    detail: str = ""


@dataclass(frozen=True)
class ReadinessSnapshot:
    ready: bool
    warmup_started: bool
    warmup_in_progress: bool
    components: tuple[ReadinessComponent, ...]
    last_error: str = ""


class SpeechExecutionRuntime:
    """Caches heavyweight models and adapters across requests."""

    def __init__(self, *, synthesis_config: SynthesisConfig | None = None) -> None:
        self._lock = threading.Lock()
        self._whisper_models: dict[str, object] = {}
        self._whisper_model_load_locks: dict[str, threading.Lock] = {}
        self._transcription_service: object | None = None
        self._diarization_pipeline: object | None = None
        self._diarization_pipeline_load_lock = threading.Lock()
        self._synthesis_config = synthesis_config or SynthesisConfig.from_env()
        self._tts_backends: dict[SynthesisEngine, object] = {}
        self._synthesis_service: object | None = None
        self._alignment_backend: object | None = None
        self._warmup_thread: threading.Thread | None = None
        self._warmup_started = False
        self._warmup_in_progress = False
        self._warmup_last_error = ""
        self._readiness_components = {
            "transcription": False,
            "diarization": False,
            "synthesis": False,
        }

    def _set_component_ready(self, component: str, ready: bool) -> None:
        with self._lock:
            self._readiness_components[component] = ready

    def start_background_warmup(self) -> None:
        with self._lock:
            if self._warmup_started:
                return
            self._warmup_started = True
            self._warmup_in_progress = True
            self._warmup_thread = threading.Thread(
                target=self._run_warmup,
                name="dictator-runtime-warmup",
                daemon=True,
            )
            self._warmup_thread.start()

    def _run_warmup(self) -> None:
        try:
            self.get_whisper_model("base")
            self.get_diarization_pipeline()
            backend = self.get_synthesis_service().backends[SynthesisEngine.QWEN3]
            backend.load()
            self.mark_synthesis_ready()
        except Exception as exc:  # pragma: no cover - exercised via snapshot
            logging.exception("runtime warmup failed")
            with self._lock:
                self._warmup_last_error = str(exc)
                self._warmup_in_progress = False
            return
        with self._lock:
            self._warmup_in_progress = False
            self._warmup_last_error = ""

    def readiness_snapshot(self) -> ReadinessSnapshot:
        with self._lock:
            components = tuple(
                ReadinessComponent(name=name, ready=ready)
                for name, ready in self._readiness_components.items()
            )
            ready = all(component.ready for component in components) and not self._warmup_in_progress
            return ReadinessSnapshot(
                ready=ready,
                warmup_started=self._warmup_started,
                warmup_in_progress=self._warmup_in_progress,
                components=components,
                last_error=self._warmup_last_error,
            )

    def get_whisper_model(self, model_size: str) -> object:
        with self._lock:
            cached = self._whisper_models.get(model_size)
            if cached is not None:
                return cached
            load_lock = self._whisper_model_load_locks.get(model_size)
            if load_lock is None:
                load_lock = threading.Lock()
                self._whisper_model_load_locks[model_size] = load_lock
        with load_lock:
            with self._lock:
                cached = self._whisper_models.get(model_size)
                if cached is not None:
                    return cached
            from dictator.transcription.service import load_whisper_model

            loaded = load_whisper_model(model_size)
            with self._lock:
                self._whisper_models.setdefault(model_size, loaded)
                self._readiness_components["transcription"] = True
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
        with self._diarization_pipeline_load_lock:
            with self._lock:
                if self._diarization_pipeline is not None:
                    return self._diarization_pipeline
            from dictator.extraction.service import load_diarization_pipeline

            loaded = load_diarization_pipeline()
            with self._lock:
                if self._diarization_pipeline is None:
                    self._diarization_pipeline = loaded
                    self._readiness_components["diarization"] = True
                return self._diarization_pipeline

    def get_synthesis_service(self):
        from dictator.synthesis.service import Qwen3TTSBackend, SpeechSynthesisService

        with self._lock:
            if self._synthesis_service is None:
                if SynthesisEngine.QWEN3 not in self._tts_backends:
                    self._tts_backends[SynthesisEngine.QWEN3] = Qwen3TTSBackend(
                        model_id=self._synthesis_config.qwen3_model_id,
                        dtype=self._synthesis_config.qwen3_dtype,
                        text_token_budget=self._synthesis_config.qwen3_text_token_budget,
                    )
                self._synthesis_service = SpeechSynthesisService(backends=dict(self._tts_backends))
            return self._synthesis_service

    def mark_synthesis_ready(self) -> None:
        self._set_component_ready("synthesis", True)

    def get_alignment_service(self):
        from dictator.alignment.service import AlignmentService
        from dictator.alignment.whisperx_backend import WhisperXAlignmentBackend

        with self._lock:
            if self._alignment_backend is None:
                self._alignment_backend = WhisperXAlignmentBackend()
            backend = self._alignment_backend
        return AlignmentService(backend=backend)

    def get_diarization_service(self):
        from dictator.diarization.service import DiarizationService

        return DiarizationService(
            transcription_service=self.get_transcription_service(),
            diarization_pipeline_loader=self.get_diarization_pipeline,
        )

    def get_subtitle_service(self):
        from dictator.subtitles.service import SubtitleService

        return SubtitleService(
            transcription_service=self.get_transcription_service(),
            alignment_service=self.get_alignment_service(),
        )

    def get_reference_extraction_service(self):
        from dictator.extraction.service import ReferenceExtractionService

        return ReferenceExtractionService()
