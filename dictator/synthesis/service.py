"""Engine-aware speech synthesis service."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Mapping, Protocol, Sequence

from dictator.runtime import DependencyError, ValidationError

from .config import (
    DEFAULT_COSYVOICE3_MODEL_DIR,
    DEFAULT_QWEN3_MODEL_ID,
    DEFAULT_QWEN3_TEXT_TOKEN_BUDGET,
    DEFAULT_XTTS_MODEL_ID,
    QWEN3_FAST_ATTENTION_IMPLEMENTATION,
    SynthesisConfig,
)
from .models import (
    SpeechSegment,
    SynthesisedAudioChunk,
    SynthesisChunk,
    SynthesisEngine,
    SynthesisRequest,
    SynthesisResult,
)
from .text import build_chunks, clean, split_into_sentences

QWEN3_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
}

COSYVOICE3_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


class SynthesisSession(Protocol):
    """Per-request synthesis session with any reusable prompt state."""

    def build_chunks(self, text: str) -> Sequence[SynthesisChunk]:
        ...

    def refine_chunk(self, chunk: SynthesisChunk) -> Sequence[SynthesisChunk]:
        ...

    def synthesise_to_file(self, text: str, output_path: Path) -> None:
        ...


class SynthesisBackend(Protocol):
    """Engine implementation contract."""

    engine: SynthesisEngine

    def open_session(self, request: SynthesisRequest) -> SynthesisSession:
        ...


class LegacySynthesisSession:
    """Adapter for older backend fakes that only expose synthesise_to_file()."""

    def __init__(self, backend, *, speaker_wav: Path, language_code: str) -> None:
        self._backend = backend
        self._speaker_wav = speaker_wav
        self._language_code = language_code

    def build_chunks(self, text: str) -> Sequence[SynthesisChunk]:
        return tuple(_chunk_from_text(chunk_text) for chunk_text in build_chunks(text))

    def refine_chunk(self, chunk: SynthesisChunk) -> Sequence[SynthesisChunk]:
        return (chunk,)

    def synthesise_to_file(self, text: str, output_path: Path) -> None:
        self._backend.synthesise_to_file(
            text,
            self._speaker_wav,
            self._language_code,
            output_path,
        )


def _qwen3_language_name(language_code: str) -> str:
    normalized = (language_code or "").strip().lower().replace("_", "-")
    base_language = normalized.split("-", 1)[0]
    language_name = QWEN3_LANGUAGE_NAMES.get(base_language)
    if language_name is None:
        raise ValidationError(
            "dictator.synthesis.qwen3.language_unsupported",
            f"qwen3 synthesis does not support language_code={language_code!r}",
        )
    return language_name


def _sentence_units(text: str) -> tuple[str, ...]:
    return tuple(sentence.strip() for sentence in split_into_sentences(text) if sentence.strip())


def _chunk_from_text(text: str) -> SynthesisChunk:
    return SynthesisChunk.from_units(_sentence_units(text) or (text,))


class XTTSSynthesisSession:
    def __init__(self, tts, *, speaker_wav: Path, language_code: str) -> None:
        self._tts = tts
        self._speaker_wav = speaker_wav
        self._language_code = language_code

    def build_chunks(self, text: str) -> Sequence[SynthesisChunk]:
        return tuple(_chunk_from_text(chunk_text) for chunk_text in build_chunks(text))

    def refine_chunk(self, chunk: SynthesisChunk) -> Sequence[SynthesisChunk]:
        return (chunk,)

    def synthesise_to_file(self, text: str, output_path: Path) -> None:
        self._tts.tts_to_file(
            text=text,
            speaker_wav=str(self._speaker_wav),
            language=self._language_code,
            file_path=str(output_path),
        )


class XTTSBackend:
    """Lazy XTTS model wrapper."""

    engine = SynthesisEngine.XTTS

    def __init__(self, model_id: str = DEFAULT_XTTS_MODEL_ID) -> None:
        self.model_id = model_id
        self._tts = None
        self._load_lock = threading.Lock()

    def load(self):
        if self._tts is not None:
            return self._tts
        with self._load_lock:
            if self._tts is None:
                import torch
                from TTS.api import TTS

                device = "cuda" if torch.cuda.is_available() else "cpu"
                logging.info("loading xtts model %s on %s", self.model_id, device)
                self._tts = TTS(self.model_id).to(device)
        return self._tts

    def open_session(self, request: SynthesisRequest) -> SynthesisSession:
        return XTTSSynthesisSession(
            self.load(),
            speaker_wav=request.speaker_wav,
            language_code=request.language_code,
        )

    def synthesise_to_file(
        self,
        text: str,
        speaker_wav: Path,
        language_code: str,
        output_path: Path,
    ) -> None:
        self.open_session(
            SynthesisRequest(
                engine=SynthesisEngine.XTTS,
                speaker_wav=speaker_wav,
                text=text,
                language_code=language_code,
                cap_seconds=None,
            )
        ).synthesise_to_file(text, output_path)


class Qwen3SynthesisSession:
    def __init__(
        self,
        model,
        *,
        voice_clone_prompt,
        language_name: str,
        text_token_budget: int,
    ) -> None:
        self._model = model
        self._voice_clone_prompt = voice_clone_prompt
        self._language_name = language_name
        self._text_token_budget = text_token_budget

    def _estimate_text_tokens(self, text: str) -> int:
        assistant_text = self._model._build_assistant_text(text)
        tokenized = self._model._tokenize_texts([assistant_text])
        return int(tokenized[0].shape[-1])

    def _pack_units(self, units: Sequence[str]) -> tuple[SynthesisChunk, ...]:
        chunks: list[SynthesisChunk] = []
        buffer: list[str] = []
        for unit in units:
            candidate_units = [*buffer, unit]
            candidate_text = " ".join(candidate_units)
            candidate_tokens = self._estimate_text_tokens(candidate_text)
            if buffer and candidate_tokens > self._text_token_budget:
                chunks.append(SynthesisChunk.from_units(buffer))
                buffer = [unit]
                continue
            buffer = candidate_units
        if buffer:
            chunks.append(SynthesisChunk.from_units(buffer))
        return tuple(chunks)

    def build_chunks(self, text: str) -> Sequence[SynthesisChunk]:
        units = _sentence_units(text)
        packed_chunks = self._pack_units(units)
        logging.info(
            "qwen3 packed %d sentence units into %d chunks with token budget=%d",
            len(units),
            len(packed_chunks),
            self._text_token_budget,
        )
        return packed_chunks

    def refine_chunk(self, chunk: SynthesisChunk) -> Sequence[SynthesisChunk]:
        if len(chunk.units) <= 1:
            return (chunk,)
        midpoint = len(chunk.units) // 2
        return (
            SynthesisChunk.from_units(chunk.units[:midpoint]),
            SynthesisChunk.from_units(chunk.units[midpoint:]),
        )

    def synthesise_chunk(self, text: str) -> SynthesisedAudioChunk:
        wavs, sample_rate = self._model.generate_voice_clone(
            text=text,
            language=self._language_name,
            voice_clone_prompt=self._voice_clone_prompt,
            non_streaming_mode=True,
        )
        if not wavs:
            raise ValueError("qwen3 synthesis returned no audio")
        samples = wavs[0]
        duration_seconds = len(samples) / sample_rate if sample_rate else 0.0
        return SynthesisedAudioChunk(
            samples=samples,
            sample_rate=sample_rate,
            duration_seconds=duration_seconds,
        )

    def synthesise_to_file(self, text: str, output_path: Path) -> None:
        import soundfile as sf

        chunk = self.synthesise_chunk(text)
        sf.write(output_path, chunk.samples, chunk.sample_rate)


class Qwen3TTSBackend:
    """Lazy Qwen3-TTS base model wrapper for voice cloning."""

    engine = SynthesisEngine.QWEN3

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_QWEN3_MODEL_ID,
        dtype: str = "auto",
        text_token_budget: int = DEFAULT_QWEN3_TEXT_TOKEN_BUDGET,
    ) -> None:
        self.model_id = model_id
        self.dtype = dtype
        self.text_token_budget = text_token_budget
        self._model = None
        self._load_lock = threading.Lock()
        self._prompt_cache: dict[tuple[str, str, str], object] = {}
        self._prompt_cache_lock = threading.Lock()

    def _resolve_dtype(self, torch):
        name = self.dtype.lower()
        if name == "auto":
            return torch.bfloat16 if torch.cuda.is_available() else torch.float32
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if name not in mapping:
            raise ValueError(
                f"unsupported qwen3 dtype {self.dtype!r}; expected auto, bfloat16, float16, or float32"
            )
        return mapping[name]

    def load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                import torch
                from qwen_tts import Qwen3TTSModel

                load_kwargs = {
                    "device_map": "cuda:0" if torch.cuda.is_available() else "cpu",
                    "dtype": self._resolve_dtype(torch),
                    "attn_implementation": QWEN3_FAST_ATTENTION_IMPLEMENTATION,
                }
                logging.info(
                    "loading qwen3 model %s on %s dtype=%s attn_implementation=%s",
                    self.model_id,
                    load_kwargs["device_map"],
                    self.dtype,
                    QWEN3_FAST_ATTENTION_IMPLEMENTATION,
                )
                self._model = Qwen3TTSModel.from_pretrained(self.model_id, **load_kwargs)
                logging.info("qwen3 fast attention is enabled via %s", QWEN3_FAST_ATTENTION_IMPLEMENTATION)
        return self._model

    def _prompt_cache_key(
        self,
        request: SynthesisRequest,
        *,
        speaker_transcript_text: str,
    ) -> tuple[str, str, str] | None:
        if not request.speaker_artifact_id:
            return None
        transcript_hash = hashlib.sha256(speaker_transcript_text.encode("utf-8")).hexdigest()
        return (request.speaker_artifact_id, transcript_hash, self.engine.value)

    def _create_voice_clone_prompt(self, model, request: SynthesisRequest, *, speaker_transcript_text: str):
        return model.create_voice_clone_prompt(
            ref_audio=str(request.speaker_wav),
            ref_text=speaker_transcript_text,
            x_vector_only_mode=False,
        )

    def open_session(self, request: SynthesisRequest) -> SynthesisSession:
        speaker_transcript_text = (request.speaker_transcript_text or "").strip()
        if not speaker_transcript_text:
            raise ValidationError(
                "dictator.synthesis.qwen3.reference_text_required",
                "speaker_transcript_text or speaker_transcript_artifact_id is required for qwen3 synthesis",
            )
        model = self.load()
        cache_key = self._prompt_cache_key(request, speaker_transcript_text=speaker_transcript_text)
        if cache_key is None:
            voice_clone_prompt = self._create_voice_clone_prompt(
                model,
                request,
                speaker_transcript_text=speaker_transcript_text,
            )
        else:
            with self._prompt_cache_lock:
                voice_clone_prompt = self._prompt_cache.get(cache_key)
            if voice_clone_prompt is None:
                logging.info("building qwen3 clone prompt for speaker artifact %s", request.speaker_artifact_id)
                built_prompt = self._create_voice_clone_prompt(
                    model,
                    request,
                    speaker_transcript_text=speaker_transcript_text,
                )
                with self._prompt_cache_lock:
                    voice_clone_prompt = self._prompt_cache.setdefault(cache_key, built_prompt)
            else:
                logging.info("reusing cached qwen3 clone prompt for speaker artifact %s", request.speaker_artifact_id)
        return Qwen3SynthesisSession(
            model,
            voice_clone_prompt=voice_clone_prompt,
            language_name=_qwen3_language_name(request.language_code),
            text_token_budget=self.text_token_budget,
        )


class CosyVoice3SynthesisSession:
    def __init__(
        self,
        model,
        *,
        speaker_wav: Path,
        speaker_transcript_text: str,
    ) -> None:
        self._model = model
        self._speaker_wav = speaker_wav
        self._speaker_transcript_text = speaker_transcript_text

    def build_chunks(self, text: str) -> Sequence[SynthesisChunk]:
        return tuple(
            SynthesisChunk.from_text(sentence)
            for sentence in split_into_sentences(text)
            if sentence.strip()
        )

    def refine_chunk(self, chunk: SynthesisChunk) -> Sequence[SynthesisChunk]:
        return (chunk,)

    def synthesise_chunk(self, text: str) -> SynthesisedAudioChunk:
        import numpy as np

        sample_rate = int(getattr(self._model, "sample_rate", 24000))
        sample_chunks: list[np.ndarray] = []
        prompt_text = f"{COSYVOICE3_PROMPT_PREFIX}{self._speaker_transcript_text}"
        for result in self._model.inference_zero_shot(
            text,
            prompt_text,
            str(self._speaker_wav),
            stream=False,
        ):
            samples = result.get("tts_speech")
            if samples is None:
                continue
            if hasattr(samples, "detach"):
                samples = samples.detach().cpu().numpy()
            array = np.asarray(samples, dtype=np.float32).reshape(-1)
            if array.size:
                sample_chunks.append(array)
        if not sample_chunks:
            raise ValueError("cosyvoice3 synthesis returned no audio")
        merged_samples = np.concatenate(sample_chunks)
        return SynthesisedAudioChunk(
            samples=merged_samples,
            sample_rate=sample_rate,
            duration_seconds=len(merged_samples) / sample_rate if sample_rate else 0.0,
        )

    def synthesise_to_file(self, text: str, output_path: Path) -> None:
        import soundfile as sf

        chunk = self.synthesise_chunk(text)
        sf.write(output_path, chunk.samples, chunk.sample_rate)


class CosyVoice3Backend:
    """Lazy CosyVoice 3 zero-shot model wrapper."""

    engine = SynthesisEngine.COSYVOICE3

    def __init__(self, *, model_dir: str = DEFAULT_COSYVOICE3_MODEL_DIR) -> None:
        self.model_dir = model_dir
        self._model = None
        self._load_lock = threading.Lock()

    def load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                model_path = Path(self.model_dir)
                if not model_path.exists():
                    raise DependencyError(
                        "dictator.synthesis.cosyvoice3.model_dir_missing",
                        f"CosyVoice3 model_dir does not exist: {model_path}",
                    )
                try:
                    from cosyvoice.cli.cosyvoice import AutoModel
                except ImportError as exc:
                    raise DependencyError(
                        "dictator.synthesis.cosyvoice3.unavailable",
                        "CosyVoice3 is unavailable; install the official CosyVoice repository and its dependencies.",
                    ) from exc
                logging.info("loading cosyvoice3 model from %s", model_path)
                self._model = AutoModel(model_dir=str(model_path))
        return self._model

    def open_session(self, request: SynthesisRequest) -> SynthesisSession:
        speaker_transcript_text = (request.speaker_transcript_text or "").strip()
        if not speaker_transcript_text:
            raise ValidationError(
                "dictator.synthesis.cosyvoice3.reference_text_required",
                "speaker_transcript_text or speaker_transcript_artifact_id is required for cosyvoice3 synthesis",
            )
        return CosyVoice3SynthesisSession(
            self.load(),
            speaker_wav=request.speaker_wav,
            speaker_transcript_text=speaker_transcript_text,
        )


class SpeechSynthesisService:
    """Service layer for request-safe, engine-aware synthesis."""

    def __init__(
        self,
        backend: SynthesisBackend | None = None,
        backends: Mapping[SynthesisEngine, SynthesisBackend] | None = None,
    ) -> None:
        if backend is not None and backends is not None:
            raise ValueError("set backend or backends, not both")
        if backend is not None:
            engine = getattr(backend, "engine", SynthesisEngine.XTTS)
            backends = {engine: backend}
        if backends is None:
            synthesis_config = SynthesisConfig.from_env()
            backends = {
                SynthesisEngine.XTTS: XTTSBackend(model_id=synthesis_config.xtts_model_id),
                SynthesisEngine.QWEN3: Qwen3TTSBackend(
                    model_id=synthesis_config.qwen3_model_id,
                    dtype=synthesis_config.qwen3_dtype,
                    text_token_budget=synthesis_config.qwen3_text_token_budget,
                ),
                SynthesisEngine.COSYVOICE3: CosyVoice3Backend(
                    model_dir=synthesis_config.cosyvoice3_model_dir,
                ),
            }
        self.backends = dict(backends)

    def _resolve_backend(self, engine: SynthesisEngine) -> SynthesisBackend:
        backend = self.backends.get(engine)
        if backend is None:
            raise ValidationError(
                "dictator.synthesis.engine_unsupported",
                f"unsupported synthesis engine: {engine.value}",
            )
        return backend

    def _open_session(self, backend: SynthesisBackend, request: SynthesisRequest) -> SynthesisSession:
        open_session = getattr(backend, "open_session", None)
        if callable(open_session):
            return open_session(request)
        synthesise_to_file = getattr(backend, "synthesise_to_file", None)
        if callable(synthesise_to_file):
            return LegacySynthesisSession(
                backend,
                speaker_wav=request.speaker_wav,
                language_code=request.language_code,
            )
        raise ValueError(f"backend {backend!r} does not support synthesis sessions")

    def _synthesise_chunks(
        self,
        *,
        session: SynthesisSession,
        chunks: Sequence[SynthesisChunk],
        cap_seconds: float | None,
    ) -> SynthesisResult:
        if not chunks:
            raise ValueError("No text chunks provided")

        temp_dir = Path(tempfile.mkdtemp(prefix="dictator_tts_"))
        wav_paths: list[Path] = []
        segments: list[SpeechSegment] = []
        elapsed = 0.0
        previous_duration = 0.0
        pending_chunks = list(chunks)
        chunk_index = 0

        while pending_chunks:
            if cap_seconds is not None and elapsed >= cap_seconds:
                break

            chunk = pending_chunks.pop(0)
            wav_path = temp_dir / f"{chunk_index:04d}.wav"
            synthesise_chunk = getattr(session, "synthesise_chunk", None)
            generated_chunk = None
            if callable(synthesise_chunk):
                generated_chunk = synthesise_chunk(chunk.text)
                duration_seconds = generated_chunk.duration_seconds
                previous_duration = duration_seconds
            else:
                session.synthesise_to_file(chunk.text, wav_path)

                import soundfile as sf

                info = sf.info(wav_path)
                duration_seconds = info.frames / info.samplerate
                previous_duration = duration_seconds

            if cap_seconds is not None and elapsed + duration_seconds > cap_seconds:
                wav_path.unlink(missing_ok=True)
                refined_chunks = tuple(session.refine_chunk(chunk))
                if len(refined_chunks) > 1:
                    logging.info(
                        "Refining %d-unit chunk for remaining cap %.1fs",
                        len(chunk.units),
                        cap_seconds - elapsed,
                    )
                    pending_chunks = list(refined_chunks) + pending_chunks
                    continue
                logging.warning(
                    "Chunk %.1fs longer than remaining cap (%.1fs) - skipped",
                    duration_seconds,
                    cap_seconds - elapsed,
                )
                break

            if generated_chunk is not None:
                import soundfile as sf

                sf.write(wav_path, generated_chunk.samples, generated_chunk.sample_rate)

            wav_paths.append(wav_path)
            segments.append(
                SpeechSegment(
                    text=chunk.text,
                    start_seconds=elapsed,
                    end_seconds=elapsed + duration_seconds,
                )
            )
            elapsed += duration_seconds
            logging.info(
                "chunk %03d  %.1f s  (cumulative %.1f s)",
                chunk_index,
                duration_seconds,
                elapsed,
            )
            chunk_index += 1

        if cap_seconds is not None and not wav_paths:
            logging.error("First chunk (%.1fs) exceeds --length - nothing generated", previous_duration)
            raise ValueError("No chunks fit within the length cap")

        return SynthesisResult(
            temp_dir=temp_dir,
            wav_paths=tuple(wav_paths),
            segments=tuple(segments),
        )

    def synthesise(
        self,
        speaker_wav: Path,
        chunks: Sequence[str],
        cap_seconds: float | None,
        language_code: str,
        *,
        engine: SynthesisEngine = SynthesisEngine.XTTS,
        speaker_transcript_text: str | None = None,
    ) -> SynthesisResult:
        normalized_chunks = tuple(SynthesisChunk.from_text(chunk) for chunk in chunks)
        request = SynthesisRequest(
            engine=engine,
            speaker_wav=speaker_wav,
            text=" ".join(chunk.text for chunk in normalized_chunks).strip(),
            language_code=language_code,
            cap_seconds=cap_seconds,
            speaker_artifact_id=None,
            speaker_transcript_text=speaker_transcript_text,
        )
        session = self._open_session(self._resolve_backend(engine), request)
        return self._synthesise_chunks(
            session=session,
            chunks=normalized_chunks,
            cap_seconds=cap_seconds,
        )

    def synthesise_text(self, request: SynthesisRequest) -> SynthesisResult:
        cleaned_text = clean(request.text)
        normalized_request = SynthesisRequest(
            engine=request.engine,
            speaker_wav=request.speaker_wav,
            text=cleaned_text,
            language_code=request.language_code,
            cap_seconds=request.cap_seconds,
            speaker_artifact_id=request.speaker_artifact_id,
            speaker_transcript_text=request.speaker_transcript_text,
        )
        session = self._open_session(self._resolve_backend(request.engine), normalized_request)
        chunks = tuple(session.build_chunks(cleaned_text))
        return self._synthesise_chunks(session=session, chunks=chunks, cap_seconds=request.cap_seconds)


def cleanup_synthesis_result(result: SynthesisResult) -> None:
    """Remove a synthesis result's temporary directory."""
    shutil.rmtree(result.temp_dir, ignore_errors=True)


def synthesise(
    speaker_wav: Path,
    chunks: Sequence[str],
    cap: float | None,
    language_code: str,
    *,
    engine: SynthesisEngine = SynthesisEngine.XTTS,
    speaker_transcript_text: str | None = None,
) -> tuple[list[Path], list[dict[str, float | str]]]:
    """Compatibility wrapper returning temporary chunk paths and legacy timeline dicts."""
    result = SpeechSynthesisService().synthesise(
        speaker_wav,
        chunks,
        cap,
        language_code,
        engine=engine,
        speaker_transcript_text=speaker_transcript_text,
    )
    return list(result.wav_paths), [segment.to_legacy_dict() for segment in result.segments]
