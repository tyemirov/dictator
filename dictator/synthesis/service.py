"""Qwen3-TTS speech synthesis service."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Callable, Mapping, Protocol, Sequence
import xml.etree.ElementTree as ET

from dictator.runtime import DependencyError, ValidationError

from .config import (
    DEFAULT_QWEN3_MODEL_ID,
    DEFAULT_SILERO_RU_DEFAULT_SPEAKER,
    DEFAULT_SILERO_RU_MODEL_SHA256,
    DEFAULT_SILERO_RU_MODEL_URL,
    DEFAULT_SILERO_RU_SAMPLE_RATE,
    DEFAULT_SILERO_RU_TEXT_CHAR_BUDGET,
    DEFAULT_QWEN3_TEXT_TOKEN_BUDGET,
    QWEN3_FAST_ATTENTION_IMPLEMENTATION,
    SynthesisConfig,
)
from .models import (
    SILERO_RU_NATIVE_SAMPLE_RATES,
    SILERO_RU_SUPPORTED_SPEAKERS,
    SpeechSegment,
    SynthesisedAudioChunk,
    SynthesisChunk,
    SynthesisEngine,
    SynthesisRequest,
    SynthesisResult,
    SynthesisTextFormat,
)
from .text import clean, join_synthesis_units, split_into_sentences

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
SILERO_RU_LANGUAGE_CODE = "ru"
INTER_CHUNK_SILENCE_SECONDS = 0.18
ProgressCallback = Callable[[int, int], None]
SILERO_SSML_SUPPORTED_TAGS = {"speak", "break", "prosody", "p", "s"}
SILERO_SSML_SUPPORTED_ATTRIBUTES = {
    "break": {"time", "strength"},
    "prosody": {"rate", "pitch"},
}
SILERO_RU_VOWELS = frozenset("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


class SynthesisSession(Protocol):
    """Per-request synthesis session with reusable prompt state."""

    def build_chunks(self, text: str) -> Sequence[SynthesisChunk]:
        ...

    def refine_chunk(self, chunk: SynthesisChunk) -> Sequence[SynthesisChunk]:
        ...

    def synthesise_chunk(self, text: str) -> SynthesisedAudioChunk:
        ...


class SynthesisBackend(Protocol):
    """Speech synthesis engine contract."""

    engine: SynthesisEngine

    def open_session(self, request: SynthesisRequest) -> SynthesisSession:
        ...


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


def _base_language_code(language_code: str) -> str:
    normalized = (language_code or "").strip().lower().replace("_", "-")
    return normalized.split("-", 1)[0]


def _pack_text_units_by_chars(units: Sequence[str], char_budget: int) -> tuple[SynthesisChunk, ...]:
    chunks: list[SynthesisChunk] = []
    buffer: list[str] = []
    for unit in units:
        candidate_units = [*buffer, unit]
        candidate_text = join_synthesis_units(candidate_units)
        if buffer and len(candidate_text) > char_budget:
            chunks.append(SynthesisChunk.from_units(buffer))
            buffer = [unit]
            continue
        buffer = candidate_units
    if buffer:
        chunks.append(SynthesisChunk.from_units(buffer))
    return tuple(chunks)


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if tag.startswith("{") else tag


def _looks_like_ssml(text: str) -> bool:
    return text.lstrip().lower().startswith("<speak")


def _validate_silero_ssml(text: str) -> ET.Element:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValidationError(
            "dictator.synthesis.silero_ru.invalid_ssml",
            f"silero_ru SSML text is not well-formed XML: {exc}",
        ) from exc
    if _xml_local_name(root.tag) != "speak":
        raise ValidationError(
            "dictator.synthesis.silero_ru.invalid_ssml",
            "silero_ru SSML text must use <speak> as the root tag",
        )
    for element in root.iter():
        tag_name = _xml_local_name(element.tag)
        if tag_name not in SILERO_SSML_SUPPORTED_TAGS:
            supported = ", ".join(f"<{tag}>" for tag in sorted(SILERO_SSML_SUPPORTED_TAGS))
            raise ValidationError(
                "dictator.synthesis.silero_ru.unsupported_ssml",
                f"silero_ru SSML tag <{tag_name}> is unsupported; supported tags: {supported}",
            )
        supported_attributes = SILERO_SSML_SUPPORTED_ATTRIBUTES.get(tag_name, set())
        unsupported_attributes = sorted(set(element.attrib) - supported_attributes)
        if unsupported_attributes:
            raise ValidationError(
                "dictator.synthesis.silero_ru.unsupported_ssml",
                f"silero_ru SSML tag <{tag_name}> does not support attributes: {', '.join(unsupported_attributes)}",
            )
    return root


def _ssml_timeline_text(root: ET.Element) -> str:
    return _strip_silero_stress_markers(" ".join(" ".join(root.itertext()).split()))


def _strip_silero_stress_markers(text: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "+" and index + 1 < len(text) and text[index + 1] in SILERO_RU_VOWELS:
            index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


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
            candidate_text = join_synthesis_units(candidate_units)
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


class Qwen3TTSBackend:
    """Lazy Qwen3-TTS model wrapper for voice cloning."""

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
                }
                if torch.cuda.is_available():
                    try:
                        import flash_attn  # noqa: F401
                    except ImportError:
                        raise DependencyError(
                            "dictator.synthesis.qwen3.flash_attn_required",
                            f"flash-attn is required for qwen3 synthesis on CUDA hosts; install it before loading {self.model_id}",
                        )
                    load_kwargs["attn_implementation"] = QWEN3_FAST_ATTENTION_IMPLEMENTATION
                logging.info(
                    "loading qwen3 model %s on %s dtype=%s attn_implementation=%s",
                    self.model_id,
                    load_kwargs["device_map"],
                    self.dtype,
                    load_kwargs.get("attn_implementation", "default"),
                )
                self._model = Qwen3TTSModel.from_pretrained(self.model_id, **load_kwargs)
                if load_kwargs.get("attn_implementation") == QWEN3_FAST_ATTENTION_IMPLEMENTATION:
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
        if request.text_format == SynthesisTextFormat.SSML:
            raise ValidationError(
                "dictator.synthesis.qwen3.text_format_unsupported",
                "qwen3 synthesis does not support SSML text_format",
            )
        speaker_transcript_text = (request.speaker_transcript_text or "").strip()
        if not speaker_transcript_text:
            raise ValidationError(
                "dictator.synthesis.qwen3.reference_text_required",
                "speaker_transcript_text is required for qwen3 synthesis",
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


class SileroRuSynthesisSession:
    def __init__(
        self,
        model,
        *,
        speaker: str,
        sample_rate: int,
        text_char_budget: int,
        text_format: SynthesisTextFormat = SynthesisTextFormat.AUTO,
    ) -> None:
        self._model = model
        self._speaker = speaker
        self._sample_rate = sample_rate
        self._text_char_budget = text_char_budget
        self._text_format = text_format

    def _uses_ssml(self, text: str) -> bool:
        return self._text_format == SynthesisTextFormat.SSML or (
            self._text_format == SynthesisTextFormat.AUTO and _looks_like_ssml(text)
        )

    def build_chunks(self, text: str) -> Sequence[SynthesisChunk]:
        if self._uses_ssml(text):
            root = _validate_silero_ssml(text)
            timeline_text = _ssml_timeline_text(root)
            if not timeline_text:
                raise ValidationError(
                    "dictator.synthesis.silero_ru.empty_ssml",
                    "silero_ru SSML text must contain speakable text",
                )
            if len(timeline_text) > self._text_char_budget:
                raise ValidationError(
                    "dictator.synthesis.silero_ru.ssml_text_too_long",
                    f"silero_ru SSML speakable text exceeds char budget {self._text_char_budget}",
                )
            logging.info("silero_ru using one SSML chunk")
            return (SynthesisChunk.from_text(text, timeline_text=timeline_text),)
        units = _sentence_units(text) or (text,)
        chunks = _pack_text_units_by_chars(units, self._text_char_budget)
        logging.info(
            "silero_ru packed %d sentence units into %d chunks with char budget=%d",
            len(units),
            len(chunks),
            self._text_char_budget,
        )
        return chunks

    def refine_chunk(self, chunk: SynthesisChunk) -> Sequence[SynthesisChunk]:
        if len(chunk.units) <= 1:
            return (chunk,)
        midpoint = len(chunk.units) // 2
        return (
            SynthesisChunk.from_units(chunk.units[:midpoint]),
            SynthesisChunk.from_units(chunk.units[midpoint:]),
        )

    def synthesise_chunk(self, text: str) -> SynthesisedAudioChunk:
        if self._uses_ssml(text):
            _validate_silero_ssml(text)
            try:
                samples = self._model.apply_tts(
                    ssml_text=text,
                    speaker=self._speaker,
                    sample_rate=self._sample_rate,
                )
            except TypeError as exc:
                raise DependencyError(
                    "dictator.synthesis.silero_ru.ssml_unsupported",
                    "loaded silero_ru model does not support apply_tts(ssml_text=...)",
                ) from exc
            if hasattr(samples, "detach"):
                samples = samples.detach().cpu().numpy()
            duration_seconds = len(samples) / self._sample_rate if self._sample_rate else 0.0
            return SynthesisedAudioChunk(
                samples=samples,
                sample_rate=self._sample_rate,
                duration_seconds=duration_seconds,
            )
        kwargs = {
            "text": text,
            "speaker": self._speaker,
            "sample_rate": self._sample_rate,
            "put_accent": True,
            "put_yo": True,
        }
        try:
            samples = self._model.apply_tts(**kwargs)
        except TypeError:
            kwargs.pop("put_accent")
            kwargs.pop("put_yo")
            samples = self._model.apply_tts(**kwargs)
        if hasattr(samples, "detach"):
            samples = samples.detach().cpu().numpy()
        duration_seconds = len(samples) / self._sample_rate if self._sample_rate else 0.0
        return SynthesisedAudioChunk(
            samples=samples,
            sample_rate=self._sample_rate,
            duration_seconds=duration_seconds,
        )


class SileroRuTTSBackend:
    """Lazy Silero v5 Russian preset-speaker TTS wrapper."""

    engine = SynthesisEngine.SILERO_RU

    def __init__(
        self,
        *,
        model_path: str = "",
        model_url: str = DEFAULT_SILERO_RU_MODEL_URL,
        model_sha256: str = DEFAULT_SILERO_RU_MODEL_SHA256,
        default_speaker: str = DEFAULT_SILERO_RU_DEFAULT_SPEAKER,
        sample_rate: int = DEFAULT_SILERO_RU_SAMPLE_RATE,
        text_char_budget: int = DEFAULT_SILERO_RU_TEXT_CHAR_BUDGET,
    ) -> None:
        self.model_path = model_path
        self.model_url = model_url
        self.model_sha256 = model_sha256.strip().lower()
        self.default_speaker = default_speaker
        self.sample_rate = sample_rate
        self.text_char_budget = text_char_budget
        self._model = None
        self._load_lock = threading.Lock()

    def _resolve_speaker(self, preset_speaker: str | None) -> str:
        speaker = (preset_speaker or self.default_speaker).strip().lower()
        if speaker not in SILERO_RU_SUPPORTED_SPEAKERS:
            supported = ", ".join(sorted(SILERO_RU_SUPPORTED_SPEAKERS))
            raise ValidationError(
                "dictator.synthesis.silero_ru.speaker_unsupported",
                f"silero_ru speaker must be one of: {supported}",
            )
        return speaker

    def _resolve_sample_rate(self, request: SynthesisRequest) -> int:
        if request.audio_format is None:
            return self.sample_rate
        requested_sample_rate = request.audio_format.sample_rate_hz
        if requested_sample_rate in SILERO_RU_NATIVE_SAMPLE_RATES:
            return requested_sample_rate
        return self.sample_rate

    def _download_model(self, torch, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(self.model_url, str(destination))
        return destination

    def _verify_model_file(self, path: Path) -> None:
        if not self.model_sha256:
            return
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if digest != self.model_sha256:
            raise DependencyError(
                "dictator.synthesis.silero_ru.model_digest_mismatch",
                f"silero_ru model digest mismatch for {path}: expected {self.model_sha256}, got {digest}",
            )

    def _model_cache_path(self) -> Path:
        return Path.home() / ".cache" / "dictator" / "models" / "silero" / Path(self.model_url).name

    def _ensure_model_path(self, torch) -> Path:
        if self.model_path:
            configured = Path(self.model_path)
            if configured.exists():
                self._verify_model_file(configured)
                return configured
            downloaded = self._download_model(torch, configured)
            try:
                self._verify_model_file(downloaded)
            except DependencyError:
                downloaded.unlink(missing_ok=True)
                raise
            return downloaded
        cached = self._model_cache_path()
        if cached.exists():
            try:
                self._verify_model_file(cached)
                return cached
            except DependencyError:
                cached.unlink(missing_ok=True)
        downloaded = self._download_model(torch, cached)
        try:
            self._verify_model_file(downloaded)
        except DependencyError:
            downloaded.unlink(missing_ok=True)
            raise
        return downloaded

    def load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                import torch

                model_path = self._ensure_model_path(torch)
                logging.info("loading silero_ru model from %s", model_path)
                importer = torch.package.PackageImporter(str(model_path))
                model = importer.load_pickle("tts_models", "model")
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)
                self._model = model
        return self._model

    def open_session(self, request: SynthesisRequest) -> SynthesisSession:
        if _base_language_code(request.language_code) != SILERO_RU_LANGUAGE_CODE:
            raise ValidationError(
                "dictator.synthesis.silero_ru.language_unsupported",
                f"silero_ru synthesis requires language_code='ru', got {request.language_code!r}",
            )
        return SileroRuSynthesisSession(
            self.load(),
            speaker=self._resolve_speaker(request.preset_speaker),
            sample_rate=self._resolve_sample_rate(request),
            text_char_budget=self.text_char_budget,
            text_format=request.text_format,
        )


class SpeechSynthesisService:
    """Service layer for request-safe speech synthesis."""

    def __init__(self, backends: Mapping[SynthesisEngine, SynthesisBackend] | None = None) -> None:
        if backends is None:
            synthesis_config = SynthesisConfig.from_env()
            backends = {
                SynthesisEngine.QWEN3: Qwen3TTSBackend(
                    model_id=synthesis_config.qwen3_model_id,
                    dtype=synthesis_config.qwen3_dtype,
                    text_token_budget=synthesis_config.qwen3_text_token_budget,
                ),
                SynthesisEngine.SILERO_RU: SileroRuTTSBackend(
                    model_path=synthesis_config.silero_ru_model_path,
                    model_url=synthesis_config.silero_ru_model_url,
                    model_sha256=synthesis_config.silero_ru_model_sha256,
                    default_speaker=synthesis_config.silero_ru_default_speaker,
                    sample_rate=synthesis_config.silero_ru_sample_rate,
                    text_char_budget=synthesis_config.silero_ru_text_char_budget,
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

    def _synthesise_chunks(
        self,
        *,
        session: SynthesisSession,
        chunks: Sequence[SynthesisChunk],
        cap_seconds: float | None,
        progress_callback: ProgressCallback | None = None,
    ) -> SynthesisResult:
        if not chunks:
            raise ValueError("No text chunks provided")
        estimated_total_chunks = len(chunks)
        if progress_callback is not None:
            progress_callback(0, estimated_total_chunks)

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
            generated_chunk = session.synthesise_chunk(_render_synthesis_text(chunk))
            duration_seconds = generated_chunk.duration_seconds
            leading_silence_seconds = INTER_CHUNK_SILENCE_SECONDS if wav_paths else 0.0
            total_duration_seconds = leading_silence_seconds + duration_seconds
            previous_duration = total_duration_seconds

            if cap_seconds is not None and elapsed + total_duration_seconds > cap_seconds:
                refined_chunks = tuple(session.refine_chunk(chunk))
                if len(refined_chunks) > 1:
                    logging.info(
                        "Refining %d-unit chunk for remaining cap %.1fs",
                        len(chunk.units),
                        cap_seconds - elapsed,
                    )
                    estimated_total_chunks += len(refined_chunks) - 1
                    pending_chunks = list(refined_chunks) + pending_chunks
                    if progress_callback is not None:
                        progress_callback(len(wav_paths), estimated_total_chunks)
                    continue
                logging.warning(
                    "Chunk %.1fs longer than remaining cap (%.1fs) - skipped",
                    total_duration_seconds,
                    cap_seconds - elapsed,
                )
                break

            chunk_samples = _prepend_silence(
                generated_chunk.samples,
                generated_chunk.sample_rate,
                silence_seconds=leading_silence_seconds,
            )
            import soundfile as sf

            sf.write(wav_path, chunk_samples, generated_chunk.sample_rate)
            wav_paths.append(wav_path)
            segments.append(
                SpeechSegment(
                    text=chunk.timeline_text or chunk.text,
                    start_seconds=elapsed + leading_silence_seconds,
                    end_seconds=elapsed + total_duration_seconds,
                )
            )
            elapsed += total_duration_seconds
            logging.info(
                "chunk %03d  %.1f s speech + %.1f s pause  (cumulative %.1f s)",
                chunk_index,
                duration_seconds,
                leading_silence_seconds,
                elapsed,
            )
            if progress_callback is not None:
                progress_callback(len(wav_paths), estimated_total_chunks)
            chunk_index += 1

        if cap_seconds is not None and not wav_paths:
            logging.error("First chunk (%.1fs) exceeds --length - nothing generated", previous_duration)
            raise ValueError("No chunks fit within the length cap")

        return SynthesisResult(
            temp_dir=temp_dir,
            wav_paths=tuple(wav_paths),
            segments=tuple(segments),
        )

    def synthesise_text(
        self,
        request: SynthesisRequest,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> SynthesisResult:
        cleaned_text = clean(request.text)
        normalized_request = SynthesisRequest(
            engine=request.engine,
            speaker_wav=request.speaker_wav,
            text=cleaned_text,
            language_code=request.language_code,
            cap_seconds=request.cap_seconds,
            speaker_artifact_id=request.speaker_artifact_id,
            speaker_transcript_text=request.speaker_transcript_text,
            preset_speaker=request.preset_speaker,
            audio_format=request.audio_format,
            text_format=request.text_format,
        )
        session = self._resolve_backend(normalized_request.engine).open_session(normalized_request)
        chunks = tuple(session.build_chunks(cleaned_text))
        return self._synthesise_chunks(
            session=session,
            chunks=chunks,
            cap_seconds=request.cap_seconds,
            progress_callback=progress_callback,
        )


def _render_synthesis_text(chunk: SynthesisChunk) -> str:
    return join_synthesis_units(chunk.units)


def _prepend_silence(samples, sample_rate: int, *, silence_seconds: float):
    if silence_seconds <= 0.0:
        return samples

    import numpy as np

    sample_array = np.asarray(samples)
    silence_frame_count = max(1, int(round(sample_rate * silence_seconds)))
    silence_shape = (silence_frame_count, *sample_array.shape[1:])
    silence = np.zeros(silence_shape, dtype=sample_array.dtype)
    return np.concatenate((silence, sample_array), axis=0)


def cleanup_synthesis_result(result: SynthesisResult) -> None:
    """Remove a synthesis result's temporary directory."""
    shutil.rmtree(result.temp_dir, ignore_errors=True)
