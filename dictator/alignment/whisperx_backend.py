"""WhisperX-backed forced alignment backend."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import numbers
import os
from pathlib import Path
import threading
from types import ModuleType
from typing import Iterable, cast

from dictator.runtime.errors import DependencyError, ProcessingError, ValidationError

from .models import AlignedWord, SUPPORTED_LANGUAGE_CODES
from .text import is_punctuation_token, strip_punctuation_from_token

LOGGER = logging.getLogger(__name__)

ALIGNMENT_CODE = "dictator.alignment.align.failed"
ALIGN_MODEL_CODE = "dictator.alignment.align.model"
ALIGNMENT_TIMESTAMP_CODE = "dictator.alignment.align.missing_timestamps"
ALIGNMENT_INFERRED_TIMESTAMPS_CODE = "dictator.alignment.align.inferred_timestamps"
DEVICE_UNAVAILABLE_CODE = "dictator.alignment.device.unavailable"
TORCH_VERSION_CODE = "dictator.alignment.dependency.torch_version"
TORCHAUDIO_METADATA_CODE = "dictator.alignment.dependency.torchaudio_metadata"
DEFAULT_MISSING_TOKEN_SECONDS = 0.25
TORCH_HOME_ENV = "TORCH_HOME"
TORCH_MIN_VERSION = (2, 6)
TORCH_MIN_VERSION_TEXT = "2.6"
TORCHAUDIO_ALIGNMENT_LANGUAGES = {"en", "fr", "de", "es", "it"}
CORRUPTED_TORCH_ARCHIVE_PATTERNS = (
    "PytorchStreamReader failed reading zip archive",
    "failed finding central directory",
)

AlignmentModelKey = tuple[str, str]
AlignmentModelPayload = tuple[object, dict[str, object]]
_alignment_model_cache: dict[AlignmentModelKey, AlignmentModelPayload] = {}
_alignment_model_load_locks: dict[AlignmentModelKey, threading.Lock] = {}
_alignment_model_registry_lock = threading.Lock()


def parse_torch_version(version: str) -> tuple[int, int] | None:
    parts = version.strip().split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return int(parts[0]), int(parts[1])


def ensure_torch_version(torch_module: ModuleType, language: str) -> None:
    version = str(getattr(torch_module, "__version__", "")).strip()
    parsed = parse_torch_version(version)
    if parsed is None:
        raise DependencyError(TORCH_VERSION_CODE, f"torch version is invalid: {version!r}")
    if parsed < TORCH_MIN_VERSION:
        raise DependencyError(
            TORCH_VERSION_CODE,
            (
                f"torch >= {TORCH_MIN_VERSION_TEXT} is required for "
                f"{language!r} alignment models (installed: {version})"
            ),
        )


def load_torch_module() -> ModuleType:
    try:
        import torch
    except Exception as exc:
        raise DependencyError(DEVICE_UNAVAILABLE_CODE, f"torch is unavailable: {exc}") from exc
    return cast(ModuleType, torch)


def ensure_torchaudio_metadata() -> None:
    try:
        import torchaudio
    except Exception as exc:
        raise DependencyError(DEVICE_UNAVAILABLE_CODE, f"torchaudio is unavailable: {exc}") from exc
    if hasattr(torchaudio, "AudioMetaData"):
        return
    audio_metadata_type = None
    for module_name in ("torchaudio.backend.common", "torchaudio._backend.common"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        audio_metadata_type = getattr(module, "AudioMetaData", None)
        if audio_metadata_type is not None:
            break
    if audio_metadata_type is None:
        raise DependencyError(
            TORCHAUDIO_METADATA_CODE,
            "torchaudio is missing AudioMetaData; install torchaudio>=2.6",
        )
    setattr(torchaudio, "AudioMetaData", audio_metadata_type)


def load_whisperx_alignment_modules() -> tuple[ModuleType, ModuleType]:
    ensure_torchaudio_metadata()
    package_spec = importlib.util.find_spec("whisperx")
    if package_spec is None or package_spec.submodule_search_locations is None:
        raise DependencyError(ALIGNMENT_CODE, "whisperx is unavailable")
    try:
        alignment_module = importlib.import_module("whisperx.alignment")
        audio_module = importlib.import_module("whisperx.audio")
    except Exception as exc:
        raise DependencyError(ALIGNMENT_CODE, f"whisperx alignment import failed: {exc}") from exc
    return alignment_module, audio_module


def is_corrupted_torch_archive_error(message: str) -> bool:
    return any(pattern in message for pattern in CORRUPTED_TORCH_ARCHIVE_PATTERNS)


def resolve_torch_hub_checkpoints_dir() -> Path:
    torch_home = os.environ.get(TORCH_HOME_ENV, "").strip()
    if torch_home:
        return Path(torch_home) / "hub" / "checkpoints"
    return Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def clear_torch_hub_checkpoints() -> int:
    checkpoints_dir = resolve_torch_hub_checkpoints_dir()
    if not checkpoints_dir.exists():
        return 0
    removed = 0
    for candidate in checkpoints_dir.iterdir():
        if not candidate.is_file():
            continue
        candidate.unlink()
        removed += 1
    return removed


def clear_alignment_model_cache() -> None:
    with _alignment_model_registry_lock:
        _alignment_model_cache.clear()
        _alignment_model_load_locks.clear()


def normalize_alignment_model_key(language: str, device: str) -> AlignmentModelKey:
    normalized_language = language.strip().lower()
    normalized_device = device.strip().lower()
    if not normalized_language:
        raise ValidationError("dictator.alignment.input.invalid_language", "language must be non-empty")
    if normalized_language not in SUPPORTED_LANGUAGE_CODES:
        raise ValidationError(
            "dictator.alignment.input.invalid_language",
            f"unsupported language: {normalized_language!r}",
        )
    if not normalized_device:
        raise ValidationError(DEVICE_UNAVAILABLE_CODE, "device must be non-empty")
    return normalized_language, normalized_device


def load_alignment_model(
    language: str,
    device: str,
    alignment_module: ModuleType,
) -> AlignmentModelPayload:
    if language not in TORCHAUDIO_ALIGNMENT_LANGUAGES:
        ensure_torch_version(load_torch_module(), language)
    try:
        return alignment_module.load_align_model(language_code=language, device=device)
    except Exception as exc:
        if is_corrupted_torch_archive_error(str(exc)):
            removed_files = clear_torch_hub_checkpoints()
            LOGGER.warning(
                "%s: cleared %d torch checkpoint file(s) and retrying model load",
                ALIGN_MODEL_CODE,
                removed_files,
            )
            try:
                return alignment_module.load_align_model(language_code=language, device=device)
            except Exception as retry_exc:
                raise ProcessingError(
                    ALIGN_MODEL_CODE,
                    f"align model load failed after checkpoint cache reset: {retry_exc}",
                ) from retry_exc
        raise ProcessingError(ALIGN_MODEL_CODE, f"align model load failed: {exc}") from exc


def load_cached_alignment_model(
    language: str,
    device: str,
    alignment_module: ModuleType,
) -> AlignmentModelPayload:
    key = normalize_alignment_model_key(language, device)
    with _alignment_model_registry_lock:
        cached_model = _alignment_model_cache.get(key)
        if cached_model is not None:
            return cached_model
        load_lock = _alignment_model_load_locks.get(key)
        if load_lock is None:
            load_lock = threading.Lock()
            _alignment_model_load_locks[key] = load_lock
    with load_lock:
        with _alignment_model_registry_lock:
            cached_model = _alignment_model_cache.get(key)
            if cached_model is not None:
                return cached_model
        loaded_model = load_alignment_model(key[0], key[1], alignment_module)
        with _alignment_model_registry_lock:
            _alignment_model_cache[key] = loaded_model
        return loaded_model


def resolve_device(device: str = "auto") -> str:
    normalized = device.strip().lower() or "auto"
    if normalized != "auto":
        return normalized
    torch_module = load_torch_module()
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def preload_alignment_model(language: str, device: str = "auto") -> None:
    alignment_module, _ = load_whisperx_alignment_modules()
    load_cached_alignment_model(language, resolve_device(device), alignment_module)


def coerce_timestamp(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        candidate = float(value)
        if math.isfinite(candidate):
            return candidate
    return None


def default_segment_bounds(
    token_count: int,
    last_known_end: float | None,
) -> tuple[float, float]:
    start_seconds = last_known_end if last_known_end is not None else 0.0
    window_seconds = DEFAULT_MISSING_TOKEN_SECONDS * float(token_count)
    return start_seconds, start_seconds + window_seconds


def segment_bounds(
    segment: dict[str, object],
    fallback: tuple[float, float],
) -> tuple[float, float]:
    start_seconds = coerce_timestamp(segment.get("start"))
    end_seconds = coerce_timestamp(segment.get("end"))
    if start_seconds is not None and end_seconds is not None and end_seconds > start_seconds:
        return start_seconds, end_seconds
    return fallback


def segment_bounds_from_tokens(tokens: list[dict[str, object]]) -> tuple[float, float] | None:
    starts: list[float] = []
    ends: list[float] = []
    for token in tokens:
        start_value = coerce_timestamp(token.get("start"))
        end_value = coerce_timestamp(token.get("end"))
        if start_value is not None and end_value is not None and end_value > start_value:
            starts.append(start_value)
            ends.append(end_value)
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def token_weight(text_value: str) -> int:
    compact = "".join(part for part in text_value.split() if part)
    return max(1, len(compact))


def infer_missing_timings(
    tokens: list[dict[str, object]],
    segment_start: float,
    segment_end: float,
) -> None:
    missing_texts: list[str] = []
    for token in tokens:
        start_value = coerce_timestamp(token.get("start"))
        end_value = coerce_timestamp(token.get("end"))
        if start_value is not None and end_value is not None:
            token["start"] = start_value
            token["end"] = end_value
            continue
        token["start"] = None
        token["end"] = None
        missing_texts.append(str(token.get("text", "")).strip())

    if not missing_texts:
        return

    preview = ", ".join(missing_texts[:8])
    extra = "" if len(missing_texts) <= 8 else f" (+{len(missing_texts) - 8} more)"
    LOGGER.warning(
        "%s: inferring timestamps for %d token(s): %s%s",
        ALIGNMENT_INFERRED_TIMESTAMPS_CODE,
        len(missing_texts),
        preview,
        extra,
    )

    index = 0
    while index < len(tokens):
        if isinstance(tokens[index].get("start"), float):
            index += 1
            continue
        run_start = index
        while index < len(tokens) and tokens[index].get("start") is None:
            index += 1
        run_end = index

        left_bound = segment_start
        if run_start > 0:
            left_bound = cast(float, tokens[run_start - 1].get("end"))
        right_bound = segment_end
        if run_end < len(tokens):
            right_bound = cast(float, tokens[run_end].get("start"))
        if right_bound <= left_bound:
            right_bound = left_bound + (0.001 * float(run_end - run_start))

        window_seconds = right_bound - left_bound
        weights = [token_weight(str(tokens[i].get("text", "")).strip()) for i in range(run_start, run_end)]
        total_weight = float(sum(weights)) or 1.0
        cursor = left_bound
        for offset, weight in enumerate(weights):
            share = window_seconds * (float(weight) / total_weight)
            start_seconds = cursor
            end_seconds = cursor + share
            token = tokens[run_start + offset]
            token["start"] = start_seconds
            token["end"] = end_seconds
            cursor = end_seconds


def extract_aligned_words(
    segments: Iterable[dict[str, object]],
    remove_punctuation: bool = False,
) -> tuple[AlignedWord, ...]:
    words: list[AlignedWord] = []
    pending_prefix_tokens: list[str] = []
    last_known_end: float | None = None
    for segment in segments:
        if not isinstance(segment, dict):
            raise ProcessingError(ALIGNMENT_CODE, "alignment segment payload must be an object")
        raw_words = segment.get("words", [])
        if not isinstance(raw_words, list):
            raise ProcessingError(ALIGNMENT_CODE, "alignment segment words must be a list")

        tokens: list[dict[str, object]] = []
        for raw_word in raw_words:
            if not isinstance(raw_word, dict):
                raise ProcessingError(ALIGNMENT_CODE, "alignment word payload must be an object")
            raw_text = str(raw_word.get("word", "")).strip()
            text_value = raw_text
            if remove_punctuation:
                text_value = strip_punctuation_from_token(raw_text)
                if not text_value:
                    if is_punctuation_token(raw_text):
                        LOGGER.info("dictator.alignment.dropped_punctuation: %s", raw_text)
                        continue
                    raise ProcessingError(
                        ALIGNMENT_TIMESTAMP_CODE,
                        "aligned word text is empty after punctuation removal",
                    )
            start_value = coerce_timestamp(raw_word.get("start"))
            end_value = coerce_timestamp(raw_word.get("end"))
            if start_value is None or end_value is None:
                if is_punctuation_token(text_value):
                    if tokens:
                        previous_text = str(tokens[-1].get("text", "")).strip()
                        tokens[-1]["text"] = f"{previous_text} {text_value}".strip()
                    elif words:
                        previous = words[-1]
                        words[-1] = AlignedWord(
                            text=f"{previous.text} {text_value}".strip(),
                            start_seconds=previous.start_seconds,
                            end_seconds=previous.end_seconds,
                        )
                    else:
                        pending_prefix_tokens.append(text_value)
                    continue

            if pending_prefix_tokens:
                text_value = " ".join([*pending_prefix_tokens, text_value])
                pending_prefix_tokens.clear()
            tokens.append({"text": text_value, "start": start_value, "end": end_value})

        if not tokens:
            continue

        fallback = segment_bounds_from_tokens(tokens)
        if fallback is None:
            LOGGER.warning(
                "%s: inferring segment bounds for %d token(s)",
                ALIGNMENT_INFERRED_TIMESTAMPS_CODE,
                len(tokens),
            )
            fallback = default_segment_bounds(len(tokens), last_known_end)
        segment_start, segment_end = segment_bounds(segment, fallback)
        infer_missing_timings(tokens, segment_start, segment_end)
        for token in tokens:
            token_text = str(token.get("text", "")).strip()
            start_value = coerce_timestamp(token.get("start"))
            end_value = coerce_timestamp(token.get("end"))
            words.append(
                AlignedWord(
                    text=token_text,
                    start_seconds=cast(float, start_value),
                    end_seconds=cast(float, end_value),
                )
            )
            last_known_end = cast(float, end_value)

    if not words:
        raise ProcessingError(ALIGNMENT_CODE, "alignment produced no words")
    return tuple(words)


class WhisperXAlignmentBackend:
    """Reusable WhisperX aligner with per-language model caching."""

    def align(
        self,
        audio_path: Path,
        transcript_text: str,
        language: str,
        device: str = "auto",
        remove_punctuation: bool = False,
    ) -> tuple[AlignedWord, ...]:
        alignment_module, audio_module = load_whisperx_alignment_modules()
        resolved_device = resolve_device(device)
        audio = audio_module.load_audio(str(audio_path))
        audio_duration = float(len(audio)) / float(audio_module.SAMPLE_RATE)
        segments = [{"start": 0.0, "end": audio_duration, "text": transcript_text}]

        try:
            align_model, metadata = load_cached_alignment_model(
                language,
                resolved_device,
                alignment_module,
            )
            result = alignment_module.align(
                segments,
                align_model,
                metadata,
                audio,
                resolved_device,
                return_char_alignments=False,
            )
        except Exception as exc:
            error_message = str(exc)
            error_code = ALIGNMENT_TIMESTAMP_CODE if "missing timestamps" in error_message else ALIGNMENT_CODE
            raise ProcessingError(error_code, f"alignment failed: {error_message}") from exc

        return extract_aligned_words(
            result.get("segments", []),
            remove_punctuation=remove_punctuation,
        )
