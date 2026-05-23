"""Configuration for speech synthesis engines."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

DEFAULT_QWEN3_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_QWEN3_TEXT_TOKEN_BUDGET = 192
QWEN3_FAST_ATTENTION_IMPLEMENTATION = "flash_attention_2"
DEFAULT_SILERO_RU_MODEL_URL = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"
DEFAULT_SILERO_RU_MODEL_SHA256 = "50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437"
DEFAULT_SILERO_RU_MODEL_FILENAME = "v5_5_ru.pt"
DEFAULT_SILERO_RU_DEFAULT_SPEAKER = "baya"
DEFAULT_SILERO_RU_SAMPLE_RATE = 24_000
DEFAULT_SILERO_RU_TEXT_CHAR_BUDGET = 900

QWEN3_MODEL_ID_ENV = "DICTATOR_QWEN3_TTS_MODEL_ID"
QWEN3_DTYPE_ENV = "DICTATOR_QWEN3_TTS_DTYPE"
QWEN3_TEXT_TOKEN_BUDGET_ENV = "DICTATOR_QWEN3_TTS_TEXT_TOKEN_BUDGET"
MODEL_ROOT_ENV = "DICTATOR_MODEL_ROOT"
SILERO_RU_MODEL_PATH_ENV = "DICTATOR_SILERO_RU_MODEL_PATH"
SILERO_RU_MODEL_URL_ENV = "DICTATOR_SILERO_RU_MODEL_URL"
SILERO_RU_MODEL_SHA256_ENV = "DICTATOR_SILERO_RU_MODEL_SHA256"
SILERO_RU_DEFAULT_SPEAKER_ENV = "DICTATOR_SILERO_RU_DEFAULT_SPEAKER"
SILERO_RU_SAMPLE_RATE_ENV = "DICTATOR_SILERO_RU_SAMPLE_RATE"
SILERO_RU_TEXT_CHAR_BUDGET_ENV = "DICTATOR_SILERO_RU_TEXT_CHAR_BUDGET"


def _positive_int_from_env(
    source: Mapping[str, str],
    env_name: str,
    default: int,
) -> int:
    raw_value = source.get(env_name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{env_name} must be a positive integer")
    return value


def _default_silero_ru_model_path(source: Mapping[str, str]) -> str:
    model_root = source.get(MODEL_ROOT_ENV, "").strip()
    if not model_root:
        return ""
    return str(os.path.join(model_root, "silero", DEFAULT_SILERO_RU_MODEL_FILENAME))


@dataclass(frozen=True)
class SynthesisConfig:
    """Runtime synthesis model configuration."""

    qwen3_model_id: str = DEFAULT_QWEN3_MODEL_ID
    qwen3_dtype: str = "auto"
    qwen3_text_token_budget: int = DEFAULT_QWEN3_TEXT_TOKEN_BUDGET
    silero_ru_model_path: str = ""
    silero_ru_model_url: str = DEFAULT_SILERO_RU_MODEL_URL
    silero_ru_model_sha256: str = DEFAULT_SILERO_RU_MODEL_SHA256
    silero_ru_default_speaker: str = DEFAULT_SILERO_RU_DEFAULT_SPEAKER
    silero_ru_sample_rate: int = DEFAULT_SILERO_RU_SAMPLE_RATE
    silero_ru_text_char_budget: int = DEFAULT_SILERO_RU_TEXT_CHAR_BUDGET

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SynthesisConfig":
        source = dict(os.environ if env is None else env)
        qwen3_dtype = source.get(QWEN3_DTYPE_ENV, "auto").strip().lower() or "auto"
        silero_ru_model_path = (
            source.get(SILERO_RU_MODEL_PATH_ENV, "").strip()
            or _default_silero_ru_model_path(source)
        )
        return cls(
            qwen3_model_id=source.get(QWEN3_MODEL_ID_ENV, DEFAULT_QWEN3_MODEL_ID).strip() or DEFAULT_QWEN3_MODEL_ID,
            qwen3_dtype=qwen3_dtype,
            qwen3_text_token_budget=_positive_int_from_env(
                source,
                QWEN3_TEXT_TOKEN_BUDGET_ENV,
                DEFAULT_QWEN3_TEXT_TOKEN_BUDGET,
            ),
            silero_ru_model_path=silero_ru_model_path,
            silero_ru_model_url=source.get(SILERO_RU_MODEL_URL_ENV, DEFAULT_SILERO_RU_MODEL_URL).strip()
            or DEFAULT_SILERO_RU_MODEL_URL,
            silero_ru_model_sha256=source.get(
                SILERO_RU_MODEL_SHA256_ENV,
                DEFAULT_SILERO_RU_MODEL_SHA256,
            ).strip()
            or DEFAULT_SILERO_RU_MODEL_SHA256,
            silero_ru_default_speaker=source.get(
                SILERO_RU_DEFAULT_SPEAKER_ENV,
                DEFAULT_SILERO_RU_DEFAULT_SPEAKER,
            ).strip()
            or DEFAULT_SILERO_RU_DEFAULT_SPEAKER,
            silero_ru_sample_rate=_positive_int_from_env(
                source,
                SILERO_RU_SAMPLE_RATE_ENV,
                DEFAULT_SILERO_RU_SAMPLE_RATE,
            ),
            silero_ru_text_char_budget=_positive_int_from_env(
                source,
                SILERO_RU_TEXT_CHAR_BUDGET_ENV,
                DEFAULT_SILERO_RU_TEXT_CHAR_BUDGET,
            ),
        )
