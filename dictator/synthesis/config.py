"""Configuration for Qwen3-TTS synthesis."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

DEFAULT_QWEN3_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_QWEN3_TEXT_TOKEN_BUDGET = 192
QWEN3_FAST_ATTENTION_IMPLEMENTATION = "flash_attention_2"

QWEN3_MODEL_ID_ENV = "DICTATOR_QWEN3_TTS_MODEL_ID"
QWEN3_DTYPE_ENV = "DICTATOR_QWEN3_TTS_DTYPE"
QWEN3_TEXT_TOKEN_BUDGET_ENV = "DICTATOR_QWEN3_TTS_TEXT_TOKEN_BUDGET"


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


@dataclass(frozen=True)
class SynthesisConfig:
    """Runtime synthesis model configuration."""

    qwen3_model_id: str = DEFAULT_QWEN3_MODEL_ID
    qwen3_dtype: str = "auto"
    qwen3_text_token_budget: int = DEFAULT_QWEN3_TEXT_TOKEN_BUDGET

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SynthesisConfig":
        source = dict(os.environ if env is None else env)
        qwen3_dtype = source.get(QWEN3_DTYPE_ENV, "auto").strip().lower() or "auto"
        return cls(
            qwen3_model_id=source.get(QWEN3_MODEL_ID_ENV, DEFAULT_QWEN3_MODEL_ID).strip() or DEFAULT_QWEN3_MODEL_ID,
            qwen3_dtype=qwen3_dtype,
            qwen3_text_token_budget=_positive_int_from_env(
                source,
                QWEN3_TEXT_TOKEN_BUDGET_ENV,
                DEFAULT_QWEN3_TEXT_TOKEN_BUDGET,
            ),
        )
