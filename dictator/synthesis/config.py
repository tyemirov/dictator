"""Configuration for speech synthesis backends."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Mapping

DEFAULT_XTTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_QWEN3_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

XTTS_MODEL_ID_ENV = "DICTATOR_XTTS_MODEL_ID"
QWEN3_MODEL_ID_ENV = "DICTATOR_QWEN3_TTS_MODEL_ID"
QWEN3_ATTN_IMPLEMENTATION_ENV = "DICTATOR_QWEN3_TTS_ATTN_IMPLEMENTATION"
QWEN3_DTYPE_ENV = "DICTATOR_QWEN3_TTS_DTYPE"


@dataclass(frozen=True)
class SynthesisConfig:
    """Runtime synthesis model configuration."""

    xtts_model_id: str = DEFAULT_XTTS_MODEL_ID
    qwen3_model_id: str = DEFAULT_QWEN3_MODEL_ID
    qwen3_attn_implementation: str | None = None
    qwen3_dtype: str = "auto"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "SynthesisConfig":
        source = dict(os.environ if env is None else env)
        qwen3_attn_implementation = source.get(QWEN3_ATTN_IMPLEMENTATION_ENV, "").strip() or None
        qwen3_dtype = source.get(QWEN3_DTYPE_ENV, "auto").strip().lower() or "auto"
        return cls(
            xtts_model_id=source.get(XTTS_MODEL_ID_ENV, DEFAULT_XTTS_MODEL_ID).strip() or DEFAULT_XTTS_MODEL_ID,
            qwen3_model_id=source.get(QWEN3_MODEL_ID_ENV, DEFAULT_QWEN3_MODEL_ID).strip() or DEFAULT_QWEN3_MODEL_ID,
            qwen3_attn_implementation=qwen3_attn_implementation,
            qwen3_dtype=qwen3_dtype,
        )
