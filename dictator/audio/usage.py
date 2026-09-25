"""Exact quantities from decoded audio submitted for processing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputAudioUsage:
    """One input's mono samples, independent of model passes and output length."""

    sample_count: int
    sample_rate_hz: int

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or not 0 <= self.sample_count < 2**64:
            raise ValueError("input audio sample_count must be an unsigned 64-bit integer")
        if type(self.sample_rate_hz) is not int or not 0 < self.sample_rate_hz < 2**32:
            raise ValueError("input audio sample_rate_hz must be a positive unsigned 32-bit integer")

    def to_json_dict(self) -> dict[str, int]:
        return {"sample_count": self.sample_count, "sample_rate_hz": self.sample_rate_hz}
