"""Text normalization helpers for speech synthesis."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence

CTRL_REMOVE = {c: None for c in range(32) if c not in (10, 13)} | {127: None}


def clean(text: str) -> str:
    """Unicode-normalise, strip controls, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text.translate(CTRL_REMOVE))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=[.!?])\n+", " ", text)
    text = text.replace("\n", "")
    return " ".join(text.split())


def split_into_sentences(text: str) -> List[str]:
    """Return sentences including their terminal punctuation."""
    return [
        sentence
        for sentence in re.split(r"(?<=[.!?])(?:\s+|(?=[A-ZА-ЯЁ0-9\"']))", text)
        if sentence
    ]


def join_synthesis_units(units: Sequence[str]) -> str:
    """Join sentence units with strong separators so the synthesiser sees boundaries."""
    normalized_units = tuple(unit.strip() for unit in units if unit.strip())
    if not normalized_units:
        raise ValueError("synthesis units cannot be empty")
    return "\n\n".join(normalized_units)


def parse_length(spec: Optional[str]) -> Optional[float]:
    """Parse strings like 90s, 2m, or 1.5h."""
    if not spec:
        return None
    match = re.fullmatch(r"\s*([\d.]+)\s*([smhSMH])\s*", spec)
    if not match:
        raise ValueError("--length must look like '90s', '2m' or '1.5h'")
    value, unit = match.groups()
    return float(value) * {"s": 1, "m": 60, "h": 3600}[unit.lower()]
