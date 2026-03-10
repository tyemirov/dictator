"""Text normalization helpers for speech synthesis."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional
CTRL_REMOVE = {c: None for c in range(32)} | {127: None}


def clean(text: str) -> str:
    """Unicode-normalise, strip controls, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text.translate(CTRL_REMOVE))
    return " ".join(text.split())


def split_into_sentences(text: str) -> List[str]:
    """Return sentences including their terminal punctuation."""
    return re.split(r"(?<=[.!?])\s+", text)


def parse_length(spec: Optional[str]) -> Optional[float]:
    """Parse strings like 90s, 2m, or 1.5h."""
    if not spec:
        return None
    match = re.fullmatch(r"\s*([\d.]+)\s*([smhSMH])\s*", spec)
    if not match:
        raise ValueError("--length must look like '90s', '2m' or '1.5h'")
    value, unit = match.groups()
    return float(value) * {"s": 1, "m": 60, "h": 3600}[unit.lower()]
