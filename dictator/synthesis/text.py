"""Text normalization and chunking for XTTS."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional

from dictator.audio.constants import XTTS_BYTE_BUDGET

BYTE_BUDGET = XTTS_BYTE_BUDGET
CTRL_REMOVE = {c: None for c in range(32)} | {127: None}


def clean(text: str) -> str:
    """Unicode-normalise, strip controls, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text.translate(CTRL_REMOVE))
    return " ".join(text.split())


def split_into_sentences(text: str) -> List[str]:
    """Return sentences including their terminal punctuation."""
    return re.split(r"(?<=[.!?])\s+", text)


def fits_xtts(chunk: str, budget: int = BYTE_BUDGET) -> bool:
    """Return True when a chunk fits XTTS's UTF-8 byte budget."""
    return len(chunk.encode("utf-8")) <= budget


def trim_utf8(text: str, budget: int) -> str:
    """Trim text to at most the given UTF-8 byte budget."""
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    return encoded[:budget].decode("utf-8", errors="ignore")


def build_chunks(text: str, budget: int = BYTE_BUDGET) -> List[str]:
    """Greedy byte-budget splitter tuned for XTTS-v2."""
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    buffer = ""

    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if fits_xtts(candidate, budget):
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            buffer = sentence if fits_xtts(sentence, budget) else trim_utf8(sentence, budget)
    if buffer:
        chunks.append(buffer)

    merged: List[str] = []
    for chunk in chunks:
        if merged and len(chunk.encode("utf-8")) < 80 and fits_xtts(f"{merged[-1]} {chunk}", budget):
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)
    return merged


def parse_length(spec: Optional[str]) -> Optional[float]:
    """Parse strings like 90s, 2m, or 1.5h."""
    if not spec:
        return None
    match = re.fullmatch(r"\s*([\d.]+)\s*([smhSMH])\s*", spec)
    if not match:
        raise ValueError("--length must look like '90s', '2m' or '1.5h'")
    value, unit = match.groups()
    return float(value) * {"s": 1, "m": 60, "h": 3600}[unit.lower()]
