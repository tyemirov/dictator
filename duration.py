import argparse
import re


def parse_duration(value: str) -> float:
    """Parse human-friendly duration strings.

    Accepts plain numbers (seconds) or a number with a suffix:
    "s" for seconds or "m" for minutes. Examples:
    ``"60"``, ``"60s"`` and ``"1m"`` all return ``60.0``.
    """
    match = re.fullmatch(r"\s*(\d+(?:\.\d*)?)\s*([sm]?)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid duration: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "m":
        return amount * 60.0
    # default seconds for "s" or empty suffix
    return amount
