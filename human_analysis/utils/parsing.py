"""Safe parsing helpers for list encoded CSV cells."""

from __future__ import annotations

import ast
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


def parse_list_cell(value: Any) -> list[Any]:
    """Return one CSV cell as a list without raising on malformed input."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Real) and pd.isna(value):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "none"}:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        return [parsed]
    if isinstance(value, Real):
        return [float(value)]
    return []


def parse_numeric_list(value: Any) -> list[float]:
    """Return only finite numeric values from a list encoded cell."""
    result: list[float] = []
    for item in parse_list_cell(value):
        if isinstance(item, Real) and np.isfinite(item):
            result.append(float(item))
    return result
