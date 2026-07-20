"""Distance conversion and validation for the experiment mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_DISTANCE_CODES = frozenset({1.0, 2.0, 3.0, 4.0, 5.0})
DISTANCES_METRES = frozenset({2.0, 4.0, 6.0, 8.0, 10.0})


def distance_code_to_metres(value: object) -> float:
    """Convert one raw mapping code into physical metres."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or float(numeric) == 0.0:
        return np.nan
    numeric = float(numeric)
    if numeric not in RAW_DISTANCE_CODES:
        raise ValueError(f"Unexpected raw distPed code: {numeric}")
    return numeric * 2.0


def distance_codes_to_metres(series: pd.Series) -> pd.Series:
    """Convert raw mapping codes into physical metres once."""
    numeric = pd.to_numeric(series, errors="coerce")
    valid_codes = numeric.isin(RAW_DISTANCE_CODES)
    invalid = numeric.notna() & numeric.ne(0) & ~valid_codes
    if invalid.any():
        unexpected = sorted(numeric.loc[invalid].unique().tolist())
        raise ValueError(f"Unexpected raw distPed codes: {unexpected}")
    converted = numeric * 2.0
    converted.loc[numeric.eq(0)] = np.nan
    return converted


def validate_distances_metres(series: pd.Series) -> pd.Series:
    """Validate a series that is already expressed in metres."""
    numeric = pd.to_numeric(series, errors="coerce")
    invalid = numeric.notna() & numeric.ne(0) & ~numeric.isin(DISTANCES_METRES)
    if invalid.any():
        unexpected = sorted(numeric.loc[invalid].unique().tolist())
        raise ValueError(f"Unexpected distPed_m values: {unexpected}")
    numeric.loc[numeric.eq(0)] = np.nan
    return numeric
