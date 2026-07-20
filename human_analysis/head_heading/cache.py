from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

import common
from custom_logger import CustomLogger
from .settings import *

logger = CustomLogger(__name__)


def _mapping_hash(mapping: pd.DataFrame) -> str:
    stable = mapping.sort_values("video_id").reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(stable, index=True).values.tobytes()).hexdigest()

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _input_signature(
    matrix_paths: list[Path],
    mapping: pd.DataFrame,
    source_cache_key: str | None = None,
) -> dict[str, object]:
    signature: dict[str, object] = {
        "cache_version": CACHE_VERSION,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mapping_sha256": _mapping_hash(mapping),
        "source_cache_key": source_cache_key,
    }
    if source_cache_key is None:
        # Standalone use has no validated processed-data pickle. Hash the
        # matrix contents so cache reuse never depends on file timestamps.
        signature["matrices"] = [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in matrix_paths
        ]
    else:
        # analysis.py restores these files directly from a validated pickle.
        # Its created_utc identifier changes whenever raw data are rebuilt.
        signature["matrices"] = [
            {"name": path.name, "size": path.stat().st_size}
            for path in matrix_paths
        ]
    return signature

def _load_manifest(path: Path) -> dict[str, object] | None:
    try:
        manifest = pd.read_csv(path)
        if len(manifest) != 1 or "signature_json" not in manifest.columns:
            return None
        return json.loads(str(manifest.loc[0, "signature_json"]))
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return None
