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


def unity_heading_from_quaternion(q: np.ndarray) -> float:
    """Return Unity horizontal heading in radians for scalar-first quaternion."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return np.nan
    norm = np.linalg.norm(q)
    if norm <= np.finfo(float).eps:
        return np.nan
    w, x, y, z = q / norm
    forward_x = 2.0 * (x * z + w * y)
    forward_z = 1.0 - 2.0 * (x * x + y * y)
    if math.hypot(forward_x, forward_z) <= np.finfo(float).eps:
        return np.nan
    return math.atan2(forward_x, forward_z)

def average_quaternions_markley(quaternions: np.ndarray) -> np.ndarray:
    """Average scalar-first unit quaternions using Markley's eigenvector method."""
    q = np.asarray(quaternions, dtype=float)
    q = q[np.all(np.isfinite(q), axis=1)]
    if not len(q):
        return np.full(4, np.nan)
    norms = np.linalg.norm(q, axis=1)
    q = q[norms > np.finfo(float).eps]
    norms = norms[norms > np.finfo(float).eps]
    if not len(q):
        return np.full(4, np.nan)
    q = q / norms[:, None]
    reference = q[0]
    q[np.dot(q, reference) < 0] *= -1
    eigenvalues, eigenvectors = np.linalg.eigh(q.T @ q / len(q))
    avg = eigenvectors[:, np.argmax(eigenvalues)]
    return avg if avg[0] >= 0 else -avg

def parse_cell_heading(value: object) -> float:
    """Parse a matrix cell and return its within-bin horizontal heading."""
    if not isinstance(value, str) or value == "[]" or not value.strip():
        return np.nan
    # Matrix cells contain only numeric lists. Removing brackets and using
    # fromstring is substantially faster than literal_eval over ~1.5 M cells.
    raw = value.replace("[", "").replace("]", "")
    vals = np.fromstring(raw, sep=",")
    if not len(vals) or len(vals) % 4:
        return np.nan
    quats = vals.reshape(-1, 4)
    if len(quats) == 1:
        return unity_heading_from_quaternion(quats[0])
    return unity_heading_from_quaternion(average_quaternions_markley(quats))

def mean_in_window(time: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    mask = (time >= start) & (time < end) & np.isfinite(values)
    return float(np.mean(values[mask])) if np.any(mask) else np.nan

def smooth_nan(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that ignores missing samples."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values).astype(float)
    filled = np.where(np.isfinite(values), values, 0.0)
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(filled, kernel, mode="same")
    denominator = np.convolve(valid, kernel, mode="same")
    out = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return out

def sustained_recovery_onset(
    time: np.ndarray,
    smooth_heading: np.ndarray,
    minimum_index: int,
    baseline: float,
    fraction: float,
    end_time: float,
    sustained_samples: int,
) -> float:
    """Start of the final sustained recovery before the participant is passed.

    A short early rebound is visible in several trajectories. Treating its first
    threshold crossing as the recovery onset would be misleading because the head
    subsequently turns left again. A candidate onset must therefore (a) remain
    above the threshold for the initial sustained interval and (b) remain above it
    for at least 90% of the samples through ``end_time``. The earliest candidate
    satisfying both requirements is returned.
    """
    minimum = smooth_heading[minimum_index]
    amplitude = baseline - minimum
    if not np.isfinite(amplitude) or amplitude < math.radians(20.0):
        return np.nan
    target = minimum + fraction * amplitude
    in_range = (np.arange(len(time)) >= minimum_index) & (time <= end_time)
    above = in_range & np.isfinite(smooth_heading) & (smooth_heading >= target)
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    end_indices = np.flatnonzero(in_range & np.isfinite(smooth_heading))
    if not len(end_indices):
        return np.nan
    final_index = int(end_indices[-1])
    for start in starts:
        initial_end = start + sustained_samples
        if initial_end > len(above) or not np.all(above[start:initial_end]):
            continue
        tail = above[start : final_index + 1]
        if len(tail) and np.mean(tail) >= 0.90:
            return float(time[start])
    return np.nan

def event_delta(time: np.ndarray, heading: np.ndarray, event_time: float) -> tuple[float, float, float]:
    if not np.isfinite(event_time):
        return np.nan, np.nan, np.nan
    pre = mean_in_window(time, heading, event_time - 0.5, event_time)
    post = mean_in_window(time, heading, event_time, event_time + 0.5)
    return pre, post, post - pre

def participant_pass_time(row: pd.Series) -> float:
    # camera=0 is avatar-first/participant-second: participant is passed at cross_p1.
    # camera=1 is participant-first/avatar-second: participant is passed at cross_p2.
    return float(row["cross_p1_time_s"] if int(row["camera"]) == 0 else row["cross_p2_time_s"])

def trial_features(
    matrix_path: Path,
    mapping_row: pd.Series,
    curve_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    matrix = pd.read_csv(matrix_path)
    time = pd.to_numeric(matrix["Timestamp"], errors="coerce").to_numpy(float)
    participant_columns = sorted(
        [c for c in matrix.columns if PARTICIPANT_RE.match(c)],
        key=lambda c: int(PARTICIPANT_RE.match(c).group(1)),
    )
    video_id = str(mapping_row["video_id"])
    pass_time = participant_pass_time(mapping_row)
    order = "AF_PS" if int(mapping_row["camera"]) == 0 else "PF_AS"
    spacing_m = 2 * int(mapping_row["distPed"])
    base_metadata = {
        "video_id": video_id,
        "yielding": int(mapping_row["yielding"]),
        "eHMIOn": int(mapping_row["eHMIOn"]),
        "order": order,
        "spacing_m": spacing_m,
        "participant_pass_time_s": pass_time,
    }

    features: list[dict[str, object]] = []
    for participant_column in participant_columns:
        participant = int(PARTICIPANT_RE.match(participant_column).group(1))
        heading = np.array([parse_cell_heading(v) for v in matrix[participant_column].array], dtype=float)
        valid = np.isfinite(heading)
        if np.sum(valid) < 10:
            continue
        # Preserve temporal continuity if an angle approaches the +/-pi boundary.
        unwrapped = heading.copy()
        unwrapped[valid] = np.unwrap(unwrapped[valid])
        baseline = mean_in_window(time, unwrapped, 0.02, 0.30)
        heading_bc = unwrapped - baseline
        smooth = smooth_nan(heading_bc, window=11)  # approximately 0.22 s at 50 Hz

        search = (time >= 0.50) & (time <= pass_time - 0.20) & np.isfinite(smooth)
        if np.any(search):
            search_indices = np.flatnonzero(search)
            minimum_index = int(search_indices[np.argmin(smooth[search])])
            minimum_heading = float(smooth[minimum_index])
            minimum_rel_pass = float(time[minimum_index] - pass_time)
        else:
            minimum_index = 0
            minimum_heading = np.nan
            minimum_rel_pass = np.nan

        record: dict[str, object] = {
            **base_metadata,
            "participant": participant,
            "valid_samples": int(np.sum(valid)),
            "valid_fraction": float(np.mean(valid)),
            "baseline_heading_deg": math.degrees(baseline),
            "minimum_heading_deg": math.degrees(minimum_heading),
            "minimum_time_rel_pass_s": minimum_rel_pass,
        }

        for fraction in (0.10, 0.20, 0.30):
            onset = sustained_recovery_onset(
                time,
                smooth,
                minimum_index,
                baseline=0.0,
                fraction=fraction,
                end_time=pass_time,
                sustained_samples=10,
            )
            record[f"recovery_onset_{int(fraction * 100)}pct_rel_pass_s"] = (
                onset - pass_time if np.isfinite(onset) else np.nan
            )

        far = mean_in_window(time, heading_bc, pass_time - 3.0, pass_time - 2.0)
        immediate_pre = mean_in_window(time, heading_bc, pass_time - 0.5, pass_time)
        immediate_post = mean_in_window(time, heading_bc, pass_time, pass_time + 0.5)
        at_pass = mean_in_window(time, heading_bc, pass_time - 0.1, pass_time + 0.1)
        record.update(
            {
                "far_pre_heading_deg": math.degrees(far),
                "immediate_pre_heading_deg": math.degrees(immediate_pre),
                "immediate_post_heading_deg": math.degrees(immediate_post),
                "heading_at_pass_deg": math.degrees(at_pass),
                "far_to_pre_change_deg": math.degrees(immediate_pre - far),
                "passage_change_deg": math.degrees(immediate_post - immediate_pre),
                "recovered_by_pass_deg": math.degrees(at_pass - minimum_heading),
            }
        )

        for event_name, event_column in (
            ("braking", "yield_start_time_s"),
            ("stopping", "yield_stop_time_s"),
            ("resumption", "yield_resume_time_s"),
        ):
            event_time = pd.to_numeric(mapping_row.get(event_column, np.nan), errors="coerce")
            pre, post, delta = event_delta(time, heading_bc, float(event_time))
            record[f"{event_name}_pre_heading_deg"] = math.degrees(pre)
            record[f"{event_name}_post_heading_deg"] = math.degrees(post)
            record[f"{event_name}_change_deg"] = math.degrees(delta)

        features.append(record)

        # Match the existing yaw figures' configured temporal resolution (20 ms
        # in the supplied configuration), then aggregate participants downstream.
        rel = time - pass_time
        bin_width = HEAD_HEADING_TIME_BIN_S
        n_event_bins = int(round(6.0 / bin_width))
        # Use the same small boundary tolerance as the project's established
        # half-open binning helper. Without it, floating-point representations
        # such as 0.2999999999999999 can create alternating empty 20-ms bins.
        bins = np.floor((rel + 5.0 + bin_width * 1e-9) / bin_width).astype(int)
        time_decimals = max(2, int(math.ceil(-math.log10(bin_width))) + 1)
        for bin_index in range(n_event_bins):  # [-5, +1) s
            selected = (bins == bin_index) & np.isfinite(heading_bc)
            if not np.any(selected):
                continue
            curve_records.append(
                {
                    **base_metadata,
                    "participant": participant,
                    "time_rel_pass_s": round(
                        -5.0 + (bin_index + 0.5) * bin_width,
                        time_decimals,
                    ),
                    "heading_deg": math.degrees(float(np.mean(heading_bc[selected]))),
                }
            )
    return features
