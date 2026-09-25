"""Scripted vehicle event schedule used to define analysis windows.

The AV followed the same scripted trajectory in every trial of a given vehicle
behaviour, so vehicle event times are constant across trials. The per-condition
times in ``mapping.csv`` come from a simulator log whose values scatter for two
technical reasons:

* standstill and drive-off were detected from a speed estimate that the
  simulator updated only every 0.5 s (``CarMovement.speedSampleInterval``), so
  the logged times lag the true events by up to about 0.5 s and the logged
  standstill durations are exactly 2.50 s or 3.00 s;
* braking onset and pedestrian passages were detected once per 20-ms physics
  step, so identical events can be logged one to three steps apart.

The yielding trajectory is two spline segments: the approach segment ends
(standstill) 11 s after trial onset and the drive-off segment starts 14 s after
onset (``CarMovement.DriveCar``: ``firstAni = 11``, ``secDel = 14``). Braking
onset and passage times are the modal logged values: one value per vehicle
behaviour for the first roadside position and one value per vehicle behaviour
and inter-pedestrian distance for the second position.

The Unity script has since been changed so that future experiments do not
have this scatter: the AV brakes at a constant 2.4 m/s^2 and every event is
logged at its exact time (``CarMovement.UpdateConstantDecelYield``).
"""

from __future__ import annotations

import hashlib
from typing import Dict

import pandas as pd

# Scripted yielding events (s after trial onset), identical in all yielding trials.
YIELDING_BRAKING_ONSET_S = 5.76
YIELDING_STANDSTILL_S = 11.00
YIELDING_DRIVE_OFF_S = 14.00

EVENT_TIME_COLUMNS = (
    "yield_start_time_s",
    "yield_stop_time_s",
    "yield_resume_time_s",
    "yield_end_time_s",
    "cross_p1_time_s",
    "cross_p2_time_s",
)


def _modal_value(values: pd.Series) -> float:
    """Most frequent value; ties resolve to the smallest value."""
    counts = values.round(2).value_counts()
    top = counts[counts == counts.max()].index
    return float(min(top))


def apply_scripted_vehicle_events(mapping: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``mapping`` with constant vehicle event times.

    Main-trial rows (``video_*``) receive one value per event and geometry:
    yielding events are set to the scripted schedule, and passage times of the
    first (P2) and second (P1) roadside positions are set to the modal logged
    value for each vehicle behaviour and inter-pedestrian distance. Practice
    trials are left unchanged.
    """
    current = mapping.copy()
    main = current["video_id"].astype(str).str.startswith("video_")
    yielding = pd.to_numeric(current["yielding"], errors="coerce").eq(1)
    for column in EVENT_TIME_COLUMNS:
        if column in current.columns:
            current[column] = pd.to_numeric(current[column], errors="coerce")

    scripted: Dict[str, float] = {
        "yield_start_time_s": YIELDING_BRAKING_ONSET_S,
        "yield_stop_time_s": YIELDING_STANDSTILL_S,
        "yield_resume_time_s": YIELDING_DRIVE_OFF_S,
    }
    for column, value in scripted.items():
        if column in current.columns:
            current.loc[main & yielding, column] = value

    # The first position (P2) is at the same place in every trial, so its
    # passage and the end of yielding depend only on the vehicle behaviour.
    # The second position (P1) lies d metres downstream, so its passage time
    # also depends on the inter-pedestrian distance.
    groupings = {
        "yield_end_time_s": ["yielding"],
        "cross_p2_time_s": ["yielding"],
        "cross_p1_time_s": ["yielding", "distPed"],
    }
    for column, keys in groupings.items():
        if column not in current.columns:
            continue
        for _, index in current.loc[main].groupby(keys).groups.items():
            values = current.loc[index, column].dropna()
            if not values.empty:
                current.loc[index, column] = _modal_value(values)
    return current


def vehicle_event_schedule_key(mapping: pd.DataFrame) -> str:
    """Short hash of the event times, used to invalidate cached statistics."""
    columns = ["video_id"] + [c for c in EVENT_TIME_COLUMNS if c in mapping.columns]
    table = mapping[columns].sort_values("video_id").round(3)
    return hashlib.sha1(table.to_csv(index=False).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "YIELDING_BRAKING_ONSET_S",
    "YIELDING_STANDSTILL_S",
    "YIELDING_DRIVE_OFF_S",
    "apply_scripted_vehicle_events",
    "vehicle_event_schedule_key",
]
