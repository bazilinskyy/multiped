from __future__ import annotations

import ast
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import expit
from scipy.stats import chi2
from scipy.stats import norm
from scipy.stats import shapiro
from scipy.stats import skew
from scipy.stats import t as student_t
from scipy.stats import ttest_rel

import common
from custom_logger import CustomLogger

import warnings


ADVANCED_STATS_SPECIFICATION = "reviewer_response_v4_bounded_common_window"

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import build_design_matrices, dmatrix
from statsmodels.tools.sm_exceptions import ConvergenceWarning


# Create a module level logger so every save and model fitting step is traceable.
logger = CustomLogger(__name__)


# Use a dedicated dataclass so statistical results can move cleanly between
# computations, tables, and plots without relying on positional tuples.

from .results import TOSTResult
from ..utils.parsing import parse_numeric_list
from ..utils.distance import distance_code_to_metres, distance_codes_to_metres, validate_distances_metres


class CoreStatsMixin:
    """Focused method group extracted without changing calculation logic."""

    def __init__(
        self,
        helper,
        mapping_df: pd.DataFrame,
        output_dir: Optional[str] = None,
    ) -> None:

        """Initialise the runner and normalise mapping table columns.

        Args:
            helper: Helper object that provides plot saving utilities and config access.
            mapping_df: DataFrame describing scenario metadata for each video.
            output_dir: Optional output directory override.

        Returns:
            None
        """
        # Store shared dependencies and project level configuration once during setup.
        # Store the shared helper and core input tables on the runner instance.
        self.helper = helper
        self.mapping_df = mapping_df.copy()
        self.output_dir = output_dir or common.get_configs("output")
        self.stats_dir = os.path.join(self.output_dir, "statistics")
        self.fig_dir = common.get_configs("figures")
        self.template = common.get_configs("plotly_template")
        self.font_size = common.get_configs("font_size")
        self.font_family = common.get_configs("font_family")
        self.common_window_event = self._config_or_default(
            "common_window_event", "participant_passage"
        )
        self.common_window_pre_s = float(
            self._config_or_default("common_window_pre_s", 5.0)
        )
        self.common_window_post_s = float(
            self._config_or_default("common_window_post_s", 0.0)
        )
        self.raw_trigger_sampling_hz = float(
            self._config_or_default("raw_trigger_sampling_hz", 50.0)
        )
        self.trigger_bin_seconds = float(
            self._config_or_default(
                "trigger_bin_interval_ms",
                self._config_or_default("kp_resolution", 100),
            )
        ) / 1000.0
        if self.common_window_event != "participant_passage":
            raise ValueError(
                "Only common_window_event='participant_passage' is currently supported."
            )
        if self.common_window_pre_s <= 0 or self.common_window_post_s != 0:
            raise ValueError(
                "common_window_pre_s must be positive and common_window_post_s "
                "must be 0 for the pre-passage analysis."
            )
        if self.raw_trigger_sampling_hz <= 0 or self.trigger_bin_seconds <= 0:
            raise ValueError("Raw trigger frequency and aggregation interval must be positive.")

        # Ensure output folders exist before any table or figure writing occurs.
        # Create any required output directories before later save operations.
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

        # Normalise identifier style early so later joins use consistent string keys.
        # Normalise identifier columns to consistent string types for later joins.
        if "video_id" in self.mapping_df.columns:
            self.mapping_df["video_id"] = self.mapping_df["video_id"].astype(str)
        if "condition_name" in self.mapping_df.columns:
            self.mapping_df["condition_name"] = self.mapping_df["condition_name"].astype(str)

        # Coerce mapping columns that should be numeric so downstream comparisons and
        # calculations behave predictably.
        # Coerce mapping metadata columns that should participate in numeric comparisons.
        for col in [
            "distPed",
            "yielding",
            "eHMIOn",
            "camera",
            "cross_p1_time_s",
            "cross_p2_time_s",
            "yield_start_time_s",
            "yield_stop_time_s",
            "yield_resume_time_s",
        ]:
            if col in self.mapping_df.columns:
                self.mapping_df[col] = pd.to_numeric(self.mapping_df[col], errors="coerce")

        if "distPed" in self.mapping_df.columns:
            self.mapping_df["distPed_m"] = self._distance_series_to_meters(self.mapping_df["distPed"])

    @staticmethod
    def _config_or_default(key: str, default):
        """Read one project setting while remaining compatible with older configs."""
        try:
            return common.get_configs(key)
        except (KeyError, TypeError, AttributeError):
            return default

    @staticmethod
    def _common_window_bounds(
        participant_passage_s: float,
        pre_s: float = 5.0,
        post_s: float = 0.0,
    ) -> Tuple[float, float]:
        """Return a fixed-duration window aligned to participant passage."""
        passage = float(participant_passage_s)
        pre = float(pre_s)
        post = float(post_s)
        if not np.isfinite(passage):
            raise ValueError("participant_passage_s must be finite.")
        if pre <= 0 or post < 0:
            raise ValueError("pre_s must be positive and post_s must be nonnegative.")
        return passage - pre, passage + post

    def _analysis_window_specs(
        self,
        map_row: pd.Series,
        participant_passage_s: float,
    ) -> Dict[str, Tuple[float, float]]:
        """Return primary and yielding-event half-open five-second windows."""
        start, end = self._common_window_bounds(
            participant_passage_s,
            pre_s=self.common_window_pre_s,
            post_s=self.common_window_post_s,
        )
        specs: Dict[str, Tuple[float, float]] = {
            "common_window": (start, end),
        }
        yielding_value = pd.to_numeric(map_row.get("yielding"), errors="coerce")
        if pd.notna(yielding_value) and int(yielding_value) == 1:
            event_definitions = {
                # Early deceleration and the onset of the animated conditional eHMI.
                "braking_onset_window": ("yield_start_time_s", "post"),
                # The five seconds immediately preceding the full stop.
                "stopping_window": ("yield_stop_time_s", "pre"),
                # The final deceleration/waiting interval before acceleration resumes.
                "resumption_window": ("yield_resume_time_s", "pre"),
            }
            for label, (column, direction) in event_definitions.items():
                event_time = pd.to_numeric(map_row.get(column), errors="coerce")
                if pd.isna(event_time):
                    continue
                event_time = float(event_time)
                if direction == "post":
                    specs[label] = (
                        event_time,
                        event_time + self.common_window_pre_s,
                    )
                else:
                    specs[label] = (
                        event_time - self.common_window_pre_s,
                        event_time,
                    )
        return specs

    @staticmethod
    def _distance_code_to_meters(value) -> float:
        """Convert one raw mapping code into physical metres."""
        return distance_code_to_metres(value)

    @classmethod
    def _distance_series_to_meters(cls, series: pd.Series) -> pd.Series:
        """Convert raw mapping distance codes into physical metres."""
        return distance_codes_to_metres(series)

    @staticmethod
    def _distance_meters_series(series: pd.Series) -> pd.Series:
        """Validate distances that are already expressed in metres."""
        return validate_distances_metres(series)

    @staticmethod
    def _extract_numeric_values(cell) -> List[float]:
        """Return finite numeric values from a list encoded CSV cell."""
        return parse_numeric_list(cell)

    @staticmethod
    def _lag1_binary_autocorrelation(states: Iterable[int]) -> float:
        """Return lag-1 correlation for a binary sequence, or NaN if constant."""
        values = np.asarray(list(states), dtype=float)
        if values.size < 3:
            return np.nan
        left = values[:-1]
        right = values[1:]
        if np.std(left) == 0 or np.std(right) == 0:
            return np.nan
        return float(np.corrcoef(left, right)[0, 1])

    @staticmethod
    def _holm_adjust(p_values: Iterable[float]) -> np.ndarray:
        """Return Holm family-wise adjusted p values, preserving input order."""
        values = np.asarray(list(p_values), dtype=float)
        adjusted = np.full(values.shape, np.nan, dtype=float)
        finite_indexes = np.flatnonzero(np.isfinite(values))
        if finite_indexes.size == 0:
            return adjusted
        ordered = finite_indexes[np.argsort(values[finite_indexes])]
        running = 0.0
        m = len(ordered)
        for rank, index in enumerate(ordered):
            candidate = min(1.0, (m - rank) * values[index])
            running = max(running, candidate)
            adjusted[index] = running
        return adjusted

    def _mapping_row_for_video(self, video_id: str) -> Optional[pd.Series]:

        """Fetch the first mapping row that matches a video identifier.

        Args:
            video_id: Video identifier to search for.

        Returns:
            The matching mapping row, or ``None`` when the video is unknown.
        """
        # The mapping table can contain multiple columns, but only the first matched row is used
        # because each video is expected to map to exactly one scenario definition.
        tmp = self.mapping_df.loc[self.mapping_df["video_id"] == str(video_id)]
        if tmp.empty:
            return None
        return tmp.iloc[0]

    @staticmethod
    def _cutoff_from_mapping(row: Optional[pd.Series]) -> Optional[float]:

        """Determine the analysis cutoff time for a scenario.

        The cutoff is taken from the crossing time that corresponds to the
        active camera view.

        Args:
            row: Optional scenario mapping row.

        Returns:
            The cutoff time in seconds, or ``None`` when it cannot be derived.
        """
        # Pick the cutoff that corresponds to the visible pedestrian in the current camera view.
        if row is None:
            return None
        # Read the camera flag that determines which crossing time applies.
        camera_raw = row.get("camera")
        if camera_raw is None:
            return None

        try:
            camera = int(float(camera_raw))
        except (TypeError, ValueError):
            return None
        # Use the first pedestrian crossing time for camera zero scenarios.
        if camera == 0:
            cross_p1 = row.get("cross_p1_time_s")
            if cross_p1 is not None and pd.notna(cross_p1):
                return float(cross_p1)
        # Use the second pedestrian crossing time for camera one scenarios.
        if camera == 1:
            cross_p2 = row.get("cross_p2_time_s")
            if cross_p2 is not None and pd.notna(cross_p2):
                return float(cross_p2)

        return None

    def _save_table(self, df: pd.DataFrame, filename: str) -> str:

        """Save a DataFrame into the statistics directory.

        Args:
            df: Table to save.
            filename: Output CSV file name.

        Returns:
            The fully resolved output path.
        """
        # Save to the statistics subdirectory rather than the generic output root to keep
        # generated tables grouped together.
        path = os.path.join(self.stats_dir, filename)
        df.to_csv(path, index=False)
        logger.info(f"Saved table: {path}")
        return path
