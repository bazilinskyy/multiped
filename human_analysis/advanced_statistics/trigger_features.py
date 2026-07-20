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


class TriggerFeatureMixin:
    """Focused method group extracted without changing calculation logic."""

    def build_trigger_feature_table(
        self,
        column_name: str = "TriggerValueRight",
        threshold: float = 0.05,
        force: bool = False,
    ) -> pd.DataFrame:

        """Create participant by video trigger features from exported matrices.

        Each participant matrix is scanned per video. The method parses the
        trigger values stored in each timestamp bin, applies the scenario
        specific analysis cutoff, derives per participant summary features,
        and caches the resulting trial level table.

        Args:
            column_name: Name of the exported trigger matrix variant to load.
            threshold: Trigger values strictly greater than this threshold mark
                a timestamp bin as pressed. The default of 0.05 is used because
                the trigger was pressure-sensitive and light contact could
                produce small values.
            force: Whether to ignore a cached feature table and rebuild it.

        Returns:
            A DataFrame containing one derived feature row per participant
            and video pair.

        Raises:
            FileNotFoundError: If no participant trigger matrices exist.
            ValueError: If feature extraction yields no usable rows.
        """
        # Reuse a cached feature table when possible because parsing every participant matrix
        # can be expensive for large exports.
        # Define the cache location for the derived trigger feature table.
        primary_threshold = float(
            self._config_or_default("primary_trigger_threshold", threshold)
        )
        threshold_label = f"{int(round(float(threshold) * 100)):02d}pct"
        if np.isclose(float(threshold), primary_threshold):
            out_csv = os.path.join(self.stats_dir, "trigger_time_series_features.csv")
        else:
            sensitivity_dir = os.path.join(
                self.stats_dir, "common_window_threshold_sensitivity"
            )
            os.makedirs(sensitivity_dir, exist_ok=True)
            out_csv = os.path.join(
                sensitivity_dir,
                f"trigger_time_series_features_{threshold_label}.csv",
            )
        if os.path.isfile(out_csv) and not force:
            cached = pd.read_csv(out_csv)
            required_common_columns = {
                "unsafe_prop_common_window_pct",
                "unsafe_bins_common_window",
                "valid_bins_common_window",
                "common_window_start_s",
                "common_window_end_s",
                "common_window_pre_s",
                "common_window_post_s",
                "common_window_interval_closure",
                "trigger_bin_interval_s",
                "raw_trigger_sampling_hz",
            }
            if (
                "trigger_threshold" in cached.columns
                and required_common_columns.issubset(cached.columns)
            ):
                cached_thresholds = pd.to_numeric(cached["trigger_threshold"], errors="coerce").dropna().unique()
                cached_pre = pd.to_numeric(
                    cached["common_window_pre_s"], errors="coerce"
                ).dropna().unique()
                cached_post = pd.to_numeric(
                    cached["common_window_post_s"], errors="coerce"
                ).dropna().unique()
                if (
                    len(cached_thresholds) == 1
                    and np.isclose(cached_thresholds[0], threshold)
                    and len(cached_pre) == 1
                    and np.isclose(cached_pre[0], self.common_window_pre_s)
                    and len(cached_post) == 1
                    and np.isclose(cached_post[0], self.common_window_post_s)
                ):
                    logger.info(f"Loading cached trigger feature table: {out_csv}")
                    return cached
            logger.info(
                "Cached trigger feature table lacks the requested threshold or "
                "common-window definition; rebuilding."
            )

        # Discover all participant by video matrices that match the requested trigger column.
        # Build the file pattern used to discover exported participant matrices.
        pattern = os.path.join(self.output_dir, f"participant_{column_name}_video_*.csv")
        file_list = sorted(glob.glob(pattern))
        if not file_list:
            raise FileNotFoundError(
                "No participant trigger matrices were found. Run the trigger export or heatmap step first."
            )

        # Collect one derived record per participant and video pair.
        records: List[Dict[str, object]] = []

        # Local helper to coerce mapping values to floats without repeated boilerplate.
        def _as_float(value: object) -> float:
            numeric = pd.to_numeric(value, errors="coerce")
            return float(numeric) if pd.notna(numeric) else np.nan

        # Process each exported video matrix independently so feature rows can be traced
        # back to their source file.
        # Iterate over every discovered matrix file and derive features independently.
        for file_path in file_list:
            base = os.path.basename(file_path)
            match = re.search(r"(video_\d+)\.csv$", base, flags=re.IGNORECASE)
            if match is None:
                continue

        # Recover the canonical video identifier from the file name.
        # Extract the canonical video identifier from the file name.
            video_id = match.group(1)
            map_row = self._mapping_row_for_video(video_id)
            if map_row is None:
                logger.warning(f"Skipping {video_id}: missing mapping row")
                continue

        # Resolve the scenario specific cutoff before loading participant samples.
            cutoff = self._cutoff_from_mapping(map_row)
            common_window_start = np.nan
            common_window_end = np.nan
            window_specs: Dict[str, Tuple[float, float]] = {}
            if cutoff is not None and np.isfinite(cutoff):
                window_specs = self._analysis_window_specs(
                    map_row,
                    participant_passage_s=float(cutoff),
                )
                common_window_start, common_window_end = window_specs["common_window"]

        # Load the participant matrix that contains one timestamp column and one column per participant.
            df = pd.read_csv(file_path)
            if "Timestamp" not in df.columns:
                logger.warning(f"Skipping {video_id}: Timestamp column missing in {file_path}")
                continue

        # Clean timestamps so time based filtering and spacing calculations are reliable.
            df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

            if df.empty:
                logger.warning(f"Skipping {video_id}: no timestamp rows are available")
                continue

        # Derive the effective time grid from the retained timestamps.
            unique_ts = np.sort(df["Timestamp"].unique())
            if len(unique_ts) > 1:
                dt_seconds = float(np.nanmedian(np.diff(unique_ts)))
            else:
                dt_seconds = float(common.get_configs("kp_resolution")) / 1000.0

            # Whole-trial legacy features stop at participant passage, whereas
            # event-aligned sensitivity windows may extend later in the video.
            analysis_unique_ts = (
                unique_ts[unique_ts <= float(cutoff)]
                if cutoff is not None and np.isfinite(cutoff)
                else unique_ts
            )
            total_duration = float(len(analysis_unique_ts) * dt_seconds)

            # Every non timestamp column is treated as a participant specific time series.
            participant_cols = [col for col in df.columns if col != "Timestamp"]

        # Convert each participant column into a trial level summary record.
            for participant_col in participant_cols:
                pm = re.search(r"P(\d+)", str(participant_col))
                if pm is None:
                    continue

                participant = int(pm.group(1))

        # Initialise per participant containers for time series summaries.
                ts_used: List[float] = []
                bin_means: List[float] = []
                pressed_states: List[int] = []
                all_values: List[float] = []
                event_window_timestamps: Dict[str, List[float]] = {
                    label: [] for label in window_specs
                }
                event_window_pressed_states: Dict[str, List[int]] = {
                    label: [] for label in window_specs
                }
                event_window_raw_sample_counts: Dict[str, List[int]] = {
                    label: [] for label in window_specs
                }

        # Visit each timestamp bin to compute per bin trigger summaries.
                for ts_raw, cell_value in df[["Timestamp", participant_col]].itertuples(index=False, name=None):
                    ts = float(ts_raw)
                    values = self._extract_numeric_values(cell_value)

                    if values:
                        bin_mean = float(np.mean(values))
                        pressed = int(any(v > threshold for v in values))
                    else:
                        bin_mean = 0.0
                        pressed = 0

                    # Preserve whole-trial summaries only through participant
                    # passage. Do not truncate the source matrix because later
                    # vehicle-event windows may occur after that instant.
                    if cutoff is None or not np.isfinite(cutoff) or ts <= float(cutoff):
                        ts_used.append(ts)
                        bin_means.append(bin_mean)
                        pressed_states.append(pressed)
                        all_values.extend(values)
                    # Primary and event-aligned windows are left-closed and
                    # right-open. Only populated raw-sample bins are valid;
                    # missing bins are not silently classified as safe.
                    if values:
                        for label, (window_start, window_end) in window_specs.items():
                            if window_start <= ts < window_end:
                                event_window_timestamps[label].append(ts)
                                event_window_pressed_states[label].append(pressed)
                                event_window_raw_sample_counts[label].append(len(values))

        # Skip participants that still have no usable time bins after cleaning.
                if not ts_used:
                    continue

        # Collapse the bin level summaries into whole trial features.
                mean_raw = float(np.mean(bin_means)) if bin_means else np.nan
                peak_raw = float(np.max(bin_means)) if bin_means else np.nan
                auc_raw = float(np.sum(bin_means) * dt_seconds) if bin_means else np.nan
                unsafe_prop = float(np.mean(pressed_states)) if pressed_states else np.nan
                time_pressed = float(np.sum(pressed_states) * dt_seconds) if pressed_states else np.nan
                switch_count = int(np.sum(np.abs(np.diff(pressed_states)))) if len(pressed_states) > 1 else 0
                common_window_timestamps = event_window_timestamps.get(
                    "common_window", []
                )
                common_window_pressed_states = event_window_pressed_states.get(
                    "common_window", []
                )
                common_window_raw_counts = event_window_raw_sample_counts.get(
                    "common_window", []
                )
                valid_bins_common = int(len(common_window_pressed_states))
                unsafe_bins_common = int(np.sum(common_window_pressed_states))
                unsafe_prop_common = (
                    float(unsafe_bins_common / valid_bins_common)
                    if valid_bins_common > 0
                    else np.nan
                )
                common_window_duration = (
                    float(common_window_end - common_window_start)
                    if (
                        valid_bins_common > 0
                        and np.isfinite(common_window_start)
                        and np.isfinite(common_window_end)
                    )
                    else np.nan
                )
                expected_bins_common = (
                    int(round(common_window_duration / dt_seconds))
                    if np.isfinite(common_window_duration) and dt_seconds > 0
                    else 0
                )
                common_window_lag1 = self._lag1_binary_autocorrelation(
                    common_window_pressed_states
                )

        # Locate the first press event and the first subsequent release if present.
                first_press_idx = next((idx for idx, state in enumerate(pressed_states) if state == 1), None)
                first_press_latency = np.nan
                first_release_latency = np.nan

                if first_press_idx is not None:
                    first_press_latency = float(ts_used[first_press_idx])

                    first_release_idx = next(
                        (
                            idx
                            for idx in range(first_press_idx + 1, len(pressed_states))
                            if pressed_states[idx] == 0
                        ),
                        None,
                    )
                    if first_release_idx is not None:
                        first_release_latency = float(ts_used[first_release_idx])

                # Describe the spread of the raw trigger values that fed the summaries.
                value_sd = float(np.std(all_values, ddof=1)) if len(all_values) > 1 else np.nan
                n_samples = int(len(all_values))

                # Pull scenario level metadata from the mapping row so the feature table can be
                # merged directly onto trial level analyses later.
                dist_ped_code = _as_float(map_row.get("distPed"))
                dist_ped_m = _as_float(map_row.get("distPed_m"))
                if pd.isna(dist_ped_m):
                    dist_ped_m = self._distance_code_to_meters(dist_ped_code)
                elif dist_ped_m not in {2.0, 4.0, 6.0, 8.0, 10.0}:
                    raise ValueError(f"Unexpected distPed_m value: {dist_ped_m}")

                yielding = _as_float(map_row.get("yielding"))
                ehmi_on = _as_float(map_row.get("eHMIOn"))
                camera = _as_float(map_row.get("camera"))
                analysis_cutoff_s = float(cutoff) if cutoff is not None else np.nan

                # Package feature values and scenario metadata into one row dictionary.
                record: Dict[str, object] = {
                    "participant": participant,
                    "video_id": video_id,
                    "condition_name": str(map_row.get("condition_name", video_id)),
                    "yielding": yielding,
                    "eHMIOn": ehmi_on,
                    "camera": camera,
                    "distPed": dist_ped_m,
                    "distPed_m": dist_ped_m,
                    "distPed_code": dist_ped_code,
                    "analysis_cutoff_s": analysis_cutoff_s,
                    "dt_seconds": dt_seconds,
                    "analysis_duration_s": total_duration,
                    "n_bins": int(len(ts_used)),
                    "n_trigger_samples": n_samples,
                    "trigger_threshold": float(threshold),
                    "mean_trigger_raw": mean_raw,
                    "mean_trigger_pct": mean_raw * 100.0 if pd.notna(mean_raw) else np.nan,
                    "peak_trigger_raw": peak_raw,
                    "peak_trigger_pct": peak_raw * 100.0 if pd.notna(peak_raw) else np.nan,
                    "auc_trigger_raw_s": auc_raw,
                    "auc_trigger_pct_s": auc_raw * 100.0 if pd.notna(auc_raw) else np.nan,
                    "unsafe_prop": unsafe_prop,
                    "unsafe_prop_pct": unsafe_prop * 100.0 if pd.notna(unsafe_prop) else np.nan,
                    "unsafe_prop_common_window": unsafe_prop_common,
                    "unsafe_prop_common_window_pct": (
                        unsafe_prop_common * 100.0
                        if pd.notna(unsafe_prop_common)
                        else np.nan
                    ),
                    "unsafe_bins_common_window": unsafe_bins_common,
                    "valid_bins_common_window": valid_bins_common,
                    "expected_bins_common_window": expected_bins_common,
                    "common_window_event": self.common_window_event,
                    "common_window_pre_s": self.common_window_pre_s,
                    "common_window_post_s": self.common_window_post_s,
                    "common_window_start_s": common_window_start,
                    "common_window_end_s": common_window_end,
                    "common_window_observed_start_s": (
                        min(common_window_timestamps)
                        if common_window_timestamps
                        else np.nan
                    ),
                    "common_window_observed_end_s": (
                        max(common_window_timestamps)
                        if common_window_timestamps
                        else np.nan
                    ),
                    "common_window_duration_s": common_window_duration,
                    "common_window_interval_closure": "[start,end)",
                    "common_window_bin_duration_s": (
                        float(valid_bins_common * dt_seconds)
                        if valid_bins_common > 0
                        else np.nan
                    ),
                    "trigger_bin_interval_s": dt_seconds,
                    "raw_trigger_sampling_hz": self.raw_trigger_sampling_hz,
                    "expected_raw_samples_per_bin": (
                        self.raw_trigger_sampling_hz * dt_seconds
                    ),
                    "common_window_raw_samples_per_bin_mean": (
                        float(np.mean(common_window_raw_counts))
                        if common_window_raw_counts
                        else np.nan
                    ),
                    "common_window_raw_samples_per_bin_min": (
                        int(np.min(common_window_raw_counts))
                        if common_window_raw_counts
                        else np.nan
                    ),
                    "common_window_raw_samples_per_bin_max": (
                        int(np.max(common_window_raw_counts))
                        if common_window_raw_counts
                        else np.nan
                    ),
                    "common_window_lag1_autocorrelation": common_window_lag1,
                    "trigger_bin_state_rule": "any(raw_value > threshold)",
                    "time_pressed_s": time_pressed,
                    "switch_count": switch_count,
                    "first_press_latency_s": first_press_latency,
                    "first_release_latency_s": first_release_latency,
                    "trigger_value_sd": value_sd,
                }

                for window_label in (
                    "braking_onset_window",
                    "stopping_window",
                    "resumption_window",
                ):
                    prefix = window_label
                    states = event_window_pressed_states.get(window_label, [])
                    timestamps = event_window_timestamps.get(window_label, [])
                    raw_counts = event_window_raw_sample_counts.get(window_label, [])
                    bounds = window_specs.get(window_label)
                    valid_count = int(len(states)) if bounds is not None else 0
                    unsafe_count = int(np.sum(states)) if bounds is not None else 0
                    record[f"{prefix}_start_s"] = (
                        float(bounds[0]) if bounds is not None else np.nan
                    )
                    record[f"{prefix}_end_s"] = (
                        float(bounds[1]) if bounds is not None else np.nan
                    )
                    record[f"{prefix}_valid_bins"] = (
                        valid_count if bounds is not None else np.nan
                    )
                    expected_event_bins = int(round(5.0 / dt_seconds))
                    record[f"{prefix}_expected_bins"] = (
                        expected_event_bins if bounds is not None else np.nan
                    )
                    record[f"{prefix}_complete"] = (
                        bool(valid_count == expected_event_bins)
                        if bounds is not None
                        else np.nan
                    )
                    record[f"{prefix}_unsafe_bins"] = (
                        unsafe_count if bounds is not None else np.nan
                    )
                    record[f"{prefix}_unsafe_prop"] = (
                        float(unsafe_count / valid_count)
                        if bounds is not None and valid_count > 0
                        else np.nan
                    )
                    record[f"{prefix}_unsafe_pct"] = (
                        100.0 * float(unsafe_count / valid_count)
                        if bounds is not None and valid_count > 0
                        else np.nan
                    )
                    record[f"{prefix}_observed_start_s"] = (
                        min(timestamps) if timestamps else np.nan
                    )
                    record[f"{prefix}_observed_end_s"] = (
                        max(timestamps) if timestamps else np.nan
                    )
                    record[f"{prefix}_raw_samples_per_bin_mean"] = (
                        float(np.mean(raw_counts)) if raw_counts else np.nan
                    )
                    record[f"{prefix}_lag1_autocorrelation"] = (
                        self._lag1_binary_autocorrelation(states)
                        if states
                        else np.nan
                    )
                records.append(record)

        # Convert all accumulated row dictionaries into a single DataFrame.
        feature_df = pd.DataFrame.from_records(records)
        if feature_df.empty:
            raise ValueError("Trigger feature extraction produced an empty table.")

        # Apply a deterministic ordering before saving the feature table.
        feature_df = feature_df.sort_values(["participant", "video_id"]).reset_index(drop=True)

        # Every populated five-second primary window must contain exactly the
        # configured number of 100-ms bins. Failing here prevents silent use of
        # inclusive endpoints or missing raw samples.
        invalid_primary = feature_df.loc[
            feature_df["valid_bins_common_window"]
            != feature_df["expected_bins_common_window"]
        ]
        if not invalid_primary.empty:
            example = invalid_primary[
                [
                    "participant",
                    "video_id",
                    "valid_bins_common_window",
                    "expected_bins_common_window",
                ]
            ].head(10)
            raise ValueError(
                "Primary half-open common windows do not all contain the "
                "expected number of populated bins. Examples:\n"
                + example.to_string(index=False)
            )
        feature_df.to_csv(out_csv, index=False)

        logger.info(
            f"Built trigger feature table with {len(feature_df)} rows across "
            f"{feature_df['participant'].nunique()} participants and "
            f"{feature_df['video_id'].nunique()} videos"
        )
        logger.info(f"Saved table: {out_csv}")
        return feature_df
