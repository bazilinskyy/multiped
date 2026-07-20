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
@dataclass
class TOSTResult:

    """Container for paired TOST and paired t test results.

    Attributes:
        label: Human readable label for the comparison that was run.
        n: Number of paired observations used in the analysis.
        mean_diff: Mean of the paired differences.
        sd_diff: Sample standard deviation of the paired differences.
        se_diff: Standard error of the paired differences.
        ci90_low: Lower bound of the 90 percent confidence interval.
        ci90_high: Upper bound of the 90 percent confidence interval.
        ci95_low: Lower bound of the 95 percent confidence interval.
        ci95_high: Upper bound of the 95 percent confidence interval.
        t_lower: Test statistic for the lower bound one sided TOST test.
        p_lower: P value for the lower bound one sided TOST test.
        t_upper: Test statistic for the upper bound one sided TOST test.
        p_upper: P value for the upper bound one sided TOST test.
        p_tost: Final TOST p value, defined as the larger one sided p value.
        equivalent: Whether both one sided tests passed at the requested alpha.
        t_paired: Test statistic from the conventional paired t test against zero.
        p_paired: P value from the conventional paired t test against zero.
        margin_low: Lower equivalence margin supplied by the caller.
        margin_high: Upper equivalence margin supplied by the caller.
    """
    label: str
    n: int
    mean_diff: float
    sd_diff: float
    se_diff: float
    ci90_low: float
    ci90_high: float
    ci95_low: float
    ci95_high: float
    t_lower: float
    p_lower: float
    t_upper: float
    p_upper: float
    p_tost: float
    equivalent: bool
    t_paired: float
    p_paired: float
    margin_low: float
    margin_high: float


class AdvancedStatsRunner:

    """Run advanced statistical analyses and figure generation.

    This runner extends the existing helper based workflow. It consumes
    participant by video trigger matrices that have already been exported,
    derives trial level trigger features, runs equivalence testing and
    mixed effect style models, and writes publication ready tables and figures.

    Attributes:
        helper: Project specific helper that exposes save_plotly and config helpers.
        mapping_df: Scenario level mapping table used to enrich each video.
        output_dir: Root output directory for tables and cached files.
        data_root: Root data directory for raw project inputs.
        stats_dir: Directory used for generated statistics tables.
        template: Plotly template name from the project config.
        font_size: Default figure font size from the project config.
        font_family: Default figure font family from the project config.
    """

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
        """Convert one raw mapping code in 1..5 to physical metres."""
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return np.nan
        numeric = float(numeric)
        if numeric == 0:
            return np.nan
        if numeric not in {1.0, 2.0, 3.0, 4.0, 5.0}:
            raise ValueError(f"Unexpected raw distPed code: {numeric}")
        return numeric * 2.0

    @classmethod
    def _distance_series_to_meters(cls, series: pd.Series) -> pd.Series:
        """Convert a Series of raw mapping codes to physical metres once."""
        numeric = pd.to_numeric(series, errors="coerce")
        mask_code = numeric.isin([1, 2, 3, 4, 5])  # type: ignore
        invalid = numeric.notna() & (numeric != 0) & ~mask_code
        if invalid.any():
            unexpected = sorted(numeric.loc[invalid].unique().tolist())
            raise ValueError(f"Unexpected raw distPed codes: {unexpected}")
        mapped = numeric * 2.0
        mapped.loc[numeric == 0] = np.nan
        return mapped

    @staticmethod
    def _distance_meters_series(series: pd.Series) -> pd.Series:
        """Validate a Series that is already expressed in physical metres."""
        numeric = pd.to_numeric(series, errors="coerce")
        allowed = {2.0, 4.0, 6.0, 8.0, 10.0}
        invalid = numeric.notna() & (numeric != 0) & ~numeric.isin(allowed)
        if invalid.any():
            unexpected = sorted(numeric.loc[invalid].unique().tolist())
            raise ValueError(f"Unexpected distPed_m values: {unexpected}")
        numeric.loc[numeric == 0] = np.nan
        return numeric

    @staticmethod
    def _extract_numeric_values(cell) -> List[float]:

        """Parse a cell that should contain a list of numeric trigger values.

        Args:
            cell: Raw value taken from a participant trigger matrix cell.

        Returns:
            A list of finite numeric values. Invalid, missing, or malformed
            inputs return an empty list instead of raising.
        """
        # Treat missing values and malformed cells as empty bins rather than failing the run.
        # Treat missing trigger cells as empty observations.
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return []
        # Some exports store a Python list as text, so literal_eval is used to recover it.
        try:
            # Recover list literals that were serialised to text in CSV files.
            parsed = ast.literal_eval(cell) if isinstance(cell, str) else cell
        except Exception:
            return []
        # Ignore anything that is not a list because the feature extraction logic expects
        # a sequence of per bin values.
        # Reject malformed parsed values that do not produce a list.
        if not isinstance(parsed, list):
            return []
        # Keep only finite numeric entries to protect the summary statistics from invalid values.
        # Accumulate only the finite numeric values that survive validation.
        out: List[float] = []
        for item in parsed:
            if isinstance(item, (int, float)) and np.isfinite(item):
                out.append(float(item))
        return out

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

    @staticmethod
    def _paired_tost(diff: Iterable[float], low_eq: float, high_eq: float,
                     alpha: float = 0.05) -> TOSTResult:

        """Run a paired two one sided tests procedure.

        Args:
            diff: Iterable of paired differences.
            low_eq: Lower equivalence bound.
            high_eq: Upper equivalence bound.
            alpha: Significance level for each one sided test.

        Returns:
            A populated ``TOSTResult`` instance containing sample statistics,
            confidence intervals, one sided TOST results, and a conventional
            paired t test against zero.
        """
        # Convert the incoming paired differences into a numeric NumPy array.
        diff_arr = np.asarray(list(diff), dtype=float)
        diff_arr = diff_arr[np.isfinite(diff_arr)]
        n = int(len(diff_arr))

        # Return early when there are too few observations for paired inference.
        if n < 2:
            return TOSTResult(
                label="",
                n=n,
                mean_diff=np.nan,
                sd_diff=np.nan,
                se_diff=np.nan,
                ci90_low=np.nan,
                ci90_high=np.nan,
                ci95_low=np.nan,
                ci95_high=np.nan,
                t_lower=np.nan,
                p_lower=np.nan,
                t_upper=np.nan,
                p_upper=np.nan,
                p_tost=np.nan,
                equivalent=False,
                t_paired=np.nan,
                p_paired=np.nan,
                margin_low=float(low_eq),
                margin_high=float(high_eq),
            )

        # Compute the core paired sample summary statistics.
        mean_diff = float(np.mean(diff_arr))
        sd_diff = float(np.std(diff_arr, ddof=1))
        se_diff = float(sd_diff / np.sqrt(n))
        dfree = n - 1

        # Handle degenerate standard errors without raising an exception.
        if not np.isfinite(se_diff) or se_diff == 0.0:
            return TOSTResult(
                label="",
                n=n,
                mean_diff=mean_diff,
                sd_diff=sd_diff,
                se_diff=se_diff,
                ci90_low=np.nan,
                ci90_high=np.nan,
                ci95_low=np.nan,
                ci95_high=np.nan,
                t_lower=np.nan,
                p_lower=np.nan,
                t_upper=np.nan,
                p_upper=np.nan,
                p_tost=np.nan,
                equivalent=False,
                t_paired=np.nan,
                p_paired=np.nan,
                margin_low=float(low_eq),
                margin_high=float(high_eq),
            )

        # Compute the critical values needed for confidence intervals.
        crit90 = float(student_t.ppf(1.0 - alpha, df=dfree))
        crit95 = float(student_t.ppf(1.0 - alpha / 2.0, df=dfree))

        # Report both 90 percent and 95 percent intervals because the 90 percent interval is
        # directly relevant for equivalence testing.
        ci90_low = float(mean_diff - crit90 * se_diff)
        ci90_high = float(mean_diff + crit90 * se_diff)
        ci95_low = float(mean_diff - crit95 * se_diff)
        ci95_high = float(mean_diff + crit95 * se_diff)

        # Form the two one sided test statistics for the equivalence bounds.
        t_lower = float((mean_diff - low_eq) / se_diff)
        p_lower = float(1.0 - student_t.cdf(t_lower, df=dfree))

        t_upper = float((mean_diff - high_eq) / se_diff)
        p_upper = float(student_t.cdf(t_upper, df=dfree))

        # Combine the two one sided p values into the final TOST decision quantity.
        p_tost = float(max(p_lower, p_upper))
        equivalent = bool((p_lower < alpha) and (p_upper < alpha))

        # Also compute the conventional paired t test against zero for reference.
        paired_result = ttest_rel(diff_arr, np.zeros_like(diff_arr))
        t_paired = float(paired_result.statistic)  # pyright: ignore[reportAttributeAccessIssue]
        p_paired = float(paired_result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]

        return TOSTResult(
            label="",
            n=n,
            mean_diff=mean_diff,
            sd_diff=sd_diff,
            se_diff=se_diff,
            ci90_low=ci90_low,
            ci90_high=ci90_high,
            ci95_low=ci95_low,
            ci95_high=ci95_high,
            t_lower=t_lower,
            p_lower=p_lower,
            t_upper=t_upper,
            p_upper=p_upper,
            p_tost=p_tost,
            equivalent=equivalent,
            t_paired=t_paired,
            p_paired=p_paired,
            margin_low=float(low_eq),
            margin_high=float(high_eq),
        )

    def run_equivalence_tests(self, trial_df: pd.DataFrame, outcome: str = "crossing_risk",
                              low_distances_m: Tuple[int, int] = (2, 4),
                              high_distances_m: Tuple[int, int] = (8, 10), equivalence_margin: float = 5.0,
                              alpha: float = 0.05) -> pd.DataFrame:

        """Run paired TOST comparisons for near versus far distances.

        The comparison is calculated once across all contexts and again within
        each yielding by eHMI by camera combination.

        Args:
            trial_df: Trial level DataFrame with scenario metadata.
            outcome: Column to compare between near and far distances.
            low_distances_m: Actual distance values in metres to treat as near.
            high_distances_m: Actual distance values in metres to treat as far.
            equivalence_margin: Symmetric equivalence margin in outcome units.
            alpha: Significance level for the one sided tests.

        Returns:
            A DataFrame with one TOST summary row per comparison context.

        Raises:
            ValueError: If no valid data remains after filtering.
        """
        df = trial_df.copy()
        if "distPed_m" not in df.columns:
            if "distPed" not in df.columns:
                raise ValueError("Neither 'distPed_m' nor 'distPed' is available in the trial table.")
            df["distPed_m"] = self._distance_series_to_meters(df["distPed"])
        else:
            # distPed_m is already physical distance. Never pass it through the
            # raw-code converter because 2 and 4 are valid in both domains.
            df["distPed_m"] = self._distance_meters_series(df["distPed_m"])

        df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
        df = df.dropna(subset=["participant", "distPed_m", outcome, "yielding", "eHMIOn", "camera"])
        if df.empty:
            raise ValueError(f"No valid rows are available for equivalence testing of {outcome}.")

        df["distance_band"] = pd.Series(pd.NA, index=df.index, dtype="object")
        df.loc[df["distPed_m"].isin(low_distances_m), "distance_band"] = "near"
        df.loc[df["distPed_m"].isin(high_distances_m), "distance_band"] = "far"
        df = df.loc[df["distance_band"].isin(["near", "far"])].copy()
        if df.empty:
            raise ValueError("No near/far rows remained after distance band selection.")

        def _yield_label(val: object) -> str:
            return "Yielding" if int(val) == 1 else "Non-yielding"  # pyright: ignore[reportArgumentType]

        def _ehmi_label(val: object) -> str:
            return "eHMI" if int(val) == 1 else "No eHMI"  # pyright: ignore[reportArgumentType]

        def _order_label(val: object) -> str:
            return (
                "Participant first / avatar second"
                if int(val) == 1
                else "Avatar first / participant second"
            )  # type: ignore

        result_records: List[Dict[str, object]] = []

        overall = (
            df.groupby(["participant", "distance_band"], as_index=False)[outcome]
            .mean()
            .pivot(index="participant", columns="distance_band", values=outcome)
            .dropna(subset=["near", "far"])
        )
        if not overall.empty:
            tost = self._paired_tost(
                overall["near"] - overall["far"],
                low_eq=-equivalence_margin,
                high_eq=equivalence_margin,
                alpha=alpha,
            )
            result_records.append({
                **tost.__dict__,
                "label": "Overall",
                "context": "Overall",
                "display_label": "Overall",
                "yielding": np.nan,
                "eHMIOn": np.nan,
                "camera": np.nan,
            })
            logger.info(
                f"Overall TOST for {outcome}: mean diff = {tost.mean_diff:.3f}, "
                f"90% CI [{tost.ci90_low:.3f}, {tost.ci90_high:.3f}], p_tost = {tost.p_tost:.4g}, "
                f"equivalent = {tost.equivalent}"
            )

        ctx_cols = ["yielding", "eHMIOn", "camera"]
        for ctx, ctx_df in df.groupby(ctx_cols):
            pivot = (
                ctx_df.groupby(["participant", "distance_band"], as_index=False)[outcome]
                .mean()
                .pivot(index="participant", columns="distance_band", values=outcome)
                .dropna(subset=["near", "far"])
            )
            if pivot.empty:
                continue

            tost = self._paired_tost(
                pivot["near"] - pivot["far"],
                low_eq=-equivalence_margin,
                high_eq=equivalence_margin,
                alpha=alpha,
            )
            label = f"Y{int(ctx[0])} H{int(ctx[1])} C{int(ctx[2])}"
            display_label = f"{_yield_label(ctx[0])}, {_ehmi_label(ctx[1])}, {_order_label(ctx[2])}"
            result_records.append({
                **tost.__dict__,
                "label": label,
                "context": label,
                "display_label": display_label,
                "yielding": int(ctx[0]),
                "eHMIOn": int(ctx[1]),
                "camera": int(ctx[2]),
            })
            logger.info(
                f"Context {label} TOST for {outcome}: mean diff = {tost.mean_diff:.3f}, "
                f"90% CI [{tost.ci90_low:.3f}, {tost.ci90_high:.3f}], p_tost = {tost.p_tost:.4g}, "
                f"equivalent = {tost.equivalent}"
            )

        results_df = pd.DataFrame(result_records)
        if results_df.empty:
            raise ValueError("No equivalence results could be computed.")

        self._save_table(results_df, f"equivalence_near_vs_far_{outcome}.csv")
        # Build a faceted equivalence figure.
        # Rows separate relative order, columns separate eHMI, and each panel shows
        # two yielding states. This keeps labels short and publication friendly.
        fig = make_subplots(
            rows=3,
            cols=2,
            specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]],
            subplot_titles=[
                "Overall",
                "Avatar first / participant second | No eHMI",
                "Avatar first / participant second | eHMI",
                "Participant first / avatar second | No eHMI",
                "Participant first / avatar second | eHMI",
            ],
            shared_xaxes=True,
            shared_yaxes=False,
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
            row_heights=[0.20, 0.40, 0.40],
        )

        plot_df = results_df.copy()
        finite_bounds = pd.concat(
            [plot_df["ci90_low"], plot_df["ci90_high"], plot_df["mean_diff"]],
            ignore_index=True,
        )
        finite_bounds = pd.to_numeric(finite_bounds, errors="coerce")
        finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
        if finite_bounds.empty:
            x_limit = float(equivalence_margin + 1.0)
        else:
            x_limit = float(max(equivalence_margin, np.abs(finite_bounds).max()))
            x_limit += max(1.0, 0.08 * x_limit)

        panel_positions = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2)]
        for row_idx, col_idx in panel_positions:
            fig.add_vrect(
                x0=-equivalence_margin,
                x1=equivalence_margin,
                fillcolor="rgba(50, 50, 50, 0.08)",
                line_width=0,
                row=row_idx,  # pyright: ignore[reportArgumentType]
                col=col_idx,  # pyright: ignore[reportArgumentType]
            )
            fig.add_vline(x=0, line_dash="dash", line_color="black", row=row_idx, col=col_idx)  # type: ignore
            fig.update_xaxes(range=[-x_limit, x_limit], row=row_idx, col=col_idx)

        overall_df = plot_df.loc[plot_df["label"] == "Overall"]
        if not overall_df.empty:
            row = overall_df.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[row["mean_diff"]],
                    y=["Overall"],
                    mode="markers",
                    marker=dict(size=12, symbol="diamond-open" if not row["equivalent"] else "diamond"),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[row["ci90_high"] - row["mean_diff"]],
                        arrayminus=[row["mean_diff"] - row["ci90_low"]],
                        thickness=1.8,
                        width=0,
                    ),
                    showlegend=False,
                    hovertemplate=(
                        "<b>Overall</b><br>Near minus far: %{x:.2f}<br>"
                        f"TOST p: {row['p_tost']:.4g}<br>"
                        f"Equivalent: {row['equivalent']}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=["Overall"],
                row=1,
                col=1,
            )

        context_df = plot_df.loc[plot_df["label"] != "Overall"].copy()
        context_df["yield_label"] = context_df["yielding"].map(_yield_label)
        context_df["panel_row"] = context_df["camera"].map({0: 2, 1: 3})
        context_df["panel_col"] = context_df["eHMIOn"].map({0: 1, 1: 2})

        for _, row in context_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["mean_diff"]],
                    y=[row["yield_label"]],
                    mode="markers",
                    marker=dict(size=11, symbol="circle-open" if not row["equivalent"] else "circle"),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[row["ci90_high"] - row["mean_diff"]],
                        arrayminus=[row["mean_diff"] - row["ci90_low"]],
                        thickness=1.6,
                        width=0,
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{row['display_label']}</b><br>Near minus far: %{{x:.2f}}<br>"
                        f"TOST p: {row['p_tost']:.4g}<br>"
                        f"Equivalent: {row['equivalent']}<extra></extra>"
                    ),
                ),
                row=int(row["panel_row"]),
                col=int(row["panel_col"]),
            )

        for r in [2, 3]:
            for c in [1, 2]:
                fig.update_yaxes(
                    categoryorder="array",
                    categoryarray=["Non-yielding", "Yielding"],
                    row=r,
                    col=c,
                )

        fig.update_layout(
            template=self.template,
            title="",
            font=dict(family=self.font_family, size=self.font_size + 2),
            margin=dict(l=0, r=0, t=0, b=0),
            height=900,
        )
        fig.update_annotations(font=dict(family=self.font_family, size=self.font_size + 4))

        # Only the left panels need a y axis title.
        fig.update_yaxes(title_text="", title_font=dict(family=self.font_family, size=self.font_size + 10),
                         tickfont=dict(family=self.font_family, size=self.font_size + 6), row=1, col=1)
        fig.update_yaxes(title_text="", title_font=dict(family=self.font_family, size=self.font_size + 8),
                         tickfont=dict(family=self.font_family, size=self.font_size + 6), row=2, col=1)
        fig.update_yaxes(title_text="", title_font=dict(family=self.font_family, size=self.font_size + 8),
                         tickfont=dict(family=self.font_family, size=self.font_size + 6), row=3, col=1)
        fig.update_yaxes(tickfont=dict(family=self.font_family, size=self.font_size + 6), row=2, col=2)
        fig.update_yaxes(tickfont=dict(family=self.font_family, size=self.font_size + 6), row=3, col=2)

        for r in [1, 2, 3]:
            for c in [1, 2]:
                fig.update_xaxes(
                    title_font=dict(family=self.font_family, size=self.font_size + 10),
                    tickfont=dict(family=self.font_family, size=self.font_size + 6),
                    automargin=True,
                    row=r,
                    col=c,
                )
        pretty_outcome = self._pretty_outcome_label(outcome)
        fig.update_xaxes(title_text=f"Near minus far difference in {pretty_outcome}", row=3, col=1)
        fig.update_xaxes(title_text=f"Near minus far difference in {pretty_outcome}", row=3, col=2)

        self.helper.save_plotly(
            fig=fig,
            name=f"equivalence_near_vs_far_{outcome}",
            width=1300,
            height=900,
            save_final=True,
            open_browser=True,
        )
        logger.info(f"Saved figure set for: equivalence_near_vs_far_{outcome}")
        return results_df

    @staticmethod
    def _pretty_outcome_label(outcome: str) -> str:

        """Convert raw outcome column names into human readable labels."""
        mapping = {
            "crossing_risk": "crossing risk",
            "unsafe_prop_pct": "unsafe time (%)",
            "first_press_latency_s": "first press latency (s)",
            "peak_trigger_pct": "peak trigger (0–100)",
            "auc_trigger_pct_s": "Trigger AUC",
            "switch_count": "switch count",
        }
        return mapping.get(outcome, outcome.replace("_", " "))

    @staticmethod
    def _pretty_term(term: str) -> str:

        """Convert raw model term names into human readable labels.

        Args:
            term: Raw term emitted by statsmodels.

        Returns:
            A friendlier display label for tables and figures.
        """
        # Map raw statsmodels term names to display labels that read well in tables and figures.
        # Translate raw model term names into cleaner display labels.
        mapping = {
            "Intercept": "Intercept",
            "C(yielding)[T.1]": "Yielding",
            "C(eHMIOn)[T.1]": "eHMI",
            "C(camera)[T.1]": "Participant-first / avatar-second order",
            "distPed_m": "Distance (m)",
            "within_score": "Within participant",
            "between_score": "Between participant",
            "Group Var": "Random intercept variance",
            "C(yielding)[T.1]:C(eHMIOn)[T.1]": "Yielding × eHMI",
            "C(yielding)[T.1]:C(camera)[T.1]": "Yielding × relative pedestrian order",
            "C(eHMIOn)[T.1]:C(camera)[T.1]": "eHMI × relative pedestrian order",
        }
        return mapping.get(term, term)

    @staticmethod
    def _collect_convergence_messages(caught_warnings: List[warnings.WarningMessage]) -> List[str]:

        """Extract statsmodels convergence warning messages from a warning list."""
        messages: List[str] = []
        for warning_obj in caught_warnings:
            if issubclass(warning_obj.category, ConvergenceWarning):
                messages.append(str(warning_obj.message))
        return messages

    @staticmethod
    def _has_hard_convergence_failure(fit, warning_messages: List[str]) -> bool:

        """Decide whether a fitted mixed model should be treated as failed."""
        converged = bool(getattr(fit, "converged", False))
        if not converged:
            return True

        lowered = [msg.lower() for msg in warning_messages]
        hard_markers = [
            "failed to converge",
            "optimization failed",
            "gradient optimization failed",
            "check mle_retvals",
        ]
        return any(marker in msg for marker in hard_markers for msg in lowered)

    def _fit_model(self, df: pd.DataFrame, formula: str, group_col: str = "participant",
                   re_formula: Optional[str] = None):

        """Fit a mixed effects model with the requested random effects structure.

        Args:
            df: Modelling DataFrame.
            formula: Statsmodels formula string.
            group_col: Column that defines grouping for random effects.
            re_formula: Optional random effects formula.

        Returns:
            The fitted statsmodels result object.

        Raises:
            RuntimeError: If statsmodels is unavailable or the model does not converge.
        """
        # Guard modelling code when statsmodels is unavailable in the runtime.
        if smf is None:
            raise RuntimeError("statsmodels is not available in this environment.")

        model = smf.mixedlm(formula, df, groups=df[group_col], re_formula=re_formula)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            fit = model.fit(
                reml=False,
                method=["lbfgs", "bfgs", "cg"],
                maxiter=500,
                disp=False,
            )

        warning_messages = self._collect_convergence_messages(caught)
        if self._has_hard_convergence_failure(fit, warning_messages):
            raise RuntimeError(
                "MixedLM did not converge "
                f"(re_formula={re_formula!r}, warnings={warning_messages}, "
                f"mle_retvals={getattr(fit, 'mle_retvals', None)})"
            )

        if warning_messages:
            logger.warning(
                f"MixedLM converged with warnings for formula '{formula}' "
                f"(re_formula={re_formula}): {warning_messages}"
            )

        return fit

    def _fit_model_with_fallbacks(
        self,
        df: pd.DataFrame,
        formula: str,
        group_col: str = "participant",
        re_formula: Optional[str] = None,
        re_formula_fallbacks: Optional[Iterable[str]] = None,
    ):

        """Fit a model using progressively simpler fallbacks.

        The method first attempts the requested mixed model, then a random
        intercept only version, and finally a clustered OLS model when mixed
        models fail to converge.

        Args:
            df: Modelling DataFrame.
            formula: Statsmodels formula string.
            group_col: Column that defines grouping for clustered inference.
            re_formula: Optional random effects formula.

        Returns:
            A tuple of ``(fit, model_name)``. Either value may be ``None`` if
            every fitting strategy fails.
        """
        # Try more expressive models first and fall back to simpler ones when convergence fails.
        if smf is None:
            logger.warning(f"statsmodels is unavailable. Skipping model: {formula}")
            return None, None

        # Prepare the ordered list of model fitting strategies. Random-slope
        # structures must be attempted before the random-intercept fallback in
        # this fully within-participant experiment.
        attempts: List[Tuple[str, Optional[str]]] = []
        requested_re_formulas: List[str] = []
        if re_formula is not None:
            requested_re_formulas.append(str(re_formula))
        if re_formula_fallbacks is not None:
            for candidate in re_formula_fallbacks:
                candidate = str(candidate)
                if candidate not in requested_re_formulas:
                    requested_re_formulas.append(candidate)

        for index, candidate in enumerate(requested_re_formulas, start=1):
            attempts.append((f"mixed_random_slopes_{index}", candidate))
        attempts.append(("mixed_random_intercept_fallback", None))
        # Try each modelling strategy until one converges successfully.
        for model_name, current_re_formula in attempts:
            try:
                fit = self._fit_model(df=df, formula=formula, group_col=group_col, re_formula=current_re_formula)
                return fit, model_name
            except Exception as exc:
                logger.warning(
                    f"Model attempt failed ({model_name}, re_formula={current_re_formula}): {exc}"
                )

        # If all mixed models fail, fall back to clustered OLS so the analysis can still produce
        # coefficient estimates with participant level dependence accounted for.
        try:
            fit = smf.ols(formula, data=df).fit(
                cov_type="cluster",
                cov_kwds={"groups": df[group_col]},
            )
            logger.warning(
                f"Falling back to clustered OLS for formula '{formula}' after mixed model failures."
            )
            return fit, "ols_clustered"
        except Exception as exc:
            logger.error(f"All model attempts failed for formula '{formula}': {exc}")
            return None, None

    def _coef_frame(self, fit, outcome: str, model_name: str,
                    keep_terms: Optional[Iterable[str]] = None) -> pd.DataFrame:

        """Convert a fitted model into a tidy coefficient table.

        Args:
            fit: Fitted statsmodels result object.
            outcome: Outcome label to attach to every coefficient row.
            model_name: Name of the fitting strategy that succeeded.
            keep_terms: Optional iterable of coefficient names to keep.

        Returns:
            A tidy DataFrame with estimates, standard errors, p values,
            confidence intervals, and pretty labels.
        """
        # Pull the fitted coefficients and uncertainty estimates from the model result.
        params = fit.params
        pvalues = fit.pvalues
        conf = fit.conf_int()
        bse = fit.bse
        # Reshape coefficient vectors into a tidy tabular format.
        coef_df = pd.DataFrame(
            {
                "outcome": outcome,
                "model": model_name,
                "term": params.index,
                "estimate": params.values,
                "std_error": bse.values,
                "p_value": pvalues.values,
                "ci_lower": conf.iloc[:, 0].values,
                "ci_upper": conf.iloc[:, 1].values,
            }
        )
        # Attach human readable term labels after extracting the raw parameter names.
        coef_df["pretty_term"] = coef_df["term"].map(self._pretty_term)
        # Optionally retain only the subset of coefficients relevant for reporting.
        if keep_terms is not None:
            keep_terms = set(keep_terms)
            coef_df = coef_df.loc[coef_df["term"].isin(keep_terms)].copy()
        return coef_df.reset_index(drop=True)  # type: ignore

    @staticmethod
    def _participant_cell_summary_with_ci(
        trial_df: pd.DataFrame,
        outcome: str,
        group_cols: Optional[List[str]] = None,
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """Summarise participant-level condition values with t-based intervals."""
        group_cols = group_cols or ["distPed_m", "yielding", "eHMIOn", "camera"]
        required = ["participant", outcome] + list(group_cols)
        current = trial_df.copy()
        for col in required:
            if col not in current.columns:
                raise ValueError(f"Missing column required for uncertainty summary: {col}")
        current[outcome] = pd.to_numeric(current[outcome], errors="coerce")
        current = current.dropna(subset=required)
        if current.empty:
            raise ValueError(f"No valid participant-level values for outcome '{outcome}'.")

        summary = (
            current.groupby(group_cols, as_index=False)[outcome]
            .agg(mean="mean", sd="std", n="count")
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
        summary["se"] = summary["sd"] / np.sqrt(summary["n"])
        alpha = 1.0 - float(confidence)
        summary["critical_t"] = summary["n"].map(
            lambda n: student_t.ppf(1.0 - alpha / 2.0, int(n) - 1)
            if int(n) > 1
            else np.nan
        )
        summary["ci_half_width"] = summary["critical_t"] * summary["se"]
        summary["ci_lower"] = summary["mean"] - summary["ci_half_width"]
        summary["ci_upper"] = summary["mean"] + summary["ci_half_width"]
        summary["outcome"] = outcome
        summary["confidence_level"] = float(confidence)
        return summary

    @staticmethod
    def _fixed_effect_components(fit):
        """Extract fixed-effect estimates and their covariance from a fitted model."""
        if hasattr(fit, "fe_params"):
            beta = fit.fe_params.copy()
        else:
            beta = fit.params.copy()
        names = list(beta.index)
        covariance = fit.cov_params().loc[names, names]
        return beta, covariance

    def _estimated_marginal_means_and_contrasts(
        self,
        fit,
        distances: Iterable[float] = (2, 4, 6, 8, 10),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate model-based marginal means and revised-analysis contrasts."""
        try:
            from patsy import build_design_matrices
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("patsy is required for estimated marginal means.") from exc

        grid = pd.DataFrame.from_records(
            [
                {
                    "yielding": yielding,
                    "eHMIOn": ehmi,
                    "camera": camera,
                    "distPed_m": float(distance),
                    "distPed_centered": float(distance) - 6.0,
                    "trial_number_centered": 0.0,
                }
                for yielding in (0, 1)
                for ehmi in (0, 1)
                for camera in (0, 1)
                for distance in distances
            ]
        )
        design_info = fit.model.data.design_info
        design = build_design_matrices(
            [design_info], grid, return_type="dataframe"
        )[0]
        beta, covariance = self._fixed_effect_components(fit)
        design = design.loc[:, beta.index]

        linear_functions: Dict[Tuple[int, int, int], np.ndarray] = {}
        emm_records: List[Dict[str, object]] = []
        for keys, indexes in grid.groupby(["yielding", "eHMIOn", "camera"]).groups.items():
            yielding, ehmi, camera = (int(value) for value in keys)
            linear_function = design.loc[list(indexes)].mean(axis=0).to_numpy(dtype=float)
            estimate = float(linear_function @ beta.to_numpy(dtype=float))
            variance = float(linear_function @ covariance.to_numpy(dtype=float) @ linear_function)
            se = float(np.sqrt(max(variance, 0.0)))
            linear_functions[(yielding, ehmi, camera)] = linear_function
            emm_records.append(
                {
                    "yielding": yielding,
                    "eHMIOn": ehmi,
                    "camera": camera,
                    "estimate": estimate,
                    "std_error": se,
                    "ci_lower": estimate - 1.96 * se,
                    "ci_upper": estimate + 1.96 * se,
                    "averaged_over_distances_m": "2,4,6,8,10",
                }
            )

        beta_values = beta.to_numpy(dtype=float)
        covariance_values = covariance.to_numpy(dtype=float)
        contrast_records: List[Dict[str, object]] = []

        def add_contrast(
            contrast: str,
            reference_key: Tuple[int, int, int],
            comparison_key: Tuple[int, int, int],
            conditioning: str,
        ) -> None:
            linear_function = (
                linear_functions[comparison_key] - linear_functions[reference_key]
            )
            estimate = float(linear_function @ beta_values)
            variance = float(linear_function @ covariance_values @ linear_function)
            se = float(np.sqrt(max(variance, 0.0)))
            z_value = estimate / se if se > 0 else np.nan
            p_value = 2.0 * norm.sf(abs(z_value)) if np.isfinite(z_value) else np.nan
            contrast_records.append(
                {
                    "contrast": contrast,
                    "conditioning": conditioning,
                    "reference": str(reference_key),
                    "comparison": str(comparison_key),
                    "estimate": estimate,
                    "std_error": se,
                    "z_value": z_value,
                    "p_value": p_value,
                    "ci_lower": estimate - 1.96 * se,
                    "ci_upper": estimate + 1.96 * se,
                }
            )

        for yielding in (0, 1):
            for camera in (0, 1):
                add_contrast(
                    "eHMI minus no eHMI",
                    (yielding, 0, camera),
                    (yielding, 1, camera),
                    f"yielding={yielding}; camera={camera}",
                )
        for yielding in (0, 1):
            for ehmi in (0, 1):
                add_contrast(
                    "participant-first minus avatar-first order",
                    (yielding, ehmi, 0),
                    (yielding, ehmi, 1),
                    f"yielding={yielding}; eHMIOn={ehmi}",
                )
        for ehmi in (0, 1):
            for camera in (0, 1):
                add_contrast(
                    "yielding minus non-yielding",
                    (0, ehmi, camera),
                    (1, ehmi, camera),
                    f"eHMIOn={ehmi}; camera={camera}",
                )

        return pd.DataFrame(emm_records), pd.DataFrame(contrast_records)

    @staticmethod
    def _model_diagnostic_row(
        fit,
        model_name: str,
        formula: str,
        random_effects_formula: Optional[str],
        current: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return convergence and residual diagnostics for one fitted model."""
        residuals = np.asarray(getattr(fit, "resid", []), dtype=float)
        residuals = residuals[np.isfinite(residuals)]
        fitted = np.asarray(getattr(fit, "fittedvalues", []), dtype=float)
        fitted = fitted[np.isfinite(fitted)]
        shapiro_p = np.nan
        if 3 <= residuals.size <= 5000:
            shapiro_p = float(shapiro(residuals).pvalue)
        return pd.DataFrame(
            [
                {
                    "model": model_name,
                    "formula": formula,
                    "random_effects_formula": random_effects_formula or "1",
                    "converged": bool(getattr(fit, "converged", True)),
                    "n_observations": int(getattr(fit, "nobs", len(current))),
                    "n_participants": int(current["participant"].nunique()),
                    "log_likelihood": float(getattr(fit, "llf", np.nan)),
                    "aic": float(getattr(fit, "aic", np.nan)),
                    "bic": float(getattr(fit, "bic", np.nan)),
                    "residual_mean": float(np.mean(residuals)) if residuals.size else np.nan,
                    "residual_sd": float(np.std(residuals, ddof=1)) if residuals.size > 1 else np.nan,
                    "residual_skew": float(skew(residuals)) if residuals.size > 2 else np.nan,
                    "residual_shapiro_p": shapiro_p,
                    "fitted_min": float(np.min(fitted)) if fitted.size else np.nan,
                    "fitted_max": float(np.max(fitted)) if fitted.size else np.nan,
                }
            ]
        )

    def run_improved_repeated_measures_model(
        self,
        trial_df: pd.DataFrame,
        outcome: str,
        analysis_label: str,
    ) -> Dict[str, pd.DataFrame]:
        """Fit categorical-distance models with participant random slopes."""
        requested_thresholds = (
            pd.to_numeric(trial_df["trigger_threshold"], errors="coerce")
            .dropna()
            .unique()
            if "trigger_threshold" in trial_df.columns
            else np.array([])
        )
        output_paths = {
            "coefficients": os.path.join(
                self.stats_dir, f"{analysis_label}_mixed_model_coefficients.csv"
            ),
            "diagnostics": os.path.join(
                self.stats_dir, f"{analysis_label}_mixed_model_diagnostics.csv"
            ),
            "estimated_marginal_means": os.path.join(
                self.stats_dir, f"{analysis_label}_estimated_marginal_means.csv"
            ),
            "simple_contrasts": os.path.join(
                self.stats_dir, f"{analysis_label}_simple_contrasts.csv"
            ),
        }
        if (
            getattr(self.helper, "reuse_statistical_results", False)
            and all(os.path.isfile(path) for path in output_paths.values())
        ):
            cached = {
                name: pd.read_csv(path) for name, path in output_paths.items()
            }
            settings_table = cached["coefficients"]
            cached_pre = (
                pd.to_numeric(
                    settings_table["common_window_pre_s"], errors="coerce"
                ).dropna().unique()
                if "common_window_pre_s" in settings_table.columns
                else np.array([])
            )
            cached_post = (
                pd.to_numeric(
                    settings_table["common_window_post_s"], errors="coerce"
                ).dropna().unique()
                if "common_window_post_s" in settings_table.columns
                else np.array([])
            )
            cached_thresholds = (
                pd.to_numeric(
                    settings_table["trigger_threshold"], errors="coerce"
                ).dropna().unique()
                if "trigger_threshold" in settings_table.columns
                else np.array([])
            )
            if (
                len(cached_pre) == 1
                and np.isclose(cached_pre[0], self.common_window_pre_s)
                and len(cached_post) == 1
                and np.isclose(cached_post[0], self.common_window_post_s)
                and (
                    len(requested_thresholds) == 0
                    or (
                        len(cached_thresholds) == 1
                        and np.isclose(
                            cached_thresholds[0], requested_thresholds[0]
                        )
                    )
                )
            ):
                logger.info(
                    f"Reused cached improved model tables for {analysis_label}."
                )
                return cached
        required = [
            "participant",
            outcome,
            "yielding",
            "eHMIOn",
            "camera",
            "distPed_m",
            "trial_number",
        ]
        missing = [column for column in required if column not in trial_df.columns]
        if missing:
            raise ValueError(
                f"Missing columns for improved model '{analysis_label}': {missing}"
            )
        current = trial_df.copy()
        for column in required:
            current[column] = pd.to_numeric(current[column], errors="coerce")
        current = current.dropna(subset=required).copy()
        current["distPed_centered"] = current["distPed_m"] - 6.0
        current["trial_number_centered"] = (
            current["trial_number"]
            - current.groupby("participant")["trial_number"].transform("mean")
        )
        if current.empty:
            raise ValueError(f"No valid rows for improved model '{analysis_label}'.")

        formula = (
            f"{outcome} ~ C(yielding) * C(eHMIOn) * C(camera) + "
            "C(distPed_m) * (C(yielding) + C(camera)) + "
            "trial_number_centered + I(trial_number_centered ** 2)"
        )
        random_formulas = [
            "~C(yielding) + C(eHMIOn) + C(camera) + distPed_centered + trial_number_centered",
            "~C(yielding) + C(eHMIOn) + C(camera)",
            "~C(yielding) + C(eHMIOn)",
            "~distPed_centered",
        ]
        fit, model_name = self._fit_model_with_fallbacks(
            current,
            formula=formula,
            group_col="participant",
            re_formula=random_formulas[0],
            re_formula_fallbacks=random_formulas[1:],
        )
        if fit is None or model_name is None:
            raise RuntimeError(f"All improved model attempts failed for {analysis_label}.")

        selected_re_formula = None
        if model_name.startswith("mixed_random_slopes_"):
            try:
                selected_index = int(model_name.rsplit("_", 1)[1]) - 1
                selected_re_formula = random_formulas[selected_index]
            except (ValueError, IndexError):
                selected_re_formula = "unknown"

        fixed_names = (
            list(fit.fe_params.index)
            if hasattr(fit, "fe_params")
            else list(fit.params.index)
        )
        coefficients = self._coef_frame(
            fit,
            outcome=outcome,
            model_name=model_name,
            keep_terms=fixed_names,
        )
        coefficients["analysis"] = analysis_label
        coefficients["formula"] = formula
        coefficients["random_effects_formula"] = selected_re_formula or "1"
        coefficients["common_window_pre_s"] = self.common_window_pre_s
        coefficients["common_window_post_s"] = self.common_window_post_s
        coefficients["trigger_threshold"] = (
            requested_thresholds[0] if len(requested_thresholds) == 1 else np.nan
        )
        diagnostics = self._model_diagnostic_row(
            fit,
            model_name=model_name,
            formula=formula,
            random_effects_formula=selected_re_formula,
            current=current,
        )
        emmeans, contrasts = self._estimated_marginal_means_and_contrasts(fit)
        emmeans["analysis"] = analysis_label
        contrasts["analysis"] = analysis_label
        for table in (diagnostics, emmeans, contrasts):
            table["common_window_pre_s"] = self.common_window_pre_s
            table["common_window_post_s"] = self.common_window_post_s
            table["trigger_threshold"] = (
                requested_thresholds[0]
                if len(requested_thresholds) == 1
                else np.nan
            )

        self._save_table(
            coefficients,
            f"{analysis_label}_mixed_model_coefficients.csv",
        )
        self._save_table(
            diagnostics,
            f"{analysis_label}_mixed_model_diagnostics.csv",
        )
        self._save_table(emmeans, f"{analysis_label}_estimated_marginal_means.csv")
        self._save_table(contrasts, f"{analysis_label}_simple_contrasts.csv")
        return {
            "coefficients": coefficients,
            "diagnostics": diagnostics,
            "estimated_marginal_means": emmeans,
            "simple_contrasts": contrasts,
        }

    @staticmethod
    def _joint_wald_row(
        fit,
        term_names: Iterable[str],
        test: str,
        analysis: str,
    ) -> Dict[str, object]:
        """Calculate one joint Wald chi-square test for named coefficients."""
        available = list(fit.params.index)
        selected = [name for name in term_names if name in available]
        if not selected:
            return {
                "analysis": analysis,
                "test": test,
                "n_terms": 0,
                "terms": "",
                "wald_chi2": np.nan,
                "df": 0,
                "p_value": np.nan,
            }
        beta = fit.params.loc[selected].to_numpy(dtype=float)
        covariance = fit.cov_params().loc[selected, selected].to_numpy(dtype=float)
        statistic = float(beta @ np.linalg.pinv(covariance) @ beta)
        degrees = int(np.linalg.matrix_rank(covariance))
        p_value = float(chi2.sf(statistic, degrees)) if degrees > 0 else np.nan
        return {
            "analysis": analysis,
            "test": test,
            "n_terms": len(selected),
            "terms": " | ".join(selected),
            "wald_chi2": statistic,
            "df": degrees,
            "p_value": p_value,
        }

    def _binomial_omnibus_tests(self, fit, analysis: str) -> pd.DataFrame:
        """Return global tests for distance, retained interactions, and learning."""
        names = list(fit.params.index)

        def containing(*tokens: str) -> List[str]:
            return [name for name in names if all(token in name for token in tokens)]

        distance_main = [
            name
            for name in names
            if name.startswith("C(distPed_m)") and ":" not in name
        ]
        distance_yielding = containing("C(distPed_m)", "C(yielding)")
        distance_ehmi = containing("C(distPed_m)", "C(eHMIOn)")
        distance_order = containing("C(distPed_m)", "C(camera)")
        distance_all = [name for name in names if "C(distPed_m)" in name]
        ehmi_learning = containing("trial_number_centered", "C(eHMIOn)")
        three_way_learning = containing(
            "trial_number_centered", "C(yielding)", "C(eHMIOn)"
        )
        tests = {
            "categorical_distance": distance_main,
            "distance_by_av_behaviour": distance_yielding,
            "distance_by_conditional_ehmi": distance_ehmi,
            "distance_by_relative_order": distance_order,
            "all_distance_related_terms": distance_all,
            "conditional_ehmi_learning_terms": ehmi_learning,
            "trial_by_yielding_by_conditional_ehmi": three_way_learning,
        }
        return pd.DataFrame.from_records(
            [
                self._joint_wald_row(fit, terms, test, analysis)
                for test, terms in tests.items()
            ]
        )

    @staticmethod
    def _binomial_prediction_components(fit, design_info, grid: pd.DataFrame):
        """Return response-scale predictions and delta-method gradients."""
        design = build_design_matrices(
            [design_info], grid, return_type="dataframe"
        )[0]
        design = design.loc[:, fit.params.index]
        matrix = design.to_numpy(dtype=float)
        beta = fit.params.to_numpy(dtype=float)
        probabilities = expit(matrix @ beta)
        gradients = probabilities[:, None] * (1.0 - probabilities[:, None]) * matrix
        return probabilities, gradients

    def _binomial_marginal_probabilities(
        self,
        fit,
        design_info,
        analysis: str,
        model_mode: str,
        distances: Iterable[float],
    ) -> Tuple[pd.DataFrame, Dict[Tuple[int, int, int], np.ndarray]]:
        """Calculate response-scale marginal predicted probabilities."""
        distances = [float(value) for value in distances]
        records: List[Dict[str, object]] = []
        if model_mode == "full":
            records = [
                {
                    "yielding": yielding,
                    "eHMIOn": ehmi,
                    "camera": camera,
                    "distPed_m": distance,
                    "trial_number_centered": 0.0,
                }
                for yielding in (0, 1)
                for ehmi in (0, 1)
                for camera in (0, 1)
                for distance in distances
            ]
            group_columns = ["yielding", "eHMIOn", "camera"]
        elif model_mode == "participant_first":
            records = [
                {
                    "yielding": yielding,
                    "eHMIOn": ehmi,
                    "camera": 1,
                    "distPed_m": distance,
                    "trial_number_centered": 0.0,
                }
                for yielding in (0, 1)
                for ehmi in (0, 1)
                for distance in distances
            ]
            group_columns = ["yielding", "eHMIOn", "camera", "distPed_m"]
        else:
            records = [
                {
                    "yielding": 1,
                    "eHMIOn": ehmi,
                    "camera": 1,
                    "distPed_m": distance,
                    "trial_number_centered": 0.0,
                }
                for ehmi in (0, 1)
                for distance in distances
            ]
            group_columns = ["yielding", "eHMIOn", "camera", "distPed_m"]

        grid = pd.DataFrame.from_records(records)
        probabilities, gradients = self._binomial_prediction_components(
            fit, design_info, grid
        )
        grid["predicted_probability_row"] = probabilities
        covariance = fit.cov_params().to_numpy(dtype=float)
        output: List[Dict[str, object]] = []
        full_gradients: Dict[Tuple[int, int, int], np.ndarray] = {}
        for keys, indexes in grid.groupby(group_columns, sort=True).groups.items():
            if not isinstance(keys, tuple):
                keys = (keys,)
            positions = grid.index.get_indexer(list(indexes))
            mean_probability = float(np.mean(probabilities[positions]))
            gradient = np.mean(gradients[positions, :], axis=0)
            variance = float(gradient @ covariance @ gradient)
            standard_error = float(np.sqrt(max(variance, 0.0)))
            row = dict(zip(group_columns, keys))
            row.update(
                {
                    "analysis": analysis,
                    "predicted_probability": mean_probability,
                    "predicted_percentage": 100.0 * mean_probability,
                    "std_error_probability": standard_error,
                    "ci_lower_probability": max(0.0, mean_probability - 1.96 * standard_error),
                    "ci_upper_probability": min(1.0, mean_probability + 1.96 * standard_error),
                    "ci_lower_percentage": 100.0 * max(0.0, mean_probability - 1.96 * standard_error),
                    "ci_upper_percentage": 100.0 * min(1.0, mean_probability + 1.96 * standard_error),
                    "averaging": (
                        "equal weight over distances 2,4,6,8,10 m"
                        if model_mode == "full"
                        else "distance-specific prediction"
                    ),
                }
            )
            output.append(row)
            if model_mode == "full":
                key = (int(row["yielding"]), int(row["eHMIOn"]), int(row["camera"]))
                full_gradients[key] = gradient
        return pd.DataFrame.from_records(output), full_gradients

    def _binomial_revised_contrasts(
        self,
        fit,
        marginal_probabilities: pd.DataFrame,
        gradients: Dict[Tuple[int, int, int], np.ndarray],
        analysis: str,
    ) -> pd.DataFrame:
        """Calculate the 12 revised-analysis contrasts with Holm correction."""
        probability_lookup = {
            (int(row.yielding), int(row.eHMIOn), int(row.camera)): float(
                row.predicted_probability
            )
            for row in marginal_probabilities.itertuples()
        }
        covariance = fit.cov_params().to_numpy(dtype=float)
        specifications: List[Tuple[str, str, Tuple[int, int, int], Tuple[int, int, int], str]] = []
        for yielding in (0, 1):
            for camera in (0, 1):
                specifications.append(
                    (
                        "conditional_ehmi",
                        f"eHMI on minus off | yielding={yielding}, order={camera}",
                        (yielding, 0, camera),
                        (yielding, 1, camera),
                        f"yielding={yielding}; camera={camera}",
                    )
                )
        for yielding in (0, 1):
            for ehmi in (0, 1):
                specifications.append(
                    (
                        "relative_order",
                        f"participant first minus avatar first | yielding={yielding}, eHMI={ehmi}",
                        (yielding, ehmi, 0),
                        (yielding, ehmi, 1),
                        f"yielding={yielding}; eHMIOn={ehmi}",
                    )
                )
        for ehmi in (0, 1):
            for camera in (0, 1):
                specifications.append(
                    (
                        "av_behaviour",
                        f"yielding minus non-yielding | eHMI={ehmi}, order={camera}",
                        (0, ehmi, camera),
                        (1, ehmi, camera),
                        f"eHMIOn={ehmi}; camera={camera}",
                    )
                )

        rows: List[Dict[str, object]] = []
        for family, label, reference, comparison, conditioning in specifications:
            gradient = gradients[comparison] - gradients[reference]
            estimate = probability_lookup[comparison] - probability_lookup[reference]
            variance = float(gradient @ covariance @ gradient)
            standard_error = float(np.sqrt(max(variance, 0.0)))
            z_value = estimate / standard_error if standard_error > 0 else np.nan
            p_value = float(2.0 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
            rows.append(
                {
                    "analysis": analysis,
                    "contrast_family": family,
                    "contrast": label,
                    "conditioning": conditioning,
                    "reference_cell": str(reference),
                    "comparison_cell": str(comparison),
                    "estimate_probability": estimate,
                    "estimate_percentage_points": 100.0 * estimate,
                    "std_error_probability": standard_error,
                    "ci_lower_percentage_points": 100.0 * (estimate - 1.96 * standard_error),
                    "ci_upper_percentage_points": 100.0 * (estimate + 1.96 * standard_error),
                    "z_value": z_value,
                    "p_value_unadjusted": p_value,
                }
            )
        contrasts = pd.DataFrame.from_records(rows)
        contrasts["p_value_holm"] = np.nan
        for _, indexes in contrasts.groupby("contrast_family").groups.items():
            contrasts.loc[list(indexes), "p_value_holm"] = self._holm_adjust(
                contrasts.loc[list(indexes), "p_value_unadjusted"]
            )
        contrasts["multiplicity_method"] = "Holm within each four-contrast family"
        return contrasts

    def run_primary_grouped_binomial_analysis(self, trial_df: pd.DataFrame,
                                              analysis_label: str = "common_window_primary",
                                              success_column: str = "unsafe_bins_common_window",
                                              valid_column: str = "valid_bins_common_window",
                                              model_mode: str = "full") -> Dict[str, pd.DataFrame]:
        """
        Fit the primary bounded marginal binomial model with participant clustering.
        """
        if sm is None or dmatrix is None or build_design_matrices is None:
            raise RuntimeError("statsmodels and patsy are required for the binomial analysis.")
        if model_mode not in {"full", "participant_first", "event_participant_first_yielding"}:
            raise ValueError(f"Unknown binomial model mode: {model_mode}")
        required = [
            "participant",
            success_column,
            valid_column,
            "yielding",
            "eHMIOn",
            "camera",
            "distPed_m",
            "trial_number",
        ]
        missing = [column for column in required if column not in trial_df.columns]
        if missing:
            raise ValueError(f"Missing columns for {analysis_label}: {missing}")
        current = trial_df.copy()
        for column in required:
            current[column] = pd.to_numeric(current[column], errors="coerce")
        current = current.dropna(subset=required)
        current = current.loc[current[valid_column] > 0].copy()
        if model_mode == "participant_first":
            current = current.loc[current["camera"].eq(1)].copy()
        elif model_mode == "event_participant_first_yielding":
            current = current.loc[current["camera"].eq(1) & current["yielding"].eq(1)].copy()
            expected_bins = int(round(5.0 / self.trigger_bin_seconds))
            incomplete_count = int((current[valid_column] != expected_bins).sum())
            if incomplete_count:
                logger.warning(
                    f"Excluded {incomplete_count} incomplete event windows from {analysis_label}."
                )
                current = current.loc[current[valid_column].eq(expected_bins)].copy()
        if current.empty:
            raise ValueError(f"No valid observations for {analysis_label}.")
        current["trial_number_centered"] = (
            current["trial_number"]
            - current.groupby("participant")["trial_number"].transform("mean")
        )
        if model_mode == "full":
            rhs_formula = (
                "C(yielding) * C(eHMIOn) * C(camera) + "
                "C(distPed_m) * (C(yielding) + C(eHMIOn) + C(camera)) + "
                "trial_number_centered * C(yielding) * C(eHMIOn) + "
                "I(trial_number_centered ** 2)"
            )
        elif model_mode == "participant_first":
            rhs_formula = (
                "C(yielding) * C(eHMIOn) + "
                "C(distPed_m) * (C(yielding) + C(eHMIOn)) + "
                "trial_number_centered * C(yielding) * C(eHMIOn) + "
                "I(trial_number_centered ** 2)"
            )
        else:
            rhs_formula = (
                "C(distPed_m) * C(eHMIOn) + "
                "trial_number_centered * C(eHMIOn) + "
                "I(trial_number_centered ** 2)"
            )
        formula = f"{success_column}/{valid_column} ~ {rhs_formula}"
        design = dmatrix(rhs_formula, current, return_type="dataframe")
        successes = current[success_column].to_numpy(dtype=float)
        failures = (current[valid_column] - current[success_column]).to_numpy(dtype=float)
        if np.any(successes < 0) or np.any(failures < 0):
            raise ValueError(f"Invalid grouped-binomial counts in {analysis_label}.")
        fit = sm.GLM(
            endog=np.column_stack([successes, failures]),
            exog=design,
            family=sm.families.Binomial(),
        ).fit(
            cov_type="cluster",
            cov_kwds={"groups": current["participant"]},
        )
        threshold_values = (
            pd.to_numeric(current.get("trigger_threshold"), errors="coerce")
            .dropna()
            .unique()
            if "trigger_threshold" in current.columns
            else np.array([])
        )
        threshold = float(threshold_values[0]) if len(threshold_values) == 1 else np.nan
        coefficients = self._coef_frame(
            fit,
            outcome=f"{success_column}/{valid_column}",
            model_name="marginal_binomial_glm_participant_clustered",
        )
        coefficients["analysis"] = analysis_label
        coefficients["analysis_version"] = "primary_grouped_binomial_clustered_v3"
        coefficients["formula"] = formula
        coefficients["trigger_threshold"] = threshold

        distances = sorted(current["distPed_m"].dropna().unique().tolist())
        marginal, gradients = self._binomial_marginal_probabilities(
            fit, design.design_info, analysis_label, model_mode, distances
        )
        marginal["trigger_threshold"] = threshold
        contrasts = (
            self._binomial_revised_contrasts(
                fit, marginal, gradients, analysis_label
            )
            if model_mode == "full"
            else pd.DataFrame()
        )
        if not contrasts.empty:
            contrasts["trigger_threshold"] = threshold
        omnibus = self._binomial_omnibus_tests(fit, analysis_label)
        omnibus["trigger_threshold"] = threshold
        pearson_scale = (
            float(fit.pearson_chi2 / fit.df_resid)
            if getattr(fit, "df_resid", 0) > 0
            else np.nan
        )
        lag1_column = (
            success_column.replace("_unsafe_bins", "_lag1_autocorrelation")
            if success_column.endswith("_unsafe_bins")
            else "common_window_lag1_autocorrelation"
        )
        lag1 = (
            pd.to_numeric(current[lag1_column], errors="coerce")
            if lag1_column in current.columns
            else pd.Series(dtype=float)
        )
        diagnostics = pd.DataFrame.from_records(
            [
                {
                    "analysis": analysis_label,
                    "analysis_version": "primary_grouped_binomial_clustered_v3",
                    "model": "marginal_binomial_glm_participant_clustered",
                    "formula": formula,
                    "n_trials": len(current),
                    "n_participants": current["participant"].nunique(),
                    "n_aggregated_bins": int(current[valid_column].sum()),
                    "pearson_dispersion": pearson_scale,
                    "deviance": float(fit.deviance),
                    "cluster_covariance": "participant-level sandwich; arbitrary within-participant dependence",
                    "median_trial_lag1_binary_autocorrelation": float(lag1.median()) if not lag1.dropna().empty else np.nan,
                    "lag1_source_column": lag1_column,
                    "trigger_threshold": threshold,
                    "success_column": success_column,
                    "valid_column": valid_column,
                }
            ]
        )

        self._save_table(coefficients, f"{analysis_label}_binomial_coefficients.csv")
        self._save_table(marginal, f"{analysis_label}_marginal_probabilities.csv")
        self._save_table(omnibus, f"{analysis_label}_omnibus_tests.csv")
        self._save_table(diagnostics, f"{analysis_label}_binomial_diagnostics.csv")
        if not contrasts.empty:
            self._save_table(contrasts, f"{analysis_label}_revised_contrasts.csv")
        if analysis_label == "common_window_primary":
            self._save_table(
                coefficients,
                "common_window_binomial_clustered_coefficients.csv",
            )
        return {
            "coefficients": coefficients,
            "marginal_probabilities": marginal,
            "revised_contrasts": contrasts,
            "omnibus_tests": omnibus,
            "diagnostics": diagnostics,
        }

    def create_common_window_figure_with_uncertainty(self, trial_df: pd.DataFrame,
                                                     outcome: str = "perceived_unsafety_common_window_pct") -> pd.DataFrame:
        """
        Regenerate the primary spacing figure with participant-level 95% CIs.
        """
        summary = self._participant_cell_summary_with_ci(trial_df, outcome)
        summary["yielding_label"] = summary["yielding"].map(
            {0: "Non-yielding", 1: "Yielding"}
        )
        summary["eHMI_label"] = summary["eHMIOn"].map(
            {0: "No eHMI", 1: "Conditional eHMI"}
        )
        summary["order_label"] = summary["camera"].map(
            {
                0: "Avatar first / participant second",
                1: "Participant first / avatar second",
            }
        )
        self._save_table(summary, "common_window_figure7_cell_summary.csv")

        figure = px.line(
            summary,
            x="distPed_m",
            y="mean",
            error_y="ci_half_width",
            color="order_label",
            facet_col="eHMI_label",
            facet_row="yielding_label",
            markers=True,
            category_orders={
                "eHMI_label": ["No eHMI", "Conditional eHMI"],
                "yielding_label": ["Non-yielding", "Yielding"],
                "order_label": [
                    "Avatar first / participant second",
                    "Participant first / avatar second",
                ],
            },
            labels={
                "distPed_m": "Inter-pedestrian spacing (m)",
                "mean": "Perceived-unsafety time (%)",
                "order_label": "Relative pedestrian order",
                "eHMI_label": "Conditional eHMI logic",
                "yielding_label": "AV behaviour",
            },
            template=self.template,
            title="",
        )
        figure.for_each_annotation(
            lambda annotation: annotation.update(
                text=annotation.text.split("=")[-1]
            )
        )
        figure.update_layout(
            font=dict(
                family=self.font_family,
                size=22,
            ),
            legend=dict(
                title=dict(text=""),
                orientation="h",
                x=0.5,
                xanchor="center",
                y=0.5,
                yanchor="bottom",
                font=dict(
                    family=self.font_family,
                    size=self.font_size+8,
                ),
            ),
            margin=dict(
                t=30
            ),
        )
        figure.update_xaxes(
            tickvals=[2, 4, 6, 8, 10],
            tickfont=dict(
                family=self.font_family,
                size=self.font_size+8,
            ),
            title_font=dict(
                family=self.font_family,
                size=self.font_size+8,
            ),
        )

        figure.update_yaxes(
            rangemode="tozero",
            tickfont=dict(
                family=self.font_family,
                size=self.font_size+8,
            ),
            title_font=dict(
                family=self.font_family,
                size=self.font_size+8,
            ),
        )
        for figure_name in [
            "perceived_unsafety_common_window_full_factorial",
            "crossing_risk_full_factorial",
        ]:
            self.helper.save_plotly(
                fig=figure,
                name=figure_name,
                width=1320,
                height=760,
                save_html=True,
                save_png=True,
                save_eps=True,
                save_final=True,
                open_browser=True,
            )

            required_paths = [
                os.path.join(root, f"{figure_name}.{extension}")
                for root in (self.output_dir, self.fig_dir)
                for extension in ("html", "png", "eps")
            ]
            missing_paths = [
                path for path in required_paths if not os.path.isfile(path)
            ]
            if missing_paths:
                raise FileNotFoundError(
                    "The common-window figure was not saved to all required "
                    f"locations and formats: {missing_paths}"
                )
            logger.info(
                "Verified common-window figure in output and figures directories: "
                f"{required_paths}"
            )
        return summary

    def create_common_window_participant_descriptives(
        self,
        trial_df: pd.DataFrame,
        outcome: str = "perceived_unsafety_common_window_pct",
    ) -> pd.DataFrame:
        """Export participant-level means, SDs, and 95% CIs for Table 3."""
        required = ["participant", outcome, "yielding", "eHMIOn", "camera", "distPed_m"]
        current = trial_df.copy()
        for column in required:
            current[column] = pd.to_numeric(current[column], errors="coerce")
        current = current.dropna(subset=required)
        if current.empty:
            raise ValueError("No valid common-window values for participant descriptives.")

        tables: List[pd.DataFrame] = []
        specifications = [
            ("overall", []),
            ("av_behaviour", ["yielding"]),
            ("conditional_ehmi_logic", ["eHMIOn"]),
            ("relative_order", ["camera"]),
            ("distance", ["distPed_m"]),
            ("av_behaviour_by_relative_order", ["yielding", "camera"]),
        ]
        for summary_name, factors in specifications:
            participant_means = current.groupby(
                ["participant"] + factors, as_index=False
            )[outcome].mean()
            if factors:
                result = participant_means.groupby(factors, as_index=False)[outcome].agg(
                    mean="mean", participant_sd="std", n_participants="count"
                )
            else:
                result = pd.DataFrame.from_records(
                    [
                        {
                            "mean": participant_means[outcome].mean(),
                            "participant_sd": participant_means[outcome].std(ddof=1),
                            "n_participants": participant_means["participant"].nunique(),
                        }
                    ]
                )
            result["summary"] = summary_name
            result["std_error"] = result["participant_sd"] / np.sqrt(result["n_participants"])
            result["critical_t"] = result["n_participants"].map(
                lambda n: student_t.ppf(0.975, int(n) - 1) if int(n) > 1 else np.nan
            )
            result["ci_lower"] = result["mean"] - result["critical_t"] * result["std_error"]
            result["ci_upper"] = result["mean"] + result["critical_t"] * result["std_error"]
            tables.append(result)
        output = pd.concat(tables, ignore_index=True, sort=False)
        output["outcome"] = outcome
        self._save_table(output, "common_window_participant_level_descriptives.csv")
        return output

    def summarise_common_window_quality(self, trial_df: pd.DataFrame) -> pd.DataFrame:
        """Export coverage checks for the fixed pre-passage window."""
        current = trial_df.copy()
        if "expected_bins_common_window" not in current.columns:
            expected_bins = int(
                round(
                    (self.common_window_pre_s + self.common_window_post_s)
                    / self.trigger_bin_seconds
                )
            )
            current["expected_bins_common_window"] = expected_bins
            logger.warning(
                "Derived expected_bins_common_window from the configured "
                f"half-open window and bin interval: {expected_bins} bins."
            )
        required = [
            "participant",
            "yielding",
            "eHMIOn",
            "camera",
            "distPed_m",
            "common_window_duration_s",
            "valid_bins_common_window",
            "expected_bins_common_window",
            "common_window_observed_start_s",
            "common_window_observed_end_s",
            "common_window_raw_samples_per_bin_mean",
            "common_window_lag1_autocorrelation",
        ]
        missing = [column for column in required if column not in current.columns]
        if missing:
            raise ValueError(f"Missing common-window quality columns: {missing}")
        for column in required:
            current[column] = pd.to_numeric(current[column], errors="coerce")
        coverage_required = [
            column
            for column in required
            if column != "common_window_lag1_autocorrelation"
        ]
        current = current.dropna(subset=coverage_required)
        group_cols = ["distPed_m", "yielding", "eHMIOn", "camera"]
        quality = (
            current.groupby(group_cols, as_index=False)
            .agg(
                n_participants=("participant", "nunique"),
                n_trials=("participant", "count"),
                duration_min_s=("common_window_duration_s", "min"),
                duration_max_s=("common_window_duration_s", "max"),
                valid_bins_min=("valid_bins_common_window", "min"),
                valid_bins_max=("valid_bins_common_window", "max"),
                expected_bins_min=("expected_bins_common_window", "min"),
                expected_bins_max=("expected_bins_common_window", "max"),
                observed_start_min_s=("common_window_observed_start_s", "min"),
                observed_start_max_s=("common_window_observed_start_s", "max"),
                observed_end_min_s=("common_window_observed_end_s", "min"),
                observed_end_max_s=("common_window_observed_end_s", "max"),
                raw_samples_per_bin_mean=(
                    "common_window_raw_samples_per_bin_mean",
                    "mean",
                ),
                lag1_autocorrelation_mean=(
                    "common_window_lag1_autocorrelation",
                    "mean",
                ),
                lag1_autocorrelation_n=(
                    "common_window_lag1_autocorrelation",
                    "count",
                ),
            )
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
        quality["nominal_window_pre_s"] = self.common_window_pre_s
        quality["nominal_window_post_s"] = self.common_window_post_s
        quality["interval_closure"] = "[start,end)"
        quality["aggregation_rule"] = "any(raw_value > threshold)"
        self._save_table(quality, "common_window_quality_by_condition.csv")
        return quality

    def run_within_between_models(self, trial_df: pd.DataFrame) -> pd.DataFrame:

        """Estimate within person and between person links to crossing risk.

        For each questionnaire item, the raw score is decomposed into a
        participant mean and a trial specific deviation from that mean. This
        separates between participant differences from within participant
        trial to trial fluctuations.

        Args:
            trial_df: Trial level DataFrame containing ratings and predictors.

        Returns:
            A tidy coefficient table for the within and between score terms.

        Raises:
            ValueError: If no models can be fitted successfully.
        """
        # Declare the minimum columns needed for within versus between analyses.
        needed = ["participant", "crossing_risk", "yielding", "eHMIOn", "camera", "distPed_m"]
        results: List[pd.DataFrame] = []
        # Run the same decomposition and model for each questionnaire item.
        for q_col in ["Q1", "Q2", "Q3"]:
            current = trial_df.copy()
            current[q_col] = pd.to_numeric(current[q_col], errors="coerce")
            current["crossing_risk"] = pd.to_numeric(current["crossing_risk"], errors="coerce")
            current = current.dropna(subset=needed + [q_col])
            if current.empty:
                logger.warning(f"Skipping within/between model for {q_col}: no valid rows")
                continue
        # Decompose the rating into between participant and within participant components.
            current["between_score"] = current.groupby("participant")[q_col].transform("mean")
            current["within_score"] = current[q_col] - current["between_score"]
        # Specify the fixed effect structure used by the current model.
            formula = (
                "crossing_risk ~ within_score + between_score + C(yielding) + C(eHMIOn) + "
                "C(camera) + distPed_m"
            )
        # Estimate the current feature model with robust fallbacks.
            fit, model_name = self._fit_model_with_fallbacks(
                current,
                formula=formula,
                group_col="participant",
                re_formula="~distPed_m",
            )
            if fit is None:
                continue
        # Extract the coefficients that will be exported and plotted.
            coef_df = self._coef_frame(
                fit,
                outcome=q_col,
                model_name=model_name,  # type: ignore
                keep_terms=["within_score", "between_score"],
            )
            results.append(coef_df)

            logger.info(
                f"Within/between model for {q_col} fitted with {model_name}.\n"
                f"{coef_df[['pretty_term', 'estimate', 'ci_lower', 'ci_upper', 'p_value']].to_string(index=False)}"
            )

        # Fail loudly when none of the requested models produce a usable fit.
        if not results:
            raise ValueError("No within/between models were successfully fitted.")
        # Combine the per rating model outputs into one coefficient table.
        results_df = pd.concat(results, ignore_index=True)
        self._save_table(results_df, "within_between_models_crossing_risk.csv")
        # Build a coefficient plot for the within and between estimates.
        fig = px.scatter(
            results_df,
            x="estimate",
            y="outcome",
            color="pretty_term",
            error_x=results_df["ci_upper"] - results_df["estimate"],
            error_x_minus=results_df["estimate"] - results_df["ci_lower"],
            labels={
                "estimate": "Coefficient on crossing risk",
                "outcome": "Rating",
                "pretty_term": "Effect",
            },
            template=self.template,
            title="",
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(font=dict(family=self.font_family, size=self.font_size))
        self.helper.save_plotly(
            fig=fig,
            name="within_between_crossing_risk_coefficients",
            width=1100,
            height=650,
            save_final=True,
            open_browser=True,
        )
        logger.info("Saved figure set for: within_between_crossing_risk_coefficients")
        return results_df

    def merge_trigger_features(self, trial_df: pd.DataFrame, feature_df: Optional[pd.DataFrame] = None,
                               save: bool = True, trigger_threshold: float = 0.05) -> pd.DataFrame:

        """Merge derived trigger features onto a trial level table.

        Args:
            trial_df: Base trial level DataFrame.
            feature_df: Optional precomputed trigger feature table.
            save: Whether to write the merged table to disk.
            trigger_threshold: Threshold used if feature_df needs to be built on demand.

        Returns:
            The enriched trial level DataFrame.
        """
        # Generate trigger features on demand when they were not supplied.
        if feature_df is None:
            feature_df = self.build_trigger_feature_table(threshold=trigger_threshold)

        # Copy both input tables to avoid mutating caller owned DataFrames.
        left = trial_df.copy()
        right = feature_df.copy()

        # Standardise merge key types before joining the tables.
        left["participant"] = pd.to_numeric(left["participant"], errors="coerce")
        right["participant"] = pd.to_numeric(right["participant"], errors="coerce")

        left["video_id"] = left["video_id"].astype(str)
        right["video_id"] = right["video_id"].astype(str)

        # List the derived trigger feature columns that may be appended.
        merge_cols = [
            "participant",
            "video_id",
            "mean_trigger_raw",
            "mean_trigger_pct",
            "peak_trigger_raw",
            "peak_trigger_pct",
            "auc_trigger_raw_s",
            "auc_trigger_pct_s",
            "unsafe_prop",
            "unsafe_prop_pct",
            "unsafe_prop_common_window",
            "unsafe_prop_common_window_pct",
            "unsafe_bins_common_window",
            "valid_bins_common_window",
            "expected_bins_common_window",
            "common_window_event",
            "common_window_pre_s",
            "common_window_post_s",
            "common_window_start_s",
            "common_window_end_s",
            "common_window_observed_start_s",
            "common_window_observed_end_s",
            "common_window_duration_s",
            "common_window_bin_duration_s",
            "common_window_interval_closure",
            "trigger_bin_interval_s",
            "raw_trigger_sampling_hz",
            "expected_raw_samples_per_bin",
            "common_window_raw_samples_per_bin_mean",
            "common_window_raw_samples_per_bin_min",
            "common_window_raw_samples_per_bin_max",
            "common_window_lag1_autocorrelation",
            "trigger_bin_state_rule",
            "time_pressed_s",
            "switch_count",
            "first_press_latency_s",
            "first_release_latency_s",
            "trigger_value_sd",
            "analysis_duration_s",
            "analysis_cutoff_s",
            "dt_seconds",
            "n_bins",
            "n_trigger_samples",
            "trigger_threshold",
        ]

        for column in right.columns:
            if column.startswith(
                (
                    "braking_onset_window_",
                    "stopping_window_",
                    "resumption_window_",
                )
            ) and column not in merge_cols:
                merge_cols.append(column)

        keep_cols = [c for c in merge_cols if c in right.columns]
        # Join the feature table onto the trial table while retaining all trial rows.
        enriched = left.merge(
            right[keep_cols],
            on=["participant", "video_id"],
            how="left",
        )

        # Both inputs can carry the analysis threshold. Pandas then creates
        # trigger_threshold_x/y, which previously left the exported canonical
        # trigger_threshold column missing. Coalesce the two audited values.
        if "trigger_threshold_x" in enriched.columns or "trigger_threshold_y" in enriched.columns:
            threshold_left = (
                pd.to_numeric(enriched["trigger_threshold_x"], errors="coerce")
                if "trigger_threshold_x" in enriched.columns
                else pd.Series(np.nan, index=enriched.index)
            )
            threshold_right = (
                pd.to_numeric(enriched["trigger_threshold_y"], errors="coerce")
                if "trigger_threshold_y" in enriched.columns
                else pd.Series(np.nan, index=enriched.index)
            )
            # The feature table is authoritative because it is rebuilt for
            # each sensitivity threshold; the base trial table can still
            # carry the primary 0.10 label.
            enriched["trigger_threshold"] = threshold_right.combine_first(threshold_left)
            enriched = enriched.drop(
                columns=["trigger_threshold_x", "trigger_threshold_y"],
                errors="ignore",
            )
        # Keep the primary outcome definition consistent across the pipeline.
        # The manuscript defines crossing risk as the percentage of analysed
        # time bins where the pressure-sensitive trigger exceeded the threshold,
        # not as the mean trigger pressure. If unsafe_prop_pct is available, it
        # is therefore the authoritative crossing_risk value. The incoming value
        # is preserved for auditing.
        if "unsafe_prop_pct" in enriched.columns:
            unsafe_pct = pd.to_numeric(enriched["unsafe_prop_pct"], errors="coerce")
            if "crossing_risk" in enriched.columns:
                incoming_risk = pd.to_numeric(enriched["crossing_risk"], errors="coerce")
                enriched["crossing_risk_input"] = incoming_risk
                enriched["crossing_risk"] = unsafe_pct.combine_first(incoming_risk)
            else:
                enriched["crossing_risk"] = unsafe_pct
            logger.info("Set crossing_risk from unsafe_prop_pct where available.")

        if "unsafe_prop_common_window_pct" in enriched.columns:
            enriched["perceived_unsafety_common_window_pct"] = pd.to_numeric(
                enriched["unsafe_prop_common_window_pct"], errors="coerce"
            )
            logger.info(
                "Added perceived_unsafety_common_window_pct from the fixed "
                "pre-passage window."
            )
        # Persist the enriched table when the caller requests an output file.
        if save:
            out_csv = os.path.join(self.output_dir, "trial_level_enriched_with_trigger_features.csv")
            enriched.to_csv(out_csv, index=False)
            logger.info(f"Saved enriched trial table: {out_csv}")

        return enriched

    def run_feature_models_and_figures(self, trial_df: pd.DataFrame,
                                       feature_outcomes: Optional[List[str]] = None) -> pd.DataFrame:

        """Fit models for derived trigger features and create figures.

        The method summarises each requested feature across distance and
        context, fits one model per feature, writes tidy coefficient tables,
        and generates profile and coefficient plots.

        Args:
            trial_df: Trial level DataFrame enriched with trigger features.
            feature_outcomes: Optional list of trigger feature columns to model.

        Returns:
            A concatenated coefficient DataFrame across all fitted features.

        Raises:
            ValueError: If no feature models can be fitted successfully.
        """
        # Use the default set of trigger outcomes when none are specified.
        if feature_outcomes is None:
            feature_outcomes = [
                "peak_trigger_pct",
                "auc_trigger_pct_s",
                "switch_count",
                "first_press_latency_s",
                "unsafe_prop_pct",
            ]
        # Collect coefficient tables and descriptive summaries across all feature models.
        all_coef_frames: List[pd.DataFrame] = []
        outcome_summary_frames: List[pd.DataFrame] = []
        # Fit one model and build one descriptive summary per trigger feature.
        for outcome in feature_outcomes:
            current = trial_df.copy()
            current[outcome] = pd.to_numeric(current[outcome], errors="coerce")
            current = current.dropna(
                subset=["participant", outcome, "yielding", "eHMIOn", "camera", "distPed_m"]
            )
            if current.empty:
                logger.warning(f"Skipping feature model for {outcome}: no valid rows")
                continue
            # Aggregate the current feature over distance and scenario context for plotting.
            summary = (
                current.groupby(["distPed_m", "yielding", "eHMIOn", "camera"], as_index=False)[outcome]
                .mean()
                .sort_values(["yielding", "eHMIOn", "camera", "distPed_m"])
            )
            summary["feature"] = outcome
            outcome_summary_frames.append(summary)

            # Use the same core predictors across features so coefficients are comparable.
            formula = (
                f"{outcome} ~ C(yielding) + C(eHMIOn) + C(camera) + distPed_m + "
                "C(yielding):C(eHMIOn) + C(yielding):C(camera) + C(eHMIOn):C(camera)"
            )
            fit, model_name = self._fit_model_with_fallbacks(
                current,
                formula=formula,
                group_col="participant",
                re_formula="~distPed_m",
            )
            if fit is None:
                continue

            # Keep only the main effects that are intended for the exported coefficient summary.
            coef_df = self._coef_frame(
                fit,
                outcome=outcome,
                model_name=model_name,  # type: ignore
                keep_terms=[
                    "C(yielding)[T.1]",
                    "C(eHMIOn)[T.1]",
                    "C(camera)[T.1]",
                    "distPed_m",
                ],
            )
            all_coef_frames.append(coef_df)
            logger.info(
                f"Feature model for {outcome} fitted with {model_name}.\n"
                f"{coef_df[['pretty_term', 'estimate', 'ci_lower', 'ci_upper', 'p_value']].to_string(index=False)}"
            )
        # Write and visualise descriptive feature summaries when any are available.
        if outcome_summary_frames:
            summary_df = pd.concat(outcome_summary_frames, ignore_index=True)
            self._save_table(summary_df, "trigger_feature_distance_profiles.csv")
        # Define human readable labels for the requested trigger features.
            feature_labels = {
                "peak_trigger_pct": "Peak trigger (0–100)",
                "auc_trigger_pct_s": "Trigger AUC",
                "switch_count": "Switch count",
                "first_press_latency_s": "First press latency (s)",
                "unsafe_prop_pct": "Unsafe time (%)",
            }
            summary_df["feature_label"] = summary_df["feature"].map(feature_labels).fillna(summary_df["feature"])
            summary_df["yielding_label"] = summary_df["yielding"].map({0: "Non-yielding", 1: "Yielding"})
            summary_df["eHMI_label"] = summary_df["eHMIOn"].map({0: "No eHMI", 1: "eHMI"})
            summary_df["order_label"] = summary_df["camera"].map(
                {
                    0: "Avatar first / participant second",
                    1: "Participant first / avatar second",
                }
            )
            summary_df["panel_title"] = (
                summary_df["order_label"] + " | " + summary_df["eHMI_label"]
            )
            # Log compact descriptive insights that help interpret the profile figure.
            for feat in feature_outcomes:
                feat_summary = summary_df.loc[summary_df["feature"] == feat].copy()
                feat_trials = trial_df.copy()
                feat_trials[feat] = pd.to_numeric(feat_trials[feat], errors="coerce")
                feat_trials = feat_trials.dropna(
                    subset=[feat, "yielding", "eHMIOn", "camera", "distPed_m", "participant"]
                )
                if feat_summary.empty or feat_trials.empty:
                    continue

                feature_name = feature_labels.get(feat, feat)
                distances = sorted(pd.to_numeric(feat_summary["distPed_m"],
                                                 errors="coerce").dropna().unique().tolist())
                if not distances:
                    continue
                near_dist = float(distances[0])
                far_dist = float(distances[-1])

                overall_mean = float(feat_trials[feat].mean())
                overall_sd = float(feat_trials[feat].std(ddof=1)) if len(feat_trials) > 1 else float("nan")
                overall_min = float(feat_trials[feat].min())
                overall_max = float(feat_trials[feat].max())
                logger.info(
                    f"{feature_name} profile summary: n={len(feat_trials)} trials, "
                    f"{feat_trials['participant'].nunique()} participants, "
                    f"distances={[int(d) if float(d).is_integer() else float(d) for d in distances]}, "
                    f"mean={overall_mean:.3f}, sd={overall_sd:.3f}, "
                    f"min={overall_min:.3f}, max={overall_max:.3f}"
                )

                nearest_mean = float(feat_summary.loc[feat_summary["distPed_m"] == near_dist, feat].mean())
                farthest_mean = float(feat_summary.loc[feat_summary["distPed_m"] == far_dist, feat].mean())
                logger.info(
                    f"{feature_name} distance contrast: nearest {near_dist:.1f} m "
                    f"mean={nearest_mean:.3f}, farthest {far_dist:.1f} m "
                    f"mean={farthest_mean:.3f}, far minus near={farthest_mean - nearest_mean:.3f}"
                )

                scenario_changes = []
                for (yielding_value, ehmi_value, camera_value), ctx_df in feat_summary.groupby(["yielding",
                                                                                                "eHMIOn", "camera"]):
                    ctx_df = ctx_df.sort_values("distPed_m")  # pyright: ignore[reportCallIssue]
                    if ctx_df.empty:
                        continue
                    first_val = float(ctx_df.iloc[0][feat])
                    last_val = float(ctx_df.iloc[-1][feat])
                    delta = last_val - first_val
                    scenario_changes.append({
                        "yielding": int(yielding_value),
                        "eHMIOn": int(ehmi_value),
                        "camera": int(camera_value),
                        "start": first_val,
                        "end": last_val,
                        "delta": delta,
                    })
                if scenario_changes:
                    strongest = max(scenario_changes, key=lambda row: abs(row["delta"]))
                    yielding_txt = "Yielding" if strongest["yielding"] == 1 else "Non-yielding"
                    ehmi_txt = "eHMI" if strongest["eHMIOn"] == 1 else "No eHMI"
                    order_txt = (
                        "Participant first / avatar second"
                        if strongest["camera"] == 1
                        else "Avatar first / participant second"
                    )
                    logger.info(
                        f"{feature_name} strongest profile change: {yielding_txt}, "
                        f"{ehmi_txt}, {order_txt} changed by {strongest['delta']:.3f} "
                        f"from {strongest['start']:.3f} to {strongest['end']:.3f} across distance"
                    )

                yielding_means = feat_summary.groupby("yielding", as_index=False)[feat].mean()
                if set(yielding_means["yielding"].tolist()) == {0, 1}:
                    not_yielding_mean = float(yielding_means.loc[yielding_means["yielding"] == 0, feat].iloc[0])
                    yielding_mean = float(yielding_means.loc[yielding_means["yielding"] == 1, feat].iloc[0])
                    logger.info(
                        f"{feature_name} yielding contrast: yielding mean={yielding_mean:.3f}, "
                        f"not yielding mean={not_yielding_mean:.3f}, "
                        f"difference={yielding_mean - not_yielding_mean:.3f}"
                    )
        # Build a faceted profile plot so each panel only contains two lines.
            feature_order = [
                feat for feat in feature_outcomes if feat in summary_df["feature"].unique().tolist()
            ]
            panel_order = [
                {"eHMIOn": 0, "camera": 0, "title": "Avatar first / participant second | No eHMI"},
                {"eHMIOn": 1, "camera": 0, "title": "Avatar first / participant second | eHMI"},
                {"eHMIOn": 0, "camera": 1, "title": "Participant first / avatar second | No eHMI"},
                {"eHMIOn": 1, "camera": 1, "title": "Participant first / avatar second | eHMI"},
            ]
            yielding_styles = {
                0: {
                    "name": "Non-yielding",
                    "color": "rgba(85, 98, 112, 0.95)",
                    "dash": "dot",
                    "symbol": "circle-open",
                },
                1: {
                    "name": "Yielding",
                    "color": "rgba(31, 119, 180, 0.95)",
                    "dash": "solid",
                    "symbol": "circle",
                },
            }

            if feature_order:
                subplot_titles = []
                for row_idx, _ in enumerate(feature_order, start=1):
                    if row_idx == 1:
                        subplot_titles.extend([panel["title"] for panel in panel_order])
                    else:
                        subplot_titles.extend([""] * len(panel_order))

                fig = make_subplots(
                    rows=len(feature_order),
                    cols=len(panel_order),
                    shared_xaxes=True,
                    horizontal_spacing=0.05,
                    vertical_spacing=0.08,
                    subplot_titles=subplot_titles,
                )

                tickvals = sorted(pd.to_numeric(summary_df["distPed_m"],
                                                errors="coerce").dropna().unique().tolist())  # type: ignore

                for row_idx, feat in enumerate(feature_order, start=1):
                    feat_df = summary_df.loc[summary_df["feature"] == feat].copy()

                    for col_idx, panel in enumerate(panel_order, start=1):
                        panel_df = feat_df.loc[
                            (feat_df["eHMIOn"] == panel["eHMIOn"]) &
                            (feat_df["camera"] == panel["camera"])
                        ].copy()
                        if panel_df.empty:
                            continue

                        for yielding_value in [0, 1]:
                            trace_df = panel_df.loc[panel_df["yielding"] == yielding_value].copy()
                            if trace_df.empty:
                                continue
                            trace_df = trace_df.sort_values("distPed_m")
                            style = yielding_styles[yielding_value]

                            fig.add_trace(
                                go.Scatter(
                                    x=trace_df["distPed_m"],
                                    y=trace_df[feat],
                                    mode="lines+markers",
                                    name=style["name"],
                                    legendgroup=style["name"],
                                    showlegend=(row_idx == 1 and col_idx == 1),
                                    line=dict(color=style["color"], dash=style["dash"], width=2.5),
                                    marker=dict(color=style["color"], symbol=style["symbol"], size=8),
                                    hovertemplate=(
                                        f"<b>{feature_labels.get(feat, feat)}</b><br>"
                                        f"{panel['title']}<br>"
                                        f"{style['name']}<br>"
                                        "Distance: %{x:.0f} m<br>"
                                        "Value: %{y:.2f}<extra></extra>"
                                    ),
                                ),
                                row=row_idx,
                                col=col_idx,
                            )

                        fig.update_xaxes(
                            tickmode="array",
                            tickvals=tickvals,
                            tickfont=dict(family=self.font_family, size=self.font_size + 2),
                            row=row_idx,
                            col=col_idx,
                        )
                        fig.update_yaxes(
                            tickfont=dict(family=self.font_family, size=self.font_size + 2),
                            row=row_idx,
                            col=col_idx,
                        )

                    fig.update_yaxes(
                        title_text=feature_labels.get(feat, feat),
                        title_font=dict(family=self.font_family, size=self.font_size + 8),
                        title_standoff=24,
                        automargin=True,
                        row=row_idx,
                        col=1,
                    )

                for col_idx in range(1, len(panel_order) + 1):
                    fig.update_xaxes(
                        title_text="Distance between pedestrians (m)",
                        title_font=dict(family=self.font_family, size=self.font_size + 8),
                        row=len(feature_order),
                        col=col_idx,
                    )

                fig.update_layout(
                    template=self.template,
                    title="",
                    height=max(950, 260 * len(feature_order)),
                    font=dict(family=self.font_family, size=self.font_size),
                    margin=dict(l=0, r=0, t=0, b=0),
                    legend=dict(
                        title_text="",
                        orientation="h",
                        x=0.5,
                        xanchor="center",
                        y=0.5,
                        yanchor="bottom",
                    ),
                )

                for annotation in fig.layout.annotations:  # type: ignore
                    annotation.font = dict(family=self.font_family, size=self.font_size + 4)

                self.helper.save_plotly(
                    fig=fig,
                    name="trigger_feature_distance_profiles",
                    width=2100,
                    height=max(950, 260 * len(feature_order)),
                    save_final=True,
                    open_browser=True,
                )
                logger.info("Saved figure set for: trigger_feature_distance_profiles")
        # Stop with a clear error when every feature model fails.
        if not all_coef_frames:
            raise ValueError("No trigger feature models were successfully fitted.")
        # Concatenate the fitted feature model coefficients into one table.
        coef_df = pd.concat(all_coef_frames, ignore_index=True)
        self._save_table(coef_df, "trigger_feature_model_coefficients.csv")
        # Create the final cross feature coefficient summary plot.
        fig_coef = px.scatter(
            coef_df,
            x="estimate",
            y="outcome",
            color="pretty_term",
            error_x=coef_df["ci_upper"] - coef_df["estimate"],
            error_x_minus=coef_df["estimate"] - coef_df["ci_lower"],
            labels={
                "estimate": "Coefficient",
                "outcome": "Feature",
                "pretty_term": "Predictor",
            },
            template=self.template,
            title="",
        )
        fig_coef.add_vline(x=0, line_dash="dash", line_color="black")
        fig_coef.update_layout(font=dict(family=self.font_family, size=self.font_size))
        self.helper.save_plotly(
            fig=fig_coef,
            name="trigger_feature_model_coefficients",
            width=1150,
            height=700,
            save_final=True,
            open_browser=True,
        )
        logger.info("Saved figure set for: trigger_feature_model_coefficients")
        return coef_df

    def run_all(self, trial_df: pd.DataFrame, equivalence_margin: float = 5.0, trigger_threshold: float = 0.05) -> Dict[str, pd.DataFrame]:

        """Run the full advanced statistics pipeline end to end.

        Args:
            trial_df: Trial level input DataFrame.
            equivalence_margin: Symmetric equivalence margin used in the
                near versus far TOST comparison.
            trigger_threshold: Threshold on the 0..1 pressure-sensitive trigger
                signal used to define a binary pressed/risk state.

        Returns:
            A dictionary containing the main intermediate and final result
            tables generated by the pipeline.
        """
        # Log the start of the end to end pipeline for traceability.
        logger.info("Starting advanced statistics pipeline")
        logger.info(
            "Advanced statistics specification: "
            f"{ADVANCED_STATS_SPECIFICATION}"
        )
        logger.info(
            "Common event-aligned trigger window: "
            f"{self.common_window_pre_s:.2f} s before participant passage to "
            f"{self.common_window_post_s:.2f} s after passage."
        )
        # Step 1: derive participant by video trigger features.
        feature_df = self.build_trigger_feature_table(threshold=trigger_threshold)
        enriched_trial_df = self.merge_trigger_features(
            trial_df,
            feature_df=feature_df,
            save=True,
        )
        # Generate and verify the primary manuscript figure before model fitting
        # so it remains available even if a later statistical model fails.
        logger.info(
            "Generating perceived_unsafety_common_window_full_factorial in "
            f"'{self.output_dir}' and '{self.fig_dir}'."
        )
        common_window_figure_summary = (
            self.create_common_window_figure_with_uncertainty(enriched_trial_df)
        )
        # Step 2: fit the primary bounded model. Participant-clustered sandwich
        # uncertainty allows arbitrary dependence among a participant's trial
        # totals, including the effect of serially dependent trigger bins.
        common_window_binomial = self.run_primary_grouped_binomial_analysis(
            enriched_trial_df,
            analysis_label="common_window_primary",
            success_column="unsafe_bins_common_window",
            valid_column="valid_bins_common_window",
            model_mode="full",
        )
        participant_first_binomial = self.run_primary_grouped_binomial_analysis(
            enriched_trial_df,
            analysis_label="common_window_participant_first",
            success_column="unsafe_bins_common_window",
            valid_column="valid_bins_common_window",
            model_mode="participant_first",
        )

        # Event-aligned sensitivity analyses use only yielding trials in the
        # participant-first subset, where spacing is not confounded with the
        # avatar-first participant-passage geometry.
        event_results: Dict[str, Dict[str, pd.DataFrame]] = {}
        for event_name in (
            "braking_onset_window",
            "stopping_window",
            "resumption_window",
        ):
            event_results[event_name] = self.run_primary_grouped_binomial_analysis(
                enriched_trial_df,
                analysis_label=f"{event_name}_participant_first",
                success_column=f"{event_name}_unsafe_bins",
                valid_column=f"{event_name}_valid_bins",
                model_mode="event_participant_first_yielding",
            )

        # Repeat the new primary common-window model at every configured trigger
        # threshold. This is distinct from the superseded legacy-window check.
        threshold_results: Dict[str, Dict[str, pd.DataFrame]] = {}
        configured_thresholds = [
            float(value)
            for value in self._config_or_default(
                "trigger_threshold", [trigger_threshold]
            )
        ]
        for current_threshold in configured_thresholds:
            threshold_key = f"{int(round(current_threshold * 100)):02d}pct"
            if np.isclose(current_threshold, trigger_threshold):
                threshold_results[threshold_key] = common_window_binomial
                continue
            threshold_features = self.build_trigger_feature_table(
                threshold=current_threshold
            )
            threshold_trials = self.merge_trigger_features(
                trial_df,
                feature_df=threshold_features,
                save=False,
            )
            threshold_results[threshold_key] = self.run_primary_grouped_binomial_analysis(
                threshold_trials,
                analysis_label=f"common_window_threshold_{threshold_key}",
                success_column="unsafe_bins_common_window",
                valid_column="valid_bins_common_window",
                model_mode="full",
            )
        for table_key, filename in (
            ("marginal_probabilities", "common_window_threshold_marginal_probabilities.csv"),
            ("revised_contrasts", "common_window_threshold_revised_contrasts.csv"),
            ("omnibus_tests", "common_window_threshold_omnibus_tests.csv"),
            ("diagnostics", "common_window_threshold_binomial_diagnostics.csv"),
        ):
            frames = [
                result[table_key]
                for result in threshold_results.values()
                if not result[table_key].empty
            ]
            if frames:
                self._save_table(pd.concat(frames, ignore_index=True), filename)

        # Retain the Gaussian repeated-measures fit only as a secondary model
        # sensitivity analysis; it is no longer the primary inference.
        common_window_gaussian = self.run_improved_repeated_measures_model(
            enriched_trial_df,
            outcome="perceived_unsafety_common_window_pct",
            analysis_label="common_window_gaussian_secondary",
        )
        common_window_quality = self.summarise_common_window_quality(
            enriched_trial_df
        )
        common_window_descriptives = self.create_common_window_participant_descriptives(
            enriched_trial_df
        )
        # Step 3: test near versus far equivalence on the legacy-window data.
        # This remains a secondary exploratory output for continuity.
        equivalence_df = self.run_equivalence_tests(
            enriched_trial_df,
            outcome="crossing_risk",
            equivalence_margin=equivalence_margin,
        )
        # Step 4: fit the trigger feature models and export their figures.
        feature_coef_df = self.run_feature_models_and_figures(enriched_trial_df)
        # Step 5: fit the within versus between participant rating models.
        within_between_df = self.run_within_between_models(enriched_trial_df)
        logger.info("Finished advanced statistics pipeline")
        # Return all major outputs so callers can inspect or reuse them programmatically.
        return {
            "features": feature_df,
            "trial_enriched": enriched_trial_df,
            "common_window_binomial_coefficients": common_window_binomial["coefficients"],
            "common_window_marginal_probabilities": common_window_binomial["marginal_probabilities"],
            "common_window_revised_contrasts": common_window_binomial["revised_contrasts"],
            "common_window_omnibus_tests": common_window_binomial["omnibus_tests"],
            "common_window_binomial_diagnostics": common_window_binomial["diagnostics"],
            "participant_first_binomial": participant_first_binomial,
            "event_aligned_binomial": event_results,
            "threshold_binomial": threshold_results,
            "common_window_gaussian_secondary": common_window_gaussian,
            "common_window_quality": common_window_quality,
            "common_window_participant_descriptives": common_window_descriptives,
            "common_window_figure_summary": common_window_figure_summary,
            "equivalence": equivalence_df,
            "feature_coefficients": feature_coef_df,
            "within_between": within_between_df,
        }
