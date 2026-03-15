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
from scipy.stats import t as student_t
from scipy.stats import ttest_rel

import common
from custom_logger import CustomLogger

import warnings

try:
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except ImportError:  # pragma: no cover
    smf = None  # type: ignore
    ConvergenceWarning = Warning  # type: ignore


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
        fig_dir: Directory used for generated figures.
        template: Plotly template name from the project config.
        font_size: Default figure font size from the project config.
        font_family: Default figure font family from the project config.
    """

    def __init__(
        self,
        helper,
        mapping_df: pd.DataFrame,
        output_dir: Optional[str] = None,
        data_root: Optional[str] = None,
    ) -> None:

        """Initialise the runner and normalise mapping table columns.

        Args:
            helper: Helper object that provides plot saving utilities and config access.
            mapping_df: DataFrame describing scenario metadata for each video.
            output_dir: Optional output directory override.
            data_root: Optional data directory override.

        Returns:
            None
        """
        # Store shared dependencies and project level configuration once during setup.
        # ------------------------------------------------------------------
        # Store the shared helper and core input tables on the runner instance.
        # ------------------------------------------------------------------
        self.helper = helper
        self.mapping_df = mapping_df.copy()
        self.output_dir = output_dir or common.get_configs("output")
        self.data_root = data_root or common.get_configs("data")
        self.stats_dir = os.path.join(self.output_dir, "statistics")
        self.fig_dir = common.get_configs("figures")
        self.template = common.get_configs("plotly_template")
        self.font_size = common.get_configs("font_size")
        self.font_family = common.get_configs("font_family")

        # Ensure output folders exist before any table or figure writing occurs.
        # ------------------------------------------------------------------
        # Create any required output directories before later save operations.
        # ------------------------------------------------------------------
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

        # Normalise identifier style early so later joins use consistent string keys.
        # ------------------------------------------------------------------
        # Normalise identifier columns to consistent string types for later joins.
        # ------------------------------------------------------------------
        if "video_id" in self.mapping_df.columns:
            self.mapping_df["video_id"] = self.mapping_df["video_id"].astype(str)
        if "condition_name" in self.mapping_df.columns:
            self.mapping_df["condition_name"] = self.mapping_df["condition_name"].astype(str)

        # Coerce mapping columns that should be numeric so downstream comparisons and
        # calculations behave predictably.
        # ------------------------------------------------------------------
        # Coerce mapping metadata columns that should participate in numeric comparisons.
        # ------------------------------------------------------------------
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
    def _distance_code_to_meters(value) -> float:
        """Convert coded distance levels 1..5 to 2..10 m while preserving metre values."""
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return np.nan
        numeric = float(numeric)
        if numeric == 0:
            return np.nan
        if numeric in {1.0, 2.0, 3.0, 4.0, 5.0}:
            return numeric * 2.0
        return numeric

    @classmethod
    def _distance_series_to_meters(cls, series: pd.Series) -> pd.Series:
        """Vectorised distance conversion for pandas Series."""
        numeric = pd.to_numeric(series, errors="coerce")
        mapped = numeric.copy()  # pyright: ignore[reportAttributeAccessIssue]
        mask_code = numeric.isin([1, 2, 3, 4, 5])  # type: ignore
        mapped.loc[mask_code] = numeric.loc[mask_code] * 2.0  # type: ignore
        mapped.loc[numeric == 0] = np.nan
        return mapped

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
        # ------------------------------------------------------------------
        # Treat missing trigger cells as empty observations.
        # ------------------------------------------------------------------
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return []
        # Some exports store a Python list as text, so literal_eval is used to recover it.
        try:
            # ------------------------------------------------------------------
            # Recover list literals that were serialised to text in CSV files.
            # ------------------------------------------------------------------
            parsed = ast.literal_eval(cell) if isinstance(cell, str) else cell
        except Exception:
            return []
        # Ignore anything that is not a list because the feature extraction logic expects
        # a sequence of per bin values.
        # ------------------------------------------------------------------
        # Reject malformed parsed values that do not produce a list.
        # ------------------------------------------------------------------
        if not isinstance(parsed, list):
            return []
        # Keep only finite numeric entries to protect the summary statistics from invalid values.
        # ------------------------------------------------------------------
        # Accumulate only the finite numeric values that survive validation.
        # ------------------------------------------------------------------
        out: List[float] = []
        for item in parsed:
            if isinstance(item, (int, float)) and np.isfinite(item):
                out.append(float(item))
        return out

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:

        """Safely divide two numeric values.

        Args:
            numerator: Value to place in the numerator.
            denominator: Value to place in the denominator.

        Returns:
            The floating point quotient, or NaN when the denominator is zero
            or missing.
        """
        # Guard division based summaries against zero denominators and propagated NaNs.
        if denominator == 0 or np.isnan(denominator):
            return np.nan
        return float(numerator / denominator)

    @staticmethod
    def _context_label(row: pd.Series) -> str:

        """Create a compact context label from a mapping row.

        Args:
            row: Row containing yielding, eHMIOn, and camera fields.

        Returns:
            A compact label such as ``Y1_H0_C1``.
        """
        # Build a compact label that is stable enough for tables, legends, and logs.
        return (
            f"Y{int(row['yielding'])}_"
            f"H{int(row['eHMIOn'])}_"
            f"C{int(row['camera'])}"
        )

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

        # ------------------------------------------------------------------
        # Read the camera flag that determines which crossing time applies.
        # ------------------------------------------------------------------
        camera_raw = row.get("camera")
        if camera_raw is None:
            return None

        try:
            camera = int(float(camera_raw))
        except (TypeError, ValueError):
            return None

        # ------------------------------------------------------------------
        # Use the first pedestrian crossing time for camera zero scenarios.
        # ------------------------------------------------------------------
        if camera == 0:
            cross_p1 = row.get("cross_p1_time_s")
            if cross_p1 is not None and pd.notna(cross_p1):
                return float(cross_p1)

        # ------------------------------------------------------------------
        # Use the second pedestrian crossing time for camera one scenarios.
        # ------------------------------------------------------------------
        if camera == 1:
            cross_p2 = row.get("cross_p2_time_s")
            if cross_p2 is not None and pd.notna(cross_p2):
                return float(cross_p2)

        return None

    def _output_path(self, *parts: str) -> str:

        """Build an absolute path inside the output directory.

        Args:
            *parts: Path components to join under ``self.output_dir``.

        Returns:
            The combined path as a string.
        """
        # Centralise output path creation so other methods do not repeat the base directory join.
        return os.path.join(self.output_dir, *parts)

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
            threshold: Threshold used to mark a timestamp bin as pressed.
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
        # ------------------------------------------------------------------
        # Define the cache location for the derived trigger feature table.
        # ------------------------------------------------------------------
        out_csv = os.path.join(self.stats_dir, "trigger_time_series_features.csv")
        if os.path.isfile(out_csv) and not force:
            logger.info(f"Loading cached trigger feature table: {out_csv}")
            return pd.read_csv(out_csv)

        # Discover all participant by video matrices that match the requested trigger column.
        # ------------------------------------------------------------------
        # Build the file pattern used to discover exported participant matrices.
        # ------------------------------------------------------------------
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
        # ------------------------------------------------------------------
        # Iterate over every discovered matrix file and derive features independently.
        # ------------------------------------------------------------------
        for file_path in file_list:
            base = os.path.basename(file_path)
            match = re.search(r"(video_\d+)\.csv$", base, flags=re.IGNORECASE)
            if match is None:
                continue

        # Recover the canonical video identifier from the file name.
        # ------------------------------------------------------------------
        # Extract the canonical video identifier from the file name.
        # ------------------------------------------------------------------
            video_id = match.group(1)
            map_row = self._mapping_row_for_video(video_id)
            if map_row is None:
                logger.warning(f"Skipping {video_id}: missing mapping row")
                continue

        # Resolve the scenario specific cutoff before loading participant samples.
        # ------------------------------------------------------------------
            cutoff = self._cutoff_from_mapping(map_row)

        # Load the participant matrix that contains one timestamp column and one column per participant.
            df = pd.read_csv(file_path)
            if "Timestamp" not in df.columns:
                logger.warning(f"Skipping {video_id}: Timestamp column missing in {file_path}")
                continue

        # Clean timestamps so time based filtering and spacing calculations are reliable.
        # ------------------------------------------------------------------
            df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
            df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

        # Trim each matrix to the pre crossing analysis window when a cutoff is available.
        # ------------------------------------------------------------------
            if cutoff is not None and np.isfinite(cutoff):
                df = df.loc[df["Timestamp"] <= cutoff].copy()

            if df.empty:
                logger.warning(f"Skipping {video_id}: no rows remain after cutoff filtering")
                continue

        # Derive the effective time grid from the retained timestamps.
        # ------------------------------------------------------------------
            unique_ts = np.sort(df["Timestamp"].unique())
            if len(unique_ts) > 1:
                dt_seconds = float(np.nanmedian(np.diff(unique_ts)))
            else:
                dt_seconds = float(common.get_configs("kp_resolution")) / 1000.0

            # Total analysed duration is based on the number of unique bins retained.
            total_duration = float(len(unique_ts) * dt_seconds)

            # Every non timestamp column is treated as a participant specific time series.
            participant_cols = [col for col in df.columns if col != "Timestamp"]

        # Convert each participant column into a trial level summary record.
        # ------------------------------------------------------------------
            for participant_col in participant_cols:
                pm = re.search(r"P(\d+)", str(participant_col))
                if pm is None:
                    continue

                participant = int(pm.group(1))

        # Initialise per participant containers for time series summaries.
        # ------------------------------------------------------------------
                ts_used: List[float] = []
                bin_means: List[float] = []
                pressed_states: List[int] = []
                all_values: List[float] = []

        # Visit each timestamp bin to compute per bin trigger summaries.
        # ------------------------------------------------------------------
                for ts_raw, cell_value in df[["Timestamp", participant_col]].itertuples(index=False, name=None):
                    ts = float(ts_raw)
                    values = self._extract_numeric_values(cell_value)

                    ts_used.append(ts)

        # Use the observed values when the current timestamp bin is populated.
        # ------------------------------------------------------------------
                    if values:
                        bin_mean = float(np.mean(values))
                        pressed = int(any(v > threshold for v in values))
                        all_values.extend(values)
                    else:
                        bin_mean = 0.0
                        pressed = 0

                    # Preserve the per bin summaries so whole trial features can be derived afterward.
                    bin_means.append(bin_mean)
                    pressed_states.append(pressed)

        # Skip participants that still have no usable time bins after cleaning.
        # ------------------------------------------------------------------
                if not ts_used:
                    continue

        # Collapse the bin level summaries into whole trial features.
        # ------------------------------------------------------------------
                mean_raw = float(np.mean(bin_means)) if bin_means else np.nan
                peak_raw = float(np.max(bin_means)) if bin_means else np.nan
                auc_raw = float(np.sum(bin_means) * dt_seconds) if bin_means else np.nan
                unsafe_prop = float(np.mean(pressed_states)) if pressed_states else np.nan
                time_pressed = float(np.sum(pressed_states) * dt_seconds) if pressed_states else np.nan
                switch_count = int(np.sum(np.abs(np.diff(pressed_states)))) if len(pressed_states) > 1 else 0

        # Locate the first press event and the first subsequent release if present.
        # ------------------------------------------------------------------
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
                # ------------------------------------------------------------------
                value_sd = float(np.std(all_values, ddof=1)) if len(all_values) > 1 else np.nan
                n_samples = int(len(all_values))

                # Pull scenario level metadata from the mapping row so the feature table can be
                # merged directly onto trial level analyses later.
                dist_ped_code = _as_float(map_row.get("distPed"))
                dist_ped_m = self._distance_code_to_meters(map_row.get("distPed_m", dist_ped_code))

                yielding = _as_float(map_row.get("yielding"))
                ehmi_on = _as_float(map_row.get("eHMIOn"))
                camera = _as_float(map_row.get("camera"))
                analysis_cutoff_s = float(cutoff) if cutoff is not None else np.nan

                # Package feature values and scenario metadata into one row dictionary.
                # ------------------------------------------------------------------
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
                    "mean_trigger_raw": mean_raw,
                    "mean_trigger_pct": mean_raw * 100.0 if pd.notna(mean_raw) else np.nan,
                    "peak_trigger_raw": peak_raw,
                    "peak_trigger_pct": peak_raw * 100.0 if pd.notna(peak_raw) else np.nan,
                    "auc_trigger_raw_s": auc_raw,
                    "auc_trigger_pct_s": auc_raw * 100.0 if pd.notna(auc_raw) else np.nan,
                    "unsafe_prop": unsafe_prop,
                    "unsafe_prop_pct": unsafe_prop * 100.0 if pd.notna(unsafe_prop) else np.nan,
                    "time_pressed_s": time_pressed,
                    "switch_count": switch_count,
                    "first_press_latency_s": first_press_latency,
                    "first_release_latency_s": first_release_latency,
                    "trigger_value_sd": value_sd,
                }
                records.append(record)

        # Convert all accumulated row dictionaries into a single DataFrame.
        # ------------------------------------------------------------------
        feature_df = pd.DataFrame.from_records(records)
        if feature_df.empty:
            raise ValueError("Trigger feature extraction produced an empty table.")

        # Apply a deterministic ordering before saving the feature table.
        # ------------------------------------------------------------------
        feature_df = feature_df.sort_values(["participant", "video_id"]).reset_index(drop=True)
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
        # ------------------------------------------------------------------
        diff_arr = np.asarray(list(diff), dtype=float)
        diff_arr = diff_arr[np.isfinite(diff_arr)]
        n = int(len(diff_arr))

        # Return early when there are too few observations for paired inference.
        # ------------------------------------------------------------------
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
        # ------------------------------------------------------------------
        mean_diff = float(np.mean(diff_arr))
        sd_diff = float(np.std(diff_arr, ddof=1))
        se_diff = float(sd_diff / np.sqrt(n))
        dfree = n - 1

        # Handle degenerate standard errors without raising an exception.
        # ------------------------------------------------------------------
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
        # ------------------------------------------------------------------
        crit90 = float(student_t.ppf(1.0 - alpha, df=dfree))
        crit95 = float(student_t.ppf(1.0 - alpha / 2.0, df=dfree))

        # Report both 90 percent and 95 percent intervals because the 90 percent interval is
        # directly relevant for equivalence testing.
        ci90_low = float(mean_diff - crit90 * se_diff)
        ci90_high = float(mean_diff + crit90 * se_diff)
        ci95_low = float(mean_diff - crit95 * se_diff)
        ci95_high = float(mean_diff + crit95 * se_diff)

        # Form the two one sided test statistics for the equivalence bounds.
        # ------------------------------------------------------------------
        t_lower = float((mean_diff - low_eq) / se_diff)
        p_lower = float(1.0 - student_t.cdf(t_lower, df=dfree))

        t_upper = float((mean_diff - high_eq) / se_diff)
        p_upper = float(student_t.cdf(t_upper, df=dfree))

        # Combine the two one sided p values into the final TOST decision quantity.
        # ------------------------------------------------------------------
        p_tost = float(max(p_lower, p_upper))
        equivalent = bool((p_lower < alpha) and (p_upper < alpha))

        # Also compute the conventional paired t test against zero for reference.
        # ------------------------------------------------------------------
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
            df["distPed_m"] = self._distance_series_to_meters(df["distPed_m"])

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
            return "Yielding" if int(val) == 1 else "Not yielding"  # pyright: ignore[reportArgumentType]

        def _ehmi_label(val: object) -> str:
            return "eHMI on" if int(val) == 1 else "eHMI off"  # pyright: ignore[reportArgumentType]

        def _visibility_label(val: object) -> str:
            return "Other pedestrian not visible" if int(val) == 1 else "Other pedestrian visible"  # type: ignore

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
            display_label = f"{_yield_label(ctx[0])}, {_ehmi_label(ctx[1])}, {_visibility_label(ctx[2])}"
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

        # ------------------------------------------------------------------
        # Build a faceted equivalence figure.
        # Rows separate visibility, columns separate eHMI, and each panel shows
        # two yielding states. This keeps labels short and publication friendly.
        # ------------------------------------------------------------------
        fig = make_subplots(
            rows=3,
            cols=2,
            specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]],
            subplot_titles=[
                "Overall",
                "Other pedestrian visible | eHMI off",
                "Other pedestrian visible | eHMI on",
                "Other pedestrian not visible | eHMI off",
                "Other pedestrian not visible | eHMI on",
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
                    categoryarray=["Not yielding", "Yielding"],
                    row=r,
                    col=c,
                )

        fig.update_layout(
            template=self.template,
            title="",
            font=dict(family=self.font_family, size=self.font_size + 2),
            margin=dict(l=90, r=30, t=70, b=80),
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
        # ------------------------------------------------------------------
        # Translate raw model term names into cleaner display labels.
        # ------------------------------------------------------------------
        mapping = {
            "Intercept": "Intercept",
            "C(yielding)[T.1]": "Yielding",
            "C(eHMIOn)[T.1]": "eHMI on",
            "C(camera)[T.1]": "Other pedestrian not visible",
            "distPed_m": "Distance (m)",
            "within_score": "Within participant",
            "between_score": "Between participant",
            "Group Var": "Random intercept variance",
            "C(yielding)[T.1]:C(eHMIOn)[T.1]": "Yielding × eHMI on",
            "C(yielding)[T.1]:C(camera)[T.1]": "Yielding × visibility",
            "C(eHMIOn)[T.1]:C(camera)[T.1]": "eHMI on × visibility",
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
        # ------------------------------------------------------------------
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

    def _fit_model_with_fallbacks(self, df: pd.DataFrame, formula: str, group_col: str = "participant",
                                  re_formula: Optional[str] = None):

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

        # Prepare the ordered list of model fitting strategies.
        # Prefer the simpler random intercept model first because the random
        # slope on distance is the part most likely to be numerically unstable.
        # ------------------------------------------------------------------
        attempts = [("mixed_random_intercept", None)]
        if re_formula is not None:
            attempts.append(("mixed_random_slope", re_formula))  # pyright: ignore[reportArgumentType]

        # ------------------------------------------------------------------
        # Try each modelling strategy until one converges successfully.
        # ------------------------------------------------------------------
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
        # ------------------------------------------------------------------
        # Pull the fitted coefficients and uncertainty estimates from the model result.
        # ------------------------------------------------------------------
        params = fit.params
        pvalues = fit.pvalues
        conf = fit.conf_int()
        bse = fit.bse

        # ------------------------------------------------------------------
        # Reshape coefficient vectors into a tidy tabular format.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Optionally retain only the subset of coefficients relevant for reporting.
        # ------------------------------------------------------------------
        if keep_terms is not None:
            keep_terms = set(keep_terms)
            coef_df = coef_df.loc[coef_df["term"].isin(keep_terms)].copy()
        return coef_df.reset_index(drop=True)  # type: ignore

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
        # ------------------------------------------------------------------
        # Declare the minimum columns needed for within versus between analyses.
        # ------------------------------------------------------------------
        needed = ["participant", "crossing_risk", "yielding", "eHMIOn", "camera", "distPed_m"]
        results: List[pd.DataFrame] = []

        # ------------------------------------------------------------------
        # Run the same decomposition and model for each questionnaire item.
        # ------------------------------------------------------------------
        for q_col in ["Q1", "Q2", "Q3"]:
            current = trial_df.copy()
            current[q_col] = pd.to_numeric(current[q_col], errors="coerce")
            current["crossing_risk"] = pd.to_numeric(current["crossing_risk"], errors="coerce")
            current = current.dropna(subset=needed + [q_col])
            if current.empty:
                logger.warning(f"Skipping within/between model for {q_col}: no valid rows")
                continue

        # ------------------------------------------------------------------
        # Decompose the rating into between participant and within participant components.
        # ------------------------------------------------------------------
            current["between_score"] = current.groupby("participant")[q_col].transform("mean")
            current["within_score"] = current[q_col] - current["between_score"]

        # ------------------------------------------------------------------
        # Specify the fixed effect structure used by the current model.
        # ------------------------------------------------------------------
            formula = (
                "crossing_risk ~ within_score + between_score + C(yielding) + C(eHMIOn) + "
                "C(camera) + distPed_m"
            )
        # ------------------------------------------------------------------
        # Estimate the current feature model with robust fallbacks.
        # ------------------------------------------------------------------
            fit, model_name = self._fit_model_with_fallbacks(
                current,
                formula=formula,
                group_col="participant",
                re_formula="~distPed_m",
            )
            if fit is None:
                continue

        # ------------------------------------------------------------------
        # Extract the coefficients that will be exported and plotted.
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Combine the per rating model outputs into one coefficient table.
        # ------------------------------------------------------------------
        results_df = pd.concat(results, ignore_index=True)
        self._save_table(results_df, "within_between_models_crossing_risk.csv")

        # ------------------------------------------------------------------
        # Build a coefficient plot for the within and between estimates.
        # ------------------------------------------------------------------
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
                               save: bool = True) -> pd.DataFrame:

        """Merge derived trigger features onto a trial level table.

        Args:
            trial_df: Base trial level DataFrame.
            feature_df: Optional precomputed trigger feature table.
            save: Whether to write the merged table to disk.

        Returns:
            The enriched trial level DataFrame.
        """
        # ------------------------------------------------------------------
        # Generate trigger features on demand when they were not supplied.
        # ------------------------------------------------------------------
        if feature_df is None:
            feature_df = self.build_trigger_feature_table()

        # ------------------------------------------------------------------
        # Copy both input tables to avoid mutating caller owned DataFrames.
        # ------------------------------------------------------------------
        left = trial_df.copy()
        right = feature_df.copy()

        # ------------------------------------------------------------------
        # Standardise merge key types before joining the tables.
        # ------------------------------------------------------------------
        left["participant"] = pd.to_numeric(left["participant"], errors="coerce")
        right["participant"] = pd.to_numeric(right["participant"], errors="coerce")

        left["video_id"] = left["video_id"].astype(str)
        right["video_id"] = right["video_id"].astype(str)

        # ------------------------------------------------------------------
        # List the derived trigger feature columns that may be appended.
        # ------------------------------------------------------------------
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
        ]

        keep_cols = [c for c in merge_cols if c in right.columns]

        # ------------------------------------------------------------------
        # Join the feature table onto the trial table while retaining all trial rows.
        # ------------------------------------------------------------------
        enriched = left.merge(
            right[keep_cols],
            on=["participant", "video_id"],
            how="left",
        )

        # ------------------------------------------------------------------
        # Persist the enriched table when the caller requests an output file.
        # ------------------------------------------------------------------
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
        # ------------------------------------------------------------------
        # Use the default set of trigger outcomes when none are specified.
        # ------------------------------------------------------------------
        if feature_outcomes is None:
            feature_outcomes = [
                "peak_trigger_pct",
                "auc_trigger_pct_s",
                "switch_count",
                "first_press_latency_s",
                "unsafe_prop_pct",
            ]

        # ------------------------------------------------------------------
        # Collect coefficient tables and descriptive summaries across all feature models.
        # ------------------------------------------------------------------
        all_coef_frames: List[pd.DataFrame] = []
        outcome_summary_frames: List[pd.DataFrame] = []

        # ------------------------------------------------------------------
        # Fit one model and build one descriptive summary per trigger feature.
        # ------------------------------------------------------------------
        for outcome in feature_outcomes:
            current = trial_df.copy()
            current[outcome] = pd.to_numeric(current[outcome], errors="coerce")
            current = current.dropna(
                subset=["participant", outcome, "yielding", "eHMIOn", "camera", "distPed_m"]
            )
            if current.empty:
                logger.warning(f"Skipping feature model for {outcome}: no valid rows")
                continue

            # ------------------------------------------------------------------
            # Aggregate the current feature over distance and scenario context for plotting.
            # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Write and visualise descriptive feature summaries when any are available.
        # ------------------------------------------------------------------
        if outcome_summary_frames:
            summary_df = pd.concat(outcome_summary_frames, ignore_index=True)
            self._save_table(summary_df, "trigger_feature_distance_profiles.csv")

        # ------------------------------------------------------------------
        # Define human readable labels for the requested trigger features.
        # ------------------------------------------------------------------
            feature_labels = {
                "peak_trigger_pct": "Peak trigger (0–100)",
                "auc_trigger_pct_s": "Trigger AUC",
                "switch_count": "Switch count",
                "first_press_latency_s": "First press latency (s)",
                "unsafe_prop_pct": "Unsafe time (%)",
            }
            summary_df["feature_label"] = summary_df["feature"].map(feature_labels).fillna(summary_df["feature"])
            summary_df["yielding_label"] = summary_df["yielding"].map({0: "Not yielding", 1: "Yielding"})
            summary_df["eHMI_label"] = summary_df["eHMIOn"].map({0: "eHMI off", 1: "eHMI on"})
            summary_df["visibility_label"] = summary_df["camera"].map(
                {0: "Other pedestrian visible", 1: "Other pedestrian not visible"}
            )
            summary_df["panel_title"] = summary_df["visibility_label"].str.replace(
                "Other pedestrian ", "", regex=False
            ) + " | " + summary_df["eHMI_label"]

            # --------------------------------------------------------------
            # Log compact descriptive insights that help interpret the profile figure.
            # --------------------------------------------------------------
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
                    yielding_txt = "Yielding" if strongest["yielding"] == 1 else "Not yielding"
                    ehmi_txt = "eHMI on" if strongest["eHMIOn"] == 1 else "eHMI off"
                    vis_txt = (
                        "Other pedestrian not visible"
                        if strongest["camera"] == 1 else "Other pedestrian visible"
                    )
                    logger.info(
                        f"{feature_name} strongest profile change: {yielding_txt}, "
                        f"{ehmi_txt}, {vis_txt} changed by {strongest['delta']:.3f} "
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

        # ------------------------------------------------------------------
        # Build a faceted profile plot so each panel only contains two lines.
        # ------------------------------------------------------------------
            feature_order = [
                feat for feat in feature_outcomes if feat in summary_df["feature"].unique().tolist()
            ]
            panel_order = [
                {"eHMIOn": 0, "camera": 0, "title": "Visible | eHMI off"},
                {"eHMIOn": 1, "camera": 0, "title": "Visible | eHMI on"},
                {"eHMIOn": 0, "camera": 1, "title": "Not visible | eHMI off"},
                {"eHMIOn": 1, "camera": 1, "title": "Not visible | eHMI on"},
            ]
            yielding_styles = {
                0: {
                    "name": "Not yielding",
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
                    margin=dict(l=190, r=70, t=80, b=90),
                    legend=dict(
                        title_text="Scenario yielding state",
                        orientation="h",
                        x=0.5,
                        xanchor="center",
                        y=1.04,
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

        # ------------------------------------------------------------------
        # Stop with a clear error when every feature model fails.
        # ------------------------------------------------------------------
        if not all_coef_frames:
            raise ValueError("No trigger feature models were successfully fitted.")

        # ------------------------------------------------------------------
        # Concatenate the fitted feature model coefficients into one table.
        # ------------------------------------------------------------------
        coef_df = pd.concat(all_coef_frames, ignore_index=True)
        self._save_table(coef_df, "trigger_feature_model_coefficients.csv")

        # ------------------------------------------------------------------
        # Create the final cross feature coefficient summary plot.
        # ------------------------------------------------------------------
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

    def run_all(self, trial_df: pd.DataFrame, equivalence_margin: float = 5.0) -> Dict[str, pd.DataFrame]:

        """Run the full advanced statistics pipeline end to end.

        Args:
            trial_df: Trial level input DataFrame.
            equivalence_margin: Symmetric equivalence margin used in the
                near versus far TOST comparison.

        Returns:
            A dictionary containing the main intermediate and final result
            tables generated by the pipeline.
        """
        # ------------------------------------------------------------------
        # Log the start of the end to end pipeline for traceability.
        # ------------------------------------------------------------------
        logger.info("Starting advanced statistics pipeline")

        # ------------------------------------------------------------------
        # Step 1: derive participant by video trigger features.
        # ------------------------------------------------------------------
        feature_df = self.build_trigger_feature_table()
        enriched_trial_df = self.merge_trigger_features(trial_df, feature_df=feature_df, save=True)  # type: ignore

        # ------------------------------------------------------------------
        # Step 2: test near versus far equivalence on the enriched trial data.
        # ------------------------------------------------------------------
        equivalence_df = self.run_equivalence_tests(
            enriched_trial_df,
            outcome="crossing_risk",
            equivalence_margin=equivalence_margin,
        )

        # ------------------------------------------------------------------
        # Step 3: fit the trigger feature models and export their figures.
        # ------------------------------------------------------------------
        feature_coef_df = self.run_feature_models_and_figures(enriched_trial_df)

        # ------------------------------------------------------------------
        # Step 4: fit the within versus between participant rating models.
        # ------------------------------------------------------------------
        within_between_df = self.run_within_between_models(enriched_trial_df)
        logger.info("Finished advanced statistics pipeline")

        # ------------------------------------------------------------------
        # Return all major outputs so callers can inspect or reuse them programmatically.
        # ------------------------------------------------------------------
        return {
            "features": feature_df,
            "trial_enriched": enriched_trial_df,
            "equivalence": equivalence_df,
            "feature_coefficients": feature_coef_df,
            "within_between": within_between_df,
        }
