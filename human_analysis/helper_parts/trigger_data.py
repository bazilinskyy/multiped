import pandas as pd
import os
import shutil
import glob
import plotly.graph_objects as go
import plotly as py
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
# For OneEuroFilter, see https://github.com/casiez/OneEuroFilter
from OneEuroFilter import OneEuroFilter
import common
from custom_logger import CustomLogger
import re
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, t
from ..utils.HMD_helper import HMD_yaw
from ..utils.tools import Tools
from datetime import datetime
import ast
import math
from typing import Dict, Optional
import statsmodels.formula.api as smf


logger = CustomLogger(__name__)  # use custom logger

HMD_class = HMD_yaw()
extra_class = Tools()

# Consts
plotly_template = common.get_configs("plotly_template")
font_size = common.get_configs("font_size")
font_family = common.get_configs("font_family")



from ..utils.parsing import parse_numeric_list
from ..utils.distance import distance_code_to_metres, distance_codes_to_metres


class TriggerDataMixin:
    """Focused method group extracted without changing calculation logic."""

    def heat_plot(self, folder_path: str, mapping_df: pd.DataFrame, relation: str = "ratio",
                  colorscale: str = "Viridis", summary_func=np.mean,
                  trigger_threshold: float = 0.05):
        """
        Compute summary values per video CSV, rename axes using `mapping_df`,
        and show a Plotly heatmap of pairwise relations (no numbers, no colorbar).

        Additionally, this function writes a 'trigger_summary.csv' file into
        `folder_path` with columns:
            ['label', 'avg_trigger', 'sd_trigger', 'n_samples']

        Parameters
        ----------
        folder_path : str
            Folder with 'participant_TriggerValueRight_video_*.csv'.
        mapping_df : pd.DataFrame
            Must include columns: 'video_id', 'condition_name', 'camera',
            'cross_p1_time_s', 'cross_p2_time_s'.
            The cutoff time used per video is cross_p1_time_s if camera==0,
            otherwise cross_p2_time_s. Only rows with Timestamp <= cutoff are used.
        relation : {'ratio','diff'}
            'ratio' uses vb/va; 'diff' uses vb - va.
        colorscale : str
            Plotly colorscale name.
        summary_func : callable
            Aggregator applied to binary trigger-pressed states per condition
            (default: np.mean). With the default, `avg_trigger` is the
            proportion of analysed participant-time bins for which the trigger
            value was above `trigger_threshold`.
        trigger_threshold : float
            Trigger values strictly greater than this threshold are counted as
            pressed. The default of 0.05 is used because the trigger was
            pressure-sensitive and light contact could produce small values.

        Returns
        -------
        averages : dict
            {mapped_label: summary_value}  (means per condition; for compatibility)
        relation_df : pd.DataFrame
            Pairwise matrix (mapped_label × mapped_label)
        fig : plotly.graph_objects.Figure
            The heatmap figure.
        """
        # ---- Validate mapping_df ----
        required_cols = {"video_id", "condition_name", "camera", "cross_p1_time_s", "cross_p2_time_s"}
        if not required_cols.issubset(set(mapping_df.columns)):
            raise ValueError(f"mapping_df must have columns {required_cols}, got {list(mapping_df.columns)}")

        # Normalize dtypes
        md = mapping_df.copy()
        md["video_id"] = md["video_id"].astype(str)

        # Force numerics for times; invalids -> NaN
        md["cross_p1_time_s"] = pd.to_numeric(md["cross_p1_time_s"], errors="coerce")
        md["cross_p2_time_s"] = pd.to_numeric(md["cross_p2_time_s"], errors="coerce")
        md["camera"] = pd.to_numeric(md["camera"], errors="coerce").astype("Int64")

        # Build lookup: video_id -> {label, camera, p1, p2}
        mapping_info = {
            row["video_id"]: {
                "label": str(row["condition_name"]),
                "camera": int(row["camera"]) if pd.notna(row["camera"]) else None,
                "p1": float(row["cross_p1_time_s"]) if pd.notna(row["cross_p1_time_s"]) else None,
                "p2": float(row["cross_p2_time_s"]) if pd.notna(row["cross_p2_time_s"]) else None,
            }
            for _, row in md.iterrows()
        }

        # ---- Find files ----
        pattern = os.path.join(folder_path, "participant_TriggerValueRight_video_*.csv")
        file_list = glob.glob(pattern)
        if not file_list:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        # Sort by numeric video index if present (video_123), else alphabetically
        def _video_key(fn):
            m = re.search(r"video_(\d+)\.csv$", os.path.basename(fn), flags=re.IGNORECASE)
            return (0, int(m.group(1))) if m else (1, os.path.basename(fn).lower())

        file_list = sorted(file_list, key=_video_key)

        # ---- Helper: extract video_id from filename ----
        def _extract_video_id(filename: str) -> str:
            base = os.path.basename(filename)
            m = re.search(r"(video_\d+)\.csv$", base, flags=re.IGNORECASE)
            return m.group(1) if m else base

        # ---- Collect trigger-pressed states per condition label ----
        # per_label_values[label] stores one binary value per participant-time bin:
        # 1 = at least one trigger value in that bin was above trigger_threshold;
        # 0 = no trigger value in that bin was above trigger_threshold.
        # This makes avg_trigger the proportion of analysed time marked as unsafe,
        # rather than the mean trigger pressure/intensity.
        per_label_values: Dict[str, np.ndarray] = {}
        per_label_raw_values: Dict[str, np.ndarray] = {}

        for file_path in file_list:
            video_id = _extract_video_id(file_path)
            info = mapping_info.get(video_id)

            # Determine cutoff (if mapping missing or incomplete, we'll fall back to no cutoff)
            cutoff = None
            label = video_id
            if info:
                label = info["label"]
                cam = info["camera"]
                if cam == 0:
                    cutoff = info["p1"]
                elif cam == 1:
                    cutoff = info["p2"]
                else:
                    logger.warning(f"⚠️ camera not 0/1 for {video_id}; using full data (no cutoff).")

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"⚠️ Error reading {file_path}: {e}")
                continue

            if "Timestamp" not in df.columns:
                logger.warning(f"⚠️ 'Timestamp' column missing in {file_path}; using full data.")
                ts_filtered = df
            else:
                # Coerce Timestamp to numeric seconds, drop rows with invalid timestamps
                ts = pd.to_numeric(df["Timestamp"], errors="coerce")
                valid = ts.notna()  # type: ignore
                df = df.loc[valid].copy()
                ts = ts.loc[valid]  # type: ignore

                if cutoff is not None and np.isfinite(cutoff):
                    mask = ts <= float(cutoff)
                    ts_filtered = df.loc[mask].copy()
                else:
                    ts_filtered = df

            # If nothing remains after filtering, skip
            if ts_filtered.empty:
                logger.warning(f"⚠️ No rows after time filtering for {file_path}; skipping.")
                continue

            # Drop Timestamp before aggregating values
            ts_filtered = ts_filtered.drop(columns=["Timestamp"], errors="ignore")

            # Parse list-like cells. For the manuscript's primary perceived-risk
            # measure, each participant-time bin is converted into a binary state:
            # pressed = any trigger value in that bin is above the threshold.
            pressed_states = []
            raw_values = []
            for col in ts_filtered.columns:
                for val in ts_filtered[col]:
                    nums = self._extract_numeric_values_from_cell(val)
                    raw_values.extend(nums)
                    pressed_states.append(1.0 if any(x > trigger_threshold for x in nums) else 0.0)

            if not pressed_states:
                logger.warning(f"⚠️ No analysable trigger bins found (after cutoff) in {file_path}; skipping.")
            else:
                vals = np.array(pressed_states, dtype=float)
                raw_vals = np.array(raw_values, dtype=float) if raw_values else np.array([], dtype=float)
                if label not in per_label_values:
                    per_label_values[label] = vals
                    per_label_raw_values[label] = raw_vals
                else:
                    per_label_values[label] = np.concatenate([per_label_values[label], vals])
                    per_label_raw_values[label] = np.concatenate([per_label_raw_values[label], raw_vals])

        if not per_label_values:
            raise ValueError("No valid numeric data found in videos (after applying time cutoffs).")

        # ---- Build trigger summary (unsafe proportion + SD + n) ----
        trigger_summary_df = (
            pd.DataFrame(
                [
                    {
                        "label": label,
                        # Proportion of analysed participant-time bins marked unsafe.
                        "avg_trigger": float(summary_func(vals)),
                        "sd_trigger": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                        "n_samples": int(len(vals)),
                        "n_trigger_bins": int(len(vals)),
                        "n_raw_trigger_samples": int(len(per_label_raw_values.get(label, []))),
                        # Kept for diagnostics only; this is no longer the primary risk measure.
                        "mean_trigger_intensity": (
                            float(np.mean(per_label_raw_values[label]))
                            if len(per_label_raw_values.get(label, [])) > 0 else np.nan
                        ),
                        "sd_trigger_intensity": (
                            float(np.std(per_label_raw_values[label], ddof=1))
                            if len(per_label_raw_values.get(label, [])) > 1 else np.nan
                        ),
                        "trigger_threshold": float(trigger_threshold),
                    }
                    for label, vals in per_label_values.items()
                ]
            )
            .sort_values("label")
            .reset_index(drop=True)
        )

        # Backwards-compatible dict of means only (for relation matrix)
        averages = dict(
            zip(trigger_summary_df["label"], trigger_summary_df["avg_trigger"])
        )

        # ---- Save trigger_summary.csv ----
        trigger_summary_path = os.path.join(folder_path, "trigger_summary.csv")
        trigger_summary_df.to_csv(trigger_summary_path, index=False)
        logger.info(f"Saved trigger summary (unsafe proportion + SD) to: {trigger_summary_path}")

        # ---- Build pairwise relation matrix ----
        labels = list(averages.keys())
        n = len(labels)
        mat = np.full((n, n), np.nan, dtype=float)

        for i, a in enumerate(labels):
            va = averages[a]
            for j, b in enumerate(labels):
                vb = averages[b]
                if relation == "ratio":
                    mat[i, j] = (vb / va) if va not in (0, None) else np.nan
                elif relation == "diff":
                    mat[i, j] = vb - va
                else:
                    raise ValueError("relation must be 'ratio' or 'diff'")

        relation_df = pd.DataFrame(mat, index=labels, columns=labels)

        # ---- Add formatted decimal text for cells ----
        text_values = np.where(
            np.isnan(relation_df.values),
            "",
            np.round(relation_df.values, 2).astype(str)
        )

        # ---- Plotly heatmap (no cell text, no colorbar) ----
        fig = go.Figure(
            data=go.Heatmap(
                z=relation_df.values,
                x=relation_df.columns,
                y=relation_df.index,
                text=text_values,           # show decimals in cells
                texttemplate="%{text}",     # ensures text is rendered
                colorscale=colorscale,
                showscale=False,  # remove colorbar
                hovertemplate=(
                    "<b>%{y}</b> → <b>%{x}</b><br>"
                    f"{relation}: %{{z:.6f}}<extra></extra>"
                )
            )
        )

        fig.update_layout(
            title="",
            xaxis=dict(
                title="",
                tickangle=45,
                tickfont=dict(size=18, color="black"),
                titlefont=dict(size=16, color="black")
            ),
            yaxis=dict(
                title="",
                autorange="reversed",
                tickfont=dict(size=18, color="black"),
                titlefont=dict(size=16, color="black")
            ),
            margin=dict(l=80, r=40, t=60, b=90),
            width=2400,
            height=2400,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        self.save_plotly(fig, 'heatmap', height=2400, width=2400, save_final=True)

        # Optional: also save the pairwise relation matrix
        relation_path = os.path.join(folder_path, "trigger_relation_matrix.csv")
        relation_df.to_csv(relation_path)
        logger.info(f"Saved trigger relation matrix to: {relation_path}")

        return averages, relation_df, fig

    @staticmethod
    def _extract_numeric_values_from_cell(value):
        """Return finite numeric values from a list encoded CSV cell."""
        return parse_numeric_list(value)

    @classmethod
    def _trigger_pressed_state(cls, value, threshold=0.05):
        """Return 1 when any finite trigger value in a matrix cell is above threshold."""
        values = cls._extract_numeric_values_from_cell(value)
        return 1.0 if any(item > threshold for item in values) else 0.0

    @classmethod
    def _threshold_trigger_matrix(cls, df, threshold=0.05):
        """Convert list-valued trigger matrix cells into binary pressed-state cells.

        Each non-timestamp cell becomes ``[1.0]`` when any finite trigger value in
        that participant-time bin is greater than ``threshold`` and ``[0.0]``
        otherwise. Keeping a list in each cell preserves compatibility with the
        existing averaging and time-series helper functions.
        """
        out = df.copy()
        for col in out.columns:
            if col == "Timestamp":
                continue
            out[col] = out[col].apply(lambda value: [cls._trigger_pressed_state(value, threshold)])
        return out

    def _compute_trial_level_trigger_summary(self, trigger_matrices_dir, mapping_df, trigger_threshold: float = 0.05):
        """Build participant x video unsafe-time summaries from exported trigger matrices."""
        mapping_info = mapping_df.copy()
        for col in ["video_id", "condition_name"]:
            mapping_info[col] = mapping_info[col].astype(str)
        for col in ["camera", "cross_p1_time_s", "cross_p2_time_s", "distPed", "yielding", "eHMIOn"]:
            if col in mapping_info.columns:
                mapping_info[col] = pd.to_numeric(mapping_info[col], errors="coerce")

        records = []
        pattern = os.path.join(trigger_matrices_dir, "participant_TriggerValueRight_video_*.csv")
        for fp in sorted(glob.glob(pattern)):
            base = os.path.basename(fp)
            m = re.search(r"(video_\d+)\.csv$", base, flags=re.IGNORECASE)
            if not m:
                continue
            video_id = m.group(1)
            map_row = mapping_info.loc[mapping_info["video_id"] == video_id]
            if map_row.empty:
                continue
            row0 = map_row.iloc[0]
            cutoff = None
            camera = row0.get("camera")
            if pd.notna(camera):
                camera = int(camera)
                if camera == 0 and pd.notna(row0.get("cross_p1_time_s")):
                    cutoff = float(row0["cross_p1_time_s"])
                elif camera == 1 and pd.notna(row0.get("cross_p2_time_s")):
                    cutoff = float(row0["cross_p2_time_s"])

            df = pd.read_csv(fp)
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
                df = df.dropna(subset=["Timestamp"])
                if cutoff is not None and np.isfinite(cutoff):
                    df = df[df["Timestamp"] <= cutoff]

            participant_cols = [c for c in df.columns if c != "Timestamp"]
            for participant_col in participant_cols:
                pm = re.search(r"P(\d+)", str(participant_col))
                if not pm:
                    continue
                participant = int(pm.group(1))
                raw_values = []
                pressed_states = []
                for cell in df[participant_col].tolist():
                    cell_values = self._extract_numeric_values_from_cell(cell)
                    raw_values.extend(cell_values)
                    pressed_states.append(1.0 if any(v > trigger_threshold for v in cell_values) else 0.0)

                if not pressed_states:
                    unsafe_prop = np.nan
                    sd_unsafe_prop = np.nan
                    n_bins = 0
                else:
                    state_arr = np.asarray(pressed_states, dtype=float)
                    unsafe_prop = float(np.mean(state_arr))
                    sd_unsafe_prop = float(np.std(state_arr, ddof=1)) if len(state_arr) > 1 else np.nan
                    n_bins = int(len(state_arr))

                if raw_values:
                    raw_arr = np.asarray(raw_values, dtype=float)
                    mean_trigger_intensity = float(np.mean(raw_arr))
                    sd_trigger_intensity = float(np.std(raw_arr, ddof=1)) if len(raw_arr) > 1 else np.nan
                    n_raw_samples = int(len(raw_arr))
                else:
                    mean_trigger_intensity = np.nan
                    sd_trigger_intensity = np.nan
                    n_raw_samples = 0

                records.append({
                    "participant": participant,
                    "video_id": video_id,
                    "condition_name": str(row0["condition_name"]),
                    # avg_trigger is retained as the public column used downstream,
                    # but it now means unsafe-time proportion, not mean trigger intensity.
                    "avg_trigger": unsafe_prop,
                    "sd_trigger": sd_unsafe_prop,
                    "n_trigger_samples": n_bins,
                    "n_trigger_bins": n_bins,
                    "n_raw_trigger_samples": n_raw_samples,
                    "mean_trigger_intensity": mean_trigger_intensity,
                    "sd_trigger_intensity": sd_trigger_intensity,
                    "trigger_threshold": float(trigger_threshold),
                })

        if not records:
            raise ValueError(
                "No participant-level trigger matrices were found. Run plot_column/heat_plot first so "
                "participant_TriggerValueRight_video_*.csv files exist in the output folder."
            )
        return pd.DataFrame.from_records(records)

    def load_and_average_Q2(
        self,
        trigger_summary_csv: str,
        responses_root: str,
        mapping_df: pd.DataFrame,
        n_participants: int = 50,
        response_col_index: int = 2,
        save_combined: bool = True,
        trigger_threshold: float = 0.05,
        trigger_matrices_dir: Optional[str] = None,
        ratings_df: Optional[pd.DataFrame] = None,
    ):
        """Merge trigger risk with cached or raw participant Q1/Q2/Q3 ratings."""
        trigger_df = pd.read_csv(trigger_summary_csv)
        if "label" in trigger_df.columns and "condition_name" not in trigger_df.columns:
            trigger_df = trigger_df.rename(columns={"label": "condition_name"})
        if "condition_name" not in trigger_df.columns or "avg_trigger" not in trigger_df.columns:
            raise ValueError(
                "trigger_summary_csv must contain condition_name or label, plus avg_trigger."
            )
        if "sd_trigger" not in trigger_df.columns:
            trigger_df["sd_trigger"] = np.nan
        trigger_df["condition_name"] = trigger_df["condition_name"].astype(str)

        map_cols = [
            "video_id", "condition_name", "distPed", "yielding", "eHMIOn", "camera",
            "cross_p1_time_s", "cross_p2_time_s"
        ]
        missing_map = [c for c in map_cols if c not in mapping_df.columns]
        if missing_map:
            raise ValueError(f"mapping_df missing required columns: {missing_map}")
        map_df = mapping_df[map_cols].copy()
        map_df["video_id"] = map_df["video_id"].astype(str)
        map_df["condition_name"] = map_df["condition_name"].astype(str)

        if ratings_df is None:
            ratings_df = self.load_trial_ratings(
                responses_root=responses_root,
                n_participants=n_participants,
                response_col_index=response_col_index,
            )
        else:
            ratings_df = ratings_df.copy()
            required_rating_columns = {"participant", "video_id", "Q1", "Q2", "Q3"}
            missing_ratings = sorted(required_rating_columns.difference(ratings_df.columns))
            if missing_ratings:
                raise ValueError(
                    f"ratings_df missing required columns: {missing_ratings}"
                )
            ratings_df["video_id"] = ratings_df["video_id"].astype(str)
        for q_col in ["Q1", "Q2", "Q3"]:
            ratings_df[q_col] = pd.to_numeric(ratings_df[q_col], errors="coerce")

        output_dir = os.path.dirname(trigger_summary_csv) or self.output_folder
        trigger_matrices_dir = trigger_matrices_dir or output_dir
        participant_trigger_df = self._compute_trial_level_trigger_summary(
            trigger_matrices_dir,
            map_df,
            trigger_threshold=trigger_threshold,
        )

        trial_df = ratings_df.merge(map_df, on="video_id", how="left")
        trial_df = trial_df.merge(
            participant_trigger_df[[
                "participant", "video_id", "avg_trigger", "sd_trigger", "n_trigger_samples",
                "n_trigger_bins", "n_raw_trigger_samples", "mean_trigger_intensity",
                "sd_trigger_intensity", "trigger_threshold"
            ]],
            on=["participant", "video_id"],
            how="left",
        )
        # fallback from condition-level trigger summary if a participant-level row is missing
        trial_df = trial_df.merge(
            trigger_df[["condition_name", "avg_trigger", "sd_trigger"]].rename(
                columns={"avg_trigger": "avg_trigger_condition", "sd_trigger": "sd_trigger_condition"}),  # type:ignore
            on="condition_name",
            how="left",
        )
        trial_df["avg_trigger"] = trial_df["avg_trigger"].fillna(trial_df["avg_trigger_condition"])
        trial_df["sd_trigger"] = trial_df["sd_trigger"].fillna(trial_df["sd_trigger_condition"])
        trial_df = trial_df.drop(columns=["avg_trigger_condition", "sd_trigger_condition"], errors="ignore")
        # Preserve the raw mapping code in distPed and store physical distance
        # separately. This prevents downstream code from mistaking 2 m or 4 m
        # for raw codes and multiplying them a second time.
        trial_df["distPed_m"] = self._distance_series_to_meters(trial_df["distPed"])
        # crossing_risk is the percentage of analysed participant-time bins for
        # which the trigger was pressed (> trigger_threshold), matching the paper.
        trial_df["crossing_risk"] = pd.to_numeric(trial_df["avg_trigger"], errors="coerce") * 100.0
        trial_df["crossing_risk_sd"] = pd.to_numeric(trial_df["sd_trigger"], errors="coerce") * 100.0

        condition_df = (
            trial_df
            .groupby("condition_name", as_index=False)
            .agg(
                avg_trigger=("avg_trigger", "mean"),
                std_trigger=("avg_trigger", "std"),
                mean_Q1=("Q1", "mean"),
                std_Q1=("Q1", "std"),
                mean_Q2=("Q2", "mean"),
                std_Q2=("Q2", "std"),
                mean_Q3=("Q3", "mean"),
                std_Q3=("Q3", "std"),
                n_trials=("Q2", "size"),
                mean_trigger_intensity=("mean_trigger_intensity", "mean"),
                trigger_threshold=("trigger_threshold", "first"),
            )
            .sort_values("condition_name")
            .reset_index(drop=True)
        )

        if save_combined:
            os.makedirs(output_dir, exist_ok=True)
            trial_path = os.path.join(output_dir, "trial_level_trigger_Q123.csv")
            cond_path = os.path.join(output_dir, "condition_level_trigger_Q123.csv")
            trial_df.to_csv(trial_path, index=False)
            condition_df.to_csv(cond_path, index=False)
            logger.info(f"Saved trial-level trigger + Q1/Q2/Q3 data to: {trial_path}")
            logger.info(f"Saved condition-level trigger + Q1/Q2/Q3 data to: {cond_path}")

        return trial_df, condition_df
