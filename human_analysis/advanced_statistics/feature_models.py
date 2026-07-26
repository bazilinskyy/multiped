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


ADVANCED_STATS_SPECIFICATION = "reviewer_response_v5_participant_bootstrap"

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


class FeatureModelMixin:
    """Focused method group extracted without changing calculation logic."""

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
