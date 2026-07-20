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


class DistanceAnalysisMixin:
    """Focused method group extracted without changing calculation logic."""

    def analyze_and_plot_distance_effect_plotly(self, mapping_df, condition_df, out_dir=None, trial_df=None):
        """
        Merge condition-level averages with distance, yielding, eHMI, and camera,
        compute full-factorial summaries (means + SDs), and create Plotly figures.

        Assumes
        -------
        condition_df has at least:
            ['condition_name', 'avg_trigger', 'std_trigger',
             'mean_Q1', 'std_Q1',
             'mean_Q2', 'std_Q2',
             'mean_Q3', 'std_Q3']
            - avg_trigger: mean proportion of analysed time bins where the trigger was pressed.
            - std_trigger: SD of that thresholded trigger-based measure per condition.
            - mean_Q1/Q2/Q3: mean responses per condition (0–100).
            - std_Q1/Q2/Q3: SD of Q1/Q2/Q3 per condition.

        mapping_df has at least:
            ['condition_name', 'distPed', 'yielding', 'eHMIOn', 'camera']
        """

        # Helper: turn "var=value" facet titles into just "value"
        facet_font = dict(size=font_size, family=font_family)

        def _strip_facet_equals(fig):
            fig.for_each_annotation(
                lambda a: a.update(
                    text=a.text.split("=", 1)[-1].strip(),
                    font=facet_font,
                )
            )

        # --- REQUIRED columns ---
        required_cols = ["condition_name", "distPed", "yielding", "eHMIOn", "camera"]
        missing = [c for c in required_cols if c not in mapping_df.columns]
        if missing:
            raise ValueError(f"mapping_df missing: {missing}")

        needed_cond_cols = [
            "condition_name",
            "avg_trigger", "std_trigger",
            "mean_Q1", "std_Q1",
            "mean_Q2", "std_Q2",
            "mean_Q3", "std_Q3",
        ]
        missing2 = [c for c in needed_cond_cols if c not in condition_df.columns]
        if missing2:
            raise ValueError(f"condition_df missing: {missing2}")

        # --- Prepare mapping info ---
        mapping_df = mapping_df.copy()
        mapping_df["condition_name"] = mapping_df["condition_name"].astype(str)

        cond_plot_df = condition_df.copy()
        cond_plot_df["condition_name"] = cond_plot_df["condition_name"].astype(str)

        # Merge mapping info onto condition-level data
        cond_plot_df = cond_plot_df.merge(
            mapping_df[required_cols],
            on="condition_name",
            how="left",
        )

        # Convert the raw mapping codes once to actual metres (2, 4, 6, 8, 10).
        cond_plot_df["distPed_m"] = self._distance_series_to_meters(cond_plot_df["distPed"])

        # Scale thresholded unsafe-time proportion to 0–100.
        cond_plot_df["crossing_risk"] = cond_plot_df["avg_trigger"] * 100.0
        cond_plot_df["crossing_risk_sd"] = cond_plot_df["std_trigger"] * 100.0

        # Drop if some factors missing
        cond_plot_df = cond_plot_df.dropna(
            subset=["distPed_m", "yielding", "eHMIOn", "camera"]
        )

        # Label maps for binary factors (0/1 → text)
        label_map_yield = {0: "Non-yielding", 1: "Yielding"}
        label_map_ehmi = {0: "No eHMI", 1: "eHMI"}
        label_map_cam = {
            0: "Avatar first / participant second",
            1: "Participant first / avatar second",
        }

        cond_plot_df["yielding_label"] = cond_plot_df["yielding"].map(label_map_yield)
        cond_plot_df["eHMI_label"] = cond_plot_df["eHMIOn"].map(label_map_ehmi)
        cond_plot_df["camera_label"] = cond_plot_df["camera"].map(label_map_cam)

        # ============================
        # Full-factorial summary (means + SD from condition_df)
        # ============================
        group_cols = ["distPed_m", "yielding", "eHMIOn", "camera"]

        # If there are multiple rows per combination (e.g., multiple videos),
        # they should have identical stats; we take the mean as a safe aggregator.
        by_cond = (
            cond_plot_df
            .groupby(group_cols, as_index=False)
            .agg(
                mean_crossing_risk=("crossing_risk", "mean"),
                sd_crossing_risk=("crossing_risk_sd", "mean"),

                Q1_mean=("mean_Q1", "mean"),
                Q1_sd=("std_Q1", "mean"),

                Q2_mean=("mean_Q2", "mean"),
                Q2_sd=("std_Q2", "mean"),

                Q3_mean=("mean_Q3", "mean"),
                Q3_sd=("std_Q3", "mean"),
            )
            .sort_values(group_cols)
        )

        # Figure uncertainty must come from participant-level trials, not from
        # variation among condition means. Add t-based 95% CIs when those data
        # are available.
        by_cond["ci95_half_crossing_risk"] = np.nan
        by_cond["n_participants_crossing_risk"] = np.nan
        if trial_df is not None and not trial_df.empty:
            participant_trials = trial_df.copy()
            if "distPed_m" not in participant_trials.columns:
                participant_trials["distPed_m"] = self._distance_series_to_meters(
                    participant_trials["distPed"]
                )
            participant_trials["crossing_risk"] = pd.to_numeric(
                participant_trials["crossing_risk"], errors="coerce"
            )
            participant_summary = (
                participant_trials.dropna(subset=group_cols + ["participant", "crossing_risk"])
                .groupby(group_cols, as_index=False)["crossing_risk"]
                .agg(
                    mean_crossing_risk_participant="mean",
                    sd_crossing_risk_participant="std",
                    n_participants_crossing_risk="count",
                )
            )
            participant_summary["se_crossing_risk"] = (
                participant_summary["sd_crossing_risk_participant"]
                / np.sqrt(participant_summary["n_participants_crossing_risk"])
            )
            participant_summary["ci95_half_crossing_risk"] = participant_summary.apply(
                lambda row: (
                    t.ppf(0.975, int(row["n_participants_crossing_risk"]) - 1)
                    * row["se_crossing_risk"]
                    if int(row["n_participants_crossing_risk"]) > 1
                    else np.nan
                ),
                axis=1,
            )
            by_cond = by_cond.drop(
                columns=["ci95_half_crossing_risk", "n_participants_crossing_risk"]
            ).merge(participant_summary, on=group_cols, how="left")
            by_cond["mean_crossing_risk"] = by_cond[
                "mean_crossing_risk_participant"
            ].combine_first(by_cond["mean_crossing_risk"])

        # Add label columns for plotting facets
        by_cond["yielding_label"] = by_cond["yielding"].map(label_map_yield)
        by_cond["eHMI_label"] = by_cond["eHMIOn"].map(label_map_ehmi)
        by_cond["camera_label"] = by_cond["camera"].map(label_map_cam)

        logger.info("\n=== Full-factorial condition table (MEAN + SD, 0–100 scales) ===")
        logger.info(
            "\n=== Full-factorial condition table (MEAN + SD, 0–100 scales) ===\n{}",
            by_cond.to_string(index=False),
        )
        logger.info("===========================================================\n")

        # Common label mapping for all figures
        base_labels = {
            "distPed_m": "Distance between pedestrians (m)",
            "crossing_risk": "Perceived-unsafety time (%)",
            "mean_crossing_risk": "Perceived-unsafety time (%)",
            "sd_crossing_risk": "SD of perceived-unsafety time (%)",

            "Q1_mean": "Q1 (0–100)",
            "Q1_sd": "SD of Q1 (0–100)",
            "Q2_mean": "Q2 (0–100)",
            "Q2_sd": "SD of Q2 (0–100)",
            "Q3_mean": "Q3 (0–100)",
            "Q3_sd": "SD of Q3 (0–100)",

            "camera_label": "Relative pedestrian order",
            "yielding_label": "AV behaviour",
            "eHMI_label": "Conditional eHMI logic",
            "context": "Context (AV behaviour, conditional eHMI logic, relative pedestrian order)",
            "delta": "Near–far difference (0–100)",
            "measure": "Measure",
        }

        # Category ordering for cleaner facets / legends
        category_orders = {
            "eHMI_label": ["No eHMI", "eHMI"],
            "yielding_label": ["Non-yielding", "Yielding"],
            "camera_label": [
                "Avatar first / participant second",
                "Participant first / avatar second",
            ],
        }
        axis_title_font = dict(size=font_size, family=font_family)

        # ============================
        # Figure 1 — Mean crossing risk vs Distance (legend = camera)
        # ============================
        fig_beh = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="camera_label",
            facet_col="eHMI_label",
            facet_row="yielding_label",
            markers=True,
            error_y="ci95_half_crossing_risk",
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_beh.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_beh.update_xaxes(title_font=axis_title_font)
        fig_beh.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_beh)

        # ============================
        # Figure 2 — Q2 vs Distance (legend = camera)
        # ============================
        fig_q2 = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="camera_label",
            facet_col="eHMI_label",
            facet_row="yielding_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_q2.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.9,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_q2.update_xaxes(title_font=axis_title_font)
        fig_q2.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_q2)

        # ============================
        # EXTRA Figure A — Mean crossing risk vs distance, legend = yielding
        # ============================
        fig_beh_yield = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="yielding_label",
            facet_col="eHMI_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_beh_yield.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_beh_yield.update_xaxes(title_font=axis_title_font)
        fig_beh_yield.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_beh_yield)

        # ============================
        # EXTRA Figure B — Mean crossing risk vs distance, legend = eHMI
        # ============================
        fig_beh_ehmi = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="eHMI_label",
            facet_col="yielding_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_beh_ehmi.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_beh_ehmi.update_xaxes(title_font=axis_title_font)
        fig_beh_ehmi.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_beh_ehmi)

        # ============================
        # EXTRA Figure C — Q2 vs distance, legend = yielding
        # ============================
        fig_q2_yield = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="yielding_label",
            facet_col="eHMI_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_q2_yield.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_q2_yield.update_xaxes(title_font=axis_title_font)
        fig_q2_yield.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_q2_yield)

        # ============================
        # EXTRA Figure D — Q2 vs distance, legend = eHMI
        # ============================
        fig_q2_ehmi = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="eHMI_label",
            facet_col="yielding_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_q2_ehmi.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_q2_ehmi.update_xaxes(title_font=axis_title_font)
        fig_q2_ehmi.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_q2_ehmi)

        # ============================
        # Figure 3 — Mean crossing risk vs Q2 scatter (condition-level)
        # ============================
        fig_scatter = px.scatter(
            cond_plot_df,
            x="crossing_risk",
            y="mean_Q2",
            color="distPed_m",
            labels=base_labels,
            title="",
        )

        x_vals = cond_plot_df["crossing_risk"].values
        y_vals = cond_plot_df["mean_Q2"].values
        if len(x_vals) >= 2 and np.isfinite(x_vals).all() and np.isfinite(y_vals).all():
            b1, b0 = np.polyfit(x_vals, y_vals, 1)
            xs = np.linspace(x_vals.min(), x_vals.max(), 100)
            ys = b0 + b1 * xs
            fig_scatter.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig_scatter.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
            ),
        )

        # ============================
        # Figure 4 — NEAR (2–4 m) minus FAR (8–10 m) per context
        # ============================
        ctx_cols = ["yielding", "eHMIOn", "camera"]

        near = (
            by_cond[by_cond["distPed_m"].isin([2, 4])]
            .groupby(ctx_cols, as_index=False)
            .agg(
                crossing_risk_near=("mean_crossing_risk", "mean"),
                Q1_near=("Q1_mean", "mean"),
                Q2_near=("Q2_mean", "mean"),
                Q3_near=("Q3_mean", "mean"),
            )
        )
        far = (
            by_cond[by_cond["distPed_m"].isin([8, 10])]
            .groupby(ctx_cols, as_index=False)
            .agg(
                crossing_risk_far=("mean_crossing_risk", "mean"),
                Q1_far=("Q1_mean", "mean"),
                Q2_far=("Q2_mean", "mean"),
                Q3_far=("Q3_mean", "mean"),
            )
        )

        diff_df = near.merge(far, on=ctx_cols, how="inner")
        diff_df["delta_crossing_risk"] = (
            diff_df["crossing_risk_near"] - diff_df["crossing_risk_far"]
        )
        diff_df["delta_Q1"] = diff_df["Q1_near"] - diff_df["Q1_far"]
        diff_df["delta_Q2"] = diff_df["Q2_near"] - diff_df["Q2_far"]
        diff_df["delta_Q3"] = diff_df["Q3_near"] - diff_df["Q3_far"]

        # Context string with on/off text instead of 0/1
        diff_df["context"] = diff_df.apply(
            lambda r: (
                f"{'Yielding' if r['yielding'] == 1 else 'Non-yielding'}, "
                f"eHMI {'on' if r['eHMIOn'] == 1 else 'off'}, "
                f"{'avatar first / participant second' if int(r['camera']) == 0 else 'participant first / avatar second'}"
            ),
            axis=1,
        )

        logger.info("\n=== NEAR–FAR differences per context ===")
        logger.info(
            "\n=== NEAR–FAR differences per context ===\n{}",
            diff_df[
                ["context", "delta_crossing_risk", "delta_Q1", "delta_Q2", "delta_Q3"]
            ].to_string(index=False),
        )
        logger.info("========================================\n")

        long_diff = diff_df.melt(
            id_vars=["context"],
            value_vars=["delta_crossing_risk", "delta_Q1", "delta_Q2", "delta_Q3"],
            var_name="measure",
            value_name="delta",
        )

        long_diff["measure"] = long_diff["measure"].map({
            "delta_crossing_risk": "Perceived-unsafety time (%)",
            "delta_Q1": "Q1 (0–100)",
            "delta_Q2": "Q2 (0–100)",
            "delta_Q3": "Q3 (0–100)",
        })

        fig_diff = px.bar(
            long_diff,
            x="context",
            y="delta",
            color="measure",
            barmode="group",
            labels={**base_labels, "delta": "Near–far difference (0–100)"},
            title="",
        )

        fig_diff.update_traces(
            texttemplate="%{y:.1f}",
            textposition="outside",
            textfont=axis_title_font,
        )

        fig_diff.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.88,
                y=0.85,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
            ),
        )
        fig_diff.update_xaxes(title_font=axis_title_font, tickfont=axis_title_font)
        fig_diff.update_yaxes(title_font=axis_title_font, tickfont=axis_title_font)
        fig_diff.add_hline(y=0, line_dash="dash", line_color="black")

        # ============================
        # Stats summary
        # ============================
        logger.info(
            "Naive trial-level and condition-mean correlations are intentionally "
            "omitted; the within/between-participant models provide the clustered "
            "association analysis."
        )

        xd = by_cond["distPed_m"].values

        # Slopes vs distance for crossing risk and Q1–Q3
        yd_risk = by_cond["mean_crossing_risk"].values
        slope_risk, intercept_risk = np.polyfit(xd, yd_risk, 1)
        logger.info(
            "Overall perceived-unsafety time vs distance: "
            f"slope = {slope_risk:.4f} (risk units per 1 m)"
        )

        for q_label, col in [("Q1", "Q1_mean"), ("Q2", "Q2_mean"), ("Q3", "Q3_mean")]:
            y = by_cond[col].values
            slope_q, intercept_q = np.polyfit(xd, y, 1)
            logger.info(
                f"Overall {q_label} vs distance: "
                f"slope = {slope_q:.4f} ({q_label} units per 1 m)"
            )

        # ============================
        # Stats summary
        # ============================
        stats_out_dir = out_dir or self.output_folder
        os.makedirs(stats_out_dir, exist_ok=True)

        near_far_path = os.path.join(stats_out_dir, "near_far_differences.csv")
        diff_df.to_csv(near_far_path, index=False)

        by_cond.to_csv(os.path.join(stats_out_dir, "distance_effect_cell_summary.csv"), index=False)
        by_cond.to_csv(os.path.join(stats_out_dir, "table_descriptives_full_factorial.csv"), index=False)

        if trial_df is not None and not trial_df.empty:
            self._run_mixed_effects_model(trial_df, "crossing_risk", stats_out_dir)
            self._run_mixed_effects_model(trial_df, "Q1", stats_out_dir)
            self._run_mixed_effects_model(trial_df, "Q2", stats_out_dir)
            self._run_mixed_effects_model(trial_df, "Q3", stats_out_dir)

        # ============================
        # Save figures
        # ============================
        self.save_plotly(fig_beh, "crossing_risk_full_factorial", save_final=True)
        self.save_plotly(fig_q2, "Q2_full_factorial", save_final=True)
        self.save_plotly(fig_diff, "near_minus_far_crossing_risk_vs_Q123", save_final=True)
        self.save_plotly(fig_beh_yield, "crossing_risk_full_factorial_legend_yielding", save_final=True)
        self.save_plotly(fig_beh_ehmi, "crossing_risk_full_factorial_legend_eHMI", save_final=True)
        self.save_plotly(fig_q2_yield, "Q2_full_factorial_legend_yielding", save_final=True)
        self.save_plotly(fig_q2_ehmi, "Q2_full_factorial_legend_eHMI", save_final=True)

        return by_cond, cond_plot_df, diff_df
