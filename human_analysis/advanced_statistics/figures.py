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


class CommonWindowFigureMixin:
    """Focused method group extracted without changing calculation logic."""

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
