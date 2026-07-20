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


class SensitivityMixin:
    """Focused method group extracted without changing calculation logic."""

    @staticmethod
    def _trigger_threshold_label(threshold: float) -> str:
        """Create a filesystem-safe label for a trigger threshold."""
        return f"threshold_{int(round(float(threshold) * 100)):03d}pct"

    def run_trigger_threshold_sensitivity(
        self,
        trigger_thresholds,
        trigger_matrices_dir: str,
        responses_root: str,
        mapping_df: pd.DataFrame,
        primary_threshold: float = 0.10,
        output_dir: Optional[str] = None,
        n_participants: int = 50,
        response_col_index: int = 2,
        ratings_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Run crossing-risk sensitivity checks for several trigger thresholds.

        The primary manuscript definition treats the pressure-sensitive trigger as
        a binary state: values greater than the threshold are coded as unsafe/risk
        state 1, and values at or below the threshold are coded as 0. This method
        repeats that calculation for multiple thresholds so the effect of the
        chosen tolerance can be inspected.

        Parameters
        ----------
        trigger_thresholds : iterable
            Trigger thresholds on the 0..1 scale, for example [0.05, 0.10, 0.50].
        trigger_matrices_dir : str
            Directory containing participant_TriggerValueRight_video_*.csv files.
        responses_root : str
            Directory containing Participant_* folders with trial-wise Q1/Q2/Q3 data.
        mapping_df : pd.DataFrame
            Scenario mapping table.
        primary_threshold : float
            Threshold used for the manuscript's primary analysis. Alternative
            thresholds are compared directly with this value.
        output_dir : str, optional
            Root directory where threshold-specific output folders are written.
        n_participants : int
            Maximum participant id to scan when reading trial-wise responses.
        response_col_index : int
            Index of Q2 in the participant response CSVs; Q1 and Q3 are inferred
            as the neighbouring columns, matching load_and_average_Q2.
        ratings_df : pd.DataFrame, optional
            Cached participant Q1/Q2/Q3 trial ratings. When supplied, raw
            participant response files are not read.

        Returns
        -------
        dict
            A dictionary with combined summary, model-term, and condition-level
            tables for all thresholds.
        """
        output_dir = output_dir or os.path.join(self.output_folder, "threshold_sensitivity")
        os.makedirs(output_dir, exist_ok=True)

        threshold_values = list(dict.fromkeys(float(x) for x in trigger_thresholds))
        primary_threshold = float(primary_threshold)
        if not any(np.isclose(value, primary_threshold) for value in threshold_values):
            raise ValueError("primary_threshold must be included in trigger_thresholds")
        summary_records = []
        model_tables = []
        condition_tables = []

        for threshold in threshold_values:
            threshold_label = self._trigger_threshold_label(threshold)
            threshold_dir = os.path.join(output_dir, threshold_label)
            stats_dir = os.path.join(threshold_dir, "statistics")
            os.makedirs(threshold_dir, exist_ok=True)
            os.makedirs(stats_dir, exist_ok=True)

            logger.info(
                "Running trigger-threshold sensitivity for {} ({:.2f}).",
                threshold_label,
                threshold,
            )

            participant_trigger_df = self._compute_trial_level_trigger_summary(
                trigger_matrices_dir,
                mapping_df,
                trigger_threshold=threshold,
            )
            participant_trigger_df.to_csv(
                os.path.join(threshold_dir, "participant_level_trigger_summary.csv"),
                index=False,
            )

            trigger_summary_df = (
                participant_trigger_df
                .groupby("condition_name", as_index=False)
                .agg(
                    avg_trigger=("avg_trigger", "mean"),
                    sd_trigger=("avg_trigger", "std"),
                    n_trials=("avg_trigger", "size"),
                    n_trigger_bins=("n_trigger_bins", "sum"),
                    n_raw_trigger_samples=("n_raw_trigger_samples", "sum"),
                    mean_trigger_intensity=("mean_trigger_intensity", "mean"),
                    sd_trigger_intensity=("sd_trigger_intensity", "mean"),
                    trigger_threshold=("trigger_threshold", "first"),
                )
                .sort_values("condition_name")
                .reset_index(drop=True)
            )
            trigger_summary_path = os.path.join(threshold_dir, "trigger_summary.csv")
            trigger_summary_df.to_csv(trigger_summary_path, index=False)

            trial_df, condition_df = self.load_and_average_Q2(
                trigger_summary_csv=trigger_summary_path,
                responses_root=responses_root,
                mapping_df=mapping_df,
                n_participants=n_participants,
                response_col_index=response_col_index,
                save_combined=True,
                trigger_threshold=threshold,
                trigger_matrices_dir=trigger_matrices_dir,
                ratings_df=ratings_df,
            )
            trial_df["threshold"] = threshold
            trial_df["threshold_label"] = threshold_label
            condition_df["threshold"] = threshold
            condition_df["threshold_label"] = threshold_label
            condition_tables.append(condition_df)

            risk = pd.to_numeric(trial_df["crossing_risk"], errors="coerce").dropna()
            summary_records.append({
                "threshold": threshold,
                "threshold_label": threshold_label,
                "n_trials": int(risk.size),
                "n_participants": int(trial_df["participant"].nunique()) if "participant" in trial_df.columns else np.nan,
                "mean_crossing_risk": float(risk.mean()) if not risk.empty else np.nan,
                "sd_crossing_risk": float(risk.std(ddof=1)) if risk.size > 1 else np.nan,
                "median_crossing_risk": float(risk.median()) if not risk.empty else np.nan,
                "min_crossing_risk": float(risk.min()) if not risk.empty else np.nan,
                "max_crossing_risk": float(risk.max()) if not risk.empty else np.nan,
                "p05_crossing_risk": float(risk.quantile(0.05)) if not risk.empty else np.nan,
                "p95_crossing_risk": float(risk.quantile(0.95)) if not risk.empty else np.nan,
                "zero_risk_trial_pct": float((risk == 0).mean() * 100.0) if not risk.empty else np.nan,
            })

            try:
                _, coef_df = self._run_mixed_effects_model(trial_df, "crossing_risk", stats_dir)
                coef_df = coef_df.copy()
                coef_df["threshold"] = threshold
                coef_df["threshold_label"] = threshold_label
                model_tables.append(coef_df)
            except Exception as exc:
                logger.warning(
                    "Mixed-effects model failed for {}: {}",
                    threshold_label,
                    exc,
                )

        summary_df = pd.DataFrame(summary_records)
        model_terms_df = pd.concat(model_tables, ignore_index=True) if model_tables else pd.DataFrame()
        condition_sensitivity_df = (
            pd.concat(condition_tables, ignore_index=True) if condition_tables else pd.DataFrame()
        )

        # Provide an auditable comparison with the primary 0.10 analysis. The
        # correlations describe stability of condition patterns, while the
        # model fields show whether effect directions and significance decisions
        # are retained. These metrics are descriptive; the researcher should
        # inspect the accompanying long-form tables before claiming robustness.
        robustness_records = []
        if not condition_sensitivity_df.empty:
            primary_condition = condition_sensitivity_df.loc[
                np.isclose(condition_sensitivity_df["threshold"], primary_threshold),
                ["condition_name", "avg_trigger"],
            ].rename(columns={"avg_trigger": "avg_trigger_primary"})

            primary_model = pd.DataFrame()
            if not model_terms_df.empty and {"term", "estimate", "p_value", "threshold"}.issubset(model_terms_df.columns):
                primary_model = model_terms_df.loc[
                    np.isclose(model_terms_df["threshold"], primary_threshold),
                    ["term", "estimate", "p_value"],
                ].drop_duplicates(subset=["term"])

            for threshold in threshold_values:
                current_condition = condition_sensitivity_df.loc[
                    np.isclose(condition_sensitivity_df["threshold"], threshold),
                    ["condition_name", "avg_trigger"],
                ].rename(columns={"avg_trigger": "avg_trigger_current"})
                condition_compare = primary_condition.merge(
                    current_condition,
                    on="condition_name",
                    how="inner",
                ).dropna()

                pearson = np.nan
                spearman = np.nan
                mean_abs_difference_points = np.nan
                max_abs_difference_points = np.nan
                if len(condition_compare) >= 2:
                    pearson = condition_compare["avg_trigger_primary"].corr(
                        condition_compare["avg_trigger_current"], method="pearson"
                    )
                    spearman = condition_compare["avg_trigger_primary"].corr(
                        condition_compare["avg_trigger_current"], method="spearman"
                    )
                    difference_points = (
                        condition_compare["avg_trigger_current"]
                        - condition_compare["avg_trigger_primary"]
                    ).abs() * 100.0
                    mean_abs_difference_points = float(difference_points.mean())
                    max_abs_difference_points = float(difference_points.max())

                model_sign_agreement_pct = np.nan
                model_significance_agreement_pct = np.nan
                n_model_terms = 0
                if not primary_model.empty:
                    current_model = model_terms_df.loc[
                        np.isclose(model_terms_df["threshold"], threshold),
                        ["term", "estimate", "p_value"],
                    ].drop_duplicates(subset=["term"])
                    model_compare = primary_model.merge(
                        current_model,
                        on="term",
                        suffixes=("_primary", "_current"),
                        how="inner",
                    )
                    model_compare = model_compare.loc[
                        model_compare["term"] != "Group Var"
                    ].dropna(subset=["estimate_primary", "estimate_current"])
                    n_model_terms = int(len(model_compare))
                    if n_model_terms:
                        model_sign_agreement_pct = float(
                            (
                                np.sign(model_compare["estimate_primary"])
                                == np.sign(model_compare["estimate_current"])
                            ).mean() * 100.0
                        )
                        valid_p = model_compare.dropna(
                            subset=["p_value_primary", "p_value_current"]
                        )
                        if not valid_p.empty:
                            model_significance_agreement_pct = float(
                                (
                                    (valid_p["p_value_primary"] < 0.05)
                                    == (valid_p["p_value_current"] < 0.05)
                                ).mean() * 100.0
                            )

                robustness_records.append({
                    "primary_threshold": primary_threshold,
                    "threshold": threshold,
                    "is_primary": bool(np.isclose(threshold, primary_threshold)),
                    "n_conditions": int(len(condition_compare)),
                    "condition_pearson_r": pearson,
                    "condition_spearman_rho": spearman,
                    "condition_mean_abs_difference_points": mean_abs_difference_points,
                    "condition_max_abs_difference_points": max_abs_difference_points,
                    "n_model_terms": n_model_terms,
                    "model_sign_agreement_pct": model_sign_agreement_pct,
                    "model_significance_agreement_pct": model_significance_agreement_pct,
                })

        robustness_df = pd.DataFrame(robustness_records)

        summary_path = os.path.join(output_dir, "threshold_sensitivity_summary.csv")
        model_path = os.path.join(output_dir, "threshold_sensitivity_model_terms.csv")
        condition_path = os.path.join(output_dir, "threshold_sensitivity_condition_means.csv")
        robustness_path = os.path.join(output_dir, "threshold_sensitivity_robustness.csv")

        summary_df.to_csv(summary_path, index=False)
        model_terms_df.to_csv(model_path, index=False)
        condition_sensitivity_df.to_csv(condition_path, index=False)
        robustness_df.to_csv(robustness_path, index=False)

        logger.info(f"Saved threshold sensitivity summary to: {summary_path}")
        logger.info(f"Saved threshold sensitivity model terms to: {model_path}")
        logger.info(f"Saved threshold sensitivity condition means to: {condition_path}")
        logger.info(f"Saved threshold sensitivity robustness checks to: {robustness_path}")

        return {
            "summary": summary_df,
            "model_terms": model_terms_df,
            "condition_means": condition_sensitivity_df,
            "robustness": robustness_df,
        }
