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


class WithinBetweenMixin:
    """Focused method group extracted without changing calculation logic."""

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
