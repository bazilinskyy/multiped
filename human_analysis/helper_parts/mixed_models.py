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


class MixedModelMixin:
    """Focused method group extracted without changing calculation logic."""

    @staticmethod
    def _pretty_mixed_term(term):
        mapping = {
            "Intercept": "Intercept",
            "C(yielding)[T.1]": "Yielding",
            "C(eHMIOn)[T.1]": "eHMI",
            "C(camera)[T.1]": "Participant-first / avatar-second order",
            "distPed_m": "Distance (m)",
            "C(yielding)[T.1]:C(eHMIOn)[T.1]": "Yielding × eHMI",
            "C(yielding)[T.1]:C(camera)[T.1]": "Yielding × relative pedestrian order",
            "C(eHMIOn)[T.1]:C(camera)[T.1]": "eHMI × relative pedestrian order",
            "Group Var": "Random intercept variance",
        }
        return mapping.get(term, term)

    def _run_mixed_effects_model(self, trial_df, outcome, out_dir=None):
        """Fit categorical-distance models, attempting participant random slopes first."""
        if smf is None:
            logger.warning(f"statsmodels is not available; skipping mixed model for {outcome}")
            return None, None

        save_dir = out_dir or self.output_folder
        os.makedirs(save_dir, exist_ok=True)
        coefficients_path = os.path.join(
            save_dir, f"mixed_model_coefficients_{outcome}.csv"
        )
        analysis_version = "categorical_distance_random_slopes_v2"
        if self.reuse_statistical_results and os.path.isfile(coefficients_path):
            cached = pd.read_csv(coefficients_path)
            if (
                "analysis_version" in cached.columns
                and cached["analysis_version"].eq(analysis_version).all()
            ):
                logger.info(f"Reused cached improved mixed model for {outcome}.")
                return None, cached

        model_df = trial_df.copy()
        model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")
        needed = ["participant", outcome, "yielding", "eHMIOn", "camera", "distPed_m"]
        if "trial_number" in model_df.columns:
            needed.append("trial_number")
        model_df = model_df.dropna(subset=needed)
        if model_df.empty:
            logger.warning(f"No data available for mixed model outcome {outcome}")
            return None, None

        model_df["distPed_centered"] = model_df["distPed_m"] - 6.0
        trial_terms = ""
        random_trial_term = ""
        if "trial_number" in model_df.columns:
            model_df["trial_number_centered"] = (
                model_df["trial_number"]
                - model_df.groupby("participant")["trial_number"].transform("mean")
            )
            trial_terms = " + trial_number_centered + I(trial_number_centered ** 2)"
            random_trial_term = " + trial_number_centered"

        formula = (
            f"{outcome} ~ C(yielding) * C(eHMIOn) * C(camera) + "
            "C(distPed_m) * (C(yielding) + C(camera))"
            f"{trial_terms}"
        )
        random_formulas = [
            "~C(yielding) + C(eHMIOn) + C(camera) + distPed_centered"
            f"{random_trial_term}",
            "~C(yielding) + C(eHMIOn) + C(camera)",
            "~C(yielding) + C(eHMIOn)",
            "~distPed_centered",
            None,
        ]
        fit = None
        model_name = None
        selected_re_formula = None
        for index, re_formula in enumerate(random_formulas, start=1):
            try:
                candidate = smf.mixedlm(
                    formula,
                    model_df,
                    groups=model_df["participant"],
                    re_formula=re_formula,
                ).fit(
                    reml=False,
                    method=["lbfgs", "bfgs", "cg"],
                    maxiter=500,
                    disp=False,
                )
                if not bool(getattr(candidate, "converged", False)):
                    raise RuntimeError("model did not converge")
                fit = candidate
                model_name = (
                    f"mixed_random_slopes_{index}"
                    if re_formula is not None
                    else "mixed_random_intercept_fallback"
                )
                selected_re_formula = re_formula
                break
            except Exception as exc:
                logger.warning(
                    "Mixed model attempt {} failed for {} (re_formula={}): {}",
                    index,
                    outcome,
                    re_formula,
                    exc,
                )

        if fit is None:
            try:
                fit = smf.ols(formula, data=model_df).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": model_df["participant"]},
                )
                model_name = "ols_clustered_fallback"
                selected_re_formula = None
                logger.warning(
                    "All mixed models failed for {}; used participant-clustered OLS.",
                    outcome,
                )
            except Exception as exc:
                logger.error(f"All improved models failed for {outcome}: {exc}")
                return None, None

        coef_df = pd.DataFrame({
            "term": fit.params.index,
            "estimate": fit.params.values,
            "std_error": fit.bse.values,
            "z_value": fit.tvalues.values,
            "p_value": fit.pvalues.values,
        })
        conf = fit.conf_int()
        coef_df["ci_lower"] = conf.iloc[:, 0].values
        coef_df["ci_upper"] = conf.iloc[:, 1].values
        coef_df["predictor"] = coef_df["term"].map(self._pretty_mixed_term)
        coef_df["model"] = model_name
        coef_df["formula"] = formula
        coef_df["random_effects_formula"] = selected_re_formula or "1"
        coef_df["converged"] = bool(getattr(fit, "converged", True))
        coef_df["analysis_version"] = analysis_version
        coef_df["ci_95"] = coef_df.apply(
            lambda row: f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]",
            axis=1,
        )

        logger.info(f"\n=== Mixed model: {outcome} ===")
        logger.info(f"Formula: {formula}")
        logger.info(
            "{}",
            coef_df[["predictor", "estimate", "ci_lower", "ci_upper", "p_value"]].to_string(index=False),
        )
        logger.info("=======================\n")

        coef_df.to_csv(coefficients_path, index=False)

        term_df = coef_df[["term", "predictor", "estimate", "ci_lower", "ci_upper", "p_value"]].copy()
        term_df.insert(0, "outcome", outcome)
        term_df.to_csv(os.path.join(save_dir, f"mixed_model_terms_{outcome}.csv"), index=False)

        fixed_term_names = (
            set(fit.fe_params.index)
            if hasattr(fit, "fe_params")
            else set(fit.params.index)
        )
        manuscript_df = coef_df.loc[
            coef_df["term"].isin(fixed_term_names),
            ["predictor", "estimate", "ci_lower", "ci_upper", "ci_95", "p_value"],
        ].copy()
        manuscript_df.to_csv(os.path.join(save_dir, f"mixed_model_table_{outcome}.csv"), index=False)
        return fit, coef_df
