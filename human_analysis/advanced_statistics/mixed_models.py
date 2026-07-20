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


class RepeatedMeasuresMixin:
    """Focused method group extracted without changing calculation logic."""

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
