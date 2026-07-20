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


class BinomialMixin:
    """Focused method group extracted without changing calculation logic."""

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
