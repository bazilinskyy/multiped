"""Analyses added in the second revision in response to reviewer comments.

Every method here reuses the primary common-window data construction. The
additions are:

* direct order-by-eHMI interaction contrasts (difference between the two
  conditional eHMI contrasts) with participant-clustered and bootstrap
  intervals;
* sensitivity analyses that do not treat 100-ms bins as the unit of analysis:
  paired participant-level contrasts, a trial-level fractional logit, and a
  binary majority-of-window trial outcome;
* a learning analysis based on cumulative exposure to the yielding eHMI
  animation rather than on general trial number;
* term-wise joint Wald tests from an effect-coded refit of the primary model,
  giving one overview row for each factor and interaction;
* a check of how evenly conditions were spread over the realised trial order;
* a summary of logged vehicle event times used to define analysis windows.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix
from scipy.stats import friedmanchisquare, norm
from scipy.stats import t as student_t

import common
from custom_logger import CustomLogger


ADVANCED_STATS_SPECIFICATION = "reviewer_response_v6_second_revision"

logger = CustomLogger(__name__)

Cell = Tuple[int, int, int]  # (yielding, eHMIOn, camera)

# camera 0 = avatar first / participant second; camera 1 = participant first.
_ORDER_LABELS = {0: "avatar first", 1: "participant first"}


def _ehmi_by_order_weights(yielding: int) -> Dict[Cell, float]:
    """Weights for [eHMI effect | PF] minus [eHMI effect | AF]."""
    return {
        (yielding, 1, 1): 1.0,
        (yielding, 0, 1): -1.0,
        (yielding, 1, 0): -1.0,
        (yielding, 0, 0): 1.0,
    }


def _interaction_contrast_specifications() -> List[Tuple[str, str, Dict[Cell, float]]]:
    """Return response-scale interaction contrasts as signed cell weights."""
    yielding_weights = _ehmi_by_order_weights(1)
    non_yielding_weights = _ehmi_by_order_weights(0)
    three_way = dict(yielding_weights)
    for cell, weight in non_yielding_weights.items():
        three_way[cell] = three_way.get(cell, 0.0) - weight
    return [
        (
            "ehmi_by_order",
            "(eHMI on minus off | participant first) minus "
            "(eHMI on minus off | avatar first) | yielding",
            yielding_weights,
        ),
        (
            "ehmi_by_order",
            "(eHMI on minus off | participant first) minus "
            "(eHMI on minus off | avatar first) | non-yielding",
            non_yielding_weights,
        ),
        (
            "ehmi_by_order_by_yielding",
            "yielding minus non-yielding difference in the eHMI-by-order contrast",
            three_way,
        ),
    ]


class RevisionAnalysesMixin:
    """Second-revision analyses built on the primary common-window model."""

    # ------------------------------------------------------------------
    # Interaction contrasts
    # ------------------------------------------------------------------
    def run_interaction_contrasts(
        self,
        model_result: Dict[str, object],
        analysis_label: str,
        bootstrap_marginal_draws: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Estimate difference-in-differences contrasts on the response scale.

        The delta-method variance uses the model's participant-clustered
        covariance. When participant bootstrap draws of the eight marginal
        cell predictions are supplied, the same linear combination is applied
        to every draw to obtain a percentile interval.
        """
        fit = model_result["fit"]
        gradients: Dict[Cell, np.ndarray] = model_result["cell_gradients"]
        marginal = model_result["marginal_probabilities"]
        probabilities = {
            (int(row.yielding), int(row.eHMIOn), int(row.camera)): float(
                row.predicted_probability
            )
            for row in marginal.itertuples()
        }
        covariance = fit.cov_params().to_numpy(dtype=float)

        draw_matrix = None
        if bootstrap_marginal_draws is not None and not bootstrap_marginal_draws.empty:
            draw_matrix = bootstrap_marginal_draws.pivot_table(
                index="bootstrap_iteration",
                columns=["yielding", "eHMIOn", "camera"],
                values="predicted_percentage",
            )

        rows = []
        for family, label, weights in _interaction_contrast_specifications():
            estimate = sum(w * probabilities[c] for c, w in weights.items())
            gradient = sum(w * gradients[c] for c, w in weights.items())
            standard_error = float(np.sqrt(max(gradient @ covariance @ gradient, 0.0)))
            z_value = estimate / standard_error if standard_error > 0 else np.nan
            p_value = float(2.0 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
            row = {
                "analysis": analysis_label,
                "contrast_family": family,
                "contrast": label,
                "cell_weights": "; ".join(
                    f"{cell}:{weight:+.0f}" for cell, weight in weights.items()
                ),
                "estimate_percentage_points": 100.0 * estimate,
                "std_error_percentage_points": 100.0 * standard_error,
                "ci_lower_percentage_points": 100.0 * (estimate - 1.96 * standard_error),
                "ci_upper_percentage_points": 100.0 * (estimate + 1.96 * standard_error),
                "z_value": z_value,
                "p_value_unadjusted": p_value,
                "bootstrap_ci_lower_percentage_points": np.nan,
                "bootstrap_ci_upper_percentage_points": np.nan,
                "n_bootstrap_draws": 0,
            }
            if draw_matrix is not None:
                values = sum(
                    weight * draw_matrix[cell].to_numpy(dtype=float)
                    for cell, weight in weights.items()
                )
                values = values[np.isfinite(values)]
                if values.size:
                    row["bootstrap_ci_lower_percentage_points"] = float(
                        np.quantile(values, 0.025)
                    )
                    row["bootstrap_ci_upper_percentage_points"] = float(
                        np.quantile(values, 0.975)
                    )
                    row["n_bootstrap_draws"] = int(values.size)
            rows.append(row)
        table = pd.DataFrame.from_records(rows)
        table["p_value_holm_two_way"] = np.nan
        two_way = table["contrast_family"].eq("ehmi_by_order")
        table.loc[two_way, "p_value_holm_two_way"] = self._holm_adjust(
            table.loc[two_way, "p_value_unadjusted"]
        )
        self._save_table(table, f"{analysis_label}_interaction_contrasts.csv")
        return table

    # ------------------------------------------------------------------
    # Trial-level sensitivity analysis
    # ------------------------------------------------------------------
    @staticmethod
    def _contrast_columns(contrasts: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Select and prefix the reporting columns of a contrast table."""
        return contrasts[
            [
                "contrast",
                "estimate_percentage_points",
                "ci_lower_percentage_points",
                "ci_upper_percentage_points",
                "p_value_holm",
            ]
        ].rename(
            columns={
                "estimate_percentage_points": f"{prefix}_estimate_pp",
                "ci_lower_percentage_points": f"{prefix}_ci_lower_pp",
                "ci_upper_percentage_points": f"{prefix}_ci_upper_pp",
                "p_value_holm": f"{prefix}_p_holm",
            }
        )

    def run_participant_level_contrasts(self, trial_df: pd.DataFrame) -> pd.DataFrame:
        """Paired participant-level versions of all primary contrasts.

        Each participant's unsafe percentage is averaged over the five spacings
        within every AV behaviour x eHMI x order cell, so each participant
        contributes exactly one difference score per contrast. The one-sample
        t-tests therefore make no assumption about dependence among bins or
        trials within a participant.
        """
        label = "common_window_participant_level_paired"
        current = trial_df.copy()
        current["unsafe_pct"] = (
            100.0
            * pd.to_numeric(current["unsafe_bins_common_window"], errors="coerce")
            / pd.to_numeric(current["valid_bins_common_window"], errors="coerce")
        )
        cells = (
            current.groupby(["participant", "yielding", "eHMIOn", "camera"])["unsafe_pct"]
            .mean()
            .unstack(["yielding", "eHMIOn", "camera"])
            .dropna()
        )
        cells.columns = [tuple(int(v) for v in column) for column in cells.columns]

        specifications = [
            (family, contrast, {comparison: 1.0, reference: -1.0})
            for family, contrast, reference, comparison, _ in self._full_contrast_specifications()
        ] + _interaction_contrast_specifications()
        rows = []
        for family, contrast, weights in specifications:
            scores = sum(weight * cells[cell] for cell, weight in weights.items())
            n = int(scores.size)
            mean = float(scores.mean())
            se = float(scores.std(ddof=1) / np.sqrt(n))
            t_value = mean / se if se > 0 else np.nan
            critical = float(student_t.ppf(0.975, n - 1))
            rows.append(
                {
                    "analysis": label,
                    "contrast_family": family,
                    "contrast": contrast,
                    "n_participants": n,
                    "estimate_percentage_points": mean,
                    "std_error_percentage_points": se,
                    "ci_lower_percentage_points": mean - critical * se,
                    "ci_upper_percentage_points": mean + critical * se,
                    "t_value": t_value,
                    "df": n - 1,
                    "p_value_unadjusted": float(2 * student_t.sf(abs(t_value), n - 1)),
                }
            )
        table = pd.DataFrame.from_records(rows)
        table["p_value_holm"] = np.nan
        for _, indexes in table.groupby("contrast_family").groups.items():
            table.loc[list(indexes), "p_value_holm"] = self._holm_adjust(
                table.loc[list(indexes), "p_value_unadjusted"]
            )
        self._save_table(table, f"{label}_contrasts.csv")
        return table

    def run_trial_level_sensitivity(
        self,
        trial_df: pd.DataFrame,
        primary_result: Dict[str, object],
        majority_share: float = 0.5,
    ) -> Dict[str, pd.DataFrame]:
        """Trial-level sensitivity analyses for the bin-based primary model.

        1. A fractional logit with one equally weighted row per trial. When all
           trials contain the same number of valid bins, its estimating
           equations are proportional to those of the grouped-binomial model,
           so estimates and participant-clustered standard errors coincide.
           The fit verifies that the bins do not receive extra weight.
        2. A binary trial outcome coding a trial as unsafe when at least
           ``majority_share`` of its valid window bins were unsafe. This
           discards within-trial duration information altogether and fits one
           Bernoulli observation per trial with participant clustering.
        """
        fractional_label = "common_window_trial_level_fractional_logit"
        fractional = self.run_primary_grouped_binomial_analysis(
            trial_df,
            analysis_label=fractional_label,
            success_column="unsafe_bins_common_window",
            valid_column="valid_bins_common_window",
            model_mode="full",
            response="trial_proportion",
        )

        binary_label = "common_window_trial_level_majority_binary"
        binary_df = trial_df.copy()
        unsafe = pd.to_numeric(binary_df["unsafe_bins_common_window"], errors="coerce")
        valid = pd.to_numeric(binary_df["valid_bins_common_window"], errors="coerce")
        binary_df["unsafe_majority_trial"] = (
            unsafe >= majority_share * valid
        ).astype(float).where(valid > 0)
        binary_df["trial_unit"] = np.where(valid > 0, 1.0, np.nan)
        binary = self.run_primary_grouped_binomial_analysis(
            binary_df,
            analysis_label=binary_label,
            success_column="unsafe_majority_trial",
            valid_column="trial_unit",
            model_mode="full",
            response="grouped",
        )
        binary_interactions = self.run_interaction_contrasts(binary, binary_label)
        self.run_interaction_contrasts(fractional, fractional_label)

        primary = primary_result["revised_contrasts"][["contrast_family"]].join(
            self._contrast_columns(primary_result["revised_contrasts"], "primary")
        )
        participant_level = self.run_participant_level_contrasts(trial_df)
        comparison = (
            primary.merge(
                self._contrast_columns(participant_level, "participant_paired"),
                on="contrast",
                how="left",
            )
            .merge(
                self._contrast_columns(fractional["revised_contrasts"], "fractional"),
                on="contrast",
                how="left",
            )
            .merge(
                self._contrast_columns(binary["revised_contrasts"], "majority_binary"),
                on="contrast",
                how="left",
            )
        )
        comparison["fractional_max_abs_difference_pp"] = (
            comparison["fractional_estimate_pp"] - comparison["primary_estimate_pp"]
        ).abs()
        comparison["majority_binary_same_holm_decision_at_0_05"] = (
            comparison["primary_p_holm"].lt(0.05)
            == comparison["majority_binary_p_holm"].lt(0.05)
        )
        comparison["participant_paired_same_holm_decision_at_0_05"] = (
            comparison["primary_p_holm"].lt(0.05)
            == comparison["participant_paired_p_holm"].lt(0.05)
        )
        comparison["majority_share"] = majority_share
        self._save_table(comparison, "common_window_trial_level_vs_primary_contrasts.csv")
        logger.info(
            "Trial-level sensitivity: fractional logit max |difference| = "
            f"{comparison['fractional_max_abs_difference_pp'].max():.2e} pp; "
            "majority-binary Holm decisions agreed for "
            f"{int(comparison['majority_binary_same_holm_decision_at_0_05'].sum())}/"
            f"{len(comparison)} contrasts."
        )
        return {
            "comparison": comparison,
            "participant_level_contrasts": participant_level,
            "majority_binary_interaction_contrasts": binary_interactions,
            "majority_binary_marginal_probabilities": binary["marginal_probabilities"],
            "majority_binary_omnibus_tests": binary["omnibus_tests"],
        }

    # ------------------------------------------------------------------
    # Exposure-based learning analysis
    # ------------------------------------------------------------------
    @staticmethod
    def add_ehmi_animation_exposure(trial_df: pd.DataFrame) -> pd.DataFrame:
        """Add cumulative counts of previous yielding-eHMI animation trials.

        The inward wiping animation was displayed only in yielding trials with
        the eHMI active. ``prior_animation_exposures`` counts how many such
        main trials the participant had completed before the current trial.
        """
        current = trial_df.copy()
        current = current.sort_values(["participant", "trial_number"]).copy()
        animation = (
            pd.to_numeric(current["yielding"], errors="coerce").eq(1)
            & pd.to_numeric(current["eHMIOn"], errors="coerce").eq(1)
        ).astype(int)
        active_ehmi = pd.to_numeric(current["eHMIOn"], errors="coerce").eq(1).astype(int)
        current["animation_trial"] = animation
        current["prior_animation_exposures"] = (
            animation.groupby(current["participant"]).cumsum() - animation
        )
        current["prior_active_ehmi_exposures"] = (
            active_ehmi.groupby(current["participant"]).cumsum() - active_ehmi
        )
        return current

    def run_animation_exposure_learning(self, trial_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Test whether the yielding eHMI contrast changed with animation exposure."""
        label = "common_window_animation_exposure_learning"
        success_column = "unsafe_bins_common_window"
        valid_column = "valid_bins_common_window"
        current = self._prepare_grouped_binomial_frame(
            trial_df=trial_df,
            success_column=success_column,
            valid_column=valid_column,
            model_mode="full",
            analysis_label=label,
        )
        current = self.add_ehmi_animation_exposure(current)
        rhs = (
            "C(yielding) * C(eHMIOn) * C(camera) + "
            "C(distPed_m) * (C(yielding) + C(eHMIOn) + C(camera)) + "
            "trial_number_centered + I(trial_number_centered ** 2) + "
            "prior_animation_exposures * C(yielding) * C(eHMIOn)"
        )
        design = dmatrix(rhs, current, return_type="dataframe")
        successes = current[success_column].to_numpy(dtype=float)
        failures = (current[valid_column] - current[success_column]).to_numpy(dtype=float)
        fit = sm.GLM(
            endog=np.column_stack([successes, failures]),
            exog=design,
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": current["participant"]})

        coefficients = self._coef_frame(
            fit,
            outcome=f"{success_column}/{valid_column}",
            model_name="marginal_binomial_glm_participant_clustered",
        )
        coefficients["analysis"] = label
        coefficients["formula"] = f"{success_column}/{valid_column} ~ {rhs}"

        names = list(fit.params.index)
        three_way = [
            name for name in names
            if "prior_animation_exposures" in name
            and "C(yielding)" in name and "C(eHMIOn)" in name
        ]
        ehmi_block = [
            name for name in names
            if "prior_animation_exposures" in name and "C(eHMIOn)" in name
        ]
        all_exposure = [name for name in names if "prior_animation_exposures" in name]
        tests = pd.DataFrame.from_records(
            [
                self._joint_wald_row(fit, three_way, "exposure_by_yielding_by_ehmi", label),
                self._joint_wald_row(fit, ehmi_block, "exposure_by_ehmi_terms", label),
                self._joint_wald_row(fit, all_exposure, "all_exposure_terms", label),
            ]
        )
        # The three-way coefficient is the change per additional previous
        # animation trial in the log-odds eHMI contrast for yielding trials,
        # relative to the corresponding change in non-yielding trials.
        slope_rows = []
        if three_way:
            term = three_way[0]
            beta = float(fit.params[term])
            se = float(fit.bse[term])
            slope_rows.append(
                {
                    "analysis": label,
                    "term": term,
                    "log_odds_estimate": beta,
                    "std_error": se,
                    "odds_ratio": float(np.exp(beta)),
                    "odds_ratio_ci_lower": float(np.exp(beta - 1.96 * se)),
                    "odds_ratio_ci_upper": float(np.exp(beta + 1.96 * se)),
                    "p_value": float(fit.pvalues[term]),
                }
            )
        slope = pd.DataFrame.from_records(slope_rows)

        # Describe the design: how strongly exposure tracked trial number.
        within = current.groupby("participant").apply(
            lambda group: group["prior_animation_exposures"].corr(group["trial_number"])
        )
        descriptive = (
            current.loc[current["yielding"].eq(1)]
            .assign(
                exposure_band=lambda d: pd.cut(
                    d["prior_animation_exposures"],
                    bins=[-0.5, 2.5, 5.5, 10.5],
                    labels=["0-2", "3-5", "6-10"],
                ),
                unsafe_pct=lambda d: 100.0 * d[success_column] / d[valid_column],
            )
            .groupby(["exposure_band", "eHMIOn"], observed=True)
            .agg(
                n_trials=("unsafe_pct", "size"),
                n_participants=("participant", "nunique"),
                mean_unsafe_pct=("unsafe_pct", "mean"),
            )
            .reset_index()
        )
        descriptive["analysis"] = label
        tests["median_within_participant_corr_exposure_trial"] = float(within.median())
        tests["min_within_participant_corr_exposure_trial"] = float(within.min())

        self._save_table(coefficients, f"{label}_coefficients.csv")
        self._save_table(tests, f"{label}_tests.csv")
        self._save_table(slope, f"{label}_three_way_slope.csv")
        self._save_table(descriptive, f"{label}_descriptives.csv")
        return {
            "coefficients": coefficients,
            "tests": tests,
            "slope": slope,
            "descriptives": descriptive,
        }

    # ------------------------------------------------------------------
    # Term-wise overview table
    # ------------------------------------------------------------------
    _TERM_LABELS = {
        "C(yielding, Sum)": "Yielding",
        "C(eHMIOn, Sum)": "eHMI",
        "C(camera, Sum)": "Pedestrian order",
        "C(yielding, Sum):C(eHMIOn, Sum)": "Yielding x eHMI",
        "C(yielding, Sum):C(camera, Sum)": "Yielding x order",
        "C(eHMIOn, Sum):C(camera, Sum)": "eHMI x order",
        "C(yielding, Sum):C(eHMIOn, Sum):C(camera, Sum)": "Yielding x eHMI x order",
        "C(distPed_m, Sum)": "Distance",
        "C(distPed_m, Sum):C(yielding, Sum)": "Distance x yielding",
        "C(distPed_m, Sum):C(eHMIOn, Sum)": "Distance x eHMI",
        "C(distPed_m, Sum):C(camera, Sum)": "Distance x order",
        "trial_number_centered": "Trial number (linear)",
        "trial_number_centered:C(yielding, Sum)": "Trial x yielding",
        "trial_number_centered:C(eHMIOn, Sum)": "Trial x eHMI",
        "trial_number_centered:C(yielding, Sum):C(eHMIOn, Sum)": "Trial x yielding x eHMI",
        "I(trial_number_centered ** 2)": "Trial number (quadratic)",
    }

    def run_term_wise_wald_tests(self, trial_df: pd.DataFrame) -> pd.DataFrame:
        """Joint Wald test for every term of the primary model under effect coding.

        Effect (sum-to-zero) coding spans the same model space as the primary
        treatment-coded fit, so predictions and contrasts are unchanged, but
        each lower-order term is then tested averaged over the levels of the
        factors it interacts with rather than at a reference cell.
        """
        label = "common_window_primary_term_wise"
        success_column = "unsafe_bins_common_window"
        valid_column = "valid_bins_common_window"
        current = self._prepare_grouped_binomial_frame(
            trial_df=trial_df,
            success_column=success_column,
            valid_column=valid_column,
            model_mode="full",
            analysis_label=label,
        )
        rhs = (
            self._binomial_rhs_formula("full")
            .replace("C(yielding)", "C(yielding, Sum)")
            .replace("C(eHMIOn)", "C(eHMIOn, Sum)")
            .replace("C(camera)", "C(camera, Sum)")
            .replace("C(distPed_m)", "C(distPed_m, Sum)")
        )
        design = dmatrix(rhs, current, return_type="dataframe")
        successes = current[success_column].to_numpy(dtype=float)
        failures = (current[valid_column] - current[success_column]).to_numpy(dtype=float)
        fit = sm.GLM(
            endog=np.column_stack([successes, failures]),
            exog=design,
            family=sm.families.Binomial(),
        ).fit(cov_type="cluster", cov_kwds={"groups": current["participant"]})

        rows = []
        for term_name, columns in design.design_info.term_name_slices.items():
            if term_name == "Intercept":
                continue
            names = list(design.columns[columns])
            row = self._joint_wald_row(fit, names, term_name, label)
            row["term_label"] = self._TERM_LABELS.get(term_name, term_name)
            rows.append(row)
        table = pd.DataFrame.from_records(rows)
        table["coding"] = "effect (sum-to-zero) coding; logit scale"
        self._save_table(table, f"{label}_wald_tests.csv")
        return table

    # ------------------------------------------------------------------
    # Presentation-order balance
    # ------------------------------------------------------------------
    def summarise_presentation_order(self, trial_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Describe how evenly each condition was spread over trial positions."""
        current = trial_df.copy()
        for column in ("participant", "trial_number", "yielding", "eHMIOn", "camera", "distPed_m"):
            current[column] = pd.to_numeric(current[column], errors="coerce")
        current = current.dropna(
            subset=["participant", "trial_number", "yielding", "eHMIOn", "camera", "distPed_m"]
        )
        factors = ["yielding", "eHMIOn", "camera", "distPed_m"]

        by_condition = (
            current.groupby(factors)["trial_number"]
            .agg(
                n="size",
                mean_position="mean",
                sd_position="std",
                min_position="min",
                max_position="max",
                share_in_first_half=lambda values: float(np.mean(values <= 20)),
            )
            .reset_index()
        )
        wide = current.pivot_table(
            index="participant", columns=factors, values="trial_number"
        ).dropna()
        friedman_statistic, friedman_p = friedmanchisquare(
            *[wide[column].to_numpy() for column in wide.columns]
        )

        factor_rows = []
        for factor in ("yielding", "eHMIOn", "camera"):
            participant_means = current.pivot_table(
                index="participant", columns=factor, values="trial_number", aggfunc="mean"
            ).dropna()
            difference = participant_means[1] - participant_means[0]
            n = len(difference)
            mean = float(difference.mean())
            se = float(difference.std(ddof=1) / np.sqrt(n))
            t_value = mean / se if se > 0 else np.nan
            factor_rows.append(
                {
                    "factor": factor,
                    "mean_position_level_0": float(participant_means[0].mean()),
                    "mean_position_level_1": float(participant_means[1].mean()),
                    "mean_within_participant_difference": mean,
                    "t_value": t_value,
                    "df": n - 1,
                    "p_value": float(2 * student_t.sf(abs(t_value), n - 1)),
                }
            )
        distance_means = current.pivot_table(
            index="participant", columns="distPed_m", values="trial_number", aggfunc="mean"
        ).dropna()
        distance_statistic, distance_p = friedmanchisquare(
            *[distance_means[column].to_numpy() for column in distance_means.columns]
        )
        factor_rows.append(
            {
                "factor": "distPed_m",
                "mean_position_level_0": np.nan,
                "mean_position_level_1": np.nan,
                "mean_within_participant_difference": np.nan,
                "t_value": np.nan,
                "df": distance_means.shape[1] - 1,
                "p_value": float(distance_p),
                "friedman_chi2": float(distance_statistic),
            }
        )
        by_factor = pd.DataFrame.from_records(factor_rows)
        by_factor["all_40_conditions_friedman_chi2"] = float(friedman_statistic)
        by_factor["all_40_conditions_friedman_df"] = int(wide.shape[1] - 1)
        by_factor["all_40_conditions_friedman_p"] = float(friedman_p)
        by_factor["condition_mean_position_min"] = float(by_condition["mean_position"].min())
        by_factor["condition_mean_position_max"] = float(by_condition["mean_position"].max())

        self._save_table(by_condition, "presentation_order_by_condition.csv")
        self._save_table(by_factor, "presentation_order_balance_by_factor.csv")
        return {"by_condition": by_condition, "by_factor": by_factor}

    # ------------------------------------------------------------------
    # Logged vehicle event timing
    # ------------------------------------------------------------------
    def summarise_vehicle_event_timing(self) -> pd.DataFrame:
        """Summarise the vehicle event times used by the analysis.

        ``self.mapping_df`` carries the constant scripted schedule
        (``utils.vehicle_events``); the raw simulator log is appended for
        comparison.

        In the mapping, P2 denotes the first roadside position in the vehicle's
        path and P1 the second, irrespective of whether the participant or the
        avatar stood there.
        """
        mapping = self.mapping_df.copy()
        mapping = mapping.loc[mapping["video_id"].astype(str).str.startswith("video_")].copy()
        numeric = [
            "yielding", "camera", "yield_start_time_s", "yield_start_speed_kmh",
            "yield_start_dP2_m", "yield_stop_time_s", "yield_stop_dP2_m",
            "yield_resume_time_s", "yield_end_time_s", "cross_p2_time_s",
            "cross_p1_time_s",
        ]
        for column in numeric:
            if column in mapping.columns:
                mapping[column] = pd.to_numeric(mapping[column], errors="coerce")
        mapping["participant_passage_s"] = np.where(
            mapping["camera"].eq(1), mapping["cross_p2_time_s"], mapping["cross_p1_time_s"]
        )
        mapping["first_position_passage_s"] = mapping["cross_p2_time_s"]
        yielding = mapping["yielding"].eq(1)
        mapping.loc[yielding, "braking_duration_s"] = (
            mapping["yield_stop_time_s"] - mapping["yield_start_time_s"]
        )
        mapping.loc[yielding, "mean_deceleration_ms2"] = (
            mapping["yield_start_speed_kmh"] / 3.6 / mapping["braking_duration_s"]
        )
        mapping.loc[yielding, "standstill_duration_s"] = (
            mapping["yield_resume_time_s"] - mapping["yield_stop_time_s"]
        )
        mapping.loc[yielding, "resume_to_first_passage_s"] = (
            mapping["cross_p2_time_s"] - mapping["yield_resume_time_s"]
        )

        measures = [
            "yield_start_time_s", "yield_start_dP2_m", "yield_stop_time_s",
            "yield_stop_dP2_m", "yield_resume_time_s", "braking_duration_s",
            "mean_deceleration_ms2", "standstill_duration_s",
            "first_position_passage_s", "resume_to_first_passage_s",
            "participant_passage_s",
        ]
        rows = []
        for (yield_value, camera), group in mapping.groupby(["yielding", "camera"]):
            for measure in measures:
                values = pd.to_numeric(group.get(measure), errors="coerce").dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "yielding": int(yield_value),
                        "camera": int(camera),
                        "order": _ORDER_LABELS[int(camera)],
                        "measure": measure,
                        "n_conditions": int(values.size),
                        "min": float(values.min()),
                        "median": float(values.median()),
                        "max": float(values.max()),
                    }
                )
        table = pd.DataFrame.from_records(rows)
        table["source"] = "analysis schedule (constant scripted events)"
        # Also document the raw simulator log for comparison. Standstill and
        # drive-off there were detected from a 0.5-s speed estimate.
        try:
            logged = pd.read_csv(common.get_configs("mapping"))
            logged = logged.loc[logged["video_id"].astype(str).str.startswith("video_")]
            logged_rows = []
            for column in (
                "yield_start_time_s", "yield_stop_time_s", "yield_resume_time_s",
                "cross_p2_time_s", "cross_p1_time_s",
            ):
                for (yield_value, camera), group in logged.groupby(["yielding", "camera"]):
                    values = pd.to_numeric(group[column], errors="coerce").dropna()
                    if values.empty:
                        continue
                    logged_rows.append(
                        {
                            "yielding": int(yield_value),
                            "camera": int(camera),
                            "order": _ORDER_LABELS[int(camera)],
                            "measure": column,
                            "n_conditions": int(values.size),
                            "min": float(values.min()),
                            "median": float(values.median()),
                            "max": float(values.max()),
                            "source": "raw simulator log (mapping file)",
                        }
                    )
            table = pd.concat([table, pd.DataFrame.from_records(logged_rows)], ignore_index=True)
        except (KeyError, FileNotFoundError) as exc:
            logger.warning(f"Raw mapping file not available for timing comparison: {exc}")
        self._save_table(table, "vehicle_event_timing_summary.csv")
        return table

    # ------------------------------------------------------------------
    # Trial-wise rating models (manuscript table of Q1--Q3 coefficients)
    # ------------------------------------------------------------------
    def run_rating_models(self, trial_df: pd.DataFrame) -> pd.DataFrame:
        """Gaussian participant random-intercept models for Q1--Q3 (ML fit).

        Fixed effects: yielding, eHMI, configuration, linear distance (m) and
        the two-way interactions among yielding, eHMI and configuration.
        """
        import statsmodels.formula.api as smf

        rhs = (
            "C(yielding) + C(eHMIOn) + C(camera) + distPed_m + "
            "C(yielding):C(eHMIOn) + C(yielding):C(camera) + C(eHMIOn):C(camera)"
        )
        rows = []
        for outcome in ("Q1", "Q2", "Q3"):
            current = trial_df.dropna(subset=[outcome, "distPed_m", "participant"])
            fit = smf.mixedlm(
                f"{outcome} ~ {rhs}", current, groups=current["participant"]
            ).fit(reml=False)
            intervals = fit.conf_int()
            for term in fit.fe_params.index:
                rows.append(
                    {
                        "outcome": outcome,
                        "term": term,
                        "estimate": float(fit.params[term]),
                        "ci_lower": float(intervals.loc[term, 0]),
                        "ci_upper": float(intervals.loc[term, 1]),
                        "p_value": float(fit.pvalues[term]),
                        "model": "Gaussian participant random intercept, ML",
                        "formula": f"{outcome} ~ {rhs}",
                    }
                )
        table = pd.DataFrame.from_records(rows)
        self._save_table(table, "rating_models_random_intercept_coefficients.csv")
        return table

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run_revision_analyses(
        self,
        trial_df: pd.DataFrame,
        primary_result: Dict[str, object],
        bootstrap_result: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, object]:
        """Run every second-revision analysis on the primary data construction."""
        logger.info("Running second-revision analyses.")
        draws = (
            bootstrap_result.get("marginal_probability_draws")
            if bootstrap_result is not None
            else None
        )
        return {
            "interaction_contrasts": self.run_interaction_contrasts(
                primary_result, "common_window_primary", bootstrap_marginal_draws=draws
            ),
            "trial_level_sensitivity": self.run_trial_level_sensitivity(
                trial_df, primary_result
            ),
            "animation_exposure_learning": self.run_animation_exposure_learning(trial_df),
            "term_wise_wald_tests": self.run_term_wise_wald_tests(trial_df),
            "presentation_order": self.summarise_presentation_order(trial_df),
            "vehicle_event_timing": self.summarise_vehicle_event_timing(),
            "rating_models": self.run_rating_models(trial_df),
        }
