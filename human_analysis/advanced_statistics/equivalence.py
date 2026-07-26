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


class EquivalenceMixin:
    """Focused method group extracted without changing calculation logic."""

    @staticmethod
    def _paired_tost(diff: Iterable[float], low_eq: float, high_eq: float,
                     alpha: float = 0.05) -> TOSTResult:

        """Run a paired two one sided tests procedure.

        Args:
            diff: Iterable of paired differences.
            low_eq: Lower equivalence bound.
            high_eq: Upper equivalence bound.
            alpha: Significance level for each one sided test.

        Returns:
            A populated ``TOSTResult`` instance containing sample statistics,
            confidence intervals, one sided TOST results, and a conventional
            paired t test against zero.
        """
        # Convert the incoming paired differences into a numeric NumPy array.
        diff_arr = np.asarray(list(diff), dtype=float)
        diff_arr = diff_arr[np.isfinite(diff_arr)]
        n = int(len(diff_arr))

        # Return early when there are too few observations for paired inference.
        if n < 2:
            return TOSTResult(
                label="",
                n=n,
                mean_diff=np.nan,
                sd_diff=np.nan,
                se_diff=np.nan,
                ci90_low=np.nan,
                ci90_high=np.nan,
                ci95_low=np.nan,
                ci95_high=np.nan,
                t_lower=np.nan,
                p_lower=np.nan,
                t_upper=np.nan,
                p_upper=np.nan,
                p_tost=np.nan,
                equivalent=False,
                t_paired=np.nan,
                p_paired=np.nan,
                margin_low=float(low_eq),
                margin_high=float(high_eq),
            )

        # Compute the core paired sample summary statistics.
        mean_diff = float(np.mean(diff_arr))
        sd_diff = float(np.std(diff_arr, ddof=1))
        se_diff = float(sd_diff / np.sqrt(n))
        dfree = n - 1

        # Handle degenerate standard errors without raising an exception.
        if not np.isfinite(se_diff) or se_diff == 0.0:
            return TOSTResult(
                label="",
                n=n,
                mean_diff=mean_diff,
                sd_diff=sd_diff,
                se_diff=se_diff,
                ci90_low=np.nan,
                ci90_high=np.nan,
                ci95_low=np.nan,
                ci95_high=np.nan,
                t_lower=np.nan,
                p_lower=np.nan,
                t_upper=np.nan,
                p_upper=np.nan,
                p_tost=np.nan,
                equivalent=False,
                t_paired=np.nan,
                p_paired=np.nan,
                margin_low=float(low_eq),
                margin_high=float(high_eq),
            )

        # Compute the critical values needed for confidence intervals.
        crit90 = float(student_t.ppf(1.0 - alpha, df=dfree))
        crit95 = float(student_t.ppf(1.0 - alpha / 2.0, df=dfree))

        # Report both 90 percent and 95 percent intervals because the 90 percent interval is
        # directly relevant for equivalence testing.
        ci90_low = float(mean_diff - crit90 * se_diff)
        ci90_high = float(mean_diff + crit90 * se_diff)
        ci95_low = float(mean_diff - crit95 * se_diff)
        ci95_high = float(mean_diff + crit95 * se_diff)

        # Form the two one sided test statistics for the equivalence bounds.
        t_lower = float((mean_diff - low_eq) / se_diff)
        p_lower = float(1.0 - student_t.cdf(t_lower, df=dfree))

        t_upper = float((mean_diff - high_eq) / se_diff)
        p_upper = float(student_t.cdf(t_upper, df=dfree))

        # Combine the two one sided p values into the final TOST decision quantity.
        p_tost = float(max(p_lower, p_upper))
        equivalent = bool((p_lower < alpha) and (p_upper < alpha))

        # Also compute the conventional paired t test against zero for reference.
        paired_result = ttest_rel(diff_arr, np.zeros_like(diff_arr))
        t_paired = float(paired_result.statistic)  # pyright: ignore[reportAttributeAccessIssue]
        p_paired = float(paired_result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]

        return TOSTResult(
            label="",
            n=n,
            mean_diff=mean_diff,
            sd_diff=sd_diff,
            se_diff=se_diff,
            ci90_low=ci90_low,
            ci90_high=ci90_high,
            ci95_low=ci95_low,
            ci95_high=ci95_high,
            t_lower=t_lower,
            p_lower=p_lower,
            t_upper=t_upper,
            p_upper=p_upper,
            p_tost=p_tost,
            equivalent=equivalent,
            t_paired=t_paired,
            p_paired=p_paired,
            margin_low=float(low_eq),
            margin_high=float(high_eq),
        )

    def run_equivalence_tests(self, trial_df: pd.DataFrame, outcome: str = "crossing_risk",
                              low_distances_m: Tuple[int, int] = (2, 4),
                              high_distances_m: Tuple[int, int] = (8, 10), equivalence_margin: float = 5.0,
                              alpha: float = 0.05) -> pd.DataFrame:

        """Run paired TOST comparisons for near versus far distances.

        The comparison is calculated once across all contexts and again within
        each yielding by eHMI by camera combination.

        Args:
            trial_df: Trial level DataFrame with scenario metadata.
            outcome: Column to compare between near and far distances.
            low_distances_m: Actual distance values in metres to treat as near.
            high_distances_m: Actual distance values in metres to treat as far.
            equivalence_margin: Symmetric equivalence margin in outcome units.
            alpha: Significance level for the one sided tests.

        Returns:
            A DataFrame with one TOST summary row per comparison context.

        Raises:
            ValueError: If no valid data remains after filtering.
        """
        df = trial_df.copy()
        if "distPed_m" not in df.columns:
            if "distPed" not in df.columns:
                raise ValueError("Neither 'distPed_m' nor 'distPed' is available in the trial table.")
            df["distPed_m"] = self._distance_series_to_meters(df["distPed"])
        else:
            # distPed_m is already physical distance. Never pass it through the
            # raw-code converter because 2 and 4 are valid in both domains.
            df["distPed_m"] = self._distance_meters_series(df["distPed_m"])

        df[outcome] = pd.to_numeric(df[outcome], errors="coerce")
        df = df.dropna(subset=["participant", "distPed_m", outcome, "yielding", "eHMIOn", "camera"])
        if df.empty:
            raise ValueError(f"No valid rows are available for equivalence testing of {outcome}.")

        df["distance_band"] = pd.Series(pd.NA, index=df.index, dtype="object")
        df.loc[df["distPed_m"].isin(low_distances_m), "distance_band"] = "near"
        df.loc[df["distPed_m"].isin(high_distances_m), "distance_band"] = "far"
        df = df.loc[df["distance_band"].isin(["near", "far"])].copy()
        if df.empty:
            raise ValueError("No near/far rows remained after distance band selection.")

        def _yield_label(val: object) -> str:
            return "Yielding" if int(val) == 1 else "Non-yielding"  # pyright: ignore[reportArgumentType]

        def _ehmi_label(val: object) -> str:
            return "eHMI" if int(val) == 1 else "No eHMI"  # pyright: ignore[reportArgumentType]

        def _order_label(val: object) -> str:
            return (
                "Participant first / avatar second"
                if int(val) == 1
                else "Avatar first / participant second"
            )  # type: ignore

        result_records: List[Dict[str, object]] = []

        overall = (
            df.groupby(["participant", "distance_band"], as_index=False)[outcome]
            .mean()
            .pivot(index="participant", columns="distance_band", values=outcome)
            .dropna(subset=["near", "far"])
        )
        if not overall.empty:
            tost = self._paired_tost(
                overall["near"] - overall["far"],
                low_eq=-equivalence_margin,
                high_eq=equivalence_margin,
                alpha=alpha,
            )
            result_records.append({
                **tost.__dict__,
                "label": "Overall",
                "context": "Overall",
                "display_label": "Overall",
                "yielding": np.nan,
                "eHMIOn": np.nan,
                "camera": np.nan,
            })
            logger.info(
                f"Overall TOST for {outcome}: mean diff = {tost.mean_diff:.3f}, "
                f"90% CI [{tost.ci90_low:.3f}, {tost.ci90_high:.3f}], p_tost = {tost.p_tost:.4g}, "
                f"equivalent = {tost.equivalent}"
            )

        ctx_cols = ["yielding", "eHMIOn", "camera"]
        for ctx, ctx_df in df.groupby(ctx_cols):
            pivot = (
                ctx_df.groupby(["participant", "distance_band"], as_index=False)[outcome]
                .mean()
                .pivot(index="participant", columns="distance_band", values=outcome)
                .dropna(subset=["near", "far"])
            )
            if pivot.empty:
                continue

            tost = self._paired_tost(
                pivot["near"] - pivot["far"],
                low_eq=-equivalence_margin,
                high_eq=equivalence_margin,
                alpha=alpha,
            )
            label = f"Y{int(ctx[0])} H{int(ctx[1])} C{int(ctx[2])}"
            display_label = f"{_yield_label(ctx[0])}, {_ehmi_label(ctx[1])}, {_order_label(ctx[2])}"
            result_records.append({
                **tost.__dict__,
                "label": label,
                "context": label,
                "display_label": display_label,
                "yielding": int(ctx[0]),
                "eHMIOn": int(ctx[1]),
                "camera": int(ctx[2]),
            })
            logger.info(
                f"Context {label} TOST for {outcome}: mean diff = {tost.mean_diff:.3f}, "
                f"90% CI [{tost.ci90_low:.3f}, {tost.ci90_high:.3f}], p_tost = {tost.p_tost:.4g}, "
                f"equivalent = {tost.equivalent}"
            )

        results_df = pd.DataFrame(result_records)
        if results_df.empty:
            raise ValueError("No equivalence results could be computed.")

        self._save_table(results_df, f"equivalence_near_vs_far_{outcome}.csv")
        # Build a faceted equivalence figure.
        # Rows separate relative order, columns separate eHMI, and each panel shows
        # two yielding states. This keeps labels short and publication friendly.
        fig = make_subplots(
            rows=3,
            cols=2,
            specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]],
            subplot_titles=[
                "Overall",
                "Avatar first / participant second | No eHMI",
                "Avatar first / participant second | eHMI",
                "Participant first / avatar second | No eHMI",
                "Participant first / avatar second | eHMI",
            ],
            shared_xaxes=True,
            shared_yaxes=False,
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
            row_heights=[0.20, 0.40, 0.40],
        )

        plot_df = results_df.copy()
        finite_bounds = pd.concat(
            [plot_df["ci90_low"], plot_df["ci90_high"], plot_df["mean_diff"]],
            ignore_index=True,
        )
        finite_bounds = pd.to_numeric(finite_bounds, errors="coerce")
        finite_bounds = finite_bounds[np.isfinite(finite_bounds)]
        if finite_bounds.empty:
            x_limit = float(equivalence_margin + 1.0)
        else:
            x_limit = float(max(equivalence_margin, np.abs(finite_bounds).max()))
            x_limit += max(1.0, 0.08 * x_limit)

        panel_positions = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2)]
        for row_idx, col_idx in panel_positions:
            fig.add_vrect(
                x0=-equivalence_margin,
                x1=equivalence_margin,
                fillcolor="rgba(50, 50, 50, 0.08)",
                line_width=0,
                row=row_idx,  # pyright: ignore[reportArgumentType]
                col=col_idx,  # pyright: ignore[reportArgumentType]
            )
            fig.add_vline(x=0, line_dash="dash", line_color="black", row=row_idx, col=col_idx)  # type: ignore
            fig.update_xaxes(range=[-x_limit, x_limit], row=row_idx, col=col_idx)

        overall_df = plot_df.loc[plot_df["label"] == "Overall"]
        if not overall_df.empty:
            row = overall_df.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[row["mean_diff"]],
                    y=["Overall"],
                    mode="markers",
                    marker=dict(size=12, symbol="diamond-open" if not row["equivalent"] else "diamond"),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[row["ci90_high"] - row["mean_diff"]],
                        arrayminus=[row["mean_diff"] - row["ci90_low"]],
                        thickness=1.8,
                        width=0,
                    ),
                    showlegend=False,
                    hovertemplate=(
                        "<b>Overall</b><br>Near minus far: %{x:.2f}<br>"
                        f"TOST p: {row['p_tost']:.4g}<br>"
                        f"Equivalent: {row['equivalent']}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=["Overall"],
                row=1,
                col=1,
            )

        context_df = plot_df.loc[plot_df["label"] != "Overall"].copy()
        context_df["yield_label"] = context_df["yielding"].map(_yield_label)
        context_df["panel_row"] = context_df["camera"].map({0: 2, 1: 3})
        context_df["panel_col"] = context_df["eHMIOn"].map({0: 1, 1: 2})

        for _, row in context_df.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["mean_diff"]],
                    y=[row["yield_label"]],
                    mode="markers",
                    marker=dict(size=11, symbol="circle-open" if not row["equivalent"] else "circle"),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[row["ci90_high"] - row["mean_diff"]],
                        arrayminus=[row["mean_diff"] - row["ci90_low"]],
                        thickness=1.6,
                        width=0,
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{row['display_label']}</b><br>Near minus far: %{{x:.2f}}<br>"
                        f"TOST p: {row['p_tost']:.4g}<br>"
                        f"Equivalent: {row['equivalent']}<extra></extra>"
                    ),
                ),
                row=int(row["panel_row"]),
                col=int(row["panel_col"]),
            )

        for r in [2, 3]:
            for c in [1, 2]:
                fig.update_yaxes(
                    categoryorder="array",
                    categoryarray=["Non-yielding", "Yielding"],
                    row=r,
                    col=c,
                )

        fig.update_layout(
            template=self.template,
            title="",
            font=dict(family=self.font_family, size=self.font_size + 2),
            margin=dict(l=0, r=0, t=0, b=0),
            height=900,
        )
        fig.update_annotations(font=dict(family=self.font_family, size=self.font_size + 4))

        # Only the left panels need a y axis title.
        fig.update_yaxes(title_text="", title_font=dict(family=self.font_family, size=self.font_size + 10),
                         tickfont=dict(family=self.font_family, size=self.font_size + 6), row=1, col=1)
        fig.update_yaxes(title_text="", title_font=dict(family=self.font_family, size=self.font_size + 8),
                         tickfont=dict(family=self.font_family, size=self.font_size + 6), row=2, col=1)
        fig.update_yaxes(title_text="", title_font=dict(family=self.font_family, size=self.font_size + 8),
                         tickfont=dict(family=self.font_family, size=self.font_size + 6), row=3, col=1)
        fig.update_yaxes(tickfont=dict(family=self.font_family, size=self.font_size + 6), row=2, col=2)
        fig.update_yaxes(tickfont=dict(family=self.font_family, size=self.font_size + 6), row=3, col=2)

        for r in [1, 2, 3]:
            for c in [1, 2]:
                fig.update_xaxes(
                    title_font=dict(family=self.font_family, size=self.font_size + 10),
                    tickfont=dict(family=self.font_family, size=self.font_size + 6),
                    automargin=True,
                    row=r,
                    col=c,
                )
        pretty_outcome = self._pretty_outcome_label(outcome)
        fig.update_xaxes(title_text=f"Near minus far difference in {pretty_outcome}", row=3, col=1)
        fig.update_xaxes(title_text=f"Near minus far difference in {pretty_outcome}", row=3, col=2)

        self.helper.save_plotly(
            fig=fig,
            name=f"equivalence_near_vs_far_{outcome}",
            width=1300,
            height=900,
            save_final=True,
            open_browser=True,
        )
        logger.info(f"Saved figure set for: equivalence_near_vs_far_{outcome}")
        return results_df
