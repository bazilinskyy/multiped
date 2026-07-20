from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

import common
from custom_logger import CustomLogger
from .settings import *

logger = CustomLogger(__name__)
from .features import mean_in_window

def one_sample_summary(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n = len(values)
    if n < 2:
        return {"n": n, "estimate": np.nan, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "t": np.nan, "p_raw": np.nan}
    estimate = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    critical = float(stats.t.ppf(0.975, n - 1))
    t_value, p_value = stats.ttest_1samp(values, 0.0)
    return {
        "n": n,
        "estimate": estimate,
        "se": se,
        "ci_lower": estimate - critical * se,
        "ci_upper": estimate + critical * se,
        "t": float(t_value),
        "p_raw": float(p_value),
    }

def factor_contrast(features: pd.DataFrame, metric: str, factor: str, high: object, low: object) -> pd.Series:
    grouped = features.groupby(["participant", factor], observed=True)[metric].mean().unstack(factor)
    return grouped[high] - grouped[low]

def interaction_contrast(
    features: pd.DataFrame,
    metric: str,
    factor_a: str,
    high_a: object,
    low_a: object,
    factor_b: str,
    high_b: object,
    low_b: object,
) -> pd.Series:
    grouped = features.groupby(["participant", factor_a, factor_b], observed=True)[metric].mean().unstack([factor_a, factor_b])
    return (grouped[(high_a, high_b)] - grouped[(high_a, low_b)]) - (
        grouped[(low_a, high_b)] - grouped[(low_a, low_b)]
    )

def holm_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return pd.Series(adjusted, index=p_values.index)
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = (m - rank) * p[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index)

def repeated_measures_anova(matrix: pd.DataFrame) -> dict[str, float | int]:
    """One-factor repeated-measures ANOVA on complete participant-by-level matrix."""
    complete = matrix.dropna(axis=0, how="any").to_numpy(float)
    n, k = complete.shape if complete.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return {"n": n, "levels": k, "df1": np.nan, "df2": np.nan, "F": np.nan, "p_raw": np.nan}
    grand = complete.mean()
    participant_means = complete.mean(axis=1)
    level_means = complete.mean(axis=0)
    ss_total = np.sum((complete - grand) ** 2)
    ss_participant = k * np.sum((participant_means - grand) ** 2)
    ss_level = n * np.sum((level_means - grand) ** 2)
    ss_error = ss_total - ss_participant - ss_level
    df1 = k - 1
    df2 = (n - 1) * (k - 1)
    f_value = (ss_level / df1) / (ss_error / df2)
    return {
        "n": n,
        "levels": k,
        "df1": df1,
        "df2": df2,
        "F": float(f_value),
        "p_raw": float(stats.f.sf(f_value, df1, df2)),
    }

def build_contrast_table(features: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    definitions = [
        ("yielding_minus_non_yielding", lambda d, m: factor_contrast(d, m, "yielding", 1, 0)),
        ("conditional_eHMI_on_minus_off", lambda d, m: factor_contrast(d, m, "eHMIOn", 1, 0)),
        ("PF_AS_minus_AF_PS", lambda d, m: factor_contrast(d, m, "order", "PF_AS", "AF_PS")),
        (
            "yielding_x_conditional_eHMI",
            lambda d, m: interaction_contrast(d, m, "yielding", 1, 0, "eHMIOn", 1, 0),
        ),
        (
            "yielding_x_order",
            lambda d, m: interaction_contrast(d, m, "yielding", 1, 0, "order", "PF_AS", "AF_PS"),
        ),
        (
            "conditional_eHMI_x_order",
            lambda d, m: interaction_contrast(d, m, "eHMIOn", 1, 0, "order", "PF_AS", "AF_PS"),
        ),
    ]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for name, function in definitions:
            summary = one_sample_summary(function(features, metric))
            rows.append({"metric": metric, "contrast": name, **summary})
    result = pd.DataFrame(rows)
    result["p_holm_within_metric"] = result.groupby("metric", group_keys=False)["p_raw"].apply(holm_adjust)
    return result

def build_distance_table(features: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        base = features.groupby(["participant", "spacing_m"], observed=True)[metric].mean().unstack("spacing_m")
        rows.append({"metric": metric, "test": "distance", **repeated_measures_anova(base)})
        for factor, high, low in (("yielding", 1, 0), ("eHMIOn", 1, 0), ("order", "PF_AS", "AF_PS")):
            cells = features.groupby(["participant", "spacing_m", factor], observed=True)[metric].mean().unstack(factor)
            difference = (cells[high] - cells[low]).unstack("spacing_m")
            rows.append(
                {"metric": metric, "test": f"distance_x_{factor}", **repeated_measures_anova(difference)}
            )
    result = pd.DataFrame(rows)
    result["p_holm_within_metric"] = result.groupby("metric", group_keys=False)["p_raw"].apply(holm_adjust)
    return result

def build_simple_contrast_table(features: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Condition-specific paired contrasts, with Holm correction over 12 per metric."""
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for order in ("AF_PS", "PF_AS"):
            for ehmi in (0, 1):
                subset = features[(features["order"] == order) & (features["eHMIOn"] == ehmi)]
                values = factor_contrast(subset, metric, "yielding", 1, 0)
                rows.append(
                    {
                        "metric": metric,
                        "family": "yielding_within_order_eHMI",
                        "contrast": f"yielding_minus_non_yielding|order={order}|eHMI={ehmi}",
                        **one_sample_summary(values),
                    }
                )
        for yielding in (0, 1):
            for ehmi in (0, 1):
                subset = features[(features["yielding"] == yielding) & (features["eHMIOn"] == ehmi)]
                values = factor_contrast(subset, metric, "order", "PF_AS", "AF_PS")
                rows.append(
                    {
                        "metric": metric,
                        "family": "order_within_yielding_eHMI",
                        "contrast": f"PF_AS_minus_AF_PS|yielding={yielding}|eHMI={ehmi}",
                        **one_sample_summary(values),
                    }
                )
        for yielding in (0, 1):
            for order in ("AF_PS", "PF_AS"):
                subset = features[(features["yielding"] == yielding) & (features["order"] == order)]
                values = factor_contrast(subset, metric, "eHMIOn", 1, 0)
                rows.append(
                    {
                        "metric": metric,
                        "family": "eHMI_within_yielding_order",
                        "contrast": f"eHMI_on_minus_off|yielding={yielding}|order={order}",
                        **one_sample_summary(values),
                    }
                )
    result = pd.DataFrame(rows)
    result["p_holm_12_within_metric"] = result.groupby("metric", group_keys=False)["p_raw"].apply(holm_adjust)
    return result

def build_yielding_event_table(features: pd.DataFrame) -> pd.DataFrame:
    yielding = features[features["yielding"] == 1]
    rows: list[dict[str, object]] = []
    for metric in ("braking_change_deg", "stopping_change_deg", "resumption_change_deg"):
        overall = yielding.groupby("participant", observed=True)[metric].mean()
        rows.append({"metric": metric, "contrast": "post_minus_pre", **one_sample_summary(overall)})
        rows.append(
            {
                "metric": metric,
                "contrast": "PF_AS_minus_AF_PS",
                **one_sample_summary(factor_contrast(yielding, metric, "order", "PF_AS", "AF_PS")),
            }
        )
        rows.append(
            {
                "metric": metric,
                "contrast": "eHMI_on_minus_off",
                **one_sample_summary(factor_contrast(yielding, metric, "eHMIOn", 1, 0)),
            }
        )
        rows.append(
            {
                "metric": metric,
                "contrast": "eHMI_x_order",
                **one_sample_summary(
                    interaction_contrast(yielding, metric, "eHMIOn", 1, 0, "order", "PF_AS", "AF_PS")
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm_12"] = holm_adjust(result["p_raw"])
    return result

def condition_summary(features: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    participant_cells = (
        features.groupby(["participant", "yielding", "eHMIOn", "order", "spacing_m"], observed=True)[metrics]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    factors = ["yielding", "eHMIOn", "order", "spacing_m"]
    for keys, group in participant_cells.groupby(factors, observed=True):
        base = dict(zip(factors, keys))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            n = len(values)
            mean = float(np.mean(values)) if n else np.nan
            se = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else np.nan
            critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else np.nan
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_participants": n,
                    "mean": mean,
                    "se": se,
                    "ci_lower": mean - critical * se,
                    "ci_upper": mean + critical * se,
                }
            )
    return pd.DataFrame(rows)

def curve_summary(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Average trials within participant before estimating between-participant uncertainty.
    participant = (
        curves.groupby(["participant", "yielding", "eHMIOn", "order", "time_rel_pass_s"], observed=True)["heading_deg"]
        .mean()
        .reset_index()
    )
    group_cols = ["yielding", "eHMIOn", "order", "time_rel_pass_s"]
    summary = participant.groupby(group_cols, observed=True)["heading_deg"].agg(["count", "mean", "std"]).reset_index()
    summary = summary.rename(columns={"count": "n_participants"})
    summary["se"] = summary["std"] / np.sqrt(summary["n_participants"])
    critical = stats.t.ppf(0.975, summary["n_participants"] - 1)
    summary["ci_lower"] = summary["mean"] - critical * summary["se"]
    summary["ci_upper"] = summary["mean"] + critical * summary["se"]
    return participant, summary.drop(columns="std")

def passage_order_tests(features: pd.DataFrame) -> pd.DataFrame:
    """Paired order contrasts for the pre-specified passage window in each panel.

    The event-aligned figure is descriptive at every time point.  To avoid a
    serially dependent sequence of pointwise tests, significance annotations
    refer only to the participant-level mean heading in the 200-ms passage
    window (the same outcome used in the numerical passage analysis).
    """
    participant = (
        features.groupby(["participant", "yielding", "order"], observed=True)["heading_at_pass_deg"]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for yielding in (0, 1):
        wide = participant[participant["yielding"].eq(yielding)].pivot(
            index="participant", columns="order", values="heading_at_pass_deg"
        )
        wide = wide.dropna(subset=["AF_PS", "PF_AS"])
        summary = one_sample_summary(wide["PF_AS"] - wide["AF_PS"])
        rows.append({"yielding": yielding, **summary})
    result = pd.DataFrame(rows)
    result["p_holm_across_panels"] = holm_adjust(result["p_raw"])

    def significance_stars(value: float) -> str:
        if value < 0.001:
            return "***"
        if value < 0.01:
            return "**"
        if value < 0.05:
            return "*"
        return "n.s."

    result["significance"] = result["p_holm_across_panels"].map(significance_stars)
    return result

def pointwise_order_ttests(
    curves: pd.DataFrame,
    p_threshold: float = 0.001,
) -> pd.DataFrame:
    """Match the existing yaw figures' two-sided paired t-test marker method.

    For every configured event-aligned yaw bin, trials are first averaged within each
    participant, AV-behaviour panel, and relative-order condition.  PF/AS and
    AF/PS are then compared with a two-sided paired t-test.  As in the existing
    HMD plots, a bin is marked when its raw p value is below the configured
    ``p_value`` threshold (0.001 in the supplied configuration).
    """
    participant = (
        curves.groupby(
            ["participant", "yielding", "order", "time_rel_pass_s"],
            observed=True,
        )["heading_deg"]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for (yielding, time_rel), group in participant.groupby(
        ["yielding", "time_rel_pass_s"], observed=True
    ):
        wide = group.pivot(index="participant", columns="order", values="heading_deg")
        if not {"AF_PS", "PF_AS"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["AF_PS", "PF_AS"])
        if len(wide) < 2:
            t_statistic, p_value, estimate = np.nan, 1.0, np.nan
        else:
            t_statistic, p_value = stats.ttest_rel(
                wide["PF_AS"],
                wide["AF_PS"],
                alternative="two-sided",
            )
            if not np.isfinite(p_value):
                p_value = 1.0
            estimate = float((wide["PF_AS"] - wide["AF_PS"]).mean())
        rows.append(
            {
                "yielding": int(yielding),
                "time_rel_pass_s": float(time_rel),
                "n_pairs": int(len(wide)),
                "PF_AS_minus_AF_PS_deg": estimate,
                "t_statistic": float(t_statistic),
                "p_raw": float(p_value),
                "p_threshold": float(p_threshold),
                "significant": int(float(p_value) < float(p_threshold)),
            }
        )
    return pd.DataFrame(rows).sort_values(["yielding", "time_rel_pass_s"]).reset_index(drop=True)

def _significant_time_ranges(panel_tests: pd.DataFrame) -> str:
    """Compact consecutive significant event-aligned bins for the results log."""
    times = np.sort(
        panel_tests.loc[panel_tests["significant"].astype(bool), "time_rel_pass_s"].to_numpy(float)
    )
    if len(times) == 0:
        return "none"
    all_times = np.sort(panel_tests["time_rel_pass_s"].dropna().unique().astype(float))
    if len(all_times) > 1:
        consecutive_tolerance = 1.01 * float(np.median(np.diff(all_times)))
    else:
        consecutive_tolerance = 1.01 * HEAD_HEADING_TIME_BIN_S
    ranges: list[tuple[float, float]] = []
    start = previous = float(times[0])
    for value in times[1:]:
        value = float(value)
        if value - previous > consecutive_tolerance:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return ", ".join(
        f"{start:.2f} s" if abs(start - end) < 1e-9 else f"{start:.2f} to {end:.2f} s"
        for start, end in ranges
    )
