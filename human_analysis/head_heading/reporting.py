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
from .statistics import _significant_time_ranges
from .plots import _p_text

def _log_results(
    features: pd.DataFrame,
    contrasts: pd.DataFrame,
    distance: pd.DataFrame,
    event_contrasts: pd.DataFrame,
    figure_tests: pd.DataFrame,
    pointwise_tests: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    logger.info('\n=== Participant-level horizontal head-heading analysis ===')
    quality_map = quality.set_index("measure")["value"]
    logger.info(
        f"Records: {int(quality_map['trials']):,} participant-trials; "
        f"participants: {int(quality_map['participants'])}; "
        f"mean valid samples: {100.0 * float(quality_map['mean_valid_fraction']):.2f}%."
    )

    minimum_cells = (
        features.groupby(["participant", "yielding", "order"], observed=True)["minimum_heading_deg"]
        .mean()
        .reset_index()
    )
    minimum_rows: list[dict[str, object]] = []
    for (yielding, order), group in minimum_cells.groupby(["yielding", "order"], observed=True):
        values = group["minimum_heading_deg"].dropna()
        n = len(values)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(n))
        half = float(stats.t.ppf(0.975, n - 1) * se)
        minimum_rows.append(
            {
                "AV behaviour": "Yielding" if yielding else "Non-yielding",
                "Relative order": order,
                "Mean_deg": mean,
                "CI_low": mean - half,
                "CI_high": mean + half,
            }
        )
    minimum_table = pd.DataFrame(minimum_rows).to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}",
    )
    logger.info(
        "Minimum leftward heading before participant passage "
        f"(participant-marginal degrees; 95% CI):\n{minimum_table}"
    )
    minimum = contrasts[contrasts["metric"].eq("minimum_heading_deg")].set_index("contrast")
    minimum_main = minimum.loc[
        ["yielding_minus_non_yielding", "PF_AS_minus_AF_PS", "conditional_eHMI_on_minus_off"]
    ]
    logger.info(
        f"Minimum-heading main contrasts were not supported after Holm correction (smallest {_p_text(float(minimum_main['p_holm_within_metric'].min()))})."
    )
    participant_cells = (
        features.groupby(["participant", "yielding", "order"], observed=True)["heading_at_pass_deg"]
        .mean()
        .reset_index()
    )
    cell_rows: list[dict[str, object]] = []
    for (yielding, order), group in participant_cells.groupby(["yielding", "order"], observed=True):
        values = group["heading_at_pass_deg"].dropna()
        n = len(values)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(n))
        half = float(stats.t.ppf(0.975, n - 1) * se)
        cell_rows.append(
            {
                "AV behaviour": "Yielding" if yielding else "Non-yielding",
                "Relative order": order,
                "Mean_deg": mean,
                "CI_low": mean - half,
                "CI_high": mean + half,
            }
        )
    passage_table = pd.DataFrame(cell_rows).to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}",
    )
    logger.info(
        "Heading at participant passage "
        f"(participant-marginal degrees; 95% CI):\n{passage_table}"
    )

    passage = contrasts[contrasts["metric"].eq("heading_at_pass_deg")].set_index("contrast")
    for contrast_name, label in (
        ("yielding_minus_non_yielding", "Yielding minus non-yielding"),
        ("PF_AS_minus_AF_PS", "Participant-first minus avatar-first order"),
        ("conditional_eHMI_on_minus_off", "Conditional eHMI on minus off"),
    ):
        row = passage.loc[contrast_name]
        logger.info(
            f"{label}: {row['estimate']:.2f} deg, "
            f"95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}], "
            f"{_p_text(float(row['p_holm_within_metric']))}."
        )
    for yielding, label in ((0, "Non-yielding"), (1, "Yielding")):
        row = figure_tests[figure_tests["yielding"].eq(yielding)].iloc[0]
        logger.info(
            f"{label} passage-window order contrast (PF/AS minus AF/PS): "
            f"{row['estimate']:.2f} deg, "
            f"95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}], "
            f"{_p_text(float(row['p_holm_across_panels']))} across the two panels."
        )
        panel_pointwise = pointwise_tests[pointwise_tests["yielding"].eq(yielding)]
        significant_count = int(panel_pointwise["significant"].sum())
        logger.info(
            f"{label} event-aligned pointwise paired t-tests (PF/AS vs AF/PS): "
            f"{significant_count} of {len(panel_pointwise)} "
            f"{HEAD_HEADING_TIME_BIN_MS}-ms bins had raw "
            f"p<{float(panel_pointwise['p_threshold'].iloc[0]):.3f}; "
            f"marker ranges: {_significant_time_ranges(panel_pointwise)}."
        )
    spacing = distance[
        distance["metric"].eq("heading_at_pass_deg") & distance["test"].eq("distance")
    ].iloc[0]
    logger.info(
        f"Global spacing test for passage heading: "
        f"F({spacing['df1']:.0f},{spacing['df2']:.0f})={spacing['F']:.2f}, "
        f"p={spacing['p_raw']:.3f}, Holm p={spacing['p_holm_within_metric']:.3f}."
    )
    onset = contrasts[
        contrasts["metric"].eq("recovery_onset_20pct_rel_pass_s")
        & contrasts["contrast"].eq("yielding_minus_non_yielding")
    ].iloc[0]
    available = int(quality_map["onset_20pct_available"])
    logger.info(
        "Exploratory final-sustained 20% recovery onset, yielding minus "
        f"non-yielding: {onset['estimate']:.2f} s, "
        f"95% CI [{onset['ci_lower']:.2f}, {onset['ci_upper']:.2f}], "
        f"Holm p<.001; available in {available}/{len(features)} trials "
        f"({100.0 * available / len(features):.1f}%)."
    )
    stop_rows = event_contrasts[event_contrasts["metric"].eq("stopping_change_deg")].set_index("contrast")
    for contrast_name, label in (
        ("post_minus_pre", "Stop-aligned post-minus-pre change"),
        ("PF_AS_minus_AF_PS", "Order difference in stop-aligned change"),
        ("eHMI_on_minus_off", "eHMI difference in stop-aligned change"),
    ):
        row = stop_rows.loc[contrast_name]
        logger.info(
            f"{label}: {row['estimate']:.2f} deg, "
            f"95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}], "
            f"{_p_text(float(row['p_holm_12']))}."
        )
    logger.info(
        'Interpretation: horizontal HMD orientation only, not gaze or attention;' \
        'relative order remains a compound geometry/visibility factor.'
    )
    logger.info('=========================================================\n')
