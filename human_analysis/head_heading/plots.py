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
from OneEuroFilter import OneEuroFilter

import common
from custom_logger import CustomLogger
from .settings import *

logger = CustomLogger(__name__)
from .statistics import _significant_time_ranges

__all__ = [
    "smooth_heading_for_display",
    "create_event_aligned_plotly",
    "create_passage_summary_plotly",
    "_save_plotly_only",
    "_p_text",
]

def smooth_heading_for_display(values: pd.Series | np.ndarray) -> np.ndarray:
    """Smooth heading values using the configured One Euro filter for display only."""
    numeric = pd.Series(values, dtype=float).fillna(0.0).tolist()
    if not bool(common.get_configs("smoothen_signal")):
        return np.asarray(numeric, dtype=float)

    heading_filter = OneEuroFilter(
        freq=common.get_configs("freq"),
        mincutoff=common.get_configs("mincutoff"),
        beta=common.get_configs("beta"),
    )
    return np.asarray([heading_filter(value) for value in numeric], dtype=float)

def create_event_aligned_plotly(curves: pd.DataFrame, pointwise_tests: pd.DataFrame) -> Any:
    """Return the event-aligned figure using the project's Plotly stack."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    participant = (
        curves.groupby(["participant", "yielding", "order", "time_rel_pass_s"], observed=True)["heading_deg"]
        .mean()
        .reset_index()
    )
    summary = (
        participant.groupby(["yielding", "order", "time_rel_pass_s"], observed=True)["heading_deg"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    critical = stats.t.ppf(0.975, summary["count"] - 1)
    summary["lower"] = summary["mean"] - critical * summary["se"]
    summary["upper"] = summary["mean"] + critical * summary["se"]
    data_min = float(summary["lower"].min())
    data_max = float(summary["upper"].max())
    marker_y = data_min - HEAD_HEADING_TTEST_ROW_HEIGHT_DEG
    plot_min = data_min - max(4.0 * HEAD_HEADING_TTEST_ROW_HEIGHT_DEG, 2.0)
    plot_max = data_max + max(0.02 * (data_max - data_min), 1.0)
    labels = {"AF_PS": "Avatar first / participant second", "PF_AS": "Participant first / avatar second"}
    colors = {"AF_PS": "#3569b8", "PF_AS": "#d95032"}
    fills = {"AF_PS": "rgba(53,105,184,0.18)", "PF_AS": "rgba(217,80,50,0.18)"}
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Non-yielding", "Yielding"))
    fig.update_annotations(
        font=dict(
            family=common.get_configs("font_family"),
            size=common.get_configs("font_size")+8,
        )
    )
    for column, yielding in enumerate((0, 1), 1):
        for order in ("AF_PS", "PF_AS"):
            group = summary[(summary["yielding"] == yielding) & (summary["order"] == order)].sort_values(
                "time_rel_pass_s"
            ).copy()
            group["mean"] = smooth_heading_for_display(group["mean"])
            group["lower"] = smooth_heading_for_display(group["lower"])
            group["upper"] = smooth_heading_for_display(group["upper"])
            fig.add_trace(
                go.Scatter(
                    x=group["time_rel_pass_s"],
                    y=group["upper"],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            fig.add_trace(
                go.Scatter(
                    x=group["time_rel_pass_s"],
                    y=group["lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fills[order],
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            fig.add_trace(
                go.Scatter(
                    x=group["time_rel_pass_s"],
                    y=group["mean"],
                    mode="lines",
                    line=dict(color=colors[order], width=HEAD_HEADING_LINE_WIDTH),
                    name=labels[order],
                    legendgroup=order,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        fig.add_vline(x=0, line_dash="dash", line_color="#222222", row=1, col=column)
        fig.add_hline(y=0, line_dash="dot", line_color="#777777", row=1, col=column)
        fig.update_xaxes(
            title_text="Time relative to vehicle passing participant (s)",
            title_font=dict(
                family=common.get_configs("font_family"),
                size=common.get_configs("font_size")+8,
            ),
            tickfont=dict(
                family=common.get_configs("font_family"),
                size=common.get_configs("font_size")+8,
            ),
            range=[-5, 1],
            dtick=HEAD_HEADING_XAXIS_STEP,
            row=1,
            col=column,
        )
        significant = pointwise_tests[
            pointwise_tests["yielding"].eq(yielding)
            & pointwise_tests["significant"].astype(bool)
        ]
        fig.add_trace(
            go.Scatter(
                x=significant["time_rel_pass_s"],
                y=np.full(len(significant), marker_y),
                mode="text",
                text=["*"] * len(significant),
                name="__significance_markers__",
                textfont=dict(
                    family=common.get_configs("font_family"),
                    size=HEAD_HEADING_TTEST_MARKER_SIZE,
                    color="black",
                ),
                customdata=significant["p_raw"],
                hovertemplate="PF/AS vs AF/PS: time=%{x}, p=%{customdata:.4g}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    fig.update_yaxes(
        title_text="Baseline-corrected horizontal head heading (degrees)",
        title_font=dict(
            family=common.get_configs("font_family"),
            size=common.get_configs("font_size")+8,
        ),
        tickfont=dict(
            family=common.get_configs("font_family"),
            size=common.get_configs("font_size")+8,
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        range=[plot_min, plot_max],
        dtick=HEAD_HEADING_YAXIS_STEP_DEG,
        tickfont=dict(
            family=common.get_configs("font_family"),
            size=common.get_configs("font_size")+8,
        ),
    )
    fig.update_layout(
        template=common.get_configs("plotly_template"),
        font=dict(
            family=common.get_configs("font_family"),
            size=common.get_configs("font_size")+8,
        ),
        legend=dict(
            orientation="h",
            y=HEAD_HEADING_LEGEND_Y,
            x=HEAD_HEADING_LEGEND_X,
        ),
        margin=dict(l=120, r=2, t=75, b=60),
    )
    return fig

def create_passage_summary_plotly(features: pd.DataFrame) -> Any:
    """Return a manuscript-ready passage-heading point/interval figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    participant = (
        features.groupby(["participant", "yielding", "eHMIOn", "order"], observed=True)["heading_at_pass_deg"]
        .mean()
        .reset_index()
    )
    summary = (
        participant.groupby(["yielding", "eHMIOn", "order"], observed=True)["heading_at_pass_deg"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["half_ci"] = stats.t.ppf(0.975, summary["count"] - 1) * summary["se"]
    labels = {"AF_PS": "Avatar first / participant second", "PF_AS": "Participant first / avatar second"}
    colors = {0: "#3569b8", 1: "#d95032"}
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Non-yielding", "Yielding"))
    for column, yielding in enumerate((0, 1), 1):
        for ehmi in (0, 1):
            group = summary[(summary["yielding"] == yielding) & (summary["eHMIOn"] == ehmi)]
            group = group.set_index("order").loc[["AF_PS", "PF_AS"]].reset_index()
            fig.add_trace(
                go.Scatter(
                    x=[labels[x] for x in group["order"]],
                    y=group["mean"],
                    error_y=dict(type="data", array=group["half_ci"], visible=True, thickness=1.5, width=5),
                    mode="lines+markers",
                    marker=dict(size=9),
                    line=dict(color=colors[ehmi], width=2),
                    name="Conditional eHMI" if ehmi else "No eHMI",
                    legendgroup=f"eHMI{ehmi}",
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        fig.add_hline(y=0, line_dash="dot", line_color="#777777", row=1, col=column)
    fig.update_yaxes(title_text="Heading at participant passage (degrees)", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        title=dict(text="Horizontal head heading at participant passage", x=0.5),
        legend=dict(orientation="h", y=1.13, x=0),
        margin=dict(l=80, r=30, t=100, b=120),
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text="Participant-marginal means averaged over spacing; error bars are between-participant 95% CIs.",
                x=0.5,
                y=-0.24,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )
    return fig

def _save_plotly_only(
    fig: Any,
    name: str,
    width: int,
    height: int,
    open_browser: bool = True,
) -> None:
    """Save one figure with Plotly only to the configured output root."""
    import plotly.offline as plotly_offline

    figure_root = Path(common.get_configs("output"))
    figure_root.mkdir(parents=True, exist_ok=True)
    output_base = figure_root / name

    plotly_offline.plot(
        fig,
        filename=str(output_base.with_suffix(".html")),
        auto_open=bool(open_browser),
    )
    logger.info(f"Saved Plotly figure: {output_base.with_suffix('.html')}")

    for suffix, label in ((".eps", "EPS"), (".png", "PNG")):
        output_path = output_base.with_suffix(suffix)
        try:
            fig.write_image(str(output_path), width=width, height=height)
            logger.info(f"Saved Plotly figure: {output_path}")
        except Exception as exc:
            logger.warning(
                f"Skipping {label} export for '{name}' because Plotly/Kaleido "
                f"could not create the file: {exc}"
            )

def _p_text(value: float, label: str = "Holm p") -> str:
    """Format p values consistently for the human-readable results log."""
    return f"{label}<.001" if value < 0.001 else f"{label}={value:.3f}"
