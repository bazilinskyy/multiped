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


class RatingPlotMixin:
    """Focused method group extracted without changing calculation logic."""

    def plot_2x4_violins(self, responses_csv: str, mapping, name):
        """
        Create a 2x4 grid (8 subplots) of violin plots for all combinations
        of (yielding x eHMIOn x camera) defined in the mapping file.
        """

        # 1. Load data
        responses = pd.read_csv(responses_csv)
        # 2. Drop baselines and reshape to long format
        responses = responses.drop(columns=["baseline_1", "baseline_2"], errors="ignore")
        video_cols = [c for c in responses.columns if c.startswith("video_")]

        long_df = responses.melt(
            id_vars=["participant_id"],
            value_vars=video_cols,
            var_name="video_id",
            value_name="rating"
        )

        long_df["video_id"] = long_df["video_id"].astype(str)
        mapping["video_id"] = mapping["video_id"].astype(str)

        # 3. Merge with mapping
        mapping_cond = mapping[["video_id", "yielding", "eHMIOn", "camera"]].drop_duplicates()
        long_cond = long_df.merge(mapping_cond, on="video_id", how="left")

        long_cond["rating"] = pd.to_numeric(long_cond["rating"], errors="coerce")
        long_cond = long_cond.dropna(subset=["rating", "yielding", "eHMIOn", "camera"])

        # Keep Plotly's original full violin density, including its soft tails,
        # but provide only nonnegative y-axis tick positions.
        rating_max = float(long_cond["rating"].max()) if not long_cond.empty else 1.0
        rough_tick_step = max(rating_max, 1.0) / 5.0
        tick_magnitude = 10.0 ** math.floor(math.log10(rough_tick_step))
        normalized_step = rough_tick_step / tick_magnitude
        if normalized_step <= 1.0:
            nice_step = 1.0 * tick_magnitude
        elif normalized_step <= 2.0:
            nice_step = 2.0 * tick_magnitude
        elif normalized_step <= 5.0:
            nice_step = 5.0 * tick_magnitude
        else:
            nice_step = 10.0 * tick_magnitude
        tick_upper = math.ceil(max(rating_max, 0.0) / nice_step) * nice_step
        nonnegative_rating_ticks = np.arange(
            0.0,
            tick_upper + nice_step * 0.5,
            nice_step,
        )

        # 4. Unique condition combinations (should be 8)
        conds = (
            long_cond[["yielding", "eHMIOn", "camera"]]
            .drop_duplicates()
            .sort_values(["yielding", "eHMIOn", "camera"])
            .reset_index(drop=True)
        )

        max_plots = 8
        if len(conds) > max_plots:
            conds = conds.iloc[:max_plots]

        def camera_label(cam):
            return (
                "Avatar first / participant second"
                if cam == 0
                else "Participant first / avatar second"
            )

        # Two-line subplot title, single-line trace label
        def case_title(row):
            line1 = f"{'Yielding' if row['yielding'] == 1 else 'Non-yielding'}, {'eHMI' if row['eHMIOn'] == 1 else 'No eHMI'}"  # noqa: E501
            line2 = camera_label(int(row['camera']))
            return f"{line1}<br>{line2}"

        def case_name(row):
            # for hover / legend (one line)
            return (
                f"{'Yielding' if row['yielding'] == 1 else 'Non-yielding'}, "
                f"eHMI {'on' if row['eHMIOn'] == 1 else 'off'}, "
                f"{camera_label(int(row['camera']))}"
            )

        titles = [case_title(row) for _, row in conds.iterrows()]
        names = [case_name(row) for _, row in conds.iterrows()]

        # 5. Create 2x4 subplot figure and add violins
        fig = make_subplots(
            rows=2,
            cols=4,
            subplot_titles=titles
        )

        for i, cond_row in conds.iterrows():
            sub = long_cond[
                (long_cond["yielding"] == cond_row["yielding"]) &
                (long_cond["eHMIOn"] == cond_row["eHMIOn"]) &
                (long_cond["camera"] == cond_row["camera"])
            ]

            r = i // 4 + 1   # type: ignore # row 1–2
            c = i % 4 + 1    # type: ignore # col 1–4

            fig.add_trace(
                go.Violin(
                    y=sub["rating"],
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    name=names[i],   # one-line label for hover # type: ignore
                    showlegend=False
                ),
                row=r,
                col=c
            )

        # 6. Layout
        fig.update_layout(
            height=900,
            width=1600,
            title_text="",
            template=plotly_template
        )

        # Preserve the original autorange and full violin shape. Only the tick
        # positions are restricted, so negative density tails remain visible.
        for r in range(1, 3):
            for c in range(1, 5):
                fig.update_yaxes(
                    tickmode="array",
                    tickvals=nonnegative_rating_ticks,
                    row=r,
                    col=c,
                )
            fig.update_yaxes(title_text="Rating", row=r, col=1)

        # Hide x tick labels (titles already describe conditions)
        for r in range(1, 3):
            for c in range(1, 5):
                fig.update_xaxes(showticklabels=False, row=r, col=c)

        self.save_plotly(
            fig, name, save_final=True
        )
