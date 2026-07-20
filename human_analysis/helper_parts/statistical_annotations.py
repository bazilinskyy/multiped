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


class StatisticalAnnotationMixin:
    """Focused plotting responsibility extracted from the legacy helper."""

    def ttest(self, signal_1, signal_2, type='two-sided', paired=True):
        """
        Perform a t-test on two signals, computing p-values and significance.

        Args:
            signal_1 (list): First signal, a list of numeric values.
            signal_2 (list): Second signal, a list of numeric values.
            type (str, optional): Type of t-test to perform. Options are "two-sided",
                                  "greater", or "less". Defaults to "two-sided".
            paired (bool, optional): Indicates whether to perform a paired t-test
                                     (ttest_rel) or an independent t-test (ttest_ind).
                                     Defaults to True (paired).

        Returns:
            list: A list containing two elements:
                  - p_values (list): Raw p-values for each bin.
                  - significance (list): Binary flags (0 or 1) indicating whether
                    the p-value for each bin is below the threshold configured in
                    tr.common.get_configs('p_value').
        """
        # Check if the lengths of the two signals are the same
        if len(signal_1) != len(signal_2):
            logger.error('The lengths of signal_1 and signal_2 must be the same.')
            return -1

        p_values = []
        significance = []
        threshold = common.get_configs("p_value")

        for i in range(len(signal_1)):
            data1 = signal_1[i]
            data2 = signal_2[i]

            # Skip if data is empty
            if not data1 or not data2 or (paired and len(data1) != len(data2)):
                p_values.append(1.0)
                significance.append(0)
                continue

            try:
                if paired:
                    t_stat, p_val = ttest_rel(data1, data2, alternative=type)
                else:
                    t_stat, p_val = ttest_ind(data1, data2, equal_var=False, alternative=type)

                # Handles the nan cases
                if np.isnan(p_val):  # type: ignore
                    p_val = 1.0
            except Exception as e:
                logger.warning(f"Skipping t-test at time index {i} due to error: {e}")
                p_val = 1.0

            p_values.append(p_val)
            significance.append(int(p_val < threshold))

        return [p_values, significance]

    def draw_ttest(self, fig, times, name_file, yaxis_range, yaxis_step, ttest_signals,
                   ttest_marker_size, ttest_marker_colour, ttest_annotations_font_size,
                   ttest_annotations_colour, ttest_row_height, ttest_annotation_x,
                   flag_trigger=False, reuse_statistical_csv=False):
        """Draw the pointwise paired t test row.

        Args:
            fig (figure): figure object.
            name_file (str): name of file to save.
            yaxis_range (list): range of y axis in format [min, max] for the keypress plot.
            yaxis_step (int): step between ticks on y axis.
            ttest_signals (list): signals to compare with ttest. None = do not compare.
            ttest_marker_size (int): size of markers for the ttest.
            ttest_marker_colour (str): colour of markers for the ttest.
            ttest_annotations_font_size (int): font size of annotations for ttest.
            ttest_annotations_colour (str): colour of annotations for ttest.
            ttest_row_height (float): height of the t test marker row in y units.
        """
        # Save original axis limits (bottom/top of the main data area)
        original_min, original_max = yaxis_range
        # Counters for marker rows
        counter_ttest = 0
        counter_anova = 0

        # calculate resolution based on the param
        if flag_trigger:
            resolution = common.get_configs("kp_resolution") / 1000.0
        else:
            resolution = common.get_configs("yaw_resolution") / 1000.0

        # --- t-test markers ---
        if ttest_signals:
            for comp in ttest_signals:
                # Save csv. Keep nested plot output folders intact when name_file
                # contains a subdirectory, for example:
                #   kp_threshold_sensitivity/threshold_05pct/all_videos_...
                # The statistics file is then saved as:
                #   _output/statistics/kp_threshold_sensitivity/threshold_05pct/video_2_all_videos_....csv
                # instead of accidentally creating a directory called
                #   video_2_kp_threshold_sensitivity/...
                times_csv = [round(i * resolution, 2) for i in range(len(comp['signal_1']))]
                name_dir = os.path.dirname(name_file)
                name_base = os.path.basename(name_file)
                stats_name_file = f"{comp['label']}_{name_base}.csv"
                if name_dir:
                    stats_name_file = os.path.join(name_dir, stats_name_file)
                stats_path = os.path.join(
                    common.get_configs("output"),
                    self.folder_stats,
                    stats_name_file,
                )

                p_vals = None
                sig = None
                if reuse_statistical_csv and os.path.isfile(stats_path):
                    cached_stats = pd.read_csv(stats_path)
                    if (
                        "p-value" in cached_stats.columns
                        and len(cached_stats) == len(comp["signal_1"])
                    ):
                        p_vals = (
                            pd.to_numeric(cached_stats["p-value"], errors="coerce")
                            .fillna(1.0)
                            .tolist()
                        )
                        threshold = common.get_configs("p_value")
                        sig = [int(value < threshold) for value in p_vals]
                        logger.info(f"Reused cached statistical test CSV: {stats_path}")

                if p_vals is None or sig is None:
                    p_vals, sig = self.ttest(
                        signal_1=comp['signal_1'],
                        signal_2=comp['signal_2'],
                        paired=comp['paired'],
                    )  # type: ignore
                    self.save_stats_csv(
                        t=times_csv,
                        p_values=p_vals,
                        name_file=stats_name_file,
                    )
                    self.statistical_cache_changed = True

                if any(sig):
                    # Place this row below the curves, one row further down per comparison
                    # (same logic for kp/yaw; ttest_row_height is in the same units as y)
                    y_offset = original_min - ttest_row_height * (counter_ttest + 1)

                    significant_indices = [
                        index for index, is_significant in enumerate(sig)
                        if is_significant
                    ]
                    xs = [times[index] for index in significant_indices]
                    significant_p_values = [
                        p_vals[index] for index in significant_indices
                    ]

                    # One vectorised text trace is substantially faster than
                    # adding a separate Plotly annotation for every time bin.
                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=[y_offset] * len(xs),
                        mode="text",
                        text=["*"] * len(xs),
                        name="__significance_markers__",
                        textfont=dict(
                            family=common.get_configs("font_family"),
                            size=ttest_marker_size,
                            color=ttest_marker_colour,
                        ),
                        customdata=significant_p_values,
                        hovertemplate=(
                            f"{comp['label']}: time=%{{x}}, "
                            "p=%{customdata:.4g}<extra></extra>"
                        ),
                        showlegend=False,
                    ))

                    # label row
                    fig.add_annotation(x=ttest_annotation_x,
                                       y=y_offset,
                                       text=comp['label'],
                                       xanchor='right',
                                       showarrow=False,
                                       font=dict(family=common.get_configs("font_family"),
                                                 size=ttest_annotations_font_size,
                                                 color=ttest_annotations_colour))
                    counter_ttest += 1

        # TODO: ANOVA support is currently broken in original code; left untouched other than counting.
        # If you later add ANOVA rows, increment `counter_anova` similarly and compute their y_offset.

        # --- Adjust axis to include marker rows ---
        if counter_ttest or counter_anova:
            n_rows = max(counter_ttest, counter_anova)
            # Extend the axis downward enough to include all rows, plus one extra row of padding
            min_y = original_min - ttest_row_height * (n_rows + 1)

            fig.update_layout(yaxis=dict(
                range=[min_y, original_max],
                dtick=yaxis_step,
                tickformat='.2f'
            ))

    def save_stats_csv(self, t, p_values, name_file):
        """Save results of statistical test in csv.

        Args:
            t (list): list of time slices.
            p_values (list): list of p values.
            name_file (str): name of file. This may include a relative
                subdirectory, for example
                ``kp_threshold_sensitivity/threshold_05pct/file.csv``.
        """
        path = os.path.join(common.get_configs("output"), self.folder_stats)  # where to save csv
        df = pd.DataFrame(columns=['t', 'p-value'])  # dataframe to save to csv
        df['t'] = t
        df['p-value'] = p_values

        out_path = os.path.join(path, name_file)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        df.to_csv(out_path, index=False)
        logger.info(f"Saved statistical test CSV: {out_path}")

    @staticmethod
    def draw_events(fig, yaxis_range, events, events_width, events_dash, events_colour,
                    events_annotations_font_size, events_annotations_colour):
        """Draw vertical lines and text labels for events (no arrows), with grouping by 'id'.

        - Events with the same 'id' share a horizontal row near the top of the plot.
        - Row with id == 1 (e.g. 'Car decelerates', 'Car stops', 'Car accelerates')
          is placed very close to the top.
        - Labels are horizontally centered on their own vertical line (x = start).
        """

        if not events:
            return

        y_min, y_max = yaxis_range
        height = max(y_max - y_min, 1e-6)  # avoid zero height

        # Group events by 'id'. Events without an id get their own group.
        groups = {}
        for idx, ev in enumerate(events):
            key = ev.get("id")
            if key is None:
                key = f"_noid_{idx}"
            groups.setdefault(key, []).append(idx)

        # Base spacing between row bands (for ids other than 1)
        row_height_frac = 0.10    # fraction of plot height between rows
        base_offset_frac = 0.05   # offset below top of plot for non-id-1 rows

        # Iterate bands in insertion order (id=1 first, then id=2, etc.)
        for row_index, (group_key, idx_list) in enumerate(groups.items()):
            if not idx_list:
                continue

            # Fractional vertical position for this row
            if str(group_key) == "1":
                # put id=1 row very close to the top
                frac = 0.01
            else:
                frac = base_offset_frac + row_index * row_height_frac

            # keep inside the plot
            frac = min(max(frac, 0.0), 0.95)

            label_y = y_max - frac * height

            for event_index in idx_list:
                ev = events[event_index]
                start = float(ev["start"])
                end = float(ev["end"])
                label = ev.get("annotation", "")

                # Center label horizontally on its own line
                label_x = 0.5 * (start + end) if start != end else start

                # --- Vertical line(s) ---
                fig.add_shape(
                    type="line",
                    x0=start,
                    y0=y_min,
                    x1=start,
                    y1=y_max,
                    line=dict(
                        color=events_colour,
                        dash=events_dash,
                        width=events_width,
                    ),
                )

                if start != end:
                    fig.add_shape(
                        type="line",
                        x0=end,
                        y0=y_min,
                        x1=end,
                        y1=y_max,
                        line=dict(
                            color=events_colour,
                            dash=events_dash,
                            width=events_width,
                        ),
                    )

                # --- Text label ---
                fig.add_annotation(
                    text=label,
                    x=label_x,
                    y=label_y,
                    xanchor="center",
                    yanchor="bottom",
                    showarrow=False,
                    font=dict(
                        family=common.get_configs("font_family"),
                        size=int(events_annotations_font_size * 3.3),
                        color=events_annotations_colour,
                    ),
                )
