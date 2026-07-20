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


class HeadingPlotMixin:
    """Focused method group extracted without changing calculation logic."""

    def plot_yaw(self, mapping, column_name="Yaw", parameter=None, parameter_value=None,
                 additional_parameter=None, additional_parameter_value=None, compare_trial="video_1",
                 xaxis_title=None, xaxis_range=None, yaxis_range=None,
                 margin=None, name=None, recompute=False):
        """
        Generate a comparison plot of horizontal Unity head heading data and
        subjective slider ratings
        for multiple video trials relative to a test condition.

        The function processes trigger matrices for each participant and trial,
        aligns time series data, attaches subjective slider-based ratings (annoyance,
        informativeness, noticeability), and prepares the data for visualization.
        Significance testing (paired t-tests) is performed between the test condition
        and each other trial.

        Args:
            mapping (pd.DataFrame): DataFrame with video metadata, including
                'video_id', 'sound_clip_name', 'display_name', and 'colour'.
            column_name (str, optional): The matrix column to process (default "Yaw").
            parameter / parameter_value (optional): Filter `mapping` by column == value.
            additional_parameter / additional_parameter_value (optional): Second filter.
            compare_trial (str, optional): Reference trial video_id.
            xaxis_title (str, optional): Custom label for the x-axis.
            xaxis_range (list, optional): x-axis [min, max] limits for the plot.
            yaxis_range (list, optional): y-axis [min, max] limits for the plot.
            margin (dict, optional): Custom plot margin dictionary.
            name (str, optional): (currently unused).
            recompute (bool, optional): If True, regenerate CSVs/TXT even if they exist.
                                        If False, reuse existing files when present.
        """
        # ensure yaxis_range is mutable (plot_kp modifies it in-place)
        if isinstance(yaxis_range, tuple):
            yaxis_range = list(yaxis_range)

        # Find the video_length for the given video_id
        lens = mapping.loc[mapping["video_id"].eq(compare_trial), "video_length"].unique()

        if len(lens) == 0:
            raise ValueError(f"No rows found for video_id='{compare_trial}'")
        elif len(lens) > 1:
            # If the same video_id appears with different lengths, keep all matching lengths
            mapping_filtered = mapping[mapping["video_length"].isin(lens)].copy()
        else:
            # Typical case: one length
            mapping_filtered = mapping[mapping["video_length"].eq(lens[0])].copy()

        if parameter is not None:
            mapping_filtered = mapping_filtered[mapping_filtered[parameter] == parameter_value]

        if additional_parameter is not None:
            mapping_filtered = mapping_filtered[mapping_filtered[additional_parameter] == additional_parameter_value]

        # Filter out control/test video IDs for comparison
        mapping_filtered = mapping_filtered[~mapping_filtered["video_id"].isin(["baseline_1", "baseline_2"])]
        plot_videos = mapping_filtered["video_id"]

        all_dfs = []          # List to collect DataFrames for each trial
        all_labels = []       # Corresponding list of human-friendly trial labels
        ttest_signals = []    # Store t-test pairs for stats annotations

        data_folder = common.get_configs("data")  # Get path to raw data

        # Export HMD quaternions and compute horizontal Unity heading per timestamp.
        test_participant_csv = os.path.join(
            self.output_folder,
            f"participant_{column_name}_{compare_trial}.csv"
        )

        # Export participant quaternion matrix for reference trial (only if needed)
        if recompute or not os.path.exists(test_participant_csv):
            self.export_participant_quaternion_matrix(
                data_folder=self.data_folder,   # keep original behaviour for reference
                video_id=compare_trial,
                output_file=test_participant_csv,
                mapping=mapping
            )

        # Compute average horizontal heading for the reference trial and save.
        test_yaw_csv = os.path.join(
            self.output_folder,
            f"yaw_avg_{compare_trial}.csv"     # IMPORTANT: separate file from participant_*.csv
        )

        if recompute or not os.path.exists(test_yaw_csv):
            HMD_class.compute_avg_yaw_from_matrix_csv(
                input_csv=test_participant_csv,
                output_csv=test_yaw_csv,
                force=recompute,
            )

        def heading_bins(video_id, participant_csv):
            """Use processed per-bin headings before converting quaternions."""
            if self.processed_data_cache is not None:
                cached_bins = self.processed_data_cache.get("head_heading_bins", {})
                if str(video_id) in cached_bins:
                    return cached_bins[str(video_id)]
            return extra_class.all_yaws_per_bin(input_csv=participant_csv)

        # Matrix for t-tests: must use participant-level per-bin headings, not
        # only the averaged yaw CSV.
        test_matrix = heading_bins(compare_trial, test_participant_csv)

        # === Iterate through each video trial (excluding control/test) ===
        for video in plot_videos:
            # Get display name for current trial
            display_name = mapping.loc[mapping["video_id"] == video, "video_id"].values[0]
            participant_csv = os.path.join(
                self.output_folder,
                f"participant_{column_name}_{video}.csv"
            )

            # Export quaternion/yaw matrix for this trial (if needed)
            if recompute or not os.path.exists(participant_csv):
                self.export_participant_quaternion_matrix(
                    data_folder=data_folder,
                    video_id=video,
                    output_file=participant_csv,
                    mapping=mapping
                )

            # Compute avg yaw for this trial (if needed)
            yaw_csv = os.path.join(self.output_folder, f"yaw_avg_{video}.csv")
            if recompute or not os.path.exists(yaw_csv):
                HMD_class.compute_avg_yaw_from_matrix_csv(
                    input_csv=participant_csv,
                    output_csv=yaw_csv,
                    force=recompute,
                )

            df = pd.read_csv(yaw_csv)
            all_dfs.append(df)
            all_labels.append(display_name)

            # Extract all per-bin yaw values (for saving and t-test)
            trial_matrix = heading_bins(video, participant_csv)

            yaw_values = extra_class.flatten_trial_matrix(trial_matrix)
            yaw_values = yaw_values[~np.isnan(yaw_values)]  # Remove NaNs if present

            trial_txt_path = os.path.join(self.output_folder, f"yaw_values_{video}.txt")
            if recompute or not os.path.exists(trial_txt_path):
                np.savetxt(trial_txt_path, yaw_values)

            # Prepare for t-test: compare each trial vs. test reference (exclude self-comparison)
            if video != compare_trial:
                ttest_signals.append({
                    "signal_1": test_matrix,
                    "signal_2": trial_matrix,
                    "paired": True,
                    "label": f"{display_name}"
                })

        # === Combine all trial DataFrames into a single one for plotting ===
        if not all_dfs:
            raise RuntimeError("No data frames found to plot.")

        combined_df = pd.DataFrame()
        combined_df["Timestamp"] = all_dfs[0]["Timestamp"]

        # Add trial average yaw series as columns
        for df, label in zip(all_dfs, all_labels):
            combined_df[label] = df["AvgYaw"]

        # === Helper for event times (ignore ±0.02 s by rounding + mode) ===
        def _get_mode_time(df, col, round_decimals=2):
            """Return the mode of a time column, ignoring NaNs and
            small numeric differences by rounding first."""
            if col not in df.columns:
                return None

            series = df[col].dropna()
            if series.empty:
                return None

            rounded = series.round(round_decimals)
            mode_vals = rounded.mode()
            if mode_vals.empty:
                return None

            return float(mode_vals.iloc[0])

        # === Build events from mapping_filtered timing columns ===
        events = []

        # Row group 1: all main car events at the same (top) height (id=1)
        first_row_events = [
            ("yield_start_time_s",   "Car decelerates"),
            ("yield_stop_time_s",    "Car stops"),
            ("yield_resume_time_s",  "Car accelerates"),
        ]
        for col_name, label in first_row_events:
            t = _get_mode_time(mapping_filtered, col_name)
            if t is not None and not np.isnan(t):
                events.append({
                    "id": 1,
                    "start": t,
                    "end": t,
                    "annotation": label
                })

        # Row group 2: crossing event on its own lower row (id=2)
        second_row_events = [
            ("cross_p2_time_s", "Car crosses the 1st pedestrian"),
        ]
        for col_name, label in second_row_events:
            t = _get_mode_time(mapping_filtered, col_name)
            if t is not None and not np.isnan(t):
                events.append({
                    "id": 2,
                    "start": t,
                    "end": t,
                    "annotation": label
                })

        has_top_row = any(ev.get("id") == 1 for ev in events)
        if not has_top_row:
            for ev in events:
                if ev.get("id") is not None:
                    ev["id"] = 1

        # === cross_p1_time_s: per-line time (one marker per plotted line) ===
        cross_p1_times = {}
        if "cross_p1_time_s" in mapping_filtered.columns:
            for video, label in zip(plot_videos, all_labels):
                series = mapping_filtered.loc[
                    mapping_filtered["video_id"] == video, "cross_p1_time_s"
                ].dropna()
                if not series.empty:
                    # take first non-NaN for that video/line
                    cross_p1_times[label] = float(series.iloc[0])

        # Choose line style: dashed for test trial, solid for others
        custom_line_dashes = []
        for label in all_labels:
            vid = mapping.loc[mapping["video_id"] == label, "video_id"].values[0]
            if vid == compare_trial:
                custom_line_dashes.append("dot")
            else:
                custom_line_dashes.append("solid")

        # === Call central plotting function with all visualization & stats options ===
        self.plot_kp(
            df=combined_df,
            y=all_labels,
            y_legend_kp=all_labels,
            xaxis_range=xaxis_range,
            yaxis_range=yaxis_range,
            xaxis_title=xaxis_title,  # type: ignore
            yaxis_title="Horizontal head heading, [radians]",
            xaxis_title_offset=-0.047,  # type: ignore
            name_file=f"{name}",
            show_text_labels=True,
            pretty_text=True,
            events=events,
            events_width=2,
            events_annotations_font_size=common.get_configs("font_size") - 8,
            stacked=False,
            ttest_signals=ttest_signals,
            ttest_row_height=0.006,
            ttest_annotations_font_size=common.get_configs("font_size") - 8,
            ttest_annotation_x=0.001,  # type: ignore
            ttest_marker_size=common.get_configs("font_size") - 6,
            xaxis_step=1,
            yaxis_step=0.20,  # type: ignore
            legend_x=0,
            legend_y=1.225,
            legend_columns=2,
            line_width=3,
            fig_save_width=1470,
            fig_save_height=850,
            font_size=common.get_configs("font_size"),
            save_file=True,
            save_final=True,
            custom_line_dashes=custom_line_dashes,
            flag_trigger=False,
            margin=margin,
            cross_p1_times=cross_p1_times,
            reuse_statistical_csv=self.reuse_statistical_results,
        )

    def plot_yaw_frequencies_by_condition(self, mapping, yaw_files_dir):
        yaw_files_dir = os.path.abspath(yaw_files_dir)

        # Remove baseline videos and copy to avoid SettingWithCopyWarning
        mapping = mapping.loc[~mapping["video_id"].isin(["baseline_1", "baseline_2"])].copy()

        # Keys for grouping
        # case_key: (yielding, eHMIOn)  -> for colour mapping

        mapping["case_key"] = (
            "y" + mapping["yielding"].astype(str) +
            "_e" + mapping["eHMIOn"].astype(str)
        )
        # condition_key: (yielding, eHMIOn, camera)  -> for the actual curves
        mapping["condition_key"] = mapping["case_key"] + "_c" + mapping["camera"].astype(str)
        # Distances (for the 10 subplots)
        mapping["distPed_m"] = self._distance_series_to_meters(mapping["distPed"])
        dist_values = sorted(mapping["distPed_m"].dropna().unique())

        if len(dist_values) != 5:
            logger.warning(
                f"Expected 5 distinct inter-pedestrian distances in metres, found {len(dist_values)}: {dist_values}"
            )
        # G10 colours mapped by case_key so the same case shares colour across all figures
        colors = px.colors.qualitative.G10

        case_keys = sorted(mapping["case_key"].unique())
        color_map = {ck: colors[i % len(colors)] for i, ck in enumerate(case_keys)}
        line_width = 6

        # Histogram settings (no smoothing) in [-90, 90]
        bins = np.linspace(-90, 90, 181)   # 1° bins from -90 to 90

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # lists to collect curve metrics
        metrics_cam0 = []

        metrics_cam1 = []
        metrics_grid = []   # metrics for the 10 (camera, distPed) plots

        def _compute_freq_for_subset(sub_subset, cam_value, metrics_list=None, dist_value=None):
            """

            Given a subset of mapping (already filtered by camera and possibly distPed),
            compute frequencies per (yielding, eHMIOn) case, and optionally store metrics.
            Returns
            -------

            curves : list of (case_key, cond_label, color, freq_array)
            local_max : float
                Maximum frequency value among curves for this subset.
            """
            curves = []
            local_max = 0.0
            # Group by (yielding, eHMIOn, camera) via condition_key
            for cond_key, sub in sub_subset.groupby("condition_key", sort=True):

                case_key = sub["case_key"].iloc[0]
                color = color_map[case_key]
                # --- Legend label based ONLY on yielding & eHMI ---
                y_val = int(sub["yielding"].iloc[0])

                e_val = int(sub["eHMIOn"].iloc[0])
                yielding_label = "Non-yielding" if y_val == 0 else "Yielding"
                ehmi_label = "No eHMI" if e_val == 0 else "eHMI"

                cond_label = f"{yielding_label}, {ehmi_label}"

                all_yaws_deg = []

                for video_id in sub["video_id"].unique():
                    yaw_path = os.path.join(yaw_files_dir, f"yaw_values_{video_id}.txt")

                    if not os.path.exists(yaw_path):
                        logger.warning(f"Warning: {yaw_path} not found, skipping this video.")
                        continue
                    yaws_rad = np.loadtxt(yaw_path)
                    yaws_rad = np.atleast_1d(yaws_rad)

                    # radians → degrees, wrap to [-180, 180]
                    yaws_deg = np.degrees(yaws_rad)

                    yaws_deg = (yaws_deg + 180) % 360 - 180
                    # keep only [-90, 90] observations
                    mask = (yaws_deg >= -90) & (yaws_deg <= 90)

                    yaws_deg = yaws_deg[mask]
                    if yaws_deg.size == 0:
                        continue

                    all_yaws_deg.append(yaws_deg)

                if not all_yaws_deg:
                    logger.error(f"Warning: no yaw samples for condition '{cond_label}'.")

                    continue
                all_yaws_deg = np.concatenate(all_yaws_deg)

                counts, _ = np.histogram(all_yaws_deg, bins=bins)
                if counts.sum() == 0:

                    continue
                # Frequency (%) – NO smoothing
                freq = counts / counts.sum() * 100.0

                local_max = max(local_max, freq.max())
                # Metrics (if a metrics_list is provided)
                if metrics_list is not None:

                    area_total = freq.sum()  # should be ~100
                    left_mask = bin_centers < 0
                    right_mask = bin_centers > 0

                    central_mask = np.abs(bin_centers) <= 15
                    area_left = freq[left_mask].sum()
                    area_right = freq[right_mask].sum()

                    area_central = freq[central_mask].sum()
                    peak_idx = np.argmax(freq)
                    peak_yaw = bin_centers[peak_idx]

                    peak_freq = freq[peak_idx]
                    mean_yaw = np.sum(bin_centers * freq) / area_total
                    var_yaw = np.sum(((bin_centers - mean_yaw) ** 2) * freq) / area_total

                    std_yaw = np.sqrt(var_yaw)
                    metrics_list.append(
                        {
                            "camera": cam_value,
                            "distPed_m": dist_value,
                            "condition": cond_label,
                            "case_key": case_key,
                            "area_total_pct": area_total,
                            "area_left_pct": area_left,
                            "area_right_pct": area_right,
                            "area_central_±15_pct": area_central,
                            "peak_yaw_deg": peak_yaw,
                            "peak_freq_pct": peak_freq,
                            "mean_yaw_deg": mean_yaw,
                            "std_yaw_deg": std_yaw,
                        }
                    )
                curves.append((case_key, cond_label, color, freq))

            return curves, local_max

        def build_figure_for_camera(cam_value, metrics_list):
            sub_cam = mapping[mapping["camera"] == cam_value]

            if sub_cam.empty:
                return None, 0.0
            fig = go.Figure()
            curves, max_y = _compute_freq_for_subset(

                sub_cam,
                cam_value=cam_value,
                metrics_list=metrics_list,
                dist_value=None,  # aggregated across distances
            )
            for case_key, cond_label, color, freq in curves:
                fig.add_trace(

                    go.Scatter(
                        x=bin_centers,
                        y=freq,
                        mode="lines",
                        name=cond_label,
                        line=dict(width=line_width, color=color, dash="solid"),
                    )
                )
            return fig, max_y

        # === 1) Build the two main camera-level figures (camera 0, camera 1) ===
        fig_cam0, max0 = build_figure_for_camera(0, metrics_cam0)

        fig_cam1, max1 = build_figure_for_camera(1, metrics_cam1)
        if fig_cam0 is None and fig_cam1 is None:
            raise ValueError("No yaw samples found for any camera.")

        global_max = max(max0, max1)
        if global_max <= 0:

            raise ValueError("Non-positive maximum frequency encountered.")
        # Common y-axis range & ticks for main figs: 1, 2, 3, ...
        ymax_axis = math.ceil(global_max)

        if ymax_axis < 1:
            ymax_axis = 1  # at least one tick
        # Log metrics as tables for your report (camera-level)
        metrics_cam0_df = pd.DataFrame(metrics_cam0)

        metrics_cam1_df = pd.DataFrame(metrics_cam1)
        if not metrics_cam0_df.empty:
            logger.info(f"Camera 0 metrics:\n{metrics_cam0_df.to_string(index=False)}")
        if not metrics_cam1_df.empty:
            logger.info(f"Camera 1 metrics:\n{metrics_cam1_df.to_string(index=False)}")

        def finalize_figure(fig):
            if fig is None:

                return None
            # Vertical reference line at 0° (thin, grey, dotted)
            fig.add_shape(

                type="line",
                x0=0, x1=0,
                y0=0, y1=ymax_axis,
                line=dict(color="gray", dash="dot", width=3.0),
            )
            # White background, grids, solid axes, legend inside
            fig.update_layout(
                template="none",
                paper_bgcolor="white",
                plot_bgcolor="white",
                title="",
                xaxis_title="</b>Horizontal head heading (deg)</b>",
                yaxis_title="</b>Frequency</b>",
                legend_title=None,
                legend=dict(
                    x=0.02,
                    y=0.98,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    font=dict(
                        family=font_family, size=font_size+20
                    ),
                ),
                xaxis_title_font=dict(family=font_family, size=font_size+20),
                yaxis_title_font=dict(family=font_family, size=font_size+20),
            )
            fig.update_xaxes(
                range=[-90, 90],

                showgrid=True,
                gridcolor="lightgray",
                tickmode="array",
                tickvals=[-90, -60, -30, 0, 30, 60, 90],  # every 30°, excluding 0
                zeroline=False,                        # no built-in zero line
                showline=True,
                linecolor="black",
                tickfont=dict(family=font_family, size=font_size+20),
            )
            fig.update_yaxes(
                range=[0, ymax_axis],
                tick0=1,
                dtick=1,                               # ticks & grid at 1,2,3,...
                showgrid=True,
                gridcolor="lightgray",
                zeroline=True,
                zerolinecolor="black",
                showline=True,
                linecolor="black",
                tickfont=dict(family=font_family, size=font_size+8),
            )
            return fig

        fig_cam0 = finalize_figure(fig_cam0)
        fig_cam1 = finalize_figure(fig_cam1)

        # Save main camera-level figs
        if fig_cam0 is not None:
            self.save_plotly(
                fig_cam0,
                "yaw_hist_can_see",
                width=1600,
                height=900,
                save_final=True,
            )

        if fig_cam1 is not None:
            self.save_plotly(
                fig_cam1,
                "yaw_hist_cannot_see",
                width=1600,
                height=900,
                save_final=True,
            )

        # === 2) Build the additional 10 plots: camera × distPed in a 2×5 grid ===

        # Subplot titles: only show distance on top row, blank on bottom row
        subplot_titles = []
        for r in range(2):
            for dist in dist_values[:5]:
                if r == 0:
                    subplot_titles.append(f"Distance = {int(dist)} m")
                else:
                    subplot_titles.append("")  # no title in second row

        fig_grid = make_subplots(
            rows=2,
            cols=5,
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.03,
            vertical_spacing=0.10,
            subplot_titles=subplot_titles,
        )

        # For grid: fix y_max (you can use 8 if you want stricter cap)
        grid_ymax = 9

        for r, cam in enumerate([0, 1], start=1):
            for c, dist in enumerate(dist_values[:5], start=1):
                sub_camdist = mapping[
                    (mapping["camera"] == cam) &
                    (mapping["distPed_m"] == dist)
                ]
                if sub_camdist.empty:
                    continue

                curves, local_max = _compute_freq_for_subset(
                    sub_camdist,
                    cam_value=cam,
                    metrics_list=metrics_grid,   # collect metrics for grid as well
                    dist_value=dist,
                )

                for case_key, cond_label, color, freq in curves:
                    # Only show legend entry once (top-left subplot),
                    # but link ALL traces of the same case via legendgroup
                    show_legend = (r == 1 and c == 1)

                    fig_grid.add_trace(
                        go.Scatter(
                            x=bin_centers,
                            y=freq,
                            mode="lines",
                            name=cond_label,
                            legendgroup=case_key,
                            line=dict(width=line_width-3, color=color, dash="solid"),
                            showlegend=show_legend,
                        ),
                        row=r,
                        col=c,
                    )

                # Vertical line at 0° for this subplot
                fig_grid.add_vline(
                    x=0,
                    line_dash="dot",
                    line_color="gray",
                    line_width=1.5,
                    row=r,  # type: ignore
                    col=c,  # type: ignore
                )

        # Log metrics for the 10 camera×distPed plots
        metrics_grid_df = pd.DataFrame(metrics_grid)
        if not metrics_grid_df.empty:
            logger.info(f"Grid (camera × distance in metres) metrics:\n{metrics_grid_df.to_string(index=False)}")

        # Common layout for the grid figure
        fig_grid.update_layout(
            template="none",
            paper_bgcolor="white",
            plot_bgcolor="white",
            title="",
            legend_title=None,
            legend=dict(
                x=0.87,
                y=0.99,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(family=font_family, size=font_size),
            ),
        )

        # Bigger tick labels for grid
        fig_grid.update_xaxes(
            range=[-90, 90],
            showgrid=True,
            gridcolor="lightgray",
            tickmode="array",
            tickvals=[-90, -60, -30, 0, 30, 60, 90],
            zeroline=False,
            showline=True,
            linecolor="black",
            tickfont=dict(family=font_family, size=font_size),
        )
        fig_grid.update_yaxes(
            range=[0, grid_ymax],
            tick0=1,
            dtick=1,
            showgrid=True,
            gridcolor="lightgray",
            zeroline=True,
            zerolinecolor="black",
            showline=True,
            linecolor="black",
            tickfont=dict(family=font_family, size=font_size),
        )

        # Axis labels for the grid
        fig_grid.update_yaxes(title_text="Frequency", row=1, col=1)
        fig_grid.update_yaxes(title_text="Frequency", row=2, col=1)
        fig_grid.update_xaxes(title_text="Horizontal head heading (deg)", row=2, col=3)

        # Row labels on extreme left: "Can see the person" / "Cannot see the person"
        # Use y-axis domains of first column in each row to place the labels nicely
        # try:
        #     dom_row1 = fig_grid.layout.yaxis.domain
        #     dom_row2 = fig_grid.layout.yaxis6.domain  # first yaxis in second row
        #     y_row1 = 0.5 * (dom_row1[0] + dom_row1[1])
        #     y_row2 = 0.5 * (dom_row2[0] + dom_row2[1])
        # except Exception:
        #     # Fallback approximate positions if domains aren't available
        #     y_row1, y_row2 = 0.75, 0.25

        # fig_grid.add_annotation(
        #     xref="paper",
        #     yref="paper",
        #     x=-0.04,
        #     y=y_row1,
        #     text="Can see the person",
        #     showarrow=False,
        #     textangle=-90,
        #     font=dict(family="Arial", size=14),
        # )
        # fig_grid.add_annotation(
        #     xref="paper",
        #     yref="paper",
        #     x=-0.04,
        #     y=y_row2,
        #     text="Cannot see the person",
        #     showarrow=False,
        #     textangle=-90,
        #     font=dict(family="Arial", size=14),
        # )

        # Save grid figure
        self.save_plotly(
            fig_grid,
            "yaw_hist_cam_dist",
            remove_margins=True,
            width=1600,
            height=900,
            save_final=True,
        )
