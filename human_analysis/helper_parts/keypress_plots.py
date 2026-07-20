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


class KeypressPlotMixin:
    """Focused plotting responsibility extracted from the legacy helper."""

    def plot_kp(self, df, y: list, y_legend_kp=None, x=None, events=None, events_width=1,
                events_dash='dot', events_colour='black', events_annotations_font_size=20,
                events_annotations_colour='black', xaxis_title='Time (s)',
                yaxis_title='Percentage of trials with response key pressed',
                xaxis_title_offset=0, yaxis_title_offset=0,
                xaxis_range=None, yaxis_range=None, stacked=False,
                pretty_text=False, show_text_labels=False,
                name_file='kp', save_file=False, save_final=False,
                fig_save_width=1320, fig_save_height=680, legend_x=0.7, legend_y=0.95, legend_columns=1,
                font_family=None, font_size=None, ttest_signals=None,
                ttest_marker_size=3, ttest_marker_colour='black', ttest_annotations_font_size=10,
                ttest_annotation_x=0, ttest_annotations_colour='black', ttest_row_height=0.5,
                xaxis_step=5, yaxis_step=5, line_width=1,
                custom_line_colors=None, custom_line_dashes=None, flag_trigger=False, margin=None,
                cross_p1_times=None, cross_p1_marker='diamond',
                cross_p1_marker_size=10, cross_p1_marker_colour='black',
                reuse_statistical_csv=None):
        """
        Plots keypress (response) data from a dataframe using Plotly, with options for custom lines,
        annotations, t-test result overlays, event markers, per-line cross_p1 markers,
        and customisable styling and saving.
        """

        logger.info('Creating keypress figure.')
        # calculate times
        times = df['Timestamp'].values
        # plotly
        fig = go.Figure()

        # ensure yaxis_range is mutable if provided as a tuple
        if isinstance(yaxis_range, tuple):
            yaxis_range = list(yaxis_range)

        # track plotted values to compute min/max for ticks
        all_values = []

        # plot keypress data
        for row_number, key in enumerate(y):
            values = df[key]
            if y_legend_kp:
                name = y_legend_kp[row_number]
            else:
                name = key

            # smoothen signal
            if self.smoothen_signal:
                if isinstance(values, pd.Series):
                    # Replace NaNs with 0 before smoothing
                    values = values.fillna(0).tolist()
                    values = self.smoothen_filter(values)
            else:
                # If not smoothing, ensure no NaNs anyway
                if isinstance(values, pd.Series):
                    values = values.fillna(0).tolist()
                else:
                    values = [v if not pd.isna(v) else 0 for v in values]

            # convert to 0-100%
            if flag_trigger:
                values = [v * 100 for v in values]  # type: ignore
            else:
                values = [v for v in values]  # type: ignore

            # collect values for y-axis tick range
            all_values.extend(values)  # type: ignore

            name = y_legend_kp[row_number] if y_legend_kp else key

            # main line
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines',
                x=times,
                line=dict(
                    width=line_width,
                    color=custom_line_colors[row_number] if custom_line_colors else None,
                    dash=custom_line_dashes[row_number] if custom_line_dashes else None,
                ),
                name=name
            ))

            # --- NEW: marker for cross_p1_time_s on this line ---
            if cross_p1_times and name in cross_p1_times:
                t_cross = cross_p1_times[name]

                # find nearest timestamp index (handles small timing mismatches)
                times_array = np.array(times, dtype=float)
                idx = int(np.abs(times_array - t_cross).argmin())

                x_marker = float(times_array[idx])
                y_marker = values[idx]

                fig.add_trace(go.Scatter(
                    x=[x_marker],
                    y=[y_marker],
                    mode='markers',
                    marker=dict(
                        symbol=cross_p1_marker,
                        size=cross_p1_marker_size,
                        color=cross_p1_marker_colour,
                    ),
                    name=f"{name} P1 cross",
                    showlegend=False
                ))

        # --- if no yaxis_range provided, derive it from the data so it's never None ---
        if yaxis_range is None:
            if all_values:  # safeguard against empty data
                actual_ymin = min(all_values)
                actual_ymax = max(all_values)
                yaxis_range = [actual_ymin, actual_ymax]
            else:
                # fallback range if for some reason there's no data
                yaxis_range = [0, 1]

        # draw events
        self.draw_events(fig=fig,
                         yaxis_range=yaxis_range,
                         events=events,
                         events_width=events_width,
                         events_dash=events_dash,
                         events_colour=events_colour,
                         events_annotations_font_size=events_annotations_font_size,
                         events_annotations_colour=events_annotations_colour)

        # update x-axis
        if xaxis_step:
            fig.update_xaxes(title_text=xaxis_title,
                             range=xaxis_range,
                             dtick=xaxis_step,
                             title_font=dict(family=font_family,
                                             size=font_size or common.get_configs('font_size'))
                             )
        else:
            fig.update_xaxes(title_text=xaxis_title,
                             range=xaxis_range,
                             title_font=dict(family=font_family,
                                             size=font_size or common.get_configs('font_size')))

        # Find actual y range across all series (for tick generation only)
        actual_ymin = min(all_values)
        actual_ymax = max(all_values)

        # Generate ticks from 0 up to actual_ymax
        positive_ticks = np.arange(0, actual_ymax + yaxis_step, yaxis_step)
        formatted_positive_ticks = [int(tick) if tick.is_integer() else tick for tick in positive_ticks]

        # Generate ticks from 0 down to actual_ymin (note: ymin is negative)
        negative_ticks = np.arange(0, actual_ymin - yaxis_step, -yaxis_step)
        formatted_negative_ticks = [int(tick) if tick.is_integer() else tick for tick in negative_ticks]

        # Combine and sort ticks
        visible_ticks = np.sort(np.unique(
            np.concatenate((formatted_negative_ticks, formatted_positive_ticks))
        ))

        tick_labels = [str(int(t)) if t.is_integer() else f"{t:.2f}" for t in visible_ticks]

        # Update y-axis with only relevant tick marks
        fig.update_yaxes(
            showgrid=True,
            range=yaxis_range,
            tickvals=visible_ticks,  # only show ticks for data range
            ticktext=tick_labels,
            automargin=True,
            title=dict(
                text="",
                font=dict(family=font_family,
                          size=font_size or common.get_configs('font_size')),
                standoff=0
            )
        )

        fig.add_annotation(
            text=yaxis_title,
            xref='paper',
            yref='paper',
            x=xaxis_title_offset,     # still left side
            y=0.5 + yaxis_title_offset,
            showarrow=False,
            textangle=-90,
            font=dict(family=font_family,
                      size=font_size or common.get_configs('font_size')),
            xanchor='center',
            yanchor='middle'
        )

        # prettify text
        if pretty_text:
            for variable in y:
                # check if column contains strings
                if isinstance(df.iloc[0][variable], str):
                    # replace underscores with spaces
                    df[variable] = df[variable].str.replace('_', ' ')
                    # capitalise
                    df[variable] = df[variable].str.capitalize()

        # use index of df if none is given
        if not x:
            x = df.index

        # draw ttest and anova rows
        if reuse_statistical_csv is None:
            reuse_statistical_csv = self.reuse_statistical_results

        self.draw_ttest(fig=fig,
                              times=times,
                              name_file=name_file,
                              yaxis_range=yaxis_range,
                              yaxis_step=yaxis_step,
                              ttest_signals=ttest_signals,
                              ttest_marker_size=ttest_marker_size,
                              ttest_marker_colour=ttest_marker_colour,
                              ttest_annotations_font_size=ttest_annotations_font_size,
                              ttest_annotations_colour=ttest_annotations_colour,
                              ttest_row_height=ttest_row_height,
                              ttest_annotation_x=ttest_annotation_x,
                              flag_trigger=flag_trigger,
                              reuse_statistical_csv=reuse_statistical_csv)

        # update template
        fig.update_layout(template=self.template)

        # format text labels
        if show_text_labels:
            fig.update_traces(texttemplate='%{text:.2f}')
            # Significance stars are literal text rather than numeric values.
            # Restore their template after formatting the normal data traces;
            # otherwise Plotly renders "*" as NaN.
            fig.update_traces(
                texttemplate="%{text}",
                selector=dict(name="__significance_markers__"),
            )

        # stacked bar chart
        if stacked:
            fig.update_layout(barmode='stack')

        # legend
        if legend_columns == 1:  # single column
            fig.update_layout(legend=dict(x=legend_x,
                                          y=legend_y,
                                          bgcolor='rgba(0,0,0,0)',
                                          font=dict(family=font_family,
                                                    size=font_size or common.get_configs('font_size') - 6)))

        # multiple columns
        elif legend_columns == 2:
            fig.update_layout(
                legend=dict(
                    x=legend_x,
                    y=legend_y,
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(size=font_size or common.get_configs('font_size')),
                    orientation='h',
                    traceorder='normal',
                    itemwidth=30,
                    itemsizing='constant'
                ),
                legend_title_text='',
                legend_tracegroupgap=5,
                legend_groupclick='toggleitem',
                legend_itemclick='toggleothers',
                legend_itemdoubleclick='toggle',
            )

        # adjust margins because of hardcoded ylim axis
        if margin:
            fig.update_layout(margin=margin)

        # update font family
        if font_family:
            fig.update_layout(font=dict(family=font_family))
        else:
            fig.update_layout(font=dict(family=common.get_configs('font_family')))

        # update font size
        if font_size:
            fig.update_layout(font=dict(size=font_size))
        else:
            fig.update_layout(font=dict(size=common.get_configs('font_size')))

        # save file to local output folder
        if save_file:
            self.save_plotly(fig=fig,
                             name=name_file,
                             remove_margins=False,
                             width=fig_save_width,
                             height=fig_save_height,
                             save_final=save_final)
        else:
            fig.show()

    def plot_column(self, mapping, column_name="TriggerValueRight", parameter=None, parameter_value=None,
                    additional_parameter=None, additional_parameter_value=None,
                    compare_trial="video_1", xaxis_title=None, yaxis_title=None, xaxis_range=None,
                    yaxis_range=[0, 100], margin=None, name=None,
                    trigger_threshold=0.05, output_subdir=None, save_final=True):
        """
        Generate a comparison plot of keypress data (or other time-series columns) and subjective slider ratings
        across multiple video trials relative to a test/reference condition.

        This function processes participant trigger matrices for each trial,
        aligns timestamps, attaches slider-based subjective ratings (annoyance,
        informativeness, noticeability), and prepares data for visualisation,
        including significance testing (paired t-tests) between the test trial and each comparison trial.

        Args:
            mapping (pd.DataFrame): DataFrame containing video metadata, including
                'video_id', 'sound_clip_name', 'display_name', and 'colour'.
            column_name (str): The column to extract for plotting (e.g., 'TriggerValueRight').
            xaxis_title (str, optional): Custom label for the x-axis.
            xaxis_range (list, optional): x-axis [min, max] limits for the plot.
            yaxis_range (list, optional): y-axis [min, max] limits for the plot.
            margin (dict, optional): Custom plot margin dictionary.
            trigger_threshold (float, optional): Trigger values strictly greater
                than this threshold are coded as pressed. The default, 0.05,
                is used for pressure-sensitive trigger data.
            output_subdir (str, optional): Relative subdirectory inside the
                output and figures folders where the plot should be saved.
            save_final (bool, optional): Whether to also save a copy in the
                configured figures folder.
        """

        # make yaxis_range mutable if it's a tuple
        if isinstance(yaxis_range, tuple):
            yaxis_range = list(yaxis_range)

        # === Filter mapping to same video_length as reference trial ===
        lens = mapping.loc[mapping["video_id"].eq(compare_trial), "video_length"].unique()

        if len(lens) == 0:
            raise ValueError(f"No rows found for video_id='{compare_trial}'")
        elif len(lens) > 1:
            # same video_id appears with different lengths; keep all those lengths
            mapping_filtered = mapping[mapping["video_length"].isin(lens)].copy()
        else:
            mapping_filtered = mapping[mapping["video_length"].eq(lens[0])].copy()

        if parameter is not None:
            mapping_filtered = mapping_filtered[mapping_filtered[parameter] == parameter_value]

        if additional_parameter is not None:
            mapping_filtered = mapping_filtered[mapping_filtered[additional_parameter] == additional_parameter_value]

        # Filter out control/test video IDs for comparison
        mapping_filtered = mapping_filtered[~mapping_filtered["video_id"].isin(["baseline_1", "baseline_2"])]
        plot_videos = mapping_filtered["video_id"]

        # Prepare containers for results and stats
        all_dfs = []        # averaged time-series for each trial
        all_labels = []     # display names for legend
        ttest_signals = []  # for significance testing

        # === Export trigger matrix for test (reference) trial ===
        test_output_csv = os.path.join(
            common.get_configs("output"),
            f"participant_{column_name}_{compare_trial}.csv"
        )

        self.export_participant_trigger_matrix(
            data_folder=self.data_folder,
            video_id=compare_trial,
            output_file=test_output_csv,
            column_name=column_name,
            mapping=mapping_filtered
        )

        # Read matrix, threshold pressure-sensitive trigger values, and extract
        # the binary pressed-state time series for the reference trial.
        test_raw_df = pd.read_csv(test_output_csv)
        if column_name == "TriggerValueRight":
            test_raw_df_for_analysis = self._threshold_trigger_matrix(test_raw_df, trigger_threshold)
        else:
            test_raw_df_for_analysis = test_raw_df
        test_matrix = extra_class.extract_time_series_values(test_raw_df_for_analysis)

        # === Loop through each trial (including reference) ===
        for video in plot_videos:
            # Get human-readable display name for this trial
            display_name = mapping_filtered.loc[mapping_filtered["video_id"] == video, "video_id"].values[0]

            trial_output_csv = os.path.join(
                common.get_configs("output"),
                f"participant_{column_name}_{video}.csv"
            )

            # Export trigger matrix for this video
            self.export_participant_trigger_matrix(
                data_folder=self.data_folder,
                video_id=video,
                output_file=trial_output_csv,
                column_name=column_name,
                mapping=mapping_filtered
            )

            # Read and process the trigger matrix to extract time series for this trial.
            # For pressure-sensitive trigger data, convert each participant-time bin
            # to 1 when any value exceeds the threshold and 0 otherwise.
            trial_raw_df = pd.read_csv(trial_output_csv)
            if column_name == "TriggerValueRight":
                trial_raw_df_for_analysis = self._threshold_trigger_matrix(trial_raw_df, trigger_threshold)
            else:
                trial_raw_df_for_analysis = trial_raw_df
            trial_matrix = extra_class.extract_time_series_values(trial_raw_df_for_analysis)

            # Compute participant-averaged pressed-state time series by timestamp.
            avg_df = extra_class.average_dataframe_vectors_with_timestamp(
                trial_raw_df_for_analysis,
                column_name=f"{column_name}"
            )

            all_dfs.append(avg_df)
            all_labels.append(display_name)

            # Prepare paired t-test between reference trial and each comparison trial
            if video != compare_trial:
                ttest_signals.append({
                    "signal_1": test_matrix,
                    "signal_2": trial_matrix,
                    "paired": True,
                    "label": f"{display_name}"
                })

        # === Combine all trial DataFrames for multi-trial plotting ===
        if not all_dfs:
            raise RuntimeError("No data frames found to plot.")

        combined_df = pd.DataFrame()
        combined_df["Timestamp"] = all_dfs[0]["Timestamp"]

        for df, label in zip(all_dfs, all_labels):
            combined_df[label] = df[column_name]

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

        # Row 1: all main car events at same height
        first_row_events = [
            ("yield_start_time_s", "Car decelerates"),
            ("yield_stop_time_s",  "Car stops"),
            ("yield_resume_time_s", "Car accelerates"),
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

        # Row 2: crossing event on its own line below
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

        # === cross_p1_time_s: per-line marker time for each video ===
        cross_p1_times = {}
        if "cross_p1_time_s" in mapping_filtered.columns:
            for video, label in zip(plot_videos, all_labels):
                series = mapping_filtered.loc[
                    mapping_filtered["video_id"] == video, "cross_p1_time_s"
                ].dropna()
                if not series.empty:
                    cross_p1_times[label] = float(series.iloc[0])

        # === Set line style: dashed for reference (compare_trial), solid for others ===
        custom_line_dashes = []
        for label in all_labels:
            vid = mapping_filtered.loc[mapping_filtered["video_id"] == label, "video_id"].values[0]
            if vid == compare_trial:
                custom_line_dashes.append("dot")
            else:
                custom_line_dashes.append("solid")

        # === Generate the main plot (delegated to plot_kp helper) ===
        base_name = self._short_kp_file_stem(name)
        name_file = os.path.join(output_subdir, base_name) if output_subdir else base_name

        self.plot_kp(
            df=combined_df,
            y=all_labels,
            y_legend_kp=all_labels,
            yaxis_range=yaxis_range,
            xaxis_range=xaxis_range,
            xaxis_title=xaxis_title,  # type: ignore
            yaxis_title=yaxis_title,  # type: ignore
            xaxis_title_offset=-0.04,  # type: ignore
            yaxis_title_offset=0.18,   # type: ignore
            name_file=name_file,
            show_text_labels=True,
            pretty_text=True,
            events=events,
            events_width=2,
            events_annotations_font_size=common.get_configs("font_size") - 8,
            stacked=False,
            ttest_signals=ttest_signals,
            ttest_row_height=6,
            ttest_annotations_font_size=common.get_configs("font_size") - 8,
            ttest_annotation_x=0.001,  # type: ignore
            ttest_marker_size=common.get_configs("font_size")-6,
            legend_x=0,
            legend_y=1.225,
            legend_columns=2,
            xaxis_step=1,
            yaxis_step=20,  # type: ignore
            line_width=3,
            font_size=common.get_configs("font_size"),
            fig_save_width=1470,
            fig_save_height=850,
            save_file=True,
            save_final=save_final,
            custom_line_dashes=custom_line_dashes,
            flag_trigger=True,
            margin=margin,
            cross_p1_times=cross_p1_times,
        )
