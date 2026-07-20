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
from utils.HMD_helper import HMD_yaw
from utils.tools import Tools
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


class HMD_helper:

    def __init__(self):
        self.template = common.get_configs('plotly_template')
        self.smoothen_signal = common.get_configs('smoothen_signal')
        self.folder_figures = common.get_configs('figures')  # subdirectory to save figures
        self.folder_stats = 'statistics'  # subdirectory to save statistical output
        self.data_folder = common.get_configs("data")  # Get path to participant data
        self.output_folder = common.get_configs("output")
        self.processed_data_cache = None
        self.statistical_cache_changed = False
        self.reuse_statistical_results = False

    def set_processed_data_cache(self, payload, reuse_statistical_results=False):
        """Attach the loaded processed-data payload for graph-only reruns."""
        self.processed_data_cache = payload
        self.statistical_cache_changed = False
        self.reuse_statistical_results = bool(reuse_statistical_results)

    @staticmethod
    def _half_open_bin_start(timestamps, resolution):
        """Map raw timestamps to starts of half-open bins ``[t, t + resolution)``."""
        numeric = pd.to_numeric(timestamps, errors="coerce").astype(float)
        resolution = float(resolution)
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be a positive finite number")
        # The tiny tolerance keeps a value represented as 0.2999999999999999
        # on the intended 0.3-second boundary without moving genuinely earlier
        # samples into the following bin.
        return np.floor((numeric + resolution * 1e-9) / resolution) * resolution

    @staticmethod
    def _short_kp_file_stem(name):
        """Return a short, stable filename stem for keypress condition plots.

        The older keypress filenames were long, for example
        ``all_videos_kp_slider_plot_eHMI_off_yielding``. This helper keeps
        the saved files easier to scan and avoids path length problems while
        preserving the condition meaning in a compact form.
        """
        short_names = {
            "all_values_with_yielding": "kp_all_y",
            "all_values_without_yielding": "kp_all_ny",
            "eHMI_off_yielding": "kp_e0_y",
            "eHMI_on_yielding": "kp_e1_y",
            "eHMI_off_non-yielding": "kp_e0_ny",
            "eHMI_on_non-yielding": "kp_e1_ny",
            "first_eHMI_on_non-yielding": "kp_p1_e1_ny",
            "first_eHMI_on_yielding": "kp_p1_e1_y",
            "first_eHMI_off_non-yielding": "kp_p1_e0_ny",
            "first_eHMI_off_yielding": "kp_p1_e0_y",
            "second_eHMI_on_non-yielding": "kp_p2_e1_ny",
            "second_eHMI_on_yielding": "kp_p2_e1_y",
            "second_eHMI_off_non-yielding": "kp_p2_e0_ny",
            "second_eHMI_off_yielding": "kp_p2_e0_y",
        }
        if name in short_names:
            return short_names[name]

        safe_name = str(name or "kp").strip()
        safe_name = re.sub(r"[^0-9A-Za-z]+", "_", safe_name).strip("_")
        return f"kp_{safe_name[:35]}" if safe_name else "kp_plot"

    @staticmethod
    def _short_question_file_stem(column_name, tag=None):
        """Return short, stable filenames for questionnaire figures.

        Long survey questions used to become long filenames. This mapping keeps
        the exported HTML/PNG/EPS names compact while preserving the meaning of
        each questionnaire item. Unknown columns still fall back to a short,
        sanitised stem.
        """
        short_names = {
            "Do you consent to participate in this study as described in the information provided above?": "consent",
            "Have you read and understood the above instructions?": "instructions",
            "What is your gender?": "gender",
            "Are you wearing any seeing aids during the experiments?": "seeing_aids",
            "Do you have problems with hearing?": "hearing",
            "How often in the last month have you experienced virtual reality?": "vr_exp",
            "I am comfortable with walking in areas with dense traffic.": "comfort_traffic",
            "The presence of another pedestrian reduces my willingness to cross the street when a car is driving towards me.": "ped_reduces_crossing",
            "What is your primary mode of transportation?": "transport",
            "On average, how often did you drive a vehicle in the last 12 months?": "driving_freq",
            "About how many kilometers did you drive in last 12 months?": "driving_km",
            "How often do you do the following?: Becoming angered by a particular type of driver, and indicate your hostility by whatever means you can.": "driver_anger",
            "How often do you do the following?: Disregarding the speed limit on a motorway.": "speed_motorway",
            "How often do you do the following?: Disregarding the speed limit on a residential road. ": "speed_residential",
            "How many accidents were you involved in when driving a car in the last 3 years? (please include all accidents, regardless of how they were caused, how slight they were, or where they happened)": "accidents",
            "How often do you do the following?: Driving so close to the car in front that it would be difficult to stop in an emergency. ": "tailgating",
            "How often do you do the following?: Racing away from traffic lights with the intention of beating the driver next to you. ": "racing_lights",
            "How often do you do the following?: Sounding your horn to indicate your annoyance with another road user. ": "horn",
            "How often do you do the following?: Using a mobile phone without a hands free kit.": "mobile_phone",
            "How often do you do the following?: Doing my best not to be obstacle for other drivers.": "not_obstacle",
            "I would like to communicate with other road users while crossing the road (for instance, using eye contact, gestures, verbal communication, etc.).": "road_user_comm",
            "I trust an automated car more than a manually driven car.": "trust_av",
            "The presence of another pedestrian influenced my willingness to cross the road.": "ped_influence",
            "The type of car (with eHMI or without eHMI) affected my decision to cross the road.": "car_type_effect",
            "What is your age (in years)?": "age",
            "At what age did you obtain your first license for driving a car or motorcycle?": "licence_age",
            "How stressful did you feel during the experiment?": "stress",
            "How anxious did you feel during the experiment?": "anxiety",
            "How realistic did you find the experiment?": "realism",
            "How would you rate your overall experience in this experiment?": "overall_experience",
        }

        key = str(column_name).strip()
        stem = short_names.get(key) or short_names.get(str(column_name))
        if stem is None:
            stem = re.sub(r"[^0-9A-Za-z]+", "_", key.lower()).strip("_")
            stem = stem[:45] if stem else "question"

        if tag:
            return f"{stem}_{tag}"
        return stem

    @staticmethod
    def _distance_code_to_meters(value):
        """Convert one raw mapping code in 1..5 to physical metres."""
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return np.nan
        numeric = float(numeric)
        if numeric == 0:
            return np.nan
        if numeric not in {1.0, 2.0, 3.0, 4.0, 5.0}:
            raise ValueError(f"Unexpected raw distPed code: {numeric}")
        return numeric * 2.0

    @classmethod
    def _distance_series_to_meters(cls, series):
        """Convert a Series of raw mapping codes to physical metres once."""
        numeric = pd.to_numeric(series, errors="coerce")
        mask_code = numeric.isin([1, 2, 3, 4, 5])
        invalid = numeric.notna() & (numeric != 0) & ~mask_code
        if invalid.any():
            unexpected = sorted(numeric.loc[invalid].unique().tolist())
            raise ValueError(f"Unexpected raw distPed codes: {unexpected}")
        mapped = numeric * 2.0
        mapped.loc[numeric == 0] = np.nan
        return mapped

    def smoothen_filter(self, signal, type_flter='OneEuroFilter'):
        """Smoothen list with a filter.

        Args:
            signal (list): input signal to smoothen
            type_flter (str, optional): type_flter of filter to use.

        Returns:
            list: list with smoothened data.
        """
        if type_flter == 'OneEuroFilter':
            filter_kp = OneEuroFilter(freq=common.get_configs('freq'),            # frequency
                                      mincutoff=common.get_configs('mincutoff'),  # minimum cutoff frequency
                                      beta=common.get_configs('beta'))            # beta value
            return [filter_kp(value) for value in signal]
        else:
            logger.error(f"Specified filter {type_flter} not implemented.")
            return -1

    def plot_column_distribution(self, df, columns, save_file=True, tag=None):
        """
        Plots and prints distributions of specified survey columns.

        Parameters:
            df (DataFrame or str): DataFrame or path to CSV.
            columns (list): List of column names to analyse.
            save_file (bool): Whether to save plots or just show them.
        """
        if isinstance(df, str):
            df = pd.read_csv(df)

        for column in columns:
            if column not in df.columns:
                logger.error(f"Column not found: {column}")
                continue

            logger.info(f"Distribution for: '{column}'")
            # Drop missing
            data = df[column].dropna().astype(str).str.strip()
            value_counts = data.value_counts()

            # Print counts
            for value, count in value_counts.items():
                logger.info(f"{value}: {count}")

            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(labels=value_counts.index, values=value_counts.values, hole=0.0)
            ])

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )

            # Save or display
            if save_file:
                filename = self._short_question_file_stem(column, tag=tag)
                self.save_plotly(fig, filename, save_final=True)
            else:
                fig.show()

    def distribution_plots(self, df, column_names, save_file=True):

        if isinstance(df, str):
            df = pd.read_csv(df)

        current_year = datetime.now().year

        for column_name in column_names:
            if column_name not in df.columns:
                logger.warning(f"Column not found: {column_name}")
                continue

            # Try numeric conversion
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            is_numeric = pd.api.types.is_numeric_dtype(temp_series)

            # Drop NaNs
            df_clean = df.dropna(subset=[column_name]).copy()

            if df_clean.empty:
                logger.warning(f"No valid data in column: {column_name}")
                continue

            if is_numeric:
                # Numeric column processing
                df_clean[column_name] = pd.to_numeric(df_clean[column_name], errors='coerce')

                # Special cleaning for age column
                if column_name.strip().lower() == "what is your age (in years)?".lower():
                    cleaned_ages = []
                    for val in df_clean[column_name]:
                        if 18 <= val <= 99:  # Valid age
                            cleaned_ages.append(val)
                        elif 1900 <= val <= current_year:  # Looks like year of birth
                            age = current_year - val
                            if 18 <= age <= 99:
                                cleaned_ages.append(age)
                        # Else: ignore nonsensical values
                    df_clean[column_name] = cleaned_ages

                mean_val = df_clean[column_name].mean()
                std_val = df_clean[column_name].std()
                logger.info(f"{column_name} - Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}")

                value_counts = df_clean[column_name].round().value_counts().sort_index()
                labels = [f"{int(v)}" for v in value_counts.index]
                values = value_counts.values
            else:
                # Categorical column processing
                df_clean[column_name] = df_clean[column_name].astype(str).str.strip()
                value_counts = df_clean[column_name].value_counts()
                labels = value_counts.index.tolist()
                values = value_counts.values.tolist()
                logger.info(f"{column_name} - Response counts: {dict(zip(labels, values))}")

            # Plotting
            fig = go.Figure(data=[
                go.Pie(labels=labels, values=values, hole=0.0, showlegend=True, sort=False)
            ])

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )

            # Save or display
            if save_file:
                filename = self._short_question_file_stem(column_name)
                self.save_plotly(fig, filename, save_final=True)
            else:
                fig.show()

    def plot_gender_by_nationality(self, csv_path,
                                   gender_col="What is your gender?",
                                   nationality_col="Nationality"):
        """
        Reads a CSV file and generates an interactive Plotly bar chart
        showing gender distribution for each nationality.
        """

        df = csv_path.copy() if isinstance(csv_path, pd.DataFrame) else pd.read_csv(csv_path)

        nationality_map = {
            "Pakistani": "Pakistan",
            "Yemini": "Yemen",
            "Yemeni": "Yemen",
            "Nepalese": "Nepal",
            "Chinese": "China",
            "chinese": "China",
            " Chinese": "China",
            "Polish": "Poland",
            "Indian ": "India",
            "Dutch ": "Netherlands",
            "Dutch": "Netherlands",
            "Iranian": "Iran",
            "Romanian": "Romania",
            "Spanish": "Spain",
            "Colombian": "Colombia",
            "portuguese": "Portugal",
            "Taiwanese": "Taiwan",
            "German": "Germany",
            "Indian": "India",
            "dutch": "Netherlands"
        }

        # Apply mapping
        df[nationality_col] = df[nationality_col].map(nationality_map).fillna(
            df[nationality_col].str.capitalize()
        )

        # --- Group data ---
        s = df.groupby([nationality_col, gender_col]).size()
        s.name = "Count"
        grouped = s.reset_index()

        # --- Plot ---
        fig = px.bar(
            grouped,
            x=nationality_col,
            y="Count",
            color=gender_col,
            barmode="group",
            title=""
        )

        fig.show()

    @staticmethod
    def read_slider_data(data_folder, output_folder):
        """
        Reads participant slider CSVs from all participant folders, aggregates the
        ratings (noticeability, informativeness, annoyance) for all trials,
        and saves a summary CSV per slider to the output folder.

        Args:
            data_folder (str): Path to the folder containing participant subfolders.
            output_folder (str): Directory to save aggregated CSVs for each slider.
        """
        participant_data = {}  # Store per-participant DataFrames
        all_trials = set()  # Collect all unique trial IDs

        # Iterate over each participant's folder
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue

            # Parse participant ID from folder name
            match = re.match(r'Participant_(\d+)', folder)
            if not match:
                continue
            participant_id = int(match.group(1))

            # Find the CSV with slider data for this participant
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # Expected pattern: Participant_[id]_[number]_[number].csv
                if re.match(rf'Participant_{participant_id}_\d+_\d+\.csv', file):
                    # Assume no header: columns are trial, noticeability, info, annoyance
                    df = pd.read_csv(file_path,
                                     header=None,
                                     names=["trial", "behaviour", "distance", "intention"])
                    df.set_index("trial", inplace=True)
                    participant_data[participant_id] = df
                    all_trials.update(df.index)
                    break  # Stop at first valid slider CSV

        # Build a sorted trial list (with 'test' first if present)
        all_trials = sorted([t for t in all_trials if t != "test"],
                            key=lambda x: int(re.search(r'\d+', x).group()))  # type: ignore
        all_trials.insert(0, "test") if "test" in all_trials else None

        # Prepare dict to aggregate each slider rating across all participants
        slider_data = {"behaviour": [], "distance": [], "intention": []}

        # For each participant, gather ratings for all trials, filling missing with None
        for participant_id, df in sorted(participant_data.items()):
            row = {"participant_id": participant_id}
            for trial in all_trials:
                if trial in df.index:
                    row[trial] = df.loc[trial].to_list()
                else:
                    row[trial] = [None, None, None]

            # Split values for each slider
            slider_data["behaviour"].append([participant_id] + [vals[0] for vals in row.values() if isinstance(vals, list)])  # noqa: E501
            slider_data["distance"].append([participant_id] + [vals[1] for vals in row.values() if isinstance(vals, list)])  # noqa: E501
            slider_data["intention"].append([participant_id] + [vals[2] for vals in row.values() if isinstance(vals, list)])  # noqa: E501

        # Convert lists to DataFrames, rename columns, and add average row
        for slider, data in slider_data.items():
            df = pd.DataFrame(data, columns=["participant_id"] + all_trials)
            # Rename trial columns using mapping (video_id to sound_clip_name)
            # df.rename(columns={trial: mapping_dict.get(trial, trial) for trial in all_trials}, inplace=True)

            # Add average row at the end (ignoring participant_id)
            avg_values = df.iloc[:, 1:].mean(skipna=True)
            avg_row = pd.DataFrame([["average"] + avg_values.tolist()], columns=df.columns)  # type: ignore
            df = pd.concat([df, avg_row], ignore_index=True)

            # Save the aggregated slider data to CSV
            output_path = os.path.join(output_folder, f"slider_input_{slider}.csv")
            df.to_csv(output_path, index=False)

    def save_plotly(self, fig, name, remove_margins=False, width=1320, height=680, save_eps=True, save_png=True,
                    save_html=True, open_browser=True, save_mp4=False, save_final=False):
        """
        Save a Plotly figure as HTML and image files.

        The ``name`` argument may include subdirectories, for example
        ``kp_threshold_sensitivity/threshold_05pct/my_figure``. Any missing
        output folders are created automatically.
        """
        # disable mathjax globally for Kaleido
        pio.kaleido.scope.mathjax = None

        output_root = os.path.join(common.get_configs("output"))
        final_root = self.folder_figures
        os.makedirs(output_root, exist_ok=True)
        if save_final:
            os.makedirs(final_root, exist_ok=True)

        # Keep only safe path components while preserving intentional folders.
        name = str(name).replace("\\", os.sep).replace("/", os.sep)
        name = os.path.normpath(name)
        if name.startswith("..") or os.path.isabs(name):
            raise ValueError(f"Figure name must be a relative path, got: {name}")
        if name == "head_heading" or name.startswith(f"head_heading{os.sep}"):
            name = os.path.basename(name)

        output_base = os.path.join(output_root, name)
        final_base = os.path.join(final_root, name)
        os.makedirs(os.path.dirname(output_base), exist_ok=True)
        if save_final:
            os.makedirs(os.path.dirname(final_base), exist_ok=True)

        # Limit only the file stem when paths become too long.
        output_dir = os.path.dirname(output_base)
        final_dir = os.path.dirname(final_base)
        stem = os.path.basename(output_base)
        max_dir_len = max(len(output_dir), len(final_dir))
        if max_dir_len + len(stem) > 195:
            safe_len = max(20, 190 - max_dir_len)
            stem = stem[:safe_len]
            output_base = os.path.join(output_dir, stem)
            final_base = os.path.join(final_dir, stem)

        # Save once in the output directory, then copy the exact file into the
        # configured figures directory. This guarantees that both locations
        # contain the same figure and avoids rendering the same figure twice.
        if save_html:
            output_html = output_base + ".html"
            py.offline.plot(fig, filename=output_html, auto_open=open_browser)
            logger.info(f"Saved figure: {output_html}")

            if save_final:
                final_html = final_base + ".html"
                shutil.copy2(output_html, final_html)
                logger.info(f"Saved figure: {final_html}")

        # remove white margins
        if remove_margins:
            fig.update_layout(margin=dict(l=100, r=2, t=20, b=12))

        # save as eps
        if save_eps:
            try:
                output_eps = output_base + ".eps"
                fig.write_image(output_eps, width=width, height=height)
                logger.info(f"Saved figure: {output_eps}")

                if save_final:
                    final_eps = final_base + ".eps"
                    shutil.copy2(output_eps, final_eps)
                    logger.info(f"Saved figure: {final_eps}")
            except Exception as exc:
                logger.warning(
                    f"Skipping EPS export for '{name}' because Plotly/Kaleido could not create the EPS file: {exc}"
                )

        # save as png
        if save_png:
            try:
                output_png = output_base + ".png"
                fig.write_image(output_png, width=width, height=height)
                logger.info(f"Saved figure: {output_png}")

                if save_final:
                    final_png = final_base + ".png"
                    shutil.copy2(output_png, final_png)
                    logger.info(f"Saved figure: {final_png}")
            except Exception as exc:
                logger.warning(
                    f"Skipping PNG export for '{name}' because Plotly/Kaleido could not create the PNG file: {exc}"
                )

        # save as mp4
        if save_mp4:
            try:
                output_mp4 = output_base + '.mp4'
                fig.write_image(output_mp4, width=width, height=height)
                logger.info(f"Saved figure: {output_mp4}")
            except Exception as exc:
                logger.warning(
                    f"Skipping MP4 export for '{name}' because Plotly/Kaleido could not create the MP4 file: {exc}"
                )

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
        HMD_helper.draw_events(fig=fig,
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

    def export_participant_trigger_matrix(
        self,
        data_folder,
        video_id,
        output_file,
        column_name,
        mapping,
        overwrite=False,
    ):
        """
        Export a matrix of trigger (or other column) values per participant for a given video.

        Each cell contains a list of values (one per frame or timepoint) for that participant and timestamp.
        Missing data is left as NaN, not zero.

        Args:
            data_folder (str): Path to folder containing participant subfolders with CSVs.
            video_id (str): Target video identifier (e.g. '002', 'test', etc.).
            output_file (str): Path to output CSV file (e.g. '_output/participant_trigger_002.csv').
            column_name (str): Name of the column to export (e.g. 'TriggerValueRight').
            mapping (pd.DataFrame): Mapping DataFrame containing at least 'video_id' and 'video_length'.
            overwrite (bool): Rebuild from raw participant files even if output exists.
        """

        if not overwrite and os.path.isfile(output_file):
            return
        if not overwrite and not bool(common.get_configs("always_analyse")):
            raise FileNotFoundError(
                f"A trigger matrix required by the processed-data cache is missing: {output_file}. "
                "Set always_analyse to true for one run to rebuild the cache."
            )

        participant_matrix = {}    # Store trigger value lists for each participant, keyed by timestamp
        all_timestamps = set()     # Collect all observed timestamps for alignment

        # Calculate time bin resolution (in seconds) from config
        resolution = common.get_configs("kp_resolution") / 1000.0

        # Iterate over participant folders
        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue  # Ignore files, only process directories

            # Extract participant ID from folder name (expecting "Participant_###_...")
            match = re.match(r'Participant_(\d+)', folder)
            if not match:
                continue
            participant_id = int(match.group(1))

            # Search for this participant's file matching the video ID
            for file in os.listdir(folder_path):
                if f"{video_id}.csv" in file:
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    # Check required columns
                    if "Timestamp" not in df or column_name not in df:
                        continue

                    # Aggregate 50-Hz raw samples into half-open 100-ms bins.
                    # Floor-based binning gives [0.0, 0.1), [0.1, 0.2), ...
                    # rather than nearest-bin rounding around bin centres.
                    df["Timestamp"] = self._half_open_bin_start(
                        df["Timestamp"], resolution
                    ).round(6)

                    # Group by timestamp, collect all values in a list per bin
                    grouped = df.groupby("Timestamp", as_index=True)[column_name].apply(list)

                    # Store the resulting dict: timestamp -> list of values
                    participant_matrix[f"P{participant_id}"] = grouped.to_dict()
                    all_timestamps.update(grouped.index)
                    break  # Only process the first matching file for this participant

        # Get the expected timeline from mapping for alignment (using video_length)
        video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
        if not video_length_row.empty:
            video_length_sec = video_length_row.values[0] / 1000  # Convert ms to seconds
            all_timestamps = np.round(np.arange(0.0, video_length_sec + resolution, resolution), 2).tolist()
        else:
            logger.warning(f"Video length not found in mapping for video_id {video_id}")

        # Build DataFrame with one row per timestamp
        combined_df = pd.DataFrame({"Timestamp": all_timestamps})

        # For each participant, add a column: each entry is a list or NaN (if no data for that timestamp)
        for participant, values in participant_matrix.items():
            combined_df[participant] = combined_df["Timestamp"].map(values)

        # Save matrix to CSV (do NOT fill missing with zero; keep NaN for clarity)
        combined_df.to_csv(output_file, index=False)

    def export_participant_quaternion_matrix(self, data_folder, video_id, output_file, mapping, overwrite=False):
        """
        Export a matrix of raw HMD quaternions per participant per timestamp for a given video.
        If overwrite=False and output_file exists, it is reused.
        """

        # short-circuit if already exists
        if not overwrite and os.path.isfile(output_file):
            return
        if not overwrite and not bool(common.get_configs("always_analyse")):
            raise FileNotFoundError(
                f"A quaternion matrix required by the processed-data cache is missing: {output_file}. "
                "Set always_analyse to true for one run to rebuild the cache."
            )

        participant_matrix = {}
        all_timestamps = set()

        resolution = common.get_configs("yaw_resolution") / 1000.0

        for folder in sorted(os.listdir(data_folder)):
            folder_path = os.path.join(data_folder, folder)
            if not os.path.isdir(folder_path):
                continue

            match = re.match(r"Participant_(\d+)$", folder, re.IGNORECASE)
            if not match:
                continue

            participant_id = int(match.group(1))

            for file in os.listdir(folder_path):
                if file == f"{video_id}.csv":
                    file_path = os.path.join(folder_path, file)
                    df = pd.read_csv(file_path)

                    required_cols = {
                        "Timestamp",
                        "HMDRotationW",
                        "HMDRotationX",
                        "HMDRotationY",
                        "HMDRotationZ",
                    }
                    if not required_cols.issubset(df.columns):
                        continue

                    df["Timestamp"] = (
                        (df["Timestamp"] / resolution).round() * resolution
                    ).round(2)

                    quats_by_time = (
                        df.groupby("Timestamp")[["HMDRotationW", "HMDRotationX", "HMDRotationY", "HMDRotationZ"]]
                        .apply(lambda g: g.values.tolist()).to_dict()  # type: ignore
                    )

                    participant_matrix[f"P{participant_id}"] = quats_by_time
                    all_timestamps.update(quats_by_time.keys())
                    break

        video_length_row = mapping.loc[mapping["video_id"] == video_id, "video_length"]
        if not video_length_row.empty:
            video_length_sec = video_length_row.values[0] / 1000.0
            all_timestamps = (
                np.round(
                    np.arange(0, video_length_sec + resolution, resolution), 2
                ).tolist()
            )
        else:
            logger.warning(f"Video length not found in mapping for video_id {video_id}")
            all_timestamps = sorted(all_timestamps)

        combined_df = pd.DataFrame({"Timestamp": all_timestamps})
        for participant, values in participant_matrix.items():
            combined_df[participant] = combined_df["Timestamp"].map(
                lambda ts: str(values.get(ts, []))
            )

        combined_df.to_csv(output_file, index=False)

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

    def heat_plot(self, folder_path: str, mapping_df: pd.DataFrame, relation: str = "ratio",
                  colorscale: str = "Viridis", summary_func=np.mean,
                  trigger_threshold: float = 0.05):
        """
        Compute summary values per video CSV, rename axes using `mapping_df`,
        and show a Plotly heatmap of pairwise relations (no numbers, no colorbar).

        Additionally, this function writes a 'trigger_summary.csv' file into
        `folder_path` with columns:
            ['label', 'avg_trigger', 'sd_trigger', 'n_samples']

        Parameters
        ----------
        folder_path : str
            Folder with 'participant_TriggerValueRight_video_*.csv'.
        mapping_df : pd.DataFrame
            Must include columns: 'video_id', 'condition_name', 'camera',
            'cross_p1_time_s', 'cross_p2_time_s'.
            The cutoff time used per video is cross_p1_time_s if camera==0,
            otherwise cross_p2_time_s. Only rows with Timestamp <= cutoff are used.
        relation : {'ratio','diff'}
            'ratio' uses vb/va; 'diff' uses vb - va.
        colorscale : str
            Plotly colorscale name.
        summary_func : callable
            Aggregator applied to binary trigger-pressed states per condition
            (default: np.mean). With the default, `avg_trigger` is the
            proportion of analysed participant-time bins for which the trigger
            value was above `trigger_threshold`.
        trigger_threshold : float
            Trigger values strictly greater than this threshold are counted as
            pressed. The default of 0.05 is used because the trigger was
            pressure-sensitive and light contact could produce small values.

        Returns
        -------
        averages : dict
            {mapped_label: summary_value}  (means per condition; for compatibility)
        relation_df : pd.DataFrame
            Pairwise matrix (mapped_label × mapped_label)
        fig : plotly.graph_objects.Figure
            The heatmap figure.
        """
        # ---- Validate mapping_df ----
        required_cols = {"video_id", "condition_name", "camera", "cross_p1_time_s", "cross_p2_time_s"}
        if not required_cols.issubset(set(mapping_df.columns)):
            raise ValueError(f"mapping_df must have columns {required_cols}, got {list(mapping_df.columns)}")

        # Normalize dtypes
        md = mapping_df.copy()
        md["video_id"] = md["video_id"].astype(str)

        # Force numerics for times; invalids -> NaN
        md["cross_p1_time_s"] = pd.to_numeric(md["cross_p1_time_s"], errors="coerce")
        md["cross_p2_time_s"] = pd.to_numeric(md["cross_p2_time_s"], errors="coerce")
        md["camera"] = pd.to_numeric(md["camera"], errors="coerce").astype("Int64")

        # Build lookup: video_id -> {label, camera, p1, p2}
        mapping_info = {
            row["video_id"]: {
                "label": str(row["condition_name"]),
                "camera": int(row["camera"]) if pd.notna(row["camera"]) else None,
                "p1": float(row["cross_p1_time_s"]) if pd.notna(row["cross_p1_time_s"]) else None,
                "p2": float(row["cross_p2_time_s"]) if pd.notna(row["cross_p2_time_s"]) else None,
            }
            for _, row in md.iterrows()
        }

        # ---- Find files ----
        pattern = os.path.join(folder_path, "participant_TriggerValueRight_video_*.csv")
        file_list = glob.glob(pattern)
        if not file_list:
            raise FileNotFoundError(f"No files matched pattern: {pattern}")

        # Sort by numeric video index if present (video_123), else alphabetically
        def _video_key(fn):
            m = re.search(r"video_(\d+)\.csv$", os.path.basename(fn), flags=re.IGNORECASE)
            return (0, int(m.group(1))) if m else (1, os.path.basename(fn).lower())

        file_list = sorted(file_list, key=_video_key)

        # ---- Helper: extract video_id from filename ----
        def _extract_video_id(filename: str) -> str:
            base = os.path.basename(filename)
            m = re.search(r"(video_\d+)\.csv$", base, flags=re.IGNORECASE)
            return m.group(1) if m else base

        # ---- Collect trigger-pressed states per condition label ----
        # per_label_values[label] stores one binary value per participant-time bin:
        # 1 = at least one trigger value in that bin was above trigger_threshold;
        # 0 = no trigger value in that bin was above trigger_threshold.
        # This makes avg_trigger the proportion of analysed time marked as unsafe,
        # rather than the mean trigger pressure/intensity.
        per_label_values: Dict[str, np.ndarray] = {}
        per_label_raw_values: Dict[str, np.ndarray] = {}

        for file_path in file_list:
            video_id = _extract_video_id(file_path)
            info = mapping_info.get(video_id)

            # Determine cutoff (if mapping missing or incomplete, we'll fall back to no cutoff)
            cutoff = None
            label = video_id
            if info:
                label = info["label"]
                cam = info["camera"]
                if cam == 0:
                    cutoff = info["p1"]
                elif cam == 1:
                    cutoff = info["p2"]
                else:
                    logger.warning(f"⚠️ camera not 0/1 for {video_id}; using full data (no cutoff).")

            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"⚠️ Error reading {file_path}: {e}")
                continue

            if "Timestamp" not in df.columns:
                logger.warning(f"⚠️ 'Timestamp' column missing in {file_path}; using full data.")
                ts_filtered = df
            else:
                # Coerce Timestamp to numeric seconds, drop rows with invalid timestamps
                ts = pd.to_numeric(df["Timestamp"], errors="coerce")
                valid = ts.notna()  # type: ignore
                df = df.loc[valid].copy()
                ts = ts.loc[valid]  # type: ignore

                if cutoff is not None and np.isfinite(cutoff):
                    mask = ts <= float(cutoff)
                    ts_filtered = df.loc[mask].copy()
                else:
                    ts_filtered = df

            # If nothing remains after filtering, skip
            if ts_filtered.empty:
                logger.warning(f"⚠️ No rows after time filtering for {file_path}; skipping.")
                continue

            # Drop Timestamp before aggregating values
            ts_filtered = ts_filtered.drop(columns=["Timestamp"], errors="ignore")

            # Parse list-like cells. For the manuscript's primary perceived-risk
            # measure, each participant-time bin is converted into a binary state:
            # pressed = any trigger value in that bin is above the threshold.
            pressed_states = []
            raw_values = []
            for col in ts_filtered.columns:
                for val in ts_filtered[col]:
                    nums = self._extract_numeric_values_from_cell(val)
                    raw_values.extend(nums)
                    pressed_states.append(1.0 if any(x > trigger_threshold for x in nums) else 0.0)

            if not pressed_states:
                logger.warning(f"⚠️ No analysable trigger bins found (after cutoff) in {file_path}; skipping.")
            else:
                vals = np.array(pressed_states, dtype=float)
                raw_vals = np.array(raw_values, dtype=float) if raw_values else np.array([], dtype=float)
                if label not in per_label_values:
                    per_label_values[label] = vals
                    per_label_raw_values[label] = raw_vals
                else:
                    per_label_values[label] = np.concatenate([per_label_values[label], vals])
                    per_label_raw_values[label] = np.concatenate([per_label_raw_values[label], raw_vals])

        if not per_label_values:
            raise ValueError("No valid numeric data found in videos (after applying time cutoffs).")

        # ---- Build trigger summary (unsafe proportion + SD + n) ----
        trigger_summary_df = (
            pd.DataFrame(
                [
                    {
                        "label": label,
                        # Proportion of analysed participant-time bins marked unsafe.
                        "avg_trigger": float(summary_func(vals)),
                        "sd_trigger": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
                        "n_samples": int(len(vals)),
                        "n_trigger_bins": int(len(vals)),
                        "n_raw_trigger_samples": int(len(per_label_raw_values.get(label, []))),
                        # Kept for diagnostics only; this is no longer the primary risk measure.
                        "mean_trigger_intensity": (
                            float(np.mean(per_label_raw_values[label]))
                            if len(per_label_raw_values.get(label, [])) > 0 else np.nan
                        ),
                        "sd_trigger_intensity": (
                            float(np.std(per_label_raw_values[label], ddof=1))
                            if len(per_label_raw_values.get(label, [])) > 1 else np.nan
                        ),
                        "trigger_threshold": float(trigger_threshold),
                    }
                    for label, vals in per_label_values.items()
                ]
            )
            .sort_values("label")
            .reset_index(drop=True)
        )

        # Backwards-compatible dict of means only (for relation matrix)
        averages = dict(
            zip(trigger_summary_df["label"], trigger_summary_df["avg_trigger"])
        )

        # ---- Save trigger_summary.csv ----
        trigger_summary_path = os.path.join(folder_path, "trigger_summary.csv")
        trigger_summary_df.to_csv(trigger_summary_path, index=False)
        logger.info(f"Saved trigger summary (unsafe proportion + SD) to: {trigger_summary_path}")

        # ---- Build pairwise relation matrix ----
        labels = list(averages.keys())
        n = len(labels)
        mat = np.full((n, n), np.nan, dtype=float)

        for i, a in enumerate(labels):
            va = averages[a]
            for j, b in enumerate(labels):
                vb = averages[b]
                if relation == "ratio":
                    mat[i, j] = (vb / va) if va not in (0, None) else np.nan
                elif relation == "diff":
                    mat[i, j] = vb - va
                else:
                    raise ValueError("relation must be 'ratio' or 'diff'")

        relation_df = pd.DataFrame(mat, index=labels, columns=labels)

        # ---- Add formatted decimal text for cells ----
        text_values = np.where(
            np.isnan(relation_df.values),
            "",
            np.round(relation_df.values, 2).astype(str)
        )

        # ---- Plotly heatmap (no cell text, no colorbar) ----
        fig = go.Figure(
            data=go.Heatmap(
                z=relation_df.values,
                x=relation_df.columns,
                y=relation_df.index,
                text=text_values,           # show decimals in cells
                texttemplate="%{text}",     # ensures text is rendered
                colorscale=colorscale,
                showscale=False,  # remove colorbar
                hovertemplate=(
                    "<b>%{y}</b> → <b>%{x}</b><br>"
                    f"{relation}: %{{z:.6f}}<extra></extra>"
                )
            )
        )

        fig.update_layout(
            title="",
            xaxis=dict(
                title="",
                tickangle=45,
                tickfont=dict(size=18, color="black"),
                titlefont=dict(size=16, color="black")
            ),
            yaxis=dict(
                title="",
                autorange="reversed",
                tickfont=dict(size=18, color="black"),
                titlefont=dict(size=16, color="black")
            ),
            margin=dict(l=80, r=40, t=60, b=90),
            width=2400,
            height=2400,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        self.save_plotly(fig, 'heatmap', height=2400, width=2400, save_final=True)

        # Optional: also save the pairwise relation matrix
        relation_path = os.path.join(folder_path, "trigger_relation_matrix.csv")
        relation_df.to_csv(relation_path)
        logger.info(f"Saved trigger relation matrix to: {relation_path}")

        return averages, relation_df, fig

    @staticmethod
    def _extract_numeric_values_from_cell(value):
        """Parse a matrix cell containing a string-encoded list and return finite numeric values."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        try:
            parsed = ast.literal_eval(value) if isinstance(value, str) else value
        except Exception:
            return []
        if not isinstance(parsed, list):
            return []
        out = []
        for item in parsed:
            if isinstance(item, (int, float)) and np.isfinite(item):
                out.append(float(item))
        return out

    @classmethod
    def _trigger_pressed_state(cls, value, threshold=0.05):
        """Return 1 when any finite trigger value in a matrix cell is above threshold."""
        values = cls._extract_numeric_values_from_cell(value)
        return 1.0 if any(item > threshold for item in values) else 0.0

    @classmethod
    def _threshold_trigger_matrix(cls, df, threshold=0.05):
        """Convert list-valued trigger matrix cells into binary pressed-state cells.

        Each non-timestamp cell becomes ``[1.0]`` when any finite trigger value in
        that participant-time bin is greater than ``threshold`` and ``[0.0]``
        otherwise. Keeping a list in each cell preserves compatibility with the
        existing averaging and time-series helper functions.
        """
        out = df.copy()
        for col in out.columns:
            if col == "Timestamp":
                continue
            out[col] = out[col].apply(lambda value: [cls._trigger_pressed_state(value, threshold)])
        return out

    def _compute_trial_level_trigger_summary(self, trigger_matrices_dir, mapping_df, trigger_threshold: float = 0.05):
        """Build participant x video unsafe-time summaries from exported trigger matrices."""
        mapping_info = mapping_df.copy()
        for col in ["video_id", "condition_name"]:
            mapping_info[col] = mapping_info[col].astype(str)
        for col in ["camera", "cross_p1_time_s", "cross_p2_time_s", "distPed", "yielding", "eHMIOn"]:
            if col in mapping_info.columns:
                mapping_info[col] = pd.to_numeric(mapping_info[col], errors="coerce")

        records = []
        pattern = os.path.join(trigger_matrices_dir, "participant_TriggerValueRight_video_*.csv")
        for fp in sorted(glob.glob(pattern)):
            base = os.path.basename(fp)
            m = re.search(r"(video_\d+)\.csv$", base, flags=re.IGNORECASE)
            if not m:
                continue
            video_id = m.group(1)
            map_row = mapping_info.loc[mapping_info["video_id"] == video_id]
            if map_row.empty:
                continue
            row0 = map_row.iloc[0]
            cutoff = None
            camera = row0.get("camera")
            if pd.notna(camera):
                camera = int(camera)
                if camera == 0 and pd.notna(row0.get("cross_p1_time_s")):
                    cutoff = float(row0["cross_p1_time_s"])
                elif camera == 1 and pd.notna(row0.get("cross_p2_time_s")):
                    cutoff = float(row0["cross_p2_time_s"])

            df = pd.read_csv(fp)
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
                df = df.dropna(subset=["Timestamp"])
                if cutoff is not None and np.isfinite(cutoff):
                    df = df[df["Timestamp"] <= cutoff]

            participant_cols = [c for c in df.columns if c != "Timestamp"]
            for participant_col in participant_cols:
                pm = re.search(r"P(\d+)", str(participant_col))
                if not pm:
                    continue
                participant = int(pm.group(1))
                raw_values = []
                pressed_states = []
                for cell in df[participant_col].tolist():
                    cell_values = self._extract_numeric_values_from_cell(cell)
                    raw_values.extend(cell_values)
                    pressed_states.append(1.0 if any(v > trigger_threshold for v in cell_values) else 0.0)

                if not pressed_states:
                    unsafe_prop = np.nan
                    sd_unsafe_prop = np.nan
                    n_bins = 0
                else:
                    state_arr = np.asarray(pressed_states, dtype=float)
                    unsafe_prop = float(np.mean(state_arr))
                    sd_unsafe_prop = float(np.std(state_arr, ddof=1)) if len(state_arr) > 1 else np.nan
                    n_bins = int(len(state_arr))

                if raw_values:
                    raw_arr = np.asarray(raw_values, dtype=float)
                    mean_trigger_intensity = float(np.mean(raw_arr))
                    sd_trigger_intensity = float(np.std(raw_arr, ddof=1)) if len(raw_arr) > 1 else np.nan
                    n_raw_samples = int(len(raw_arr))
                else:
                    mean_trigger_intensity = np.nan
                    sd_trigger_intensity = np.nan
                    n_raw_samples = 0

                records.append({
                    "participant": participant,
                    "video_id": video_id,
                    "condition_name": str(row0["condition_name"]),
                    # avg_trigger is retained as the public column used downstream,
                    # but it now means unsafe-time proportion, not mean trigger intensity.
                    "avg_trigger": unsafe_prop,
                    "sd_trigger": sd_unsafe_prop,
                    "n_trigger_samples": n_bins,
                    "n_trigger_bins": n_bins,
                    "n_raw_trigger_samples": n_raw_samples,
                    "mean_trigger_intensity": mean_trigger_intensity,
                    "sd_trigger_intensity": sd_trigger_intensity,
                    "trigger_threshold": float(trigger_threshold),
                })

        if not records:
            raise ValueError(
                "No participant-level trigger matrices were found. Run plot_column/heat_plot first so "
                "participant_TriggerValueRight_video_*.csv files exist in the output folder."
            )
        return pd.DataFrame.from_records(records)

    @staticmethod
    def _pretty_mixed_term(term):
        mapping = {
            "Intercept": "Intercept",
            "C(yielding)[T.1]": "Yielding",
            "C(eHMIOn)[T.1]": "eHMI",
            "C(camera)[T.1]": "Participant-first / avatar-second order",
            "distPed_m": "Distance (m)",
            "C(yielding)[T.1]:C(eHMIOn)[T.1]": "Yielding × eHMI",
            "C(yielding)[T.1]:C(camera)[T.1]": "Yielding × relative pedestrian order",
            "C(eHMIOn)[T.1]:C(camera)[T.1]": "eHMI × relative pedestrian order",
            "Group Var": "Random intercept variance",
        }
        return mapping.get(term, term)

    def _run_mixed_effects_model(self, trial_df, outcome, out_dir=None):
        """Fit categorical-distance models, attempting participant random slopes first."""
        if smf is None:
            logger.warning(f"statsmodels is not available; skipping mixed model for {outcome}")
            return None, None

        save_dir = out_dir or self.output_folder
        os.makedirs(save_dir, exist_ok=True)
        coefficients_path = os.path.join(
            save_dir, f"mixed_model_coefficients_{outcome}.csv"
        )
        analysis_version = "categorical_distance_random_slopes_v2"
        if self.reuse_statistical_results and os.path.isfile(coefficients_path):
            cached = pd.read_csv(coefficients_path)
            if (
                "analysis_version" in cached.columns
                and cached["analysis_version"].eq(analysis_version).all()
            ):
                logger.info(f"Reused cached improved mixed model for {outcome}.")
                return None, cached

        model_df = trial_df.copy()
        model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")
        needed = ["participant", outcome, "yielding", "eHMIOn", "camera", "distPed_m"]
        if "trial_number" in model_df.columns:
            needed.append("trial_number")
        model_df = model_df.dropna(subset=needed)
        if model_df.empty:
            logger.warning(f"No data available for mixed model outcome {outcome}")
            return None, None

        model_df["distPed_centered"] = model_df["distPed_m"] - 6.0
        trial_terms = ""
        random_trial_term = ""
        if "trial_number" in model_df.columns:
            model_df["trial_number_centered"] = (
                model_df["trial_number"]
                - model_df.groupby("participant")["trial_number"].transform("mean")
            )
            trial_terms = " + trial_number_centered + I(trial_number_centered ** 2)"
            random_trial_term = " + trial_number_centered"

        formula = (
            f"{outcome} ~ C(yielding) * C(eHMIOn) * C(camera) + "
            "C(distPed_m) * (C(yielding) + C(camera))"
            f"{trial_terms}"
        )
        random_formulas = [
            "~C(yielding) + C(eHMIOn) + C(camera) + distPed_centered"
            f"{random_trial_term}",
            "~C(yielding) + C(eHMIOn) + C(camera)",
            "~C(yielding) + C(eHMIOn)",
            "~distPed_centered",
            None,
        ]
        fit = None
        model_name = None
        selected_re_formula = None
        for index, re_formula in enumerate(random_formulas, start=1):
            try:
                candidate = smf.mixedlm(
                    formula,
                    model_df,
                    groups=model_df["participant"],
                    re_formula=re_formula,
                ).fit(
                    reml=False,
                    method=["lbfgs", "bfgs", "cg"],
                    maxiter=500,
                    disp=False,
                )
                if not bool(getattr(candidate, "converged", False)):
                    raise RuntimeError("model did not converge")
                fit = candidate
                model_name = (
                    f"mixed_random_slopes_{index}"
                    if re_formula is not None
                    else "mixed_random_intercept_fallback"
                )
                selected_re_formula = re_formula
                break
            except Exception as exc:
                logger.warning(
                    "Mixed model attempt {} failed for {} (re_formula={}): {}",
                    index,
                    outcome,
                    re_formula,
                    exc,
                )

        if fit is None:
            try:
                fit = smf.ols(formula, data=model_df).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": model_df["participant"]},
                )
                model_name = "ols_clustered_fallback"
                selected_re_formula = None
                logger.warning(
                    "All mixed models failed for {}; used participant-clustered OLS.",
                    outcome,
                )
            except Exception as exc:
                logger.error(f"All improved models failed for {outcome}: {exc}")
                return None, None

        coef_df = pd.DataFrame({
            "term": fit.params.index,
            "estimate": fit.params.values,
            "std_error": fit.bse.values,
            "z_value": fit.tvalues.values,
            "p_value": fit.pvalues.values,
        })
        conf = fit.conf_int()
        coef_df["ci_lower"] = conf.iloc[:, 0].values
        coef_df["ci_upper"] = conf.iloc[:, 1].values
        coef_df["predictor"] = coef_df["term"].map(self._pretty_mixed_term)
        coef_df["model"] = model_name
        coef_df["formula"] = formula
        coef_df["random_effects_formula"] = selected_re_formula or "1"
        coef_df["converged"] = bool(getattr(fit, "converged", True))
        coef_df["analysis_version"] = analysis_version
        coef_df["ci_95"] = coef_df.apply(
            lambda row: f"[{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]",
            axis=1,
        )

        logger.info(f"\n=== Mixed model: {outcome} ===")
        logger.info(f"Formula: {formula}")
        logger.info(
            "{}",
            coef_df[["predictor", "estimate", "ci_lower", "ci_upper", "p_value"]].to_string(index=False),
        )
        logger.info("=======================\n")

        coef_df.to_csv(coefficients_path, index=False)

        term_df = coef_df[["term", "predictor", "estimate", "ci_lower", "ci_upper", "p_value"]].copy()
        term_df.insert(0, "outcome", outcome)
        term_df.to_csv(os.path.join(save_dir, f"mixed_model_terms_{outcome}.csv"), index=False)

        fixed_term_names = (
            set(fit.fe_params.index)
            if hasattr(fit, "fe_params")
            else set(fit.params.index)
        )
        manuscript_df = coef_df.loc[
            coef_df["term"].isin(fixed_term_names),
            ["predictor", "estimate", "ci_lower", "ci_upper", "ci_95", "p_value"],
        ].copy()
        manuscript_df.to_csv(os.path.join(save_dir, f"mixed_model_table_{outcome}.csv"), index=False)
        return fit, coef_df

    @staticmethod
    def load_trial_ratings(
        responses_root: str,
        n_participants: int = 50,
        response_col_index: int = 2,
    ) -> pd.DataFrame:
        """Load participant Q1/Q2/Q3 trial ratings from the raw response files."""
        q1_idx = response_col_index - 1
        q2_idx = response_col_index
        q3_idx = response_col_index + 1
        if q1_idx < 1:
            raise ValueError("response_col_index is too small to infer Q1/Q2/Q3")

        all_records = []
        for pid in range(1, n_participants + 1):
            participant_folder = os.path.join(
                responses_root,
                f"Participant_{pid}",
            )
            if not os.path.isdir(participant_folder):
                continue
            pattern = os.path.join(
                participant_folder,
                f"Participant_{pid}_*.csv",
            )
            for file_path in glob.glob(pattern):
                response_df = pd.read_csv(file_path, header=None)
                if response_df.shape[1] <= q3_idx:
                    continue
                trial_df = response_df[[0, q1_idx, q2_idx, q3_idx]].copy()
                trial_df.columns = ["video_id", "Q1", "Q2", "Q3"]  # pyright: ignore[reportAttributeAccessIssue]
                trial_df["participant"] = pid
                trial_df["video_id"] = trial_df["video_id"].astype(str)
                trial_df = trial_df[
                    trial_df["video_id"].str.startswith("video_")
                ].copy()
                # Rows were written by Unity in realised presentation order.
                # Preserve that order explicitly so learning and fatigue can be
                # tested without inferring order from video identifiers.
                trial_df["trial_number"] = np.arange(1, len(trial_df) + 1)
                all_records.append(trial_df)

        if not all_records:
            raise ValueError(
                "No participant response data found. "
                "Check responses_root and file patterns."
            )

        ratings_df = pd.concat(all_records, ignore_index=True)
        for q_col in ["Q1", "Q2", "Q3"]:
            ratings_df[q_col] = pd.to_numeric(ratings_df[q_col], errors="coerce")
        return ratings_df

    def load_and_average_Q2(
        self,
        trigger_summary_csv: str,
        responses_root: str,
        mapping_df: pd.DataFrame,
        n_participants: int = 50,
        response_col_index: int = 2,
        save_combined: bool = True,
        trigger_threshold: float = 0.05,
        trigger_matrices_dir: Optional[str] = None,
        ratings_df: Optional[pd.DataFrame] = None,
    ):
        """Merge trigger risk with cached or raw participant Q1/Q2/Q3 ratings."""
        trigger_df = pd.read_csv(trigger_summary_csv)
        if "label" in trigger_df.columns and "condition_name" not in trigger_df.columns:
            trigger_df = trigger_df.rename(columns={"label": "condition_name"})
        if "condition_name" not in trigger_df.columns or "avg_trigger" not in trigger_df.columns:
            raise ValueError(
                "trigger_summary_csv must contain condition_name or label, plus avg_trigger."
            )
        if "sd_trigger" not in trigger_df.columns:
            trigger_df["sd_trigger"] = np.nan
        trigger_df["condition_name"] = trigger_df["condition_name"].astype(str)

        map_cols = [
            "video_id", "condition_name", "distPed", "yielding", "eHMIOn", "camera",
            "cross_p1_time_s", "cross_p2_time_s"
        ]
        missing_map = [c for c in map_cols if c not in mapping_df.columns]
        if missing_map:
            raise ValueError(f"mapping_df missing required columns: {missing_map}")
        map_df = mapping_df[map_cols].copy()
        map_df["video_id"] = map_df["video_id"].astype(str)
        map_df["condition_name"] = map_df["condition_name"].astype(str)

        if ratings_df is None:
            ratings_df = self.load_trial_ratings(
                responses_root=responses_root,
                n_participants=n_participants,
                response_col_index=response_col_index,
            )
        else:
            ratings_df = ratings_df.copy()
            required_rating_columns = {"participant", "video_id", "Q1", "Q2", "Q3"}
            missing_ratings = sorted(required_rating_columns.difference(ratings_df.columns))
            if missing_ratings:
                raise ValueError(
                    f"ratings_df missing required columns: {missing_ratings}"
                )
            ratings_df["video_id"] = ratings_df["video_id"].astype(str)
        for q_col in ["Q1", "Q2", "Q3"]:
            ratings_df[q_col] = pd.to_numeric(ratings_df[q_col], errors="coerce")

        output_dir = os.path.dirname(trigger_summary_csv) or self.output_folder
        trigger_matrices_dir = trigger_matrices_dir or output_dir
        participant_trigger_df = self._compute_trial_level_trigger_summary(
            trigger_matrices_dir,
            map_df,
            trigger_threshold=trigger_threshold,
        )

        trial_df = ratings_df.merge(map_df, on="video_id", how="left")
        trial_df = trial_df.merge(
            participant_trigger_df[[
                "participant", "video_id", "avg_trigger", "sd_trigger", "n_trigger_samples",
                "n_trigger_bins", "n_raw_trigger_samples", "mean_trigger_intensity",
                "sd_trigger_intensity", "trigger_threshold"
            ]],
            on=["participant", "video_id"],
            how="left",
        )
        # fallback from condition-level trigger summary if a participant-level row is missing
        trial_df = trial_df.merge(
            trigger_df[["condition_name", "avg_trigger", "sd_trigger"]].rename(
                columns={"avg_trigger": "avg_trigger_condition", "sd_trigger": "sd_trigger_condition"}),  # type:ignore
            on="condition_name",
            how="left",
        )
        trial_df["avg_trigger"] = trial_df["avg_trigger"].fillna(trial_df["avg_trigger_condition"])
        trial_df["sd_trigger"] = trial_df["sd_trigger"].fillna(trial_df["sd_trigger_condition"])
        trial_df = trial_df.drop(columns=["avg_trigger_condition", "sd_trigger_condition"], errors="ignore")
        # Preserve the raw mapping code in distPed and store physical distance
        # separately. This prevents downstream code from mistaking 2 m or 4 m
        # for raw codes and multiplying them a second time.
        trial_df["distPed_m"] = self._distance_series_to_meters(trial_df["distPed"])
        # crossing_risk is the percentage of analysed participant-time bins for
        # which the trigger was pressed (> trigger_threshold), matching the paper.
        trial_df["crossing_risk"] = pd.to_numeric(trial_df["avg_trigger"], errors="coerce") * 100.0
        trial_df["crossing_risk_sd"] = pd.to_numeric(trial_df["sd_trigger"], errors="coerce") * 100.0

        condition_df = (
            trial_df
            .groupby("condition_name", as_index=False)
            .agg(
                avg_trigger=("avg_trigger", "mean"),
                std_trigger=("avg_trigger", "std"),
                mean_Q1=("Q1", "mean"),
                std_Q1=("Q1", "std"),
                mean_Q2=("Q2", "mean"),
                std_Q2=("Q2", "std"),
                mean_Q3=("Q3", "mean"),
                std_Q3=("Q3", "std"),
                n_trials=("Q2", "size"),
                mean_trigger_intensity=("mean_trigger_intensity", "mean"),
                trigger_threshold=("trigger_threshold", "first"),
            )
            .sort_values("condition_name")
            .reset_index(drop=True)
        )

        if save_combined:
            os.makedirs(output_dir, exist_ok=True)
            trial_path = os.path.join(output_dir, "trial_level_trigger_Q123.csv")
            cond_path = os.path.join(output_dir, "condition_level_trigger_Q123.csv")
            trial_df.to_csv(trial_path, index=False)
            condition_df.to_csv(cond_path, index=False)
            logger.info(f"Saved trial-level trigger + Q1/Q2/Q3 data to: {trial_path}")
            logger.info(f"Saved condition-level trigger + Q1/Q2/Q3 data to: {cond_path}")

        return trial_df, condition_df


    @staticmethod
    def _trigger_threshold_label(threshold: float) -> str:
        """Create a filesystem-safe label for a trigger threshold."""
        return f"threshold_{int(round(float(threshold) * 100)):03d}pct"

    def run_trigger_threshold_sensitivity(
        self,
        trigger_thresholds,
        trigger_matrices_dir: str,
        responses_root: str,
        mapping_df: pd.DataFrame,
        primary_threshold: float = 0.10,
        output_dir: Optional[str] = None,
        n_participants: int = 50,
        response_col_index: int = 2,
        ratings_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Run crossing-risk sensitivity checks for several trigger thresholds.

        The primary manuscript definition treats the pressure-sensitive trigger as
        a binary state: values greater than the threshold are coded as unsafe/risk
        state 1, and values at or below the threshold are coded as 0. This method
        repeats that calculation for multiple thresholds so the effect of the
        chosen tolerance can be inspected.

        Parameters
        ----------
        trigger_thresholds : iterable
            Trigger thresholds on the 0..1 scale, for example [0.05, 0.10, 0.50].
        trigger_matrices_dir : str
            Directory containing participant_TriggerValueRight_video_*.csv files.
        responses_root : str
            Directory containing Participant_* folders with trial-wise Q1/Q2/Q3 data.
        mapping_df : pd.DataFrame
            Scenario mapping table.
        primary_threshold : float
            Threshold used for the manuscript's primary analysis. Alternative
            thresholds are compared directly with this value.
        output_dir : str, optional
            Root directory where threshold-specific output folders are written.
        n_participants : int
            Maximum participant id to scan when reading trial-wise responses.
        response_col_index : int
            Index of Q2 in the participant response CSVs; Q1 and Q3 are inferred
            as the neighbouring columns, matching load_and_average_Q2.
        ratings_df : pd.DataFrame, optional
            Cached participant Q1/Q2/Q3 trial ratings. When supplied, raw
            participant response files are not read.

        Returns
        -------
        dict
            A dictionary with combined summary, model-term, and condition-level
            tables for all thresholds.
        """
        output_dir = output_dir or os.path.join(self.output_folder, "threshold_sensitivity")
        os.makedirs(output_dir, exist_ok=True)

        threshold_values = list(dict.fromkeys(float(x) for x in trigger_thresholds))
        primary_threshold = float(primary_threshold)
        if not any(np.isclose(value, primary_threshold) for value in threshold_values):
            raise ValueError("primary_threshold must be included in trigger_thresholds")
        summary_records = []
        model_tables = []
        condition_tables = []

        for threshold in threshold_values:
            threshold_label = self._trigger_threshold_label(threshold)
            threshold_dir = os.path.join(output_dir, threshold_label)
            stats_dir = os.path.join(threshold_dir, "statistics")
            os.makedirs(threshold_dir, exist_ok=True)
            os.makedirs(stats_dir, exist_ok=True)

            logger.info(
                "Running trigger-threshold sensitivity for {} ({:.2f}).",
                threshold_label,
                threshold,
            )

            participant_trigger_df = self._compute_trial_level_trigger_summary(
                trigger_matrices_dir,
                mapping_df,
                trigger_threshold=threshold,
            )
            participant_trigger_df.to_csv(
                os.path.join(threshold_dir, "participant_level_trigger_summary.csv"),
                index=False,
            )

            trigger_summary_df = (
                participant_trigger_df
                .groupby("condition_name", as_index=False)
                .agg(
                    avg_trigger=("avg_trigger", "mean"),
                    sd_trigger=("avg_trigger", "std"),
                    n_trials=("avg_trigger", "size"),
                    n_trigger_bins=("n_trigger_bins", "sum"),
                    n_raw_trigger_samples=("n_raw_trigger_samples", "sum"),
                    mean_trigger_intensity=("mean_trigger_intensity", "mean"),
                    sd_trigger_intensity=("sd_trigger_intensity", "mean"),
                    trigger_threshold=("trigger_threshold", "first"),
                )
                .sort_values("condition_name")
                .reset_index(drop=True)
            )
            trigger_summary_path = os.path.join(threshold_dir, "trigger_summary.csv")
            trigger_summary_df.to_csv(trigger_summary_path, index=False)

            trial_df, condition_df = self.load_and_average_Q2(
                trigger_summary_csv=trigger_summary_path,
                responses_root=responses_root,
                mapping_df=mapping_df,
                n_participants=n_participants,
                response_col_index=response_col_index,
                save_combined=True,
                trigger_threshold=threshold,
                trigger_matrices_dir=trigger_matrices_dir,
                ratings_df=ratings_df,
            )
            trial_df["threshold"] = threshold
            trial_df["threshold_label"] = threshold_label
            condition_df["threshold"] = threshold
            condition_df["threshold_label"] = threshold_label
            condition_tables.append(condition_df)

            risk = pd.to_numeric(trial_df["crossing_risk"], errors="coerce").dropna()
            summary_records.append({
                "threshold": threshold,
                "threshold_label": threshold_label,
                "n_trials": int(risk.size),
                "n_participants": int(trial_df["participant"].nunique()) if "participant" in trial_df.columns else np.nan,
                "mean_crossing_risk": float(risk.mean()) if not risk.empty else np.nan,
                "sd_crossing_risk": float(risk.std(ddof=1)) if risk.size > 1 else np.nan,
                "median_crossing_risk": float(risk.median()) if not risk.empty else np.nan,
                "min_crossing_risk": float(risk.min()) if not risk.empty else np.nan,
                "max_crossing_risk": float(risk.max()) if not risk.empty else np.nan,
                "p05_crossing_risk": float(risk.quantile(0.05)) if not risk.empty else np.nan,
                "p95_crossing_risk": float(risk.quantile(0.95)) if not risk.empty else np.nan,
                "zero_risk_trial_pct": float((risk == 0).mean() * 100.0) if not risk.empty else np.nan,
            })

            try:
                _, coef_df = self._run_mixed_effects_model(trial_df, "crossing_risk", stats_dir)
                coef_df = coef_df.copy()
                coef_df["threshold"] = threshold
                coef_df["threshold_label"] = threshold_label
                model_tables.append(coef_df)
            except Exception as exc:
                logger.warning(
                    "Mixed-effects model failed for {}: {}",
                    threshold_label,
                    exc,
                )

        summary_df = pd.DataFrame(summary_records)
        model_terms_df = pd.concat(model_tables, ignore_index=True) if model_tables else pd.DataFrame()
        condition_sensitivity_df = (
            pd.concat(condition_tables, ignore_index=True) if condition_tables else pd.DataFrame()
        )

        # Provide an auditable comparison with the primary 0.10 analysis. The
        # correlations describe stability of condition patterns, while the
        # model fields show whether effect directions and significance decisions
        # are retained. These metrics are descriptive; the researcher should
        # inspect the accompanying long-form tables before claiming robustness.
        robustness_records = []
        if not condition_sensitivity_df.empty:
            primary_condition = condition_sensitivity_df.loc[
                np.isclose(condition_sensitivity_df["threshold"], primary_threshold),
                ["condition_name", "avg_trigger"],
            ].rename(columns={"avg_trigger": "avg_trigger_primary"})

            primary_model = pd.DataFrame()
            if not model_terms_df.empty and {"term", "estimate", "p_value", "threshold"}.issubset(model_terms_df.columns):
                primary_model = model_terms_df.loc[
                    np.isclose(model_terms_df["threshold"], primary_threshold),
                    ["term", "estimate", "p_value"],
                ].drop_duplicates(subset=["term"])

            for threshold in threshold_values:
                current_condition = condition_sensitivity_df.loc[
                    np.isclose(condition_sensitivity_df["threshold"], threshold),
                    ["condition_name", "avg_trigger"],
                ].rename(columns={"avg_trigger": "avg_trigger_current"})
                condition_compare = primary_condition.merge(
                    current_condition,
                    on="condition_name",
                    how="inner",
                ).dropna()

                pearson = np.nan
                spearman = np.nan
                mean_abs_difference_points = np.nan
                max_abs_difference_points = np.nan
                if len(condition_compare) >= 2:
                    pearson = condition_compare["avg_trigger_primary"].corr(
                        condition_compare["avg_trigger_current"], method="pearson"
                    )
                    spearman = condition_compare["avg_trigger_primary"].corr(
                        condition_compare["avg_trigger_current"], method="spearman"
                    )
                    difference_points = (
                        condition_compare["avg_trigger_current"]
                        - condition_compare["avg_trigger_primary"]
                    ).abs() * 100.0
                    mean_abs_difference_points = float(difference_points.mean())
                    max_abs_difference_points = float(difference_points.max())

                model_sign_agreement_pct = np.nan
                model_significance_agreement_pct = np.nan
                n_model_terms = 0
                if not primary_model.empty:
                    current_model = model_terms_df.loc[
                        np.isclose(model_terms_df["threshold"], threshold),
                        ["term", "estimate", "p_value"],
                    ].drop_duplicates(subset=["term"])
                    model_compare = primary_model.merge(
                        current_model,
                        on="term",
                        suffixes=("_primary", "_current"),
                        how="inner",
                    )
                    model_compare = model_compare.loc[
                        model_compare["term"] != "Group Var"
                    ].dropna(subset=["estimate_primary", "estimate_current"])
                    n_model_terms = int(len(model_compare))
                    if n_model_terms:
                        model_sign_agreement_pct = float(
                            (
                                np.sign(model_compare["estimate_primary"])
                                == np.sign(model_compare["estimate_current"])
                            ).mean() * 100.0
                        )
                        valid_p = model_compare.dropna(
                            subset=["p_value_primary", "p_value_current"]
                        )
                        if not valid_p.empty:
                            model_significance_agreement_pct = float(
                                (
                                    (valid_p["p_value_primary"] < 0.05)
                                    == (valid_p["p_value_current"] < 0.05)
                                ).mean() * 100.0
                            )

                robustness_records.append({
                    "primary_threshold": primary_threshold,
                    "threshold": threshold,
                    "is_primary": bool(np.isclose(threshold, primary_threshold)),
                    "n_conditions": int(len(condition_compare)),
                    "condition_pearson_r": pearson,
                    "condition_spearman_rho": spearman,
                    "condition_mean_abs_difference_points": mean_abs_difference_points,
                    "condition_max_abs_difference_points": max_abs_difference_points,
                    "n_model_terms": n_model_terms,
                    "model_sign_agreement_pct": model_sign_agreement_pct,
                    "model_significance_agreement_pct": model_significance_agreement_pct,
                })

        robustness_df = pd.DataFrame(robustness_records)

        summary_path = os.path.join(output_dir, "threshold_sensitivity_summary.csv")
        model_path = os.path.join(output_dir, "threshold_sensitivity_model_terms.csv")
        condition_path = os.path.join(output_dir, "threshold_sensitivity_condition_means.csv")
        robustness_path = os.path.join(output_dir, "threshold_sensitivity_robustness.csv")

        summary_df.to_csv(summary_path, index=False)
        model_terms_df.to_csv(model_path, index=False)
        condition_sensitivity_df.to_csv(condition_path, index=False)
        robustness_df.to_csv(robustness_path, index=False)

        logger.info(f"Saved threshold sensitivity summary to: {summary_path}")
        logger.info(f"Saved threshold sensitivity model terms to: {model_path}")
        logger.info(f"Saved threshold sensitivity condition means to: {condition_path}")
        logger.info(f"Saved threshold sensitivity robustness checks to: {robustness_path}")

        return {
            "summary": summary_df,
            "model_terms": model_terms_df,
            "condition_means": condition_sensitivity_df,
            "robustness": robustness_df,
        }

    def analyze_and_plot_distance_effect_plotly(self, mapping_df, condition_df, out_dir=None, trial_df=None):
        """
        Merge condition-level averages with distance, yielding, eHMI, and camera,
        compute full-factorial summaries (means + SDs), and create Plotly figures.

        Assumes
        -------
        condition_df has at least:
            ['condition_name', 'avg_trigger', 'std_trigger',
             'mean_Q1', 'std_Q1',
             'mean_Q2', 'std_Q2',
             'mean_Q3', 'std_Q3']
            - avg_trigger: mean proportion of analysed time bins where the trigger was pressed.
            - std_trigger: SD of that thresholded trigger-based measure per condition.
            - mean_Q1/Q2/Q3: mean responses per condition (0–100).
            - std_Q1/Q2/Q3: SD of Q1/Q2/Q3 per condition.

        mapping_df has at least:
            ['condition_name', 'distPed', 'yielding', 'eHMIOn', 'camera']
        """

        # Helper: turn "var=value" facet titles into just "value"
        facet_font = dict(size=font_size, family=font_family)

        def _strip_facet_equals(fig):
            fig.for_each_annotation(
                lambda a: a.update(
                    text=a.text.split("=", 1)[-1].strip(),
                    font=facet_font,
                )
            )

        # --- REQUIRED columns ---
        required_cols = ["condition_name", "distPed", "yielding", "eHMIOn", "camera"]
        missing = [c for c in required_cols if c not in mapping_df.columns]
        if missing:
            raise ValueError(f"mapping_df missing: {missing}")

        needed_cond_cols = [
            "condition_name",
            "avg_trigger", "std_trigger",
            "mean_Q1", "std_Q1",
            "mean_Q2", "std_Q2",
            "mean_Q3", "std_Q3",
        ]
        missing2 = [c for c in needed_cond_cols if c not in condition_df.columns]
        if missing2:
            raise ValueError(f"condition_df missing: {missing2}")

        # --- Prepare mapping info ---
        mapping_df = mapping_df.copy()
        mapping_df["condition_name"] = mapping_df["condition_name"].astype(str)

        cond_plot_df = condition_df.copy()
        cond_plot_df["condition_name"] = cond_plot_df["condition_name"].astype(str)

        # Merge mapping info onto condition-level data
        cond_plot_df = cond_plot_df.merge(
            mapping_df[required_cols],
            on="condition_name",
            how="left",
        )

        # Convert the raw mapping codes once to actual metres (2, 4, 6, 8, 10).
        cond_plot_df["distPed_m"] = self._distance_series_to_meters(cond_plot_df["distPed"])

        # Scale thresholded unsafe-time proportion to 0–100.
        cond_plot_df["crossing_risk"] = cond_plot_df["avg_trigger"] * 100.0
        cond_plot_df["crossing_risk_sd"] = cond_plot_df["std_trigger"] * 100.0

        # Drop if some factors missing
        cond_plot_df = cond_plot_df.dropna(
            subset=["distPed_m", "yielding", "eHMIOn", "camera"]
        )

        # Label maps for binary factors (0/1 → text)
        label_map_yield = {0: "Non-yielding", 1: "Yielding"}
        label_map_ehmi = {0: "No eHMI", 1: "eHMI"}
        label_map_cam = {
            0: "Avatar first / participant second",
            1: "Participant first / avatar second",
        }

        cond_plot_df["yielding_label"] = cond_plot_df["yielding"].map(label_map_yield)
        cond_plot_df["eHMI_label"] = cond_plot_df["eHMIOn"].map(label_map_ehmi)
        cond_plot_df["camera_label"] = cond_plot_df["camera"].map(label_map_cam)

        # ============================
        # Full-factorial summary (means + SD from condition_df)
        # ============================
        group_cols = ["distPed_m", "yielding", "eHMIOn", "camera"]

        # If there are multiple rows per combination (e.g., multiple videos),
        # they should have identical stats; we take the mean as a safe aggregator.
        by_cond = (
            cond_plot_df
            .groupby(group_cols, as_index=False)
            .agg(
                mean_crossing_risk=("crossing_risk", "mean"),
                sd_crossing_risk=("crossing_risk_sd", "mean"),

                Q1_mean=("mean_Q1", "mean"),
                Q1_sd=("std_Q1", "mean"),

                Q2_mean=("mean_Q2", "mean"),
                Q2_sd=("std_Q2", "mean"),

                Q3_mean=("mean_Q3", "mean"),
                Q3_sd=("std_Q3", "mean"),
            )
            .sort_values(group_cols)
        )

        # Figure uncertainty must come from participant-level trials, not from
        # variation among condition means. Add t-based 95% CIs when those data
        # are available.
        by_cond["ci95_half_crossing_risk"] = np.nan
        by_cond["n_participants_crossing_risk"] = np.nan
        if trial_df is not None and not trial_df.empty:
            participant_trials = trial_df.copy()
            if "distPed_m" not in participant_trials.columns:
                participant_trials["distPed_m"] = self._distance_series_to_meters(
                    participant_trials["distPed"]
                )
            participant_trials["crossing_risk"] = pd.to_numeric(
                participant_trials["crossing_risk"], errors="coerce"
            )
            participant_summary = (
                participant_trials.dropna(subset=group_cols + ["participant", "crossing_risk"])
                .groupby(group_cols, as_index=False)["crossing_risk"]
                .agg(
                    mean_crossing_risk_participant="mean",
                    sd_crossing_risk_participant="std",
                    n_participants_crossing_risk="count",
                )
            )
            participant_summary["se_crossing_risk"] = (
                participant_summary["sd_crossing_risk_participant"]
                / np.sqrt(participant_summary["n_participants_crossing_risk"])
            )
            participant_summary["ci95_half_crossing_risk"] = participant_summary.apply(
                lambda row: (
                    t.ppf(0.975, int(row["n_participants_crossing_risk"]) - 1)
                    * row["se_crossing_risk"]
                    if int(row["n_participants_crossing_risk"]) > 1
                    else np.nan
                ),
                axis=1,
            )
            by_cond = by_cond.drop(
                columns=["ci95_half_crossing_risk", "n_participants_crossing_risk"]
            ).merge(participant_summary, on=group_cols, how="left")
            by_cond["mean_crossing_risk"] = by_cond[
                "mean_crossing_risk_participant"
            ].combine_first(by_cond["mean_crossing_risk"])

        # Add label columns for plotting facets
        by_cond["yielding_label"] = by_cond["yielding"].map(label_map_yield)
        by_cond["eHMI_label"] = by_cond["eHMIOn"].map(label_map_ehmi)
        by_cond["camera_label"] = by_cond["camera"].map(label_map_cam)

        logger.info("\n=== Full-factorial condition table (MEAN + SD, 0–100 scales) ===")
        logger.info(
            "\n=== Full-factorial condition table (MEAN + SD, 0–100 scales) ===\n{}",
            by_cond.to_string(index=False),
        )
        logger.info("===========================================================\n")

        # Common label mapping for all figures
        base_labels = {
            "distPed_m": "Distance between pedestrians (m)",
            "crossing_risk": "Perceived-unsafety time (%)",
            "mean_crossing_risk": "Perceived-unsafety time (%)",
            "sd_crossing_risk": "SD of perceived-unsafety time (%)",

            "Q1_mean": "Q1 (0–100)",
            "Q1_sd": "SD of Q1 (0–100)",
            "Q2_mean": "Q2 (0–100)",
            "Q2_sd": "SD of Q2 (0–100)",
            "Q3_mean": "Q3 (0–100)",
            "Q3_sd": "SD of Q3 (0–100)",

            "camera_label": "Relative pedestrian order",
            "yielding_label": "AV behaviour",
            "eHMI_label": "Conditional eHMI logic",
            "context": "Context (AV behaviour, conditional eHMI logic, relative pedestrian order)",
            "delta": "Near–far difference (0–100)",
            "measure": "Measure",
        }

        # Category ordering for cleaner facets / legends
        category_orders = {
            "eHMI_label": ["No eHMI", "eHMI"],
            "yielding_label": ["Non-yielding", "Yielding"],
            "camera_label": [
                "Avatar first / participant second",
                "Participant first / avatar second",
            ],
        }
        axis_title_font = dict(size=font_size, family=font_family)

        # ============================
        # Figure 1 — Mean crossing risk vs Distance (legend = camera)
        # ============================
        fig_beh = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="camera_label",
            facet_col="eHMI_label",
            facet_row="yielding_label",
            markers=True,
            error_y="ci95_half_crossing_risk",
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_beh.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_beh.update_xaxes(title_font=axis_title_font)
        fig_beh.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_beh)

        # ============================
        # Figure 2 — Q2 vs Distance (legend = camera)
        # ============================
        fig_q2 = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="camera_label",
            facet_col="eHMI_label",
            facet_row="yielding_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_q2.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.9,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_q2.update_xaxes(title_font=axis_title_font)
        fig_q2.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_q2)

        # ============================
        # EXTRA Figure A — Mean crossing risk vs distance, legend = yielding
        # ============================
        fig_beh_yield = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="yielding_label",
            facet_col="eHMI_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_beh_yield.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_beh_yield.update_xaxes(title_font=axis_title_font)
        fig_beh_yield.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_beh_yield)

        # ============================
        # EXTRA Figure B — Mean crossing risk vs distance, legend = eHMI
        # ============================
        fig_beh_ehmi = px.line(
            by_cond,
            x="distPed_m",
            y="mean_crossing_risk",
            color="eHMI_label",
            facet_col="yielding_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_beh_ehmi.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_beh_ehmi.update_xaxes(title_font=axis_title_font)
        fig_beh_ehmi.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_beh_ehmi)

        # ============================
        # EXTRA Figure C — Q2 vs distance, legend = yielding
        # ============================
        fig_q2_yield = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="yielding_label",
            facet_col="eHMI_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_q2_yield.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_q2_yield.update_xaxes(title_font=axis_title_font)
        fig_q2_yield.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_q2_yield)

        # ============================
        # EXTRA Figure D — Q2 vs distance, legend = eHMI
        # ============================
        fig_q2_ehmi = px.line(
            by_cond,
            x="distPed_m",
            y="Q2_mean",
            color="eHMI_label",
            facet_col="yielding_label",
            facet_row="camera_label",
            markers=True,
            labels=base_labels,
            category_orders=category_orders,
            title="",
        )

        fig_q2_ehmi.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
                title_text="",
            ),
        )
        fig_q2_ehmi.update_xaxes(title_font=axis_title_font)
        fig_q2_ehmi.update_yaxes(title_font=axis_title_font)
        _strip_facet_equals(fig_q2_ehmi)

        # ============================
        # Figure 3 — Mean crossing risk vs Q2 scatter (condition-level)
        # ============================
        fig_scatter = px.scatter(
            cond_plot_df,
            x="crossing_risk",
            y="mean_Q2",
            color="distPed_m",
            labels=base_labels,
            title="",
        )

        x_vals = cond_plot_df["crossing_risk"].values
        y_vals = cond_plot_df["mean_Q2"].values
        if len(x_vals) >= 2 and np.isfinite(x_vals).all() and np.isfinite(y_vals).all():
            b1, b0 = np.polyfit(x_vals, y_vals, 1)
            xs = np.linspace(x_vals.min(), x_vals.max(), 100)
            ys = b0 + b1 * xs
            fig_scatter.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name="",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig_scatter.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
            ),
        )

        # ============================
        # Figure 4 — NEAR (2–4 m) minus FAR (8–10 m) per context
        # ============================
        ctx_cols = ["yielding", "eHMIOn", "camera"]

        near = (
            by_cond[by_cond["distPed_m"].isin([2, 4])]
            .groupby(ctx_cols, as_index=False)
            .agg(
                crossing_risk_near=("mean_crossing_risk", "mean"),
                Q1_near=("Q1_mean", "mean"),
                Q2_near=("Q2_mean", "mean"),
                Q3_near=("Q3_mean", "mean"),
            )
        )
        far = (
            by_cond[by_cond["distPed_m"].isin([8, 10])]
            .groupby(ctx_cols, as_index=False)
            .agg(
                crossing_risk_far=("mean_crossing_risk", "mean"),
                Q1_far=("Q1_mean", "mean"),
                Q2_far=("Q2_mean", "mean"),
                Q3_far=("Q3_mean", "mean"),
            )
        )

        diff_df = near.merge(far, on=ctx_cols, how="inner")
        diff_df["delta_crossing_risk"] = (
            diff_df["crossing_risk_near"] - diff_df["crossing_risk_far"]
        )
        diff_df["delta_Q1"] = diff_df["Q1_near"] - diff_df["Q1_far"]
        diff_df["delta_Q2"] = diff_df["Q2_near"] - diff_df["Q2_far"]
        diff_df["delta_Q3"] = diff_df["Q3_near"] - diff_df["Q3_far"]

        # Context string with on/off text instead of 0/1
        diff_df["context"] = diff_df.apply(
            lambda r: (
                f"{'Yielding' if r['yielding'] == 1 else 'Non-yielding'}, "
                f"eHMI {'on' if r['eHMIOn'] == 1 else 'off'}, "
                f"{'avatar first / participant second' if int(r['camera']) == 0 else 'participant first / avatar second'}"
            ),
            axis=1,
        )

        logger.info("\n=== NEAR–FAR differences per context ===")
        logger.info(
            "\n=== NEAR–FAR differences per context ===\n{}",
            diff_df[
                ["context", "delta_crossing_risk", "delta_Q1", "delta_Q2", "delta_Q3"]
            ].to_string(index=False),
        )
        logger.info("========================================\n")

        long_diff = diff_df.melt(
            id_vars=["context"],
            value_vars=["delta_crossing_risk", "delta_Q1", "delta_Q2", "delta_Q3"],
            var_name="measure",
            value_name="delta",
        )

        long_diff["measure"] = long_diff["measure"].map({
            "delta_crossing_risk": "Perceived-unsafety time (%)",
            "delta_Q1": "Q1 (0–100)",
            "delta_Q2": "Q2 (0–100)",
            "delta_Q3": "Q3 (0–100)",
        })

        fig_diff = px.bar(
            long_diff,
            x="context",
            y="delta",
            color="measure",
            barmode="group",
            labels={**base_labels, "delta": "Near–far difference (0–100)"},
            title="",
        )

        fig_diff.update_traces(
            texttemplate="%{y:.1f}",
            textposition="outside",
            textfont=axis_title_font,
        )

        fig_diff.update_layout(
            template=plotly_template,
            legend=dict(
                x=0.88,
                y=0.85,
                xanchor="center",
                yanchor="bottom",
                font=axis_title_font,
            ),
        )
        fig_diff.update_xaxes(title_font=axis_title_font, tickfont=axis_title_font)
        fig_diff.update_yaxes(title_font=axis_title_font, tickfont=axis_title_font)
        fig_diff.add_hline(y=0, line_dash="dash", line_color="black")

        # ============================
        # Stats summary
        # ============================
        logger.info(
            "Naive trial-level and condition-mean correlations are intentionally "
            "omitted; the within/between-participant models provide the clustered "
            "association analysis."
        )

        xd = by_cond["distPed_m"].values

        # Slopes vs distance for crossing risk and Q1–Q3
        yd_risk = by_cond["mean_crossing_risk"].values
        slope_risk, intercept_risk = np.polyfit(xd, yd_risk, 1)
        logger.info(
            "Overall perceived-unsafety time vs distance: "
            f"slope = {slope_risk:.4f} (risk units per 1 m)"
        )

        for q_label, col in [("Q1", "Q1_mean"), ("Q2", "Q2_mean"), ("Q3", "Q3_mean")]:
            y = by_cond[col].values
            slope_q, intercept_q = np.polyfit(xd, y, 1)
            logger.info(
                f"Overall {q_label} vs distance: "
                f"slope = {slope_q:.4f} ({q_label} units per 1 m)"
            )

        # ============================
        # Stats summary
        # ============================
        stats_out_dir = out_dir or self.output_folder
        os.makedirs(stats_out_dir, exist_ok=True)

        near_far_path = os.path.join(stats_out_dir, "near_far_differences.csv")
        diff_df.to_csv(near_far_path, index=False)

        by_cond.to_csv(os.path.join(stats_out_dir, "distance_effect_cell_summary.csv"), index=False)
        by_cond.to_csv(os.path.join(stats_out_dir, "table_descriptives_full_factorial.csv"), index=False)

        if trial_df is not None and not trial_df.empty:
            self._run_mixed_effects_model(trial_df, "crossing_risk", stats_out_dir)
            self._run_mixed_effects_model(trial_df, "Q1", stats_out_dir)
            self._run_mixed_effects_model(trial_df, "Q2", stats_out_dir)
            self._run_mixed_effects_model(trial_df, "Q3", stats_out_dir)

        # ============================
        # Save figures
        # ============================
        self.save_plotly(fig_beh, "crossing_risk_full_factorial", save_final=True)
        self.save_plotly(fig_q2, "Q2_full_factorial", save_final=True)
        self.save_plotly(fig_diff, "near_minus_far_crossing_risk_vs_Q123", save_final=True)
        self.save_plotly(fig_beh_yield, "crossing_risk_full_factorial_legend_yielding", save_final=True)
        self.save_plotly(fig_beh_ehmi, "crossing_risk_full_factorial_legend_eHMI", save_final=True)
        self.save_plotly(fig_q2_yield, "Q2_full_factorial_legend_yielding", save_final=True)
        self.save_plotly(fig_q2_ehmi, "Q2_full_factorial_legend_eHMI", save_final=True)

        return by_cond, cond_plot_df, diff_df

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
