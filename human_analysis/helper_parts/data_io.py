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


class DataIOMixin:
    """Focused method group extracted without changing calculation logic."""

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
