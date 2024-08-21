import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio  # noqa:F401
import plotly.express as px  # noqa:F401
from plotly.subplots import make_subplots  # noqa:F401
import math
import glob
import shutil  # Added for moving files
import common
from custom_logger import CustomLogger
import re

logger = CustomLogger(__name__)  # use custom logger


# Todo: Make a code to move all the csv files in multiped-->readings--><participant_no>
class HMD_helper:
    def __init__(self):
        pass

    @staticmethod
    def quaternion_to_euler(w, x, y, z):
        """
        Convert a quaternion into Euler angles (roll, pitch, yaw)
        Roll is rotation around x-axis, pitch is rotation around y-axis, and yaw is rotation around z-axis.
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def check_participant_file_exists(directory_path):
        # Construct the search pattern
        search_pattern = os.path.join(directory_path, "pilot*.csv")
        matching_files = glob.glob(search_pattern)

        if matching_files:
            print(f"Files found: {matching_files}")
            participant_file = matching_files[0]
            # Extract the file name without the extension
            full_file_name = os.path.splitext(os.path.basename(participant_file))[0]
            # Remove the timestamp
            participant_no = full_file_name[:-15]
            return participant_no
        else:
            print("No file starting with 'participant' found.")

    def move_csv_files(self, participant_no, mapping):
        # Get the readings directory and create a folder inside it named after the participant_no
        readings_folder = common.get_configs("readings")
        participant_folder = os.path.join(readings_folder, str(participant_no))

        # Check if the participant folder exists; if not, create it
        if not os.path.exists(participant_folder):
            os.makedirs(
                participant_folder
            )  # Use makedirs to ensure all directories are created
            print(f"Folder '{participant_folder}' created.")
        else:
            print(f"Folder '{participant_folder}' already exists.")

        # Data folder where CSV files are originally located
        data_folder_path = common.get_configs("data")

        # Move the specific "participants...csv" file
        participant_file_pattern = os.path.join(
            data_folder_path, f"{participant_no}*.csv"
        )
        participant_files = glob.glob(participant_file_pattern)

        if participant_files:
            for participant_file in participant_files:
                dest_file = os.path.join(
                    participant_folder, os.path.basename(participant_file)
                )
                shutil.move(participant_file, dest_file)
                print(f"Moved '{participant_file}' to '{dest_file}'.")
        else:
            print(f"No file matching '{participant_no}*.csv' found.")

        # Iterate over video IDs in the mapping to move corresponding CSV files
        for video_id in mapping["video_id"]:
            src_file = os.path.join(data_folder_path, f"{video_id}.csv")
            dest_file = os.path.join(participant_folder, f"{video_id}.csv")

            if os.path.exists(src_file):
                shutil.move(src_file, dest_file)
                print(f"Moved '{src_file}' to '{dest_file}'.")
            else:
                print(f"File '{src_file}' does not exist.")

    @staticmethod
    def delete_unnecessary_meta_files(participant_no, mapping):
        pass

    @staticmethod
    def plot_mean_trigger_value_right(readings_folder, mapping):
        participants_folders = [
            f
            for f in os.listdir(readings_folder)
            if os.path.isdir(os.path.join(readings_folder, f))
        ]

        # Get all the csv files starting with "video_" and ending with ".csv"
        sample_folder = os.path.join(readings_folder, participants_folders[0])
        csv_files = [
            f
            for f in os.listdir(sample_folder)
            if f.startswith("video_") and f.endswith(".csv")
        ]

        # Sort files by the number in their names to ensure correct ordering
        csv_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

        # Group files in sets of 5 (1-5, 6-10, etc.)
        groups = [csv_files[i: i + 5] for i in range(0, len(csv_files), 5)]

        for group in groups:
            # Initialize the figure for each group
            fig = go.Figure()

            # Loop through each file in the group
            for csv_file in group:
                dfs = []
                for participant in participants_folders:
                    file_path = os.path.join(readings_folder, participant, csv_file)
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)

                        # Extract video number from the filename to match with mapping
                        video_number = re.findall(r"\d+", csv_file)[0]
                        video_id = f"video_{video_number}"

                        # Get the corresponding video_length from the mapping
                        video_length = mapping.loc[
                            mapping["video_id"] == video_id, "video_length"
                        ].values[0]
                        video_length = video_length / 1000

                        # Filter the DataFrame to only include rows where Timestamp >= 0 and < video_length
                        df = df[
                            (df["Timestamp"] >= 0) & (df["Timestamp"] < video_length)
                        ]

                        # Keep only the 'Timestamp' and 'TriggerValueRight' columns
                        df = df[["Timestamp", "TriggerValueRight"]]
                        dfs.append(df)

                if dfs:
                    combined_df = pd.concat(dfs)
                    mean_df = combined_df.groupby("Timestamp").mean().reset_index()

                    # Add a trace for the 'TriggerValueRight' data for this file
                    fig.add_trace(
                        go.Scatter(
                            x=mean_df["Timestamp"],
                            y=mean_df["TriggerValueRight"],
                            mode="lines",
                            name=csv_file,  # Use the file name as the label in the legend
                        )
                    )

            # Update layout to add titles, axes labels, and show the legend
            fig.update_layout(
                title=f"Mean TriggerValueRight for {', '.join(group)}",
                xaxis_title="Timestamp",
                yaxis_title="Mean TriggerValueRight",
                showlegend=True,  # Ensure legend is displayed
                )

            # Show the figure for the current group
            fig.show()
