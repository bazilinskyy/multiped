import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio  # noqa:F401
import plotly.express as px  # noqa:F401
from plotly.subplots import make_subplots
import math
import glob
import shutil  # Added for moving files
import common
from custom_logger import CustomLogger
import re
import numpy as np

logger = CustomLogger(__name__)  # use custom logger
template = common.get_configs("plotly_template")


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
        search_pattern = os.path.join(directory_path, "Participant*.csv")
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
    def plot_mean_trigger_value_right(readings_folder, mapping, output_folder, group_titles=None, legend_labels=None):
        dataframes_dict = {}
        participants_folders = [
            f for f in os.listdir(readings_folder)
            if os.path.isdir(os.path.join(readings_folder, f))
        ]

        # Dictionary to accumulate csv files found in participant folders
        csv_files_dict = {}

        # Search all participant folders for csv files
        for participant in participants_folders:
            sample_folder = os.path.join(readings_folder, participant)
            csv_files = [
                f for f in os.listdir(sample_folder)
                if f.startswith("video_") and f.endswith(".csv")
            ]
            # Add files to the dictionary to track their presence across participants
            for csv_file in csv_files:
                if csv_file not in csv_files_dict:
                    csv_files_dict[f"{participant}_{csv_file}"] = []
                csv_files_dict[f"{participant}_{csv_file}"].append(os.path.join(sample_folder, csv_file))

        # Initialize an empty dictionary to store groups
        grouped_data = {}

        # Loop through each key in the dictionary
        for key, value in sorted(csv_files_dict.items(), key=lambda x: int(x[0].split('_video_')[1].split('.')[0])):
            # Extract the video number from the key
            video_number = key.split('_video_')[1].split('.')[0]

            # Add the key-value pair to the appropriate group
            if video_number not in grouped_data:
                grouped_data[video_number] = []
            grouped_data[video_number].append({key: value})

        grouped_keys_list = [[list(item.keys())[0] for item in values] for values in grouped_data.values()]

        for jdx, csv_files in enumerate(grouped_keys_list):
            # Loop through each csv file in the current group
            for csv_file in csv_files:
                csv_file_ = "_".join(csv_file.split("_")[-2:])
                participant_ = "_".join(csv_file.split("_")[:2])
                file_path = os.path.join(readings_folder, participant_, csv_file_)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)[["Timestamp", "TriggerValueRight"]]

                    # Get the corresponding video_length from the mapping
                    video_length = mapping.loc[mapping["video_id"] == csv_file_.rstrip('.csv'),
                                               "video_length"].values[0]
                    video_length = video_length / 1000

                    # Filter the DataFrame to only include rows where Timestamp >= 0 and <= video_length
                    df = df[(df["Timestamp"] >= 0) & (df["Timestamp"] <= video_length)]

                    # Add the DataFrame to the dictionary; create a list if it doesn't exist
                    if csv_file_ in dataframes_dict:
                        dataframes_dict[csv_file_].append(df)
                    else:
                        dataframes_dict[csv_file_] = [df]

        # Merge DataFrames and calculate the average if needed
        merged_dataframes = {}
        for csv_file_, df_list in dataframes_dict.items():
            if len(df_list) > 1:
                # Concatenate and then average if there are multiple DataFrames
                merged_df = pd.concat(df_list).groupby(level=0).mean()
            else:
                # If there's only one DataFrame, use it directly
                merged_df = df_list[0]

            merged_dataframes[csv_file_] = merged_df

        # Get the total number of plots needed (each containing 5 curves)
        num_plots = len(merged_dataframes) // 5 + (1 if len(merged_dataframes) % 5 != 0 else 0)

        # Iterate over the groups of 5
        for plot_index in range(num_plots):
            # Create a new subplot for each group of 5
            fig = go.Figure()

            # Determine the start and end indices for this group
            start_idx = plot_index * 5
            end_idx = min(start_idx + 5, len(merged_dataframes))

            # Add traces for this group
            for i, (csv_file_, merged_df) in enumerate(list(merged_dataframes.items())[start_idx:end_idx]):
                fig.add_trace(
                    go.Scatter(
                        x=merged_df["Timestamp"],
                        y=merged_df["TriggerValueRight"],
                        mode='lines',
                        name=legend_labels[i],
                        line=dict(width=3)  # Increase line thickness
                    )
                )

            # Update layout for each subplot
            fig.update_layout(
                title={
                    'text': group_titles[plot_index],
                    'x': 0.5,  # Center the title
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20, 'weight': 'bold'}  # Make the title larger and bold
                },
                xaxis_title={'text': 'Time (in seconds)', 'font': {'size': 20}},  # Increase x-axis label size
                yaxis_title={'text': 'Trigger press', 'font': {'size': 20}},  # Increase y-axis label size
                legend_title=None,
                template='plotly',
                xaxis=dict(tickfont=dict(size=20)),  # Increase x-axis tick size
                yaxis=dict(tickfont=dict(size=20))   # Increase y-axis tick size
            )

            fig.update_layout(
                legend=dict(x=0.113, y=0.986, traceorder="normal", font=dict(size=24)))

            # Save the plot
            base_filename = f"group_{plot_index + 1}_trigger"
            fig.write_image(os.path.join(output_folder, base_filename + ".eps"), width=1600, height=900, scale=3)
            fig.write_image(os.path.join(output_folder, base_filename + ".png"), width=1600, height=900, scale=3)
            fig.write_html(os.path.join(output_folder, base_filename + ".html"))
            fig.write_image(os.path.join(output_folder, base_filename + ".svg"),
                            width=1600, height=900, scale=3, format="svg")

            # Show the plot
            fig.show()

    @staticmethod
    def sort_and_group_videos(grouped_videos):
        # Initialize a list to store the new grouped videos
        new_grouped_videos = []

        for group in grouped_videos:
            # Sort video names numerically by extracting numbers from the video IDs
            sorted_group = sorted(group, key=lambda x: int(re.findall(r'\d+', x)[0]))

            # Split the sorted group into subgroups of 5
            for i in range(0, len(sorted_group), 5):
                new_grouped_videos.append(sorted_group[i:i+5])

        return new_grouped_videos

    @staticmethod
    def radar_plot(readings_folder, mapping, output_folder):
        final_dict = {}  # dictionary to accumate the mean value of the slider bars
        participants_folders = [
            f for f in os.listdir(readings_folder)
            if os.path.isdir(os.path.join(readings_folder, f))
        ]
        # Dictionary to accumulate csv files found in participant folders
        csv_files_dict = {}

        # Search all participant folders for csv files
        for participant in participants_folders:
            sample_folder = os.path.join(readings_folder, participant)
            csv_files = [
                f for f in os.listdir(sample_folder)
                if f.startswith("Participant_") and f.endswith(".csv")
            ]
            # Add files to the dictionary to track their presence across participants
            for csv_file in csv_files:
                if csv_file not in csv_files_dict:
                    csv_files_dict[f"{csv_file}"] = []
                csv_files_dict[f"{csv_file}"].append(os.path.join(sample_folder, csv_file))

        # Group videos by unique combinations of the constant information
        grouped_videos = mapping.groupby(['yielding', 'p1', 'p2', 'camera'])['video_id'].apply(list).tolist()
        # Filter out groups that contain any 'baseline' videos
        grouped_videos = [group for group in grouped_videos if not any('baseline' in video for video in group)]
        new_grouped_videos = HMD_helper.sort_and_group_videos(grouped_videos)

        # Initialize dictionaries to store sums and counts for each column of each video
        video_sums = {video_name: [0, 0, 0] for video_name in mapping['video_id'].unique()}
        video_counts = {video_name: 0 for video_name in mapping['video_id'].unique()}

        # Loop through each CSV file (existing code to process files remains the same)
        for file_name, file_path in csv_files_dict.items():
            if isinstance(file_path, list) and len(file_path) > 0:
                file_path = file_path[0]

            if isinstance(file_path, str):
                try:
                    df = pd.read_csv(file_path, header=None)
                    for video_name in video_sums.keys():
                        video_rows = df[df[0] == video_name]
                        for i in range(1, 4):
                            video_sums[video_name][i-1] += video_rows[i].sum()
                        video_counts[video_name] += len(video_rows)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
            else:
                print(f"Invalid file path: {file_path}")

        # Calculate averages for each paired group
        for idx, video_group in enumerate(new_grouped_videos):
            # Calculate the mean of the first column for this paired group
            first_column_sum = sum(video_sums[video][0] for video in video_group if video_counts[video] > 0)
            first_column_count = sum(video_counts[video] for video in video_group if video_counts[video] > 0)
            first_column_mean = first_column_sum / first_column_count if first_column_count > 0 else None

            # Calculate the mean of the second column for this paired group
            second_column_sum = sum(video_sums[video][1] for video in video_group if video_counts[video] > 0)
            second_column_count = sum(video_counts[video] for video in video_group if video_counts[video] > 0)
            second_column_mean = second_column_sum / second_column_count if second_column_count > 0 else None

            # Calculate the mean of the third column for this paired group
            third_column_sum = sum(video_sums[video][2] for video in video_group if video_counts[video] > 0)
            third_column_count = sum(video_counts[video] for video in video_group if video_counts[video] > 0)
            third_column_mean = third_column_sum / third_column_count if third_column_count > 0 else None

            # Add the means to the results with a dynamic key
            group_key_prefix = f'group_{idx+1}'
            final_dict[f'{group_key_prefix}_first_column_mean'] = first_column_mean
            final_dict[f'{group_key_prefix}_second_column_mean'] = second_column_mean
            final_dict[f'{group_key_prefix}_third_column_mean'] = third_column_mean

        # Radar plot categories
        categories = ['Metric 1', 'Metric 2', 'Metric 3']

        # Create a radar plot
        fig = go.Figure()

        # Add traces for each group in the radar plot
        num_groups = len(new_grouped_videos)  # Number of groups to plot
        for idx in range(num_groups):
            group_key_prefix = f'group_{idx + 1}'

            # Extract the means for the specified columns
            values = [
                final_dict.get(f'{group_key_prefix}_first_column_mean', 0),
                final_dict.get(f'{group_key_prefix}_second_column_mean', 0),
                final_dict.get(f'{group_key_prefix}_third_column_mean', 0)
            ]

            # Add trace
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=group_key_prefix
            ))

        # Set the layout of the radar chart
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 80]  # Adjust based on the metric values, or set dynamically if needed
                )
            ),
            showlegend=True,
            title="Multi-Trace Radar Plot"
        )

        base_filename = "radar"
        fig.write_image(os.path.join(output_folder, base_filename + ".eps"), width=1600, height=900, scale=3)
        fig.write_image(os.path.join(output_folder, base_filename + ".png"), width=1600, height=900, scale=3)
        fig.write_html(os.path.join(output_folder, base_filename + ".html"))
        fig.write_image(os.path.join(output_folder, base_filename + ".svg"),
                        width=1600, height=900, scale=3, format="svg")

        # Display the radar plot
        fig.show()

    @staticmethod
    def plot_yaw_movement(readings_folder, mapping, output_folder, group_titles=None, legend_labels=None):
        dataframes_dict = {}
        participants_folders = [
            f for f in os.listdir(readings_folder)
            if os.path.isdir(os.path.join(readings_folder, f))
        ]

        # Dictionary to accumulate csv files found in participant folders
        csv_files_dict = {}

        # Search all participant folders for csv files
        for participant in participants_folders:
            sample_folder = os.path.join(readings_folder, participant)
            csv_files = [
                f for f in os.listdir(sample_folder)
                if f.startswith("video_") and f.endswith(".csv")
            ]
            # Add files to the dictionary to track their presence across participants
            for csv_file in csv_files:
                if csv_file not in csv_files_dict:
                    csv_files_dict[f"{participant}_{csv_file}"] = []
                csv_files_dict[f"{participant}_{csv_file}"].append(os.path.join(sample_folder, csv_file))

        # Initialize an empty dictionary to store groups
        grouped_data = {}

        # Loop through each key in the dictionary
        for key, value in sorted(csv_files_dict.items(), key=lambda x: int(x[0].split('_video_')[1].split('.')[0])):
            # Extract the video number from the key
            video_number = key.split('_video_')[1].split('.')[0]

            # Add the key-value pair to the appropriate group
            if video_number not in grouped_data:
                grouped_data[video_number] = []
            grouped_data[video_number].append({key: value})

        grouped_keys_list = [[list(item.keys())[0] for item in values] for values in grouped_data.values()]

        for jdx, csv_files in enumerate(grouped_keys_list):
            # Loop through each csv file in the current group
            for csv_file in csv_files:
                csv_file_ = "_".join(csv_file.split("_")[-2:])
                participant_ = "_".join(csv_file.split("_")[:2])
                file_path = os.path.join(readings_folder, participant_, csv_file_)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)[["Timestamp", "HMDRotationW", 'HMDRotationX',
                                                 'HMDRotationY', 'HMDRotationZ']]

                    # Extract quaternion columns and convert to Euler angles
                    euler_angles = df.apply(lambda row: HMD_helper.quaternion_to_euler(
                        row['HMDRotationW'], row['HMDRotationX'],
                        row['HMDRotationY'], row['HMDRotationZ']), axis=1)

                    # Split the tuple into separate columns for roll, pitch, and yaw

                    df[['Roll', 'Pitch', 'Yaw']] = pd.DataFrame(euler_angles.tolist(), index=df.index)

                    # Keep only the 'Timestamp' and 'Yaw' columns
                    df = df[['Timestamp', 'Yaw']]

                    # Get the corresponding video_length from the mapping
                    video_length = mapping.loc[mapping["video_id"] == csv_file_.rstrip('.csv'),
                                               "video_length"].values[0]
                    video_length = video_length / 1000

                    # Filter the DataFrame to only include rows where Timestamp >= 0 and <= video_length
                    df = df[(df["Timestamp"] >= 0) & (df["Timestamp"] <= video_length)]

                    # Add the DataFrame to the dictionary; create a list if it doesn't exist
                    if csv_file_ in dataframes_dict:
                        dataframes_dict[csv_file_].append(df)
                    else:
                        dataframes_dict[csv_file_] = [df]

        # Merge DataFrames and calculate the average if needed
        merged_dataframes = {}
        for csv_file_, df_list in dataframes_dict.items():
            if len(df_list) > 1:
                # Concatenate and then average if there are multiple DataFrames
                merged_df = pd.concat(df_list).groupby(level=0).mean()
            else:
                # If there's only one DataFrame, use it directly
                merged_df = df_list[0]

            merged_dataframes[csv_file_] = merged_df

        # Get the total number of plots needed (each containing 5 curves)
        num_plots = len(merged_dataframes) // 5 + (1 if len(merged_dataframes) % 5 != 0 else 0)

        # Iterate over the groups of 5
        for plot_index in range(num_plots):
            # Create a new subplot for each group of 5
            fig = go.Figure()

            # Determine the start and end indices for this group
            start_idx = plot_index * 5
            end_idx = min(start_idx + 5, len(merged_dataframes))

            # Add traces for this group
            for i, (csv_file_, merged_df) in enumerate(list(merged_dataframes.items())[start_idx:end_idx]):
                fig.add_trace(
                    go.Scatter(
                        x=merged_df["Timestamp"],
                        y=merged_df["Yaw"],
                        mode='lines',
                        name=legend_labels[i],
                        line=dict(width=3)  # Increase line thickness
                    )
                )

            # Update layout for each subplot
            fig.update_layout(
                title={
                    'text': group_titles[plot_index],
                    'x': 0.5,  # Center the title
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': {'size': 20, 'weight': 'bold'}  # Make the title larger and bold
                },
                xaxis_title={'text': 'Time (in seconds)', 'font': {'size': 20}},  # Increase x-axis label size
                yaxis_title={'text': 'Yaw Angle (in radians)', 'font': {'size': 20}},  # Increase y-axis label size
                legend_title=None,
                template='plotly',
                xaxis=dict(tickfont=dict(size=20)),  # Increase x-axis tick size
                yaxis=dict(tickfont=dict(size=20))   # Increase y-axis tick size
            )

            fig.update_layout(
                legend=dict(x=0.113, y=0.986, traceorder="normal", font=dict(size=24)))

            # Save the plot
            base_filename = f"yaw_group_{plot_index + 1}_trigger"
            fig.write_image(os.path.join(output_folder, base_filename + ".eps"), width=1600, height=900, scale=3)
            fig.write_image(os.path.join(output_folder, base_filename + ".png"), width=1600, height=900, scale=3)
            fig.write_html(os.path.join(output_folder, base_filename + ".html"))
            fig.write_image(os.path.join(output_folder, base_filename + ".svg"),
                            width=1600, height=900, scale=3, format="svg")

            # Show the plot
            fig.show()