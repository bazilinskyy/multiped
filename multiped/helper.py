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
    def plot_mean_trigger_value_right(readings_folder, mapping, output_folder, group_titles=None, legend_labels=None):
        participants_folders = [
            f for f in os.listdir(readings_folder)
            if os.path.isdir(os.path.join(readings_folder, f))
        ]

        # Get all the csv files starting with "video_" and ending with ".csv"
        sample_folder = os.path.join(readings_folder, participants_folders[0])
        csv_files = [
            f for f in os.listdir(sample_folder)
            if f.startswith("video_") and f.endswith(".csv")
        ]

        # Sort files by the number in their names to ensure correct ordering
        csv_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

        # Group files in sets of 5 (1-5, 6-10, etc.)
        groups = [csv_files[i: i + 5] for i in range(0, len(csv_files), 5)]

        for idx, group in enumerate(groups):
            # Initialize the figure for each group
            fig = go.Figure()

            # Use a custom title for the group if provided
            group_title = f"Mean TriggerValue for {', '.join(group)}" if group_titles is None else group_titles[idx]

            # Loop through each file in the group
            for jdx, csv_file in enumerate(group):
                dfs = []
                for participant in participants_folders:
                    file_path = os.path.join(readings_folder, participant, csv_file)
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)

                        # Extract video number from the filename to match with mapping
                        video_number = re.findall(r"\d+", csv_file)[0]
                        video_id = f"video_{video_number}"

                        # Get the corresponding video_length from the mapping
                        video_length = mapping.loc[mapping["video_id"] == video_id, "video_length"].values[0]
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

                    # Use a custom legend label for the file if provided
                    legend_label = csv_file if legend_labels is None else legend_labels[idx][jdx]

                    # Add a trace for the 'TriggerValueRight' data for this file
                    fig.add_trace(
                        go.Scatter(
                            x=mean_df["Timestamp"],
                            y=mean_df["TriggerValueRight"],
                            mode="lines",
                            name=legend_label,  # Use the custom legend label
                        )
                    )

            # Update layout to add titles, axes labels, and show the legend
            fig.update_layout(
                title=group_title,  # Use the custom group title
                xaxis_title="Timestamp",
                yaxis_title="Mean TriggerValueRight",
                showlegend=True,  # Ensure legend is displayed
            )
            base_filename = f"group_{idx + 1}_trigger"
            fig.write_image(os.path.join(output_folder, base_filename + ".eps"))
            fig.write_image(os.path.join(output_folder, base_filename + ".png"))
            fig.write_html(os.path.join(output_folder, base_filename + ".html"))

            # Show the figure for the current group
            fig.show()

    def plot_mean_hmd_yaw(self, readings_folder, mapping, output_folder, group_titles=None, legend_labels=None):
        participants_folders = [
            f for f in os.listdir(readings_folder)
            if os.path.isdir(os.path.join(readings_folder, f))
        ]

        # Get all the csv files starting with "video_" and ending with ".csv"
        sample_folder = os.path.join(readings_folder, participants_folders[0])
        csv_files = [
            f for f in os.listdir(sample_folder)
            if f.startswith("video_") and f.endswith(".csv")
        ]

        # Sort files by the number in their names to ensure correct ordering
        csv_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

        # Group files in sets of 5 (1-5, 6-10, etc.)
        groups = [csv_files[i: i + 5] for i in range(0, len(csv_files), 5)]

        for idx, group in enumerate(groups):
            # Initialize the figure for each group
            fig = go.Figure()

            # Use a custom title for the group if provided
            group_title = f"Mean HMD (Yaw Angle) for {', '.join(group)}" if group_titles is None else group_titles[idx]

            # Loop through each file in the group
            for jdx, csv_file in enumerate(group):
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

                        # Extract quaternion columns and convert to Euler angles
                        euler_angles = df.apply(lambda row: self.quaternion_to_euler(
                            row['HMDRotationW'], row['HMDRotationX'],
                            row['HMDRotationY'], row['HMDRotationZ']), axis=1)

                        # Convert the Euler angles tuple into separate columns and keep only the Yaw
                        df['Yaw'] = euler_angles.apply(lambda angles: angles[2])

                        # Keep only the 'Timestamp' and the 'Yaw' column
                        df = df[['Timestamp', 'Yaw']]
                        dfs.append(df)

                if dfs:
                    combined_df = pd.concat(dfs)
                    mean_df = combined_df.groupby("Timestamp").mean().reset_index()

                    # Use a custom legend label for the file if provided
                    legend_label = csv_file if legend_labels is None else legend_labels[idx][jdx]

                    # Add a trace for the Yaw data
                    fig.add_trace(
                        go.Scatter(
                            x=mean_df["Timestamp"],
                            y=mean_df["Yaw"],
                            mode="lines",
                            name=f"Yaw - {legend_label}",
                        )
                    )

            # Update layout to add titles, axes labels, and show the legend
            fig.update_layout(
                title=group_title,  # Use the custom group title
                xaxis_title="Timestamp",
                yaxis_title="Yaw Angle (Radians)",
                showlegend=True,  # Ensure legend is displayed
            )
            base_filename = f"group_{idx + 1}_hmd_yaw"
            fig.write_image(os.path.join(output_folder, base_filename + ".eps"))
            fig.write_image(os.path.join(output_folder, base_filename + ".png"))
            fig.write_html(os.path.join(output_folder, base_filename + ".html"))

            # Show the figure for the current group
            fig.show()

    @staticmethod
    def plot_video_averages(readings_folder):
        # Initialize dicts to store the averages
        video_average_1, video_average_2, video_average_3 = {}, {}, {}

        # Loop through each item in the readings folder
        for item in os.listdir(readings_folder):
            pilot_folder_path = os.path.join(readings_folder, item)

            # Process only directories that start with "pilot"
            if os.path.isdir(pilot_folder_path) and item.startswith('pilot'):
                # Find the specific CSV file that starts with the same name as the folder
                for file_name in os.listdir(pilot_folder_path):
                    if file_name.startswith(item) and file_name.endswith('.csv'):
                        file_path = os.path.join(pilot_folder_path, file_name)

                        # Read the CSV file
                        df = pd.read_csv(file_path, header=None)
                        # Convert the 2nd, 3rd, and 4th columns to integers
                        df[[1, 2, 3]] = df[[1, 2, 3]].astype(int)

                        # Filter rows where the first column starts with 'video'
                        df = df[df[0].str.startswith('video')]

                        # Loop to calculate the average in chunks of 5 rows
                        for i in range(0, len(df), 5):
                            avg_col1 = df[1].iloc[i:i+5].mean()
                            avg_col2 = df[2].iloc[i:i+5].mean()
                            avg_col3 = df[3].iloc[i:i+5].mean()

                            idx = int(i/5)
                            if idx not in video_average_1:
                                video_average_1[idx] = [avg_col1]
                            else:
                                video_average_1[idx].append(avg_col1)

                            if idx not in video_average_2:
                                video_average_2[idx] = [avg_col2]
                            else:
                                video_average_2[idx].append(avg_col2)

                            if idx not in video_average_3:
                                video_average_3[idx] = [avg_col3]
                            else:
                                video_average_3[idx].append(avg_col3)

        # Calculate the mean of the lists in the dictionaries
        for idx in video_average_1:
            video_average_1[idx] = sum(video_average_1[idx]) / len(video_average_1[idx])
            video_average_2[idx] = sum(video_average_2[idx]) / len(video_average_2[idx])
            video_average_3[idx] = sum(video_average_3[idx]) / len(video_average_3[idx])

        num_plots = len(video_average_1)
        for i in range(num_plots):
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=[f'Chunk {i+1}'],
                y=[video_average_1[i]],
                name='Average 1',
                marker_color='blue'
            ))

            fig.add_trace(go.Bar(
                x=[f'Chunk {i+1}'],
                y=[video_average_2[i]],
                name='Average 2',
                marker_color='green'
            ))

            fig.add_trace(go.Bar(
                x=[f'Chunk {i+1}'],
                y=[video_average_3[i]],
                name='Average 3',
                marker_color='red'
            ))

            # Update layout
            fig.update_layout(
                title=f'Video Averages for Chunk {i+1}',
                xaxis=dict(title='Chunks'),
                # yaxis=dict(title='Average Values'),
                barmode='group'
            )

            # Show the plot
            fig.show()

    @staticmethod
    def plot_combined(readings_folder, mapping, output_folder, group_titles, legend_labels=None):
        participants_folders = [
            f for f in os.listdir(readings_folder)
            if os.path.isdir(os.path.join(readings_folder, f))
        ]

        # Get all the csv files starting with "video_" and ending with ".csv"
        sample_folder = os.path.join(readings_folder, participants_folders[0])
        csv_files = [
            f for f in os.listdir(sample_folder)
            if f.startswith("video_") and f.endswith(".csv")
        ]

        # Sort files by the number in their names to ensure correct ordering
        csv_files.sort(key=lambda x: int(re.findall(r"\d+", x)[0]))

        # Group files in sets of 5 (1-5, 6-10, etc.)
        groups = [csv_files[i: i + 5] for i in range(0, len(csv_files), 5)]

        # Initialize dicts to store the averages
        video_average_1, video_average_2, video_average_3 = {}, {}, {}

        for idx, group in enumerate(groups):
            # Create a subplot with 1 row and 2 columns
            fig = make_subplots(rows=1, cols=2,
                                column_widths=[0.7, 0.3],
                                subplot_titles=(f"Mean TriggerValue for Group {idx+1}", "Response to the sliders"))

            # Loop through each file in the group for line plot
            for jdx, csv_file in enumerate(group):
                dfs = []
                for participant in participants_folders:
                    file_path = os.path.join(readings_folder, participant, csv_file)
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)

                        # Extract video number from the filename to match with mapping
                        video_number = re.findall(r"\d+", csv_file)[0]
                        video_id = f"video_{video_number}"

                        # Get the corresponding video_length from the mapping
                        video_length = mapping.loc[mapping["video_id"] == video_id, "video_length"].values[0]
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

                    # Use a custom legend label for the file if provided
                    legend_label = csv_file if legend_labels is None else legend_labels[idx][jdx]

                    # Add a trace for the 'TriggerValueRight' data for this file to the first subplot
                    fig.add_trace(
                        go.Scatter(
                            x=mean_df["Timestamp"],
                            y=mean_df["TriggerValueRight"],
                            mode="lines",
                            name=legend_label,  # Use the custom legend label
                        ),
                        row=1, col=1
                    )

            # Video averages bar plot
            for item in os.listdir(readings_folder):
                pilot_folder_path = os.path.join(readings_folder, item)

                if os.path.isdir(pilot_folder_path) and item.startswith('pilot'):
                    for file_name in os.listdir(pilot_folder_path):
                        if file_name.startswith(item) and file_name.endswith('.csv'):
                            file_path = os.path.join(pilot_folder_path, file_name)
                            df = pd.read_csv(file_path, header=None)
                            df[[1, 2, 3]] = df[[1, 2, 3]].astype(int)
                            df = df[df[0].str.startswith('video')]

                            for i in range(0, len(df), 5):
                                avg_col1 = df[1].iloc[i:i+5].mean()
                                avg_col2 = df[2].iloc[i:i+5].mean()
                                avg_col3 = df[3].iloc[i:i+5].mean()

                                chunk_idx = int(i/5)
                                if chunk_idx not in video_average_1:
                                    video_average_1[chunk_idx] = [avg_col1]
                                else:
                                    video_average_1[chunk_idx].append(avg_col1)

                                if chunk_idx not in video_average_2:
                                    video_average_2[chunk_idx] = [avg_col2]
                                else:
                                    video_average_2[chunk_idx].append(avg_col2)

                                if chunk_idx not in video_average_3:
                                    video_average_3[chunk_idx] = [avg_col3]
                                else:
                                    video_average_3[chunk_idx].append(avg_col3)

            fig.add_trace(go.Bar(
                # x=[f'Chunk {idx+1}'],  # Adjust the chunk label
                y=[sum(video_average_1[idx]) / len(video_average_1[idx])],
                name='Average 1',
                marker_color='blue',
                width=0.2,
                ), row=1, col=2)

            fig.add_trace(go.Bar(
                # x=[f'Chunk {idx+1}'],  # Adjust the chunk label
                y=[sum(video_average_2[idx]) / len(video_average_2[idx])],
                name='Average 2',
                marker_color='green',
                width=0.2,
                ), row=1, col=2)

            fig.add_trace(go.Bar(
                # x=[f'Chunk {idx+1}'],  # Adjust the chunk label
                y=[sum(video_average_3[idx]) / len(video_average_3[idx])],
                name='Average 3',
                marker_color='red',
                width=0.2,
                ), row=1, col=2)

            # Add annotations
            # annotations = [
            #     dict(x=10, y=1, xref="x", yref="y", text="pedestrian_cross_the_road-start",
            #          showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(color="red")),
            #     dict(x=12, y=1, xref="x", yref="y", text="pedestrian_cross_the_road-finish",
            #          showarrow=True, arrowhead=2, ax=0, ay=-40, font=dict(color="red")),]

            # fig.update_layout(annotations=annotations)

            # Update layout to add titles, axes labels, legend, and background color
            fig.update_layout(
                title=dict(
                    text=group_titles[idx],
                    x=0.5,  # Center the title
                    y=0.95,  # Adjust the vertical position
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=20, color='red')
                ),
                legend=dict(
                    x=0.65,  # Position the legend outside of the plot area
                    y=1,     # Align the legend to the top
                    traceorder="normal",
                    font=dict(
                        family="sans-serif",
                        size=12,
                        # color="white"
                    ),
                    # bgcolor="rgba(0,0,0,0)",  # Transparent background for the legend
                ),
                # paper_bgcolor='black',  # Set background color
                # plot_bgcolor='black',   # Set plot background color
                # font=dict(
                #     color='white'       # Set font color to white for better contrast
                # )
            )

            # Update axis labels
            fig.update_yaxes(
                title_text="Percentage of trials with response key pressed",
                title_font=dict(size=18),
                tickfont=dict(size=14),
                gridcolor='gray',  # Add gridlines for better visual separation
                row=1, col=1,       # Ensure it's specific to the first plot
                automargin=True   # Allow automatic margins to avoid overlap
                )
            fig.update_xaxes(
                title_text="Time(s)",
                title_font=dict(size=18),
                tickfont=dict(size=14),
                # gridcolor='gray',  # Add gridlines for better visual separation
                row=1, col=1,       # Ensure it's specific to the first plot
                automargin=True   # Allow automatic margins to avoid overlap
                )

            # Update y-axis for the bar plot (automatic scaling)
            # fig.update_yaxes(
            #     title_text="Average Values",
            #     title_font=dict(size=18),
            #     tickfont=dict(size=14),
            #     gridcolor='gray',  # Add gridlines for better visual separation
            #     row=1, col=2,       # Ensure it's specific to the second plot
            #     automargin=True   # Allow automatic margins to avoid overlap
            #     )

            # Update bar plot layout
            fig.update_xaxes(title_text="Stimlus", row=1, col=2,
                             tickvals=[f'Chunk {idx+1}'],  # Only display the chunk label
                             ticktext=[f'Chunk {idx+1}'],  # Set the custom text for the tick
                             showticklabels=True)
            # fig.update_yaxes(title_text="Average Values", row=1, col=2)

            # Save and show the figure
            base_filename = f"group_{idx + 1}_combined"
            fig.write_image(os.path.join(output_folder, base_filename + ".eps"))
            fig.write_image(os.path.join(output_folder, base_filename + ".png"))
            fig.write_html(os.path.join(output_folder, base_filename + ".html"))

            fig.show()
