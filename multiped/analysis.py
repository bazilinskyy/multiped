from helper import HMD_helper
from custom_logger import CustomLogger
from logmod import logs
import common
import pandas as pd
import os


logs(show_level="info", show_color=True)
logger = CustomLogger(__name__)  # use custom logger
template = common.get_configs("plotly_template")
asset_folder = common.get_configs("data")
readings_folder = common.get_configs("readings")

HMD = HMD_helper()
mapping = pd.read_csv("../public/videos/mapping.csv")
directory_path = common.get_configs("outputs")

try:
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
except Exception as e:
    print(f"Error occurred while creating directory: {e}")

participant_no = HMD.check_participant_file_exists(asset_folder)

if participant_no:
    HMD.move_csv_files(participant_no, mapping)
    HMD.delete_unnecessary_meta_files(participant_no, mapping)

group_titles = [
    "Yielding, Front and no eHMI",
    "Yielding, Back and no eHMI",
    "Yieding, Front and eHMI",
    "Yielding, Back and eHMI",
    "No Yielding, Front and eHMI",
    "No Yielding, Back and eHMI",
    "No Yielding, Front and no eHMI",
    "No Yielding, Back and no eHMI"
]

legend_labels = [
    ["1m", "2m", "3m", "4m", "5m"],
    ["1m", "2m", "3m", "4m", "5m"],
    ["1m", "2m", "3m", "4m", "5m"],
    ["1m", "2m", "3m", "4m", "5m"],
    ["1m", "2m", "3m", "4m", "5m"],
    ["1m", "2m", "3m", "4m", "5m"],
    ["1m", "2m", "3m", "4m", "5m"],
    ["1m", "2m", "3m", "4m", "5m"]]

# HMD.plot_mean_trigger_value_right(readings_folder, mapping, output_folder=directory_path,
#                                   group_titles=group_titles, legend_labels=legend_labels)
# HMD.plot_mean_hmd_yaw(readings_folder, mapping, output_folder=directory_path,
# group_titles=group_titles, legend_labels=legend_labels)
# HMD.plot_video_averages(readings_folder)
HMD.plot_combined(readings_folder, mapping, output_folder=directory_path,
                  group_titles=group_titles, legend_labels=legend_labels)
