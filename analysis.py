# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>

from helper import HMD_helper
from custom_logger import CustomLogger
from logmod import logs
import common
import pandas as pd
import os


logs(show_level="info", show_color=True)
logger = CustomLogger(__name__)  # use custom logger
HMD = HMD_helper()

template = common.get_configs("plotly_template")
readings_folder = common.get_configs("data")  # new location of the csv file with participant id

# Read the mapping file
mapping = pd.read_csv(common.get_configs("mapping"))
# Remove rows where 'video_id' is 'baseline_1' or 'baseline_2'
mapping = mapping[~mapping['video_id'].isin(['baseline_1', 'baseline_2'])]

directory_path = common.get_configs("output")

# Intake questionairre
first_csv = common.get_configs("input_csv")

# Post-experiment questionairre
last_csv = common.get_configs("post_input_csv")

try:
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
except Exception as e:
    print(f"Error occurred while creating directory: {e}")


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

legend_labels = ["1m", "2m", "3m", "4m", "5m"]

# HMD.plot_mean_trigger_value_right(readings_folder, mapping, output_folder=directory_path,
#                                   group_titles=group_titles, legend_labels=legend_labels)
# HMD.plot_yaw_movement(readings_folder, mapping, output_folder=directory_path,
#                       group_titles=group_titles, legend_labels=legend_labels)
# HMD.radar_plot(readings_folder, mapping, output_folder=directory_path)
# HMD.gender_distribution(first_csv, directory_path)
# HMD.age_distribution(first_csv, directory_path)
# HMD.demographic_distribution(first_csv, directory_path)
HMD.ttest(readings_folder, mapping, directory_path, group_titles,
          legend_labels=legend_labels)
