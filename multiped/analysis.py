from helper import HMD_helper
from custom_logger import CustomLogger
from logmod import logs
import common
import pandas as pd


logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger
template = common.get_configs('plotly_template')

HMD = HMD_helper()

# print(mapping)

# Uncomment the below line if all the files are in correct directory
# HMD.move_csv_files(participant_no)

# HMD.plot_button_press(mapping)
HMD.move_csv_files()
