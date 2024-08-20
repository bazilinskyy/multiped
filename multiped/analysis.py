from helper import HMD_helper
from custom_logger import CustomLogger
from logmod import logs
import common
import pandas as pd
from tqdm import tqdm


logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger
template = common.get_configs('plotly_template')
asset_folder = common.get_configs("data")
readings_folder = common.get_configs("readings")

HMD = HMD_helper()
mapping = pd.read_csv("../public/videos/mapping.csv")

participant_no = HMD.check_participant_file_exists(asset_folder)

if participant_no:
    HMD.move_csv_files(participant_no, mapping)
    HMD.delete_unnecessary_meta_files(participant_no, mapping)

HMD.plot_mean_trigger_value_right(readings_folder, mapping)