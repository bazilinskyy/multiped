import pandas as pd
import os
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import common
from custom_logger import CustomLogger
import math

logger = CustomLogger(__name__)  # use custom logger
mapping = pd.read_csv(common.get_configs('mapping'))


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
    def move_csv_files():
        global mapping
        folder_path = f'readings/{participant_no}'
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print(f"Folder '{folder_path}' created.")
        else:
            print(f"Folder '{folder_path}' already exists.")
        data_folder_path = common.get_configs("data")
        file_names = mapping["video_id"]

    def plot_button_press():
        pass
