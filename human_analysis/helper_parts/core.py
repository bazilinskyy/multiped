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


class CoreMixin:
    """Focused method group extracted without changing calculation logic."""

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
        """Convert one raw mapping code into physical metres."""
        return distance_code_to_metres(value)

    @classmethod
    def _distance_series_to_meters(cls, series):
        """Convert raw mapping distance codes into physical metres."""
        return distance_codes_to_metres(series)

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
