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


class QuestionnaireMixin:
    """Focused method group extracted without changing calculation logic."""

    def plot_column_distribution(self, df, columns, save_file=True, tag=None):
        """
        Plots and prints distributions of specified survey columns.

        Parameters:
            df (DataFrame or str): DataFrame or path to CSV.
            columns (list): List of column names to analyse.
            save_file (bool): Whether to save plots or just show them.
        """
        if isinstance(df, str):
            df = pd.read_csv(df)

        for column in columns:
            if column not in df.columns:
                logger.error(f"Column not found: {column}")
                continue

            logger.info(f"Distribution for: '{column}'")
            # Drop missing
            data = df[column].dropna().astype(str).str.strip()
            value_counts = data.value_counts()

            # Print counts
            for value, count in value_counts.items():
                logger.info(f"{value}: {count}")

            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(labels=value_counts.index, values=value_counts.values, hole=0.0)
            ])

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )

            # Save or display
            if save_file:
                filename = self._short_question_file_stem(column, tag=tag)
                self.save_plotly(fig, filename, save_final=True)
            else:
                fig.show()

    def distribution_plots(self, df, column_names, save_file=True):

        if isinstance(df, str):
            df = pd.read_csv(df)

        current_year = datetime.now().year

        for column_name in column_names:
            if column_name not in df.columns:
                logger.warning(f"Column not found: {column_name}")
                continue

            # Try numeric conversion
            temp_series = pd.to_numeric(df[column_name], errors='coerce')
            is_numeric = pd.api.types.is_numeric_dtype(temp_series)

            # Drop NaNs
            df_clean = df.dropna(subset=[column_name]).copy()

            if df_clean.empty:
                logger.warning(f"No valid data in column: {column_name}")
                continue

            if is_numeric:
                # Numeric column processing
                df_clean[column_name] = pd.to_numeric(df_clean[column_name], errors='coerce')

                # Special cleaning for age column
                if column_name.strip().lower() == "what is your age (in years)?".lower():
                    cleaned_ages = []
                    for val in df_clean[column_name]:
                        if 18 <= val <= 99:  # Valid age
                            cleaned_ages.append(val)
                        elif 1900 <= val <= current_year:  # Looks like year of birth
                            age = current_year - val
                            if 18 <= age <= 99:
                                cleaned_ages.append(age)
                        # Else: ignore nonsensical values
                    df_clean[column_name] = cleaned_ages

                mean_val = df_clean[column_name].mean()
                std_val = df_clean[column_name].std()
                logger.info(f"{column_name} - Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}")

                value_counts = df_clean[column_name].round().value_counts().sort_index()
                labels = [f"{int(v)}" for v in value_counts.index]
                values = value_counts.values
            else:
                # Categorical column processing
                df_clean[column_name] = df_clean[column_name].astype(str).str.strip()
                value_counts = df_clean[column_name].value_counts()
                labels = value_counts.index.tolist()
                values = value_counts.values.tolist()
                logger.info(f"{column_name} - Response counts: {dict(zip(labels, values))}")

            # Plotting
            fig = go.Figure(data=[
                go.Pie(labels=labels, values=values, hole=0.0, showlegend=True, sort=False)
            ])

            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10)
            )

            # Save or display
            if save_file:
                filename = self._short_question_file_stem(column_name)
                self.save_plotly(fig, filename, save_final=True)
            else:
                fig.show()

    def plot_gender_by_nationality(self, csv_path,
                                   gender_col="What is your gender?",
                                   nationality_col="Nationality"):
        """
        Reads a CSV file and generates an interactive Plotly bar chart
        showing gender distribution for each nationality.
        """

        df = csv_path.copy() if isinstance(csv_path, pd.DataFrame) else pd.read_csv(csv_path)

        nationality_map = {
            "Pakistani": "Pakistan",
            "Yemini": "Yemen",
            "Yemeni": "Yemen",
            "Nepalese": "Nepal",
            "Chinese": "China",
            "chinese": "China",
            " Chinese": "China",
            "Polish": "Poland",
            "Indian ": "India",
            "Dutch ": "Netherlands",
            "Dutch": "Netherlands",
            "Iranian": "Iran",
            "Romanian": "Romania",
            "Spanish": "Spain",
            "Colombian": "Colombia",
            "portuguese": "Portugal",
            "Taiwanese": "Taiwan",
            "German": "Germany",
            "Indian": "India",
            "dutch": "Netherlands"
        }

        # Apply mapping
        df[nationality_col] = df[nationality_col].map(nationality_map).fillna(
            df[nationality_col].str.capitalize()
        )

        # --- Group data ---
        s = df.groupby([nationality_col, gender_col]).size()
        s.name = "Count"
        grouped = s.reset_index()

        # --- Plot ---
        fig = px.bar(
            grouped,
            x=nationality_col,
            y="Count",
            color=gender_col,
            barmode="group",
            title=""
        )

        fig.show()
