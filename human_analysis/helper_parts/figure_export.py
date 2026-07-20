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


class FigureExportMixin:
    """Focused plotting responsibility extracted from the legacy helper."""

    def save_plotly(self, fig, name, remove_margins=False, width=1320, height=680, save_eps=True, save_png=True,
                    save_html=True, open_browser=True, save_mp4=False, save_final=False):
        """
        Save a Plotly figure as HTML and image files.

        The ``name`` argument may include subdirectories, for example
        ``kp_threshold_sensitivity/threshold_05pct/my_figure``. Any missing
        output folders are created automatically.
        """
        # disable mathjax globally for Kaleido
        pio.kaleido.scope.mathjax = None

        output_root = os.path.join(common.get_configs("output"))
        final_root = self.folder_figures
        os.makedirs(output_root, exist_ok=True)
        if save_final:
            os.makedirs(final_root, exist_ok=True)

        # Keep only safe path components while preserving intentional folders.
        name = str(name).replace("\\", os.sep).replace("/", os.sep)
        name = os.path.normpath(name)
        if name.startswith("..") or os.path.isabs(name):
            raise ValueError(f"Figure name must be a relative path, got: {name}")
        if name == "head_heading" or name.startswith(f"head_heading{os.sep}"):
            name = os.path.basename(name)

        output_base = os.path.join(output_root, name)
        final_base = os.path.join(final_root, name)
        os.makedirs(os.path.dirname(output_base), exist_ok=True)
        if save_final:
            os.makedirs(os.path.dirname(final_base), exist_ok=True)

        # Limit only the file stem when paths become too long.
        output_dir = os.path.dirname(output_base)
        final_dir = os.path.dirname(final_base)
        stem = os.path.basename(output_base)
        max_dir_len = max(len(output_dir), len(final_dir))
        if max_dir_len + len(stem) > 195:
            safe_len = max(20, 190 - max_dir_len)
            stem = stem[:safe_len]
            output_base = os.path.join(output_dir, stem)
            final_base = os.path.join(final_dir, stem)

        # Save once in the output directory, then copy the exact file into the
        # configured figures directory. This guarantees that both locations
        # contain the same figure and avoids rendering the same figure twice.
        if save_html:
            output_html = output_base + ".html"
            py.offline.plot(fig, filename=output_html, auto_open=open_browser)
            logger.info(f"Saved figure: {output_html}")

            if save_final:
                final_html = final_base + ".html"
                shutil.copy2(output_html, final_html)
                logger.info(f"Saved figure: {final_html}")

        # remove white margins
        if remove_margins:
            fig.update_layout(margin=dict(l=100, r=2, t=20, b=12))

        # save as eps
        if save_eps:
            try:
                output_eps = output_base + ".eps"
                fig.write_image(output_eps, width=width, height=height)
                logger.info(f"Saved figure: {output_eps}")

                if save_final:
                    final_eps = final_base + ".eps"
                    shutil.copy2(output_eps, final_eps)
                    logger.info(f"Saved figure: {final_eps}")
            except Exception as exc:
                logger.warning(
                    f"Skipping EPS export for '{name}' because Plotly/Kaleido could not create the EPS file: {exc}"
                )

        # save as png
        if save_png:
            try:
                output_png = output_base + ".png"
                fig.write_image(output_png, width=width, height=height)
                logger.info(f"Saved figure: {output_png}")

                if save_final:
                    final_png = final_base + ".png"
                    shutil.copy2(output_png, final_png)
                    logger.info(f"Saved figure: {final_png}")
            except Exception as exc:
                logger.warning(
                    f"Skipping PNG export for '{name}' because Plotly/Kaleido could not create the PNG file: {exc}"
                )

        # save as mp4
        if save_mp4:
            try:
                output_mp4 = output_base + '.mp4'
                fig.write_image(output_mp4, width=width, height=height)
                logger.info(f"Saved figure: {output_mp4}")
            except Exception as exc:
                logger.warning(
                    f"Skipping MP4 export for '{name}' because Plotly/Kaleido could not create the MP4 file: {exc}"
                )
