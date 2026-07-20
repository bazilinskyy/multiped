"""Constants for participant event aligned head heading analysis."""

from __future__ import annotations

import math
import re

import common

PARTICIPANT_RE = re.compile(r"^P(\d+)$")
VIDEO_RE = re.compile(r"participant_Yaw_video_(\d+)\.csv$")
CACHE_VERSION = "participant-event-heading-v6.1-20ms-pointwise-ttest-plotly-only"
HEAD_HEADING_POINTWISE_P_THRESHOLD = float(common.get_configs("p_value"))
HEAD_HEADING_TIME_BIN_MS = int(common.get_configs("yaw_resolution"))
HEAD_HEADING_TIME_BIN_S = HEAD_HEADING_TIME_BIN_MS / 1000.0
HEAD_HEADING_TTEST_MARKER_SIZE = max(float(common.get_configs("font_size")) - 6.0, 1.0)
HEAD_HEADING_LINE_WIDTH = 3.0
HEAD_HEADING_XAXIS_STEP = 1.0
HEAD_HEADING_YAXIS_STEP_DEG = 20.0
HEAD_HEADING_FIG_WIDTH = 1470
HEAD_HEADING_FIG_HEIGHT = 850
HEAD_HEADING_LEGEND_X = 0.0
HEAD_HEADING_LEGEND_Y = 1.225
HEAD_HEADING_TTEST_ROW_HEIGHT_DEG = math.degrees(0.006)
