# by Shadab Alam <shaadalam.5u@gmail.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from __future__ import annotations

from helper import HMD_helper
from custom_logger import CustomLogger
from logmod import logs
import hashlib
import json
import math
import re
import common
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Any, Callable

from scipy import stats

from processed_cache import ProcessedExperimentCache
import stats as stats_module


AdvancedStatsRunner = stats_module.AdvancedStatsRunner


logs(show_level="info", show_color=True)
logger = CustomLogger(__name__)  # use custom logger
HMD = HMD_helper()

data_folder = common.get_configs("data")  # new location of the csv file with participant id
mapping_source = common.get_configs("mapping")
mapping = None
output_folder = common.get_configs("output")
intake_questionnaire_source = common.get_configs("intake_questionnaire")
post_experiment_questionnaire_source = common.get_configs("post_experiment_questionnaire")
intake_questionnaire = None
post_experiment_questionnaire = None
ALWAYS_ANALYSE = bool(common.get_configs("always_analyse"))
PROCESSED_DATA_CACHE = common.get_configs("processed_data_cache")
CACHE_SETTINGS = {
    "processing_version": 2,
    "kp_resolution_ms": int(common.get_configs("kp_resolution")),
    "yaw_resolution_ms": int(common.get_configs("yaw_resolution")),
    "distance_convention": "distPed_code_times_2_metres_v1",
    "heading_convention": "unity_y_vertical_xz_heading_v1",
    "trigger_binning_convention": "floor_half_open_any_above_threshold_v2",
}

RUN_ADVANCED_STATISTICS = True
ANALYSIS_PIPELINE_VERSION = "reviewer_response_v4_bounded_common_window"
ANALYSIS_SOURCE_VERSION = "2026-07-19-head-heading-plotly-only-v6.1"
EQUIVALENCE_MARGIN_POINTS = 5.0
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
# Keep the existing yaw-plot row spacing (0.006 rad), converted because the
# event-aligned manuscript figure is displayed in degrees.
HEAD_HEADING_TTEST_ROW_HEIGHT_DEG = math.degrees(0.006)
PRIMARY_TRIGGER_THRESHOLD = float(common.get_configs("primary_trigger_threshold"))
TRIGGER_PRESS_THRESHOLDS = [
    float(value) for value in common.get_configs("trigger_threshold")
]


intake_columns_to_plot = [
    "Do you consent to participate in this study as described in the information provided above?",
    "Have you read and understood the above instructions?",
    "What is your gender?",
    "Are you wearing any seeing aids during the experiments?",
    "Do you have problems with hearing?",
    "How often in the last month have you experienced virtual reality?",
    "I am comfortable with walking in areas with dense traffic.",
    "The presence of another pedestrian reduces my willingness to cross the street when a car is driving towards me.",
    "What is your primary mode of transportation?",
    "On average, how often did you drive a vehicle in the last 12 months?",
    "About how many kilometers did you drive in last 12 months?",
    "How often do you do the following?: Becoming angered by a particular type of driver, and indicate your hostility by whatever means you can.",  # noqa: E501
    "How often do you do the following?: Disregarding the speed limit on a motorway.",
    "How often do you do the following?: Disregarding the speed limit on a residential road. ",
    "How many accidents were you involved in when driving a car in the last 3 years? (please include all accidents, regardless of how they were caused, how slight they were, or where they happened)",  # noqa: E501
    "How often do you do the following?: Driving so close to the car in front that it would be difficult to stop in an emergency. ",  # noqa: E501
    "How often do you do the following?: Racing away from traffic lights with the intention of beating the driver next to you. ",  # noqa: E501
    "How often do you do the following?: Sounding your horn to indicate your annoyance with another road user. ",
    "How often do you do the following?: Using a mobile phone without a hands free kit.",
    "How often do you do the following?: Doing my best not to be obstacle for other drivers.",
    "I would like to communicate with other road users while crossing the road (for instance, using eye contact, gestures, verbal communication, etc.).",  # noqa: E501
    "I trust an automated car more than a manually driven car."
]

post_columns_to_plot = [
    "The presence of another pedestrian influenced my willingness to cross the road.",
    "The type of car (with eHMI or without eHMI) affected my decision to cross the road.",
    "I trust an automated car more than a manually driven car."
]

# Put all the questions where one need to calculate mean and standard deviation
intake_columns_distribution_to_plot = [
    "What is your age (in years)?",
    "At what age did you obtain your first license for driving a car or motorcycle?",
]

post_columns_distribution_to_plot = [
    "How stressful did you feel during the experiment?",
    "How anxious did you feel during the experiment?",
    "How realistic did you find the experiment?",
    "How would you rate your overall experience in this experiment?",
]

try:
    # Check if the directory already exists
    if not os.path.exists(output_folder):
        # Create the directory
        os.makedirs(output_folder)
        logger.info(f"Directory '{output_folder}' created successfully.")
except Exception as e:
    logger.error(f"Error occurred while creating directory: {e}")


def ensure_slider_tables(
    data_folder: str,
    output_dir: str,
    allow_raw_processing: bool = True,
) -> None:
    """Generate slider tables only when they are missing."""
    expected = [
        os.path.join(output_dir, "slider_input_behaviour.csv"),
        os.path.join(output_dir, "slider_input_distance.csv"),
        os.path.join(output_dir, "slider_input_intention.csv"),
    ]

    missing = [path for path in expected if not os.path.isfile(path)]
    if not missing:
        logger.info("Slider input tables found. Reusing existing files.")
        return

    if not allow_raw_processing:
        raise FileNotFoundError(
            f"The processed-data cache did not restore all slider tables: {missing}. "
            "Set always_analyse to true for one run to rebuild the cache."
        )

    logger.info("Generating missing slider input tables.")
    for path in missing:
        logger.info(f"Missing slider table: {path}")

    HMD.read_slider_data(data_folder, output_dir)

    for path in expected:
        if os.path.isfile(path):
            logger.info(f"Slider table ready: {path}")
        else:
            logger.warning(f"Expected slider table was not created: {path}")


def run_advanced_statistics(trial_level_df: pd.DataFrame, trigger_threshold: float) -> None:
    """Run the added statistical analyses on top of the existing pipeline output."""
    if not RUN_ADVANCED_STATISTICS:
        logger.info("Advanced statistics disabled.")
        return

    loaded_stats_version = getattr(
        stats_module, "ADVANCED_STATS_SPECIFICATION", "missing"
    )
    loaded_stats_path = os.path.abspath(getattr(stats_module, "__file__", "unknown"))
    logger.info(f"Resolved stats module: {loaded_stats_path}")
    logger.info(f"Resolved stats specification: {loaded_stats_version}")
    if loaded_stats_version != ANALYSIS_PIPELINE_VERSION:
        raise RuntimeError(
            "analysis.py and stats.py do not belong to the same reviewer-response "
            "release. Expected stats specification "
            f"'{ANALYSIS_PIPELINE_VERSION}', loaded '{loaded_stats_version}' from "
            f"'{loaded_stats_path}'. Replace both files before rerunning."
        )

    logger.info("Running advanced statistics and extra figure generation.")
    runner = AdvancedStatsRunner(
        helper=HMD,
        mapping_df=mapping,
        output_dir=output_folder,
    )
    results = runner.run_all(
        trial_df=trial_level_df,
        equivalence_margin=EQUIVALENCE_MARGIN_POINTS,
        trigger_threshold=trigger_threshold,
    )
    required_outputs = [
        "common_window_primary_marginal_probabilities.csv",
        "common_window_primary_revised_contrasts.csv",
        "common_window_primary_omnibus_tests.csv",
        "common_window_primary_binomial_diagnostics.csv",
        "common_window_participant_first_marginal_probabilities.csv",
        "common_window_participant_first_omnibus_tests.csv",
        "braking_onset_window_participant_first_marginal_probabilities.csv",
        "stopping_window_participant_first_marginal_probabilities.csv",
        "resumption_window_participant_first_marginal_probabilities.csv",
        "common_window_threshold_marginal_probabilities.csv",
        "common_window_threshold_revised_contrasts.csv",
        "common_window_threshold_omnibus_tests.csv",
        "common_window_threshold_binomial_diagnostics.csv",
        "common_window_participant_level_descriptives.csv",
        "common_window_figure7_cell_summary.csv",
    ]
    missing_outputs = [
        filename
        for filename in required_outputs
        if not os.path.isfile(os.path.join(output_folder, "statistics", filename))
    ]
    if missing_outputs:
        raise RuntimeError(
            "Reviewer-response analysis did not create all required outputs: "
            + ", ".join(missing_outputs)
        )
    logger.info(
        "Reviewer-response output verification passed: "
        f"{len(required_outputs)} required tables are present; "
        f"result groups={sorted(results)}"
    )


def _trigger_threshold_label(trigger_threshold: float) -> str:
    """Return a compact label such as 05pct, 10pct, or 50pct."""
    pct = float(trigger_threshold) * 100.0
    if pct.is_integer():
        return f"{int(pct):02d}pct"
    return f"{str(round(pct, 3)).replace('.', 'p')}pct"


def _keypress_plot_specs():
    """Central list of all condition-specific keypress figures."""
    base_margin = dict(l=120, r=2, t=12, b=12)
    y_title = "Percentage of trials with trigger key pressed"
    x_title = "Time, [s]"
    return [
        dict(parameter=None, xaxis_range=[0, 18], compare_trial="video_1",
             xaxis_title=x_title, yaxis_title=y_title,
             name="all_values_with_yielding", margin=base_margin),
        dict(parameter=None, xaxis_range=[0, 11], compare_trial="video_21",
             xaxis_title=x_title, yaxis_title=y_title,
             name="all_values_without_yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=0, xaxis_range=[0, 18], compare_trial="video_1",
             xaxis_title=x_title, yaxis_title=y_title,
             name="eHMI_off_yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=1, xaxis_range=[0, 18], compare_trial="video_11",
             xaxis_title=x_title, yaxis_title=y_title,
             name="eHMI_on_yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=0, xaxis_range=[0, 11], compare_trial="video_31",
             xaxis_title=x_title, yaxis_title=y_title,
             name="eHMI_off_non-yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=1, xaxis_range=[0, 11], compare_trial="video_21",
             xaxis_title=x_title, yaxis_title=y_title,
             name="eHMI_on_non-yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=1, additional_parameter="camera", additional_parameter_value=0,
             xaxis_range=[0, 11], compare_trial="video_21", xaxis_title=x_title, yaxis_title=y_title,
             name="first_eHMI_on_non-yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=1, additional_parameter="camera", additional_parameter_value=0,
             xaxis_range=[0, 18], compare_trial="video_11", xaxis_title=x_title, yaxis_title=y_title,
             name="first_eHMI_on_yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=0, additional_parameter="camera", additional_parameter_value=0,
             xaxis_range=[0, 11], compare_trial="video_31", xaxis_title=x_title, yaxis_title=y_title,
             name="first_eHMI_off_non-yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=0, additional_parameter="camera", additional_parameter_value=0,
             xaxis_range=[0, 18], compare_trial="video_1", xaxis_title=x_title, yaxis_title=y_title,
             name="first_eHMI_off_yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=1, additional_parameter="camera", additional_parameter_value=1,
             xaxis_range=[0, 11], compare_trial="video_26", xaxis_title=x_title, yaxis_title=y_title,
             name="second_eHMI_on_non-yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=1, additional_parameter="camera", additional_parameter_value=1,
             xaxis_range=[0, 18], compare_trial="video_16", xaxis_title=x_title, yaxis_title=y_title,
             name="second_eHMI_on_yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=0, additional_parameter="camera", additional_parameter_value=1,
             xaxis_range=[0, 11], compare_trial="video_36", xaxis_title=x_title, yaxis_title=y_title,
             name="second_eHMI_off_non-yielding", margin=base_margin),
        dict(parameter="eHMIOn", parameter_value=0, additional_parameter="camera", additional_parameter_value=1,
             xaxis_range=[0, 18], compare_trial="video_6", xaxis_title=x_title, yaxis_title=y_title,
             name="second_eHMI_off_yielding", margin=base_margin),
    ]


def run_keypress_condition_plots(
    mapping_df: pd.DataFrame,
    trigger_threshold: float,
    output_subdir: str = None,
    save_final: bool = True,
) -> None:
    """Create all keypress figures for one trigger threshold."""
    logger.info(
        f"Generating keypress figures with trigger threshold {trigger_threshold:.3f}."
    )
    if output_subdir:
        logger.info(
            f"Keypress figures will be grouped under subfolder: {output_subdir}"
        )

    for spec in _keypress_plot_specs():
        logger.info(
            f"Creating keypress figure '{spec['name']}' at threshold {trigger_threshold:.3f}."
        )
        plot_spec = spec.copy()
        HMD.plot_column(
            mapping_df,
            trigger_threshold=trigger_threshold,
            output_subdir=output_subdir,
            save_final=save_final,
            **plot_spec,
        )


PARTICIPANT_RE = re.compile(r"^P(\d+)$")
VIDEO_RE = re.compile(r"participant_Yaw_video_(\d+)\.csv$")
CACHE_VERSION = "participant-event-heading-v6.1-20ms-pointwise-ttest-plotly-only"


def unity_heading_from_quaternion(q: np.ndarray) -> float:
    """Return Unity horizontal heading in radians for scalar-first quaternion."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return np.nan
    norm = np.linalg.norm(q)
    if norm <= np.finfo(float).eps:
        return np.nan
    w, x, y, z = q / norm
    forward_x = 2.0 * (x * z + w * y)
    forward_z = 1.0 - 2.0 * (x * x + y * y)
    if math.hypot(forward_x, forward_z) <= np.finfo(float).eps:
        return np.nan
    return math.atan2(forward_x, forward_z)


def average_quaternions_markley(quaternions: np.ndarray) -> np.ndarray:
    """Average scalar-first unit quaternions using Markley's eigenvector method."""
    q = np.asarray(quaternions, dtype=float)
    q = q[np.all(np.isfinite(q), axis=1)]
    if not len(q):
        return np.full(4, np.nan)
    norms = np.linalg.norm(q, axis=1)
    q = q[norms > np.finfo(float).eps]
    norms = norms[norms > np.finfo(float).eps]
    if not len(q):
        return np.full(4, np.nan)
    q = q / norms[:, None]
    reference = q[0]
    q[np.dot(q, reference) < 0] *= -1
    eigenvalues, eigenvectors = np.linalg.eigh(q.T @ q / len(q))
    avg = eigenvectors[:, np.argmax(eigenvalues)]
    return avg if avg[0] >= 0 else -avg


def parse_cell_heading(value: object) -> float:
    """Parse a matrix cell and return its within-bin horizontal heading."""
    if not isinstance(value, str) or value == "[]" or not value.strip():
        return np.nan
    # Matrix cells contain only numeric lists. Removing brackets and using
    # fromstring is substantially faster than literal_eval over ~1.5 M cells.
    raw = value.replace("[", "").replace("]", "")
    vals = np.fromstring(raw, sep=",")
    if not len(vals) or len(vals) % 4:
        return np.nan
    quats = vals.reshape(-1, 4)
    if len(quats) == 1:
        return unity_heading_from_quaternion(quats[0])
    return unity_heading_from_quaternion(average_quaternions_markley(quats))


def mean_in_window(time: np.ndarray, values: np.ndarray, start: float, end: float) -> float:
    mask = (time >= start) & (time < end) & np.isfinite(values)
    return float(np.mean(values[mask])) if np.any(mask) else np.nan


def smooth_nan(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that ignores missing samples."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values).astype(float)
    filled = np.where(np.isfinite(values), values, 0.0)
    kernel = np.ones(window, dtype=float)
    numerator = np.convolve(filled, kernel, mode="same")
    denominator = np.convolve(valid, kernel, mode="same")
    out = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    return out


def sustained_recovery_onset(
    time: np.ndarray,
    smooth_heading: np.ndarray,
    minimum_index: int,
    baseline: float,
    fraction: float,
    end_time: float,
    sustained_samples: int,
) -> float:
    """Start of the final sustained recovery before the participant is passed.

    A short early rebound is visible in several trajectories. Treating its first
    threshold crossing as the recovery onset would be misleading because the head
    subsequently turns left again. A candidate onset must therefore (a) remain
    above the threshold for the initial sustained interval and (b) remain above it
    for at least 90% of the samples through ``end_time``. The earliest candidate
    satisfying both requirements is returned.
    """
    minimum = smooth_heading[minimum_index]
    amplitude = baseline - minimum
    if not np.isfinite(amplitude) or amplitude < math.radians(20.0):
        return np.nan
    target = minimum + fraction * amplitude
    in_range = (np.arange(len(time)) >= minimum_index) & (time <= end_time)
    above = in_range & np.isfinite(smooth_heading) & (smooth_heading >= target)
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    end_indices = np.flatnonzero(in_range & np.isfinite(smooth_heading))
    if not len(end_indices):
        return np.nan
    final_index = int(end_indices[-1])
    for start in starts:
        initial_end = start + sustained_samples
        if initial_end > len(above) or not np.all(above[start:initial_end]):
            continue
        tail = above[start : final_index + 1]
        if len(tail) and np.mean(tail) >= 0.90:
            return float(time[start])
    return np.nan


def event_delta(time: np.ndarray, heading: np.ndarray, event_time: float) -> tuple[float, float, float]:
    if not np.isfinite(event_time):
        return np.nan, np.nan, np.nan
    pre = mean_in_window(time, heading, event_time - 0.5, event_time)
    post = mean_in_window(time, heading, event_time, event_time + 0.5)
    return pre, post, post - pre


def participant_pass_time(row: pd.Series) -> float:
    # camera=0 is avatar-first/participant-second: participant is passed at cross_p1.
    # camera=1 is participant-first/avatar-second: participant is passed at cross_p2.
    return float(row["cross_p1_time_s"] if int(row["camera"]) == 0 else row["cross_p2_time_s"])


def trial_features(
    matrix_path: Path,
    mapping_row: pd.Series,
    curve_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    matrix = pd.read_csv(matrix_path)
    time = pd.to_numeric(matrix["Timestamp"], errors="coerce").to_numpy(float)
    participant_columns = sorted(
        [c for c in matrix.columns if PARTICIPANT_RE.match(c)],
        key=lambda c: int(PARTICIPANT_RE.match(c).group(1)),
    )
    video_id = str(mapping_row["video_id"])
    pass_time = participant_pass_time(mapping_row)
    order = "AF_PS" if int(mapping_row["camera"]) == 0 else "PF_AS"
    spacing_m = 2 * int(mapping_row["distPed"])
    base_metadata = {
        "video_id": video_id,
        "yielding": int(mapping_row["yielding"]),
        "eHMIOn": int(mapping_row["eHMIOn"]),
        "order": order,
        "spacing_m": spacing_m,
        "participant_pass_time_s": pass_time,
    }

    features: list[dict[str, object]] = []
    for participant_column in participant_columns:
        participant = int(PARTICIPANT_RE.match(participant_column).group(1))
        heading = np.array([parse_cell_heading(v) for v in matrix[participant_column].array], dtype=float)
        valid = np.isfinite(heading)
        if np.sum(valid) < 10:
            continue
        # Preserve temporal continuity if an angle approaches the +/-pi boundary.
        unwrapped = heading.copy()
        unwrapped[valid] = np.unwrap(unwrapped[valid])
        baseline = mean_in_window(time, unwrapped, 0.02, 0.30)
        heading_bc = unwrapped - baseline
        smooth = smooth_nan(heading_bc, window=11)  # approximately 0.22 s at 50 Hz

        search = (time >= 0.50) & (time <= pass_time - 0.20) & np.isfinite(smooth)
        if np.any(search):
            search_indices = np.flatnonzero(search)
            minimum_index = int(search_indices[np.argmin(smooth[search])])
            minimum_heading = float(smooth[minimum_index])
            minimum_rel_pass = float(time[minimum_index] - pass_time)
        else:
            minimum_index = 0
            minimum_heading = np.nan
            minimum_rel_pass = np.nan

        record: dict[str, object] = {
            **base_metadata,
            "participant": participant,
            "valid_samples": int(np.sum(valid)),
            "valid_fraction": float(np.mean(valid)),
            "baseline_heading_deg": math.degrees(baseline),
            "minimum_heading_deg": math.degrees(minimum_heading),
            "minimum_time_rel_pass_s": minimum_rel_pass,
        }

        for fraction in (0.10, 0.20, 0.30):
            onset = sustained_recovery_onset(
                time,
                smooth,
                minimum_index,
                baseline=0.0,
                fraction=fraction,
                end_time=pass_time,
                sustained_samples=10,
            )
            record[f"recovery_onset_{int(fraction * 100)}pct_rel_pass_s"] = (
                onset - pass_time if np.isfinite(onset) else np.nan
            )

        far = mean_in_window(time, heading_bc, pass_time - 3.0, pass_time - 2.0)
        immediate_pre = mean_in_window(time, heading_bc, pass_time - 0.5, pass_time)
        immediate_post = mean_in_window(time, heading_bc, pass_time, pass_time + 0.5)
        at_pass = mean_in_window(time, heading_bc, pass_time - 0.1, pass_time + 0.1)
        record.update(
            {
                "far_pre_heading_deg": math.degrees(far),
                "immediate_pre_heading_deg": math.degrees(immediate_pre),
                "immediate_post_heading_deg": math.degrees(immediate_post),
                "heading_at_pass_deg": math.degrees(at_pass),
                "far_to_pre_change_deg": math.degrees(immediate_pre - far),
                "passage_change_deg": math.degrees(immediate_post - immediate_pre),
                "recovered_by_pass_deg": math.degrees(at_pass - minimum_heading),
            }
        )

        for event_name, event_column in (
            ("braking", "yield_start_time_s"),
            ("stopping", "yield_stop_time_s"),
            ("resumption", "yield_resume_time_s"),
        ):
            event_time = pd.to_numeric(mapping_row.get(event_column, np.nan), errors="coerce")
            pre, post, delta = event_delta(time, heading_bc, float(event_time))
            record[f"{event_name}_pre_heading_deg"] = math.degrees(pre)
            record[f"{event_name}_post_heading_deg"] = math.degrees(post)
            record[f"{event_name}_change_deg"] = math.degrees(delta)

        features.append(record)

        # Match the existing yaw figures' configured temporal resolution (20 ms
        # in the supplied configuration), then aggregate participants downstream.
        rel = time - pass_time
        bin_width = HEAD_HEADING_TIME_BIN_S
        n_event_bins = int(round(6.0 / bin_width))
        # Use the same small boundary tolerance as the project's established
        # half-open binning helper. Without it, floating-point representations
        # such as 0.2999999999999999 can create alternating empty 20-ms bins.
        bins = np.floor((rel + 5.0 + bin_width * 1e-9) / bin_width).astype(int)
        time_decimals = max(2, int(math.ceil(-math.log10(bin_width))) + 1)
        for bin_index in range(n_event_bins):  # [-5, +1) s
            selected = (bins == bin_index) & np.isfinite(heading_bc)
            if not np.any(selected):
                continue
            curve_records.append(
                {
                    **base_metadata,
                    "participant": participant,
                    "time_rel_pass_s": round(
                        -5.0 + (bin_index + 0.5) * bin_width,
                        time_decimals,
                    ),
                    "heading_deg": math.degrees(float(np.mean(heading_bc[selected]))),
                }
            )
    return features


def one_sample_summary(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n = len(values)
    if n < 2:
        return {"n": n, "estimate": np.nan, "se": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "t": np.nan, "p_raw": np.nan}
    estimate = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(n))
    critical = float(stats.t.ppf(0.975, n - 1))
    t_value, p_value = stats.ttest_1samp(values, 0.0)
    return {
        "n": n,
        "estimate": estimate,
        "se": se,
        "ci_lower": estimate - critical * se,
        "ci_upper": estimate + critical * se,
        "t": float(t_value),
        "p_raw": float(p_value),
    }


def factor_contrast(features: pd.DataFrame, metric: str, factor: str, high: object, low: object) -> pd.Series:
    grouped = features.groupby(["participant", factor], observed=True)[metric].mean().unstack(factor)
    return grouped[high] - grouped[low]


def interaction_contrast(
    features: pd.DataFrame,
    metric: str,
    factor_a: str,
    high_a: object,
    low_a: object,
    factor_b: str,
    high_b: object,
    low_b: object,
) -> pd.Series:
    grouped = features.groupby(["participant", factor_a, factor_b], observed=True)[metric].mean().unstack([factor_a, factor_b])
    return (grouped[(high_a, high_b)] - grouped[(high_a, low_b)]) - (
        grouped[(low_a, high_b)] - grouped[(low_a, low_b)]
    )


def holm_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return pd.Series(adjusted, index=p_values.index)
    order = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        candidate = (m - rank) * p[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def repeated_measures_anova(matrix: pd.DataFrame) -> dict[str, float | int]:
    """One-factor repeated-measures ANOVA on complete participant-by-level matrix."""
    complete = matrix.dropna(axis=0, how="any").to_numpy(float)
    n, k = complete.shape if complete.ndim == 2 else (0, 0)
    if n < 2 or k < 2:
        return {"n": n, "levels": k, "df1": np.nan, "df2": np.nan, "F": np.nan, "p_raw": np.nan}
    grand = complete.mean()
    participant_means = complete.mean(axis=1)
    level_means = complete.mean(axis=0)
    ss_total = np.sum((complete - grand) ** 2)
    ss_participant = k * np.sum((participant_means - grand) ** 2)
    ss_level = n * np.sum((level_means - grand) ** 2)
    ss_error = ss_total - ss_participant - ss_level
    df1 = k - 1
    df2 = (n - 1) * (k - 1)
    f_value = (ss_level / df1) / (ss_error / df2)
    return {
        "n": n,
        "levels": k,
        "df1": df1,
        "df2": df2,
        "F": float(f_value),
        "p_raw": float(stats.f.sf(f_value, df1, df2)),
    }


def build_contrast_table(features: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    definitions = [
        ("yielding_minus_non_yielding", lambda d, m: factor_contrast(d, m, "yielding", 1, 0)),
        ("conditional_eHMI_on_minus_off", lambda d, m: factor_contrast(d, m, "eHMIOn", 1, 0)),
        ("PF_AS_minus_AF_PS", lambda d, m: factor_contrast(d, m, "order", "PF_AS", "AF_PS")),
        (
            "yielding_x_conditional_eHMI",
            lambda d, m: interaction_contrast(d, m, "yielding", 1, 0, "eHMIOn", 1, 0),
        ),
        (
            "yielding_x_order",
            lambda d, m: interaction_contrast(d, m, "yielding", 1, 0, "order", "PF_AS", "AF_PS"),
        ),
        (
            "conditional_eHMI_x_order",
            lambda d, m: interaction_contrast(d, m, "eHMIOn", 1, 0, "order", "PF_AS", "AF_PS"),
        ),
    ]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for name, function in definitions:
            summary = one_sample_summary(function(features, metric))
            rows.append({"metric": metric, "contrast": name, **summary})
    result = pd.DataFrame(rows)
    result["p_holm_within_metric"] = result.groupby("metric", group_keys=False)["p_raw"].apply(holm_adjust)
    return result


def build_distance_table(features: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        base = features.groupby(["participant", "spacing_m"], observed=True)[metric].mean().unstack("spacing_m")
        rows.append({"metric": metric, "test": "distance", **repeated_measures_anova(base)})
        for factor, high, low in (("yielding", 1, 0), ("eHMIOn", 1, 0), ("order", "PF_AS", "AF_PS")):
            cells = features.groupby(["participant", "spacing_m", factor], observed=True)[metric].mean().unstack(factor)
            difference = (cells[high] - cells[low]).unstack("spacing_m")
            rows.append(
                {"metric": metric, "test": f"distance_x_{factor}", **repeated_measures_anova(difference)}
            )
    result = pd.DataFrame(rows)
    result["p_holm_within_metric"] = result.groupby("metric", group_keys=False)["p_raw"].apply(holm_adjust)
    return result


def build_simple_contrast_table(features: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Condition-specific paired contrasts, with Holm correction over 12 per metric."""
    rows: list[dict[str, object]] = []
    for metric in metrics:
        for order in ("AF_PS", "PF_AS"):
            for ehmi in (0, 1):
                subset = features[(features["order"] == order) & (features["eHMIOn"] == ehmi)]
                values = factor_contrast(subset, metric, "yielding", 1, 0)
                rows.append(
                    {
                        "metric": metric,
                        "family": "yielding_within_order_eHMI",
                        "contrast": f"yielding_minus_non_yielding|order={order}|eHMI={ehmi}",
                        **one_sample_summary(values),
                    }
                )
        for yielding in (0, 1):
            for ehmi in (0, 1):
                subset = features[(features["yielding"] == yielding) & (features["eHMIOn"] == ehmi)]
                values = factor_contrast(subset, metric, "order", "PF_AS", "AF_PS")
                rows.append(
                    {
                        "metric": metric,
                        "family": "order_within_yielding_eHMI",
                        "contrast": f"PF_AS_minus_AF_PS|yielding={yielding}|eHMI={ehmi}",
                        **one_sample_summary(values),
                    }
                )
        for yielding in (0, 1):
            for order in ("AF_PS", "PF_AS"):
                subset = features[(features["yielding"] == yielding) & (features["order"] == order)]
                values = factor_contrast(subset, metric, "eHMIOn", 1, 0)
                rows.append(
                    {
                        "metric": metric,
                        "family": "eHMI_within_yielding_order",
                        "contrast": f"eHMI_on_minus_off|yielding={yielding}|order={order}",
                        **one_sample_summary(values),
                    }
                )
    result = pd.DataFrame(rows)
    result["p_holm_12_within_metric"] = result.groupby("metric", group_keys=False)["p_raw"].apply(holm_adjust)
    return result


def build_yielding_event_table(features: pd.DataFrame) -> pd.DataFrame:
    yielding = features[features["yielding"] == 1]
    rows: list[dict[str, object]] = []
    for metric in ("braking_change_deg", "stopping_change_deg", "resumption_change_deg"):
        overall = yielding.groupby("participant", observed=True)[metric].mean()
        rows.append({"metric": metric, "contrast": "post_minus_pre", **one_sample_summary(overall)})
        rows.append(
            {
                "metric": metric,
                "contrast": "PF_AS_minus_AF_PS",
                **one_sample_summary(factor_contrast(yielding, metric, "order", "PF_AS", "AF_PS")),
            }
        )
        rows.append(
            {
                "metric": metric,
                "contrast": "eHMI_on_minus_off",
                **one_sample_summary(factor_contrast(yielding, metric, "eHMIOn", 1, 0)),
            }
        )
        rows.append(
            {
                "metric": metric,
                "contrast": "eHMI_x_order",
                **one_sample_summary(
                    interaction_contrast(yielding, metric, "eHMIOn", 1, 0, "order", "PF_AS", "AF_PS")
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm_12"] = holm_adjust(result["p_raw"])
    return result


def condition_summary(features: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    participant_cells = (
        features.groupby(["participant", "yielding", "eHMIOn", "order", "spacing_m"], observed=True)[metrics]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    factors = ["yielding", "eHMIOn", "order", "spacing_m"]
    for keys, group in participant_cells.groupby(factors, observed=True):
        base = dict(zip(factors, keys))
        for metric in metrics:
            values = group[metric].dropna().to_numpy(float)
            n = len(values)
            mean = float(np.mean(values)) if n else np.nan
            se = float(np.std(values, ddof=1) / math.sqrt(n)) if n > 1 else np.nan
            critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else np.nan
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "n_participants": n,
                    "mean": mean,
                    "se": se,
                    "ci_lower": mean - critical * se,
                    "ci_upper": mean + critical * se,
                }
            )
    return pd.DataFrame(rows)


def curve_summary(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Average trials within participant before estimating between-participant uncertainty.
    participant = (
        curves.groupby(["participant", "yielding", "eHMIOn", "order", "time_rel_pass_s"], observed=True)["heading_deg"]
        .mean()
        .reset_index()
    )
    group_cols = ["yielding", "eHMIOn", "order", "time_rel_pass_s"]
    summary = participant.groupby(group_cols, observed=True)["heading_deg"].agg(["count", "mean", "std"]).reset_index()
    summary = summary.rename(columns={"count": "n_participants"})
    summary["se"] = summary["std"] / np.sqrt(summary["n_participants"])
    critical = stats.t.ppf(0.975, summary["n_participants"] - 1)
    summary["ci_lower"] = summary["mean"] - critical * summary["se"]
    summary["ci_upper"] = summary["mean"] + critical * summary["se"]
    return participant, summary.drop(columns="std")


def smooth_heading_for_display(values: pd.Series | np.ndarray) -> np.ndarray:
    """Use the established yaw-plot smoothing configuration for display only."""
    numeric = pd.Series(values, dtype=float).fillna(0.0).tolist()
    if bool(common.get_configs("smoothen_signal")):
        return np.asarray(HMD.smoothen_filter(numeric), dtype=float)
    return np.asarray(numeric, dtype=float)


def passage_order_tests(features: pd.DataFrame) -> pd.DataFrame:
    """Paired order contrasts for the pre-specified passage window in each panel.

    The event-aligned figure is descriptive at every time point.  To avoid a
    serially dependent sequence of pointwise tests, significance annotations
    refer only to the participant-level mean heading in the 200-ms passage
    window (the same outcome used in the numerical passage analysis).
    """
    participant = (
        features.groupby(["participant", "yielding", "order"], observed=True)["heading_at_pass_deg"]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for yielding in (0, 1):
        wide = participant[participant["yielding"].eq(yielding)].pivot(
            index="participant", columns="order", values="heading_at_pass_deg"
        )
        wide = wide.dropna(subset=["AF_PS", "PF_AS"])
        summary = one_sample_summary(wide["PF_AS"] - wide["AF_PS"])
        rows.append({"yielding": yielding, **summary})
    result = pd.DataFrame(rows)
    result["p_holm_across_panels"] = holm_adjust(result["p_raw"])

    def significance_stars(value: float) -> str:
        if value < 0.001:
            return "***"
        if value < 0.01:
            return "**"
        if value < 0.05:
            return "*"
        return "n.s."

    result["significance"] = result["p_holm_across_panels"].map(significance_stars)
    return result


def pointwise_order_ttests(
    curves: pd.DataFrame,
    p_threshold: float = 0.001,
) -> pd.DataFrame:
    """Match the existing yaw figures' two-sided paired t-test marker method.

    For every configured event-aligned yaw bin, trials are first averaged within each
    participant, AV-behaviour panel, and relative-order condition.  PF/AS and
    AF/PS are then compared with a two-sided paired t-test.  As in the existing
    HMD plots, a bin is marked when its raw p value is below the configured
    ``p_value`` threshold (0.001 in the supplied configuration).
    """
    participant = (
        curves.groupby(
            ["participant", "yielding", "order", "time_rel_pass_s"],
            observed=True,
        )["heading_deg"]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for (yielding, time_rel), group in participant.groupby(
        ["yielding", "time_rel_pass_s"], observed=True
    ):
        wide = group.pivot(index="participant", columns="order", values="heading_deg")
        if not {"AF_PS", "PF_AS"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["AF_PS", "PF_AS"])
        if len(wide) < 2:
            t_statistic, p_value, estimate = np.nan, 1.0, np.nan
        else:
            t_statistic, p_value = stats.ttest_rel(
                wide["PF_AS"],
                wide["AF_PS"],
                alternative="two-sided",
            )
            if not np.isfinite(p_value):
                p_value = 1.0
            estimate = float((wide["PF_AS"] - wide["AF_PS"]).mean())
        rows.append(
            {
                "yielding": int(yielding),
                "time_rel_pass_s": float(time_rel),
                "n_pairs": int(len(wide)),
                "PF_AS_minus_AF_PS_deg": estimate,
                "t_statistic": float(t_statistic),
                "p_raw": float(p_value),
                "p_threshold": float(p_threshold),
                "significant": int(float(p_value) < float(p_threshold)),
            }
        )
    return pd.DataFrame(rows).sort_values(["yielding", "time_rel_pass_s"]).reset_index(drop=True)


def _significant_time_ranges(panel_tests: pd.DataFrame) -> str:
    """Compact consecutive significant event-aligned bins for the results log."""
    times = np.sort(
        panel_tests.loc[panel_tests["significant"].astype(bool), "time_rel_pass_s"].to_numpy(float)
    )
    if len(times) == 0:
        return "none"
    all_times = np.sort(panel_tests["time_rel_pass_s"].dropna().unique().astype(float))
    if len(all_times) > 1:
        consecutive_tolerance = 1.01 * float(np.median(np.diff(all_times)))
    else:
        consecutive_tolerance = 1.01 * HEAD_HEADING_TIME_BIN_S
    ranges: list[tuple[float, float]] = []
    start = previous = float(times[0])
    for value in times[1:]:
        value = float(value)
        if value - previous > consecutive_tolerance:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return ", ".join(
        f"{start:.2f} s" if abs(start - end) < 1e-9 else f"{start:.2f} to {end:.2f} s"
        for start, end in ranges
    )


def create_event_aligned_plotly(curves: pd.DataFrame, pointwise_tests: pd.DataFrame) -> Any:
    """Return the event-aligned figure using the project's Plotly stack."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    participant = (
        curves.groupby(["participant", "yielding", "order", "time_rel_pass_s"], observed=True)["heading_deg"]
        .mean()
        .reset_index()
    )
    summary = (
        participant.groupby(["yielding", "order", "time_rel_pass_s"], observed=True)["heading_deg"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    critical = stats.t.ppf(0.975, summary["count"] - 1)
    summary["lower"] = summary["mean"] - critical * summary["se"]
    summary["upper"] = summary["mean"] + critical * summary["se"]
    data_min = float(summary["lower"].min())
    data_max = float(summary["upper"].max())
    marker_y = data_min - HEAD_HEADING_TTEST_ROW_HEIGHT_DEG
    plot_min = data_min - max(4.0 * HEAD_HEADING_TTEST_ROW_HEIGHT_DEG, 2.0)
    plot_max = data_max + max(0.02 * (data_max - data_min), 1.0)
    labels = {"AF_PS": "Avatar first / participant second", "PF_AS": "Participant first / avatar second"}
    colors = {"AF_PS": "#3569b8", "PF_AS": "#d95032"}
    fills = {"AF_PS": "rgba(53,105,184,0.18)", "PF_AS": "rgba(217,80,50,0.18)"}
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Non-yielding", "Yielding"))
    fig.update_annotations(
        font=dict(
            family=common.get_configs("font_family"),
            size=common.get_configs("font_size")+8,
        )
    )
    for column, yielding in enumerate((0, 1), 1):
        for order in ("AF_PS", "PF_AS"):
            group = summary[(summary["yielding"] == yielding) & (summary["order"] == order)].sort_values(
                "time_rel_pass_s"
            ).copy()
            group["mean"] = smooth_heading_for_display(group["mean"])
            group["lower"] = smooth_heading_for_display(group["lower"])
            group["upper"] = smooth_heading_for_display(group["upper"])
            fig.add_trace(
                go.Scatter(
                    x=group["time_rel_pass_s"],
                    y=group["upper"],
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            fig.add_trace(
                go.Scatter(
                    x=group["time_rel_pass_s"],
                    y=group["lower"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fills[order],
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=1,
                col=column,
            )
            fig.add_trace(
                go.Scatter(
                    x=group["time_rel_pass_s"],
                    y=group["mean"],
                    mode="lines",
                    line=dict(color=colors[order], width=HEAD_HEADING_LINE_WIDTH),
                    name=labels[order],
                    legendgroup=order,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        fig.add_vline(x=0, line_dash="dash", line_color="#222222", row=1, col=column)
        fig.add_hline(y=0, line_dash="dot", line_color="#777777", row=1, col=column)
        fig.update_xaxes(
            title_text="Time relative to vehicle passing participant (s)",
            title_font=dict(
                family=common.get_configs("font_family"),
                size=common.get_configs("font_size")+8,
            ),
            tickfont=dict(
                family=common.get_configs("font_family"),
                size=common.get_configs("font_size")+8,
            ),
            range=[-5, 1],
            dtick=HEAD_HEADING_XAXIS_STEP,
            row=1,
            col=column,
        )
        significant = pointwise_tests[
            pointwise_tests["yielding"].eq(yielding)
            & pointwise_tests["significant"].astype(bool)
        ]
        fig.add_trace(
            go.Scatter(
                x=significant["time_rel_pass_s"],
                y=np.full(len(significant), marker_y),
                mode="text",
                text=["*"] * len(significant),
                name="__significance_markers__",
                textfont=dict(
                    family=common.get_configs("font_family"),
                    size=HEAD_HEADING_TTEST_MARKER_SIZE,
                    color="black",
                ),
                customdata=significant["p_raw"],
                hovertemplate="PF/AS vs AF/PS: time=%{x}, p=%{customdata:.4g}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    fig.update_yaxes(
        title_text="Baseline-corrected horizontal head heading (degrees)",
        title_font=dict(
            family=common.get_configs("font_family"),
            size=common.get_configs("font_size")+8,
        ),
        tickfont=dict(
            family=common.get_configs("font_family"),
            size=common.get_configs("font_size")+8,
        ),
        row=1,
        col=1,
    )

    fig.update_yaxes(
        range=[plot_min, plot_max],
        dtick=HEAD_HEADING_YAXIS_STEP_DEG,
        tickfont=dict(
            family=common.get_configs("font_family"),
            size=common.get_configs("font_size")+8,
        ),
    )
    fig.update_layout(
        template=common.get_configs("plotly_template"),
        font=dict(
            family=common.get_configs("font_family"),
            size=common.get_configs("font_size")+8,
        ),
        legend=dict(
            orientation="h",
            y=HEAD_HEADING_LEGEND_Y,
            x=HEAD_HEADING_LEGEND_X,
        ),
        margin=dict(l=120, r=2, t=75, b=60),
    )
    return fig


def create_passage_summary_plotly(features: pd.DataFrame) -> Any:
    """Return a manuscript-ready passage-heading point/interval figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    participant = (
        features.groupby(["participant", "yielding", "eHMIOn", "order"], observed=True)["heading_at_pass_deg"]
        .mean()
        .reset_index()
    )
    summary = (
        participant.groupby(["yielding", "eHMIOn", "order"], observed=True)["heading_at_pass_deg"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    summary["half_ci"] = stats.t.ppf(0.975, summary["count"] - 1) * summary["se"]
    labels = {"AF_PS": "Avatar first / participant second", "PF_AS": "Participant first / avatar second"}
    colors = {0: "#3569b8", 1: "#d95032"}
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Non-yielding", "Yielding"))
    for column, yielding in enumerate((0, 1), 1):
        for ehmi in (0, 1):
            group = summary[(summary["yielding"] == yielding) & (summary["eHMIOn"] == ehmi)]
            group = group.set_index("order").loc[["AF_PS", "PF_AS"]].reset_index()
            fig.add_trace(
                go.Scatter(
                    x=[labels[x] for x in group["order"]],
                    y=group["mean"],
                    error_y=dict(type="data", array=group["half_ci"], visible=True, thickness=1.5, width=5),
                    mode="lines+markers",
                    marker=dict(size=9),
                    line=dict(color=colors[ehmi], width=2),
                    name="Conditional eHMI" if ehmi else "No eHMI",
                    legendgroup=f"eHMI{ehmi}",
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        fig.add_hline(y=0, line_dash="dot", line_color="#777777", row=1, col=column)
    fig.update_yaxes(title_text="Heading at participant passage (degrees)", row=1, col=1)
    fig.update_layout(
        template="plotly_white",
        title=dict(text="Horizontal head heading at participant passage", x=0.5),
        legend=dict(orientation="h", y=1.13, x=0),
        margin=dict(l=80, r=30, t=100, b=120),
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text="Participant-marginal means averaged over spacing; error bars are between-participant 95% CIs.",
                x=0.5,
                y=-0.24,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12),
            )
        ],
    )
    return fig



def _save_plotly_only(
    fig: Any,
    name: str,
    width: int,
    height: int,
    open_browser: bool = True,
) -> None:
    """Save one figure with Plotly only to the configured output root."""
    import plotly.offline as plotly_offline

    figure_root = Path(common.get_configs("output"))
    figure_root.mkdir(parents=True, exist_ok=True)
    output_base = figure_root / name

    plotly_offline.plot(
        fig,
        filename=str(output_base.with_suffix(".html")),
        auto_open=bool(open_browser),
    )
    logger.info(f"Saved Plotly figure: {output_base.with_suffix('.html')}")

    for suffix, label in ((".eps", "EPS"), (".png", "PNG")):
        output_path = output_base.with_suffix(suffix)
        try:
            fig.write_image(str(output_path), width=width, height=height)
            logger.info(f"Saved Plotly figure: {output_path}")
        except Exception as exc:
            logger.warning(
                f"Skipping {label} export for '{name}' because Plotly/Kaleido "
                f"could not create the file: {exc}"
            )




def _mapping_hash(mapping: pd.DataFrame) -> str:
    stable = mapping.sort_values("video_id").reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(stable, index=True).values.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_signature(
    matrix_paths: list[Path],
    mapping: pd.DataFrame,
    source_cache_key: str | None = None,
) -> dict[str, object]:
    signature: dict[str, object] = {
        "cache_version": CACHE_VERSION,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mapping_sha256": _mapping_hash(mapping),
        "source_cache_key": source_cache_key,
    }
    if source_cache_key is None:
        # Standalone use has no validated processed-data pickle. Hash the
        # matrix contents so cache reuse never depends on file timestamps.
        signature["matrices"] = [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in matrix_paths
        ]
    else:
        # analysis.py restores these files directly from a validated pickle.
        # Its created_utc identifier changes whenever raw data are rebuilt.
        signature["matrices"] = [
            {"name": path.name, "size": path.stat().st_size}
            for path in matrix_paths
        ]
    return signature


def _load_manifest(path: Path) -> dict[str, object] | None:
    try:
        manifest = pd.read_csv(path)
        if len(manifest) != 1 or "signature_json" not in manifest.columns:
            return None
        return json.loads(str(manifest.loc[0, "signature_json"]))
    except (OSError, ValueError, TypeError, pd.errors.ParserError):
        return None


def _p_text(value: float, label: str = "Holm p") -> str:
    """Format p values consistently for the human-readable results log."""
    return f"{label}<.001" if value < 0.001 else f"{label}={value:.3f}"


def _log_results(
    features: pd.DataFrame,
    contrasts: pd.DataFrame,
    distance: pd.DataFrame,
    event_contrasts: pd.DataFrame,
    figure_tests: pd.DataFrame,
    pointwise_tests: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    logger.info('\n=== Participant-level horizontal head-heading analysis ===')
    quality_map = quality.set_index("measure")["value"]
    logger.info(
        f"Records: {int(quality_map['trials']):,} participant-trials; "
        f"participants: {int(quality_map['participants'])}; "
        f"mean valid samples: {100.0 * float(quality_map['mean_valid_fraction']):.2f}%."
    )

    minimum_cells = (
        features.groupby(["participant", "yielding", "order"], observed=True)["minimum_heading_deg"]
        .mean()
        .reset_index()
    )
    minimum_rows: list[dict[str, object]] = []
    for (yielding, order), group in minimum_cells.groupby(["yielding", "order"], observed=True):
        values = group["minimum_heading_deg"].dropna()
        n = len(values)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(n))
        half = float(stats.t.ppf(0.975, n - 1) * se)
        minimum_rows.append(
            {
                "AV behaviour": "Yielding" if yielding else "Non-yielding",
                "Relative order": order,
                "Mean_deg": mean,
                "CI_low": mean - half,
                "CI_high": mean + half,
            }
        )
    minimum_table = pd.DataFrame(minimum_rows).to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}",
    )
    logger.info(
        "Minimum leftward heading before participant passage "
        f"(participant-marginal degrees; 95% CI):\n{minimum_table}"
    )
    minimum = contrasts[contrasts["metric"].eq("minimum_heading_deg")].set_index("contrast")
    minimum_main = minimum.loc[
        ["yielding_minus_non_yielding", "PF_AS_minus_AF_PS", "conditional_eHMI_on_minus_off"]
    ]
    logger.info(
        f"Minimum-heading main contrasts were not supported after Holm correction (smallest {_p_text(float(minimum_main['p_holm_within_metric'].min()))})."
    )
    participant_cells = (
        features.groupby(["participant", "yielding", "order"], observed=True)["heading_at_pass_deg"]
        .mean()
        .reset_index()
    )
    cell_rows: list[dict[str, object]] = []
    for (yielding, order), group in participant_cells.groupby(["yielding", "order"], observed=True):
        values = group["heading_at_pass_deg"].dropna()
        n = len(values)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(n))
        half = float(stats.t.ppf(0.975, n - 1) * se)
        cell_rows.append(
            {
                "AV behaviour": "Yielding" if yielding else "Non-yielding",
                "Relative order": order,
                "Mean_deg": mean,
                "CI_low": mean - half,
                "CI_high": mean + half,
            }
        )
    passage_table = pd.DataFrame(cell_rows).to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}",
    )
    logger.info(
        "Heading at participant passage "
        f"(participant-marginal degrees; 95% CI):\n{passage_table}"
    )

    passage = contrasts[contrasts["metric"].eq("heading_at_pass_deg")].set_index("contrast")
    for contrast_name, label in (
        ("yielding_minus_non_yielding", "Yielding minus non-yielding"),
        ("PF_AS_minus_AF_PS", "Participant-first minus avatar-first order"),
        ("conditional_eHMI_on_minus_off", "Conditional eHMI on minus off"),
    ):
        row = passage.loc[contrast_name]
        logger.info(
            f"{label}: {row['estimate']:.2f} deg, "
            f"95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}], "
            f"{_p_text(float(row['p_holm_within_metric']))}."
        )
    for yielding, label in ((0, "Non-yielding"), (1, "Yielding")):
        row = figure_tests[figure_tests["yielding"].eq(yielding)].iloc[0]
        logger.info(
            f"{label} passage-window order contrast (PF/AS minus AF/PS): "
            f"{row['estimate']:.2f} deg, "
            f"95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}], "
            f"{_p_text(float(row['p_holm_across_panels']))} across the two panels."
        )
        panel_pointwise = pointwise_tests[pointwise_tests["yielding"].eq(yielding)]
        significant_count = int(panel_pointwise["significant"].sum())
        logger.info(
            f"{label} event-aligned pointwise paired t-tests (PF/AS vs AF/PS): "
            f"{significant_count} of {len(panel_pointwise)} "
            f"{HEAD_HEADING_TIME_BIN_MS}-ms bins had raw "
            f"p<{float(panel_pointwise['p_threshold'].iloc[0]):.3f}; "
            f"marker ranges: {_significant_time_ranges(panel_pointwise)}."
        )
    spacing = distance[
        distance["metric"].eq("heading_at_pass_deg") & distance["test"].eq("distance")
    ].iloc[0]
    logger.info(
        f"Global spacing test for passage heading: "
        f"F({spacing['df1']:.0f},{spacing['df2']:.0f})={spacing['F']:.2f}, "
        f"p={spacing['p_raw']:.3f}, Holm p={spacing['p_holm_within_metric']:.3f}."
    )
    onset = contrasts[
        contrasts["metric"].eq("recovery_onset_20pct_rel_pass_s")
        & contrasts["contrast"].eq("yielding_minus_non_yielding")
    ].iloc[0]
    available = int(quality_map["onset_20pct_available"])
    logger.info(
        "Exploratory final-sustained 20% recovery onset, yielding minus "
        f"non-yielding: {onset['estimate']:.2f} s, "
        f"95% CI [{onset['ci_lower']:.2f}, {onset['ci_upper']:.2f}], "
        f"Holm p<.001; available in {available}/{len(features)} trials "
        f"({100.0 * available / len(features):.1f}%)."
    )
    stop_rows = event_contrasts[event_contrasts["metric"].eq("stopping_change_deg")].set_index("contrast")
    for contrast_name, label in (
        ("post_minus_pre", "Stop-aligned post-minus-pre change"),
        ("PF_AS_minus_AF_PS", "Order difference in stop-aligned change"),
        ("eHMI_on_minus_off", "eHMI difference in stop-aligned change"),
    ):
        row = stop_rows.loc[contrast_name]
        logger.info(
            f"{label}: {row['estimate']:.2f} deg, "
            f"95% CI [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}], "
            f"{_p_text(float(row['p_holm_12']))}."
        )
    logger.info(
        'Interpretation: horizontal HMD orientation only, not gaze or attention;' \
        'relative order remains a compound geometry/visibility factor.'
    )
    logger.info('=========================================================\n')


def run_head_heading_analysis(
    matrices: str | Path,
    mapping: str | Path | pd.DataFrame,
    output: str | Path,
    force: bool = False,
    figure_saver: Callable[..., Any] | None = None,
    source_cache_key: str | None = None,
) -> dict[str, object]:
    """Run or reuse the participant-level analysis, log results, and save figures.

    Parameters
    ----------
    matrices
        Directory containing ``participant_Yaw_video_1.csv`` through video 40.
    mapping
        Mapping CSV path or an already loaded mapping DataFrame.
    output
        Directory for statistical CSVs. Figures are written through Plotly.
    force
        Recompute even when the input signature and cached outputs match.
    figure_saver
        Optional bound ``HMD.save_plotly`` method. When supplied, the two
        manuscript figures are saved through the project's standard Plotly
        pipeline as HTML, EPS and PNG, with browser opening enabled.
        When omitted, a Plotly-only fallback writes the same formats directly
        to the configured output folder.
    source_cache_key
        Stable identifier for a validated processed-data pickle. The integrated
        pipeline passes its ``created_utc`` value so restoring identical matrix
        CSVs does not invalidate this cache merely by changing file timestamps.
    """
    matrices = Path(matrices)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    logger.info(
        f'Head-heading event analysis: {CACHE_VERSION}; two-sided paired pointwise t-test threshold p<{HEAD_HEADING_POINTWISE_P_THRESHOLD:.3f}.'
    )
    mapping_df = mapping.copy() if isinstance(mapping, pd.DataFrame) else pd.read_csv(mapping)
    mapping_df = mapping_df[mapping_df["video_id"].astype(str).str.fullmatch(r"video_\d+")].copy()
    mapping_indexed = mapping_df.set_index("video_id", drop=False)
    matrix_paths = sorted(
        matrices.glob("participant_Yaw_video_*.csv"),
        key=lambda path: int(VIDEO_RE.search(path.name).group(1)),
    )
    if len(matrix_paths) != 40:
        raise ValueError(
            f"Expected 40 participant yaw matrices in {matrices}, found {len(matrix_paths)}. "
            "Run the existing HMD.plot_yaw calls first to generate them."
        )

    paths = {
        "features": output / "head_heading_trial_features.csv",
        "contrasts": output / "head_heading_repeated_measures_contrasts.csv",
        "simple": output / "head_heading_simple_contrasts.csv",
        "events": output / "head_heading_yielding_event_contrasts.csv",
        "distance": output / "head_heading_distance_omnibus.csv",
        "condition": output / "head_heading_condition_summary.csv",
        "participant_curves": output / "head_heading_event_aligned_participant.csv",
        "curve_summary": output / "head_heading_event_aligned_summary.csv",
        "figure_tests": output / "head_heading_event_figure_passage_tests.csv",
        "pointwise_tests": output / "head_heading_event_aligned_pointwise_ttests.csv",
        "quality": output / "head_heading_quality_summary.csv",
        # CSV is deliberate: ProcessedExperimentCache.capture_statistical_tables
        # includes it in the single processed-data pickle and restores it later.
        "manifest": output / "head_heading_cache_manifest.csv",
    }
    signature = _input_signature(
        matrix_paths,
        mapping_df,
        source_cache_key=source_cache_key,
    )
    cache_ready = (
        not force
        and all(path.exists() for key, path in paths.items() if key != "manifest")
        and _load_manifest(paths["manifest"]) == signature
    )

    if cache_ready:
        logger.info(f'Reusing participant-level head-heading results from: {output}')
        features = pd.read_csv(paths["features"])
        contrasts = pd.read_csv(paths["contrasts"])
        simple = pd.read_csv(paths["simple"])
        event_contrasts = pd.read_csv(paths["events"])
        distance = pd.read_csv(paths["distance"])
        conditions = pd.read_csv(paths["condition"])
        participant_curves = pd.read_csv(paths["participant_curves"])
        summarized_curves = pd.read_csv(paths["curve_summary"])
        figure_tests = pd.read_csv(paths["figure_tests"])
        pointwise_tests = pd.read_csv(paths["pointwise_tests"])
        quality = pd.read_csv(paths["quality"])
    else:
        reason = "force=True" if force else "cache missing or inputs/code changed"
        logger.info(f'Computing participant-level head-heading analysis ({reason}).')
        feature_records: list[dict[str, object]] = []
        curve_records: list[dict[str, object]] = []
        for index, matrix_path in enumerate(matrix_paths, 1):
            video_number = int(VIDEO_RE.search(matrix_path.name).group(1))
            video_id = f"video_{video_number}"
            if video_id not in mapping_indexed.index:
                raise ValueError(f"No mapping row for {video_id}")
            logger.info(f'Head-heading features: [{index:02d}/40] {video_id}')
            feature_records.extend(trial_features(matrix_path, mapping_indexed.loc[video_id], curve_records))

        features = pd.DataFrame(feature_records).sort_values(["participant", "video_id"])
        curves = pd.DataFrame(curve_records)
        for threshold in (10, 20, 30):
            onset_name = f"recovery_onset_{threshold}pct_rel_pass_s"
            features[f"reached_{threshold}pct_by_pass"] = features[onset_name].notna().astype(int)
        metrics = [
            "recovery_onset_10pct_rel_pass_s",
            "recovery_onset_20pct_rel_pass_s",
            "recovery_onset_30pct_rel_pass_s",
            "heading_at_pass_deg",
            "far_to_pre_change_deg",
            "passage_change_deg",
            "recovered_by_pass_deg",
            "minimum_heading_deg",
        ]
        contrasts = build_contrast_table(features, metrics)
        simple = build_simple_contrast_table(features, metrics)
        event_contrasts = build_yielding_event_table(features)
        distance = build_distance_table(features, metrics)
        conditions = condition_summary(features, metrics)
        participant_curves, summarized_curves = curve_summary(curves)
        figure_tests = passage_order_tests(features)
        pointwise_tests = pointwise_order_ttests(
            participant_curves,
            p_threshold=HEAD_HEADING_POINTWISE_P_THRESHOLD,
        )
        quality = pd.DataFrame(
            {
                "measure": [
                    "trials",
                    "participants",
                    "mean_valid_fraction",
                    "minimum_valid_fraction",
                    "onset_10pct_available",
                    "onset_20pct_available",
                    "onset_30pct_available",
                ],
                "value": [
                    len(features),
                    features["participant"].nunique(),
                    features["valid_fraction"].mean(),
                    features["valid_fraction"].min(),
                    features["recovery_onset_10pct_rel_pass_s"].notna().sum(),
                    features["recovery_onset_20pct_rel_pass_s"].notna().sum(),
                    features["recovery_onset_30pct_rel_pass_s"].notna().sum(),
                ],
            }
        )
        for key, frame in (
            ("features", features),
            ("contrasts", contrasts),
            ("simple", simple),
            ("events", event_contrasts),
            ("distance", distance),
            ("condition", conditions),
            ("participant_curves", participant_curves),
            ("curve_summary", summarized_curves),
            ("figure_tests", figure_tests),
            ("pointwise_tests", pointwise_tests),
            ("quality", quality),
        ):
            frame.to_csv(paths[key], index=False)
            logger.info(f'Saved head-heading CSV: {paths[key]}')
        pd.DataFrame(
            {"signature_json": [json.dumps(signature, sort_keys=True)]}
        ).to_csv(paths["manifest"], index=False)

    event_aligned_figure = create_event_aligned_plotly(
        participant_curves,
        pointwise_tests,
    )
    passage_summary_figure = create_passage_summary_plotly(features)

    figure_root = Path(common.get_configs("figures"))
    legacy_figure_dir = figure_root / "head_heading"
    if legacy_figure_dir.is_dir():
        figure_root.mkdir(parents=True, exist_ok=True)
        legacy_files = [path for path in legacy_figure_dir.rglob("*") if path.is_file()]
        for legacy_path in legacy_files:
            destination = figure_root / legacy_path.name
            legacy_path.replace(destination)
            logger.info(f"Moved legacy head-heading figure to: {destination}")
        legacy_directories = sorted(
            [path for path in legacy_figure_dir.rglob("*") if path.is_dir()],
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in legacy_directories:
            directory.rmdir()
        legacy_figure_dir.rmdir()
        logger.info(f"Removed legacy head-heading figure directory: {legacy_figure_dir}")

    if figure_saver is not None:
        figure_saver(
            fig=event_aligned_figure,
            name="head_heading_event_aligned_95ci",
            width=HEAD_HEADING_FIG_WIDTH,
            height=HEAD_HEADING_FIG_HEIGHT,
            save_final=True,
            open_browser=True,
        )
        figure_saver(
            fig=passage_summary_figure,
            name="head_heading_passage_summary_95ci",
            width=1320,
            height=720,
            save_final=True,
            open_browser=True,
        )
    else:
        _save_plotly_only(
            fig=event_aligned_figure,
            name="head_heading_event_aligned_95ci",
            width=HEAD_HEADING_FIG_WIDTH,
            height=HEAD_HEADING_FIG_HEIGHT,
            open_browser=True,
        )
        _save_plotly_only(
            fig=passage_summary_figure,
            name="head_heading_passage_summary_95ci",
            width=1320,
            height=720,
            open_browser=True,
        )
    _log_results(
        features,
        contrasts,
        distance,
        event_contrasts,
        figure_tests,
        pointwise_tests,
        quality,
    )
    return {
        "features": features,
        "contrasts": contrasts,
        "simple_contrasts": simple,
        "event_contrasts": event_contrasts,
        "distance_tests": distance,
        "condition_summary": conditions,
        "participant_curves": participant_curves,
        "curve_summary": summarized_curves,
        "event_figure_passage_tests": figure_tests,
        "event_aligned_pointwise_ttests": pointwise_tests,
        "quality": quality,
        "output_dir": output,
        "cache_reused": cache_ready,
    }

# Execute analysis
if __name__ == "__main__":
    logger.info("Analysis started.")
    logger.info(f"Analysis source file: {Path(__file__).resolve()}")
    logger.info(f"Analysis source version: {ANALYSIS_SOURCE_VERSION}")
    logger.info(f"Analysis pipeline version: {ANALYSIS_PIPELINE_VERSION}")

    if not TRIGGER_PRESS_THRESHOLDS:
        raise ValueError("TRIGGER_PRESS_THRESHOLDS must contain at least one threshold.")

    if not 0.0 < PRIMARY_TRIGGER_THRESHOLD < 1.0:
        raise ValueError("PRIMARY_TRIGGER_THRESHOLD must be strictly between 0 and 1.")
    if any(not 0.0 < value < 1.0 for value in TRIGGER_PRESS_THRESHOLDS):
        raise ValueError("Every trigger sensitivity threshold must be strictly between 0 and 1.")
    if not any(
        abs(value - PRIMARY_TRIGGER_THRESHOLD) < 1e-12
        for value in TRIGGER_PRESS_THRESHOLDS
    ):
        raise ValueError(
            "The primary trigger threshold must also appear in trigger_threshold."
        )

    trigger_threshold = PRIMARY_TRIGGER_THRESHOLD
    logger.info(
        f"Using trigger press threshold {trigger_threshold:.3f} for the main analysis. "
        f"Sensitivity thresholds: {TRIGGER_PRESS_THRESHOLDS}."
    )

    cache = ProcessedExperimentCache(PROCESSED_DATA_CACHE, output_folder)

    def rebuild_processed_data():
        logger.info("Reanalysing all raw human-experiment data.")
        raw_mapping = pd.read_csv(mapping_source)
        return cache.build(
            helper=HMD,
            mapping=raw_mapping,
            data_folder=data_folder,
            intake_questionnaire=intake_questionnaire_source,
            post_questionnaire=post_experiment_questionnaire_source,
            settings=CACHE_SETTINGS,
            n_participants=50,
        )

    if ALWAYS_ANALYSE:
        if cache.delete_existing():
            logger.info(f"Deleted previous processed-data cache: {cache.cache_path}")
        else:
            logger.info("No previous processed-data cache was present.")
        processed_data = rebuild_processed_data()
        cache.save(processed_data)
        reanalysed_this_run = True
        logger.info(f"Saved complete processed-data cache: {cache.cache_path}")
    else:
        logger.info(f"Loading processed human-experiment data only from: {cache.cache_path}")
        processed_data, reanalysed_this_run = cache.load_or_build(
            builder=rebuild_processed_data,
            expected_settings=CACHE_SETTINGS,
        )
        if reanalysed_this_run:
            logger.warning(
                "Processed-data cache was not present. Completed a full raw-data "
                f"analysis and created: {cache.cache_path}"
            )
        else:
            logger.info("Loaded the existing processed-data cache successfully.")

    # Existing plotting functions consume compatibility CSVs. Restore these
    # from the single pickle so no raw human data are read in cache-only mode.
    cache.restore_analysis_inputs(processed_data)
    mapping = processed_data["mapping"].copy()
    intake_questionnaire = processed_data["intake_questionnaire"].copy()
    post_experiment_questionnaire = processed_data["post_questionnaire"].copy()
    trial_ratings = processed_data["trial_ratings"].copy()
    cache_payload_changed = False
    if "trial_number" not in trial_ratings.columns:
        # Compatibility path for older processed pickles. The cached rating
        # rows retain the order in which Unity wrote each participant's trials.
        trial_ratings["trial_number"] = (
            trial_ratings.groupby("participant").cumcount() + 1
        )
        processed_data["trial_ratings"] = trial_ratings.copy()
        cache_payload_changed = True
        logger.warning(
            "Added trial_number to a legacy processed cache from within-participant "
            "row order; the upgraded cache will retain it."
        )
    HMD.set_processed_data_cache(
        processed_data,
        reuse_statistical_results=not reanalysed_this_run,
    )

    logger.info("Preparing cached inputs and outputs.")
    ensure_slider_tables(
        data_folder,
        output_folder,
        allow_raw_processing=False,
    )

    # Information on participants
    HMD.plot_gender_by_nationality(intake_questionnaire,
                                   gender_col="What is your gender?",
                                   nationality_col="What is your nationality?")

    HMD.plot_column_distribution(intake_questionnaire,
                                 intake_columns_to_plot,
                                 save_file=True,
                                 tag="intake")

    HMD.plot_column_distribution(post_experiment_questionnaire,
                                 post_columns_to_plot,
                                 save_file=True,
                                 tag="post")

    HMD.distribution_plots(intake_questionnaire,
                           intake_columns_distribution_to_plot,
                           save_file=True)

    HMD.distribution_plots(post_experiment_questionnaire,
                           post_columns_distribution_to_plot,
                           save_file=True)

    # Keypress figures for the main threshold. These are saved directly in
    # _output and figures with compact filenames, for example kp_e0_y.html/png.
    run_keypress_condition_plots(
        mapping_df=mapping,
        trigger_threshold=trigger_threshold,
        output_subdir=None,
        save_final=True,
    )

    # Keypress figures grouped by threshold. This makes it easy to inspect the
    # same condition figure at 5%, 10%, and 50% when those thresholds are listed.
    for threshold in TRIGGER_PRESS_THRESHOLDS:
        threshold_label = _trigger_threshold_label(float(threshold))
        run_keypress_condition_plots(
            mapping_df=mapping,
            trigger_threshold=float(threshold),
            output_subdir=os.path.join("kp_thr", f"t{threshold_label.replace('pct', '')}"),
            save_final=True,
        )

    logger.info("Running heat plot and distance analysis.")

    # Heatplot. Trigger values are pressure-sensitive, so values greater than
    # the configured threshold are coded as a pressed/risk state.
    HMD.heat_plot(
        folder_path=output_folder,
        mapping_df=mapping,
        trigger_threshold=trigger_threshold,
    )

    trial_level_df, condition_level_df = HMD.load_and_average_Q2(
        trigger_summary_csv=os.path.join(output_folder, "trigger_summary.csv"),
        responses_root=common.get_configs("data"),
        mapping_df=mapping,
        trigger_threshold=trigger_threshold,
        trigger_matrices_dir=output_folder,
        ratings_df=trial_ratings,
    )

    HMD.analyze_and_plot_distance_effect_plotly(
        mapping_df=mapping,
        out_dir=output_folder,
        trial_df=trial_level_df,
        condition_df=condition_level_df,
    )

    logger.info("Running trigger-threshold sensitivity analysis.")
    HMD.run_trigger_threshold_sensitivity(
        trigger_thresholds=TRIGGER_PRESS_THRESHOLDS,
        primary_threshold=PRIMARY_TRIGGER_THRESHOLD,
        trigger_matrices_dir=output_folder,
        responses_root=common.get_configs("data"),
        mapping_df=mapping,
        output_dir=os.path.join(output_folder, "threshold_sensitivity"),
        n_participants=50,
        ratings_df=trial_ratings,
    )

    logger.info("Running violin plots for behaviour, distance, and intention ratings.")

    # Violin plots
    HMD.plot_2x4_violins(
        responses_csv=os.path.join(output_folder, "slider_input_behaviour.csv"),
        mapping=mapping,
        name="behaviour_of_the_other_pedestrian"
    )

    HMD.plot_2x4_violins(
        responses_csv=os.path.join(output_folder, "slider_input_distance.csv"),
        mapping=mapping,
        name="distance_between_pedestrian"
    )

    HMD.plot_2x4_violins(
        responses_csv=os.path.join(output_folder, "slider_input_intention.csv"),
        mapping=mapping,
        name="intention_of_the_vehicle"
    )

    logger.info("Running head rotation analysis plots.")

    # Head rotation
    HMD.plot_yaw(mapping,
                 parameter=None,
                 xaxis_range=[0, 18],
                 compare_trial="video_1",
                 xaxis_title="Time, [s]",
                 name="all_yaw_values_with_yielding",
                 margin=dict(l=120, r=2, t=12, b=12),
                 recompute=reanalysed_this_run)

    HMD.plot_yaw(mapping,
                 parameter=None,
                 xaxis_range=[0, 11],
                 compare_trial="video_21",
                 xaxis_title="Time, [s]",
                 name="all_yaw_values_without_yielding",
                 margin=dict(l=120, r=2, t=12, b=12),
                 recompute=reanalysed_this_run)

    # Keypress data for yielding and eHMI criteria

    # eHMI is off and car is yielding
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 xaxis_range=[0, 18],
                 compare_trial="video_1",
                 xaxis_title="Time, [s]",
                 name="yaw_eHMI_off_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # eHMI is on and car is yielding
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 xaxis_range=[0, 18],
                 compare_trial="video_11",
                 xaxis_title="Time, [s]",
                 name="yaw_eHMI_on_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # eHMI is off and car is not yielding
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 xaxis_range=[0, 11],
                 compare_trial="video_31",
                 xaxis_title="Time, [s]",
                 name="yaw_eHMI_off_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # eHMI is off and car is not yielding
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 xaxis_range=[0, 11],
                 compare_trial="video_21",
                 xaxis_title="Time, [s]",
                 name="yaw_eHMI_on_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # Keypress data for yielding, eHMI and position criteria

    # First person view
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 additional_parameter="camera",
                 additional_parameter_value=0,
                 xaxis_range=[0, 11],
                 compare_trial="video_21",
                 xaxis_title="Time, [s]",
                 name="yaw_first_eHMI_on_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 additional_parameter="camera",
                 additional_parameter_value=0,
                 xaxis_range=[0, 18],
                 compare_trial="video_11",
                 xaxis_title="Time, [s]",
                 name="yaw_first_eHMI_on_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 additional_parameter="camera",
                 additional_parameter_value=0,
                 xaxis_range=[0, 11],
                 compare_trial="video_31",
                 xaxis_title="Time, [s]",
                 name="yaw_first_eHMI_off_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 additional_parameter="camera",
                 additional_parameter_value=0,
                 xaxis_range=[0, 18],
                 compare_trial="video_1",
                 xaxis_title="Time, [s]",
                 name="yaw_first_eHMI_off_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    # Second-person view
    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 additional_parameter="camera",
                 additional_parameter_value=1,
                 xaxis_range=[0, 11],
                 compare_trial="video_26",
                 xaxis_title="Time, [s]",
                 name="yaw_second_eHMI_on_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=1,
                 additional_parameter="camera",
                 additional_parameter_value=1,
                 xaxis_range=[0, 18],
                 compare_trial="video_16",
                 xaxis_title="Time, [s]",
                 name="yaw_second_eHMI_on_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 additional_parameter="camera",
                 additional_parameter_value=1,
                 xaxis_range=[0, 11],
                 compare_trial="video_36",
                 xaxis_title="Time, [s]",
                 name="yaw_second_eHMI_off_non-yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw(mapping,
                 parameter="eHMIOn",
                 parameter_value=0,
                 additional_parameter="camera",
                 additional_parameter_value=1,
                 xaxis_range=[0, 18],
                 compare_trial="video_6",
                 xaxis_title="Time, [s]",
                 name="yaw_second_eHMI_off_yielding",
                 margin=dict(l=120, r=2, t=12, b=12))

    HMD.plot_yaw_frequencies_by_condition(
        mapping=mapping,
        yaw_files_dir=output_folder
        )

    logger.info("Running participant-level event-aligned head-heading analysis.")
    head_heading_results = run_head_heading_analysis(
        matrices=output_folder,
        mapping=mapping,
        output=os.path.join(output_folder, "statistics", "head_heading"),
        # True both when always_analyse=true and when a missing pickle caused
        # load_or_build() to perform a complete raw-data analysis.
        force=reanalysed_this_run,
        figure_saver=HMD.save_plotly,
        # This stable pickle identifier prevents restored CSV modification
        # times from needlessly invalidating the head-heading cache.
        source_cache_key=str(processed_data["created_utc"]),
    )
    logger.info(
        "Head-heading analysis completed; cached participant-level results "
        f"reused={head_heading_results['cache_reused']}."
    )

    run_advanced_statistics(
        trial_level_df,
        trigger_threshold=trigger_threshold,
    )

    if (
        reanalysed_this_run
        or cache_payload_changed
        or not head_heading_results["cache_reused"]
        or HMD.statistical_cache_changed
        or not processed_data["statistical_tables"]
    ):
        statistical_table_count = cache.capture_statistical_tables(processed_data)
        cache.save(processed_data)
        logger.info(
            f"Stored {statistical_table_count} generated statistical tables "
            "in the processed-data cache."
        )
    logger.info("Analysis finished.")
