# by Shadab Alam <shaadalam.5u@gmail.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from __future__ import annotations

from human_analysis.helper import HMD_helper
from human_analysis.analysis_config import AnalysisConfig
from human_analysis.head_heading import run_head_heading_analysis
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

from human_analysis.processed_cache import ProcessedExperimentCache
from human_analysis import stats as stats_module


AdvancedStatsRunner = stats_module.AdvancedStatsRunner


logs(show_level="info", show_color=True)
logger = CustomLogger(__name__)  # use custom logger
HMD: HMD_helper | None = None

CONFIG = AnalysisConfig.from_project_config()
data_folder = str(CONFIG.data_directory)
mapping_source = str(CONFIG.mapping_path)
mapping = None
output_folder = str(CONFIG.output_directory)
intake_questionnaire_source = str(CONFIG.intake_questionnaire_path)
post_experiment_questionnaire_source = str(CONFIG.post_questionnaire_path)
intake_questionnaire = None
post_experiment_questionnaire = None
ALWAYS_ANALYSE = CONFIG.always_analyse
PROCESSED_DATA_CACHE = str(CONFIG.processed_cache_path)
CACHE_SETTINGS = {
    "processing_version": 2,
    "kp_resolution_ms": CONFIG.keypress_resolution_ms,
    "yaw_resolution_ms": CONFIG.yaw_resolution_ms,
    "distance_convention": "distPed_code_times_2_metres_v1",
    "heading_convention": "unity_y_vertical_xz_heading_v1",
    "trigger_binning_convention": "floor_half_open_any_above_threshold_v2",
}

RUN_ADVANCED_STATISTICS = True
ANALYSIS_PIPELINE_VERSION = "reviewer_response_v5_participant_bootstrap"
ANALYSIS_SOURCE_VERSION = "2026-07-26-participant-bootstrap-v1"
EQUIVALENCE_MARGIN_POINTS = 5.0
PRIMARY_TRIGGER_THRESHOLD = CONFIG.primary_trigger_threshold
TRIGGER_PRESS_THRESHOLDS = list(CONFIG.trigger_thresholds)
PARTICIPANT_BOOTSTRAP_RESAMPLES = CONFIG.participant_bootstrap_resamples
PARTICIPANT_BOOTSTRAP_SEED = CONFIG.participant_bootstrap_seed
PARTICIPANT_BOOTSTRAP_MINIMUM_SUCCESS_RATE = (
    CONFIG.participant_bootstrap_minimum_success_rate
)



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
        participant_bootstrap_resamples=PARTICIPANT_BOOTSTRAP_RESAMPLES,
        participant_bootstrap_seed=PARTICIPANT_BOOTSTRAP_SEED,
        participant_bootstrap_minimum_success_rate=(
            PARTICIPANT_BOOTSTRAP_MINIMUM_SUCCESS_RATE
        ),
    )
    required_outputs = [
        "common_window_primary_marginal_probabilities.csv",
        "common_window_primary_revised_contrasts.csv",
        "common_window_primary_omnibus_tests.csv",
        "common_window_primary_binomial_diagnostics.csv",
        "common_window_primary_participant_bootstrap_marginal_probabilities.csv",
        "common_window_primary_participant_bootstrap_revised_contrasts.csv",
        "common_window_primary_participant_bootstrap_diagnostics.csv",
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



def _yaw_plot_specs() -> list[dict[str, object]]:
    """Return all condition specific head heading plot configurations."""
    base = {
        "xaxis_title": "Time, [s]",
        "margin": dict(l=120, r=2, t=12, b=12),
    }
    raw_specs = [
        dict(parameter=None, xaxis_range=[0, 18], compare_trial="video_1",
             name="all_yaw_values_with_yielding", recompute_from_cache=True),
        dict(parameter=None, xaxis_range=[0, 11], compare_trial="video_21",
             name="all_yaw_values_without_yielding", recompute_from_cache=True),
        dict(parameter="eHMIOn", parameter_value=0, xaxis_range=[0, 18],
             compare_trial="video_1", name="yaw_eHMI_off_yielding"),
        dict(parameter="eHMIOn", parameter_value=1, xaxis_range=[0, 18],
             compare_trial="video_11", name="yaw_eHMI_on_yielding"),
        dict(parameter="eHMIOn", parameter_value=0, xaxis_range=[0, 11],
             compare_trial="video_31", name="yaw_eHMI_off_non-yielding"),
        dict(parameter="eHMIOn", parameter_value=1, xaxis_range=[0, 11],
             compare_trial="video_21", name="yaw_eHMI_on_non-yielding"),
        dict(parameter="eHMIOn", parameter_value=1, additional_parameter="camera",
             additional_parameter_value=0, xaxis_range=[0, 11], compare_trial="video_21",
             name="yaw_first_eHMI_on_non-yielding"),
        dict(parameter="eHMIOn", parameter_value=1, additional_parameter="camera",
             additional_parameter_value=0, xaxis_range=[0, 18], compare_trial="video_11",
             name="yaw_first_eHMI_on_yielding"),
        dict(parameter="eHMIOn", parameter_value=0, additional_parameter="camera",
             additional_parameter_value=0, xaxis_range=[0, 11], compare_trial="video_31",
             name="yaw_first_eHMI_off_non-yielding"),
        dict(parameter="eHMIOn", parameter_value=0, additional_parameter="camera",
             additional_parameter_value=0, xaxis_range=[0, 18], compare_trial="video_1",
             name="yaw_first_eHMI_off_yielding"),
        dict(parameter="eHMIOn", parameter_value=1, additional_parameter="camera",
             additional_parameter_value=1, xaxis_range=[0, 11], compare_trial="video_26",
             name="yaw_second_eHMI_on_non-yielding"),
        dict(parameter="eHMIOn", parameter_value=1, additional_parameter="camera",
             additional_parameter_value=1, xaxis_range=[0, 18], compare_trial="video_16",
             name="yaw_second_eHMI_on_yielding"),
        dict(parameter="eHMIOn", parameter_value=0, additional_parameter="camera",
             additional_parameter_value=1, xaxis_range=[0, 11], compare_trial="video_36",
             name="yaw_second_eHMI_off_non-yielding"),
        dict(parameter="eHMIOn", parameter_value=0, additional_parameter="camera",
             additional_parameter_value=1, xaxis_range=[0, 18], compare_trial="video_6",
             name="yaw_second_eHMI_off_yielding"),
    ]
    return [{**base, **spec} for spec in raw_specs]


def _run_yaw_plots(mapping_df: pd.DataFrame, recompute: bool) -> None:
    """Generate all configured head heading plots through one readable loop."""
    for spec in _yaw_plot_specs():
        plot_spec = spec.copy()
        use_recompute = bool(plot_spec.pop("recompute_from_cache", False))
        HMD.plot_yaw(
            mapping_df,
            recompute=recompute if use_recompute else False,
            **plot_spec,
        )

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


def main() -> None:
    """Run the complete human experiment analysis pipeline."""
    global HMD, mapping, intake_questionnaire, post_experiment_questionnaire
    HMD = HMD_helper()
    os.makedirs(output_folder, exist_ok=True)
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
    violin_specs = {
        "behaviour": "behaviour_of_the_other_pedestrian",
        "distance": "distance_between_pedestrian",
        "intention": "intention_of_the_vehicle",
    }
    for slider_name, figure_name in violin_specs.items():
        HMD.plot_2x4_violins(
            responses_csv=os.path.join(output_folder, f"slider_input_{slider_name}.csv"),
            mapping=mapping,
            name=figure_name,
        )

    logger.info("Running head rotation analysis plots.")
    _run_yaw_plots(mapping, recompute=reanalysed_this_run)

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


if __name__ == "__main__":
    main()
