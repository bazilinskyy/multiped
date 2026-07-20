from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

import common
from custom_logger import CustomLogger
from .settings import *

logger = CustomLogger(__name__)
from .features import trial_features
from .statistics import (
    build_contrast_table, build_distance_table, build_simple_contrast_table,
    build_yielding_event_table, condition_summary, curve_summary,
    passage_order_tests, pointwise_order_ttests,
)
from .plots import (
    create_event_aligned_plotly, create_passage_summary_plotly, _save_plotly_only,
)
from .cache import _input_signature, _load_manifest
from .reporting import _log_results

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
