"""Typed project configuration loaded once by the analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import common


@dataclass(frozen=True)
class AnalysisConfig:
    data_directory: Path
    mapping_path: Path
    output_directory: Path
    figures_directory: Path
    intake_questionnaire_path: Path
    post_questionnaire_path: Path
    processed_cache_path: Path
    always_analyse: bool
    primary_trigger_threshold: float
    trigger_thresholds: tuple[float, ...]
    participant_bootstrap_resamples: int
    participant_bootstrap_seed: int
    participant_bootstrap_minimum_success_rate: float
    keypress_resolution_ms: int
    yaw_resolution_ms: int
    p_value: float
    font_size: float

    @classmethod
    def from_project_config(cls) -> "AnalysisConfig":
        """Load and validate all settings needed by the top level pipeline."""
        config = cls(
            data_directory=Path(common.get_configs("data")),
            mapping_path=Path(common.get_configs("mapping")),
            output_directory=Path(common.get_configs("output")),
            figures_directory=Path(common.get_configs("figures")),
            intake_questionnaire_path=Path(common.get_configs("intake_questionnaire")),
            post_questionnaire_path=Path(common.get_configs("post_experiment_questionnaire")),
            processed_cache_path=Path(common.get_configs("processed_data_cache")),
            always_analyse=bool(common.get_configs("always_analyse")),
            primary_trigger_threshold=float(common.get_configs("primary_trigger_threshold")),
            trigger_thresholds=tuple(float(v) for v in common.get_configs("trigger_threshold")),
            participant_bootstrap_resamples=int(
                common.get_configs("participant_bootstrap_resamples")
            ),
            participant_bootstrap_seed=int(
                common.get_configs("participant_bootstrap_seed")
            ),
            participant_bootstrap_minimum_success_rate=float(
                common.get_configs("participant_bootstrap_minimum_success_rate")
            ),
            keypress_resolution_ms=int(common.get_configs("kp_resolution")),
            yaw_resolution_ms=int(common.get_configs("yaw_resolution")),
            p_value=float(common.get_configs("p_value")),
            font_size=float(common.get_configs("font_size")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject inconsistent threshold and resolution settings early."""
        if not self.trigger_thresholds:
            raise ValueError("trigger_threshold must contain at least one threshold")
        if not 0.0 < self.primary_trigger_threshold < 1.0:
            raise ValueError("primary_trigger_threshold must be between 0 and 1")
        if any(not 0.0 < value < 1.0 for value in self.trigger_thresholds):
            raise ValueError("Every trigger threshold must be between 0 and 1")
        if not any(
            abs(value - self.primary_trigger_threshold) < 1e-12
            for value in self.trigger_thresholds
        ):
            raise ValueError("The primary threshold must appear in trigger_threshold")
        if self.participant_bootstrap_resamples < 100:
            raise ValueError(
                "participant_bootstrap_resamples must be at least 100"
            )
        if self.participant_bootstrap_seed < 0:
            raise ValueError("participant_bootstrap_seed must be nonnegative")
        if not 0.0 < self.participant_bootstrap_minimum_success_rate <= 1.0:
            raise ValueError(
                "participant_bootstrap_minimum_success_rate must be in (0, 1]"
            )
        if self.keypress_resolution_ms <= 0 or self.yaw_resolution_ms <= 0:
            raise ValueError("Sampling resolutions must be positive")
