"""Backward compatible facade for the advanced statistical pipeline."""

from __future__ import annotations

from .advanced_statistics import (
    BinomialMixin,
    CommonWindowFigureMixin,
    CoreStatsMixin,
    EquivalenceMixin,
    FeatureModelMixin,
    RepeatedMeasuresMixin,
    RunnerMixin,
    TOSTResult,
    TriggerFeatureMixin,
    WithinBetweenMixin,
)

ADVANCED_STATS_SPECIFICATION = "reviewer_response_v4_bounded_common_window"


class AdvancedStatsRunner(
    CoreStatsMixin,
    TriggerFeatureMixin,
    EquivalenceMixin,
    RepeatedMeasuresMixin,
    BinomialMixin,
    CommonWindowFigureMixin,
    WithinBetweenMixin,
    FeatureModelMixin,
    RunnerMixin,
):
    """Compatibility runner composed from focused statistical mixins."""

    pass


__all__ = ["ADVANCED_STATS_SPECIFICATION", "TOSTResult", "AdvancedStatsRunner"]
