"""Backward compatible facade for the advanced statistical pipeline."""

from __future__ import annotations

from .advanced_statistics import (
    BinomialMixin,
    CommonWindowFigureMixin,
    CoreStatsMixin,
    EquivalenceMixin,
    FeatureModelMixin,
    RepeatedMeasuresMixin,
    RevisionAnalysesMixin,
    RunnerMixin,
    TOSTResult,
    TriggerFeatureMixin,
    WithinBetweenMixin,
)

ADVANCED_STATS_SPECIFICATION = "reviewer_response_v6_second_revision"


class AdvancedStatsRunner(
    CoreStatsMixin,
    TriggerFeatureMixin,
    EquivalenceMixin,
    RepeatedMeasuresMixin,
    BinomialMixin,
    CommonWindowFigureMixin,
    WithinBetweenMixin,
    FeatureModelMixin,
    RevisionAnalysesMixin,
    RunnerMixin,
):
    """Compatibility runner composed from focused statistical mixins."""

    pass


__all__ = ["ADVANCED_STATS_SPECIFICATION", "TOSTResult", "AdvancedStatsRunner"]
