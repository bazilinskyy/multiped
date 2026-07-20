"""Backward compatible facade for the experiment helper components."""

from __future__ import annotations

from .helper_parts import (
    CoreMixin,
    DataIOMixin,
    DistanceAnalysisMixin,
    FigureExportMixin,
    HeadingPlotMixin,
    KeypressPlotMixin,
    MixedModelMixin,
    QuestionnaireMixin,
    RatingPlotMixin,
    SensitivityMixin,
    StatisticalAnnotationMixin,
    TriggerDataMixin,
)


class HMD_helper(
    CoreMixin,
    QuestionnaireMixin,
    DataIOMixin,
    FigureExportMixin,
    KeypressPlotMixin,
    StatisticalAnnotationMixin,
    TriggerDataMixin,
    MixedModelMixin,
    SensitivityMixin,
    DistanceAnalysisMixin,
    RatingPlotMixin,
    HeadingPlotMixin,
):
    """Compatibility class composed from focused implementation mixins."""

    pass


__all__ = ["HMD_helper"]
