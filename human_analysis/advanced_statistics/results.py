"""Typed result containers for advanced statistics."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class TOSTResult:

    """Container for paired TOST and paired t test results.

    Attributes:
        label: Human readable label for the comparison that was run.
        n: Number of paired observations used in the analysis.
        mean_diff: Mean of the paired differences.
        sd_diff: Sample standard deviation of the paired differences.
        se_diff: Standard error of the paired differences.
        ci90_low: Lower bound of the 90 percent confidence interval.
        ci90_high: Upper bound of the 90 percent confidence interval.
        ci95_low: Lower bound of the 95 percent confidence interval.
        ci95_high: Upper bound of the 95 percent confidence interval.
        t_lower: Test statistic for the lower bound one sided TOST test.
        p_lower: P value for the lower bound one sided TOST test.
        t_upper: Test statistic for the upper bound one sided TOST test.
        p_upper: P value for the upper bound one sided TOST test.
        p_tost: Final TOST p value, defined as the larger one sided p value.
        equivalent: Whether both one sided tests passed at the requested alpha.
        t_paired: Test statistic from the conventional paired t test against zero.
        p_paired: P value from the conventional paired t test against zero.
        margin_low: Lower equivalence margin supplied by the caller.
        margin_high: Upper equivalence margin supplied by the caller.
    """
    label: str
    n: int
    mean_diff: float
    sd_diff: float
    se_diff: float
    ci90_low: float
    ci90_high: float
    ci95_low: float
    ci95_high: float
    t_lower: float
    p_lower: float
    t_upper: float
    p_upper: float
    p_tost: float
    equivalent: bool
    t_paired: float
    p_paired: float
    margin_low: float
    margin_high: float
