"""Participant level horizontal head heading analysis.

The public runner is imported lazily so importing a focused submodule such as
``human_analysis.head_heading.plots`` does not first import the complete
analysis pipeline and all of its dependencies.
"""

from __future__ import annotations

from typing import Any

__all__ = ["run_head_heading_analysis"]


def run_head_heading_analysis(*args: Any, **kwargs: Any) -> dict[str, object]:
    """Run the head heading analysis through a lazy compatibility wrapper."""
    from .runner import run_head_heading_analysis as _run_head_heading_analysis

    return _run_head_heading_analysis(*args, **kwargs)
