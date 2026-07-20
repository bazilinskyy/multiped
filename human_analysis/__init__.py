"""Implementation package for the human experiment analysis pipeline.

The package intentionally avoids eager imports because the complete analysis
requires project specific modules such as ``common`` and ``custom_logger``.
Import the required component from its module instead, for example
``human_analysis.helper`` or ``human_analysis.stats``.
"""

__all__: list[str] = []
