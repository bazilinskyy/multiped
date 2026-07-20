"""Reusable Plotly export service for newly written analysis code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plotly.io as pio


@dataclass(frozen=True)
class PlotlyExporter:
    output_directory: Path
    final_directory: Path

    def save(
        self,
        figure: Any,
        name: str,
        *,
        width: int = 1320,
        height: int = 680,
        save_html: bool = True,
        save_png: bool = True,
        save_eps: bool = True,
        save_final: bool = False,
    ) -> list[Path]:
        """Save a figure and return every successfully written path."""
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Figure name must be a safe relative path: {name}")
        roots = [self.output_directory]
        if save_final:
            roots.append(self.final_directory)
        written: list[Path] = []
        for root in roots:
            base = root / relative
            base.parent.mkdir(parents=True, exist_ok=True)
            if save_html:
                html_path = base.with_suffix(".html")
                figure.write_html(str(html_path))
                written.append(html_path)
            for enabled, suffix in ((save_png, ".png"), (save_eps, ".eps")):
                if not enabled:
                    continue
                image_path = base.with_suffix(suffix)
                pio.write_image(figure, str(image_path), width=width, height=height)
                written.append(image_path)
        return written
