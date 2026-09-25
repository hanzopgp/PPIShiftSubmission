"""Shared physical-size styling for the ICLR 2027 figure set."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PAPER_WIDTH = 5.5
INK = "#203040"
MUTED = "#657587"
GRID = "#dce3e9"
BLUE = "#32658a"
ORANGE = "#c47739"
PANEL_SIZE = 10.5
AXIS_SIZE = 9.2
TICK_SIZE = 9.0
CELL_SIZE = 9.0

SAMPLER_LABELS = {
    "uniform_partner": "Uniform",
    "degree_partner_matched": "Degree",
    "sequence_nuisance_matched": "Sequence",
    "esm_partner_matched": "ESM2",
    "bernett2026_ilp_exact": "Bernett",
    "tppni2025_scm_cl3_exact_balanced": "TPPNI",
}
MODEL_LABELS = {
    "symmetric_frozen_esm2_pair_mlp": "Frozen-ESM2",
    "tuna_corrected": "TUnA",
    "dscript_esm2": "D-SCRIPT",
    "rapppid": "RAPPPID",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": TICK_SIZE,
            "axes.labelsize": AXIS_SIZE,
            "axes.titlesize": PANEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_header(axis: Any, label: str) -> None:
    axis.set_title(label, loc="left", fontsize=PANEL_SIZE, fontweight="bold", pad=6)


def clean_axis(axis: Any, *, grid: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=0, pad=3)
    if grid:
        axis.grid(axis="y", color=GRID, linewidth=0.65)
        axis.set_axisbelow(True)


def text_on_color(rgba: tuple[float, float, float, float]) -> str:
    red, green, blue, _ = rgba
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.51 else INK


def heatmap(
    axis: Any,
    values: np.ndarray,
    *,
    cmap: Any,
    norm: Any,
    annotations: str | None = None,
    annotation_size: float = CELL_SIZE,
) -> Any:
    image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for edge in np.arange(-0.5, values.shape[1], 1):
        axis.axvline(edge, color="white", linewidth=0.7)
    for edge in np.arange(-0.5, values.shape[0], 1):
        axis.axhline(edge, color="white", linewidth=0.7)
    if annotations:
        for (row, column), value in np.ndenumerate(values):
            if not np.ma.is_masked(value) and np.isfinite(float(value)):
                axis.text(
                    column,
                    row,
                    annotations.format(float(value)),
                    ha="center",
                    va="center",
                    fontsize=annotation_size,
                    color=text_on_color(cmap(norm(float(value)))),
                )
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0, pad=3)
    return image
