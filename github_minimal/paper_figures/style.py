"""Shared publication style and export helpers for manuscript Figures 2-4."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


# NPG "nrc" palette from ggsci; inspired by Nature Reviews Cancer.
NPG_COLORS = (
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
)
BLUE = NPG_COLORS[3]
BAR_BLUE = BLUE
INK = "#252D36"
GRID = "#E2E5E8"
MODELS = ["Chronos-ZeroShot", "Chronos", "iTransformer", "Autoformer",
          "MLP", "RNN", "GRU", "LSTM", "TCN"]
LABELS = {"Chronos-ZeroShot": "Chronos-2 base", "Chronos": "Chronos-2 + LoRA"}
STYLES = {
    "Chronos-ZeroShot": (NPG_COLORS[3], "--", "v"),
    "Chronos": (NPG_COLORS[0], "-", "o"),
    "iTransformer": (NPG_COLORS[1], "-", "D"),
    "Autoformer": (NPG_COLORS[2], "-", "s"),
    "MLP": (NPG_COLORS[4], "-.", "x"),
    "RNN": (NPG_COLORS[5], "--", "^"),
    "GRU": (NPG_COLORS[6], ":", "P"),
    "LSTM": (NPG_COLORS[8], "-.", "+"),
    "TCN": (NPG_COLORS[9], (0, (5, 2, 1, 2)), "*"),
}
CLUSTER_COLORS = [NPG_COLORS[i] for i in (1, 0, 2, 3, 5, 8)]
CLUSTER_MARKERS = ["o", "^", "s", "D", "v", "P"]
MSE_MODES = ("relative", "thousands", "raw")
FONT_SIZE = 16
TITLE_SIZE = 18
TICK_SIZE = 14
LEGEND_SIZE = 15


def apply_style():
    """Use readable sans-serif text, with explicit portable fallbacks."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    font = next(name for name in ("Arial", "Helvetica", "DejaVu Sans") if name in installed)
    plt.rcParams.update({
        "font.family": font, "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE, "axes.titlesize": TITLE_SIZE,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "mathtext.fontset": "custom",
        "mathtext.rm": font, "mathtext.it": f"{font}:italic",
        "mathtext.bf": f"{font}:bold",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "axes.unicode_minus": True,
    })
    return font


def style_axes(ax, tick_size=TICK_SIZE):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=tick_size, length=3, width=0.7, pad=3)
    ax.grid(False)
    ax.set_axisbelow(True)


def display_mse(values, mode, reference):
    """Change display units only; never standardize individual transformer errors."""
    if mode not in MSE_MODES:
        raise ValueError(f"Unknown MSE display: {mode}")
    divisor = reference if mode == "relative" else 1000 if mode == "thousands" else 1
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("MSE display divisor must be finite and positive")
    return values / divisor


def mse_label(mode):
    return {"relative": "Relative MSE", "thousands": r"MSE / $10^3$", "raw": "MSE"}[mode]


def mse_description(mode, reference):
    if mode == "relative":
        return (f"MSE is divided by the fixed comparison reference {reference:.10g}, "
                "as recorded by the existing evaluation code. This changes display units only.")
    if mode == "thousands":
        return "MSE is divided by 1000 solely for display; this is not per-series normalization."
    return "MSE is shown on the original load scale, without normalization."


def verify_text_extents(fig, name="figure"):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    hidden_tick_labels = set()
    # Log locators cache labels beyond the visible axis range; those are not drawn.
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo <= tick.get_loc() <= hi:
                    hidden_tick_labels.update((id(tick.label1), id(tick.label2)))
    outside = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_visible() or not text.get_text().strip() or id(text) in hidden_tick_labels:
            continue
        bounds = text.get_window_extent(renderer)
        if (bounds.x0 < -2 or bounds.y0 < -2
                or bounds.x1 > fig.bbox.x1 + 2 or bounds.y1 > fig.bbox.y1 + 2):
            outside.append(text.get_text())
    if outside:
        raise ValueError(f"Text outside {name}: {outside}")


def export_figure(fig, target, dpi=600, formats=("png", "pdf", "svg"), preview_dir=None):
    """Keep physical size and margins identical in raster and vector exports."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if dpi <= 0 or not formats or set(formats) - {"png", "pdf", "svg"}:
        raise ValueError("Use positive DPI and png/pdf/svg export formats")
    verify_text_extents(fig, target.name)
    for extension in dict.fromkeys(formats):
        fig.savefig(target.parent / f"{target.name}.{extension}", dpi=dpi, facecolor="white")
    if preview_dir is not None:
        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{target.name}.png", dpi=140, facecolor="white")
