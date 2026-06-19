"""
thesis_style.py - shared plot theme for every thesis figure.

A clean, smooth, purple/blue palette with a warm complementary accent.
Usage: at the top of any figure-generating script, after importing matplotlib:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root
    from thesis_style import apply_theme, GOOD, CONTRAST, PALETTE
    apply_theme()

Then let plots use the default colour cycle, or use the named colours
(GOOD, SECOND, CONTRAST, ...) where a figure hardcodes semantic colours.
"""

import matplotlib as mpl
from cycler import cycler

# ---- Purple / blue palette (+ one warm complement for contrast series) ----
PALETTE = [
    "#5E548E",  # muted purple   (primary / "good")
    "#1F6FB2",  # blue           (secondary)
    "#9F86C0",  # light violet
    "#48BFE3",  # sky blue
    "#E07A5F",  # terracotta     (warm complement / contrast)
    "#C77DFF",  # bright violet
    "#3D348B",  # deep indigo
    "#118AB2",  # teal
]

# Named semantic colours for figures that encode meaning in colour.
PURPLE   = "#5E548E"
BLUE     = "#1F6FB2"
VIOLET   = "#9F86C0"
SKY      = "#48BFE3"
TERRA    = "#E07A5F"
INDIGO   = "#3D348B"
TEAL     = "#118AB2"

GOOD     = PURPLE   # primary / "the good" series
SECOND   = BLUE     # second series
THIRD    = VIOLET   # third series
CONTRAST = TERRA    # the "bad" / contrast series (warm, stands out on purple/blue)

# Sequential map for heatmaps / confusion-style images (light -> deep purple).
CMAP = "BuPu"


def apply_theme():
    """Apply the thesis plot theme globally (call once per script)."""
    mpl.rcParams.update({
        # typography: clean modern sans
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12.5,
        "axes.labelweight": "medium",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        # colour
        "axes.prop_cycle": cycler(color=PALETTE),
        "image.cmap": CMAP,
        # despined, light axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#3A3A3A",
        "axes.linewidth": 1.1,
        "axes.titlepad": 10,
        # subtle horizontal grid only
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#D9D9E0",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.7,
        # smooth lines / markers
        "lines.linewidth": 2.4,
        "lines.markersize": 7,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.antialiased": True,
        # bar / patch edges
        "patch.edgecolor": "white",
        "patch.linewidth": 0.8,
        "patch.antialiased": True,
        # canvas
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def despine(ax, keep=("left", "bottom")):
    """Hide all spines except those named in *keep*."""
    for name, spine in ax.spines.items():
        spine.set_visible(name in keep)


def palette(n):
    """Return *n* colours from the palette (cycling if n > len)."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]
