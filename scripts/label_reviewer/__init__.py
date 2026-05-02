"""
label_reviewer — Multi-mode label review toolkit for drone detection.

Entry points:
    python scripts/review_labels_gui.py       # GUI launcher
    python scripts/review_labels.py ...       # CLI (backward compat)
"""
from .core import LabelReviewer
from .predictor import run_prediction

__all__ = ["LabelReviewer", "run_prediction"]


def launch_gui():
    """Lazy import to avoid pulling in tkinter at package level."""
    from .gui import launch_gui as _launch
    _launch()
