#!/usr/bin/env python3
"""
review_labels_gui.py 

Usage:
    python label_reviewer/review_labels_gui.py
"""
import sys
from pathlib import Path

# Add the repo root to path so the label_reviewer package is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from label_reviewer.gui import launch_gui

if __name__ == "__main__":
    launch_gui()
