#!/usr/bin/env python3
"""
review_labels_gui.py 

Usage:
    python scripts/review_labels_gui.py
"""
import sys
from pathlib import Path

# Add scripts/ to path so label_reviewer package is importable
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from label_reviewer.gui import launch_gui

if __name__ == "__main__":
    launch_gui()
