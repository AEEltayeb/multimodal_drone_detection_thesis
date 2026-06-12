#!/usr/bin/env python3
"""
reorg_2026_06_kb.py — re-point knowledge CSV path columns after the 2026-06-11
reorganization (companion to reorg_2026_06.py, same move-map).

Single-writer compliant: every change goes through `kb.py set` (one subprocess
per row+column); this script never writes the CSVs itself.

Usage: py knowledge/_tools/reorg_2026_06_kb.py [--dry-run]
"""
import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
KB = REPO / "knowledge" / "_tools" / "kb.py"

MAP = [
    ("RGB model/dataset preparation/", "training/dataset_preparation/"),
    ("eval/results/_v5_selcom_pure_1x8/classifiers/", "models/verifiers/rgb_v5/"),
    ("eval/results/_routing_pipeline_cmp/robust8", "models/routers/robust8"),
    ("eval/results/_routing_pipeline_cmp/new_router", "models/routers/new_router"),
    ("mri/results/ir_aligned/classifiers/", "models/verifiers/ir_aligned/"),
    ("classifier/fusion_models/", "models/routers/"),
    ("classifier/runs/patches/", "models/patches/"),
    ("runs/corrective_finetune/", "models/ir/corrective_finetune/"),
    ("models/IR_", "models/ir/IR_"),
    ("RGB model/Yolo26n_", "models/rgb/Yolo26n_"),
    ("RGB model/", "training/"),
    ("scripts/review_labels_gui.py", "label_reviewer/review_labels_gui.py"),
    ("scripts/label_reviewer", "label_reviewer"),
    ("ir_gui/", "gui/"),
]

# columns eligible for rewriting, per table (only those that hold live paths/commands)
COLS = {
    "scripts": ["path", "inputs", "outputs", "reproduce_cmd"],
    "models":  ["weights_path", "trained_from_script", "reproduce_cmd", "config"],
    "evals":   ["cache_path", "source_script"],
}
# archive/ rows are historical: their recorded paths must stay as-is
SKIP_PREFIX = "archive/"

def remap(value: str) -> str:
    out = value
    for old, new in MAP:
        out = out.replace(old, new)
        out = out.replace(old.replace("/", "\\"), new.replace("/", "\\"))
    return out

def main():
    dry = "--dry-run" in sys.argv
    changes = []
    for table, cols in COLS.items():
        f = REPO / "knowledge" / f"{table}.csv"
        if not f.exists():
            continue
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            rid = row.get("id", "")
            for col in cols:
                val = row.get(col) or ""
                if not val or val.startswith(SKIP_PREFIX):
                    continue
                new = remap(val)
                if new != val:
                    changes.append((table, rid, col, val, new))
    print(f"{len(changes)} column updates to apply{' (dry-run)' if dry else ''}:")
    for table, rid, col, old, new in changes:
        print(f"  {table}.{rid}.{col}: {old[:70]} -> {new[:70]}")
        if not dry:
            r = subprocess.run([sys.executable, str(KB), "set", table, rid, f"{col}={new}"],
                               capture_output=True, text=True, cwd=str(REPO))
            if r.returncode != 0:
                print(f"    !! kb.py set FAILED: {r.stderr.strip()[:200]}")
    if not dry:
        v = subprocess.run([sys.executable, str(KB), "validate"],
                           capture_output=True, text=True, cwd=str(REPO))
        print("validate:", (v.stdout or v.stderr).strip()[:300])

if __name__ == "__main__":
    main()
