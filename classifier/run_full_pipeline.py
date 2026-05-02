"""
run_full_pipeline.py — Wait for brightness run to finish, merge datasets, retrain.

Steps:
  1. Wait for Anti-UAV brightness CSV to be ready (polls for rgb_brightness column)
  2. Merge Anti-UAV + Svanstrom frame-level datasets
  3. Train classifier on merged data
  4. Print comparison: Anti-UAV only vs merged

Usage:
    python run_full_pipeline.py
"""

import time
import subprocess
import sys
from pathlib import Path

import pandas as pd


RUNS = Path("runs")
ANTI_UAV_CSV = RUNS / "fusion_dataset.csv"
SVANSTROM_CSV = RUNS / "svanstrom_frame_dataset.csv"
MERGED_CSV = RUNS / "merged_frame_dataset.csv"


def wait_for_brightness():
    """Poll until Anti-UAV CSV has rgb_brightness column."""
    print("=" * 60)
    print("Step 1: Waiting for brightness features...")
    print("=" * 60)

    while True:
        if ANTI_UAV_CSV.exists():
            header = ANTI_UAV_CSV.open().readline().strip()
            if "rgb_brightness" in header:
                print("  Brightness columns found!")
                return True

        print(f"  [{time.strftime('%H:%M:%S')}] Still waiting... (checking every 60s)")
        time.sleep(60)


def merge_datasets():
    """Merge Anti-UAV and Svanstrom frame-level CSVs."""
    print()
    print("=" * 60)
    print("Step 2: Merging datasets")
    print("=" * 60)

    df_anti = pd.read_csv(ANTI_UAV_CSV)
    df_svan = pd.read_csv(SVANSTROM_CSV)

    print(f"  Anti-UAV: {len(df_anti)} rows ({df_anti['label'].sum()} pos, "
          f"{len(df_anti) - df_anti['label'].sum()} neg)")
    print(f"  Svanstrom: {len(df_svan)} rows ({df_svan['label'].sum()} pos, "
          f"{len(df_svan) - df_svan['label'].sum()} neg)")

    # Add dataset source column
    df_anti["dataset"] = "anti_uav"
    df_svan["dataset"] = "svanstrom"

    # Align columns — Svanstrom won't have brightness, fill with median from Anti-UAV
    for col in df_anti.columns:
        if col not in df_svan.columns:
            if col in ["rgb_brightness", "ir_brightness"]:
                median_val = df_anti[col].median() if col in df_anti.columns else 128.0
                df_svan[col] = median_val
                print(f"  Filled Svanstrom '{col}' with median={median_val:.1f}")
            elif col == "dataset":
                pass  # already set
            else:
                df_svan[col] = 0

    # Keep only shared columns + dataset
    shared_cols = [c for c in df_anti.columns if c in df_svan.columns]
    df_merged = pd.concat([df_anti[shared_cols], df_svan[shared_cols]], ignore_index=True)

    # Save
    df_merged.to_csv(MERGED_CSV, index=False)

    n_pos = df_merged["label"].sum()
    n_neg = len(df_merged) - n_pos
    print(f"\n  Merged: {len(df_merged)} rows ({n_pos} pos, {n_neg} neg)")
    print(f"  Saved to {MERGED_CSV}")

    return df_merged


def train_and_compare():
    """Train on merged data, then compare with Anti-UAV only."""
    print()
    print("=" * 60)
    print("Step 3: Training on MERGED dataset")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, "train_classifier.py", "--csv", str(MERGED_CSV)],
        capture_output=False,
    )
    if result.returncode != 0:
        print("  ERROR: Training failed!")
        return

    print()
    print("=" * 60)
    print("Step 4: Training on Anti-UAV ONLY (for comparison)")
    print("=" * 60)

    # Save current model, train on anti-uav only
    import shutil
    if (RUNS / "classifier.joblib").exists():
        shutil.copy(RUNS / "classifier.joblib", RUNS / "classifier_merged.joblib")
    if (RUNS / "metrics.json").exists():
        shutil.copy(RUNS / "metrics.json", RUNS / "metrics_merged.json")
    if (RUNS / "feature_importance.png").exists():
        shutil.copy(RUNS / "feature_importance.png", RUNS / "feature_importance_merged.png")

    result = subprocess.run(
        [sys.executable, "train_classifier.py", "--csv", str(ANTI_UAV_CSV)],
        capture_output=False,
    )

    # Restore merged as primary
    if (RUNS / "classifier_merged.joblib").exists():
        shutil.copy(RUNS / "classifier_merged.joblib", RUNS / "classifier.joblib")
    if (RUNS / "metrics_merged.json").exists():
        shutil.copy(RUNS / "metrics_merged.json", RUNS / "metrics.json")

    print()
    print("=" * 60)
    print("DONE — Compare metrics_merged.json vs the Anti-UAV-only run above")
    print("=" * 60)


def main():
    # Verify Svanstrom CSV exists
    if not SVANSTROM_CSV.exists():
        print(f"ERROR: {SVANSTROM_CSV} not found. Run Svanstrom build first.")
        return

    wait_for_brightness()
    merge_datasets()
    train_and_compare()


if __name__ == "__main__":
    main()
