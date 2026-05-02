"""
compute_detection_pr.py — Compute frame-level precision/recall for
individual modalities vs fusion approaches.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "runs" / "reliability" / "fusion"


def compute_pr(detected, has_tp, drone_present):
    """Compute frame-level P, R, F1."""
    fires = detected.sum()
    correct = (detected & has_tp).sum()
    fp = (detected & ~has_tp).sum()
    missed = (drone_present & ~has_tp).sum()
    n_drone = drone_present.sum()

    p = correct / fires if fires > 0 else 0
    r = correct / n_drone if n_drone > 0 else 0
    f1 = 2 * p * r / (p + r + 1e-9)
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "tp_frames": int(correct),
        "fp_frames": int(fp),
        "missed": int(missed),
        "fires": int(fires),
        "n_drone": int(n_drone),
    }


def main():
    df = pd.read_csv(DATA_DIR / "fusion_dataset.csv")
    print(f"Loaded {len(df):,} frames\n")

    drone = df["drone_present"] == 1
    rgb_det = df["rgb_detected"] == 1
    ir_det = df["ir_detected"] == 1
    rgb_tp = df["rgb_has_tp"] == 1
    ir_tp = df["ir_has_tp"] == 1

    # ── OVERALL ────────────────────────────────────────────────
    print("=" * 75)
    print("FRAME-LEVEL DETECTION PRECISION / RECALL (all 152K paired frames)")
    print("=" * 75)

    header = (f"  {'System':<25s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}"
              f" {'TP':>8s} {'FP':>8s} {'Missed':>8s}")
    sep = f"  {'-' * 70}"

    print(header)
    print(sep)

    # RGB alone
    m = compute_pr(rgb_det, rgb_tp, drone)
    print(f"  {'RGB YOLO alone':<25s} {m['precision']:>7.4f} {m['recall']:>7.4f}"
          f" {m['f1']:>7.4f} {m['tp_frames']:>8,} {m['fp_frames']:>8,}"
          f" {m['missed']:>8,}")

    # IR alone
    m = compute_pr(ir_det, ir_tp, drone)
    print(f"  {'IR YOLO alone':<25s} {m['precision']:>7.4f} {m['recall']:>7.4f}"
          f" {m['f1']:>7.4f} {m['tp_frames']:>8,} {m['fp_frames']:>8,}"
          f" {m['missed']:>8,}")

    # OR gate
    or_det = rgb_det | ir_det
    or_tp = rgb_tp | ir_tp
    m = compute_pr(or_det, or_tp, drone)
    print(f"  {'OR gate (either)':<25s} {m['precision']:>7.4f} {m['recall']:>7.4f}"
          f" {m['f1']:>7.4f} {m['tp_frames']:>8,} {m['fp_frames']:>8,}"
          f" {m['missed']:>8,}")

    # AND gate
    and_det = rgb_det & ir_det
    and_tp = rgb_tp & ir_tp
    m = compute_pr(and_det, and_tp, drone)
    print(f"  {'AND gate (both)':<25s} {m['precision']:>7.4f} {m['recall']:>7.4f}"
          f" {m['f1']:>7.4f} {m['tp_frames']:>8,} {m['fp_frames']:>8,}"
          f" {m['missed']:>8,}")

    # ── PER DATASET ────────────────────────────────────────────
    datasets = sorted(df["source_dataset"].unique())
    for ds in datasets:
        s = df[df["source_dataset"] == ds]
        sd = s["drone_present"] == 1

        print(f"\n{'=' * 75}")
        print(f"DATASET: {ds} ({len(s):,} frames, {sd.sum():,} with drone)")
        print(f"{'=' * 75}")
        print(header)
        print(sep)

        for name, det_col, tp_col in [
            ("RGB YOLO alone", "rgb_detected", "rgb_has_tp"),
            ("IR YOLO alone", "ir_detected", "ir_has_tp"),
        ]:
            det = s[det_col] == 1
            tp = s[tp_col] == 1
            m = compute_pr(det, tp, sd)
            print(f"  {name:<25s} {m['precision']:>7.4f} {m['recall']:>7.4f}"
                  f" {m['f1']:>7.4f} {m['tp_frames']:>8,} {m['fp_frames']:>8,}"
                  f" {m['missed']:>8,}")

        # OR gate
        or_d = (s["rgb_detected"] == 1) | (s["ir_detected"] == 1)
        or_t = (s["rgb_has_tp"] == 1) | (s["ir_has_tp"] == 1)
        m = compute_pr(or_d, or_t, sd)
        print(f"  {'OR gate (either)':<25s} {m['precision']:>7.4f} {m['recall']:>7.4f}"
              f" {m['f1']:>7.4f} {m['tp_frames']:>8,} {m['fp_frames']:>8,}"
              f" {m['missed']:>8,}")

        # AND gate
        and_d = (s["rgb_detected"] == 1) & (s["ir_detected"] == 1)
        and_t = (s["rgb_has_tp"] == 1) & (s["ir_has_tp"] == 1)
        m = compute_pr(and_d, and_t, sd)
        print(f"  {'AND gate (both)':<25s} {m['precision']:>7.4f} {m['recall']:>7.4f}"
              f" {m['f1']:>7.4f} {m['tp_frames']:>8,} {m['fp_frames']:>8,}"
              f" {m['missed']:>8,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
