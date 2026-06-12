"""
sweep_confuser_threshold.py — Find optimal P(confuser) veto threshold.

Runs both confuser filter models on all crops in the manifest, then sweeps
thresholds to find the best tradeoff between:
  - Confuser catch rate (airplane/bird/helicopter correctly rejected)
  - Drone pass rate (drones NOT incorrectly rejected)

Usage:
    python classifier/sweep_confuser_threshold.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PATCH_DIR = SCRIPT_DIR / "runs" / "patches"
MANIFEST_PATH = PATCH_DIR / "manifest.csv"

CONFUSER_CATEGORIES = {"airplane", "helicopter", "bird"}


def main():
    # Load models
    sys.path.insert(0, str(SCRIPT_DIR))
    from patch_verifier import PatchVerifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Manifest: {len(manifest)} crops")

    for modality in ["rgb", "ir"]:
        model_path = PATCH_DIR / f"confuser_filter_{modality}.pt"
        if not model_path.exists():
            print(f"  SKIP {modality}: {model_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  {modality.upper()} Confuser Filter Threshold Sweep")
        print(f"{'='*60}")

        verifier = PatchVerifier(model_path, device=device)
        df = manifest[manifest["modality"] == modality].copy()
        print(f"  Crops: {len(df)}")

        # Run inference on all crops
        probs = []
        labels = []  # 1=confuser, 0=pass(drone)
        categories = []
        bad_loads = 0

        for i, (_, row) in enumerate(df.iterrows()):
            img_path = str(PROJECT_ROOT / row["path"])
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                bad_loads += 1
                continue

            # Run through the model
            p = verifier.predict_crops([img])[0]
            probs.append(float(p))
            is_confuser = 1 if row["category"] in CONFUSER_CATEGORIES else 0
            labels.append(is_confuser)
            categories.append(row["category"])

            if (i + 1) % 2000 == 0:
                print(f"    {i+1}/{len(df)} crops processed...")

        if bad_loads > 0:
            print(f"  Warning: {bad_loads} crops failed to load")

        probs = np.array(probs)
        labels = np.array(labels)
        cats = np.array(categories)

        # Per-category distribution
        print(f"\n  Per-category P(confuser) distribution:")
        for cat in sorted(set(cats)):
            mask = cats == cat
            p = probs[mask]
            expected = "REJECT" if cat in CONFUSER_CATEGORIES else "PASS"
            print(f"    {cat:<12s} n={mask.sum():5d}  "
                  f"mean={p.mean():.4f}  median={np.median(p):.4f}  "
                  f"p5={np.percentile(p,5):.4f}  p95={np.percentile(p,95):.4f}  "
                  f"[{expected}]")

        # Threshold sweep
        print(f"\n  {'Thr':>6s}  {'Confuser_Catch':>14s}  {'Drone_FalseRej':>14s}  "
              f"{'Confuser_Miss':>13s}  {'F1':>6s}  {'Note':>20s}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*13}  {'-'*6}  {'-'*20}")

        confuser_mask = labels == 1
        drone_mask = labels == 0
        n_confuser = confuser_mask.sum()
        n_drone = drone_mask.sum()

        best_thr = 0.5
        best_score = -1

        for thr in np.arange(0.10, 1.00, 0.05):
            # pred=1 means "confuser detected" → veto
            pred_confuser = probs >= thr

            # Confuser catch rate: what fraction of actual confusers are caught
            confuser_caught = pred_confuser[confuser_mask].sum()
            confuser_catch_rate = confuser_caught / n_confuser if n_confuser > 0 else 0

            # Drone false rejection: what fraction of drones are incorrectly rejected
            drone_rejected = pred_confuser[drone_mask].sum()
            drone_false_rej = drone_rejected / n_drone if n_drone > 0 else 0

            # Confuser miss rate
            confuser_missed = n_confuser - confuser_caught
            confuser_miss_rate = confuser_missed / n_confuser if n_confuser > 0 else 0

            # F1 for confuser detection
            tp = confuser_caught
            fp = drone_rejected
            fn = confuser_missed
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = confuser_catch_rate
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Score: maximize confuser catch while penalizing drone false rejection
            # We want drone_false_rej < 2%
            note = ""
            if drone_false_rej <= 0.02 and confuser_catch_rate > best_score:
                best_score = confuser_catch_rate
                best_thr = thr
                note = "<-- BEST (FRR<=2%)"

            print(f"  {thr:6.2f}  {confuser_catch_rate:14.4f}  {drone_false_rej:14.4f}  "
                  f"{confuser_miss_rate:13.4f}  {f1:6.4f}  {note:>20s}")

        print(f"\n  >>> OPTIMAL THRESHOLD for {modality.upper()}: {best_thr:.2f}")
        print(f"      Confuser catch rate: {best_score:.4f}")

        # Also show what happens at a few key thresholds
        print(f"\n  Key thresholds detail:")
        for thr in [0.50, best_thr, 0.80, 0.90]:
            pred = probs >= thr
            print(f"\n    --- Threshold = {thr:.2f} ---")
            for cat in sorted(set(cats)):
                mask = cats == cat
                flagged = pred[mask].sum()
                total = mask.sum()
                rate = flagged / total if total > 0 else 0
                action = "REJECT" if cat in CONFUSER_CATEGORIES else "PASS"
                status = "OK" if (action == "REJECT" and rate > 0.9) or (action == "PASS" and rate < 0.02) else "WARN"
                print(f"      {cat:<12s} {flagged:5d}/{total:5d} flagged ({rate:.1%})  "
                      f"[want: {action}]  {status}")


if __name__ == "__main__":
    main()
