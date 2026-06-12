"""
check_offset.py — Estimate a fixed scale+translate mapping between
RGB and IR normalized bounding box coordinates.

If the sensors are co-mounted, there should be a consistent transform:
    rgb_cx = sx * ir_cx + tx
    rgb_cy = sy * ir_cy + ty

This script fits that transform on a subset and evaluates IoU after correction.

Usage:
    python check_offset.py
"""

import json
import random
from pathlib import Path

import numpy as np
import yaml

from utils import compute_iou


def parse_norm_centers(label_path):
    """Parse YOLO labels → list of (cx, cy, w, h) normalized."""
    boxes = []
    if not label_path.exists():
        return boxes
    text = label_path.read_text().strip()
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cx, cy, w, h))
    return boxes


def to_xyxy(cx, cy, w, h):
    return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["dataset_root"])
    rgb_lbl_dir = root / cfg["rgb_subdir"] / "labels"
    ir_lbl_dir = root / cfg["ir_subdir"] / "labels"
    rgb_img_dir = root / cfg["rgb_subdir"] / "images"
    ir_img_dir = root / cfg["ir_subdir"] / "images"

    rgb_suffix = cfg["rgb_stem_suffix"]
    ir_suffix = cfg["ir_stem_suffix"]

    # Scan label dirs directly (much faster than scanning images on external drives)
    def lbl_stem_map(lbl_dir, suffix):
        out = {}
        for f in sorted(lbl_dir.iterdir()):
            if f.suffix.lower() == ".txt":
                key = f.stem.replace(suffix, "") if suffix else f.stem
                out[key] = f
        return out

    print("Scanning label directories...")
    rgb_lbl_map = lbl_stem_map(rgb_lbl_dir, rgb_suffix)
    ir_lbl_map = lbl_stem_map(ir_lbl_dir, ir_suffix)
    shared = sorted(set(rgb_lbl_map) & set(ir_lbl_map))
    print(f"  Paired label files: {len(shared)}")

    # Collect paired GT center points (only frames with exactly 1 box each)
    rgb_pts = []
    ir_pts = []

    for stem in shared:
        rgb_lbl = rgb_lbl_map[stem]
        ir_lbl = ir_lbl_map[stem]

        rgb_boxes = parse_norm_centers(rgb_lbl)
        ir_boxes = parse_norm_centers(ir_lbl)

        # Use only single-box frames for clean pairing
        if len(rgb_boxes) == 1 and len(ir_boxes) == 1:
            rgb_pts.append(rgb_boxes[0])
            ir_pts.append(ir_boxes[0])

    rgb_pts = np.array(rgb_pts)  # (N, 4) = cx, cy, w, h
    ir_pts = np.array(ir_pts)

    print(f"Single-box paired frames: {len(rgb_pts)}")
    if len(rgb_pts) < 50:
        print("Not enough paired points to estimate transform.")
        return

    # --- Fit: rgb = scale * ir + translate (per axis) ---
    # Use 70% for fitting, 30% for testing
    n = len(rgb_pts)
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    split = int(0.7 * n)
    fit_idx, test_idx = idx[:split], idx[split:]

    # Fit scale+translate for cx, cy independently
    from numpy.polynomial.polynomial import polyfit

    # cx: rgb_cx = sx * ir_cx + tx
    sx_coeff = polyfit(ir_pts[fit_idx, 0], rgb_pts[fit_idx, 0], 1)  # [intercept, slope]
    tx, sx = sx_coeff[0], sx_coeff[1]

    # cy: rgb_cy = sy * ir_cy + ty
    sy_coeff = polyfit(ir_pts[fit_idx, 1], rgb_pts[fit_idx, 1], 1)
    ty, sy = sy_coeff[0], sy_coeff[1]

    # w: rgb_w = sw * ir_w + tw
    sw_coeff = polyfit(ir_pts[fit_idx, 2], rgb_pts[fit_idx, 2], 1)
    tw, sw = sw_coeff[0], sw_coeff[1]

    # h: rgb_h = sh * ir_h + th
    sh_coeff = polyfit(ir_pts[fit_idx, 3], rgb_pts[fit_idx, 3], 1)
    th, sh = sh_coeff[0], sh_coeff[1]

    print(f"\n--- Fitted Transform (IR → RGB normalized coords) ---")
    print(f"  cx: rgb = {sx:.4f} * ir + {tx:.4f}")
    print(f"  cy: rgb = {sy:.4f} * ir + {ty:.4f}")
    print(f"   w: rgb = {sw:.4f} * ir + {tw:.4f}")
    print(f"   h: rgb = {sh:.4f} * ir + {th:.4f}")

    # --- Evaluate on test set ---
    ious_before = []
    ious_after = []

    for i in test_idx:
        r_cx, r_cy, r_w, r_h = rgb_pts[i]
        i_cx, i_cy, i_w, i_h = ir_pts[i]

        # Before correction
        rgb_box = to_xyxy(r_cx, r_cy, r_w, r_h)
        ir_box = to_xyxy(i_cx, i_cy, i_w, i_h)
        ious_before.append(compute_iou(rgb_box, ir_box))

        # After correction: transform IR → RGB space
        corr_cx = sx * i_cx + tx
        corr_cy = sy * i_cy + ty
        corr_w = sw * i_w + tw
        corr_h = sh * i_h + th
        corr_box = to_xyxy(corr_cx, corr_cy, corr_w, corr_h)
        ious_after.append(compute_iou(rgb_box, corr_box))

    ious_before = np.array(ious_before)
    ious_after = np.array(ious_after)

    print(f"\n--- Test Set Results ({len(test_idx)} frames) ---")
    print(f"  BEFORE correction:")
    print(f"    Mean IoU:   {ious_before.mean():.4f}")
    print(f"    Median IoU: {np.median(ious_before):.4f}")
    print(f"    IoU > 0.3:  {(ious_before > 0.3).sum()} ({(ious_before > 0.3).mean()*100:.1f}%)")
    print(f"    IoU > 0.5:  {(ious_before > 0.5).sum()} ({(ious_before > 0.5).mean()*100:.1f}%)")

    print(f"\n  AFTER correction:")
    print(f"    Mean IoU:   {ious_after.mean():.4f}")
    print(f"    Median IoU: {np.median(ious_after):.4f}")
    print(f"    IoU > 0.3:  {(ious_after > 0.3).sum()} ({(ious_after > 0.3).mean()*100:.1f}%)")
    print(f"    IoU > 0.5:  {(ious_after > 0.5).sum()} ({(ious_after > 0.5).mean()*100:.1f}%)")

    # Distribution after
    print(f"\n  IoU distribution AFTER correction:")
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                    (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                    (0.8, 0.9), (0.9, 1.01)]:
        n_bin = ((ious_after >= lo) & (ious_after < hi)).sum()
        print(f"    [{lo:.1f}, {hi:.1f}): {n_bin:>5} ({n_bin/len(ious_after)*100:>5.1f}%)")

    # Residual analysis — check if transform is consistent or varies by sequence
    residuals_cx = rgb_pts[test_idx, 0] - (sx * ir_pts[test_idx, 0] + tx)
    residuals_cy = rgb_pts[test_idx, 1] - (sy * ir_pts[test_idx, 1] + ty)
    print(f"\n  Residuals (RGB_actual - RGB_predicted):")
    print(f"    cx: mean={residuals_cx.mean():.4f}, std={residuals_cx.std():.4f}")
    print(f"    cy: mean={residuals_cy.mean():.4f}, std={residuals_cy.std():.4f}")

    # Save transform
    transform = {
        "sx": float(sx), "tx": float(tx),
        "sy": float(sy), "ty": float(ty),
        "sw": float(sw), "tw": float(tw),
        "sh": float(sh), "th": float(th),
        "test_mean_iou_before": float(ious_before.mean()),
        "test_mean_iou_after": float(ious_after.mean()),
        "test_iou_gt_0.5_pct": float((ious_after > 0.5).mean() * 100),
        "n_fit": int(split),
        "n_test": int(len(test_idx)),
    }
    out_path = Path("runs/ir_to_rgb_transform.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(transform, f, indent=2)
    print(f"\n  Transform saved to {out_path}")


if __name__ == "__main__":
    main()
