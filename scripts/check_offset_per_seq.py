"""
check_offset_per_seq.py — Fit IR→RGB transform PER SEQUENCE and evaluate.

Each sequence has a fixed camera rig, so the transform should be consistent
within a sequence even if it varies across sequences.

Usage:
    python check_offset_per_seq.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from numpy.polynomial.polynomial import polyfit

from utils import compute_iou


def parse_norm_centers(label_path):
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


def extract_seq(stem):
    """Extract sequence prefix from stem like '20190925_111757_1_1_f000010'."""
    # Everything before the last _fNNNNNN
    m = re.match(r"(.+)_f\d+$", stem)
    return m.group(1) if m else stem


def fit_transform(pts_ir, pts_rgb):
    """Fit linear transform per axis. Returns (sx, tx, sy, ty, sw, tw, sh, th)."""
    if len(pts_ir) < 3:
        return None
    c = {}
    for axis, (src, tgt) in enumerate(zip(pts_ir.T, pts_rgb.T)):
        coeff = polyfit(src, tgt, 1)  # [intercept, slope]
        c[axis] = (coeff[1], coeff[0])  # (scale, translate)
    return c  # {0: (sx,tx), 1: (sy,ty), 2: (sw,tw), 3: (sh,th)}


def apply_transform(ir_cxywh, transform):
    cx = transform[0][0] * ir_cxywh[0] + transform[0][1]
    cy = transform[1][0] * ir_cxywh[1] + transform[1][1]
    w = transform[2][0] * ir_cxywh[2] + transform[2][1]
    h = transform[3][0] * ir_cxywh[3] + transform[3][1]
    return (cx, cy, max(w, 0.001), max(h, 0.001))


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["dataset_root"])
    rgb_lbl_dir = root / cfg["rgb_subdir"] / "labels"
    ir_lbl_dir = root / cfg["ir_subdir"] / "labels"

    rgb_suffix = cfg["rgb_stem_suffix"]
    ir_suffix = cfg["ir_stem_suffix"]

    def lbl_stem_map(lbl_dir, suffix):
        out = {}
        for f in sorted(lbl_dir.iterdir()):
            if f.suffix.lower() == ".txt":
                key = f.stem.replace(suffix, "") if suffix else f.stem
                out[key] = f
        return out

    print("Scanning labels...")
    rgb_map = lbl_stem_map(rgb_lbl_dir, rgb_suffix)
    ir_map = lbl_stem_map(ir_lbl_dir, ir_suffix)
    shared = sorted(set(rgb_map) & set(ir_map))

    # Group by sequence
    seq_data = defaultdict(lambda: {"rgb": [], "ir": [], "stems": []})
    for stem in shared:
        rgb_boxes = parse_norm_centers(rgb_map[stem])
        ir_boxes = parse_norm_centers(ir_map[stem])
        if len(rgb_boxes) == 1 and len(ir_boxes) == 1:
            seq = extract_seq(stem)
            seq_data[seq]["rgb"].append(rgb_boxes[0])
            seq_data[seq]["ir"].append(ir_boxes[0])
            seq_data[seq]["stems"].append(stem)

    print(f"Sequences with paired single-box frames: {len(seq_data)}")

    # Fit per-sequence, evaluate with leave-out
    all_ious_before = []
    all_ious_after = []
    seq_results = {}
    transforms = {}
    skipped = 0

    for seq, data in sorted(seq_data.items()):
        rgb_pts = np.array(data["rgb"])
        ir_pts = np.array(data["ir"])
        n = len(rgb_pts)

        if n < 5:
            skipped += 1
            continue

        # Fit on ALL frames (we're building a dataset, not evaluating the transform)
        idx = np.arange(n)
        fit_idx = idx
        # Use last 20% as sanity check
        np.random.seed(42)
        shuffled = np.random.permutation(idx)
        test_idx = shuffled[max(int(0.8 * n), 3):]
        if len(test_idx) == 0:
            test_idx = idx[-2:]

        transform = fit_transform(ir_pts[fit_idx], rgb_pts[fit_idx])
        if transform is None:
            skipped += 1
            continue

        transforms[seq] = {k: (float(v[0]), float(v[1])) for k, v in transform.items()}

        ious_b = []
        ious_a = []
        for i in test_idx:
            rgb_box = to_xyxy(*rgb_pts[i])
            ir_box = to_xyxy(*ir_pts[i])
            ious_b.append(compute_iou(rgb_box, ir_box))

            corr = apply_transform(ir_pts[i], transform)
            corr_box = to_xyxy(*corr)
            ious_a.append(compute_iou(rgb_box, corr_box))

        ious_b = np.array(ious_b)
        ious_a = np.array(ious_a)
        all_ious_before.extend(ious_b)
        all_ious_after.extend(ious_a)

        seq_results[seq] = {
            "n_frames": n,
            "n_test": len(test_idx),
            "mean_iou_before": float(ious_b.mean()),
            "mean_iou_after": float(ious_a.mean()),
            "pct_above_0.5_after": float((ious_a > 0.5).mean() * 100),
        }

    all_before = np.array(all_ious_before)
    all_after = np.array(all_ious_after)

    print(f"  Skipped sequences (too few frames): {skipped}")
    print(f"\n{'='*60}")
    print(f"GLOBAL RESULTS (all test frames across all sequences)")
    print(f"{'='*60}")
    print(f"  Test frames: {len(all_after)}")

    print(f"\n  BEFORE per-seq correction:")
    print(f"    Mean IoU:   {all_before.mean():.4f}")
    print(f"    IoU > 0.3:  {(all_before > 0.3).sum()} ({(all_before > 0.3).mean()*100:.1f}%)")
    print(f"    IoU > 0.5:  {(all_before > 0.5).sum()} ({(all_before > 0.5).mean()*100:.1f}%)")

    print(f"\n  AFTER per-seq correction:")
    print(f"    Mean IoU:   {all_after.mean():.4f}")
    print(f"    IoU > 0.3:  {(all_after > 0.3).sum()} ({(all_after > 0.3).mean()*100:.1f}%)")
    print(f"    IoU > 0.5:  {(all_after > 0.5).sum()} ({(all_after > 0.5).mean()*100:.1f}%)")

    print(f"\n  IoU distribution AFTER per-seq correction:")
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                    (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                    (0.8, 0.9), (0.9, 1.01)]:
        n_bin = ((all_after >= lo) & (all_after < hi)).sum()
        print(f"    [{lo:.1f}, {hi:.1f}): {n_bin:>5} ({n_bin/len(all_after)*100:>5.1f}%)")

    # Top/bottom sequences
    print(f"\n  Best 5 sequences:")
    ranked = sorted(seq_results.items(), key=lambda x: x[1]["mean_iou_after"], reverse=True)
    for seq, r in ranked[:5]:
        print(f"    {seq}: IoU {r['mean_iou_before']:.3f} ->{r['mean_iou_after']:.3f} "
              f"({r['pct_above_0.5_after']:.0f}% >0.5, n={r['n_frames']})")

    print(f"\n  Worst 5 sequences:")
    for seq, r in ranked[-5:]:
        print(f"    {seq}: IoU {r['mean_iou_before']:.3f} ->{r['mean_iou_after']:.3f} "
              f"({r['pct_above_0.5_after']:.0f}% >0.5, n={r['n_frames']})")

    # Save
    out = {
        "per_sequence_transforms": transforms,
        "per_sequence_results": seq_results,
        "global_mean_iou_before": float(all_before.mean()),
        "global_mean_iou_after": float(all_after.mean()),
        "global_pct_above_0.5": float((all_after > 0.5).mean() * 100),
    }
    out_path = Path("runs/per_seq_transforms.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
