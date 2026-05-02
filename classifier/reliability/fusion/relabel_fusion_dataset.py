"""
relabel_fusion_dataset.py — Patch fusion_dataset.csv with relaxed matching.

Uses IoU >= threshold OR IoP (intersection over prediction) >= 0.5,
correctly handling oversized GT boxes where predictions are tight but correct.
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "inference"
FUSION_CSV = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "fusion" / "fusion_dataset.csv"

PAIRED_DATASETS = {
    "antiuav_val":  ("antiuav_val_rgb",  "antiuav_val_ir"),
    "antiuav_test": ("antiuav_test_rgb", "antiuav_test_ir"),
    "svanstrom":    ("svanstrom_rgb",    "svanstrom_ir"),
}

CONF_THRESH = 0.4
GT_IOU = 0.5       # same as original build
IOP_THRESH = 0.5   # NEW: intersection over prediction area


def strip_modality_suffix(stem):
    """Same logic as build_fusion_dataset.py."""
    return re.sub(r"_(visible|infrared)", "", stem, flags=re.IGNORECASE)


def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def compute_iop(pred, gt):
    """Intersection over Prediction area."""
    x1 = max(pred[0], gt[0]); y1 = max(pred[1], gt[1])
    x2 = min(pred[2], gt[2]); y2 = min(pred[3], gt[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    pred_area = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
    return inter / pred_area if pred_area > 0 else 0


def parse_gt(gt_text, img_w, img_h):
    boxes = []
    if not gt_text.strip():
        return boxes
    for line in gt_text.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def match_relaxed(dets, gt_boxes):
    """Match using IoU >= GT_IOU OR IoP >= IOP_THRESH."""
    n_det = len(dets)
    n_gt = len(gt_boxes)
    if n_det == 0 or n_gt == 0:
        return 0, n_det

    det_matched = [False] * n_det
    gt_matched = [False] * n_gt

    pairs = []
    for di in range(n_det):
        for gi in range(n_gt):
            iou = compute_iou(dets[di][:4], gt_boxes[gi])
            iop = compute_iop(dets[di][:4], gt_boxes[gi])
            score = max(iou, iop)
            if iou >= GT_IOU or iop >= IOP_THRESH:
                pairs.append((score, di, gi))

    pairs.sort(reverse=True)
    for _, di, gi in pairs:
        if not det_matched[di] and not gt_matched[gi]:
            det_matched[di] = True
            gt_matched[gi] = True

    n_tp = sum(det_matched)
    return n_tp, n_det - n_tp


def assign_trust_label(rgb_has_tp, ir_has_tp, drone_present):
    if not drone_present:
        return 0
    if rgb_has_tp and ir_has_tp:
        return 3
    if rgb_has_tp:
        return 1
    if ir_has_tp:
        return 2
    return 0


def main():
    print("=" * 70)
    print("Relabeling fusion_dataset.csv with relaxed IoU+IoP matching")
    print(f"  IoU threshold: {GT_IOU}, IoP threshold: {IOP_THRESH}")
    print(f"  Conf threshold: {CONF_THRESH}")
    print("=" * 70)

    df = pd.read_csv(FUSION_CSV)
    print(f"\nLoaded {len(df):,} rows")

    # Save old labels for comparison
    old_trust = df["trust_label"].copy()

    # Load inference JSONs and build base_stem -> json_stem mappings
    # Key insight: CSV base_stems have _visible/_infrared stripped
    all_data = {}  # {ds_name: {"rgb": {base_stem: frame}, "ir": {base_stem: frame}}}

    for ds_name, (rgb_tag, ir_tag) in PAIRED_DATASETS.items():
        rgb_path = INFERENCE_DIR / f"{rgb_tag}.json"
        ir_path = INFERENCE_DIR / f"{ir_tag}.json"

        print(f"\n  Loading {rgb_tag}...", end="", flush=True)
        with open(rgb_path) as f:
            raw_rgb = json.load(f)
        # Build base_stem -> frame mapping
        rgb_by_base = {}
        for json_stem, frame in raw_rgb.items():
            base = strip_modality_suffix(json_stem)
            rgb_by_base[base] = frame
        print(f" {len(rgb_by_base):,} frames")

        print(f"  Loading {ir_tag}...", end="", flush=True)
        with open(ir_path) as f:
            raw_ir = json.load(f)
        ir_by_base = {}
        for json_stem, frame in raw_ir.items():
            base = strip_modality_suffix(json_stem)
            ir_by_base[base] = frame
        print(f" {len(ir_by_base):,} frames")

        all_data[ds_name] = {"rgb": rgb_by_base, "ir": ir_by_base}

    # Re-match each row
    t0 = time.time()
    changes = 0
    not_found = 0

    for idx in range(len(df)):
        row = df.iloc[idx]
        ds = row["source_dataset"]
        stem = row["base_stem"]

        rgb_frame = all_data[ds]["rgb"].get(stem)
        ir_frame = all_data[ds]["ir"].get(stem)

        if rgb_frame is None or ir_frame is None:
            not_found += 1
            continue

        rgb_dets = [d for d in rgb_frame["dets"] if d[4] >= CONF_THRESH]
        ir_dets = [d for d in ir_frame["dets"] if d[4] >= CONF_THRESH]

        rgb_gt = parse_gt(rgb_frame.get("gt", ""), rgb_frame["w"], rgb_frame["h"])
        ir_gt = parse_gt(ir_frame.get("gt", ""), ir_frame["w"], ir_frame["h"])

        drone_present = 1 if (rgb_gt or ir_gt) else 0

        rgb_tp, rgb_fp = match_relaxed(rgb_dets, rgb_gt)
        ir_tp, ir_fp = match_relaxed(ir_dets, ir_gt)

        rgb_has_tp = 1 if rgb_tp > 0 else 0
        ir_has_tp = 1 if ir_tp > 0 else 0
        trust = assign_trust_label(rgb_has_tp, ir_has_tp, drone_present)

        if df.at[idx, "trust_label"] != trust:
            changes += 1

        df.at[idx, "rgb_tp"] = rgb_tp
        df.at[idx, "rgb_fp"] = rgb_fp
        df.at[idx, "ir_tp"] = ir_tp
        df.at[idx, "ir_fp"] = ir_fp
        df.at[idx, "rgb_has_tp"] = rgb_has_tp
        df.at[idx, "ir_has_tp"] = ir_has_tp
        df.at[idx, "trust_label"] = trust
        df.at[idx, "drone_present"] = drone_present

        if (idx + 1) % 25000 == 0:
            elapsed = time.time() - t0
            print(f"    [{idx+1:,}/{len(df):,}] {elapsed:.1f}s, {changes:,} changed, "
                  f"{not_found} not found")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Trust label changes: {changes:,}")
    print(f"  Frames not found: {not_found}")

    # Per-dataset changes
    print("\n  Changes per dataset:")
    for ds in df["source_dataset"].unique():
        mask = df["source_dataset"] == ds
        n_diff = (old_trust[mask].values != df.loc[mask, "trust_label"].values).sum()
        print(f"    {ds:<15s}: {n_diff:,} label changes")

    # Distribution comparison
    print("\n  Trust label distribution (old -> new):")
    label_names = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
    for v, name in label_names.items():
        old_n = (old_trust == v).sum()
        new_n = (df["trust_label"] == v).sum()
        delta = new_n - old_n
        print(f"    {v} ({name:<12s}): {old_n:>8,} -> {new_n:>8,}  ({delta:>+6,})")

    # Save
    backup_path = FUSION_CSV.with_suffix(".csv.bak_strict_iou")
    if not backup_path.exists():
        import shutil
        shutil.copy2(FUSION_CSV, backup_path)
        print(f"\n  Backed up original to {backup_path.name}")

    df.to_csv(FUSION_CSV, index=False)
    print(f"  Saved updated {FUSION_CSV.name}")
    print("\nDone. Run eval_all_fusion.py to see updated results.")


if __name__ == "__main__":
    main()
