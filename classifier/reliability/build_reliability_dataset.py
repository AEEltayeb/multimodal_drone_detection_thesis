"""
build_reliability_dataset.py — Convert cached inference JSONs into per-detection
CSV datasets for training reliability classifiers.

For each detection:
  1. Compare against GT using IoU matching
  2. Label: 1=TP (IoU >= 0.5), 0=FP
  3. Extract features: conf, box geometry, clutter signals

Outputs:
  runs/reliability/rgb_reliability_dataset.csv
  runs/reliability/ir_reliability_dataset.csv

Usage:
    python build_reliability_dataset.py
    python build_reliability_dataset.py --iou-thresh 0.3
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── PATHS ───────────────────────────────────────────────────────────
INFERENCE_DIR = Path(__file__).resolve().parent.parent / "runs" / "reliability" / "inference"
OUTPUT_DIR    = Path(__file__).resolve().parent.parent / "runs" / "reliability"

# Which inference files belong to which modality
RGB_DATASETS = [
    "rgb_dataset_val",
    "rgb_dataset_test",
    "antiuav_val_rgb",
    "antiuav_test_rgb",
    "svanstrom_rgb",
]

IR_DATASETS = [
    "ir_dset_final_val",
    "ir_dset_final_test",
    "cst_antiuav_test",
    "antiuav_val_ir",
    "antiuav_test_ir",
    "svanstrom_ir",
]


# ── IoU MATCHING ────────────────────────────────────────────────────
def compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def parse_yolo_gt(gt_text, img_w, img_h):
    """Parse YOLO label text → list of [x1, y1, x2, y2] pixel boxes."""
    boxes = []
    if not gt_text.strip():
        return boxes
    for line in gt_text.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # class cx cy w h (normalized)
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def match_detections_to_gt(dets, gt_boxes, iou_thresh=0.5):
    """
    Greedy IoU matching: each detection gets label 1 (TP) or 0 (FP).
    Each GT box can only match one detection (highest IoU first).
    Returns list of labels, one per detection.
    """
    n_dets = len(dets)
    if n_dets == 0:
        return []

    labels = [0] * n_dets  # default FP

    if not gt_boxes:
        return labels  # all FP, no GT

    # Compute IoU matrix
    matched_gt = set()
    # Sort detections by confidence (highest first) for greedy matching
    det_indices = sorted(range(n_dets), key=lambda i: dets[i][4], reverse=True)

    for di in det_indices:
        det_box = dets[di][:4]
        best_iou = 0
        best_gi = -1
        for gi, gt_box in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(det_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_thresh and best_gi >= 0:
            labels[di] = 1  # TP
            matched_gt.add(best_gi)

    return labels


# ── FEATURE EXTRACTION ──────────────────────────────────────────────
def extract_detection_features(dets, det_idx, img_w, img_h):
    """Extract features for a single detection (by index)."""
    det = dets[det_idx]
    x1, y1, x2, y2, conf = det[0], det[1], det[2], det[3], det[4]

    box_w = x2 - x1
    box_h = y2 - y1
    img_area = max(1.0, img_w * img_h)
    box_area = box_w * box_h

    # All confidences in this frame, sorted descending
    all_confs = sorted([d[4] for d in dets], reverse=True)
    rank = all_confs.index(conf)  # 0 = highest conf detection

    return {
        "conf": round(conf, 6),
        "box_area_norm": round(box_area / img_area, 8),
        "aspect_ratio": round(box_w / max(box_h, 1e-6), 4),
        "box_w_norm": round(box_w / img_w, 6),
        "box_h_norm": round(box_h / img_h, 6),
        "box_center_x": round((x1 + x2) / 2 / img_w, 6),
        "box_center_y": round((y1 + y2) / 2 / img_h, 6),
        "n_dets": len(dets),
        "conf_rank": rank,  # 0 = best detection
        "conf_2nd": round(all_confs[1], 6) if len(all_confs) > 1 else 0.0,
        "conf_margin": round(all_confs[0] - all_confs[1], 6) if len(all_confs) > 1 else round(all_confs[0], 6),
        "conf_mean_frame": round(np.mean(all_confs), 6),
    }


FEATURE_COLS = [
    "conf", "box_area_norm", "aspect_ratio",
    "box_w_norm", "box_h_norm",
    "box_center_x", "box_center_y",
    "n_dets", "conf_rank",
    "conf_2nd", "conf_margin", "conf_mean_frame",
]

META_COLS = ["stem", "source_dataset", "label"]


# ── MAIN PROCESSING ────────────────────────────────────────────────
def process_dataset(tag, iou_thresh):
    """Process one inference JSON → list of feature dicts."""
    json_path = INFERENCE_DIR / f"{tag}.json"
    if not json_path.exists():
        print(f"  [SKIP] {tag}: {json_path} not found")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    n_tp, n_fp, n_frames_with_dets, n_empty = 0, 0, 0, 0

    for stem, frame in data.items():
        dets = frame["dets"]
        img_w = frame["w"]
        img_h = frame["h"]
        gt_text = frame.get("gt", "")

        if not dets:
            n_empty += 1
            continue

        n_frames_with_dets += 1

        # Parse GT and match
        gt_boxes = parse_yolo_gt(gt_text, img_w, img_h)
        labels = match_detections_to_gt(dets, gt_boxes, iou_thresh)

        # Extract features for each detection
        for i, label in enumerate(labels):
            feats = extract_detection_features(dets, i, img_w, img_h)
            feats["stem"] = stem
            feats["source_dataset"] = tag
            feats["label"] = label
            rows.append(feats)

            if label == 1:
                n_tp += 1
            else:
                n_fp += 1

    total = n_tp + n_fp
    tp_rate = n_tp / total * 100 if total > 0 else 0
    print(f"  {tag:<25s} {len(data):>7d} frames | "
          f"{n_frames_with_dets:>6d} with dets | "
          f"{total:>7d} detections | "
          f"TP: {n_tp:>6d} ({tp_rate:.1f}%), FP: {n_fp:>6d}")

    return rows


def build_dataset(dataset_tags, modality, iou_thresh):
    """Build full reliability dataset for one modality."""
    print(f"\n{'='*70}")
    print(f"Building {modality.upper()} reliability dataset")
    print(f"{'='*70}")
    print(f"  IoU threshold: {iou_thresh}")
    print(f"  Datasets: {len(dataset_tags)}")
    print()

    all_rows = []
    for tag in dataset_tags:
        rows = process_dataset(tag, iou_thresh)
        all_rows.extend(rows)

    if not all_rows:
        print(f"\n  [ERROR] No data collected for {modality}!")
        return None

    df = pd.DataFrame(all_rows)
    df = df[META_COLS + FEATURE_COLS]

    # Summary
    print(f"\n  Total: {len(df)} detections")
    print(f"    TP: {(df['label']==1).sum()} ({(df['label']==1).mean()*100:.1f}%)")
    print(f"    FP: {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")

    print(f"\n  Per-dataset breakdown:")
    print(f"    {'dataset':<25s} {'total':>7s} {'TP':>7s} {'FP':>7s} {'TP%':>7s}")
    for tag in dataset_tags:
        subset = df[df["source_dataset"] == tag]
        if len(subset) == 0:
            continue
        tp = (subset["label"] == 1).sum()
        fp = (subset["label"] == 0).sum()
        print(f"    {tag:<25s} {len(subset):>7d} {tp:>7d} {fp:>7d} "
              f"{tp/len(subset)*100:>6.1f}%")

    print(f"\n  Feature stats:")
    for col in FEATURE_COLS:
        print(f"    {col:<20s} mean={df[col].mean():.4f}  "
              f"std={df[col].std():.4f}  "
              f"min={df[col].min():.4f}  max={df[col].max():.4f}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU threshold for TP/FP matching (default: 0.5)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build RGB reliability dataset
    rgb_df = build_dataset(RGB_DATASETS, "rgb", args.iou_thresh)
    if rgb_df is not None:
        out_path = OUTPUT_DIR / "rgb_reliability_dataset.csv"
        rgb_df.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path} ({len(rgb_df)} rows)")

    # Build IR reliability dataset
    ir_df = build_dataset(IR_DATASETS, "ir", args.iou_thresh)
    if ir_df is not None:
        out_path = OUTPUT_DIR / "ir_reliability_dataset.csv"
        ir_df.to_csv(out_path, index=False)
        print(f"\n  Saved -> {out_path} ({len(ir_df)} rows)")

    # Combined summary
    if rgb_df is not None and ir_df is not None:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  RGB: {len(rgb_df):>8,} detections "
              f"({(rgb_df['label']==1).sum():,} TP, {(rgb_df['label']==0).sum():,} FP)")
        print(f"  IR:  {len(ir_df):>8,} detections "
              f"({(ir_df['label']==1).sum():,} TP, {(ir_df['label']==0).sum():,} FP)")
        print(f"\n  Ready for train_reliability.py")


if __name__ == "__main__":
    main()
