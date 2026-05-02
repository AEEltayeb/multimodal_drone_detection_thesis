"""
diagnose_ir_only.py — Diagnose why IR-only detections have low TP rate.

Runs the IR model on a random subset of IR images that have GT labels,
and evaluates detection performance at multiple IoU thresholds to see
if the issue is tight bounding boxes vs actual misdetection.

Usage:
    python diagnose_ir_only.py
    python diagnose_ir_only.py --n 500
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

from utils import compute_iou, parse_yolo_labels


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def match_detections_to_gt(det_boxes, det_confs, gt_boxes, iou_thresh):
    """Greedy match detections to GT. Returns TP count, FP count, FN count, and per-match IoUs."""
    if not det_boxes:
        return 0, 0, len(gt_boxes), []

    # Sort detections by confidence (highest first)
    order = sorted(range(len(det_boxes)), key=lambda i: det_confs[i], reverse=True)

    matched_gt = set()
    tp = 0
    fp = 0
    match_ious = []

    for di in order:
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(det_boxes[di], gt)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_iou >= iou_thresh and best_gi >= 0:
            tp += 1
            matched_gt.add(best_gi)
            match_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, match_ious


def main():
    parser = argparse.ArgumentParser(description="Diagnose IR-only detection performance")
    parser.add_argument("--n", type=int, default=500, help="Number of frames to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold (default: from config)")
    args = parser.parse_args()

    cfg = load_config()
    root = Path(cfg["dataset_root"])
    ir_img_dir = root / cfg["ir_subdir"] / "images"
    ir_lbl_dir = root / cfg["ir_subdir"] / "labels"
    ir_suffix = cfg["ir_stem_suffix"]

    # IR image dimensions
    IR_W, IR_H = 640, 512

    # Find all IR images that have GT labels
    print("Scanning IR labels...")
    labeled_frames = []
    for lbl in sorted(ir_lbl_dir.iterdir()):
        if lbl.suffix != ".txt":
            continue
        gt = parse_yolo_labels(lbl, IR_W, IR_H)
        if not gt:  # skip empty labels (no GT boxes)
            continue
        stem = lbl.stem
        img_path = ir_img_dir / (stem + ".jpg")
        if not img_path.exists():
            img_path = ir_img_dir / (stem + ".png")
        if img_path.exists():
            labeled_frames.append((img_path, lbl, gt))

    print(f"  IR frames with GT: {len(labeled_frames)}")

    # Sample
    random.seed(args.seed)
    n = min(args.n, len(labeled_frames))
    sample = random.sample(labeled_frames, n)
    print(f"  Sampled: {n} frames")

    # Load IR model
    print(f"\nLoading IR model: {cfg['ir_weights']}")
    model = YOLO(cfg["ir_weights"])

    conf_thresh = args.conf if args.conf is not None else cfg["conf"]
    print(f"  Confidence threshold: {conf_thresh}")
    print(f"  Running inference...\n")

    # Run inference and collect stats
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    stats = {t: {"tp": 0, "fp": 0, "fn": 0} for t in iou_thresholds}
    all_match_ious = []

    total_gt = 0
    total_dets = 0
    frames_with_no_det = 0
    det_conf_on_tp = []  # confidence of TP detections
    det_conf_on_fp = []  # confidence of FP detections

    t0 = time.time()
    for idx, (img_path, lbl_path, gt_boxes) in enumerate(sample):
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (n - idx - 1)
            print(f"  [{idx+1}/{n}] elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

        # Run model
        results = model(str(img_path), imgsz=cfg["imgsz"], conf=conf_thresh,
                        iou=cfg["iou_nms"], max_det=cfg["max_det"],
                        verbose=False)

        det_boxes = []
        det_confs = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                det_boxes.append(tuple(xyxy))
                det_confs.append(conf)

        total_gt += len(gt_boxes)
        total_dets += len(det_boxes)
        if not det_boxes:
            frames_with_no_det += 1

        # Evaluate at each IoU threshold
        for t in iou_thresholds:
            tp, fp, fn, match_ious = match_detections_to_gt(det_boxes, det_confs, gt_boxes, t)
            stats[t]["tp"] += tp
            stats[t]["fp"] += fp
            stats[t]["fn"] += fn

        # Collect detailed IoU info at threshold=0.01 (basically any overlap)
        _, _, _, all_ious = match_detections_to_gt(det_boxes, det_confs, gt_boxes, 0.01)
        all_match_ious.extend(all_ious)

        # TP/FP confidence breakdown at IoU 0.5
        tp5, _, _, _ = match_detections_to_gt(det_boxes, det_confs, gt_boxes, 0.5)
        # Simple: sort by conf, top tp5 are TP, rest are FP
        sorted_confs = sorted(det_confs, reverse=True)
        for i, c in enumerate(sorted_confs):
            if i < tp5:
                det_conf_on_tp.append(c)
            else:
                det_conf_on_fp.append(c)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # Report
    print(f"\n{'='*60}")
    print(f"IR MODEL STANDALONE DIAGNOSTICS ({n} frames)")
    print(f"{'='*60}")
    print(f"  Total GT boxes:     {total_gt}")
    print(f"  Total detections:   {total_dets}")
    print(f"  Frames with 0 det:  {frames_with_no_det}/{n} ({frames_with_no_det/n*100:.1f}%)")
    print(f"  Avg dets/frame:     {total_dets/n:.1f}")
    print(f"  Avg GT/frame:       {total_gt/n:.1f}")

    print(f"\n  --- Metrics at various IoU thresholds ---")
    print(f"  {'IoU':>6}  {'TP':>6}  {'FP':>6}  {'FN':>6}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
    for t in iou_thresholds:
        s = stats[t]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        f1 = 2*p*r / (p+r) if (p+r) > 0 else 0
        print(f"  {t:>6.1f}  {s['tp']:>6}  {s['fp']:>6}  {s['fn']:>6}  {p:>7.3f}  {r:>7.3f}  {f1:>7.3f}")

    # IoU distribution of matched detections
    if all_match_ious:
        ious = np.array(all_match_ious)
        print(f"\n  --- IoU distribution of matched detections (any overlap) ---")
        print(f"  N matched: {len(ious)}")
        print(f"  Mean IoU:  {ious.mean():.3f}")
        print(f"  Median:    {np.median(ious):.3f}")
        for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                        (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                        (0.8, 0.9), (0.9, 1.01)]:
            n_bin = ((ious >= lo) & (ious < hi)).sum()
            print(f"    [{lo:.1f}, {hi:.1f}): {n_bin:>5} ({n_bin/len(ious)*100:>5.1f}%)")

    # Confidence breakdown
    if det_conf_on_tp:
        tp_c = np.array(det_conf_on_tp)
        print(f"\n  --- Confidence of TP detections (IoU>=0.5) ---")
        print(f"  N={len(tp_c)}, mean={tp_c.mean():.3f}, median={np.median(tp_c):.3f}, "
              f"min={tp_c.min():.3f}, max={tp_c.max():.3f}")

    if det_conf_on_fp:
        fp_c = np.array(det_conf_on_fp)
        print(f"\n  --- Confidence of FP detections (IoU>=0.5) ---")
        print(f"  N={len(fp_c)}, mean={fp_c.mean():.3f}, median={np.median(fp_c):.3f}, "
              f"min={fp_c.min():.3f}, max={fp_c.max():.3f}")

    # Confidence histogram for both
    print(f"\n  --- Confidence distribution (all detections) ---")
    all_confs = det_conf_on_tp + det_conf_on_fp
    if all_confs:
        confs = np.array(all_confs)
        for lo in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            hi = lo + 0.1
            n_bin = ((confs >= lo) & (confs < hi)).sum()
            n_tp = ((tp_c >= lo) & (tp_c < hi)).sum() if det_conf_on_tp else 0
            print(f"    [{lo:.1f}, {hi:.1f}): {n_bin:>5} total, {n_tp:>5} TP")


if __name__ == "__main__":
    main()
