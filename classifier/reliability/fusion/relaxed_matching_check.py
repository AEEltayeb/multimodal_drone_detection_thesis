"""
relaxed_matching_check.py — Re-evaluate Svanstrom RGB with relaxed matching.

Two matching criteria (either counts as TP):
  1. IoU >= 0.2 (current strict matching)
  2. Intersection / pred_area >= 0.5 (prediction is mostly inside GT)

This catches the case where GT boxes are oversized and predictions are tight.
"""

import json
import numpy as np
from pathlib import Path

INFERENCE_DIR = Path(__file__).resolve().parent.parent.parent / "runs" / "reliability" / "inference"


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def compute_iop(pred, gt):
    """Intersection over Prediction area — is prediction inside GT?"""
    x1 = max(pred[0], gt[0])
    y1 = max(pred[1], gt[1])
    x2 = min(pred[2], gt[2])
    y2 = min(pred[3], gt[3])
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


def match_relaxed(gt_boxes, dets, iou_thresh=0.2, iop_thresh=0.5):
    """Match with IoU OR IoP (intersection over prediction)."""
    n_gt = len(gt_boxes)
    n_det = len(dets)
    gt_matched = [False] * n_gt
    det_matched = [False] * n_det

    if n_gt == 0 or n_det == 0:
        return gt_matched, det_matched

    # Build score matrix (max of IoU, IoP)
    pairs = []
    for gi in range(n_gt):
        for di in range(n_det):
            iou = compute_iou(gt_boxes[gi], dets[di][:4])
            iop = compute_iop(dets[di][:4], gt_boxes[gi])
            score = max(iou, iop)
            if iou >= iou_thresh or iop >= iop_thresh:
                pairs.append((score, gi, di))

    pairs.sort(reverse=True)
    for _, gi, di in pairs:
        if not gt_matched[gi] and not det_matched[di]:
            gt_matched[gi] = True
            det_matched[di] = True

    return gt_matched, det_matched


def match_strict(gt_boxes, dets, iou_thresh=0.2):
    """Original strict IoU matching."""
    n_gt = len(gt_boxes)
    n_det = len(dets)
    gt_matched = [False] * n_gt
    det_matched = [False] * n_det

    if n_gt == 0 or n_det == 0:
        return gt_matched, det_matched

    pairs = []
    for gi in range(n_gt):
        for di in range(n_det):
            iou = compute_iou(gt_boxes[gi], dets[di][:4])
            if iou >= iou_thresh:
                pairs.append((iou, gi, di))

    pairs.sort(reverse=True)
    for _, gi, di in pairs:
        if not gt_matched[gi] and not det_matched[di]:
            gt_matched[gi] = True
            det_matched[di] = True

    return gt_matched, det_matched


def evaluate_dataset(tag, conf_thresh=0.4):
    """Evaluate one dataset with both strict and relaxed matching."""
    json_path = INFERENCE_DIR / f"{tag}.json"
    if not json_path.exists():
        print(f"  [SKIP] {tag}")
        return

    with open(json_path) as f:
        data = json.load(f)

    # Stats for strict and relaxed
    results = {}
    for method_name, match_fn in [("strict_iou", match_strict),
                                    ("relaxed_iou+iop", match_relaxed)]:
        n_gt_total = 0
        n_tp = 0
        n_fp = 0
        n_fn = 0
        n_drone_frames = 0
        n_tp_frames = 0
        n_det_frames = 0

        for stem, frame in data.items():
            dets = [d for d in frame["dets"] if d[4] >= conf_thresh]
            gt_text = frame.get("gt", "")
            gt_boxes = parse_gt(gt_text, frame["w"], frame["h"])

            has_gt = len(gt_boxes) > 0
            has_det = len(dets) > 0

            if has_gt:
                n_drone_frames += 1

            if has_det:
                n_det_frames += 1

            gt_m, det_m = match_fn(gt_boxes, dets)

            frame_tp = sum(1 for m in gt_m if m)
            frame_fp = sum(1 for m in det_m if not m)
            frame_fn = sum(1 for m in gt_m if not m)

            n_gt_total += len(gt_boxes)
            n_tp += frame_tp
            n_fp += frame_fp
            n_fn += frame_fn

            if frame_tp > 0:
                n_tp_frames += 1

        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
        recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        frame_prec = n_tp_frames / n_det_frames if n_det_frames > 0 else 0
        frame_rec = n_tp_frames / n_drone_frames if n_drone_frames > 0 else 0
        frame_f1 = 2 * frame_prec * frame_rec / (frame_prec + frame_rec + 1e-9)

        results[method_name] = {
            "n_gt": n_gt_total, "n_tp": n_tp, "n_fp": n_fp, "n_fn": n_fn,
            "precision": precision, "recall": recall, "f1": f1,
            "frame_prec": frame_prec, "frame_rec": frame_rec, "frame_f1": frame_f1,
            "n_drone_frames": n_drone_frames, "n_det_frames": n_det_frames,
            "n_tp_frames": n_tp_frames,
        }

    return results


def main():
    print("=" * 80)
    print("Strict IoU vs Relaxed (IoU OR IoP) Matching — conf >= 0.4")
    print("=" * 80)

    datasets = [
        "svanstrom_rgb", "svanstrom_ir",
        "antiuav_test_rgb", "antiuav_test_ir",
        "antiuav_val_rgb", "antiuav_val_ir",
    ]

    for tag in datasets:
        results = evaluate_dataset(tag)
        if not results:
            continue

        print(f"\n  {tag}:")
        print(f"    {'Method':<20s} {'Det_P':>7s} {'Det_R':>7s} {'Det_F1':>7s}"
              f"   {'Fr_P':>7s} {'Fr_R':>7s} {'Fr_F1':>7s}"
              f"   {'TP':>7s} {'FP':>7s} {'FN':>7s}")
        print(f"    {'-' * 80}")

        for method, m in results.items():
            print(f"    {method:<20s} {m['precision']:>7.4f} {m['recall']:>7.4f}"
                  f" {m['f1']:>7.4f}   {m['frame_prec']:>7.4f}"
                  f" {m['frame_rec']:>7.4f} {m['frame_f1']:>7.4f}"
                  f"   {m['n_tp']:>7,} {m['n_fp']:>7,} {m['n_fn']:>7,}")

        # Delta
        s = results["strict_iou"]
        r = results["relaxed_iou+iop"]
        print(f"    {'DELTA':<20s} {r['precision']-s['precision']:>+7.4f}"
              f" {r['recall']-s['recall']:>+7.4f}"
              f" {r['f1']-s['f1']:>+7.4f}   {r['frame_prec']-s['frame_prec']:>+7.4f}"
              f" {r['frame_rec']-s['frame_rec']:>+7.4f}"
              f" {r['frame_f1']-s['frame_f1']:>+7.4f}"
              f"   {r['n_tp']-s['n_tp']:>+7,} {r['n_fp']-s['n_fp']:>+7,}"
              f" {r['n_fn']-s['n_fn']:>+7,}")


if __name__ == "__main__":
    main()
