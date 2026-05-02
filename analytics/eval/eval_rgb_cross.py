"""
Cross-evaluate IR-trained models on the RGB drone dataset.
Provides overall metrics + size-bucketed breakdown.
"""
import argparse
import csv
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

# Size buckets (normalized bbox area = w * h)
SIZE_BUCKETS = {
    "tiny":       (0.0,     0.002),    # < 0.2% of image
    "very_small": (0.002,   0.01),     # 0.2% - 1%
    "small":      (0.01,    0.04),     # 1% - 4%
    "medium":     (0.04,    0.15),     # 4% - 15%
    "large":      (0.15,    1.0),      # > 15%
}

IOU_THRESH = 0.5


def parse_yolo_labels(label_path):
    """Parse YOLO label file. Returns list of (cls, cx, cy, w, h)."""
    if not label_path.exists():
        return []
    text = label_path.read_text().strip()
    if not text:
        return []
    boxes = []
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((0, cx, cy, w, h))  # single class
        except ValueError:
            continue
    return boxes


def get_size_bucket(w, h):
    """Get size bucket name from normalized w, h."""
    area = w * h
    for name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= area < hi:
            return name
    return "large"


def compute_iou(box1, box2):
    """Compute IoU between two (cls, cx, cy, w, h) boxes."""
    _, cx1, cy1, w1, h1 = box1
    _, cx2, cy2, w2, h2 = box2

    x1_min, x1_max = cx1 - w1/2, cx1 + w1/2
    y1_min, y1_max = cy1 - h1/2, cy1 + h1/2
    x2_min, x2_max = cx2 - w2/2, cx2 + w2/2
    y2_min, y2_max = cy2 - h2/2, cy2 + h2/2

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_x * inter_y

    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def match_predictions(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Match predictions to GT. Returns (tp_list, fp_list, fn_list).
    
    Each TP: (gt_box, pred_box, iou, conf)
    Each FP: (pred_box, conf)
    Each FN: (gt_box,)
    """
    if not pred_boxes:
        return [], [], [(gt,) for gt in gt_boxes]
    if not gt_boxes:
        return [], [(p, p[5]) for p in pred_boxes], []

    # Sort predictions by confidence (descending)
    preds_sorted = sorted(pred_boxes, key=lambda x: x[5], reverse=True)
    matched_gt = set()
    tp, fp = [], []

    for pred in preds_sorted:
        best_iou = 0
        best_gt_idx = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(gt, pred[:5])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi
        
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp.append((gt_boxes[best_gt_idx], pred, best_iou, pred[5]))
            matched_gt.add(best_gt_idx)
        else:
            fp.append((pred, pred[5]))

    fn = [(gt_boxes[i],) for i in range(len(gt_boxes)) if i not in matched_gt]
    return tp, fp, fn


def evaluate_model(model_path, dataset_root, split="test", conf=0.25):
    """Run model on dataset and return per-image results."""
    from ultralytics import YOLO
    
    model = YOLO(str(model_path))
    print(f"  Model loaded: {Path(model_path).name}")
    
    # Find images
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split
    if not img_dir.exists():
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
    
    if not img_dir.exists():
        print(f"  ERROR: No images at {img_dir}")
        return []
    
    images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    print(f"  Found {len(images)} images in {split} split")
    
    results_list = []
    t0 = time.time()
    
    for idx, img_path in enumerate(images):
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    [{idx+1}/{len(images)}] {elapsed:.0f}s")
        
        # GT labels
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        gt_boxes = parse_yolo_labels(lbl_path)
        
        # Model predictions
        try:
            preds = model(str(img_path), conf=conf, verbose=False, device="0")
        except Exception:
            preds = model(str(img_path), conf=conf, verbose=False)
        
        pred_boxes = []
        for r in preds:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                xywhn = box.xywhn[0].cpu().numpy()
                c = float(box.conf[0].cpu())
                pred_boxes.append((0, float(xywhn[0]), float(xywhn[1]),
                                   float(xywhn[2]), float(xywhn[3]), c))
        
        tp, fp, fn = match_predictions(gt_boxes, pred_boxes, IOU_THRESH)
        
        results_list.append({
            "image": img_path.name,
            "gt_boxes": gt_boxes,
            "pred_boxes": pred_boxes,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
    
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return results_list


def compute_metrics(results_list):
    """Compute overall and per-size-bucket metrics."""
    # Overall
    total_tp = sum(len(r["tp"]) for r in results_list)
    total_fp = sum(len(r["fp"]) for r in results_list)
    total_fn = sum(len(r["fn"]) for r in results_list)
    total_gt = total_tp + total_fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    overall = {
        "gt": total_gt,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    # Per size bucket
    bucket_stats = {}
    for bname in SIZE_BUCKETS:
        bucket_stats[bname] = {"gt": 0, "tp": 0, "fn": 0}
    
    for r in results_list:
        # Count TPs by GT box size
        for gt, pred, iou, conf in r["tp"]:
            bucket = get_size_bucket(gt[3], gt[4])
            bucket_stats[bucket]["gt"] += 1
            bucket_stats[bucket]["tp"] += 1
        
        # Count FNs by GT box size
        for (gt,) in r["fn"]:
            bucket = get_size_bucket(gt[3], gt[4])
            bucket_stats[bucket]["gt"] += 1
            bucket_stats[bucket]["fn"] += 1
    
    for bname, stats in bucket_stats.items():
        gt = stats["gt"]
        tp = stats["tp"]
        stats["recall"] = tp / gt if gt > 0 else 0
    
    return overall, bucket_stats


def print_results(model_name, overall, bucket_stats):
    """Pretty print results."""
    print(f"\n{'='*65}")
    print(f"  {model_name}")
    print(f"{'='*65}")
    print(f"  GT: {overall['gt']:,}  |  TP: {overall['tp']:,}  |  "
          f"FP: {overall['fp']:,}  |  FN: {overall['fn']:,}")
    print(f"  Precision: {overall['precision']:.4f}  |  "
          f"Recall: {overall['recall']:.4f}  |  F1: {overall['f1']:.4f}")
    
    print(f"\n  {'Size':<14} {'GT':>6} {'TP':>6} {'FN':>6} {'Recall':>8}")
    print(f"  {'-'*42}")
    for bname in SIZE_BUCKETS:
        s = bucket_stats[bname]
        rec = f"{s['recall']:.4f}" if s['gt'] > 0 else "  n/a"
        print(f"  {bname:<14} {s['gt']:>6} {s['tp']:>6} {s['fn']:>6} {rec:>8}")


def main():
    parser = argparse.ArgumentParser(description="Cross-evaluate IR models on RGB dataset")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="RGB dataset root (YOLO format)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model paths (best.pt files)")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Model display names (same order as --models)")
    parser.add_argument("--split", default="test",
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    
    names = args.names or [Path(m).parent.parent.name for m in args.models]
    
    print(f"Dataset: {args.dataset}")
    print(f"Split:   {args.split}")
    print(f"Models:  {len(args.models)}")
    print()
    
    all_results = {}
    for model_path, name in zip(args.models, names):
        print(f"\nEvaluating: {name}")
        print(f"  Path: {model_path}")
        results = evaluate_model(model_path, args.dataset, args.split, args.conf)
        overall, bucket_stats = compute_metrics(results)
        print_results(name, overall, bucket_stats)
        all_results[name] = (overall, bucket_stats)
    
    # Comparison table
    if len(all_results) > 1:
        print(f"\n\n{'='*65}")
        print(f"  COMPARISON TABLE")
        print(f"{'='*65}")
        header = f"  {'Metric':<14}"
        for name in all_results:
            header += f" {name:>18}"
        print(header)
        print(f"  {'-'*14}" + f" {'-'*18}" * len(all_results))
        
        for metric in ["precision", "recall", "f1"]:
            row = f"  {metric:<14}"
            for name, (overall, _) in all_results.items():
                row += f" {overall[metric]:>18.4f}"
            print(row)
        
        print()
        for bname in SIZE_BUCKETS:
            row = f"  R@{bname:<11}"
            for name, (_, bucket_stats) in all_results.items():
                s = bucket_stats[bname]
                val = f"{s['recall']:.4f}" if s['gt'] > 0 else "n/a"
                row += f" {val:>18}"
            print(row)


if __name__ == "__main__":
    main()
