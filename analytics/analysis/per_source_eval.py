r"""
per_source_eval.py — Per-source error analysis for merged IR datasets.

Runs the trained model on a test split, then groups results by source
(identified by filename prefix) and computes P/R/F1 per source.

Usage:
    python scripts/analysis/per_source_eval.py \
        --weights models/ir/IR_dsetV6_188ep/best.pt \
        --dataset G:\drone\IR_dsetV6 \
        --split test \
        --threshold 0.33

    # With per-object feature CSV export (for sample selection pipeline):
    python scripts/analysis/per_source_eval.py \
        --weights models/ir/IR_dsetV6_188ep/best.pt \
        --dataset G:\drone\CST-AntiUAV_YOLO \
        --split train \
        --threshold 0.33 \
        --feature-csv error_features.csv

Output:
    - Per-source P/R/F1/mAP breakdown (printed)
    - Failure analysis: top missed GT boxes and top false positives per source
    - Saves results to runs/IR_FT_dsetV6_aug1_s0/per_source_analysis.json
    - (Optional) Per-object feature CSV for sample selection pipeline
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ── Source identification by filename prefix ──

SOURCE_PREFIXES = {
    "dv5_dv4_":  "dsetV4 (6 original sources)",
    "dv5_auv_":  "Anti-UAV",
    "svan_":     "Svanström",
}


def identify_source(filename: str) -> str:
    """Identify the source dataset from a filename prefix."""
    for prefix, source_name in SOURCE_PREFIXES.items():
        if filename.startswith(prefix):
            return source_name
    return "unknown"


# ── IoU and matching (reused from eval.py) ──

def compute_iou(box_a, box_b):
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(pred_boxes, pred_confs, gt_boxes, iou_thresh=0.5):
    """One-to-one greedy matching. Returns tp_flags, matched_gt_indices."""
    if not pred_boxes or not gt_boxes:
        return [False] * len(pred_boxes), set()

    order = sorted(range(len(pred_boxes)), key=lambda i: pred_confs[i], reverse=True)
    tp_flags = [False] * len(pred_boxes)
    matched_gt = set()

    for pred_idx in order:
        best_iou, best_gt = 0.0, -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = compute_iou(pred_boxes[pred_idx], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_iou >= iou_thresh and best_gt >= 0:
            tp_flags[pred_idx] = True
            matched_gt.add(best_gt)

    return tp_flags, matched_gt


def yolo_to_pixel(cx, cy, w, h, img_w, img_h):
    pw, ph = w * img_w, h * img_h
    x1 = cx * img_w - pw / 2
    y1 = cy * img_h - ph / 2
    return (x1, y1, x1 + pw, y1 + ph)


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def classify_size(area):
    if area < 1024:
        return "tiny"
    elif area < 9216:
        return "medium"
    else:
        return "large"


# ── Feature extraction for sample selection pipeline ──

def compute_local_contrast(img_gray, bbox_xyxy, margin_factor=1.0):
    """Target-to-background contrast: (target_mean - bg_mean) / (bg_std + eps).

    Reuses the approach from dataset_deep_diagnostic.py.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    ih, iw = img_gray.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(iw, x2), min(ih, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    target = img_gray[y1:y2, x1:x2].astype(np.float32)
    target_mean = float(target.mean())

    pw, ph = x2 - x1, y2 - y1
    mx = int(pw * margin_factor)
    my = int(ph * margin_factor)
    bx1 = max(0, x1 - mx)
    by1 = max(0, y1 - my)
    bx2 = min(iw, x2 + mx)
    by2 = min(ih, y2 + my)

    bg = img_gray[by1:by2, bx1:bx2].astype(np.float32)
    bg_mean = float(bg.mean())
    bg_std = float(bg.std())

    if bg_std < 1.0:
        return 0.0
    return (target_mean - bg_mean) / bg_std


def extract_object_features(img_gray, bbox_xyxy, img_w, img_h):
    """Extract per-object features for error analysis / sample selection."""
    x1, y1, x2, y2 = bbox_xyxy
    pw = x2 - x1
    ph = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    area = pw * ph
    img_area = img_w * img_h

    contrast = compute_local_contrast(img_gray, bbox_xyxy)
    img_mean = float(img_gray.mean())
    img_std = float(img_gray.std())
    p2 = float(np.percentile(img_gray, 2))
    p98 = float(np.percentile(img_gray, 98))

    return {
        "bbox_w_px": round(pw, 1),
        "bbox_h_px": round(ph, 1),
        "bbox_area": round(area, 1),
        "area_fraction": round(area / img_area, 6) if img_area > 0 else 0.0,
        "aspect_ratio": round(pw / ph, 3) if ph > 0 else 1.0,
        "pos_x": round(cx / img_w, 4) if img_w > 0 else 0.5,
        "pos_y": round(cy / img_h, 4) if img_h > 0 else 0.5,
        "dist_to_center": round(
            math.sqrt((cx / img_w - 0.5) ** 2 + (cy / img_h - 0.5) ** 2), 4
        ) if img_w > 0 and img_h > 0 else 0.0,
        "local_contrast": round(contrast, 4),
        "img_mean": round(img_mean, 1),
        "img_std": round(img_std, 1),
        "img_dynamic_range": round(p98 - p2, 1),
    }


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Per-source error analysis")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--dataset", required=True, help="Path to dataset root (e.g. G:\\drone\\IR_dsetV6)")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--threshold", type=float, default=0.33, help="Confidence threshold for TP/FP/FN counting")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default=None, help="Output JSON path (default: auto)")
    parser.add_argument("--sample-failures", type=int, default=50,
                        help="Number of worst failures to log per source")
    parser.add_argument("--feature-csv", default=None,
                        help="Path to write per-object feature CSV (for sample selection pipeline)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing feature CSV (skip already-processed images)")
    parser.add_argument("--clahe", action="store_true",
                        help="Apply CLAHE preprocessing before inference (clip=3.0, tile=8x8)")
    parser.add_argument("--clahe-clip", type=float, default=3.0,
                        help="CLAHE clip limit (default: 3.0)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    img_dir = dataset_root / args.split / "images"
    lbl_dir = dataset_root / args.split / "labels"

    if not img_dir.exists():
        print(f"[ERROR] Image directory not found: {img_dir}")
        sys.exit(1)

    # Load model
    from ultralytics import YOLO
    print(f"\n{'='*70}")
    print(f"  PER-SOURCE ERROR ANALYSIS")
    print(f"  Weights:   {args.weights}")
    print(f"  Dataset:   {args.dataset}")
    print(f"  Split:     {args.split}")
    print(f"  Threshold: {args.threshold}")
    if args.feature_csv:
        print(f"  Feature CSV: {args.feature_csv}")
    if args.clahe:
        print(f"  CLAHE preprocessing: ON (clip={args.clahe_clip}, tile=8x8)")
    print(f"{'='*70}\n")

    model = YOLO(args.weights)

    # ── Feature CSV setup ──
    csv_file = None
    csv_writer = None
    done_stems = set()
    FEATURE_COLUMNS = [
        "image_stem", "outcome", "source",
        "bbox_w_px", "bbox_h_px", "bbox_area", "area_fraction",
        "aspect_ratio", "pos_x", "pos_y", "dist_to_center",
        "local_contrast", "img_mean", "img_std", "img_dynamic_range",
        "best_pred_iou", "best_pred_conf", "zero_predictions",
        "size_bucket",
    ]
    if args.feature_csv:
        if args.resume and Path(args.feature_csv).exists():
            # Read existing CSV to find already-processed stems
            with open(args.feature_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    done_stems.add(row["image_stem"])
            print(f"  [RESUME] Found {len(done_stems)} already-processed images in CSV")
            csv_file = open(args.feature_csv, "a", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_file, fieldnames=FEATURE_COLUMNS)
            # No header — appending
        else:
            csv_file = open(args.feature_csv, "w", newline="", encoding="utf-8")
            csv_writer = csv.DictWriter(csv_file, fieldnames=FEATURE_COLUMNS)
            csv_writer.writeheader()

    # Collect all images grouped by source
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    all_images = sorted([
        f for f in img_dir.iterdir()
        if f.suffix.lower() in img_extensions
    ])
    print(f"  Total images: {len(all_images)}")

    # Group by source
    source_images = defaultdict(list)
    for img_file in all_images:
        source = identify_source(img_file.stem)
        source_images[source].append(img_file)

    print(f"  Sources found:")
    for source, imgs in sorted(source_images.items()):
        print(f"    {source}: {len(imgs)} images")

    # Run inference and collect per-source metrics
    print(f"\n  Running inference...")
    t_start = time.time()

    # Per-source accumulators
    source_stats = {}
    for source in source_images:
        source_stats[source] = {
            "total_images": 0,
            "positive_images": 0,
            "negative_images": 0,
            "tp": 0, "fp": 0, "fn": 0,
            "size_tp": defaultdict(int),
            "size_fp": defaultdict(int),
            "size_fn": defaultdict(int),
            "size_gt": defaultdict(int),
            "missed_gt": [],     # (filename, gt_box_area, gt_box)
            "false_positives": [],  # (filename, conf, box_area)
        }

    processed = 0
    skipped_resume = 0
    for img_file in all_images:
        # Skip if already processed (resume mode)
        if img_file.stem in done_stems:
            skipped_resume += 1
            processed += 1
            continue

        source = identify_source(img_file.stem)
        stats = source_stats[source]
        stats["total_images"] += 1

        # Load GT
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        gt_boxes = []
        if lbl_file.exists():
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = map(float, parts[1:5])
                        # We need image dimensions — we'll get them from the result
                        gt_boxes.append((cx, cy, w, h))  # keep normalized for now

        # Apply CLAHE preprocessing if enabled
        inference_source = str(img_file)
        if args.clahe:
            img_raw = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                print(f"WARNING Image Read Error {img_file}")
                processed += 1
                continue
            clahe = cv2.createCLAHE(clipLimit=args.clahe_clip, tileGridSize=(8, 8))
            img_clahe = clahe.apply(img_raw)
            # Convert back to 3-channel for YOLO
            inference_source = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

        # Run inference
        try:
            results = model.predict(
                source=inference_source,
                conf=0.001,
                iou=args.iou_thresh,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                save=False,
                max_det=300,
            )
            result = results[0]
            img_h, img_w = result.orig_shape
        except Exception as e:
            print(f"WARNING Image Read Error {img_file}  ({e})")
            processed += 1
            continue

        # Convert GT to pixel coords
        gt_boxes_px = [yolo_to_pixel(cx, cy, w, h, img_w, img_h) for cx, cy, w, h in gt_boxes]

        if gt_boxes_px:
            stats["positive_images"] += 1
        else:
            stats["negative_images"] += 1

        # Read image for feature extraction (only when writing CSV)
        img_gray = None
        if csv_writer is not None:
            img_gray = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)

        # Filter predictions at threshold
        pred_boxes = []
        pred_confs = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                if float(confs[i]) >= args.threshold:
                    pred_boxes.append(tuple(float(v) for v in xyxy[i]))
                    pred_confs.append(float(confs[i]))

        # Match
        tp_flags, matched_gt = match_predictions(pred_boxes, pred_confs, gt_boxes_px, args.iou_thresh)

        tp = sum(tp_flags)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes_px) - len(matched_gt)

        stats["tp"] += tp
        stats["fp"] += fp
        stats["fn"] += fn

        zero_preds = len(pred_boxes) == 0

        # Track size buckets
        for i, (box, is_tp) in enumerate(zip(pred_boxes, tp_flags)):
            bucket = classify_size(box_area(box))
            if is_tp:
                stats["size_tp"][bucket] += 1
            else:
                stats["size_fp"][bucket] += 1

        for gt_idx, gt_box in enumerate(gt_boxes_px):
            area = box_area(gt_box)
            bucket = classify_size(area)
            stats["size_gt"][bucket] += 1
            if gt_idx not in matched_gt:
                stats["size_fn"][bucket] += 1
                # Track missed GT for failure analysis
                if len(stats["missed_gt"]) < args.sample_failures:
                    stats["missed_gt"].append({
                        "image": img_file.stem,
                        "gt_area": round(area, 1),
                        "gt_size": bucket,
                    })

        # Track false positives for failure analysis
        for i, (box, is_tp) in enumerate(zip(pred_boxes, tp_flags)):
            if not is_tp and len(stats["false_positives"]) < args.sample_failures:
                stats["false_positives"].append({
                    "image": img_file.stem,
                    "confidence": round(pred_confs[i], 4),
                    "box_area": round(box_area(box), 1),
                    "box_size": classify_size(box_area(box)),
                })

        # ── Feature CSV: write per-object rows ──
        if csv_writer is not None and img_gray is not None:
            # GT objects (TP or FN)
            for gt_idx, gt_box in enumerate(gt_boxes_px):
                outcome = "TP" if gt_idx in matched_gt else "FN"
                feats = extract_object_features(img_gray, gt_box, img_w, img_h)

                # Find best overlapping prediction for this GT
                best_iou, best_conf = 0.0, 0.0
                for pi, pb in enumerate(pred_boxes):
                    iou_val = compute_iou(pb, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_conf = pred_confs[pi]

                row = {
                    "image_stem": img_file.stem,
                    "outcome": outcome,
                    "source": source,
                    **feats,
                    "best_pred_iou": round(best_iou, 4),
                    "best_pred_conf": round(best_conf, 4),
                    "zero_predictions": int(zero_preds),
                    "size_bucket": classify_size(box_area(gt_box)),
                }
                csv_writer.writerow(row)

            # FP detections (predictions that didn't match any GT)
            for i, (box, is_tp) in enumerate(zip(pred_boxes, tp_flags)):
                if not is_tp:
                    feats = extract_object_features(img_gray, box, img_w, img_h)
                    row = {
                        "image_stem": img_file.stem,
                        "outcome": "FP",
                        "source": source,
                        **feats,
                        "best_pred_iou": 0.0,
                        "best_pred_conf": round(pred_confs[i], 4),
                        "zero_predictions": 0,
                        "size_bucket": classify_size(box_area(box)),
                    }
                    csv_writer.writerow(row)

            # Negative frames (no GT objects)
            if not gt_boxes_px:
                img_mean = float(img_gray.mean())
                img_std_val = float(img_gray.std())
                p2 = float(np.percentile(img_gray, 2))
                p98 = float(np.percentile(img_gray, 98))

                if pred_boxes:
                    # FP on negative frame — write one row per FP
                    for i, box in enumerate(pred_boxes):
                        feats = extract_object_features(img_gray, box, img_w, img_h)
                        row = {
                            "image_stem": img_file.stem,
                            "outcome": "FP",
                            "source": source,
                            **feats,
                            "best_pred_iou": 0.0,
                            "best_pred_conf": round(pred_confs[i], 4),
                            "zero_predictions": 0,
                            "size_bucket": classify_size(box_area(box)),
                        }
                        csv_writer.writerow(row)
                else:
                    # True negative frame — write one summary row
                    row = {
                        "image_stem": img_file.stem,
                        "outcome": "NEG_FRAME",
                        "source": source,
                        "bbox_w_px": 0, "bbox_h_px": 0,
                        "bbox_area": 0, "area_fraction": 0,
                        "aspect_ratio": 0, "pos_x": 0.5, "pos_y": 0.5,
                        "dist_to_center": 0,
                        "local_contrast": 0,
                        "img_mean": round(img_mean, 1),
                        "img_std": round(img_std_val, 1),
                        "img_dynamic_range": round(p98 - p2, 1),
                        "best_pred_iou": 0, "best_pred_conf": 0,
                        "zero_predictions": 1,
                        "size_bucket": "none",
                    }
                    csv_writer.writerow(row)

        processed += 1
        if processed % 200 == 0 or processed == len(all_images):
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(all_images) - processed) / rate if rate > 0 else 0
            print(f"    [{processed}/{len(all_images)}] {rate:.1f} img/s — ETA {eta:.0f}s")

    total_time = time.time() - t_start
    print(f"\n  Inference complete in {total_time:.1f}s ({len(all_images)/total_time:.1f} img/s)")

    # ── Compute and print per-source metrics ──
    print(f"\n{'='*70}")
    print(f"  PER-SOURCE RESULTS (threshold={args.threshold})")
    print(f"{'='*70}")

    results_json = {
        "threshold": args.threshold,
        "iou_threshold": args.iou_thresh,
        "model": args.weights,
        "dataset": args.dataset,
        "split": args.split,
        "sources": {},
    }

    # Also compute overall totals
    total_tp = total_fp = total_fn = 0

    for source in sorted(source_stats.keys()):
        stats = source_stats[source]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n  ── {source} ({stats['total_images']} imgs, "
              f"{stats['positive_images']} pos, {stats['negative_images']} neg) ──")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f}")
        print(f"  F1:         {f1:.4f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}")

        # Size breakdown
        print(f"  Size breakdown:")
        for bucket in ["tiny", "medium", "large"]:
            b_tp = stats["size_tp"].get(bucket, 0)
            b_fp = stats["size_fp"].get(bucket, 0)
            b_fn = stats["size_fn"].get(bucket, 0)
            b_gt = stats["size_gt"].get(bucket, 0)
            b_p = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
            b_r = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
            print(f"    {bucket:6s}: GT={b_gt:>5}  TP={b_tp:>5}  FP={b_fp:>5}  FN={b_fn:>5}  "
                  f"P={b_p:.3f}  R={b_r:.3f}")

        # Top false positives
        if stats["false_positives"]:
            fps_sorted = sorted(stats["false_positives"], key=lambda x: x["confidence"], reverse=True)
            print(f"  Top {min(5, len(fps_sorted))} false positives (highest confidence):")
            for fp_entry in fps_sorted[:5]:
                print(f"    {fp_entry['image']}  conf={fp_entry['confidence']:.4f}  "
                      f"area={fp_entry['box_area']:.0f}  ({fp_entry['box_size']})")

        # Top missed GTs
        if stats["missed_gt"]:
            print(f"  Top {min(5, len(stats['missed_gt']))} missed GT boxes:")
            for miss in stats["missed_gt"][:5]:
                print(f"    {miss['image']}  area={miss['gt_area']:.0f}  ({miss['gt_size']})")

        # Save to JSON (without defaultdicts)
        results_json["sources"][source] = {
            "total_images": stats["total_images"],
            "positive_images": stats["positive_images"],
            "negative_images": stats["negative_images"],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "size_breakdown": {
                bucket: {
                    "gt": stats["size_gt"].get(bucket, 0),
                    "tp": stats["size_tp"].get(bucket, 0),
                    "fp": stats["size_fp"].get(bucket, 0),
                    "fn": stats["size_fn"].get(bucket, 0),
                }
                for bucket in ["tiny", "medium", "large"]
            },
            "sample_false_positives": stats["false_positives"][:10],
            "sample_missed_gt": stats["missed_gt"][:10],
        }

    # Overall
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

    print(f"\n  ── OVERALL ──")
    print(f"  Precision:  {overall_p:.4f}")
    print(f"  Recall:     {overall_r:.4f}")
    print(f"  F1:         {overall_f1:.4f}")
    print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")

    results_json["overall"] = {
        "precision": round(overall_p, 4),
        "recall": round(overall_r, 4),
        "f1": round(overall_f1, 4),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
    }

    # Save JSON
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path("runs/IR_FT_dsetV6_aug1_s0/per_source_analysis.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)

    # Close feature CSV if open
    if csv_file is not None:
        csv_file.close()
        print(f"  Feature CSV saved to: {args.feature_csv}")
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
