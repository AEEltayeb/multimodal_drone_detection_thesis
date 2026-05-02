"""
Per-source error analysis: which data sources does the model struggle with most?

Runs model on val split, groups results by filename prefix (source),
and ranks by miss rate / false positive rate.

Usage:
    python scripts/eval_per_source.py \
        --weights "ES_Drone_Detection/IR_gold_rgbcfg/IR_gold_rgbcfg/weights/best.pt" \
        --dataset "G:\drone\IR_dset_gold_duplicates_removed"
"""

import argparse
import cv2
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_source(stem: str) -> str:
    """Extract source prefix from filename."""
    # gemini/yt prefixed
    if stem.startswith("gemini_") or stem.startswith("med_gemini"):
        return "gemini_synthetic"
    if stem.startswith("yt_"):
        # yt_VIDEO-ID_sN_FRAME -> yt_VIDEO-ID
        parts = stem.split("_")
        if len(parts) >= 2:
            return f"yt_{parts[1]}"
        return "youtube"

    # dv5_dv4_goldV2_XXXXX
    if stem.startswith("dv5_dv4_"):
        return "dv5_dv4"
    if stem.startswith("dv5_"):
        return "dv5"

    # CST patterns: cst_SCENE_FRAME
    m = re.match(r"^(cst_[a-zA-Z]+(?:_\d+)?)", stem)
    if m:
        return m.group(1)

    # Svanstrom: svan_TYPE_FRAME
    m = re.match(r"^(svan_[a-zA-Z]+)", stem)
    if m:
        return m.group(1)

    # FLIR: flir_XXXXX
    if stem.startswith("flir_"):
        return "flir"

    # General: take prefix before last numeric segment
    m = re.match(r"^(.*?)_\d{3,}$", stem)
    if m:
        return m.group(1)

    return stem[:20]


def yolo_to_pixel_box(cx, cy, w, h, img_w, img_h):
    pw, ph = w * img_w, h * img_h
    x1 = cx * img_w - pw / 2
    y1 = cy * img_h - ph / 2
    return (x1, y1, x1 + pw, y1 + ph)


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def compute_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def match_preds(pred_boxes, pred_confs, gt_boxes, iou_thresh=0.5):
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_confs[i], reverse=True)
    tp_flags = [False] * len(pred_boxes)
    matched = set()
    for i in order:
        best_iou, best_j = 0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched:
                continue
            iou = compute_iou(pred_boxes[i], gb)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thresh and best_j >= 0:
            tp_flags[i] = True
            matched.add(best_j)
    return tp_flags, matched


def main():
    parser = argparse.ArgumentParser(description="Per-source error analysis")
    parser.add_argument("--weights", required=True, help="Model weights path")
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Dataset root (with train/val/test splits)")
    parser.add_argument("--split", default="val", help="Split to evaluate")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Confidence threshold (default: 0.30)")
    parser.add_argument("--min-size", type=int, default=0,
                        help="Min GT bbox area in pixels. GTs below this are ignored. "
                             "E.g. --min-size 400 ignores drones < ~20x20px")
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    img_dir = args.dataset / args.split / "images"
    lbl_dir = args.dataset / args.split / "labels"

    if not img_dir.exists():
        print(f"ERROR: {img_dir} not found")
        sys.exit(1)

    image_files = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
    total = len(image_files)
    print(f"Model:   {args.weights}")
    print(f"Dataset: {img_dir}")
    print(f"Images:  {total}")
    print(f"Conf:    {args.conf}")
    if args.min_size > 0:
        side = int(args.min_size ** 0.5)
        print(f"Min GT:  {args.min_size}px² (~{side}x{side}px) — smaller GTs ignored")
    print()

    # Size buckets (pixel area thresholds)
    SIZE_BUCKETS = [
        ("tiny",       0,    400),   # <20x20
        ("small",    400,   2500),   # 20-50px
        ("medium",  2500,  16384),   # 50-128px
        ("large",  16384, 999999),   # >128px
    ]

    def get_size_bucket(box):
        area = box_area(box)
        for name, lo, hi in SIZE_BUCKETS:
            if lo <= area < hi:
                return name, area
        return "large", area

    # Per-source stats
    src_stats = defaultdict(lambda: {
        "images": 0, "pos_images": 0, "neg_images": 0,
        "gt": 0, "tp": 0, "fp": 0, "fn": 0,
        "missed_files": [],
    })

    # Per-size-bucket stats
    size_stats = {name: {"gt": 0, "tp": 0, "fn": 0, "fp": 0} for name, _, _ in SIZE_BUCKETS}
    neg_image_fp = 0  # FPs on images with no GT (true negatives that got FP)
    skipped_small = 0

    t0 = time.time()
    for idx, img_file in enumerate(image_files):
        stem = img_file.stem
        source = extract_source(stem)

        # Load GT
        gt_boxes = []
        # Read actual image dimensions for correct coordinate mapping
        img = cv2.imread(str(img_file))
        if img is not None:
            real_h, real_w = img.shape[:2]
        else:
            real_w, real_h = 640, 512  # fallback

        lbl_file = lbl_dir / f"{stem}.txt"
        gt_sizes = []  # parallel list: size bucket for each GT box
        if lbl_file.exists():
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = map(float, parts[1:5])
                        box = yolo_to_pixel_box(cx, cy, w, h, real_w, real_h)
                        area = box_area(box)
                        if area < args.min_size:
                            skipped_small += 1
                            continue
                        gt_boxes.append(box)
                        bucket, _ = get_size_bucket(box)
                        gt_sizes.append(bucket)
                        size_stats[bucket]["gt"] += 1

        is_positive = len(gt_boxes) > 0
        src_stats[source]["images"] += 1
        src_stats[source]["gt"] += len(gt_boxes)
        if is_positive:
            src_stats[source]["pos_images"] += 1
        else:
            src_stats[source]["neg_images"] += 1

        # Run inference
        results = model.predict(
            source=str(img_file), conf=args.conf, iou=0.5,
            imgsz=640, verbose=False, save=False, max_det=300,
        )

        r = results[0]
        pred_boxes, pred_confs = [], []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                pred_boxes.append(tuple(float(v) for v in xyxy[i]))
                pred_confs.append(float(confs[i]))

        flags, matched_gt = match_preds(pred_boxes, pred_confs, gt_boxes)

        tp = sum(flags)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)

        src_stats[source]["tp"] += tp
        src_stats[source]["fp"] += fp
        src_stats[source]["fn"] += fn

        # Per-size TP/FN tracking
        for j, bucket in enumerate(gt_sizes):
            if j in matched_gt:
                size_stats[bucket]["tp"] += 1
            else:
                size_stats[bucket]["fn"] += 1

        # Per-size FP tracking (bucket FPs by prediction size)
        for i, flag in enumerate(flags):
            if not flag:  # false positive
                fp_bucket, _ = get_size_bucket(pred_boxes[i])
                size_stats[fp_bucket]["fp"] += 1
        if not is_positive and len(pred_boxes) > 0:
            neg_image_fp += len(pred_boxes)

        # Track worst misses
        if fn > 0:
            src_stats[source]["missed_files"].append((stem, fn))

        done = idx + 1
        if done % 500 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / elapsed
            print(f"  [{done:>5}/{total}] {rate:.1f} img/s", flush=True)

    # ── Results ──
    print(f"\n{'='*90}")
    print(f"  PER-SOURCE PERFORMANCE (conf={args.conf})")
    print(f"{'='*90}")

    header = (f"{'Source':<25} {'Imgs':>5} {'GT':>5} {'TP':>5} {'FP':>5} "
              f"{'FN':>5} {'P':>7} {'R':>7} {'MissR':>7}")
    print(header)
    print("─" * len(header))

    # Sort by miss rate (1 - recall) descending
    sorted_sources = sorted(
        src_stats.items(),
        key=lambda x: (x[1]["fn"] / max(1, x[1]["gt"])),
        reverse=True,
    )

    for source, s in sorted_sources:
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0
        miss = 1.0 - r if s["gt"] > 0 else 0

        # Skip pure-negative sources for miss rate ranking
        marker = " ◄" if miss > 0.3 and s["gt"] > 10 else ""

        print(f"{source:<25} {s['images']:>5} {s['gt']:>5} {s['tp']:>5} "
              f"{s['fp']:>5} {s['fn']:>5} {p:>7.3f} {r:>7.3f} {miss:>7.3f}{marker}")

    # ── Overall metrics ──
    total_tp = sum(s["tp"] for s in src_stats.values())
    total_fp = sum(s["fp"] for s in src_stats.values())
    total_fn = sum(s["fn"] for s in src_stats.values())
    total_gt = sum(s["gt"] for s in src_stats.values())
    overall_p = total_tp / max(1, total_tp + total_fp)
    overall_r = total_tp / max(1, total_tp + total_fn)
    overall_f1 = 2 * overall_p * overall_r / max(1e-9, overall_p + overall_r)

    print(f"\n{'='*90}")
    print(f"  OVERALL (conf={args.conf})")
    print(f"{'='*90}")
    print(f"  Precision: {overall_p:.4f}  Recall: {overall_r:.4f}  F1: {overall_f1:.4f}")
    print(f"  GT: {total_gt}  TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    if skipped_small > 0:
        print(f"  Skipped (below --min-size {args.min_size}): {skipped_small} GT boxes")

    # ── Per-size breakdown ──
    total_neg_images = sum(s["neg_images"] for s in src_stats.values())
    print(f"\n{'='*90}")
    print(f"  PER-SIZE BREAKDOWN (conf={args.conf})")
    print(f"{'='*90}")
    print(f"  {'Bucket':<12} {'GT':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"  {'─'*70}")

    med_large_tp = med_large_gt = med_large_fp = 0
    for name, lo, hi in SIZE_BUCKETS:
        ss = size_stats[name]
        gt = ss["gt"]
        tp = ss["tp"]
        fp = ss["fp"]
        fn = ss["fn"]
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        side_lo = int(lo ** 0.5)
        side_hi = int(hi ** 0.5) if hi < 999999 else "+"
        print(f"  {name:<12} {gt:>6} {tp:>6} {fp:>6} {fn:>6} {p:>8.1%} {r:>8.1%} {f1:>8.1%}")
        if name in ("medium", "large"):
            med_large_tp += tp
            med_large_gt += gt
            med_large_fp += fp

    med_large_r = med_large_tp / max(1, med_large_gt)
    med_large_p = med_large_tp / max(1, med_large_tp + med_large_fp)
    med_large_f1 = 2 * med_large_p * med_large_r / max(1e-9, med_large_p + med_large_r)
    print(f"  {'─'*70}")
    print(f"  {'medium+large':<12} {med_large_gt:>6} {med_large_tp:>6} {med_large_fp:>6} "
          f"{med_large_gt - med_large_tp:>6} {med_large_p:>8.1%} {med_large_r:>8.1%} {med_large_f1:>8.1%}")

    tn = total_neg_images - min(total_neg_images, neg_image_fp)
    print(f"\n  Negative images: {total_neg_images:,}")
    print(f"  TN (correct negatives): {tn:,}  FP on negatives: {neg_image_fp}")

    # ── Worst sources detail ──
    print(f"\n{'='*90}")
    print(f"  WORST SOURCES (miss rate > 30%, n > 10 GT)")
    print(f"{'='*90}")

    for source, s in sorted_sources:
        if s["gt"] < 10:
            continue
        recall = s["tp"] / max(1, s["gt"])
        if recall >= 0.7:
            continue

        miss = 1 - recall
        print(f"\n  {source}: miss rate = {miss:.1%} ({s['fn']}/{s['gt']} missed)")
        print(f"    Images: {s['images']}  Pos: {s['pos_images']}  Neg: {s['neg_images']}")
        print(f"    FP: {s['fp']}")

        # Show top missed files
        missed = sorted(s["missed_files"], key=lambda x: x[1], reverse=True)
        for fn_file, fn_count in missed[:5]:
            print(f"      missed {fn_count} GT in: {fn_file}")


if __name__ == "__main__":
    main()
