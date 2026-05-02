r"""
dataset_statistics.py — Per-source dataset statistics for dsetV6.

Computes bbox size, brightness, and aspect ratio distributions per source.
No GPU required — reads labels and images directly.

Usage:
    python scripts/analysis/dataset_statistics.py \
        --dataset G:\drone\IR_dsetV6 \
        --split test
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SOURCE_PREFIXES = {
    "dv5_dv4_":  "dsetV4",
    "dv5_auv_":  "Anti-UAV",
    "svan_":     "Svanström",
}

def identify_source(filename):
    for prefix, name in SOURCE_PREFIXES.items():
        if filename.startswith(prefix):
            return name
    return "unknown"


def yolo_to_pixel(cx, cy, w, h, iw, ih):
    pw, ph = w * iw, h * ih
    x1 = cx * iw - pw / 2
    y1 = cy * ih - ph / 2
    return (x1, y1, x1 + pw, y1 + ph, pw, ph)


def print_histogram(values, n_bins=10, label="", width=40):
    """Print a text histogram."""
    if not values:
        print(f"  {label}: no data")
        return
    arr = np.array(values)
    hist, edges = np.histogram(arr, bins=n_bins)
    max_count = max(hist) if max(hist) > 0 else 1
    print(f"  {label} (n={len(arr)}, mean={arr.mean():.1f}, median={np.median(arr):.1f}, "
          f"std={arr.std():.1f}):")
    for i, count in enumerate(hist):
        bar_len = int(count / max_count * width)
        bar = "#" * bar_len
        print(f"    [{edges[i]:7.1f}, {edges[i+1]:7.1f}): {count:>5}  {bar}")


def print_percentiles(values, label=""):
    """Print percentile summary."""
    if not values:
        return
    arr = np.array(values)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(arr, pcts)
    parts = [f"P{p}={v:.1f}" for p, v in zip(pcts, vals)]
    print(f"    {label}: {', '.join(parts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-brightness-samples", type=int, default=500,
                        help="Max images per source for brightness analysis (reading pixels is slow)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    img_dir = dataset_root / args.split / "images"
    lbl_dir = dataset_root / args.split / "labels"

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in img_extensions])

    print(f"\n{'='*70}")
    print(f"  DATASET STATISTICS — {args.dataset}")
    print(f"  Split: {args.split}  |  Total images: {len(all_images)}")
    print(f"{'='*70}")

    # ── Per-source accumulators ──
    stats = defaultdict(lambda: {
        "n_images": 0, "n_positive": 0, "n_negative": 0,
        "n_boxes": 0,
        "widths_px": [], "heights_px": [], "areas_px": [],
        "aspect_ratios": [],
        "img_mean_brightness": [],
        "img_p2": [], "img_p98": [],  # percentile range
        "img_w": [], "img_h": [],  # image dimensions
    })

    t_start = time.time()
    brightness_counts = defaultdict(int)

    for idx, img_file in enumerate(all_images):
        source = identify_source(img_file.stem)
        sd = stats[source]
        sd["n_images"] += 1

        # Load GT labels
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        gt_boxes = []
        if lbl_file.exists():
            with open(lbl_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_boxes.append(tuple(map(float, parts[1:5])))

        if gt_boxes:
            sd["n_positive"] += 1
        else:
            sd["n_negative"] += 1

        # Read image to get dimensions and brightness (sample for speed)
        do_brightness = brightness_counts[source] < args.max_brightness_samples
        if do_brightness:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ih, iw = img.shape[:2]
                sd["img_w"].append(iw)
                sd["img_h"].append(ih)

                mean_brightness = float(np.mean(img))
                sd["img_mean_brightness"].append(mean_brightness)
                sd["img_p2"].append(float(np.percentile(img, 2)))
                sd["img_p98"].append(float(np.percentile(img, 98)))
                brightness_counts[source] += 1

                # Convert GT boxes to pixel coords
                for cx, cy, w, h in gt_boxes:
                    _, _, _, _, pw, ph = yolo_to_pixel(cx, cy, w, h, iw, ih)
                    sd["n_boxes"] += 1
                    sd["widths_px"].append(pw)
                    sd["heights_px"].append(ph)
                    sd["areas_px"].append(pw * ph)
                    sd["aspect_ratios"].append(pw / ph if ph > 0 else 1.0)
            else:
                # Can't read — still count boxes with estimated dimensions
                for cx, cy, w, h in gt_boxes:
                    sd["n_boxes"] += 1
        else:
            # Just read image dimensions quickly
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ih, iw = img.shape[:2]
                for cx, cy, w, h in gt_boxes:
                    _, _, _, _, pw, ph = yolo_to_pixel(cx, cy, w, h, iw, ih)
                    sd["n_boxes"] += 1
                    sd["widths_px"].append(pw)
                    sd["heights_px"].append(ph)
                    sd["areas_px"].append(pw * ph)
                    sd["aspect_ratios"].append(pw / ph if ph > 0 else 1.0)

        if (idx + 1) % 1000 == 0 or (idx + 1) == len(all_images):
            elapsed = time.time() - t_start
            print(f"  [{idx+1}/{len(all_images)}] {(idx+1)/elapsed:.0f} img/s")

    # ── Print results per source ──
    for source in sorted(stats.keys()):
        sd = stats[source]
        neg_pct = sd["n_negative"] / sd["n_images"] * 100 if sd["n_images"] > 0 else 0

        print(f"\n{'='*70}")
        print(f"  {source}")
        print(f"{'='*70}")
        print(f"  Images: {sd['n_images']}  "
              f"(positive: {sd['n_positive']}, negative: {sd['n_negative']} = {neg_pct:.1f}%)")
        print(f"  Total GT boxes: {sd['n_boxes']}")
        if sd["n_positive"] > 0:
            print(f"  Avg boxes/positive image: {sd['n_boxes']/sd['n_positive']:.1f}")

        # Image dimensions
        if sd["img_w"]:
            w_unique = set(zip(sd["img_w"], sd["img_h"]))
            print(f"  Image dimensions: {', '.join(f'{w}×{h}' for w, h in sorted(w_unique))}")

        # Bbox size distribution
        print(f"\n  ── BBox Size Distribution ──")
        if sd["widths_px"]:
            print_histogram(sd["widths_px"], n_bins=12, label="Width (px)")
            print_percentiles(sd["widths_px"], "Width")
            print()
            print_histogram(sd["heights_px"], n_bins=12, label="Height (px)")
            print_percentiles(sd["heights_px"], "Height")
            print()
            print_histogram(sd["areas_px"], n_bins=12, label="Area (px²)")
            print_percentiles(sd["areas_px"], "Area")

            # Count tiny objects
            tiny = sum(1 for a in sd["areas_px"] if a < 100)
            small = sum(1 for a in sd["areas_px"] if 100 <= a < 1024)
            medium = sum(1 for a in sd["areas_px"] if 1024 <= a < 9216)
            large = sum(1 for a in sd["areas_px"] if a >= 9216)
            total = len(sd["areas_px"])
            print(f"\n  Size buckets:")
            print(f"    Tiny   (<100px²):   {tiny:>5} ({tiny/total*100:5.1f}%)")
            print(f"    Small  (100-1K):    {small:>5} ({small/total*100:5.1f}%)")
            print(f"    Medium (1K-9K):     {medium:>5} ({medium/total*100:5.1f}%)")
            print(f"    Large  (>9K):       {large:>5} ({large/total*100:5.1f}%)")

            # Sub-10px filter impact
            under_10w = sum(1 for w in sd["widths_px"] if w < 10)
            under_10h = sum(1 for h in sd["heights_px"] if h < 10)
            under_10_either = sum(1 for w, h in zip(sd["widths_px"], sd["heights_px"]) if w < 10 or h < 10)
            print(f"\n  Filter impact (< 10px):")
            print(f"    Boxes with width < 10px:  {under_10w:>5} ({under_10w/total*100:5.1f}%)")
            print(f"    Boxes with height < 10px: {under_10h:>5} ({under_10h/total*100:5.1f}%)")
            print(f"    Boxes with EITHER < 10px: {under_10_either:>5} ({under_10_either/total*100:5.1f}%)")

        # Aspect ratio
        if sd["aspect_ratios"]:
            print(f"\n  ── Aspect Ratio (W/H) ──")
            print_histogram(sd["aspect_ratios"], n_bins=10, label="Aspect Ratio")

        # Brightness
        if sd["img_mean_brightness"]:
            print(f"\n  ── Brightness Distribution (sampled {len(sd['img_mean_brightness'])} images) ──")
            print_histogram(sd["img_mean_brightness"], n_bins=10, label="Mean pixel intensity")
            print_percentiles(sd["img_mean_brightness"], "Mean brightness")

            # Dynamic range
            ranges = [p98 - p2 for p2, p98 in zip(sd["img_p2"], sd["img_p98"])]
            print()
            print_histogram(ranges, n_bins=10, label="Dynamic range (P98 - P2)")
            print_percentiles(ranges, "Dynamic range")

    # ── Cross-source comparison ──
    print(f"\n\n{'='*70}")
    print(f"  CROSS-SOURCE COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'Source':<15s}  {'Imgs':>6s}  {'Boxes':>6s}  {'Neg%':>5s}  "
          f"{'Med.W':>6s}  {'Med.H':>6s}  {'Med.Area':>9s}  "
          f"{'Brightness':>10s}  {'DynRange':>8s}")
    print(f"  {'-'*95}")

    for source in sorted(stats.keys()):
        sd = stats[source]
        neg_pct = sd["n_negative"] / sd["n_images"] * 100 if sd["n_images"] > 0 else 0
        med_w = np.median(sd["widths_px"]) if sd["widths_px"] else 0
        med_h = np.median(sd["heights_px"]) if sd["heights_px"] else 0
        med_a = np.median(sd["areas_px"]) if sd["areas_px"] else 0
        med_b = np.median(sd["img_mean_brightness"]) if sd["img_mean_brightness"] else 0
        ranges = [p98 - p2 for p2, p98 in zip(sd["img_p2"], sd["img_p98"])]
        med_r = np.median(ranges) if ranges else 0

        print(f"  {source:<15s}  {sd['n_images']:>6}  {sd['n_boxes']:>6}  {neg_pct:>4.1f}%  "
              f"{med_w:>6.1f}  {med_h:>6.1f}  {med_a:>9.1f}  "
              f"{med_b:>10.1f}  {med_r:>8.1f}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
