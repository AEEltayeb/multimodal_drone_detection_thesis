"""
describe_dataset.py — Compute automatic statistics for IR_dsetV3.

Reads images, labels, and split manifest to produce per-source and
combined dataset statistics: polarity distribution, drone size buckets,
image resolution, bbox counts, and negative frame counts.

Usage:
    python scripts/describe_dataset.py --dataset-dir datasets/IR_dsetV3
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Size buckets (bbox area as fraction of image area) ─────────────────────
SIZE_BUCKETS = [
    ("tiny",   0.0,   0.001),   # < 0.1%
    ("small",  0.001, 0.01),    # 0.1% – 1%
    ("medium", 0.01,  0.05),    # 1% – 5%
    ("large",  0.05,  1.0),     # > 5%
]


def classify_size(area_frac: float) -> str:
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= area_frac < hi:
            return name
    return "large"


def parse_label(lbl_path: Path):
    """Parse a YOLO label file. Returns list of (cls, xc, yc, w, h)."""
    boxes = []
    if not lbl_path.exists():
        return boxes
    text = lbl_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                cls = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append((cls, xc, yc, w, h))
            except ValueError:
                pass
    return boxes


def get_source(stem: str) -> str:
    if stem.startswith("goldV2_"):
        return "goldV2"
    elif stem.startswith("may22_"):
        return "may22"
    elif stem.startswith("roboflow_"):
        return "roboflow"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Describe IR_dsetV3 dataset")
    parser.add_argument("--dataset-dir", type=str, default="datasets/IR_dsetV3")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to split_manifest.json (default: <dataset-dir>/split_manifest.json)")
    parser.add_argument("--sample-resolution", type=int, default=10,
                        help="Number of images to sample per source for resolution check")
    args = parser.parse_args()

    dset = Path(args.dataset_dir)
    manifest_path = Path(args.manifest) if args.manifest else dset / "split_manifest.json"

    # ── Load manifest ──
    stem_info = {}  # stem -> {polarity, source, video_id, split}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for vid_id, vid_data in manifest.get("videos", {}).items():
            for frame in vid_data.get("images", []):
                stem = Path(frame["filename"]).stem
                stem_info[stem] = {
                    "polarity": vid_data.get("polarity", "UNKNOWN"),
                    "source": vid_data.get("dataset_source", "unknown"),
                    "video_id": vid_id,
                    "split": vid_data.get("split", "unknown"),
                }

    # ── Collect all images across splits ──
    stats = defaultdict(lambda: {
        "count": 0,
        "resolutions": [],
        "bbox_counts": [],
        "size_buckets": defaultdict(int),
        "aspect_ratios": [],
        "negative_frames": 0,
        "polarity": defaultdict(int),
        "splits": defaultdict(int),
    })

    for split in ["train", "valid", "test"]:
        img_dir = dset / "images" / split
        lbl_dir = dset / "labels" / split
        if not img_dir.exists():
            continue

        images = sorted(img_dir.iterdir())
        for img_path in images:
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                continue

            stem = img_path.stem
            source = get_source(stem)
            info = stem_info.get(stem, {})
            polarity = info.get("polarity", "UNKNOWN")

            s = stats[source]
            s["count"] += 1
            s["splits"][split] += 1
            s["polarity"][polarity] += 1

            # Sample resolution (only first N per source)
            if len(s["resolutions"]) < args.sample_resolution:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    s["resolutions"].append((w, h))

            # Parse labels
            lbl_path = lbl_dir / f"{stem}.txt"
            boxes = parse_label(lbl_path)
            s["bbox_counts"].append(len(boxes))

            if len(boxes) == 0:
                s["negative_frames"] += 1

            # Size + aspect ratio analysis
            # Use first sampled resolution as representative
            if s["resolutions"]:
                img_w, img_h = s["resolutions"][0]
                for cls, xc, yc, bw, bh in boxes:
                    area_frac = bw * bh  # already normalized
                    bucket = classify_size(area_frac)
                    s["size_buckets"][bucket] += 1

                    if bh > 0:
                        s["aspect_ratios"].append(bw / bh)

    # ── Print report ──
    print("=" * 70)
    print("  IR_dsetV3 DATASET CHARACTERISATION")
    print("=" * 70)

    total_frames = sum(s["count"] for s in stats.values())
    total_bboxes = sum(sum(s["bbox_counts"]) for s in stats.values())
    print(f"\n  Total: {total_frames} frames, {total_bboxes} bboxes")
    print()

    for source in ["goldV2", "may22", "roboflow"]:
        s = stats[source]
        if s["count"] == 0:
            continue

        bbox_arr = np.array(s["bbox_counts"])
        ar_arr = np.array(s["aspect_ratios"]) if s["aspect_ratios"] else np.array([0])

        print(f"  {'─' * 60}")
        print(f"  SOURCE: {source}")
        print(f"  {'─' * 60}")
        print(f"    Frames: {s['count']}")
        print(f"    Splits: {dict(s['splits'])}")

        # Resolution
        if s["resolutions"]:
            res_set = set(s["resolutions"])
            res_str = ", ".join(f"{w}×{h}" for w, h in sorted(res_set))
            print(f"    Resolution: {res_str}")

        # Polarity
        pol = dict(s["polarity"])
        print(f"    Polarity: {pol}")
        dominant = max(pol, key=pol.get) if pol else "N/A"
        dom_pct = pol.get(dominant, 0) / s["count"] * 100 if s["count"] else 0
        print(f"    Dominant polarity: {dominant} ({dom_pct:.1f}%)")

        # Bboxes
        print(f"    Bboxes: total={int(bbox_arr.sum())}, "
              f"mean={bbox_arr.mean():.2f}, median={np.median(bbox_arr):.0f}, "
              f"max={bbox_arr.max()}")
        print(f"    Negative frames (0 bboxes): {s['negative_frames']} "
              f"({s['negative_frames']/s['count']*100:.1f}%)")

        # Size buckets
        total_boxes = sum(s["size_buckets"].values())
        print(f"    Drone size distribution ({total_boxes} bboxes):")
        for bucket_name, _, _ in SIZE_BUCKETS:
            count = s["size_buckets"][bucket_name]
            pct = count / total_boxes * 100 if total_boxes else 0
            bar = "█" * int(pct / 2)
            print(f"      {bucket_name:8s}: {count:5d} ({pct:5.1f}%) {bar}")

        # Aspect ratio
        print(f"    Bbox aspect ratio (w/h): mean={ar_arr.mean():.2f}, "
              f"median={np.median(ar_arr):.2f}, std={ar_arr.std():.2f}")
        print()

    # ── Combined size distribution ──
    print(f"  {'─' * 60}")
    print(f"  COMBINED SIZE DISTRIBUTION")
    print(f"  {'─' * 60}")
    combined_buckets = defaultdict(int)
    for s in stats.values():
        for bucket_name, count in s["size_buckets"].items():
            combined_buckets[bucket_name] += count
    combined_total = sum(combined_buckets.values())
    for bucket_name, _, _ in SIZE_BUCKETS:
        count = combined_buckets[bucket_name]
        pct = count / combined_total * 100 if combined_total else 0
        bar = "█" * int(pct / 2)
        print(f"    {bucket_name:8s}: {count:5d} ({pct:5.1f}%) {bar}")

    # ── Save JSON for notebook consumption ──
    output = {}
    for source, s in stats.items():
        output[source] = {
            "count": s["count"],
            "splits": dict(s["splits"]),
            "resolutions": list(set(s["resolutions"])),
            "polarity": dict(s["polarity"]),
            "bbox_mean": float(np.mean(s["bbox_counts"])) if s["bbox_counts"] else 0,
            "bbox_median": float(np.median(s["bbox_counts"])) if s["bbox_counts"] else 0,
            "negative_frames": s["negative_frames"],
            "size_buckets": dict(s["size_buckets"]),
            "aspect_ratio_mean": float(np.mean(s["aspect_ratios"])) if s["aspect_ratios"] else 0,
            "aspect_ratio_std": float(np.std(s["aspect_ratios"])) if s["aspect_ratios"] else 0,
        }

    out_path = dset / "dataset_stats.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Stats saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
