"""
Convert the 3rd Anti-UAV dataset to YOLO image format.

Source: G:\\drone\\3rd_Anti-UAV_train_val\\
Output: G:\\drone\\3rd_AntiUAV_yolo\\

The Anti-UAV dataset has:
  - train/ and validation/ directories
  - Each contains per-sequence folders with numbered frames (000001.jpg, ...)
  - Each sequence has IR_label.json with:
      - "exist": [0/1, ...] per frame (1 = UAV visible)
      - "gt_rect": [[x, y, w, h], ...] per frame (top-left, absolute pixels)

We use the dataset's own train/validation split.
For frames where exist=1, we write YOLO labels (class 0 = drone).
For frames where exist=0, we write empty labels (negative).

Frame sampling (every Nth frame) to reduce redundancy.
"""

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

SOURCE_DIR = Path(r"G:\drone\3rd_Anti-UAV_train_val")
OUTPUT_DIR = Path(r"G:\drone\3rd_AntiUAV_yolo")

SAMPLE_RATE = 5  # Extract every Nth frame


def bbox_to_yolo(x, y, w, h, img_w, img_h):
    """Convert [x, y, w, h] (top-left, absolute) to YOLO [cx, cy, w, h] (center, normalized)."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))
    return cx, cy, nw, nh


def process_split(source_dir, split_name, output_dir, sample_rate, dry_run=False):
    """Process one split (train or validation)."""
    split_dir = source_dir / split_name
    if not split_dir.exists():
        print(f"  [SKIP] {split_dir} not found")
        return {}

    # Map split names for YOLO convention
    split_map = {"validation": "val", "track1_test": "test"}
    yolo_split = split_map.get(split_name, split_name)

    img_out = output_dir / yolo_split / "images"
    lbl_out = output_dir / yolo_split / "labels"

    if not dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    print(f"\n=== {split_name} ({len(sequences)} sequences) ===")

    stats = {
        "sequences": len(sequences),
        "total_frames": 0,
        "extracted_frames": 0,
        "with_bbox": 0,
        "without_bbox": 0,
        "missing_label": 0,
    }

    for seq_idx, seq_dir in enumerate(sequences):
        label_path = seq_dir / "IR_label.json"
        if not label_path.exists():
            print(f"  [{seq_idx+1}/{len(sequences)}] {seq_dir.name}: NO IR_label.json")
            stats["missing_label"] += 1
            continue

        # Try multiple encodings — some files have BOM or non-UTF8
        labels = None
        for enc in ["utf-8-sig", "utf-8", "latin-1"]:
            try:
                with open(label_path, "r", encoding=enc) as f:
                    labels = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        if labels is None:
            print(f"  [{seq_idx+1}/{len(sequences)}] {seq_dir.name}: Can't decode IR_label.json")
            continue

        exist_flags = labels.get("exist", [])
        gt_rects = labels.get("gt_rect", [])

        # Get sorted frame files
        frame_files = sorted(seq_dir.glob("*.jpg"))
        n_frames = len(frame_files)
        stats["total_frames"] += n_frames

        # Get image dimensions from first frame
        if n_frames == 0:
            continue
        sample_img = cv2.imread(str(frame_files[0]))
        if sample_img is None:
            print(f"  [{seq_idx+1}/{len(sequences)}] {seq_dir.name}: Can't read images")
            continue
        img_h, img_w = sample_img.shape[:2]

        seq_extracted = 0
        seq_with_bbox = 0

        for frame_idx, frame_file in enumerate(frame_files):
            if frame_idx % sample_rate != 0:
                continue

            # Unique filename: sequence_name + frame number
            out_name = f"{seq_dir.name}_{frame_file.stem}"
            img_name = f"{out_name}.jpg"
            lbl_name = f"{out_name}.txt"

            if not dry_run:
                # Copy image (faster than re-encoding)
                shutil.copy2(str(frame_file), str(img_out / img_name))

            # Write label
            has_bbox = False
            if frame_idx < len(exist_flags) and frame_idx < len(gt_rects):
                if exist_flags[frame_idx] == 1:
                    rect = gt_rects[frame_idx]
                    if isinstance(rect, list) and len(rect) == 4:
                        x, y, w, h = rect
                        if w > 0 and h > 0:
                            cx, cy, nw, nh = bbox_to_yolo(x, y, w, h, img_w, img_h)
                            if not dry_run:
                                with open(lbl_out / lbl_name, "w") as f:
                                    f.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                            has_bbox = True

            if not has_bbox and not dry_run:
                with open(lbl_out / lbl_name, "w") as f:
                    f.write("")  # empty label = negative

            seq_extracted += 1
            if has_bbox:
                seq_with_bbox += 1

        stats["extracted_frames"] += seq_extracted
        stats["with_bbox"] += seq_with_bbox
        stats["without_bbox"] += (seq_extracted - seq_with_bbox)

        if (seq_idx + 1) % 10 == 0 or (seq_idx + 1) == len(sequences):
            print(f"  [{seq_idx+1}/{len(sequences)}] {seq_dir.name}: "
                  f"{seq_extracted} frames ({seq_with_bbox} pos, {seq_extracted - seq_with_bbox} neg)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert 3rd Anti-UAV to YOLO format")
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                        help="Extract every Nth frame (default: 5)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Sample rate: every {args.sample_rate}th frame")

    all_stats = {}

    for split in ["train", "validation", "track1_test"]:
        stats = process_split(args.source, split, args.output, args.sample_rate, args.dry_run)
        if stats:
            all_stats[split] = stats

    # Write dataset.yaml
    if not args.dry_run:
        yaml_content = f"""# 3rd Anti-UAV Dataset (IR) - converted to YOLO format
# Source: 3rd Anti-UAV Challenge train+val
# Class 0 = drone (UAV)

path: {args.output}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['drone']
"""
        (args.output / "dataset.yaml").write_text(yaml_content, encoding="utf-8")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for split, s in all_stats.items():
        yolo_name = {"validation": "val", "track1_test": "test"}.get(split, split)
        print(f"\n{yolo_name}:")
        print(f"  Sequences: {s['sequences']}")
        print(f"  Total source frames: {s['total_frames']:,}")
        print(f"  Extracted (1/{args.sample_rate}): {s['extracted_frames']:,}")
        print(f"  With bbox (positive): {s['with_bbox']:,}")
        print(f"  Without bbox (negative): {s['without_bbox']:,}")

    total_extracted = sum(s["extracted_frames"] for s in all_stats.values())
    total_pos = sum(s["with_bbox"] for s in all_stats.values())
    print(f"\nTotal: {total_extracted:,} images ({total_pos:,} positive, {total_extracted - total_pos:,} negative)")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")


if __name__ == "__main__":
    main()
