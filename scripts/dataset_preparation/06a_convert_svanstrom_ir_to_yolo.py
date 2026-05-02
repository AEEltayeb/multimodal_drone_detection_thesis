"""
Convert the Video IR dataset (MATLAB .mat labels + .mp4 videos) to YOLO image format.

Source: G:\\drone\\Drone-detection-dataset-must-cite\\...\\Data\\Video_IR\\
Output: G:\\drone\\IR_video_ir_dataset\\

Classes:
  - IR_DRONE_* → class 0 (drone) with bounding box labels
  - IR_AIRPLANE_*, IR_BIRD_*, IR_HELICOPTER_* → negative samples (empty labels)

Split: 80/10/10 train/val/test at VIDEO level (no frame leakage).
"""

import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

# ── CONFIG ──────────────────────────────────────────────────────────
SOURCE_DIR = Path(r"G:\drone\Drone-detection-dataset-must-cite\Drone-detection-dataset-master\Data\Video_IR")
OUTPUT_DIR = Path(r"G:\drone\IR_video_ir_dataset")

CATEGORY_MAP = {
    "DRONE": "positive",
    "AIRPLANE": "negative",
    "BIRD": "negative",
    "HELICOPTER": "negative",
}

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
MAX_VIDEO_SHARE = 0.05  # No single video > 5% of a split's frames

SEED = 42


# ── BBOX EXTRACTION ────────────────────────────────────────────────
def extract_bboxes_from_mat(mat_path: Path, img_w: int = 320, img_h: int = 256) -> list:
    """
    Extract per-frame bounding boxes from a MATLAB groundTruth .mat file.
    
    The .mat files contain MatlabOpaque groundTruth objects. The actual bbox data
    is stored in __function_workspace__ as 4-double arrays [x, y, w, h]
    at regular intervals.
    
    Returns: list of (x, y, w, h) tuples or None for frames without valid bbox.
    """
    d = sio.loadmat(str(mat_path), squeeze_me=False)
    
    if '__function_workspace__' not in d:
        print(f"  WARNING: No __function_workspace__ in {mat_path.name}")
        return []
    
    fw = d['__function_workspace__']
    raw = bytes(fw.flat)
    
    # Search for 4-double arrays that look like bboxes
    bboxes = []
    for i in range(0, len(raw) - 40, 4):
        dtype = struct.unpack_from('<I', raw, i)[0]
        if dtype == 9:  # miDOUBLE
            nbytes = struct.unpack_from('<I', raw, i + 4)[0]
            if nbytes == 32:  # exactly 4 doubles
                vals = struct.unpack_from('<4d', raw, i + 8)
                x, y, w, h = vals
                if 0 < w < img_w and 0 < h < img_h and 0 <= x < img_w and 0 <= y < img_h:
                    bboxes.append(vals)
    
    return bboxes


def bbox_to_yolo(x, y, w, h, img_w, img_h):
    """Convert [x, y, w, h] (top-left, absolute) to YOLO [cx, cy, w, h] (center, normalized)."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    nw = max(0, min(1, nw))
    nh = max(0, min(1, nh))
    return cx, cy, nw, nh


# ── VIDEO DISCOVERY ────────────────────────────────────────────────
def discover_videos(source_dir: Path) -> dict:
    """Discover all video+label pairs, grouped by category."""
    videos = defaultdict(list)
    
    for mp4 in sorted(source_dir.glob("*.mp4")):
        name = mp4.stem  # e.g., "IR_DRONE_001"
        mat = source_dir / f"{name}_LABELS.mat"
        
        if not mat.exists():
            print(f"  WARNING: No label file for {name}")
            continue
        
        # Parse category from filename: IR_CATEGORY_NNN
        parts = name.split("_")
        category = parts[1]  # DRONE, AIRPLANE, BIRD, HELICOPTER
        
        if category not in CATEGORY_MAP:
            print(f"  WARNING: Unknown category '{category}' in {name}")
            continue
        
        # Get frame count
        cap = cv2.VideoCapture(str(mp4))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        videos[category].append({
            "name": name,
            "mp4": mp4,
            "mat": mat,
            "category": category,
            "role": CATEGORY_MAP[category],
            "n_frames": n_frames,
        })
    
    return dict(videos)


# ── VIDEO-LEVEL SPLITTING ──────────────────────────────────────────
def split_videos(videos_by_cat: dict, seed: int = SEED) -> dict:
    """
    Split videos into train/val/test at the video level.
    Stratified by category so each split has proportional representation.
    """
    rng = np.random.RandomState(seed)
    splits = {"train": [], "val": [], "test": []}
    
    for cat, vids in sorted(videos_by_cat.items()):
        n = len(vids)
        indices = list(range(n))
        rng.shuffle(indices)
        
        n_val = max(1, round(n * SPLIT_RATIOS["val"]))
        n_test = max(1, round(n * SPLIT_RATIOS["test"]))
        n_train = n - n_val - n_test
        
        for i in indices[:n_train]:
            splits["train"].append(vids[i])
        for i in indices[n_train:n_train + n_val]:
            splits["val"].append(vids[i])
        for i in indices[n_train + n_val:]:
            splits["test"].append(vids[i])
    
    return splits


# ── ANTI-DOMINATION CHECK ──────────────────────────────────────────
def compute_sample_rates(splits: dict, base_pos_rate: int, base_neg_rate: int) -> dict:
    """
    Compute per-video sample rates. If a video would contribute >5% of split
    frames, increase its sample rate to reduce its contribution.
    """
    rates = {}
    
    for split_name, vids in splits.items():
        # First pass: compute total frames at base rates
        total_frames = 0
        for v in vids:
            rate = base_pos_rate if v["role"] == "positive" else base_neg_rate
            total_frames += v["n_frames"] // rate
        
        # Second pass: check domination
        for v in vids:
            rate = base_pos_rate if v["role"] == "positive" else base_neg_rate
            my_frames = v["n_frames"] // rate
            share = my_frames / max(total_frames, 1)
            
            if share > MAX_VIDEO_SHARE:
                # Increase rate to bring share below threshold
                target_frames = int(total_frames * MAX_VIDEO_SHARE)
                rate = max(rate, v["n_frames"] // max(target_frames, 1))
            
            rates[v["name"]] = rate
    
    return rates


# ── MAIN CONVERSION ────────────────────────────────────────────────
def convert_video(video_info: dict, split_name: str, output_dir: Path, 
                  sample_rate: int) -> dict:
    """Convert a single video to YOLO format images + labels."""
    name = video_info["name"]
    category = video_info["category"]
    role = video_info["role"]
    
    img_dir = output_dir / split_name / "images"
    lbl_dir = output_dir / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_info["mp4"]))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Extract bboxes for positive videos
    bboxes = []
    if role == "positive":
        bboxes = extract_bboxes_from_mat(video_info["mat"], img_w, img_h)
    
    stats = {"name": name, "category": category, "role": role, 
             "total_frames": n_frames, "extracted": 0, "with_bbox": 0}
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_rate == 0:
            img_name = f"{name}_f{frame_idx:06d}.jpg"
            lbl_name = f"{name}_f{frame_idx:06d}.txt"
            
            cv2.imwrite(str(img_dir / img_name), frame)
            
            # Write label
            lbl_path = lbl_dir / lbl_name
            if role == "positive" and frame_idx < len(bboxes):
                x, y, w, h = bboxes[frame_idx]
                if w > 0 and h > 0:
                    cx, cy, nw, nh = bbox_to_yolo(x, y, w, h, img_w, img_h)
                    lbl_path.write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    stats["with_bbox"] += 1
                else:
                    lbl_path.write_text("")  # empty label for this frame
            else:
                lbl_path.write_text("")  # negative sample
            
            stats["extracted"] += 1
        
        frame_idx += 1
    
    cap.release()
    return stats


def write_dataset_yaml(output_dir: Path):
    """Write YOLO dataset.yaml."""
    yaml_content = f"""# IR Video Dataset — auto-generated
# Source: Drone-detection-dataset (multi-class IR videos)
# Only drone class retained (class 0), others are negatives

path: {output_dir}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['drone']
"""
    (output_dir / "dataset.yaml").write_text(yaml_content)


def main():
    parser = argparse.ArgumentParser(description="Convert Video IR dataset to YOLO format")
    parser.add_argument("--source", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--pos-sample-rate", type=int, default=3, 
                        help="Extract every Nth frame from drone videos (default: 3)")
    parser.add_argument("--neg-sample-rate", type=int, default=10,
                        help="Extract every Nth frame from non-drone videos (default: 10)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only compute split and stats, don't extract frames")
    args = parser.parse_args()
    
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Sample rates: positive={args.pos_sample_rate}, negative={args.neg_sample_rate}")
    print()
    
    # Step 1: Discover videos
    print("=== Discovering videos ===")
    videos_by_cat = discover_videos(args.source)
    for cat, vids in sorted(videos_by_cat.items()):
        total_frames = sum(v["n_frames"] for v in vids)
        print(f"  {cat}: {len(vids)} videos, {total_frames:,} total frames")
    
    # Step 2: Split videos
    print("\n=== Splitting videos (80/10/10, stratified) ===")
    splits = split_videos(videos_by_cat, args.seed)
    for split_name, vids in splits.items():
        cats = defaultdict(int)
        for v in vids:
            cats[v["category"]] += 1
        cat_str = ", ".join(f"{c}={n}" for c, n in sorted(cats.items()))
        print(f"  {split_name}: {len(vids)} videos ({cat_str})")
    
    # Step 3: Compute sample rates (anti-domination)
    print("\n=== Computing sample rates ===")
    rates = compute_sample_rates(splits, args.pos_sample_rate, args.neg_sample_rate)
    
    # Preview frame counts
    for split_name, vids in splits.items():
        total = sum(v["n_frames"] // rates[v["name"]] for v in vids)
        pos = sum(v["n_frames"] // rates[v["name"]] for v in vids if v["role"] == "positive")
        neg = total - pos
        print(f"  {split_name}: ~{total:,} frames ({pos:,} positive, {neg:,} negative)")
    
    # Check for leakage
    all_names = set()
    for split_name, vids in splits.items():
        names = {v["name"] for v in vids}
        overlap = all_names & names
        if overlap:
            print(f"  ⚠️  LEAKAGE in {split_name}: {overlap}")
            sys.exit(1)
        all_names |= names
    print("  ✅ No video leakage detected")
    
    if args.dry_run:
        print("\n[DRY RUN] Stopping here.")
        return
    
    # Step 4: Convert
    print(f"\n=== Converting to {args.output} ===")
    args.output.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    manifest = {}
    
    for split_name, vids in splits.items():
        print(f"\n--- {split_name} ({len(vids)} videos) ---")
        for i, v in enumerate(vids):
            rate = rates[v["name"]]
            stats = convert_video(v, split_name, args.output, rate)
            all_stats.append(stats)
            manifest[v["name"]] = {"split": split_name, "category": v["category"], 
                                    "sample_rate": rate, **stats}
            
            print(f"  [{i+1}/{len(vids)}] {v['name']}: {stats['extracted']} frames "
                  f"({stats['with_bbox']} with bbox), rate=1/{rate}")
    
    # Step 5: Write metadata
    write_dataset_yaml(args.output)
    
    # Split manifest
    with open(args.output / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Summary stats
    summary = {
        "splits": {},
        "total_images": sum(s["extracted"] for s in all_stats),
        "total_with_bbox": sum(s["with_bbox"] for s in all_stats),
    }
    for split_name in ["train", "val", "test"]:
        split_stats = [s for s in all_stats if manifest[s["name"]]["split"] == split_name]
        summary["splits"][split_name] = {
            "videos": len(split_stats),
            "images": sum(s["extracted"] for s in split_stats),
            "with_bbox": sum(s["with_bbox"] for s in split_stats),
            "negatives": sum(s["extracted"] - s["with_bbox"] for s in split_stats),
        }
    
    with open(args.output / "dataset_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Done ===")
    print(f"Total images: {summary['total_images']:,}")
    print(f"  With bbox: {summary['total_with_bbox']:,}")
    print(f"  Negatives: {summary['total_images'] - summary['total_with_bbox']:,}")
    for split_name, s in summary["splits"].items():
        print(f"  {split_name}: {s['images']:,} images ({s['with_bbox']:,} pos, {s['negatives']:,} neg)")


if __name__ == "__main__":
    main()
