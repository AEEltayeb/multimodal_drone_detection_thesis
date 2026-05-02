"""
build_dsetV4.py — Build IR_dsetV4 from 6 IR-only sources.

Per-source independent 80/10/10 split, then merge.
Video-aware for Gold V2, May22, Bird, DroneDetect IR.
Shuffle for Roboflow, Small Objects.
"""
import shutil
import random
import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

random.seed(42)

OUTPUT = Path("datasets/IR_dsetV4")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ═══════════════════════════════════════════════════════════
# Source definitions
# ═══════════════════════════════════════════════════════════

SOURCES = {
    "goldV2": {
        "prefix": "goldV2_",
        "split_method": "video",
        "label_dirs": [
            "datasets/goldv2_gold_labels_3_10/train",
            "datasets/goldv2_gold_labels_3_10/val",
            "datasets/goldv2_gold_labels_3_10/test",
        ],
        "image_dirs": [
            "datasets/IR_dsetV1_gold_v2/images/train",
            "datasets/IR_dsetV1_gold_v2/images/val",
            "datasets/IR_dsetV1_gold_v2/images/test",
        ],
        "video_fn": lambda stem: stem[:3] if stem[:3].isdigit() else stem.split("_")[0],
    },
    "may22": {
        "prefix": "may22_",
        "split_method": "video",
        "label_dirs": [
            "datasets/may22_gold_labels_3_10/train",
            "datasets/may22_gold_labels_3_10/val",
            "datasets/may22_gold_labels_3_10/test",
        ],
        "image_dirs": [
            "datasets/IR_thermal_may22/images/train",
            "datasets/IR_thermal_may22/images/val",
            "datasets/IR_thermal_may22/images/test",
        ],
        "video_fn": lambda stem: re.sub(r"_f\d+$", "", stem),
    },
    "roboflow": {
        "prefix": "roboflow_",
        "split_method": "shuffle",
        "label_dirs": [
            "G:/drone/Drone Detection in Various Envir.v1i.yolo26/train/labels",
            "G:/drone/Drone Detection in Various Envir.v1i.yolo26/valid/labels",
            "G:/drone/Drone Detection in Various Envir.v1i.yolo26/test/labels",
        ],
        "image_dirs": [
            "G:/drone/Drone Detection in Various Envir.v1i.yolo26/train/images",
            "G:/drone/Drone Detection in Various Envir.v1i.yolo26/valid/images",
            "G:/drone/Drone Detection in Various Envir.v1i.yolo26/test/images",
        ],
    },
    "smallobj": {
        "prefix": "smallobj_",
        "split_method": "shuffle",
        "label_dirs": [
            "datasets/small_objects_remapped/train/labels",
            "datasets/small_objects_remapped/valid/labels",
            "datasets/small_objects_remapped/test/labels",
        ],
        "image_dirs": [
            "datasets/small_objects_remapped/train/images",
            "datasets/small_objects_remapped/valid/images",
            "datasets/small_objects_remapped/test/images",
        ],
    },
    "ddetIR": {
        "prefix": "ddetIR_",
        "split_method": "video",
        "label_dirs": [
            "G:/drone/drone-detection.v1-2024-09-30.yolo26_IR+RGB_NEED_EXTRACTION/train/labels",
            "G:/drone/drone-detection.v1-2024-09-30.yolo26_IR+RGB_NEED_EXTRACTION/valid/labels",
            "G:/drone/drone-detection.v1-2024-09-30.yolo26_IR+RGB_NEED_EXTRACTION/test/labels",
        ],
        "image_dirs": [
            "G:/drone/drone-detection.v1-2024-09-30.yolo26_IR+RGB_NEED_EXTRACTION/train/images",
            "G:/drone/drone-detection.v1-2024-09-30.yolo26_IR+RGB_NEED_EXTRACTION/valid/images",
            "G:/drone/drone-detection.v1-2024-09-30.yolo26_IR+RGB_NEED_EXTRACTION/test/images",
        ],
        "filter_fn": lambda stem: stem[0].isdigit() and int(stem[0]) >= 8,
        "video_fn": lambda stem: re.sub(r"_img\d+.*$", "", stem),
    },
    "bird": {
        "prefix": "bird_",
        "split_method": "video",
        "label_dirs": ["datasets/IR_bird_negatives/labels"],
        "image_dirs": ["datasets/IR_bird_negatives/images"],
        "video_fn": lambda stem: "_".join(stem.replace("bird_", "").split("_")[:3]),
    },
}


def collect_frames(source_name, source_cfg):
    """Collect all (image_path, label_path) for a source."""
    frames = []
    filter_fn = source_cfg.get("filter_fn", lambda s: True)
    
    for lbl_dir_str, img_dir_str in zip(source_cfg["label_dirs"], source_cfg["image_dirs"]):
        lbl_dir = Path(lbl_dir_str)
        img_dir = Path(img_dir_str)
        if not lbl_dir.exists():
            print(f"  WARNING: {lbl_dir} does not exist, skipping")
            continue
        
        for lf in lbl_dir.glob("*.txt"):
            if lf.name == "classes.txt":
                continue
            stem = lf.stem
            if not filter_fn(stem):
                continue
            
            # Find matching image
            img_path = None
            for ext in IMG_EXTS:
                candidate = img_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            
            if img_path:
                frames.append((img_path, lf, stem))
            # If no image found, still include label with None image
            # (will be skipped during copy)
    
    return frames


def split_by_video(frames, video_fn, ratios=(0.80, 0.10, 0.10)):
    """Video-aware split: assign whole videos to splits using deficit-based balancing."""
    # Group by video
    videos = defaultdict(list)
    for frame in frames:
        vid = video_fn(frame[2])  # frame[2] is stem
        videos[vid].append(frame)
    
    # Sort by size (largest first)
    sorted_videos = sorted(videos.items(), key=lambda x: len(x[1]), reverse=True)
    
    target_total = len(frames)
    targets = {
        "train": target_total * ratios[0],
        "val": target_total * ratios[1],
        "test": target_total * ratios[2],
    }
    counts = {"train": 0, "val": 0, "test": 0}
    assignments = {"train": [], "val": [], "test": []}
    
    for vid_name, vid_frames in sorted_videos:
        # Find split with largest deficit
        deficits = {s: targets[s] - counts[s] for s in ["train", "val", "test"]}
        best_split = max(deficits, key=deficits.get)
        
        # Anti-domination: if this video > 30% of target split, sub-split
        if len(vid_frames) > targets[best_split] * 0.30 and len(vid_frames) > 10:
            # Temporal sub-split
            n = len(vid_frames)
            t_end = int(n * ratios[0])
            v_end = t_end + int(n * ratios[1])
            assignments["train"].extend(vid_frames[:t_end])
            assignments["val"].extend(vid_frames[t_end:v_end])
            assignments["test"].extend(vid_frames[v_end:])
            counts["train"] += t_end
            counts["val"] += v_end - t_end
            counts["test"] += n - v_end
        else:
            assignments[best_split].extend(vid_frames)
            counts[best_split] += len(vid_frames)
    
    return assignments


def split_by_shuffle(frames, ratios=(0.80, 0.10, 0.10)):
    """Shuffle split for non-video sources."""
    shuffled = list(frames)
    random.shuffle(shuffled)
    
    n = len(shuffled)
    t_end = int(n * ratios[0])
    v_end = t_end + int(n * ratios[1])
    
    return {
        "train": shuffled[:t_end],
        "val": shuffled[t_end:v_end],
        "test": shuffled[v_end:],
    }


def copy_frame(img_path, lbl_path, prefix, out_split):
    """Copy image and label to output directory with prefix."""
    stem = lbl_path.stem
    new_stem = f"{prefix}{stem}"
    
    # Copy image
    if img_path and img_path.exists():
        img_out = OUTPUT / "images" / out_split / f"{new_stem}{img_path.suffix}"
        shutil.copy2(img_path, img_out)
    
    # Copy label
    lbl_out = OUTPUT / "labels" / out_split / f"{new_stem}.txt"
    shutil.copy2(lbl_path, lbl_out)
    
    return new_stem


# ═══════════════════════════════════════════════════════════
# Main build
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("Building IR_dsetV4")
print("=" * 60)

# Create output directories
for split in ["train", "val", "test"]:
    (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

manifest = {
    "metadata": {
        "created": datetime.now().isoformat(),
        "seed": 42,
        "split_ratios": [0.80, 0.10, 0.10],
        "sources": list(SOURCES.keys()),
    },
    "source_stats": {},
    "split_stats": {"train": {}, "val": {}, "test": {}},
}

total_by_split = {"train": 0, "val": 0, "test": 0}
source_in_split = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

for source_name, cfg in SOURCES.items():
    print(f"\n--- {source_name} (prefix={cfg['prefix']}) ---")
    
    # Collect frames
    frames = collect_frames(source_name, cfg)
    print(f"  Collected: {len(frames)} frames")
    
    if not frames:
        continue
    
    # Split
    if cfg["split_method"] == "video":
        video_fn = cfg["video_fn"]
        assignments = split_by_video(frames, video_fn)
    else:
        assignments = split_by_shuffle(frames)
    
    # Copy files
    for split_name in ["train", "val", "test"]:
        split_frames = assignments[split_name]
        copied = 0
        for img_path, lbl_path, stem in split_frames:
            copy_frame(img_path, lbl_path, cfg["prefix"], split_name)
            copied += 1
        
        total_by_split[split_name] += copied
        source_in_split[split_name][source_name] = copied
        print(f"  {split_name}: {copied} frames")
    
    manifest["source_stats"][source_name] = {
        "total": len(frames),
        "train": len(assignments["train"]),
        "val": len(assignments["val"]),
        "test": len(assignments["test"]),
    }

# ═══════════════════════════════════════════════════════════
# Summary & Verification
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

grand_total = sum(total_by_split.values())
for split in ["train", "val", "test"]:
    pct = 100 * total_by_split[split] / grand_total if grand_total else 0
    print(f"  {split}: {total_by_split[split]} frames ({pct:.1f}%)")

print(f"\n  TOTAL: {grand_total} frames")

# Source proportions per split
print("\n" + "=" * 60)
print("SOURCE PROPORTIONS PER SPLIT")
print("=" * 60)

for split in ["train", "val", "test"]:
    print(f"\n  {split} ({total_by_split[split]} frames):")
    for src in SOURCES:
        n = source_in_split[split].get(src, 0)
        pct = 100 * n / total_by_split[split] if total_by_split[split] else 0
        global_pct = 100 * sum(source_in_split[s].get(src, 0) for s in ["train", "val", "test"]) / grand_total if grand_total else 0
        drift = pct - global_pct
        flag = " ⚠️" if abs(drift) > 5 else ""
        print(f"    {src:>12}: {n:>6} ({pct:>5.1f}%, drift={drift:+.1f}%){flag}")

# Save manifest
manifest_path = OUTPUT / "split_manifest.json"
for split in ["train", "val", "test"]:
    manifest["split_stats"][split] = dict(source_in_split[split])
    manifest["split_stats"][split]["_total"] = total_by_split[split]

with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n  Manifest saved to: {manifest_path}")
print("\nDone!")
