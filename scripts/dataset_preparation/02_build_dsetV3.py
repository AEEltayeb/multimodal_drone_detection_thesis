"""
build_dsetV3.py — Build IR_dsetV3 by merging Gold V2 + May22 + Roboflow.

Implements:
  - Pre-merge integrity checks
  - Video-aware grouping (prefix-based + Roboflow gap-clustering)
  - Normalized thermal polarity detection (white-hot / black-hot / mixed)
  - Temporal sub-splitting for scarce-polarity videos (wide gaps)
  - Stratified allocation by (source, polarity)
  - split_manifest.json with SHA256 fingerprint
  - Full verification output

Usage:
    python scripts/build_dsetV3.py --config configs/build_dsetV3.yaml
"""

import argparse
import cv2
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Frame:
    filename: str
    label_filename: str
    frame_number: Optional[int]  # numeric prefix for Roboflow; None for others
    bbox_count: int
    source_img_path: Path
    source_lbl_path: Optional[Path]

@dataclass
class VideoGroup:
    video_id: str
    source: str            # "goldV2", "may22", "roboflow"
    polarity: str          # "WHITE_HOT", "BLACK_HOT", "MIXED"
    frames: List[Frame] = field(default_factory=list)
    is_sub_split: bool = False
    sub_split_of: Optional[str] = None
    sub_split_range: Optional[Tuple[int, int]] = None

    @property
    def frame_count(self) -> int:
        return len(self.frames)


# ─── 1. Integrity checks ─────────────────────────────────────────────────────

def collect_image_label_pairs(img_dir: Path, lbl_dir: Path, filename_regex: str = None) -> List[Tuple[Path, Optional[Path]]]:
    """Collect (image_path, label_path) pairs, handling multiple layouts.

    Supports:
      - {train,val,test}/images/ + {train,val,test}/labels/  (Roboflow)
      - images/{train,val,test}/ + labels/{train,val,test}/  (Gold V2, May22)
      - Flat directory with images directly
    """
    pairs = []
    split_names = {"train", "val", "test", "valid"}

    # Layout 1 (Roboflow): root has {train,valid,test}/images/ subdirs
    roboflow_splits = [d for d in img_dir.iterdir()
                       if d.is_dir() and d.name in split_names and (d / "images").is_dir()]
    if roboflow_splits:
        for split_dir in sorted(roboflow_splits):
            img_sub = split_dir / "images"
            lbl_sub = split_dir / "labels"
            # Cache label stems for O(1) existence checks without disk I/O
            lbl_stems = {f.stem for f in lbl_sub.iterdir()} if lbl_sub.is_dir() else set()
            for img in sorted(img_sub.iterdir()):
                if img.suffix.lower() in IMG_EXTS:
                    if filename_regex and not re.match(filename_regex, img.name):
                        continue
                    lbl = lbl_sub / f"{img.stem}.txt"
                    pairs.append((img, lbl if img.stem in lbl_stems else None))
        return pairs

    # Layout 2 (Gold V2, May22): img_dir has split subdirs with images directly
    subdirs = [d for d in img_dir.iterdir() if d.is_dir() and d.name in split_names]
    if subdirs:
        for subdir in sorted(subdirs):
            lbl_subdir = lbl_dir / subdir.name
            lbl_stems = {f.stem for f in lbl_subdir.iterdir()} if lbl_subdir.is_dir() else set()
            for img in sorted(subdir.iterdir()):
                if img.suffix.lower() in IMG_EXTS:
                    if filename_regex and not re.match(filename_regex, img.name):
                        continue
                    lbl = lbl_subdir / f"{img.stem}.txt"
                    pairs.append((img, lbl if img.stem in lbl_stems else None))
        return pairs

    # Layout 3: flat directory with images
    lbl_stems = {f.stem for f in lbl_dir.iterdir()} if lbl_dir.is_dir() else set()
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() in IMG_EXTS:
            if filename_regex and not re.match(filename_regex, img.name):
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            pairs.append((img, lbl if img.stem in lbl_stems else None))

    return pairs


def check_integrity(img_dir: Path, lbl_dir: Path, source_name: str, filename_regex: str = None) -> List[Frame]:
    """Validate images/labels and return Frame list. Halts on errors."""
    errors = []
    warnings = []
    frames = []

    pairs = collect_image_label_pairs(img_dir, lbl_dir, filename_regex)

    seen_stems = set()
    total_pairs = len(pairs)
    for idx, (img_path, lbl_path) in enumerate(pairs):
        if (idx + 1) % 500 == 0:
            print(f"    ... checking {idx+1}/{total_pairs}", flush=True)
        stem = img_path.stem
        if stem in seen_stems:
            errors.append(f"Duplicate image: {img_path.name}")
        seen_stems.add(stem)

        bbox_count = 0
        if lbl_path is None:
            warnings.append(f"Missing label for {img_path.name} — treating as negative")
        else:
            content = lbl_path.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                for line_num, line in enumerate(content.splitlines(), 1):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    except ValueError:
                        errors.append(f"{stem}.txt line {line_num}: non-numeric values")
                        continue
                    if w <= 0 or h <= 0:
                        errors.append(f"{stem}.txt line {line_num}: zero/negative box size ({w}, {h})")
                    if xc - w/2 < -0.01 or yc - h/2 < -0.01 or xc + w/2 > 1.01 or yc + h/2 > 1.01:
                        warnings.append(f"{stem}.txt line {line_num}: OOB coords ({xc},{yc},{w},{h}) — will be clipped")
                    bbox_count += 1

        # Extract frame number for Roboflow
        m = re.match(r'^(\d+)_', stem)
        frame_num = int(m.group(1)) if m and source_name == "roboflow" else None

        frames.append(Frame(
            filename=img_path.name,
            label_filename=f"{stem}.txt",
            frame_number=frame_num,
            bbox_count=bbox_count,
            source_img_path=img_path,
            source_lbl_path=lbl_path,
        ))

    print(f"  [{source_name}] {len(frames)} images, {len(errors)} errors, {len(warnings)} warnings")
    for w in warnings[:5]:
        print(f"    ⚠ {w}")
    if len(warnings) > 5:
        print(f"    ... and {len(warnings) - 5} more warnings")

    if errors:
        print(f"\n  ERRORS in {source_name}:")
        for e in errors:
            print(f"    ✗ {e}")
        print(f"\n  [HALT] Fix {len(errors)} error(s) before proceeding.")
        sys.exit(1)

    return frames


# ─── 2. Video grouping ───────────────────────────────────────────────────────

def _extract_goldv2_prefix_and_number(stem: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract (prefix, number) from a Gold V2 filename stem.

    Patterns:
      01079       -> ("num",   1079)
      c100        -> ("c",     100)
      image_100   -> ("image", 100)
      l_42        -> ("l",     42)
    """
    # Purely numeric
    m = re.match(r'^(\d+)$', stem)
    if m:
        return ("num", int(m.group(1)))
    # Prefix + number: c100, image_100, l_42, etc.
    m = re.match(r'^([a-zA-Z]+)_?(\d+)$', stem)
    if m:
        return (m.group(1).lower(), int(m.group(2)))
    return (None, None)


def group_gold_v2(frames: List[Frame], delta_max: int = 10,
                  min_cluster: int = 30) -> Tuple[List[VideoGroup], dict]:
    """Group Gold V2 frames into videos by prefix + gap-clustering.

    Gold V2 has 4 filename patterns (num, c, image, l), each representing
    a separate contiguous video sequence. We gap-cluster each independently.
    """
    prefix_groups = defaultdict(list)  # prefix -> [(number, frame)]
    unknowns = []

    for f in frames:
        prefix, number = _extract_goldv2_prefix_and_number(f.source_img_path.stem)
        if prefix is not None:
            prefix_groups[prefix].append((number, f))
        else:
            unknowns.append(f)

    all_video_clusters = []  # List of (prefix, [frames])

    for prefix, numbered in sorted(prefix_groups.items()):
        numbered.sort(key=lambda x: x[0])

        clusters = []
        current_cluster = []
        for num, frame in numbered:
            if current_cluster and (num - current_cluster[-1][0]) > delta_max:
                clusters.append(current_cluster)
                current_cluster = []
            current_cluster.append((num, frame))
        if current_cluster:
            clusters.append(current_cluster)

        # Micro-cluster merge-back (with distance check)
        merged = []
        for cluster in clusters:
            if (merged and
                    len(cluster) < min_cluster and
                    (cluster[0][0] - merged[-1][-1][0]) <= 5 * delta_max):
                merged[-1].extend(cluster)
            else:
                merged.append(cluster)

        for cluster in merged:
            all_video_clusters.append((prefix, [f for _, f in cluster]))

    # Handle unknowns — treat as one group
    if unknowns:
        all_video_clusters.append(("other", unknowns))

    videos = []
    for i, (prefix, cluster_frames) in enumerate(all_video_clusters):
        videos.append(VideoGroup(
            video_id=f"goldV2_{prefix}_{i:03d}",
            source="goldV2",
            polarity="UNKNOWN",
            frames=cluster_frames,
        ))

    stats = {
        "num_clusters": len(all_video_clusters),
        "avg_frames": sum(v.frame_count for v in videos) / max(len(videos), 1),
        "min_frames": min(v.frame_count for v in videos) if videos else 0,
        "max_frames": max(v.frame_count for v in videos) if videos else 0,
    }
    return videos, stats


def group_may22(frames: List[Frame]) -> List[VideoGroup]:
    """Group May22 frames by video prefix.

    Source filenames look like: 20220312_140053_IR_H264_f000145.jpg
    We extract the date+time prefix as the video ID: 20220312_140053
    Or for UAV files: uav1_..., uav2_...
    """
    groups = defaultdict(list)
    for f in frames:
        stem = f.source_img_path.stem
        # Try date pattern: YYYYMMDD_HHMMSS_...
        m = re.search(r'(\d{8}_\d{6})', stem)
        if m:
            prefix = m.group(1)
        # Try uav pattern
        elif stem.lower().startswith('uav'):
            parts = stem.split('_')
            prefix = parts[0]  # "uav1", "uav2"
        else:
            prefix = "unknown"
        groups[prefix].append(f)

    videos = []
    for prefix, frs in groups.items():
        videos.append(VideoGroup(
            video_id=f"may22_{prefix}",
            source="may22",
            polarity="UNKNOWN",
            frames=sorted(frs, key=lambda x: x.filename),
        ))
    return videos


def group_roboflow(frames: List[Frame], delta_max: int = 10,
                   min_cluster: int = 30) -> Tuple[List[VideoGroup], dict]:
    """Cluster Roboflow frames into virtual videos by frame-number proximity."""
    # Sort by frame number
    numbered = [(f.frame_number, f) for f in frames if f.frame_number is not None]
    unnumbered = [f for f in frames if f.frame_number is None]
    numbered.sort(key=lambda x: x[0])

    clusters = []
    current_cluster = []

    for num, frame in numbered:
        if current_cluster and (num - current_cluster[-1][0]) > delta_max:
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append((num, frame))

    if current_cluster:
        clusters.append(current_cluster)

    # Micro-cluster merge-back (with distance check)
    merged_clusters = []
    for cluster in clusters:
        if (merged_clusters and
                len(cluster) < min_cluster and
                (cluster[0][0] - merged_clusters[-1][-1][0]) <= 5 * delta_max):
            merged_clusters[-1].extend(cluster)
        else:
            merged_clusters.append(cluster)

    # Build VideoGroups
    videos = []
    for i, cluster in enumerate(merged_clusters):
        cluster_frames = [f for _, f in cluster]
        videos.append(VideoGroup(
            video_id=f"roboflow_cluster_{i:03d}",
            source="roboflow",
            polarity="UNKNOWN",
            frames=cluster_frames,
        ))

    # Add unnumbered as a single group
    if unnumbered:
        videos.append(VideoGroup(
            video_id="roboflow_unnumbered",
            source="roboflow",
            polarity="UNKNOWN",
            frames=unnumbered,
        ))

    stats = {
        "num_clusters": len(merged_clusters),
        "avg_frames": sum(len(c) for c in merged_clusters) / max(len(merged_clusters), 1),
        "min_frames": min(len(c) for c in merged_clusters) if merged_clusters else 0,
        "max_frames": max(len(c) for c in merged_clusters) if merged_clusters else 0,
    }

    return videos, stats


# ─── 3. Polarity detection ───────────────────────────────────────────────────

def detect_frame_polarity(img_path: Path, lbl_path: Optional[Path],
                          min_bbox_px: int = 15, margin_px: int = 10) -> str:
    """Classify a single frame as WHITE_HOT, BLACK_HOT, or UNKNOWN."""
    if lbl_path is None or not lbl_path.exists():
        return "NO_BBOX"

    content = lbl_path.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return "NO_BBOX"

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "UNKNOWN"

    h, w = img.shape
    sigma_img = max(float(np.std(img)), 1.0)  # safety floor

    votes = []
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Convert to pixel coords
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_w = x2 - x1
        box_h = y2 - y1

        # Skip tiny boxes
        if box_w < min_bbox_px or box_h < min_bbox_px:
            continue

        I_in = float(np.mean(img[y1:y2, x1:x2]))

        # Dilated margin
        mx1 = max(0, x1 - margin_px)
        my1 = max(0, y1 - margin_px)
        mx2 = min(w, x2 + margin_px)
        my2 = min(h, y2 + margin_px)

        # Margin = dilated region minus inner box
        mask = np.zeros((my2 - my1, mx2 - mx1), dtype=bool)
        mask[:, :] = True
        inner_y1 = y1 - my1
        inner_y2 = y2 - my1
        inner_x1 = x1 - mx1
        inner_x2 = x2 - mx1
        mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

        margin_region = img[my1:my2, mx1:mx2]
        if mask.any():
            I_out = float(np.mean(margin_region[mask]))
        else:
            continue

        C = (I_in - I_out) / sigma_img
        if C > 0.5:
            votes.append("WHITE_HOT")
        elif C < -0.5:
            votes.append("BLACK_HOT")

    if not votes:
        return "UNKNOWN"

    wh = votes.count("WHITE_HOT")
    bh = votes.count("BLACK_HOT")
    if wh > bh:
        return "WHITE_HOT"
    elif bh > wh:
        return "BLACK_HOT"
    return "UNKNOWN"


def classify_video_polarity(video: VideoGroup, min_voting_frames: int = 10) -> str:
    """Classify a video's polarity by frame-level consensus."""
    votes = []
    for frame in video.frames:
        result = detect_frame_polarity(frame.source_img_path, frame.source_lbl_path)
        if result in ("WHITE_HOT", "BLACK_HOT"):
            votes.append(result)

    # Too few voting frames → MIXED
    if len(votes) < min_voting_frames:
        return "MIXED"

    wh = votes.count("WHITE_HOT")
    bh = votes.count("BLACK_HOT")
    total = len(votes)

    if wh / total > 0.6:
        return "WHITE_HOT"
    elif bh / total > 0.6:
        return "BLACK_HOT"
    return "MIXED"


# ─── 4. Scarce-polarity sub-splitting ────────────────────────────────────────

def should_sub_split(video: VideoGroup, all_videos: List[VideoGroup]) -> bool:
    """Check if this video dominates its polarity class or is a massive mega-cluster."""
    # Always split massive clusters to prevent skewing split ratios
    if video.frame_count > 800:
        return True

    # Otherwise, only split if it dominates a scarce polarity class
    same_polarity = [v for v in all_videos
                     if v.polarity == video.polarity and v.polarity != "MIXED"]
    if len(same_polarity) < 3:
        total_frames = sum(v.frame_count for v in same_polarity)
        if total_frames > 0 and video.frame_count / total_frames > 0.5:
            return True
    return False


def temporal_sub_split(video: VideoGroup) -> List[VideoGroup]:
    """Split a video into widely-separated train/val/test segments."""
    frames = sorted(video.frames, key=lambda f: f.filename)
    N = len(frames)

    # Train: 0 → 60%, gap, val: 65% → 75%, gap, test: 85% → 100%
    train_end = math.floor(0.60 * N)
    val_start = math.floor(0.65 * N)
    val_end = math.floor(0.75 * N)
    test_start = math.floor(0.85 * N)

    segments = {
        "train": frames[:train_end],
        "valid": frames[val_start:val_end],
        "test": frames[test_start:],
    }

    sub_videos = []
    for split_name, split_frames in segments.items():
        if not split_frames:
            continue
        sub_videos.append(VideoGroup(
            video_id=f"{video.video_id}_sub_{split_name}",
            source=video.source,
            polarity=video.polarity,
            frames=split_frames,
            is_sub_split=True,
            sub_split_of=video.video_id,
        ))

    discarded = N - sum(len(s) for s in segments.values())
    print(f"    Sub-split {video.video_id} ({N} frames): "
          f"train={len(segments['train'])}, val={len(segments['valid'])}, "
          f"test={len(segments['test'])}, discarded(gaps)={discarded}")

    return sub_videos


# ─── 5. Stratified allocation ────────────────────────────────────────────────

def allocate_videos(videos: List[VideoGroup], seed: int = 42) -> Dict[str, List[VideoGroup]]:
    """Stratified allocation by (source, polarity) with deterministic tie-break."""
    splits = {"train": [], "valid": [], "test": []}
    ratios = {"train": 0.8, "valid": 0.1, "test": 0.1}

    # Identify scarce-polarity videos for sub-splitting
    sub_split_queue = []
    remaining = []
    for v in videos:
        if should_sub_split(v, videos):
            sub_split_queue.append(v)
        else:
            remaining.append(v)

    # Step 1: Sub-split scarce-polarity videos
    if sub_split_queue:
        print(f"\n  Sub-splitting {len(sub_split_queue)} scarce-polarity video(s):")
    for v in sub_split_queue:
        sub_videos = temporal_sub_split(v)
        for sv in sub_videos:
            target_split = sv.video_id.rsplit("_sub_", 1)[-1]
            splits[target_split].append(sv)

    # Step 2: Allocate remaining by (source, polarity) bins
    bins = defaultdict(list)
    for v in remaining:
        bins[(v.source, v.polarity)].append(v)

    for (source, polarity), bin_videos in sorted(bins.items()):
        bin_videos.sort(key=lambda v: v.video_id)
        random.seed(seed)
        random.shuffle(bin_videos)

        total = sum(v.frame_count for v in bin_videos)
        targets = {s: total * ratios[s] for s in splits}
        counts = {s: 0 for s in splits}

        for v in bin_videos:
            distances = {s: targets[s] - counts[s] for s in splits}
            # Largest remainder + deterministic tie-break: train > valid > test
            best = max(splits.keys(),
                       key=lambda s: (distances[s], {"train": 3, "valid": 2, "test": 1}[s]))
            splits[best].append(v)
            counts[best] += v.frame_count

    return splits


# ─── 6. Copy files & generate outputs ────────────────────────────────────────

def _copy_and_clip_label(src: Path, dst: Path) -> None:
    """Copy a YOLO label file, clipping any out-of-bounds coordinates to [0, 1]."""
    lines_out = []
    for line in src.read_text(encoding="utf-8", errors="replace").strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            lines_out.append(line.strip())
            continue
        try:
            cls = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            lines_out.append(line.strip())
            continue

        # Clip to [0, 1]
        x1 = max(0.0, xc - w / 2)
        y1 = max(0.0, yc - h / 2)
        x2 = min(1.0, xc + w / 2)
        y2 = min(1.0, yc + h / 2)
        new_w = x2 - x1
        new_h = y2 - y1
        if new_w <= 0 or new_h <= 0:
            continue  # box clipped to nothing
        new_xc = x1 + new_w / 2
        new_yc = y1 + new_h / 2
        lines_out.append(f"{cls} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")

    dst.write_text("\n".join(lines_out) + "\n" if lines_out else "", encoding="utf-8")


def copy_dataset(splits: Dict[str, List[VideoGroup]], output_dir: Path,
                 source_lbl_dirs: Dict[str, Path]) -> None:
    """Copy all images/labels to output directory structure."""
    for split_name, videos in splits.items():
        img_out = output_dir / "images" / split_name
        lbl_out = output_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for video in videos:
            for frame in video.frames:
                # Determine prefixed filename
                prefix = {"goldV2": "goldV2", "may22": "may22", "roboflow": "roboflow"}[video.source]
                stem = frame.source_img_path.stem
                ext = frame.source_img_path.suffix

                # Only add prefix if not already prefixed
                if not stem.startswith(prefix):
                    new_img_name = f"{prefix}_{stem}{ext}"
                    new_lbl_name = f"{prefix}_{stem}.txt"
                else:
                    new_img_name = f"{stem}{ext}"
                    new_lbl_name = f"{stem}.txt"

                shutil.copy2(str(frame.source_img_path), str(img_out / new_img_name))

                if frame.source_lbl_path and frame.source_lbl_path.exists():
                    # Copy with OOB clipping
                    _copy_and_clip_label(frame.source_lbl_path, lbl_out / new_lbl_name)
                else:
                    # Empty label = negative frame
                    (lbl_out / new_lbl_name).touch()

                # Update frame filenames for manifest
                frame.filename = new_img_name
                frame.label_filename = new_lbl_name


def generate_manifest(splits: Dict[str, List[VideoGroup]], output_dir: Path,
                      seed: int) -> dict:
    """Generate split_manifest.json."""
    manifest = {
        "dataset_version": "IR_dsetV3",
        "seed": seed,
        "total_images": 0,
        "sources": ["goldV2", "may22", "roboflow"],
        "videos": {},
    }

    total = 0
    for split_name, videos in splits.items():
        for video in videos:
            entry = {
                "dataset_source": video.source,
                "polarity": video.polarity,
                "split": split_name,
                "frame_count": video.frame_count,
                "images": [],
            }
            if video.is_sub_split:
                entry["sub_split_of"] = video.sub_split_of

            for frame in video.frames:
                entry["images"].append({
                    "filename": frame.filename,
                    "label_filename": frame.label_filename,
                    "frame_number": frame.frame_number,
                    "bbox_count": frame.bbox_count,
                })

            manifest["videos"][video.video_id] = entry
            total += video.frame_count

    manifest["total_images"] = total

    # Compute SHA256 fingerprint
    manifest_str = json.dumps(manifest, sort_keys=True)
    manifest["manifest_sha256"] = hashlib.sha256(manifest_str.encode()).hexdigest()

    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def generate_yaml(output_dir: Path, name: str) -> Path:
    """Generate dataset.yaml."""
    yaml_content = (
        f"# {name}\n"
        f"path: {output_dir.resolve()}\n\n"
        f"train: images/train\n"
        f"val: images/valid\n"
        f"test: images/test\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: drone\n"
    )
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_path


# ─── 7. Verification ─────────────────────────────────────────────────────────

def verify_and_print(splits: Dict[str, List[VideoGroup]],
                     goldv2_stats: dict, roboflow_stats: dict) -> bool:
    """Print verification summary. Returns True if zero leakage."""
    print(f"\n{'=' * 70}")
    print(f"  DATASET V3 MERGE — VERIFICATION")
    print(f"{'=' * 70}")

    # Totals
    total = sum(v.frame_count for vl in splits.values() for v in vl)
    total_videos = sum(len(vl) for vl in splits.values())
    print(f"  Total: {total} images | {total_videos} video groups\n")

    # Split distribution
    print(f"  Split Distribution:")
    for s in ["train", "valid", "test"]:
        n = sum(v.frame_count for v in splits[s])
        nv = len(splits[s])
        pct = n / total * 100 if total else 0
        print(f"    {s:>5}: {n:>5} ({pct:>5.1f}%) | {nv:>3} groups")

    # Leakage check
    split_video_ids = {}
    leakage = False
    for s, videos in splits.items():
        for v in videos:
            base_id = v.sub_split_of or v.video_id
            if base_id in split_video_ids and split_video_ids[base_id] != s:
                # Sub-splits of the same video in different splits is EXPECTED
                if v.is_sub_split:
                    continue
                print(f"    ✗ LEAKAGE: {base_id} in {split_video_ids[base_id]} AND {s}")
                leakage = True
            if not v.is_sub_split:
                split_video_ids[base_id] = s

    if not leakage:
        print(f"    LEAKAGE CHECK: ✓ Zero overlapping video IDs\n")
    else:
        print(f"    LEAKAGE CHECK: ✗ FAILED\n")

    # Video length distribution
    lengths = [v.frame_count for vl in splits.values() for v in vl]
    print(f"  Video Length: Min {min(lengths)} | "
          f"Median {sorted(lengths)[len(lengths)//2]} | Max {max(lengths)}\n")

    # Clustering stats
    print(f"  Gold V2 Clustering:")
    print(f"    Clusters: {goldv2_stats['num_clusters']} | "
          f"Avg: {goldv2_stats['avg_frames']:.1f} frames | "
          f"Min: {goldv2_stats['min_frames']} | "
          f"Max: {goldv2_stats['max_frames']}")
    print(f"  Roboflow Clustering:")
    print(f"    Clusters: {roboflow_stats['num_clusters']} | "
          f"Avg: {roboflow_stats['avg_frames']:.1f} frames | "
          f"Min: {roboflow_stats['min_frames']} | "
          f"Max: {roboflow_stats['max_frames']}\n")

    # Polarity distribution
    print(f"  Polarity Distribution:")
    header = f"    {'':>10}"
    for p in ["WHITE_HOT", "BLACK_HOT", "MIXED"]:
        header += f"  {p:>10}"
    print(header)
    for s in ["train", "valid", "test"]:
        row = f"    {s:>10}"
        s_total = sum(v.frame_count for v in splits[s])
        for p in ["WHITE_HOT", "BLACK_HOT", "MIXED"]:
            n = sum(v.frame_count for v in splits[s] if v.polarity == p)
            pct = n / s_total * 100 if s_total else 0
            row += f"  {n:>5}({pct:>4.0f}%)"
        print(row)

    # Source distribution
    print(f"\n  Source Distribution:")
    header = f"    {'':>10}"
    for src in ["goldV2", "may22", "roboflow"]:
        header += f"  {src:>10}"
    print(header)
    for s in ["train", "valid", "test"]:
        row = f"    {s:>10}"
        s_total = sum(v.frame_count for v in splits[s])
        for src in ["goldV2", "may22", "roboflow"]:
            n = sum(v.frame_count for v in splits[s] if v.source == src)
            pct = n / s_total * 100 if s_total else 0
            row += f"  {n:>5}({pct:>4.0f}%)"
        print(row)

    # Avg bboxes per image by source
    print(f"\n  Avg Bboxes/Image:")
    for src in ["goldV2", "may22", "roboflow"]:
        all_frames = [f for vl in splits.values() for v in vl for f in v.frames if v.source == src]
        if all_frames:
            avg = sum(f.bbox_count for f in all_frames) / len(all_frames)
            print(f"    {src}: {avg:.2f}")

    print(f"{'=' * 70}")
    return not leakage


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build IR_dsetV3")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to build config YAML")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run all checks and verification but don't copy files")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(cfg["output_dir"])
    seed = cfg.get("seed", 42)
    dataset_name = cfg.get("dataset_name", "IR_dsetV3")

    print(f"\n{'=' * 70}")
    print(f"  BUILD {dataset_name}")
    print(f"{'=' * 70}")

    # ── Step 1: Integrity checks ──
    print(f"\n[1/6] Pre-merge integrity checks...")
    all_frames = {}
    for src_cfg in cfg["sources"]:
        name = src_cfg["name"]
        img_dir = Path(src_cfg["images"])
        lbl_dir = Path(src_cfg["labels"])
        regex = src_cfg.get("filename_regex")
        all_frames[name] = check_integrity(img_dir, lbl_dir, name, regex)

    # ── Step 2: Video grouping ──
    print(f"\n[2/6] Grouping frames into videos...")
    all_videos = []
    goldv2_stats = {"num_clusters": 0, "avg_frames": 0, "min_frames": 0, "max_frames": 0}
    roboflow_stats = {"num_clusters": 0, "avg_frames": 0, "min_frames": 0, "max_frames": 0}

    if "goldV2" in all_frames:
        delta_max = cfg.get("roboflow_delta_max", 10)
        min_cluster = cfg.get("roboflow_min_cluster", 30)
        gv, goldv2_stats = group_gold_v2(all_frames["goldV2"], delta_max, min_cluster)
        print(f"  goldV2: {len(gv)} clusters ({sum(v.frame_count for v in gv)} frames)")
        print(f"    Avg: {goldv2_stats['avg_frames']:.1f} | "
              f"Min: {goldv2_stats['min_frames']} | Max: {goldv2_stats['max_frames']}")
        all_videos.extend(gv)

    if "may22" in all_frames:
        mv = group_may22(all_frames["may22"])
        print(f"  may22: {len(mv)} video groups ({sum(v.frame_count for v in mv)} frames)")
        all_videos.extend(mv)

    if "roboflow" in all_frames:
        delta_max = cfg.get("roboflow_delta_max", 10)
        min_cluster = cfg.get("roboflow_min_cluster", 30)
        rv, roboflow_stats = group_roboflow(all_frames["roboflow"], delta_max, min_cluster)
        print(f"  roboflow: {len(rv)} clusters ({sum(v.frame_count for v in rv)} frames)")
        print(f"    Avg: {roboflow_stats['avg_frames']:.1f} | "
              f"Min: {roboflow_stats['min_frames']} | Max: {roboflow_stats['max_frames']}")
        all_videos.extend(rv)

    # ── Step 3: Polarity detection ──
    print(f"\n[3/6] Detecting thermal polarity...")
    min_voting = cfg.get("min_voting_frames", 10)
    for i, video in enumerate(all_videos):
        video.polarity = classify_video_polarity(video, min_voting_frames=min_voting)
        if (i + 1) % 50 == 0 or (i + 1) == len(all_videos):
            print(f"  [{i+1}/{len(all_videos)}] classified", flush=True)

    polarity_counts = Counter(v.polarity for v in all_videos)
    polarity_frame_counts = defaultdict(int)
    for v in all_videos:
        polarity_frame_counts[v.polarity] += v.frame_count
    print(f"  Polarity summary (videos): {dict(polarity_counts)}")
    print(f"  Polarity summary (frames): {dict(polarity_frame_counts)}")

    # ── Step 4: Allocate ──
    print(f"\n[4/6] Stratified allocation...")
    splits = allocate_videos(all_videos, seed=seed)

    # ── Step 5: Verify ──
    print(f"\n[5/6] Verification...")
    ok = verify_and_print(splits, goldv2_stats, roboflow_stats)

    if not ok:
        print("\n  [HALT] Leakage detected. Aborting.")
        sys.exit(1)

    if args.dry_run:
        print("\n  [DRY RUN] Skipping file copy. Verification passed.")
        return

    # ── Step 6: Copy files ──
    print(f"\n[6/6] Copying files to {output_dir}...")
    if output_dir.exists():
        print(f"  Output dir exists. Removing...")
        shutil.rmtree(output_dir)

    # Build source label dir mapping (for reference)
    source_lbl_dirs = {}
    for src_cfg in cfg["sources"]:
        source_lbl_dirs[src_cfg["name"]] = Path(src_cfg["labels"])

    copy_dataset(splits, output_dir, source_lbl_dirs)

    # Generate YAML
    yaml_path = generate_yaml(output_dir, dataset_name)
    print(f"  dataset.yaml: {yaml_path}")

    # Generate manifest
    manifest = generate_manifest(splits, output_dir, seed)
    print(f"  split_manifest.json: SHA256={manifest['manifest_sha256'][:16]}...")

    print(f"\n{'=' * 70}")
    print(f"  BUILD COMPLETE: {manifest['total_images']} images → {output_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
