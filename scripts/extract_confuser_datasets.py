"""
extract_confuser_datasets.py — Extract frames from test videos and create
per-video YOLO-format datasets.

Mirrors the source video folder structure:
  G:\drone\drone detection video tests\rgb\{category}\{video}.mp4
  ->
  datasets\drone detection video tests\rgb\{category}\{sanitized_name}\images\test\
  datasets\drone detection video tests\rgb\{category}\{sanitized_name}\labels\test\

For each video:
  1. Probe FPS and total frame count
  2. Compute adaptive stride (target ~1 fps, but ensure short videos get
     enough frames — minimum 20 frames per video)
  3. Extract frames as JPEGs into images/test/
  4. Create matching empty label files in labels/test/

Usage:
    python scripts/extract_confuser_datasets.py
    python scripts/extract_confuser_datasets.py --min-frames 30 --target-fps 2
    python scripts/extract_confuser_datasets.py --categories drone
"""

from __future__ import annotations

import argparse
import re
import json
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent
VIDEO_ROOT = Path(r"G:\drone\drone detection video tests\rgb")
OUTPUT_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"

# All categories in the video folder
ALL_CATEGORIES = ["airplanes", "birds", "drone", "helicopters"]


def sanitize_name(filename: str) -> str:
    """Create a short, filesystem-safe name from a video filename."""
    name = Path(filename).stem
    # Take the descriptive prefix before the first parenthesis
    if "(" in name:
        name = name[:name.index("(")].strip()
    # Collapse whitespace and special chars
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    name = name.strip("_").lower()
    # Truncate if too long
    if len(name) > 60:
        name = name[:60].rstrip("_")
    return name


def compute_stride(fps: float, total_frames: int, target_fps: float,
                   min_frames: int) -> int:
    """Compute adaptive stride.

    - Default: stride = fps / target_fps  (~1 frame per second at target_fps=1)
    - If that would produce fewer than min_frames, reduce stride so we get
      at least min_frames from the video.
    """
    stride = max(1, int(round(fps / target_fps)))

    expected = total_frames // stride
    if expected < min_frames and total_frames >= min_frames:
        stride = max(1, total_frames // min_frames)
    elif total_frames < min_frames:
        stride = 1

    return stride


def extract_video(video_path: Path, category: str, target_fps: float,
                  min_frames: int, output_root: Path,
                  force_stride: int = 0) -> dict:
    """Extract frames from a single video into a YOLO dataset structure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0

    short_name = sanitize_name(video_path.name)

    # Mirror the folder structure: rgb/{category}/{short_name}/
    ds_dir = output_root / category / short_name
    img_dir = ds_dir / "images" / "test"
    lbl_dir = ds_dir / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already extracted
    existing = list(img_dir.glob("*.jpg"))
    if existing:
        print(f"\n  Video: {video_path.name}")
        print(f"    SKIP: Already extracted ({len(existing)} frames in {ds_dir.relative_to(REPO)})")
        return {
            "dataset_name": short_name,
            "category": category,
            "video": video_path.name,
            "fps": fps,
            "total_frames": total_frames,
            "duration_s": round(duration_s, 1),
            "stride": 0,
            "extracted": len(existing),
            "path": str(ds_dir),
            "skipped": True,
        }

    stride = force_stride if force_stride > 0 else compute_stride(fps, total_frames, target_fps, min_frames)
    expected_frames = total_frames // stride

    print(f"\n  Video: {video_path.name}")
    print(f"    Category:  {category}")
    print(f"    FPS:       {fps:.1f}")
    print(f"    Frames:    {total_frames:,}")
    print(f"    Duration:  {duration_s:.1f}s")
    print(f"    Stride:    {stride}")
    print(f"    Expected:  ~{expected_frames} frames")

    extracted = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        stem = f"frame_{extracted + 1:05d}"
        img_path = img_dir / f"{stem}.jpg"
        lbl_path = lbl_dir / f"{stem}.txt"

        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl_path.write_text("")  # Empty label

        extracted += 1

    cap.release()
    print(f"    Extracted: {extracted} frames -> {ds_dir.relative_to(REPO)}")

    return {
        "dataset_name": short_name,
        "category": category,
        "video": video_path.name,
        "fps": fps,
        "total_frames": total_frames,
        "duration_s": round(duration_s, 1),
        "stride": stride,
        "extracted": extracted,
        "path": str(ds_dir),
        "skipped": False,
    }


def main():
    ap = argparse.ArgumentParser(description="Extract video test datasets")
    ap.add_argument("--target-fps", type=float, default=1.0,
                    help="Target extraction rate in frames/second (default: 1)")
    ap.add_argument("--min-frames", type=int, default=20,
                    help="Minimum frames to extract per video (default: 20)")
    ap.add_argument("--force-stride", type=int, default=0,
                    help="Override stride for all videos (0 = adaptive)")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="Specific categories to process (default: all)")
    args = ap.parse_args()

    categories = args.categories or ALL_CATEGORIES

    print(f"Video Test Dataset Extraction")
    print(f"  Source: {VIDEO_ROOT}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Target FPS: {args.target_fps}")
    print(f"  Min frames: {args.min_frames}")
    print(f"  Categories: {categories}")

    all_info = []
    for cat in categories:
        cat_dir = VIDEO_ROOT / cat
        if not cat_dir.exists():
            print(f"  WARNING: {cat_dir} not found, skipping")
            continue
        videos = sorted(cat_dir.glob("*.mp4"))
        if not videos:
            print(f"  WARNING: No .mp4 files in {cat_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"  Category: {cat.upper()} ({len(videos)} videos)")
        print(f"{'='*70}")

        for vpath in videos:
            info = extract_video(vpath, cat, args.target_fps,
                                 args.min_frames, OUTPUT_ROOT,
                                 force_stride=args.force_stride)
            if info:
                all_info.append(info)

    # Summary
    print(f"\n{'='*70}")
    print(f"  EXTRACTION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Dataset':<45s} {'Cat':<12s} {'Dur(s)':<8s} {'Stride':<8s} {'Frames':<8s}")
    print(f"  {'-'*85}")
    total_frames = 0
    for info in all_info:
        skipped = " (cached)" if info.get("skipped") else ""
        print(f"  {info['dataset_name']:<45s} {info['category']:<12s} "
              f"{info['duration_s']:<8.1f} {info['stride']:<8d} "
              f"{info['extracted']:<8d}{skipped}")
        total_frames += info["extracted"]
    print(f"  {'-'*85}")
    print(f"  {'TOTAL':<45s} {'':12s} {'':8s} {'':8s} {total_frames:<8d}")

    # Save manifest
    manifest_path = OUTPUT_ROOT / "extraction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(all_info, f, indent=2)
    print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
