"""
extract_patches_v2.py — Augment the confuser filter training set with:

  1. Drone crops from the YOLO training datasets (IR + RGB)
     → "other" class, diverse drone appearances
  2. Hard-negative confuser crops from YouTube OOD videos
     → YOLO FP mining: run YOLO, crop each detection as confuser

Appends new rows to the existing manifest.csv.  Safe to re-run:
skips crops that already exist on disk.

Sources:
  Drone (other):
    IR:  G:/drone/IR_dset_final/{train,test}/  (YOLO labels)
    RGB: G:/drone/dataset/dataset/             (YOLO labels)
  Confusers (hard-negative mining):
    IR:  ir_gui/demo_outputs/yt_*.mp4          (thermal YouTube)
    RGB: D:/Downloads/youtube_classifier_videos/*.mp4

Usage:
    python classifier/extract_patches_v2.py
    python classifier/extract_patches_v2.py --skip-youtube   # only YOLO datasets
    python classifier/extract_patches_v2.py --skip-yolo      # only YouTube mining
    python classifier/extract_patches_v2.py --dry-run        # show plan, no writes
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
PATCH_DIR = SCRIPT_DIR / "runs" / "patches"
MANIFEST_PATH = PATCH_DIR / "manifest.csv"

# ── YOLO dataset sources (drone crops → "other") ──────────────────

IR_DSET = Path(r"G:/drone/IR_dset_final")
RGB_DSET_IMAGES = Path(r"G:/drone/dataset/dataset/images")
RGB_DSET_LABELS = Path(r"G:/drone/dataset/dataset/labels")

YOLO_SOURCES = [
    # (images_dir, labels_dir, modality, split_name)
    (IR_DSET / "train" / "images", IR_DSET / "train" / "labels", "ir", "ir_dset_train"),
    (IR_DSET / "test" / "images",  IR_DSET / "test" / "labels",  "ir", "ir_dset_test"),
    (RGB_DSET_IMAGES / "train",    RGB_DSET_LABELS / "train",    "rgb", "rgb_dset_train"),
    (RGB_DSET_IMAGES / "test",     RGB_DSET_LABELS / "test",     "rgb", "rgb_dset_test"),
]

# ── YouTube confuser videos (hard-negative mining) ────────────────

YT_IR_DIR = REPO / "ir_gui" / "demo_outputs"
YT_RGB_DIR = Path(r"D:/Downloads/youtube_classifier_videos")

YT_IR_CONFUSERS = {
    "yt_EdOX8tJZDzw.mp4": "helicopter",
    "yt_gg0Da0AtWJk.mp4": "airplane",
    "yt_LflkvbKEEr8.mp4": "airplane",
    "yt_UwOMwAGVwvs.mp4": "airplane",
    "yt_oon2AjhmAE8.mp4": "airplane",
    "yt_vfLc8n8mcKo.mp4": "airplane",
    "yt_r5tBDvY7MrA.mp4": "airplane",
    "yt_5BYnJQfMvrg.mp4": "airplane",
    "yt_omoX_2UYb0s.mp4": "bird",
    "yt_NEANQ74oTew.mp4": "bird",
}

YT_RGB_CONFUSERS = {
    "airplane_rgb.mp4": "airplane",
    "airplane_rgb_2.mp4": "airplane",
    "airplane_rgb_3.mp4": "airplane",
    "airplane_rgb_compilation.mp4": "airplane",
    "heli_rgb.mp4": "helicopter",
    "heli_rgb_2.mp4": "helicopter",
    "bird_rgb.mp4": "bird",
    "birds_flock_rgb.mp4": "bird",
}

# RGB confuser videos that live in demo_outputs (alongside IR videos)
YT_RGB_DEMO_CONFUSERS = {
    "yt_Z8HJNypu_1Y.mp4": "airplane",
    "yt_JkK2KcJVXpg.mp4": "airplane",
    "yt_1U7Bu2pSUwU.mp4": "helicopter",
    "yt_ZO5lV0gh5i4.mp4": "bird",
}

# ── Cropping ──────────────────────────────────────────────────────

def crop_with_context(img, x1, y1, x2, y2, pad_frac=0.5, min_side=24):
    """Square crop around bbox with context padding (same as PatchVerifier)."""
    ih, iw = img.shape[:2]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + 2.0 * pad_frac)
    side = max(side, float(min_side))
    ax1 = int(round(cx - side / 2))
    ay1 = int(round(cy - side / 2))
    ax2 = int(round(cx + side / 2))
    ay2 = int(round(cy + side / 2))
    ax1 = max(0, ax1); ay1 = max(0, ay1)
    ax2 = min(iw, ax2); ay2 = min(ih, ay2)
    if ax2 - ax1 < min_side or ay2 - ay1 < min_side:
        return None
    return img[ay1:ay2, ax1:ax2]


# ── 1. YOLO dataset extraction ───────────────────────────────────

def extract_yolo_drone_crops(sources, out_root, max_crops_per_split=3000,
                              sample_every=1, dry_run=False):
    """Read YOLO labels, crop around class-0 (drone) boxes."""
    rows = []
    for img_dir, lbl_dir, modality, split_name in sources:
        if not img_dir.exists():
            print(f"  [skip] {img_dir} not found")
            continue
        if not lbl_dir.exists():
            print(f"  [skip] {lbl_dir} not found")
            continue

        out_dir = out_root / modality / "drone"
        out_dir.mkdir(parents=True, exist_ok=True)

        # List images
        imgs = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        # Stride-sample to target max_crops_per_split
        # Each image may have multiple boxes, so sample more conservatively
        step = max(1, len(imgs) // (max_crops_per_split * 2))
        sampled = imgs[::step]

        print(f"\n  [{modality}/{split_name}] {len(imgs):,} images -> "
              f"sampling every {step} -> {len(sampled):,} candidates")

        count = 0
        t0 = time.time()
        for idx, img_path in enumerate(sampled):
            if count >= max_crops_per_split:
                break

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Read image lazily only if we have labels
            img = None
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(parts[0])
                except ValueError:
                    continue
                if cls != 0:  # only drone class
                    continue

                if img is None:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        break
                h, w = img.shape[:2]

                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h

                crop = crop_with_context(img, x1, y1, x2, y2)
                if crop is None:
                    continue

                stem = f"yolo_{split_name}_{img_path.stem}_b{count}"
                out_path = out_dir / f"{stem}.jpg"

                if not dry_run and not out_path.exists():
                    cv2.imwrite(str(out_path), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])

                rows.append({
                    "stem": stem,
                    "path": str(out_path),
                    "modality": modality,
                    "label": "drone",
                    "category": "drone",
                    "video": f"yolo_{split_name}",
                })
                count += 1
                if count >= max_crops_per_split:
                    break

            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"    [{idx+1}/{len(sampled)}] {count} crops  "
                      f"({elapsed:.0f}s)")

        print(f"  -> {count} drone crops from {split_name}")

    return rows


# ── 2. YouTube hard-negative mining ───────────────────────────────

def mine_youtube_confusers(video_map, video_dir, modality, model_path,
                           conf, out_root, stride=3, max_per_video=200,
                           dry_run=False, grayscale=False):
    """Run YOLO on confuser videos, crop every FP detection.
    
    If grayscale=True, converts frames to 3-channel grayscale before
    inference (for running IR model on RGB videos). Saved crops are
    also grayscale.
    """
    from ultralytics import YOLO

    rows = []
    available = {
        name: cat for name, cat in video_map.items()
        if (video_dir / name).exists()
    }
    if not available:
        print(f"  [skip] no {modality} YouTube videos found in {video_dir}")
        return rows

    print(f"\n  Loading {modality.upper()} YOLO model: {model_path}")
    model = YOLO(str(model_path))

    for vid_name, category in available.items():
        vid_path = video_dir / vid_name
        out_dir = out_root / modality / category
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"  [skip] cannot open {vid_name}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = -1
        count = 0

        while count < max_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % stride != 0:
                continue

            # Convert to grayscale if needed (for IR model on RGB videos)
            infer_frame = frame
            if grayscale:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                infer_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            res = model.predict(infer_frame, conf=conf, iou=0.45, imgsz=640,
                                verbose=False, device=0, max_det=50)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue

            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

            for bi in range(len(confs)):
                if count >= max_per_video:
                    break
                x1, y1, x2, y2 = xyxy[bi]
                # Crop from the infer_frame (grayscale if applicable)
                crop = crop_with_context(infer_frame, x1, y1, x2, y2)
                if crop is None:
                    continue

                stem = f"ythn_{modality}_{vid_name.replace('.mp4','')}_f{frame_idx:06d}_b{bi}"
                out_path = out_dir / f"{stem}.jpg"

                if not dry_run and not out_path.exists():
                    cv2.imwrite(str(out_path), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])

                rows.append({
                    "stem": stem,
                    "path": str(out_path),
                    "modality": modality,
                    "label": "aerial",
                    "category": category,
                    "video": vid_name,
                })
                count += 1

        cap.release()
        print(f"    {category:12s} {vid_name:40s} -> {count} crops "
              f"(from {frame_idx+1} frames)")

    return rows


# ── Main ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-drone-crops", type=int, default=3000,
                   help="Max drone crops per YOLO dataset split (default 3000)")
    p.add_argument("--max-yt-per-video", type=int, default=200,
                   help="Max confuser crops per YouTube video (default 200)")
    p.add_argument("--yt-stride", type=int, default=3,
                   help="Process every Nth frame for YouTube (default 3)")
    p.add_argument("--skip-yolo", action="store_true",
                   help="Skip YOLO dataset drone extraction")
    p.add_argument("--skip-youtube", action="store_true",
                   help="Skip YouTube hard-negative mining")
    p.add_argument("--dry-run", action="store_true",
                   help="Show plan without writing files")
    p.add_argument("--ir-conf", type=float, default=0.40)
    p.add_argument("--rgb-conf", type=float, default=0.25)
    args = p.parse_args()

    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing manifest to check for duplicates
    existing_stems = set()
    if MANIFEST_PATH.exists():
        import pandas as pd
        old = pd.read_csv(MANIFEST_PATH)
        existing_stems = set(old["stem"].tolist())
        print(f"Existing manifest: {len(old)} rows")
    else:
        print("No existing manifest found — creating fresh")

    all_new_rows = []

    # ── 1. YOLO dataset drone crops ──────────────────────────────
    if not args.skip_yolo:
        print("\n" + "=" * 60)
        print("PHASE 1: YOLO Dataset Drone Crops")
        print("=" * 60)
        yolo_rows = extract_yolo_drone_crops(
            YOLO_SOURCES, PATCH_DIR,
            max_crops_per_split=args.max_drone_crops,
            dry_run=args.dry_run,
        )
        # Deduplicate against existing manifest
        new_yolo = [r for r in yolo_rows if r["stem"] not in existing_stems]
        print(f"\n  YOLO total: {len(yolo_rows)} extracted, "
              f"{len(new_yolo)} new (after dedup)")
        all_new_rows.extend(new_yolo)

    # ── 2. YouTube hard-negative mining ──────────────────────────
    if not args.skip_youtube:
        print("\n" + "=" * 60)
        print("PHASE 2: YouTube Hard-Negative Mining")
        print("=" * 60)

        # Load model paths from settings
        settings_path = REPO / "ir_gui" / "fusion_settings.json"
        with open(settings_path) as f:
            settings = json.load(f)

        # IR confusers
        print("\n  --- IR YouTube Confusers ---")
        ir_rows = mine_youtube_confusers(
            YT_IR_CONFUSERS, YT_IR_DIR, "ir",
            model_path=settings["ir_model"],
            conf=args.ir_conf,
            out_root=PATCH_DIR,
            stride=args.yt_stride,
            max_per_video=args.max_yt_per_video,
            dry_run=args.dry_run,
        )

        # RGB confusers
        print("\n  --- RGB YouTube Confusers ---")
        rgb_rows = mine_youtube_confusers(
            YT_RGB_CONFUSERS, YT_RGB_DIR, "rgb",
            model_path=settings["rgb_model"],
            conf=args.rgb_conf,
            out_root=PATCH_DIR,
            stride=args.yt_stride,
            max_per_video=args.max_yt_per_video,
            dry_run=args.dry_run,
        )

        yt_rows = ir_rows + rgb_rows

        # RGB confusers from demo_outputs dir
        print("\n  --- RGB YouTube Confusers (demo_outputs) ---")
        rgb_demo_rows = mine_youtube_confusers(
            YT_RGB_DEMO_CONFUSERS, YT_IR_DIR, "rgb",
            model_path=settings["rgb_model"],
            conf=args.rgb_conf,
            out_root=PATCH_DIR,
            stride=args.yt_stride,
            max_per_video=args.max_yt_per_video,
            dry_run=args.dry_run,
        )
        yt_rows += rgb_demo_rows

        # IR confusers from demo_outputs RGB videos (grayscale conversion)
        print("\n  --- IR YouTube Confusers (grayscale from RGB videos) ---")
        ir_gray_rows = mine_youtube_confusers(
            YT_RGB_DEMO_CONFUSERS, YT_IR_DIR, "ir",
            model_path=settings["ir_model"],
            conf=args.ir_conf,
            out_root=PATCH_DIR,
            stride=args.yt_stride,
            max_per_video=args.max_yt_per_video,
            dry_run=args.dry_run,
            grayscale=True,
        )
        yt_rows += ir_gray_rows
        new_yt = [r for r in yt_rows if r["stem"] not in existing_stems]
        print(f"\n  YouTube total: {len(yt_rows)} extracted, "
              f"{len(new_yt)} new (after dedup)")
        all_new_rows.extend(new_yt)

    # ── Summary & write ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY OF NEW CROPS")
    print("=" * 60)

    by_mod_cat = Counter()
    for r in all_new_rows:
        by_mod_cat[(r["modality"], r["category"])] += 1
    for (mod, cat), n in sorted(by_mod_cat.items()):
        print(f"  {mod:>3s}/{cat:<12s} {n:>6d}")
    print(f"  {'TOTAL':<16s} {len(all_new_rows):>6d}")

    if args.dry_run:
        print("\n  [dry-run] No files written.")
        return

    if not all_new_rows:
        print("\n  No new rows to add.")
        return

    # Append to manifest
    print(f"\nAppending {len(all_new_rows)} rows to {MANIFEST_PATH}")
    file_exists = MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["stem", "path", "modality", "label", "category", "video"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        for row in all_new_rows:
            w.writerow(row)

    # Print final distribution
    if MANIFEST_PATH.exists():
        import pandas as pd
        final = pd.read_csv(MANIFEST_PATH)
        print(f"\nFinal manifest: {len(final)} rows")
        print(final.groupby(["modality", "category"]).size()
              .unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
