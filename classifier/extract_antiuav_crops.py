"""
extract_antiuav_crops.py — Extract drone crops from Anti-UAV YOLO dataset
and retrain the confuser filter (binary: confuser vs pass).

Steps:
  1. Sample Anti-UAV images + YOLO labels
  2. Crop GT boxes, save as drone negative patches
  3. Update manifest with new crops
  4. Retrain two confuser filters (RGB + IR)

Usage:
    python classifier/extract_antiuav_crops.py
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_DIR = SCRIPT_DIR / "runs" / "patches"
MANIFEST_PATH = PATCH_DIR / "manifest.csv"

ANTIUAV_BASE = Path(r"G:\drone\Anti-UAV-RGBT_yolo_converted")
MIN_CROP_SIZE = 16  # Skip tiny boxes


def extract_crops(max_per_modality=2500, sample_every=30, seed=42):
    """Extract drone crops from Anti-UAV YOLO dataset."""
    random.seed(seed)
    np.random.seed(seed)

    # Load existing manifest
    manifest = pd.read_csv(MANIFEST_PATH)
    existing_stems = set(manifest["stem"].values)
    print(f"Existing manifest: {len(manifest)} rows")

    new_rows = []

    for split in ["test", "val"]:
        for mod_key, mod_folder in [("rgb", "RGB"), ("ir", "IR")]:
            img_dir = ANTIUAV_BASE / split / mod_folder / "images"
            lbl_dir = ANTIUAV_BASE / split / mod_folder / "labels"

            if not img_dir.exists():
                print(f"  SKIP {split}/{mod_folder}: not found")
                continue

            # Get all image files
            img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

            # Sample every Nth frame
            sampled = img_files[::sample_every]
            random.shuffle(sampled)

            print(f"  {split}/{mod_folder}: {len(img_files)} total, {len(sampled)} sampled")

            count = 0
            for img_path in sampled:
                if count >= max_per_modality:
                    break

                # Find corresponding label
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue

                # Read label
                content = lbl_path.read_text().strip()
                if not content:
                    continue  # No drone visible

                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

                # Parse YOLO boxes (may have multiple)
                for line_idx, line in enumerate(content.split("\n")):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    # YOLO format: class_id cx cy bw bh (normalized)
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                    # Convert to pixel coords
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)

                    # Add 20% padding
                    pad_w = int((x2 - x1) * 0.2)
                    pad_h = int((y2 - y1) * 0.2)
                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(w, x2 + pad_w)
                    y2 = min(h, y2 + pad_h)

                    # Skip tiny crops
                    if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                        continue

                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Save
                    stem = f"antiuav_{split}_{img_path.stem}_b{line_idx}"
                    if stem in existing_stems:
                        continue

                    out_dir = PATCH_DIR / mod_key / "drone"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{stem}.jpg"
                    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

                    new_rows.append({
                        "stem": stem,
                        "path": str(out_path.relative_to(SCRIPT_DIR)),
                        "modality": mod_key,
                        "label": "drone",
                        "category": "drone",
                        "video": f"antiuav_{split}_{img_path.stem.rsplit('_', 1)[0]}",
                    })
                    existing_stems.add(stem)
                    count += 1
                    break  # Only take first box per frame

            print(f"    -> Extracted {count} crops")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        updated = pd.concat([manifest, new_df], ignore_index=True)
        updated.to_csv(MANIFEST_PATH, index=False)
        print(f"\nManifest updated: {len(manifest)} -> {len(updated)} rows")
    else:
        print("\nNo new crops extracted")
        updated = manifest

    # Print final distribution
    print("\n=== Final distribution ===")
    print(updated.groupby(["modality", "category"]).size().unstack(fill_value=0).to_string())

    return updated


if __name__ == "__main__":
    extract_crops()
