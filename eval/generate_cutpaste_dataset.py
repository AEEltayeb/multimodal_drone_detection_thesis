"""
generate_cutpaste_dataset.py - Cut-paste paired RGB+IR synthetic drone dataset.

Extracts real drone crops from Anti-UAV and Svanstrom (paired RGB+IR),
pastes them onto paired backgrounds at controlled sizes,
producing a paired evaluation dataset with exact ground-truth bboxes.

KEY DESIGN DECISIONS:
  - RGB crops use RGB labels; IR crops use IR labels (cameras have different FOV)
  - Backgrounds are drone-positive frames with the drone region inpainted out
  - Both modalities share the same NORMALIZED paste position + size
  - Minimum drone size enforced to avoid invisible targets

Usage:
    python generate_cutpaste_dataset.py --n-samples 10 --output G:/drone/cutpaste_eval --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


# -- Source dataset paths -----------------------------------------------------

ANTIUAV_ROOT = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test")
ANTIUAV_RGB_IMG = ANTIUAV_ROOT / "RGB/images"
ANTIUAV_RGB_LBL = ANTIUAV_ROOT / "RGB/labels"
ANTIUAV_IR_IMG = ANTIUAV_ROOT / "IR/images"
ANTIUAV_IR_LBL = ANTIUAV_ROOT / "IR/labels"

SVAN_ROOT = Path("G:/drone/svanstrom_paired")
SVAN_RGB_IMG = SVAN_ROOT / "RGB/images"
SVAN_RGB_LBL = SVAN_ROOT / "RGB/labels"
SVAN_IR_IMG = SVAN_ROOT / "IR/images"
SVAN_IR_LBL = SVAN_ROOT / "IR/labels"

# Output resolution (uniform for both modalities)
OUT_W, OUT_H = 640, 512


# -- Data structures ----------------------------------------------------------

class CropPair(NamedTuple):
    rgb_crop: np.ndarray
    ir_crop: np.ndarray
    source: str
    stem: str
    orig_w_norm: float
    orig_h_norm: float


class BackgroundPair(NamedTuple):
    rgb_bg: np.ndarray  # already resized + inpainted
    ir_bg: np.ndarray
    stem: str


# -- YOLO label helpers -------------------------------------------------------

def read_yolo_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read YOLO label -> list of (cls, cx, cy, w, h) normalized."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((cls_id, cx, cy, w, h))
        except ValueError:
            continue
    return boxes


def find_image(stem: str, img_dir: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# -- Inpainting: remove drone from frame to create clean background -----------

def inpaint_drone(img: np.ndarray, boxes: list[tuple], expand: float = 0.3) -> np.ndarray:
    """Inpaint drone regions out of an image using its YOLO boxes.
    
    Args:
        img: BGR image
        boxes: list of (cls, cx, cy, w, h) in normalized coords
        expand: how much to expand the mask beyond the bbox (fraction)
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for _, cx, cy, bw, bh in boxes:
        # Pixel coords with expanded margin
        ex = bw * expand
        ey = bh * expand
        x1 = max(0, int((cx - bw / 2 - ex) * w))
        y1 = max(0, int((cy - bh / 2 - ey) * h))
        x2 = min(w, int((cx + bw / 2 + ex) * w))
        y2 = min(h, int((cy + bh / 2 + ey) * h))
        mask[y1:y2, x1:x2] = 255

    if mask.max() == 0:
        return img

    # Use Telea inpainting (fast, good for sky/uniform regions)
    radius = max(3, int(min(w, h) * 0.02))
    result = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return result


# -- Step 1: Create inpainted backgrounds ------------------------------------

def create_backgrounds(max_count: int = 200, stride: int = 100) -> list[BackgroundPair]:
    """Create clean paired backgrounds by inpainting drones out of positive frames.
    
    Uses drone-positive frames (where we know exactly where the drone is)
    and inpaints that region to create a clean background.
    """
    print("Creating inpainted backgrounds from Anti-UAV positive frames...")
    backgrounds = []
    rgb_labels = sorted(ANTIUAV_RGB_LBL.glob("*.txt"))

    for i, rgb_lbl in enumerate(rgb_labels):
        if i % stride != 0:
            continue

        # Must have drone bbox in RGB
        rgb_boxes = read_yolo_boxes(rgb_lbl)
        if not rgb_boxes:
            continue

        rgb_stem = rgb_lbl.stem
        ir_stem = rgb_stem.replace("_visible_", "_infrared_")

        # Must have drone bbox in IR too
        ir_lbl = ANTIUAV_IR_LBL / f"{ir_stem}.txt"
        ir_boxes = read_yolo_boxes(ir_lbl)
        if not ir_boxes:
            continue

        rgb_path = find_image(rgb_stem, ANTIUAV_RGB_IMG)
        ir_path = find_image(ir_stem, ANTIUAV_IR_IMG)
        if not rgb_path or not ir_path:
            continue

        rgb_img = cv2.imread(str(rgb_path))
        ir_img = cv2.imread(str(ir_path))
        if rgb_img is None or ir_img is None:
            continue

        # Inpaint drone out of BOTH modalities (using their own labels)
        rgb_clean = inpaint_drone(rgb_img, rgb_boxes, expand=0.4)
        ir_clean = inpaint_drone(ir_img, ir_boxes, expand=0.4)

        # Resize to output resolution
        rgb_clean = cv2.resize(rgb_clean, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        ir_clean = cv2.resize(ir_clean, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

        backgrounds.append(BackgroundPair(rgb_clean, ir_clean, rgb_stem))

        if len(backgrounds) >= max_count:
            break

    print(f"  Created {len(backgrounds)} inpainted background pairs")
    return backgrounds


# -- Step 2: Extract drone crop pairs ----------------------------------------

def extract_crop(img: np.ndarray, cx: float, cy: float, w: float, h: float,
                 padding: float = 0.1) -> np.ndarray | None:
    """Extract a crop from image using normalized YOLO coords with padding."""
    ih, iw = img.shape[:2]
    x1 = int((cx - w / 2) * iw)
    y1 = int((cy - h / 2) * ih)
    x2 = int((cx + w / 2) * iw)
    y2 = int((cy + h / 2) * ih)

    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(iw, x2 + pad_x)
    y2 = min(ih, y2 + pad_y)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return crop


def extract_drone_crops(max_per_source: int = 200, stride: int = 50) -> list[CropPair]:
    """Extract paired drone crops using EACH modality's OWN labels."""
    crops = []

    # --- Anti-UAV ---
    print("Extracting Anti-UAV drone crops (using per-modality labels)...")
    antiuav_count = 0
    for i, rgb_lbl in enumerate(sorted(ANTIUAV_RGB_LBL.glob("*.txt"))):
        if i % stride != 0:
            continue

        rgb_boxes = read_yolo_boxes(rgb_lbl)
        rgb_drones = [b for b in rgb_boxes if b[0] == 0]
        if not rgb_drones:
            continue

        rgb_stem = rgb_lbl.stem
        ir_stem = rgb_stem.replace("_visible_", "_infrared_")

        # IR must also have its own drone label
        ir_lbl = ANTIUAV_IR_LBL / f"{ir_stem}.txt"
        ir_boxes = read_yolo_boxes(ir_lbl)
        ir_drones = [b for b in ir_boxes if b[0] == 0]
        if not ir_drones:
            continue

        rgb_path = find_image(rgb_stem, ANTIUAV_RGB_IMG)
        ir_path = find_image(ir_stem, ANTIUAV_IR_IMG)
        if not rgb_path or not ir_path:
            continue

        rgb_img = cv2.imread(str(rgb_path))
        ir_img = cv2.imread(str(ir_path))
        if rgb_img is None or ir_img is None:
            continue

        # Use RGB labels for RGB crop, IR labels for IR crop
        _, rcx, rcy, rw, rh = rgb_drones[0]
        _, icx, icy, iw, ih = ir_drones[0]

        # Filter: not touching edges
        if (rcx - rw/2 < 0.02 or rcy - rh/2 < 0.02 or
            rcx + rw/2 > 0.98 or rcy + rh/2 > 0.98):
            continue
        if (icx - iw/2 < 0.02 or icy - ih/2 < 0.02 or
            icx + iw/2 > 0.98 or icy + ih/2 > 0.98):
            continue

        # Filter: minimum pixel size
        rgb_min_px = min(rw * rgb_img.shape[1], rh * rgb_img.shape[0])
        ir_min_px = min(iw * ir_img.shape[1], ih * ir_img.shape[0])
        if rgb_min_px < 12 or ir_min_px < 8:
            continue

        rgb_crop = extract_crop(rgb_img, rcx, rcy, rw, rh)
        ir_crop = extract_crop(ir_img, icx, icy, iw, ih)
        if rgb_crop is None or ir_crop is None:
            continue

        # Validate IR crop isn't just black/uniform
        if ir_crop.std() < 5:
            continue

        crops.append(CropPair(rgb_crop, ir_crop, "antiuav", rgb_stem, rw, rh))
        antiuav_count += 1
        if antiuav_count >= max_per_source:
            break

    print(f"  Anti-UAV: {antiuav_count} crop pairs")

    # --- Svanstrom ---
    print("Extracting Svanstrom drone crops (using per-modality labels)...")
    svan_count = 0
    drone_labels = sorted(f for f in SVAN_RGB_LBL.glob("*.txt") if "_DRONE_" in f.stem)
    for i, rgb_lbl in enumerate(drone_labels):
        if i % stride != 0:
            continue

        rgb_boxes = read_yolo_boxes(rgb_lbl)
        rgb_drones = [b for b in rgb_boxes if b[0] == 0]
        if not rgb_drones:
            continue

        rgb_stem = rgb_lbl.stem
        ir_stem = rgb_stem.replace("_visible", "_infrared")

        ir_lbl = SVAN_IR_LBL / f"{ir_stem}.txt"
        ir_boxes = read_yolo_boxes(ir_lbl)
        ir_drones = [b for b in ir_boxes if b[0] == 0]
        if not ir_drones:
            continue

        rgb_path = find_image(rgb_stem, SVAN_RGB_IMG)
        ir_path = find_image(ir_stem, SVAN_IR_IMG)
        if not rgb_path or not ir_path:
            continue

        rgb_img = cv2.imread(str(rgb_path))
        ir_img = cv2.imread(str(ir_path))
        if rgb_img is None or ir_img is None:
            continue

        _, rcx, rcy, rw, rh = rgb_drones[0]
        _, icx, icy, iw, ih = ir_drones[0]

        if (rcx - rw/2 < 0.02 or rcy - rh/2 < 0.02 or
            rcx + rw/2 > 0.98 or rcy + rh/2 > 0.98):
            continue
        if (icx - iw/2 < 0.02 or icy - ih/2 < 0.02 or
            icx + iw/2 > 0.98 or icy + ih/2 > 0.98):
            continue

        rgb_min_px = min(rw * rgb_img.shape[1], rh * rgb_img.shape[0])
        ir_min_px = min(iw * ir_img.shape[1], ih * ir_img.shape[0])
        if rgb_min_px < 10 or ir_min_px < 6:
            continue

        rgb_crop = extract_crop(rgb_img, rcx, rcy, rw, rh)
        ir_crop = extract_crop(ir_img, icx, icy, iw, ih)
        if rgb_crop is None or ir_crop is None:
            continue

        if ir_crop.std() < 5:
            continue

        crops.append(CropPair(rgb_crop, ir_crop, "svanstrom", rgb_stem, rw, rh))
        svan_count += 1
        if svan_count >= max_per_source:
            break

    print(f"  Svanstrom: {svan_count} crop pairs")
    print(f"  Total crop bank: {len(crops)}")
    return crops


# -- Step 3: Compose synthetic paired frames ----------------------------------

def create_alpha_mask(h: int, w: int, feather_frac: float = 0.15) -> np.ndarray:
    """Create a soft alpha mask with smooth feathered edges."""
    mask = np.ones((h, w), dtype=np.float32)
    feather = max(2, int(min(h, w) * feather_frac))

    for i in range(feather):
        alpha = (i + 1) / (feather + 1)
        mask[i, :] *= alpha
        mask[h - 1 - i, :] *= alpha
        mask[:, i] *= alpha
        mask[:, w - 1 - i] *= alpha
    return mask


def paste_crop_onto_bg(bg: np.ndarray, crop: np.ndarray,
                       cx_norm: float, cy_norm: float,
                       target_w: int, target_h: int) -> np.ndarray:
    """Paste a resized crop onto background with alpha feathering."""
    result = bg.copy()
    bh, bw = bg.shape[:2]

    resized = cv2.resize(crop, (target_w, target_h),
                         interpolation=cv2.INTER_AREA if target_w < crop.shape[1] else cv2.INTER_LINEAR)

    px = int(cx_norm * bw - target_w / 2)
    py = int(cy_norm * bh - target_h / 2)

    # Compute valid paste region
    src_x1 = max(0, -px)
    src_y1 = max(0, -py)
    dst_x1 = max(0, px)
    dst_y1 = max(0, py)
    src_x2 = min(target_w, bw - px)
    src_y2 = min(target_h, bh - py)
    dst_x2 = min(bw, px + target_w)
    dst_y2 = min(bh, py + target_h)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return result

    paste_h = dst_y2 - dst_y1
    paste_w = dst_x2 - dst_x1
    paste_region = resized[src_y1:src_y1 + paste_h, src_x1:src_x1 + paste_w]
    alpha = create_alpha_mask(paste_h, paste_w)[:, :, np.newaxis]

    result[dst_y1:dst_y2, dst_x1:dst_x2] = (
        paste_region.astype(np.float32) * alpha +
        result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32) * (1 - alpha)
    ).astype(np.uint8)

    return result


def pick_difficulty(tier: str, rng: random.Random) -> float:
    """Return target bbox area as fraction of image area."""
    if tier == "easy":
        return rng.uniform(0.015, 0.05)     # 1.5-5% of image -> clearly visible
    elif tier == "medium":
        return rng.uniform(0.003, 0.015)    # 0.3-1.5%
    elif tier == "hard":
        return rng.uniform(0.001, 0.003)    # 0.1-0.3% -> small but still visible
    else:
        return rng.uniform(0.003, 0.015)


def pick_position(tier: str, rng: random.Random) -> tuple[float, float]:
    """Pick normalized paste center position."""
    if tier == "easy":
        return rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75)
    elif tier == "medium":
        return rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
    else:
        return rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)


def generate_sample(crop: CropPair, bg: BackgroundPair, tier: str,
                    rng: random.Random) -> tuple | None:
    """Generate one paired synthetic sample."""
    rgb_bg = bg.rgb_bg.copy()
    ir_bg = bg.ir_bg.copy()

    # Target area
    target_area_frac = pick_difficulty(tier, rng)
    total_pixels = OUT_W * OUT_H

    # Target dimensions preserving RGB crop aspect ratio
    crop_aspect = crop.rgb_crop.shape[1] / max(crop.rgb_crop.shape[0], 1)
    target_area_px = target_area_frac * total_pixels
    target_h = max(8, int(np.sqrt(target_area_px / max(crop_aspect, 0.1))))
    target_w = max(8, int(target_h * crop_aspect))

    # Enforce minimum visible size
    MIN_SIZE = 12  # minimum 12px in any dimension
    target_w = max(MIN_SIZE, min(target_w, OUT_W - 20))
    target_h = max(MIN_SIZE, min(target_h, OUT_H - 20))

    # Pick position
    cx_norm, cy_norm = pick_position(tier, rng)

    # Ensure bbox stays within image
    half_w = (target_w / 2 + 5) / OUT_W
    half_h = (target_h / 2 + 5) / OUT_H
    cx_norm = max(half_w, min(1.0 - half_w, cx_norm))
    cy_norm = max(half_h, min(1.0 - half_h, cy_norm))

    # Paste onto BOTH modalities at same normalized position & size
    rgb_result = paste_crop_onto_bg(rgb_bg, crop.rgb_crop, cx_norm, cy_norm, target_w, target_h)
    ir_result = paste_crop_onto_bg(ir_bg, crop.ir_crop, cx_norm, cy_norm, target_w, target_h)

    bbox_w_norm = target_w / OUT_W
    bbox_h_norm = target_h / OUT_H

    meta = {
        "tier": tier,
        "bg_stem": bg.stem,
        "crop_source": crop.source,
        "crop_stem": crop.stem,
        "paste_cx_norm": round(cx_norm, 6),
        "paste_cy_norm": round(cy_norm, 6),
        "paste_w_norm": round(bbox_w_norm, 6),
        "paste_h_norm": round(bbox_h_norm, 6),
        "target_area_frac": round(target_area_frac, 6),
        "paste_w_px": target_w,
        "paste_h_px": target_h,
    }
    return rgb_result, ir_result, meta


# -- Step 4: Main generation loop --------------------------------------------

def generate_dataset(output_dir: Path, n_samples: int = 10, seed: int = 42):
    """Generate a cut-paste paired RGB+IR dataset."""
    rng = random.Random(seed)
    np.random.seed(seed)

    # Create output structure
    rgb_img_dir = output_dir / "RGB" / "images"
    rgb_lbl_dir = output_dir / "RGB" / "labels"
    ir_img_dir = output_dir / "IR" / "images"
    ir_lbl_dir = output_dir / "IR" / "labels"
    for d in [rgb_img_dir, rgb_lbl_dir, ir_img_dir, ir_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Index assets
    backgrounds = create_backgrounds(max_count=100, stride=200)
    if not backgrounds:
        print("ERROR: No backgrounds found!")
        return

    crops = extract_drone_crops(max_per_source=100, stride=50)
    if not crops:
        print("ERROR: No drone crops found!")
        return

    # Assign tiers
    tiers = []
    n_easy = max(1, int(n_samples * 0.3))
    n_hard = max(1, int(n_samples * 0.3))
    n_medium = n_samples - n_easy - n_hard
    tiers.extend(["easy"] * n_easy)
    tiers.extend(["medium"] * n_medium)
    tiers.extend(["hard"] * n_hard)
    rng.shuffle(tiers)

    metadata = []
    generated = 0

    print(f"\nGenerating {n_samples} paired samples...")
    for idx in range(n_samples):
        tier = tiers[idx]
        crop = rng.choice(crops)
        bg = rng.choice(backgrounds)

        result = generate_sample(crop, bg, tier, rng)
        if result is None:
            print(f"  [{idx}] FAILED - skipping")
            continue

        rgb_img, ir_img, meta = result

        seq = f"{generated:06d}"
        rgb_name = f"cp_{meta['tier']}_{seq}_visible"
        ir_name = f"cp_{meta['tier']}_{seq}_infrared"

        cv2.imwrite(str(rgb_img_dir / f"{rgb_name}.jpg"), rgb_img)
        cv2.imwrite(str(ir_img_dir / f"{ir_name}.jpg"), ir_img)

        label_line = (f"0 {meta['paste_cx_norm']:.6f} {meta['paste_cy_norm']:.6f} "
                      f"{meta['paste_w_norm']:.6f} {meta['paste_h_norm']:.6f}\n")
        (rgb_lbl_dir / f"{rgb_name}.txt").write_text(label_line)
        (ir_lbl_dir / f"{ir_name}.txt").write_text(label_line)

        print(f"  [{generated}] {meta['tier']:6s} | {meta['paste_w_px']}x{meta['paste_h_px']}px "
              f"({meta['target_area_frac']*100:.2f}%) | crop={meta['crop_source']} "
              f"| ir_crop_std={crop.ir_crop.std():.0f}")

        meta["rgb_file"] = f"{rgb_name}.jpg"
        meta["ir_file"] = f"{ir_name}.jpg"
        metadata.append(meta)
        generated += 1

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"config": {"out_w": OUT_W, "out_h": OUT_H, "seed": seed,
                               "n_samples": n_samples},
                   "samples": metadata}, f, indent=2)

    print(f"\nDone! Generated {generated} paired samples -> {output_dir}")


# -- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate cut-paste paired RGB+IR drone eval dataset")
    parser.add_argument("--output", type=str, default="G:/drone/cutpaste_eval",
                        help="Output directory")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Number of paired samples to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_dataset(Path(args.output), n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
