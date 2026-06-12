"""
predictor.py — Model inference for label generation.

Wraps YOLO inference to produce YOLO-format labels and optional review images.
Extracted from auto_label_ir.py for use by the GUI launcher.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Confidence tiers
TIERS = {
    "high":   (0.80, 1.00),
    "medium": (0.50, 0.80),
    "low":    (0.25, 0.50),
}

TIER_COLORS = {
    "high":   (0, 255, 0),
    "medium": (0, 255, 255),
    "low":    (0, 0, 255),
    "none":   (128, 128, 128),
}


def classify_tier(conf: float) -> str:
    """Classify a detection confidence into a tier."""
    for tier, (lo, hi) in TIERS.items():
        if lo <= conf < hi or (tier == "high" and conf >= hi):
            return tier
    return "low"


def draw_detections(img: np.ndarray, boxes: list, img_w: int, img_h: int) -> np.ndarray:
    """Draw bounding boxes on image with tier-colored labels."""
    vis = img.copy()
    for cls, xc, yc, w, h, conf in boxes:
        tier = classify_tier(conf)
        color = TIER_COLORS.get(tier, (255, 255, 255))

        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{conf:.2f} [{tier[0].upper()}]"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


def run_prediction(weights: str, source_dir: Path, output_dir: Path,
                   conf_threshold: float = 0.25, save_review: bool = True,
                   image_size: int = 640, device: str = "auto",
                   progress_callback=None):
    """Run model inference and write YOLO labels + optional review images.

    Args:
        weights: Path to model weights (.pt)
        source_dir: Directory of images to predict on
        output_dir: Output directory for labels/ and review/
        conf_threshold: Minimum detection confidence
        save_review: Whether to generate annotated review images
        image_size: YOLO inference image size
        device: CUDA device or 'cpu'
        progress_callback: Optional fn(current, total, message) for GUI updates

    Returns:
        dict with summary statistics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return None

    print(f"\n{'=' * 60}")
    print(f"  PREDICT: Model Inference")
    print(f"  Weights:    {weights}")
    print(f"  Source:     {source_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Conf threshold: {conf_threshold}")
    print(f"{'=' * 60}\n")

    model = YOLO(weights)

    # Auto-detect device
    if device == "auto":
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device} ({'CUDA' if device == '0' else 'CPU'})")

    # Collect images
    images = sorted([f for f in source_dir.iterdir()
                     if f.suffix.lower() in IMG_EXTS])
    print(f"  Found {len(images)} images\n")

    if not images:
        print("  [ERROR] No images found!")
        return None

    # Create output dirs
    label_dir = output_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    review_dir = None
    tier_subdirs = {}
    if save_review:
        review_dir = output_dir / "review"
        for tier in list(TIERS.keys()) + ["none"]:
            d = review_dir / tier
            d.mkdir(parents=True, exist_ok=True)
            tier_subdirs[tier] = d

    # Stats
    manifest = []
    tier_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
    total_detections = 0
    skipped = 0

    for i, img_path in enumerate(images):
        # Skip images we already predicted (resume support)
        existing_label = label_dir / f"{img_path.stem}.txt"
        if existing_label.exists():
            skipped += 1
            continue

        # Run inference
        try:
            results = model.predict(
                source=str(img_path),
                conf=conf_threshold,
                imgsz=image_size,
                verbose=False,
                device=device,
            )
            result = results[0]
        except Exception as e:
            print(f"  WARNING: Inference failed for {img_path.name}: {e}")
            # Write empty label so we skip on resume
            with open(label_dir / f"{img_path.stem}.txt", "w") as f:
                pass
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: Image read error {img_path.name}, skipping")
            with open(label_dir / f"{img_path.stem}.txt", "w") as f:
                pass
            skipped += 1
            continue
        img_h, img_w = img.shape[:2]

        # Extract detections
        boxes_data = []
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                xywhn = box.xywhn[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                boxes_data.append((cls, float(xywhn[0]), float(xywhn[1]),
                                   float(xywhn[2]), float(xywhn[3]), conf))

        # Determine frame tier
        if boxes_data:
            max_conf = max(b[5] for b in boxes_data)
            frame_tier = classify_tier(max_conf)
        else:
            max_conf = 0.0
            frame_tier = "none"

        tier_counts[frame_tier] += 1
        total_detections += len(boxes_data)

        # Write YOLO label
        with open(label_dir / f"{img_path.stem}.txt", "w") as f:
            for cls, xc, yc, w, h, conf in boxes_data:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # Generate review image
        if save_review and review_dir:
            vis = draw_detections(img, boxes_data, img_w, img_h)
            info = f"{img_path.name}  |  {len(boxes_data)} det  |  tier: {frame_tier.upper()}"
            cv2.putText(vis, info, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            save_path = tier_subdirs[frame_tier] / f"{img_path.stem}.jpg"
            cv2.imwrite(str(save_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # Manifest entry
        manifest.append({
            "stem": img_path.stem,
            "num_detections": len(boxes_data),
            "max_confidence": round(max_conf, 4),
            "tier": frame_tier,
            "confidences": [round(b[5], 4) for b in boxes_data],
        })

        # Progress
        if progress_callback:
            progress_callback(i + 1, len(images),
                              f"Det={total_detections} H={tier_counts['high']} "
                              f"M={tier_counts['medium']} L={tier_counts['low']}")
        if (i + 1) % 100 == 0 or (i + 1) == len(images):
            print(f"  [{i+1}/{len(images)}] "
                  f"det={total_detections}  "
                  f"H={tier_counts['high']} M={tier_counts['medium']} "
                  f"L={tier_counts['low']} N={tier_counts['none']}")

    # Save manifest
    manifest_path = output_dir / "labeling_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "weights": str(weights),
            "conf_threshold": conf_threshold,
            "total_images": len(images),
            "total_detections": total_detections,
            "skipped_existing": skipped,
            "tier_counts": tier_counts,
            "frames": manifest,
        }, f, indent=2)

    summary = {
        "total_images": len(images),
        "total_detections": total_detections,
        "skipped": skipped,
        "tier_counts": tier_counts,
        "label_dir": str(label_dir),
        "manifest_path": str(manifest_path),
    }

    print(f"\n{'=' * 60}")
    print(f"  DONE: {total_detections} detections across {len(images)} images")
    print(f"  Tiers: H={tier_counts['high']} M={tier_counts['medium']} "
          f"L={tier_counts['low']} N={tier_counts['none']}")
    print(f"  Labels:   {label_dir}")
    print(f"  Manifest: {manifest_path}")
    if save_review:
        print(f"  Review:   {review_dir}")
    print(f"{'=' * 60}\n")

    return summary
