"""
check_gt_alignment.py — Compare RGB vs IR ground truth bounding boxes
to understand the spatial relationship between modalities.

Usage:
    python check_gt_alignment.py
"""

from pathlib import Path
import yaml
from utils import compute_iou, parse_yolo_labels


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["dataset_root"])
    rgb_img_dir = root / cfg["rgb_subdir"] / "images"
    ir_img_dir = root / cfg["ir_subdir"] / "images"
    rgb_lbl_dir = root / cfg["rgb_subdir"] / "labels"
    ir_lbl_dir = root / cfg["ir_subdir"] / "labels"

    rgb_suffix = cfg["rgb_stem_suffix"]
    ir_suffix = cfg["ir_stem_suffix"]
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    # Build paired stem map
    def stem_map(img_dir, suffix):
        out = {}
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() in img_exts:
                s = f.stem
                key = s.replace(suffix, "") if suffix else s
                out[key] = f
        return out

    rgb_map = stem_map(rgb_img_dir, rgb_suffix)
    ir_map = stem_map(ir_img_dir, ir_suffix)
    shared = sorted(set(rgb_map) & set(ir_map))
    print(f"Paired frames: {len(shared)}")

    # We need image dimensions to convert YOLO labels to pixel coords.
    # Since RGB and IR may have different resolutions, read one of each to check.
    import cv2
    sample_rgb = cv2.imread(str(rgb_map[shared[0]]))
    sample_ir = cv2.imread(str(ir_map[shared[0]]))
    print(f"RGB resolution: {sample_rgb.shape[1]}x{sample_rgb.shape[0]}")
    print(f"IR  resolution: {sample_ir.shape[1]}x{sample_ir.shape[0]}")

    ious = []
    both_have_gt = 0
    rgb_only_gt = 0
    ir_only_gt = 0
    neither = 0
    diff_count = 0  # different number of GT boxes

    for stem in shared:
        rgb_lbl = rgb_lbl_dir / (rgb_map[stem].stem + ".txt")
        ir_lbl = ir_lbl_dir / (ir_map[stem].stem + ".txt")

        # Parse as NORMALIZED boxes (don't convert to pixels) so we can
        # compare location regardless of resolution differences
        rgb_boxes_norm = _parse_norm(rgb_lbl)
        ir_boxes_norm = _parse_norm(ir_lbl)

        has_rgb = len(rgb_boxes_norm) > 0
        has_ir = len(ir_boxes_norm) > 0

        if has_rgb and has_ir:
            both_have_gt += 1
            if len(rgb_boxes_norm) != len(ir_boxes_norm):
                diff_count += 1
            # Compute IoU between normalized boxes (greedy best match)
            for rb in rgb_boxes_norm:
                best = max((compute_iou(rb, ib) for ib in ir_boxes_norm), default=0)
                ious.append(best)
        elif has_rgb:
            rgb_only_gt += 1
        elif has_ir:
            ir_only_gt += 1
        else:
            neither += 1

    import numpy as np
    ious = np.array(ious)

    print(f"\n--- GT Presence ---")
    print(f"  Both have GT:    {both_have_gt}")
    print(f"  RGB GT only:     {rgb_only_gt}")
    print(f"  IR GT only:      {ir_only_gt}")
    print(f"  Neither:         {neither}")
    print(f"  Different # boxes: {diff_count}")

    print(f"\n--- Normalized IoU between RGB and IR GT ---")
    print(f"  Pairs compared:  {len(ious)}")
    if len(ious) > 0:
        print(f"  Mean IoU:        {ious.mean():.4f}")
        print(f"  Median IoU:      {np.median(ious):.4f}")
        print(f"  IoU > 0.5:       {(ious > 0.5).sum()} ({(ious > 0.5).mean()*100:.1f}%)")
        print(f"  IoU > 0.3:       {(ious > 0.3).sum()} ({(ious > 0.3).mean()*100:.1f}%)")
        print(f"  IoU > 0.1:       {(ious > 0.1).sum()} ({(ious > 0.1).mean()*100:.1f}%)")
        print(f"  IoU == 0:        {(ious == 0).sum()} ({(ious == 0).mean()*100:.1f}%)")

        # Distribution
        print(f"\n--- IoU Distribution ---")
        for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                        (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                        (0.8, 0.9), (0.9, 1.01)]:
            n = ((ious >= lo) & (ious < hi)).sum()
            print(f"  [{lo:.1f}, {hi:.1f}): {n:>6} ({n/len(ious)*100:>5.1f}%)")


def _parse_norm(label_path):
    """Parse YOLO label file, return normalized (x1,y1,x2,y2) boxes."""
    boxes = []
    if not label_path.exists():
        return boxes
    text = label_path.read_text().strip()
    if not text:
        return boxes
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2))
    return boxes


if __name__ == "__main__":
    main()
