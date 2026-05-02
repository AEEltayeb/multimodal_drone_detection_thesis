"""Quick debug: check calibration + alignment on specific rgb_only TP frames."""

import re
from pathlib import Path

import yaml
from ultralytics import YOLO

from utils import align_detections, compute_iou, parse_yolo_labels
from generate_fusion_data import (
    calibrate_ir_box, compute_per_frame_offset, parse_gt_norm, run_inference,
)


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stem_to_filename(stem, suffix):
    m = re.match(r"(.+)(_f\d+)$", stem)
    if m:
        return m.group(1) + suffix + m.group(2)
    return stem + suffix


def main():
    cfg = load_config()
    root = Path(cfg["dataset_root"])
    rgb_img_dir = root / cfg["rgb_subdir"] / "images"
    ir_img_dir = root / cfg["ir_subdir"] / "images"
    rgb_lbl_dir = root / cfg["rgb_subdir"] / "labels"
    ir_lbl_dir = root / cfg["ir_subdir"] / "labels"
    rgb_suffix = cfg["rgb_stem_suffix"]
    ir_suffix = cfg["ir_stem_suffix"]

    # A few of the rgb_only TP stems from the visualization
    test_stems = [
        "20190925_111757_1_6_f000085",
        "20190925_111757_1_4_f000175",
        "20190925_111757_1_1_f000020",
        "20190925_111757_1_5_f000153",
        "20190925_111757_1_8_f000216",
    ]

    print("Loading models...")
    rgb_model = YOLO(cfg["rgb_weights"])
    ir_model = YOLO(cfg["ir_weights"])

    for stem in test_stems:
        print(f"\n{'='*70}")
        print(f"STEM: {stem}")

        rgb_img = rgb_img_dir / (stem_to_filename(stem, rgb_suffix) + ".jpg")
        ir_img = ir_img_dir / (stem_to_filename(stem, ir_suffix) + ".jpg")
        rgb_lbl = rgb_lbl_dir / (stem_to_filename(stem, rgb_suffix) + ".txt")
        ir_lbl = ir_lbl_dir / (stem_to_filename(stem, ir_suffix) + ".txt")

        if not rgb_img.exists() or not ir_img.exists():
            print(f"  IMAGE NOT FOUND")
            continue

        # GT in normalized coords
        gt_rgb_norm = parse_gt_norm(rgb_lbl)
        gt_ir_norm = parse_gt_norm(ir_lbl)
        print(f"  GT RGB norm: {gt_rgb_norm}")
        print(f"  GT IR norm:  {gt_ir_norm}")

        # Per-frame offset
        offset = compute_per_frame_offset(gt_rgb_norm, gt_ir_norm)
        print(f"  Offset: {offset}")

        # Run inference
        rgb_dets, rgb_w, rgb_h = run_inference(rgb_model, rgb_img, cfg)
        ir_dets, ir_w, ir_h = run_inference(ir_model, ir_img, cfg)

        print(f"  RGB image: {rgb_w}x{rgb_h}, IR image: {ir_w}x{ir_h}")
        print(f"  RGB dets (conf>0.25): {[(round(c,3), [round(v) for v in b]) for b,c in rgb_dets if c>0.25]}")
        print(f"  IR dets (conf>0.25):  {[(round(c,3), [round(v) for v in b]) for b,c in ir_dets if c>0.25]}")

        if offset and ir_dets:
            print(f"\n  Calibrating IR dets into RGB space:")
            ir_calibrated = []
            for box, conf in ir_dets:
                if conf < 0.25:
                    continue
                cal_box = calibrate_ir_box(box, offset, ir_w, ir_h, rgb_w, rgb_h)
                ir_calibrated.append((cal_box, conf))
                print(f"    IR {[round(v) for v in box]} conf={conf:.3f}")
                print(f"    -> calibrated: {[round(v) for v in cal_box]}")

                # Check IoU with each RGB det
                for rb, rc in rgb_dets:
                    if rc > 0.25:
                        iou = compute_iou(cal_box, rb)
                        print(f"    -> IoU with RGB det {[round(v) for v in rb]}: {iou:.4f}")

            # Also check: what does align_detections produce?
            all_ir_cal = [(calibrate_ir_box(b, offset, ir_w, ir_h, rgb_w, rgb_h), c) for b, c in ir_dets]
            matched, rgb_only, ir_only = align_detections(rgb_dets, all_ir_cal, iou_thresh=cfg["alignment_iou"])
            print(f"\n  Alignment result: {len(matched)} matched, {len(rgb_only)} rgb_only, {len(ir_only)} ir_only")
        else:
            print(f"  No offset or no IR dets")


if __name__ == "__main__":
    main()
