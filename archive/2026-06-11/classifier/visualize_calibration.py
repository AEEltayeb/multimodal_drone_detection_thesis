"""
Visualize calibration: draw IR detection boxes (calibrated to RGB space)
overlaid on RGB images, alongside RGB detections and GT boxes.

Shows 20 random POSITIVE frames where both models have detections.
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_gt_norm(label_path):
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    for line in p.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append(tuple(map(float, parts[1:5])))
    return boxes


def parse_yolo_px(label_path, img_w, img_h):
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    for line in p.read_text().strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        boxes.append((
            int((cx - w/2) * img_w), int((cy - h/2) * img_h),
            int((cx + w/2) * img_w), int((cy + h/2) * img_h),
        ))
    return boxes


def compute_per_frame_offset(gt_rgb_norm, gt_ir_norm):
    if not gt_rgb_norm or not gt_ir_norm:
        return None
    r = gt_rgb_norm[0]
    i = gt_ir_norm[0]
    if i[2] <= 0 or i[3] <= 0:
        return None
    return (r[0] - i[0], r[1] - i[1], r[2] / i[2], r[3] / i[3])


def calibrate_ir_box(box_xyxy, offset, ir_w, ir_h, rgb_w, rgb_h):
    dcx, dcy, sw, sh = offset
    x1, y1, x2, y2 = box_xyxy
    cx_n = ((x1 + x2) / 2) / ir_w
    cy_n = ((y1 + y2) / 2) / ir_h
    w_n = (x2 - x1) / ir_w
    h_n = (y2 - y1) / ir_h
    new_cx = cx_n + dcx
    new_cy = cy_n + dcy
    new_w = w_n * sw
    new_h = h_n * sh
    rx1 = int((new_cx - new_w / 2) * rgb_w)
    ry1 = int((new_cy - new_h / 2) * rgb_h)
    rx2 = int((new_cx + new_w / 2) * rgb_w)
    ry2 = int((new_cy + new_h / 2) * rgb_h)
    return (rx1, ry1, rx2, ry2)


def compute_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0


def resolve_image_path(stem, cfg, modality):
    """Find the actual image file for a stem."""
    if modality == "rgb":
        img_dir = Path(cfg["dataset_root"]) / cfg["rgb_subdir"] / "images"
        suffix = cfg["rgb_stem_suffix"]
    else:
        img_dir = Path(cfg["dataset_root"]) / cfg["ir_subdir"] / "images"
        suffix = cfg["ir_stem_suffix"]
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = img_dir / f"{stem}{suffix}{ext}"
        if p.exists():
            return p
    return None


def main():
    cfg = load_config()
    out_dir = Path("runs/calibration_vis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading detections...")
    with open("runs/inference_checkpoint.json", "r") as f:
        detections = json.load(f)
    print(f"  {len(detections)} frames")

    # Find positive frames where both models have high-conf detections
    good_frames = []
    for stem, d in detections.items():
        gt_rgb = parse_gt_norm(d["rgb_lbl"])
        gt_ir = parse_gt_norm(d["ir_lbl"])
        if not gt_rgb or not gt_ir:
            continue
        rgb_confs = [det[4] for det in d["rgb_dets"]]
        ir_confs = [det[4] for det in d["ir_dets"]]
        if not rgb_confs or not ir_confs:
            continue
        if max(rgb_confs) > 0.3 and max(ir_confs) > 0.3:
            good_frames.append(stem)

    print(f"  {len(good_frames)} frames with high-conf detections in both modalities")
    random.seed(42)
    selected = random.sample(good_frames, min(20, len(good_frames)))

    for idx, stem in enumerate(selected):
        d = detections[stem]
        img_w, img_h = d["rgb_w"], d["rgb_h"]
        ir_w, ir_h = d["ir_w"], d["ir_h"]

        # Load RGB image
        rgb_path = resolve_image_path(stem, cfg, "rgb")
        if rgb_path is None:
            print(f"  [SKIP] {stem}: RGB image not found")
            continue
        img = cv2.imread(str(rgb_path))
        if img is None:
            print(f"  [SKIP] {stem}: failed to load RGB image")
            continue

        # Compute calibration offset from GT
        gt_rgb = parse_gt_norm(d["rgb_lbl"])
        gt_ir = parse_gt_norm(d["ir_lbl"])
        offset = compute_per_frame_offset(gt_rgb, gt_ir)

        # Parse GT boxes in RGB pixel space
        gt_boxes = parse_yolo_px(d["rgb_lbl"], img_w, img_h)

        # RGB detections (raw, in RGB pixel space)
        rgb_dets = [(tuple(int(v) for v in det[:4]), det[4]) for det in d["rgb_dets"]]

        # IR detections: raw and calibrated
        ir_dets_raw = [(tuple(int(v) for v in det[:4]), det[4]) for det in d["ir_dets"]]

        if offset:
            ir_dets_cal = [
                (calibrate_ir_box(tuple(det[:4]), offset, ir_w, ir_h, img_w, img_h), det[4])
                for det in d["ir_dets"]
            ]
        else:
            ir_dets_cal = [(b, c) for b, c in ir_dets_raw]  # uncalibrated fallback

        # === Draw on RGB image ===
        canvas = img.copy()

        # GT boxes: thick blue
        for gt in gt_boxes:
            cv2.rectangle(canvas, (gt[0], gt[1]), (gt[2], gt[3]),
                         (255, 150, 0), 3)  # blue
            cv2.putText(canvas, "GT", (gt[0], gt[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

        # RGB detections: green (only conf > 0.1 to reduce clutter)
        for box, conf in rgb_dets:
            if conf < 0.1:
                continue
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]),
                         (0, 255, 0), 2)  # green
            cv2.putText(canvas, f"RGB {conf:.2f}", (box[0], box[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calibrated IR detections: red (only conf > 0.1)
        for box, conf in ir_dets_cal:
            if conf < 0.1:
                continue
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]),
                         (0, 0, 255), 2)  # red
            cv2.putText(canvas, f"IR_cal {conf:.2f}", (box[0], box[3] + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Compute IoU between calibrated IR and GT / RGB
        info_lines = [f"Stem: {stem}", f"Offset: {offset}"]
        for i, (ir_box, ir_conf) in enumerate(ir_dets_cal):
            if ir_conf < 0.1:
                continue
            # IoU with GT
            best_gt_iou = max((compute_iou(ir_box, gt) for gt in gt_boxes), default=0)
            # IoU with best RGB det
            best_rgb_iou = max((compute_iou(ir_box, rb) for rb, rc in rgb_dets if rc > 0.1),
                               default=0)
            info_lines.append(
                f"IR#{i} conf={ir_conf:.2f}: IoU(GT)={best_gt_iou:.3f}, IoU(RGB)={best_rgb_iou:.3f}")

        # Draw info text
        y_text = 25
        for line in info_lines:
            cv2.putText(canvas, line, (10, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(canvas, line, (10, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_text += 20

        # Legend
        h = canvas.shape[0]
        cv2.putText(canvas, "Blue=GT  Green=RGB det  Red=IR calibrated",
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, "Blue=GT  Green=RGB det  Red=IR calibrated",
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        out_path = out_dir / f"{idx+1:02d}_{stem}.jpg"
        cv2.imwrite(str(out_path), canvas)
        print(f"  [{idx+1}/20] Saved {out_path.name}")
        for line in info_lines[1:]:
            print(f"          {line}")

    print(f"\nDone. Visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
