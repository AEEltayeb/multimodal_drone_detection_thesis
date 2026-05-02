"""
test_ir_on_rgb.py — Test IR YOLO model on grayscale-converted RGB images.

Samples 1/10 of all RGB datasets, converts to grayscale (3-channel),
runs IR YOLO, and compares against GT labels.

Quick viability check for "dual-model, single-camera" fusion.
"""

import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WORKSPACE = Path(r"C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection")
IR_WEIGHTS = WORKSPACE / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"

# All RGB datasets with labels
RGB_DATASETS = [
    ("rgb_dataset_val",    r"G:\drone\dataset\dataset\images\val",
                           r"G:\drone\dataset\dataset\labels\val"),
    ("rgb_dataset_test",   r"G:\drone\dataset\dataset\images\test",
                           r"G:\drone\dataset\dataset\labels\test"),
    ("antiuav_val_rgb",    r"G:\drone\Anti-UAV-RGBT_yolo_converted\val\RGB\images",
                           r"G:\drone\Anti-UAV-RGBT_yolo_converted\val\RGB\labels"),
    ("antiuav_test_rgb",   r"G:\drone\Anti-UAV-RGBT_yolo_converted\test\RGB\images",
                           r"G:\drone\Anti-UAV-RGBT_yolo_converted\test\RGB\labels"),
    ("svanstrom_rgb",      r"G:\drone\svanstrom_paired\RGB\images",
                           r"G:\drone\svanstrom_paired\RGB\labels"),
]

SAMPLE_RATIO = 0.1  # 1/10
CONF_THRESH = 0.4
IOU_NMS = 0.45
IMGSZ = 640
IOU_MATCH = 0.2
IOP_MATCH = 0.5  # intersection over prediction (for oversized GT)


def parse_gt(label_path, img_w, img_h):
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
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0


def compute_iop(pred, gt):
    x1 = max(pred[0], gt[0]); y1 = max(pred[1], gt[1])
    x2 = min(pred[2], gt[2]); y2 = min(pred[3], gt[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    pa = max(0, pred[2] - pred[0]) * max(0, pred[3] - pred[1])
    return inter / pa if pa > 0 else 0


def match_relaxed(dets, gt_boxes):
    n_det = len(dets)
    n_gt = len(gt_boxes)
    if n_det == 0 or n_gt == 0:
        return 0, n_det, n_gt

    det_matched = [False] * n_det
    gt_matched = [False] * n_gt
    pairs = []
    for di in range(n_det):
        for gi in range(n_gt):
            iou = compute_iou(dets[di], gt_boxes[gi])
            iop = compute_iop(dets[di], gt_boxes[gi])
            score = max(iou, iop)
            if iou >= IOU_MATCH or iop >= IOP_MATCH:
                pairs.append((score, di, gi))
    pairs.sort(reverse=True)
    for _, di, gi in pairs:
        if not det_matched[di] and not gt_matched[gi]:
            det_matched[di] = True
            gt_matched[gi] = True
    tp = sum(det_matched)
    return tp, n_det - tp, n_gt - tp  # tp, fp, fn


def main():
    print("=" * 70)
    print("Testing IR YOLO on grayscale RGB images")
    print(f"  Weights: {IR_WEIGHTS.name}")
    print(f"  Sample: {SAMPLE_RATIO*100:.0f}% of each dataset")
    print(f"  Conf: {CONF_THRESH}, IoU NMS: {IOU_NMS}")
    print("=" * 70)

    model = YOLO(str(IR_WEIGHTS))
    print("Model loaded.\n")

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    for ds_name, img_dir, lbl_dir in RGB_DATASETS:
        img_dir = Path(img_dir)
        lbl_dir = Path(lbl_dir)

        if not img_dir.exists():
            print(f"  [SKIP] {ds_name}: {img_dir} not found")
            continue

        # List and sample images
        all_imgs = sorted([f for f in img_dir.iterdir()
                          if f.suffix.lower() in IMG_EXTS])
        n_total = len(all_imgs)
        random.seed(42)
        sampled = sorted(random.sample(all_imgs, max(1, int(n_total * SAMPLE_RATIO))))
        n_sample = len(sampled)

        print(f"\n  {ds_name}: {n_total:,} images, sampling {n_sample:,}")

        tp_total = 0
        fp_total = 0
        fn_total = 0
        n_drone_frames = 0
        n_detected_frames = 0
        n_tp_frames = 0
        t0 = time.time()

        for idx, img_path in enumerate(sampled):
            # Load and convert to grayscale 3-channel
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.merge([gray, gray, gray])

            # Run IR YOLO on grayscale
            results = model.predict(
                gray_3ch, conf=CONF_THRESH, iou=IOU_NMS,
                imgsz=IMGSZ, max_det=20, verbose=False, device=0
            )

            # Extract detections
            dets = []
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    dets.append([x1, y1, x2, y2, conf])

            # Load GT
            stem = img_path.stem
            label_path = lbl_dir / f"{stem}.txt"
            gt_boxes = parse_gt(label_path, w, h)

            has_gt = len(gt_boxes) > 0
            has_det = len(dets) > 0

            if has_gt:
                n_drone_frames += 1
            if has_det:
                n_detected_frames += 1

            # Match
            det_boxes = [d[:4] for d in dets]
            tp, fp, fn = match_relaxed(det_boxes, gt_boxes)
            tp_total += tp
            fp_total += fp
            fn_total += fn

            if tp > 0:
                n_tp_frames += 1

            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"    [{idx+1}/{n_sample}] {elapsed:.1f}s")

        elapsed = time.time() - t0
        det_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        det_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        det_f1 = 2 * det_p * det_r / (det_p + det_r + 1e-9)

        fr_p = n_tp_frames / n_detected_frames if n_detected_frames > 0 else 0
        fr_r = n_tp_frames / n_drone_frames if n_drone_frames > 0 else 0
        fr_f1 = 2 * fr_p * fr_r / (fr_p + fr_r + 1e-9)

        print(f"    Done in {elapsed:.1f}s ({n_sample/elapsed:.1f} fps)")
        print(f"    Detection level: P={det_p:.4f} R={det_r:.4f} F1={det_f1:.4f}"
              f"  TP={tp_total:,} FP={fp_total:,} FN={fn_total:,}")
        print(f"    Frame level:     P={fr_p:.4f} R={fr_r:.4f} F1={fr_f1:.4f}"
              f"  drone_frames={n_drone_frames:,} det_frames={n_detected_frames:,}"
              f"  tp_frames={n_tp_frames:,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
