"""
sweep_ir_on_rgb.py — Find optimal IR model confidence for grayscale RGB.

Runs IR YOLO at conf=0.001 on sampled RGB images (converted to grayscale),
caches all detections, then sweeps confidence thresholds to find best F1.

Usage:
    python -u sweep_ir_on_rgb.py
"""

import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WORKSPACE = Path(r"C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection")
IR_WEIGHTS = WORKSPACE / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"

RGB_DATASETS = [
    ("rgb_dataset_test",   r"G:\drone\dataset\dataset\images\test",
                           r"G:\drone\dataset\dataset\labels\test"),
    ("antiuav_test_rgb",   r"G:\drone\Anti-UAV-RGBT_yolo_converted\test\RGB\images",
                           r"G:\drone\Anti-UAV-RGBT_yolo_converted\test\RGB\labels"),
    ("svanstrom_rgb",      r"G:\drone\svanstrom_paired\RGB\images",
                           r"G:\drone\svanstrom_paired\RGB\labels"),
]

SAMPLE_RATIO = 0.1
RAW_CONF = 0.001   # capture everything
IOU_NMS = 0.45
IMGSZ = 640
IOU_MATCH = 0.2
IOP_MATCH = 0.5
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CACHE_DIR = WORKSPACE / "classifier" / "runs" / "reliability" / "fusion" / "ir_on_rgb_cache"


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
    det_m = [False] * n_det
    gt_m = [False] * n_gt
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
        if not det_m[di] and not gt_m[gi]:
            det_m[di] = True
            gt_m[gi] = True
    tp = sum(det_m)
    return tp, n_det - tp, n_gt - tp


def run_inference(model):
    """Run IR model on sampled grayscale RGB images, cache all detections."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_frames = {}  # {ds_name: [{stem, dets, gt_boxes, has_gt}, ...]}

    for ds_name, img_dir, lbl_dir in RGB_DATASETS:
        cache_path = CACHE_DIR / f"{ds_name}.json"
        if cache_path.exists():
            print(f"\n  [CACHED] {ds_name}")
            with open(cache_path) as f:
                all_frames[ds_name] = json.load(f)
            print(f"    {len(all_frames[ds_name]):,} frames loaded from cache")
            continue

        img_dir = Path(img_dir)
        lbl_dir = Path(lbl_dir)
        if not img_dir.exists():
            print(f"  [SKIP] {ds_name}")
            continue

        all_imgs = sorted([f for f in img_dir.iterdir()
                          if f.suffix.lower() in IMG_EXTS])
        random.seed(42)
        sampled = sorted(random.sample(all_imgs, max(1, int(len(all_imgs) * SAMPLE_RATIO))))

        print(f"\n  {ds_name}: {len(all_imgs):,} total, sampling {len(sampled):,}")
        frames = []
        t0 = time.time()

        for idx, img_path in enumerate(sampled):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_3ch = cv2.merge([gray, gray, gray])

            results = model.predict(
                gray_3ch, conf=RAW_CONF, iou=IOU_NMS,
                imgsz=IMGSZ, max_det=20, verbose=False, device=0
            )

            dets = []
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    dets.append([x1, y1, x2, y2, conf])

            label_path = lbl_dir / f"{img_path.stem}.txt"
            gt_boxes = parse_gt(label_path, w, h)

            frames.append({
                "stem": img_path.stem,
                "dets": dets,
                "gt": [list(map(float, g)) for g in gt_boxes],
            })

            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                fps = (idx + 1) / elapsed
                print(f"    [{idx+1}/{len(sampled)}] {elapsed:.1f}s ({fps:.1f} fps)")

        elapsed = time.time() - t0
        print(f"    Done: {len(frames):,} frames in {elapsed:.1f}s")

        with open(cache_path, "w") as f:
            json.dump(frames, f)
        all_frames[ds_name] = frames

    return all_frames


def sweep_thresholds(all_frames):
    """Sweep confidence thresholds and find optimal F1."""
    thresholds = np.arange(0.05, 0.95, 0.05)

    print(f"\n{'='*80}")
    print("CONFIDENCE SWEEP (IR model on grayscale RGB)")
    print(f"{'='*80}")
    print(f"\n  {'conf':>6s}  {'P':>7s} {'R':>7s} {'F1':>7s}  {'TP':>6s} {'FP':>6s} {'FN':>6s}  {'Fr_P':>6s} {'Fr_R':>6s} {'Fr_F1':>6s}")
    print(f"  {'-'*75}")

    best_f1 = -1
    best_conf = 0

    for conf_t in thresholds:
        tp_all, fp_all, fn_all = 0, 0, 0
        n_drone_fr, n_det_fr, n_tp_fr = 0, 0, 0

        for ds_name, frames in all_frames.items():
            for frame in frames:
                dets = [d for d in frame["dets"] if d[4] >= conf_t]
                gt = frame["gt"]

                has_gt = len(gt) > 0
                has_det = len(dets) > 0

                if has_gt:
                    n_drone_fr += 1
                if has_det:
                    n_det_fr += 1

                det_boxes = [d[:4] for d in dets]
                tp, fp, fn = match_relaxed(det_boxes, gt)
                tp_all += tp
                fp_all += fp
                fn_all += fn
                if tp > 0:
                    n_tp_fr += 1

        det_p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
        det_r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
        det_f1 = 2 * det_p * det_r / (det_p + det_r + 1e-9)

        fr_p = n_tp_fr / n_det_fr if n_det_fr > 0 else 0
        fr_r = n_tp_fr / n_drone_fr if n_drone_fr > 0 else 0
        fr_f1 = 2 * fr_p * fr_r / (fr_p + fr_r + 1e-9)

        marker = " <-- BEST" if det_f1 > best_f1 else ""
        print(f"  {conf_t:>6.2f}  {det_p:>7.4f} {det_r:>7.4f} {det_f1:>7.4f}"
              f"  {tp_all:>6,} {fp_all:>6,} {fn_all:>6,}"
              f"  {fr_p:>6.3f} {fr_r:>6.3f} {fr_f1:>6.3f}{marker}")

        if det_f1 > best_f1:
            best_f1 = det_f1
            best_conf = conf_t

    print(f"\n  OPTIMAL: conf={best_conf:.2f}, F1={best_f1:.4f}")
    print(f"\n  Recommendation:")
    print(f"    Real IR input:      conf = 0.40")
    print(f"    Grayscale RGB input: conf = {best_conf:.2f}")

    # Per-dataset at optimal threshold
    print(f"\n  Per-dataset at conf={best_conf:.2f}:")
    print(f"  {'dataset':<20s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
    print(f"  {'-'*60}")
    for ds_name, frames in all_frames.items():
        tp, fp, fn = 0, 0, 0
        for frame in frames:
            dets = [d for d in frame["dets"] if d[4] >= best_conf]
            det_boxes = [d[:4] for d in dets]
            t, f, m = match_relaxed(det_boxes, frame["gt"])
            tp += t; fp += f; fn += m
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r + 1e-9)
        print(f"  {ds_name:<20s} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {tp:>6,} {fp:>6,} {fn:>6,}")


def main():
    print("=" * 70)
    print("IR on Grayscale RGB — Confidence Sweep")
    print(f"  Model: {IR_WEIGHTS.name}")
    print(f"  Raw conf: {RAW_CONF} (capture all, sweep post-hoc)")
    print(f"  Sample: {SAMPLE_RATIO*100:.0f}%")
    print("=" * 70)

    model = YOLO(str(IR_WEIGHTS))
    print("Model loaded.\n")

    all_frames = run_inference(model)
    sweep_thresholds(all_frames)
    print("\nDone.")


if __name__ == "__main__":
    main()
