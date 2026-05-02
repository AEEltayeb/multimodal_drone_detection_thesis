"""
Verify hypothesis: ir_only detections are being mislabeled because
they're matched against RGB GT after imperfect calibration.

Test: match raw (uncalibrated) ir_only detections against IR GT directly.
"""
import json
import csv
from pathlib import Path

import yaml

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def compute_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    ab = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0

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
            (cx - w/2) * img_w, (cy - h/2) * img_h,
            (cx + w/2) * img_w, (cy + h/2) * img_h,
        ))
    return boxes


print("Loading detections...")
with open("runs/inference_checkpoint.json", "r") as f:
    detections = json.load(f)
print(f"  {len(detections)} frames")

# For each frame, find IR detections that DON'T match any RGB detection
# (these are the "ir_only" candidates), then check them against IR GT
cfg = load_config()
alignment_iou = cfg["alignment_iou"]  # 0.3

bands = [
    ("0.001-0.05", 0.001, 0.05),
    ("0.05-0.10",  0.05,  0.10),
    ("0.10-0.25",  0.10,  0.25),
    ("0.25-0.50",  0.25,  0.50),
    ("0.50-0.75",  0.50,  0.75),
    ("0.75-1.00",  0.75,  1.00),
]
band_counts = {b[0]: {"total": 0, "tp_ir_gt": 0, "tp_rgb_gt": 0} for b in bands}

for stem, d in detections.items():
    rgb_dets = [(tuple(det[:4]), det[4]) for det in d["rgb_dets"]]
    ir_dets = [(tuple(det[:4]), det[4]) for det in d["ir_dets"]]
    ir_w, ir_h = d["ir_w"], d["ir_h"]
    rgb_w, rgb_h = d["rgb_w"], d["rgb_h"]

    # Find ir_only: IR dets that don't match any RGB det after calibration
    # (For simplicity, check raw IR boxes against raw RGB boxes — if no IoU
    #  match, it's ir_only regardless of calibration)
    matched_ir = set()
    for ri, (rb, rc) in enumerate(rgb_dets):
        for ii, (ib, ic) in enumerate(ir_dets):
            if compute_iou(rb, ib) >= alignment_iou:
                matched_ir.add(ii)

    # ir_only = IR dets not matched to any RGB det
    ir_gt = parse_yolo_px(d["ir_lbl"], ir_w, ir_h)
    rgb_gt = parse_yolo_px(d["rgb_lbl"], rgb_w, rgb_h)

    for ii, (ib, ic) in enumerate(ir_dets):
        if ii in matched_ir:
            continue  # this is a "both" detection

        # This is an ir_only detection — check against IR GT (native coords)
        best_ir_iou = max((compute_iou(ib, gt) for gt in ir_gt), default=0.0)
        # Also check against RGB GT (different coord system!) for comparison
        best_rgb_iou = max((compute_iou(ib, gt) for gt in rgb_gt), default=0.0)

        for band_name, lo, hi in bands:
            if lo <= ic < hi:
                band_counts[band_name]["total"] += 1
                if best_ir_iou >= 0.5:
                    band_counts[band_name]["tp_ir_gt"] += 1
                if best_rgb_iou >= 0.5:
                    band_counts[band_name]["tp_rgb_gt"] += 1
                break

print(f"\n{'='*70}")
print(f"  IR-ONLY detections: labeled against IR GT vs RGB GT")
print(f"{'='*70}")
print(f"  {'Band':<15} {'Total':>7} {'TP(IR GT)':>10} {'TP(RGB GT)':>10}  {'IR%':>6} {'RGB%':>6}")
print(f"  {'-'*60}")

total_all = 0
total_ir = 0
total_rgb = 0
for band_name, lo, hi in bands:
    c = band_counts[band_name]
    t = c["total"]
    ir_tp = c["tp_ir_gt"]
    rgb_tp = c["tp_rgb_gt"]
    total_all += t
    total_ir += ir_tp
    total_rgb += rgb_tp
    ir_pct = f"{ir_tp/t*100:.1f}%" if t > 0 else "N/A"
    rgb_pct = f"{rgb_tp/t*100:.1f}%" if t > 0 else "N/A"
    print(f"  {band_name:<15} {t:>7} {ir_tp:>10} {rgb_tp:>10}  {ir_pct:>6} {rgb_pct:>6}")

print(f"  {'-'*60}")
print(f"  {'TOTAL':<15} {total_all:>7} {total_ir:>10} {total_rgb:>10}  "
      f"{total_ir/total_all*100:.1f}% {total_rgb/total_all*100:.1f}%")
print(f"\n  If IR GT match >> RGB GT match → the calibration is losing real detections")
