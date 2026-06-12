"""
conf_sweep_ft2.py — Confidence-threshold sweep for ft2_1280 on selcom val.

Runs inference ONCE at conf=0.01 (catches everything), then re-thresholds
offline at many confidence levels and reports P/R/F1 + F1-optimal threshold.

Usage:
    python "training/conf_sweep_ft2.py"
    python "training/conf_sweep_ft2.py" --imgsz 1280 --weights "<other.pt>"
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RGB model"))
from finetune_selcom import load_gt, iop   # noqa: E402

DEFAULT_WEIGHTS = ROOT / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt"
DEFAULT_IMAGES  = Path(r"G:/drone/_finetune_selcom_mixed_ft2/images/val")
DEFAULT_LABELS  = Path(r"G:/drone/_finetune_selcom_mixed_ft2/labels/val")

CONF_LEVELS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35,
               0.40, 0.50, 0.60, 0.70]
IOP_THRESH  = 0.5
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}


def score(preds_all, gt, conf_thr):
    preds = [(p[0], p[1], p[2], p[3]) for p in preds_all if p[4] >= conf_thr]
    matched, tp, fp = set(), 0, 0
    for pr in preds:
        best, bi = 0.0, -1
        for j, g in enumerate(gt):
            s = iop(pr, g)
            if s > best:
                best, bi = s, j
        if best >= IOP_THRESH and bi not in matched:
            tp += 1; matched.add(bi)
        else:
            fp += 1
    return tp, fp, len(gt) - len(matched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--images",  default=str(DEFAULT_IMAGES))
    ap.add_argument("--labels",  default=str(DEFAULT_LABELS))
    ap.add_argument("--imgsz",   type=int, default=1280)
    ap.add_argument("--name",    default="ft2_1280_selcom_val")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    imgs = sorted(Path(args.images).iterdir())
    imgs = [p for p in imgs if p.suffix.lower() in IMG_EXTS]
    print(f"Sweeping {len(CONF_LEVELS)} conf levels  ×  {len(imgs)} images  @ imgsz={args.imgsz}")
    print(f"Weights: {args.weights}")

    counters = {c: [0, 0, 0] for c in CONF_LEVELS}
    t0 = time.perf_counter()

    for i, ip in enumerate(imgs):
        f = cv2.imread(str(ip))
        if f is None:
            continue
        h, w = f.shape[:2]
        r = model.predict(f, conf=0.01, iou=0.45,
                          imgsz=args.imgsz, verbose=False, device=0)[0]
        preds = []
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                preds.append((x1/w, y1/h, x2/w, y2/h, float(r.boxes.conf[j])))
        gt = load_gt(Path(args.labels) / (ip.stem + ".txt"))
        for c in CONF_LEVELS:
            tp, fp, fn = score(preds, gt, c)
            counters[c][0] += tp; counters[c][1] += fp; counters[c][2] += fn
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(imgs)} ({time.perf_counter()-t0:.0f}s)")

    # Output
    rows = []
    for c in CONF_LEVELS:
        tp, fp, fn = counters[c]
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        rows.append((c, tp, fp, fn, p, r, f1))

    out_dir = ROOT / "runs" / "conf_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.name}.csv"
    with out_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["conf", "tp", "fp", "fn", "precision", "recall", "f1"])
        for row in rows:
            w.writerow([f"{row[0]:.2f}", *row[1:4],
                        f"{row[4]:.4f}", f"{row[5]:.4f}", f"{row[6]:.4f}"])

    print(f"\n{'='*72}\nCONF SWEEP — {args.name}\n{'='*72}")
    print(f"{'conf':>6s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  "
          f"{'P':>7s}  {'R':>7s}  {'F1':>7s}")
    print("-" * 56)
    best = max(rows, key=lambda r: r[6])
    for row in rows:
        marker = "  <-- best F1" if row is best else ""
        print(f"{row[0]:>6.2f}  {row[1]:>5d}  {row[2]:>5d}  {row[3]:>5d}  "
              f"{row[4]:>7.4f}  {row[5]:>7.4f}  {row[6]:>7.4f}{marker}")
    print(f"\nF1-optimal conf: {best[0]:.2f}  (P={best[4]:.3f}, R={best[5]:.3f}, F1={best[6]:.3f})")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
