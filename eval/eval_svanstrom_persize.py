"""
eval_svanstrom_persize.py — Per-size drone P/R/F1 on Svanstrom for every RGB model.

Runs each RGB model on the Svanstrom RGB split and buckets TP / FP / FN by GT
box area fraction. Writes:

  eval/results/svanstrom_persize/<model>_persize.csv
      Rows: (model, category, size_bucket, TP, FP, FN, precision, recall, f1, n_gt)
  eval/results/svanstrom_persize/summary.csv
      Aggregated long-format across all models.

Confuser categories (BIRD / AIRPLANE / HELICOPTER) have no drone GT, so they
contribute only FP counts bucketed by det box area.

Usage:
  python eval/eval_svanstrom_persize.py
  python eval/eval_svanstrom_persize.py --models baseline retrained_v2 selcom_1280
  python eval/eval_svanstrom_persize.py --imgsz 1280 --conf 0.25 --iop-thr 0.5
"""

from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from datasets import PairedDataset, load_config  # noqa: E402
from metrics import (  # noqa: E402
    SIZE_BUCKETS, classify_size, score_per_size, iou_iop,
)


MODELS = {
    "baseline":       REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt",
    "hardneg_v3more": REPO / "RGB model" / "Yolo26n_hardneg_v3_more" / "weights" / "best.pt",
    "retrained_v2":   REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt",
    "selcom_1280":    REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
    "selcom_960":     REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
    "selcom_640":     REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
}

MODEL_IMGSZ_DEFAULT = {
    "selcom_960": 960,
    "selcom_640": 640,
    # everything else uses --imgsz arg (default 1280 for Svanstrom)
}


def precision(tp, fp): return tp / (tp + fp) if (tp + fp) > 0 else 0.0
def recall(tp, fn):    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
def f1(p, r):          return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def auto_stride(n: int, cap: int = 5000, floor: int = 2000) -> int:
    """Cap eval at ~cap stems; below floor use stride 1."""
    if n < floor:
        return 1
    return max(1, -(-n // cap))


def main():
    ap = argparse.ArgumentParser(description="Svanstrom per-size eval")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="YOLO inference imgsz (Svanstrom needs 1280 per project memory)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iop-thr", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--output-dir", type=str,
                    default=str(EVAL_DIR / "results" / "svanstrom_persize"))
    ap.add_argument("--stride", type=int, default=0,
                    help="0 = auto (cap 5k, floor 2k); 1 = every frame; N>1 = every Nth")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    ds = PairedDataset(cfg["datasets"]["svanstrom"])
    stems = ds.list_stems()
    n_total = len(stems)
    stride = args.stride if args.stride > 0 else auto_stride(n_total)
    if stride > 1:
        stems = stems[::stride]
    print(f"Loaded Svanstrom: {n_total} stems, stride={stride} -> {len(stems)} frames")

    all_rows: list[dict] = []

    for mname in args.models:
        if mname not in MODELS:
            print(f"  SKIP unknown model: {mname}")
            continue
        wpath = MODELS[mname]
        if not wpath.exists():
            print(f"  SKIP missing weights: {wpath}")
            continue

        imgsz = MODEL_IMGSZ_DEFAULT.get(mname, args.imgsz)
        print(f"\n=== Model: {mname}  (imgsz={imgsz}, conf={args.conf}) ===")
        model = YOLO(str(wpath))

        # Per-category, per-size counters
        # cat -> bucket -> {tp, fp, fn, n_gt}
        counts = {
            cat: {b: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0} for b in SIZE_BUCKETS}
            for cat in ("DRONE", "BIRD", "AIRPLANE", "HELICOPTER")
        }

        t0 = time.time()
        for i, stem in enumerate(stems):
            frame = ds.load_frame(stem)
            if frame is None:
                continue
            img = frame["rgb_img"]
            gts = frame["rgb_gt"]
            w, h = frame["rgb_w"], frame["rgb_h"]
            cat = frame["category"]
            if cat not in counts:
                continue

            res = model.predict(img, imgsz=imgsz, conf=args.conf,
                                device=args.device, verbose=False)
            r0 = res[0]
            dets = []
            if r0.boxes is not None and len(r0.boxes) > 0:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                confs = r0.boxes.conf.cpu().numpy()
                dets = [(tuple(map(float, b)), float(c)) for b, c in zip(xyxy, confs)]

            # Count GTs per size
            for g in gts:
                counts[cat][classify_size(g, w, h)]["n_gt"] += 1

            # DRONE frames have real GTs -> proper TP/FP/FN bucketing.
            # Confuser frames (BIRD/AIRPLANE/HELI) have gts=[]; every det is an FP
            # bucketed by det size.
            ps = score_per_size(dets, gts, w, h, iou_thr=0.5, iop_thr=args.iop_thr)
            for b in SIZE_BUCKETS:
                counts[cat][b]["tp"] += ps["iop"][b]["tp"]
                counts[cat][b]["fp"] += ps["iop"][b]["fp"]
                counts[cat][b]["fn"] += ps["iop"][b]["fn"]

            if (i + 1) % 200 == 0:
                print(f"  .. {i+1}/{len(stems)}  ({time.time()-t0:.0f}s)")

        # Emit per-model CSV
        rows = []
        for cat in counts:
            for b in SIZE_BUCKETS:
                c = counts[cat][b]
                p = precision(c["tp"], c["fp"])
                r = recall(c["tp"], c["fn"])
                rows.append({
                    "model": mname, "category": cat, "size_bucket": b,
                    "TP": c["tp"], "FP": c["fp"], "FN": c["fn"], "n_gt": c["n_gt"],
                    "precision": round(p, 4), "recall": round(r, 4),
                    "f1": round(f1(p, r), 4),
                })
        csv_path = out_dir / f"{mname}_persize.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  -> {csv_path}")
        all_rows.extend(rows)

    # Combined summary
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [
            "model", "category", "size_bucket", "TP", "FP", "FN", "n_gt", "precision", "recall", "f1"
        ])
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {summary_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
