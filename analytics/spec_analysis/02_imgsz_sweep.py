"""
02_imgsz_sweep.py — P/R/F1 + inference latency at multiple imgsz values.

Runs each (model, dataset, imgsz) cell as one inference pass per frame at
conf=0.05 (catches everything), then re-thresholds offline at conf=0.25
(production operating point) and scores with IoP@0.5.

For each cell, records:
  - tp, fp, fn, precision, recall, f1
  - mean_ms_per_frame (wall-clock inference time per image)

Outputs analytics/spec_analysis/results/imgsz_sweep.csv (appends per cell so
a crash mid-run leaves results on disk).

GPU only. Run AFTER any other GPU job finishes — single-GPU sharing breaks here.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RGB model"))
from finetune_selcom import load_gt, iop  # noqa: E402

OUT_DIR = ROOT / "analytics" / "spec_analysis" / "results"
OUT_CSV = OUT_DIR / "imgsz_sweep.csv"

IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}
IOP_THRESH = 0.5
CONF_PROD  = 0.25   # production conf threshold for the headline P/R/F1


MODELS = {
    "old_baseline":   ROOT / "RGB model" / "Yolo26n_trained"               / "weights" / "best_pre_finetune.pt",
    "ft2_1280":       ROOT / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
    "hardneg_v3more": ROOT / "RGB model" / "Yolo26n_hardneg_v3_more"       / "weights" / "best.pt",
    "retrained_v2":   ROOT / "RGB model" / "Yolo26n_retrained_v2"          / "weights" / "best.pt",
}

DATASETS = {
    "selcom_val":  dict(images=Path(r"G:/drone/_finetune_selcom_mixed_ft2/images/val"),
                         labels=Path(r"G:/drone/_finetune_selcom_mixed_ft2/labels/val"),
                         stride=1, max_n=None),
    "dataset_rgb": dict(images=Path(r"G:/drone/dataset/dataset/images/test"),
                         labels=Path(r"G:/drone/dataset/dataset/labels/test"),
                         stride=10, max_n=None),   # ~1721 imgs per cell
    "antiuav":     dict(images=Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
                         labels=Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
                         stride=400, max_n=250),    # sample for time
}

# (model_tag, dataset_tag, imgsz)
GRID = (
    # Big sweep on the two production-track models
    *[("old_baseline", d, sz) for d in ("selcom_val", "dataset_rgb", "antiuav") for sz in (640, 960, 1280, 1920)],
    *[("ft2_1280",     d, sz) for d in ("selcom_val", "dataset_rgb", "antiuav") for sz in (640, 960, 1280, 1920)],
    # Smaller sweep on the others
    *[("hardneg_v3more", d, sz) for d in ("selcom_val", "antiuav") for sz in (640, 1280)],
    *[("retrained_v2",   "selcom_val", sz) for sz in (640, 1280)],
)


def img_list(d: Path, stride=1, max_n=None):
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    if stride > 1: imgs = imgs[::stride]
    if max_n: imgs = imgs[:max_n]
    return imgs


def eval_cell(model, model_tag, ds_tag, ds_cfg, imgsz):
    imgs = img_list(ds_cfg["images"], ds_cfg.get("stride", 1), ds_cfg.get("max_n"))
    if not imgs:
        return None
    tp = fp = fn = 0
    n_frames = 0
    t0 = time.perf_counter()
    inf_times_ms = []
    for ip in imgs:
        frame = cv2.imread(str(ip))
        if frame is None: continue
        h, w = frame.shape[:2]

        t_inf = time.perf_counter()
        r = model.predict(frame, conf=0.05, iou=0.30,
                          imgsz=imgsz, verbose=False, device=0)[0]
        inf_times_ms.append(1000 * (time.perf_counter() - t_inf))

        preds = []
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                if float(r.boxes.conf[j]) < CONF_PROD: continue
                x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                preds.append((x1/w, y1/h, x2/w, y2/h, float(r.boxes.conf[j])))

        gt = load_gt(ds_cfg["labels"] / (ip.stem + ".txt"))
        matched = set()
        for px1, py1, px2, py2, _ in preds:
            best, bi = 0.0, -1
            for j, gb in enumerate(gt):
                s = iop((px1, py1, px2, py2), gb)
                if s > best: best, bi = s, j
            if best >= IOP_THRESH and bi not in matched:
                tp += 1; matched.add(bi)
            else:
                fp += 1
        fn += len(gt) - len(matched)
        n_frames += 1

    total_s = time.perf_counter() - t0
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    mean_ms = float(np.mean(inf_times_ms)) if inf_times_ms else 0.0
    return dict(
        model=model_tag, dataset=ds_tag, imgsz=imgsz,
        n_images=n_frames, tp=tp, fp=fp, fn=fn,
        precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4),
        mean_ms_per_frame=round(mean_ms, 2),
        total_seconds=round(total_s, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=None,
                    help="Subset of cells as 'model:dataset:imgsz' (defaults to full grid)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip cells already present in imgsz_sweep.csv")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = list(GRID)
    if args.cells:
        wanted = set()
        for c in args.cells:
            m, d, sz = c.split(":")
            wanted.add((m, d, int(sz)))
        grid = [(m, d, sz) for (m, d, sz) in grid if (m, d, sz) in wanted]

    done = set()
    if args.resume and OUT_CSV.exists():
        with OUT_CSV.open() as f:
            for row in csv.DictReader(f):
                done.add((row["model"], row["dataset"], int(row["imgsz"])))
        print(f"Resume: {len(done)} cells already done")

    from ultralytics import YOLO
    loaded_models = {}

    fieldnames = ["model", "dataset", "imgsz", "n_images", "tp", "fp", "fn",
                  "precision", "recall", "f1", "mean_ms_per_frame", "total_seconds"]
    is_new = not OUT_CSV.exists()
    f_out = OUT_CSV.open("a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if is_new:
        writer.writeheader(); f_out.flush()

    total = len(grid)
    for i, (m, d, sz) in enumerate(grid, 1):
        if (m, d, sz) in done:
            print(f"[{i}/{total}] {m} x {d} x imgsz={sz}  [skip, already in CSV]")
            continue
        if not MODELS[m].exists():
            print(f"[{i}/{total}] {m} weights missing — skipping")
            continue
        if m not in loaded_models:
            print(f"  loading {m} ...")
            loaded_models[m] = YOLO(str(MODELS[m]))
        print(f"[{i}/{total}] {m} x {d} x imgsz={sz}", flush=True)
        try:
            row = eval_cell(loaded_models[m], m, d, DATASETS[d], sz)
            if row is None:
                print(f"  no images found")
                continue
            writer.writerow(row); f_out.flush()
            print(f"  P={row['precision']:.4f}  R={row['recall']:.4f}  "
                  f"F1={row['f1']:.4f}  ms/frame={row['mean_ms_per_frame']:.1f}  "
                  f"({row['n_images']} imgs, {row['total_seconds']:.0f}s)")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

    f_out.close()
    print(f"\nDONE. Results: {OUT_CSV}")


if __name__ == "__main__":
    main()
