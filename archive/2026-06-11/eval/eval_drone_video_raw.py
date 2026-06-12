"""
eval_drone_video_raw.py — Raw RGB detector eval on drone-video tests
(no classifier, no cascade). Mirrors Step 1 of eval_1000_results.md.

Sampled subset (stride) for quick model comparison.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import cv2

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from metrics import score_detections, compute_prf  # noqa: E402
from datasets import read_yolo_labels  # noqa: E402
from eval_detector import MODEL_REGISTRY  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DRONE_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
SCORING = "iop"
IOP_THR = 0.5


def enumerate_frames():
    frames = []
    for cat in ("drone", "birds", "airplanes", "helicopters"):
        cat_root = DRONE_ROOT / cat
        if not cat_root.exists():
            continue
        for cdir in sorted(cat_root.iterdir()):
            if not cdir.is_dir():
                continue
            img_dir = cdir / "images" / "test"
            lbl_dir = cdir / "labels" / "test"
            if not img_dir.exists():
                continue
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                frames.append((cat, cdir.name, p, lbl_dir / f"{p.stem}.txt"))
    return frames


def eval_model(name, weights, imgsz, conf, frames):
    print(f"\n{'='*70}")
    print(f"  {name}  imgsz={imgsz} conf={conf}  ({len(frames)} frames)")
    print(f"{'='*70}")
    model = YOLO(str(weights))
    tp = fp = fn = 0
    drone_tp = drone_fp = drone_fn = 0
    conf_tp = conf_fp = 0  # confuser categories
    t0 = time.perf_counter()
    for i, (cat, clip, img_path, lbl_path) in enumerate(frames):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        r = model.predict(frame, conf=conf, iou=0.45, imgsz=imgsz,
                          verbose=False, device=0)[0]
        preds = []
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                preds.append(((x1/w, y1/h, x2/w, y2/h), float(r.boxes.conf[j])))

        if cat == "drone":
            gt_boxes = []
            if lbl_path.exists():
                for ln in lbl_path.read_text().splitlines():
                    parts = ln.strip().split()
                    if len(parts) < 5 or int(parts[0]) != 0:
                        continue
                    cx, cy, bw, bh = (float(parts[1]), float(parts[2]),
                                       float(parts[3]), float(parts[4]))
                    gt_boxes.append((cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2))
            _tp, _fp, _fn = score_detections(preds, gt_boxes, rule=SCORING, iop_thr=IOP_THR)
            tp += _tp; fp += _fp; fn += _fn
            drone_tp += _tp; drone_fp += _fp; drone_fn += _fn
        else:
            # confuser frames — every detection is a FP
            fp += len(preds)
            conf_fp += len(preds)
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(frames)}  ({time.perf_counter()-t0:.0f}s)")
    elapsed = time.perf_counter() - t0
    _prf = compute_prf(tp, fp, fn); P, R, F1 = _prf["precision"], _prf["recall"], _prf["f1"]
    _dprf = compute_prf(drone_tp, drone_fp, drone_fn); dP, dR, dF1 = _dprf["precision"], _dprf["recall"], _dprf["f1"]
    print(f"  Combined: TP={tp} FP={fp} FN={fn}  P={P:.4f} R={R:.4f} F1={F1:.4f}")
    print(f"  Drone only: TP={drone_tp} FP={drone_fp} FN={drone_fn}  P={dP:.4f} R={dR:.4f} F1={dF1:.4f}")
    print(f"  Confuser FPs: {conf_fp}  (lower=better)")
    print(f"  Elapsed: {elapsed:.0f}s")
    return dict(combined=(tp, fp, fn, P, R, F1),
                drone=(drone_tp, drone_fp, drone_fn, dP, dR, dF1),
                confuser_fps=conf_fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=7,
                    help="Sample every Nth frame (1359 frames / stride=7 -> ~194)")
    args = ap.parse_args()

    frames = enumerate_frames()
    print(f"Total frames available: {len(frames)}")
    frames = frames[::args.stride]
    print(f"After stride={args.stride}: {len(frames)} frames")

    all_results = {}
    for name in args.models:
        if name not in MODEL_REGISTRY:
            print(f"[skip] unknown model: {name}")
            continue
        weights, imgsz, modality, conf = MODEL_REGISTRY[name]
        if modality != "rgb":
            print(f"[skip] not RGB: {name}")
            continue
        all_results[name] = eval_model(name, weights, imgsz, conf, frames)

    print(f"\n{'='*70}")
    print(f"COMPARISON SUMMARY  (stride={args.stride}, {len(frames)} frames)")
    print(f"{'='*70}")
    print(f"\nDrone-only (per-frame detection metrics on drone clips):")
    print(f"  {'Model':<18s} {'TP':>5s} {'FP':>5s} {'FN':>5s}  {'P':>7s} {'R':>7s} {'F1':>7s}")
    for name, r in all_results.items():
        tp, fp, fn, P, R, F1 = r["drone"]
        print(f"  {name:<18s} {tp:>5d} {fp:>5d} {fn:>5d}  {P:>7.4f} {R:>7.4f} {F1:>7.4f}")
    print(f"\nConfuser FP count (lower=better, all frames negative):")
    for name, r in all_results.items():
        print(f"  {name:<18s} {r['confuser_fps']:>5d}")


if __name__ == "__main__":
    main()
