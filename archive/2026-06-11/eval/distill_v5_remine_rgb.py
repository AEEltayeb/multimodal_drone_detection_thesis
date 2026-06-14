#!/usr/bin/env python3
"""
V5.2 — rgb_dataset coverage boost via net-new re-mine + Phase-2 retrain.

Diagnosis (EVIDENCE_LEDGER §13.8, journey §14): V5 caps at ~0.66-0.77 recall on
rgb_dataset_test and vetoes real drones. Threshold sweep can't fix it (missed
drones score confidently low) and the Svan-weight rebalance is a measured no-op.
The remaining hypothesis: the gap is feature-space COVERAGE — the test-split
drones live in a region the cache's 9,500 rgb_dataset train+val drones (mined at
stride 8/3) don't cover.

This script tests that hypothesis the only way that can move it: add MORE
rgb_dataset drone diversity. It mines NET-NEW rgb_dataset frames at a finer
stride and APPENDS them to the existing pure_1x8 cache — no removal, no
re-mining of the other 10 sources, so Svan/selcom/Anti-UAV/confuser signal is
untouched (low cross-surface regression risk).

Net-new guarantee (no duplication with the original mine):
    Original rgb_dataset_train used stride 8 -> only frames where (idx % 8 == 0).
    Original rgb_dataset_val   used stride 3 -> only frames where (idx % 3 == 0).
    This script mines train at the finer --train-stride but SKIPS idx % 8 == 0,
    and val at --val-stride but SKIPS idx % 3 == 0. Every appended frame is
    therefore guaranteed unused by the original cache.

Usage:
    python eval/distill_v5_remine_rgb.py            # defaults: train stride 3, val stride 1
    python eval/distill_v5_remine_rgb.py --train-stride 3 --train-cap 10000 --val-cap 2000

Outputs:
    eval/results/_v5_remine_rgb/training_data.npz
    eval/results/_v5_remine_rgb/classifiers/mlp_v5.pt
    eval/results/_v5_remine_rgb/training_meta.json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO / "eval"))
from distill_v5_p3p5_ft4 import (
    DetectInputHook, _extract_detection_features, _match_det_to_gt,
    _resolve_labels_dir, INPUT_DIM, MODEL_PATHS, is_jpg, CONF_THR,
)
from distill_v5_swap_selcom import train_v5_mlp  # reuse the exact same trainer

# Source cache: production V5 (pure selcom) — the one we are augmenting
SRC_CACHE = REPO / "eval" / "results" / "_v5_selcom_pure_1x8" / "training_data.npz"
OUT_DIR = REPO / "eval" / "results" / "_v5_remine_rgb"

# rgb_dataset paths (mirror distill_v5_p3p5_ft4.py)
RGB_DATASET_TRAIN = Path("G:/drone/dataset/dataset/images/train")
RGB_DATASET_VAL = Path("G:/drone/dataset/dataset/images/val")

# rgb_dataset weights in the original cache (kept identical so appended samples
# blend in at the same per-sample weight as the existing rgb_dataset rows).
RGB_WEIGHT_DRONE = 1.0
RGB_WEIGHT_CONFUSER = 1.0

IMGSZ = 640          # rgb_dataset production imgsz
MATCH_RULE = "iou"   # rgb_dataset uses IoU (not a paired/CCTV surface)

# Original strides used by distill_v5_p3p5_ft4.py for these sources — frames on
# these grids are ALREADY in the cache and must be skipped to avoid duplicates.
ORIG_TRAIN_STRIDE = 8
ORIG_VAL_STRIDE = 3


def mine_netnew(model, hook, img_dir: Path, fine_stride: int, skip_modulo: int,
                cap_drones: int, cap_confusers: int, imgsz: int):
    """Mine frames on the fine_stride grid, SKIPPING any whose sorted index is a
    multiple of skip_modulo (already in the original cache). Returns
    (X_tp, y_tp, w_tp, X_fp, y_fp, w_fp)."""
    if not img_dir.exists():
        raise SystemExit(f"FATAL: {img_dir} not found")
    labels_dir = _resolve_labels_dir(img_dir)
    print(f"  Dir: {img_dir}")
    print(f"  Labels: {labels_dir} (exists={labels_dir.exists()})")

    all_images = sorted(p for p in img_dir.iterdir() if is_jpg(p))
    # fine grid minus original grid
    selected = [p for i, p in enumerate(all_images)
                if (i % fine_stride == 0) and (i % skip_modulo != 0)]
    print(f"  {len(all_images)} total imgs -> {len(selected)} net-new frames "
          f"(stride {fine_stride}, skipping idx %% {skip_modulo} == 0)")

    X_tp, X_fp = [], []
    t0 = time.time()
    n_processed = 0
    for img_path in selected:
        if len(X_tp) >= cap_drones and len(X_fp) >= cap_confusers:
            break
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        n_processed += 1
        ih, iw = img_bgr.shape[:2]

        gt_boxes = []
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2) * iw
                    y1 = (yc - bh / 2) * ih
                    x2 = (xc + bw / 2) * iw
                    y2 = (yc + bh / 2) * ih
                    gt_boxes.append((x1, y1, x2, y2))

        hook.clear()
        results = model.predict(img_bgr, imgsz=imgsz, conf=CONF_THR,
                                verbose=False, device="cuda")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        for i in range(len(boxes)):
            det_box = tuple(boxes.xyxy[i].cpu().numpy().tolist())
            det_conf = float(boxes.conf[i])
            is_tp = _match_det_to_gt(det_box, gt_boxes, MATCH_RULE)
            if is_tp and len(X_tp) >= cap_drones:
                continue
            if (not is_tp) and len(X_fp) >= cap_confusers:
                continue
            feat = _extract_detection_features(hook, det_box, (ih, iw), det_conf)
            (X_tp if is_tp else X_fp).append(feat)

    dt = max(time.time() - t0, 0.1)
    print(f"  Mined: {len(X_tp)} drones + {len(X_fp)} confusers from "
          f"{n_processed} imgs ({n_processed/dt:.1f} fps, {dt:.0f}s)")

    def arr(lst):
        return (np.asarray(lst, dtype=np.float32) if lst
                else np.empty((0, INPUT_DIM), dtype=np.float32))
    X_tp_a, X_fp_a = arr(X_tp), arr(X_fp)
    return (X_tp_a, np.ones(len(X_tp), np.float32),
            np.full(len(X_tp), RGB_WEIGHT_DRONE, np.float32),
            X_fp_a, np.zeros(len(X_fp), np.float32),
            np.full(len(X_fp), RGB_WEIGHT_CONFUSER, np.float32))


def main():
    p = argparse.ArgumentParser(description="V5.2 rgb_dataset coverage boost")
    p.add_argument("--train-stride", type=int, default=3,
                   help="Fine stride for rgb_dataset_train net-new mine (default 3)")
    p.add_argument("--val-stride", type=int, default=1,
                   help="Fine stride for rgb_dataset_val net-new mine (default 1)")
    p.add_argument("--train-cap", type=int, default=12000,
                   help="Max net-new train drones to add (default 12000)")
    p.add_argument("--val-cap", type=int, default=2500,
                   help="Max net-new val drones to add (default 2500)")
    p.add_argument("--conf-cap", type=int, default=4000,
                   help="Max net-new rgb_dataset confusers to add (default 4000)")
    p.add_argument("--cache", type=Path, default=SRC_CACHE)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classifiers").mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  V5.2 — rgb_dataset coverage boost (net-new re-mine + Phase-2 retrain)")
    print(f"  train stride {ORIG_TRAIN_STRIDE} -> {args.train_stride} "
          f"(skip idx %% {ORIG_TRAIN_STRIDE}); cap {args.train_cap} drones")
    print(f"  val   stride {ORIG_VAL_STRIDE} -> {args.val_stride} "
          f"(skip idx %% {ORIG_VAL_STRIDE}); cap {args.val_cap} drones")
    print(f"  Source cache: {args.cache}")
    print(f"  Output:       {OUT_DIR}")
    print("=" * 72)

    print("\n[1/4] Loading pure_1x8 cache ...")
    if not args.cache.exists():
        raise SystemExit(f"FATAL: {args.cache} not found")
    z = np.load(args.cache)
    X0, y0, w0 = (z["X"].astype(np.float32), z["y"].astype(np.float32),
                  z["w"].astype(np.float32))
    print(f"  {len(X0)} samples ({int((y0==1).sum())} drones, "
          f"{int((y0==0).sum())} confusers)")

    print("\n[2/4] Mining net-new rgb_dataset features ...")
    yolo = YOLO(MODEL_PATHS["ft4_r3"])
    hook = DetectInputHook()
    handle = hook.register(yolo)
    try:
        Xt_tp, yt_tp, wt_tp, Xt_fp, yt_fp, wt_fp = mine_netnew(
            yolo, hook, RGB_DATASET_TRAIN, args.train_stride, ORIG_TRAIN_STRIDE,
            args.train_cap, args.conf_cap, IMGSZ)
        Xv_tp, yv_tp, wv_tp, Xv_fp, yv_fp, wv_fp = mine_netnew(
            yolo, hook, RGB_DATASET_VAL, args.val_stride, ORIG_VAL_STRIDE,
            args.val_cap, 0, IMGSZ)
    finally:
        handle.remove()

    n_new_drone = len(Xt_tp) + len(Xv_tp)
    n_new_conf = len(Xt_fp) + len(Xv_fp)
    print(f"\n  Net-new added: {n_new_drone} drones + {n_new_conf} confusers")

    print("\n[3/4] Concatenating + shuffling ...")
    chunks = [(X0, y0, w0), (Xt_tp, yt_tp, wt_tp), (Xt_fp, yt_fp, wt_fp),
              (Xv_tp, yv_tp, wv_tp), (Xv_fp, yv_fp, wv_fp)]
    chunks = [c for c in chunks if len(c[0]) > 0]
    X = np.concatenate([c[0] for c in chunks], axis=0)
    y = np.concatenate([c[1] for c in chunks], axis=0)
    w = np.concatenate([c[2] for c in chunks], axis=0)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(X))
    X, y, w = X[perm], y[perm], w[perm]
    print(f"  New cache: {len(X)} samples "
          f"({int((y==1).sum())} drones, {int((y==0).sum())} confusers)")

    np.savez_compressed(OUT_DIR / "training_data.npz", X=X, y=y, w=w)
    with open(OUT_DIR / "training_meta.json", "w") as f:
        json.dump({
            "variant": "v5.2_remine_rgb",
            "train_stride": args.train_stride, "val_stride": args.val_stride,
            "train_cap": args.train_cap, "val_cap": args.val_cap,
            "conf_cap": args.conf_cap,
            "n_new_drones": int(n_new_drone), "n_new_confusers": int(n_new_conf),
            "n_total": int(len(X)), "n_drone": int((y == 1).sum()),
            "n_confuser": int((y == 0).sum()),
            "parent_cache": str(args.cache),
            "note": "Additive net-new rgb_dataset mine. No removal, no other sources touched.",
        }, f, indent=2)
    print(f"  Saved cache + meta to {OUT_DIR}")

    print("\n[4/4] Retraining MLP V5 (Phase 2 only) ...")
    artifact = OUT_DIR / "classifiers" / "mlp_v5.pt"
    cv_f1, cv_std = train_v5_mlp(X, y, w, artifact)

    print("\n" + "=" * 72)
    print(f"  Done. CV F1={cv_f1:.4f} +- {cv_std:.4f}")
    print(f"  Artifact: {artifact}")
    print(f"  Next: py eval/eval_pipeline_v5_quick.py --n-images 300 "
          f"--mlp-weights {artifact}")
    print("=" * 72)


if __name__ == "__main__":
    main()
