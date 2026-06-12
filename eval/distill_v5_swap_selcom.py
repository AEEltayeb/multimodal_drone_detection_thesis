#!/usr/bin/env python3
"""
Surgical selcom-source swap for the V5 cache + Phase 2 retrain.

The mixed selcom_train at `_finetune_selcom_mixed_ft2/images/train` is 80%
general drone data + 20% pure CCTV. Phase 3 eval shows V5 over-vetos on pure
CCTV (selcom_val F1 0.24 vs bare 0.59). Diagnosis: train-deploy distribution
mismatch — V5 learned selcom features on mostly-general data.

This script swaps the selcom training pool from mixed -> pure CCTV
(`G:/drone/selcom_dataset` minus the 311 selcom_val filenames) and rebuilds
the V5 MLP without re-mining the other 10 sources.

Usage:
    # Variant A: pure selcom at same 1.8x weight (isolate the source effect)
    python eval/distill_v5_swap_selcom.py --variant pure --weight 1.8

    # Variant B: pure selcom at 3.5x weight (source + optimization pressure)
    python eval/distill_v5_swap_selcom.py --variant pure_3x5 --weight 3.5

Outputs:
    eval/results/_v5_<variant>/training_data.npz
    eval/results/_v5_<variant>/classifiers/mlp_v5.pt
"""
from __future__ import annotations

import argparse
import json
import pickle
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
    _resolve_labels_dir, INPUT_DIM, META_DIM, P3_GRID, P5_GRID, _P3_DIM, _P5_DIM,
    MODEL_PATHS, MLPWrapper, LogRegWrapper, RFWrapper, XGBWrapper,
    cross_val_score_f1, is_jpg, IOU_THR, IOP_THR, CONF_THR, SEED,
)

# Existing V5 cache (the mixed-selcom baseline)
EXISTING_CACHE = REPO / "eval" / "results" / "_v5_p3p5_ft4_distill" / "training_data.npz"

# New PURE selcom source
PURE_SELCOM_IMAGES = Path("G:/drone/selcom_dataset/images")
# Block these filenames (they're in selcom_val, V5 evaluates there)
SELCOM_VAL_DIR = Path("G:/drone/_finetune_selcom_mixed_ft2/images/val")

# Selcom samples in existing cache are uniquely identifiable by weight:
#   weight == 1.8 -> selcom drone
#   weight == 1.5 -> selcom confuser
# Use a small tolerance for float comparison.
SELCOM_DRONE_WEIGHT = 1.8
SELCOM_CONFUSER_WEIGHT = 1.5
WEIGHT_TOL = 0.01


def slice_out_selcom(X, y, w):
    """Return (X, y, w) with all existing selcom rows removed."""
    is_selcom = (np.abs(w - SELCOM_DRONE_WEIGHT) < WEIGHT_TOL) | \
                (np.abs(w - SELCOM_CONFUSER_WEIGHT) < WEIGHT_TOL)
    keep = ~is_selcom
    n_removed = int(is_selcom.sum())
    n_drone_removed = int(((y == 1) & is_selcom).sum())
    n_conf_removed = int(((y == 0) & is_selcom).sum())
    print(f"  Slicing out existing selcom samples:")
    print(f"    {n_drone_removed} drones (weight={SELCOM_DRONE_WEIGHT})")
    print(f"    {n_conf_removed} confusers (weight={SELCOM_CONFUSER_WEIGHT})")
    print(f"    {n_removed} total removed; {keep.sum()} non-selcom samples kept")
    return X[keep], y[keep], w[keep]


def mine_pure_selcom(model, hook, weight_drone, weight_confuser, imgsz=1280):
    """Mine pure-CCTV selcom features from G:/drone/selcom_dataset/images
    minus the 311 selcom_val filenames. Returns X, y, w arrays."""
    if not PURE_SELCOM_IMAGES.exists():
        raise SystemExit(f"FATAL: {PURE_SELCOM_IMAGES} not found")

    # Build blocklist of selcom_val filenames
    blocklist = set()
    if SELCOM_VAL_DIR.exists():
        blocklist = {p.name for p in SELCOM_VAL_DIR.iterdir() if is_jpg(p)}
        print(f"  Blocklist: {len(blocklist)} selcom_val filenames will be skipped")

    labels_dir = _resolve_labels_dir(PURE_SELCOM_IMAGES)
    print(f"  Labels dir: {labels_dir}  (exists={labels_dir.exists()})")

    all_images = sorted(p for p in PURE_SELCOM_IMAGES.iterdir() if is_jpg(p))
    train_images = [p for p in all_images if p.name not in blocklist]
    print(f"  Pure selcom: {len(all_images)} total, {len(train_images)} after "
          f"blocklist (skipped {len(all_images) - len(train_images)})")

    X_tp, X_fp = [], []
    t0 = time.time()
    n_processed = 0

    for img_path in train_images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        n_processed += 1
        ih, iw = img_bgr.shape[:2]

        # Load GT
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

        # Run YOLO at selcom production imgsz
        hook.clear()
        results = model.predict(img_bgr, imgsz=imgsz, conf=CONF_THR,
                                verbose=False, device="cuda")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        for i in range(len(boxes)):
            det_box = tuple(boxes.xyxy[i].cpu().numpy().tolist())
            det_conf = float(boxes.conf[i])
            is_tp = _match_det_to_gt(det_box, gt_boxes, "iop")
            feat = _extract_detection_features(hook, det_box, (ih, iw), det_conf)
            if is_tp:
                X_tp.append(feat)
            else:
                X_fp.append(feat)

    dt = max(time.time() - t0, 0.1)
    fps = n_processed / dt
    print(f"  Mined: {len(X_tp)} TPs + {len(X_fp)} FPs from {n_processed} imgs "
          f"({fps:.1f} fps, {dt:.0f}s total)")

    X_tp_arr = np.asarray(X_tp, dtype=np.float32) if X_tp else np.empty((0, INPUT_DIM), dtype=np.float32)
    X_fp_arr = np.asarray(X_fp, dtype=np.float32) if X_fp else np.empty((0, INPUT_DIM), dtype=np.float32)
    y_tp = np.ones(len(X_tp), dtype=np.float32)
    y_fp = np.zeros(len(X_fp), dtype=np.float32)
    w_tp = np.full(len(X_tp), weight_drone, dtype=np.float32)
    w_fp = np.full(len(X_fp), weight_confuser, dtype=np.float32)
    return X_tp_arr, y_tp, w_tp, X_fp_arr, y_fp, w_fp


def train_v5_mlp(X, y, w, out_path: Path, cv_folds: int = 5):
    """Retrain the V5 MLP on the patched cache. Saves mlp_v5.pt."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  Training MLP V5 on {len(X)} samples, dim {X.shape[1]} ...")
    print(f"  Class balance: {int((y==1).sum())} drones / {int((y==0).sum())} confusers")
    print(f"  Weight stats: min={w.min():.2f} mean={w.mean():.2f} max={w.max():.2f}")

    mean_f1, std_f1, best_m = cross_val_score_f1(
        MLPWrapper, {"input_dim": X.shape[1]}, X, y, sample_weight=w,
        folds=cv_folds, seed=SEED)
    print(f"  CV F1 (mlp_meta+yolo, sample-weighted): {mean_f1:.4f} +- {std_f1:.4f}")

    # Persist the callable verifier artifact (same schema as mlp_v5.pt)
    torch.save({
        "state_dict": best_m.net.state_dict(),
        "scaler_mean": torch.from_numpy(best_m.scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(best_m.scaler.scale_.astype(np.float32)),
        "input_dim": int(best_m.input_dim),
        "hidden_dims": list(best_m.hidden_dims),
        "threshold": 0.5,
        "cv_f1": float(mean_f1),
        "cv_std": float(std_f1),
        "feature_schema": (f"{META_DIM} metadata + p3@{P3_GRID[0]}x{P3_GRID[1]} "
                           f"({_P3_DIM}-D) + p5@{P5_GRID[0]}x{P5_GRID[1]} "
                           f"({_P5_DIM}-D) = {META_DIM + (_P3_DIM + _P5_DIM)}-D"),
        "metadata_order": ["conf", "log_area", "aspect", "rel_cx", "rel_cy"],
        "p3_grid": list(P3_GRID),
        "p5_grid": list(P5_GRID),
        "use_batchnorm": True,
        "dropout": 0.3,
        "base_detector": MODEL_PATHS["ft4_r3"],
        "selcom_source": "pure_selcom_dataset (G:/drone/selcom_dataset, minus 311 selcom_val files)",
    }, out_path)
    print(f"  Saved: {out_path}")
    return mean_f1, std_f1


def main():
    p = argparse.ArgumentParser(description="V5 selcom-source ablation runner")
    p.add_argument("--variant", required=True,
                   choices=["pure", "pure_3x5"],
                   help="pure: same 1.8/1.5 weights, just source swap. "
                        "pure_3x5: source swap + bump weights to 3.5/2.5.")
    p.add_argument("--weight-drone", type=float, default=None,
                   help="Override drone weight (default by variant)")
    p.add_argument("--weight-confuser", type=float, default=None,
                   help="Override confuser weight (default by variant)")
    p.add_argument("--existing-cache", type=Path, default=EXISTING_CACHE,
                   help="Path to the V5 mixed-selcom cache")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="YOLO imgsz for pure-selcom mining (default: 1280)")
    args = p.parse_args()

    # Resolve weights
    if args.variant == "pure":
        w_drone = args.weight_drone if args.weight_drone is not None else 1.8
        w_conf  = args.weight_confuser if args.weight_confuser is not None else 1.5
        variant_dir = "_v5_selcom_pure_1x8"
    else:  # pure_3x5
        w_drone = args.weight_drone if args.weight_drone is not None else 3.5
        w_conf  = args.weight_confuser if args.weight_confuser is not None else 2.5
        variant_dir = "_v5_selcom_pure_3x5"

    out_root = REPO / "eval" / "results" / variant_dir
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "classifiers").mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  V5 selcom-source swap: variant={args.variant}")
    print(f"  Drone weight: {w_drone}    Confuser weight: {w_conf}")
    print(f"  Output dir: {out_root}")
    print("=" * 72)

    # Step 1: load existing cache, slice out old selcom
    print("\n[1/4] Loading existing cache ...")
    if not args.existing_cache.exists():
        raise SystemExit(f"FATAL: {args.existing_cache} not found. "
                          f"Run `python eval/distill_v5_p3p5_ft4.py` first.")
    z = np.load(args.existing_cache)
    X0, y0, w0 = z["X"].astype(np.float32), z["y"].astype(np.float32), z["w"].astype(np.float32)
    print(f"  Cache has {len(X0)} samples, dim {X0.shape[1]}")
    print(f"  ({int((y0==1).sum())} drones, {int((y0==0).sum())} confusers)")
    X_keep, y_keep, w_keep = slice_out_selcom(X0, y0, w0)

    # Step 2: mine pure-selcom features
    print(f"\n[2/4] Mining pure-CCTV selcom features at imgsz={args.imgsz} ...")
    yolo = YOLO(MODEL_PATHS["ft4_r3"])
    hook = DetectInputHook()
    handle = hook.register(yolo)
    try:
        Xn_tp, yn_tp, wn_tp, Xn_fp, yn_fp, wn_fp = mine_pure_selcom(
            yolo, hook, w_drone, w_conf, imgsz=args.imgsz)
    finally:
        handle.remove()

    # Step 3: concat, shuffle, save patched cache
    print("\n[3/4] Concatenating + shuffling patched cache ...")
    chunks_X = [X_keep, Xn_tp, Xn_fp]
    chunks_y = [y_keep, yn_tp, yn_fp]
    chunks_w = [w_keep, wn_tp, wn_fp]
    chunks_X = [c for c in chunks_X if len(c) > 0]
    chunks_y = [c for c in chunks_y if len(c) > 0]
    chunks_w = [c for c in chunks_w if len(c) > 0]
    X = np.concatenate(chunks_X, axis=0)
    y = np.concatenate(chunks_y, axis=0)
    w = np.concatenate(chunks_w, axis=0)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    X, y, w = X[perm], y[perm], w[perm]
    n_d = int((y == 1).sum())
    n_c = int((y == 0).sum())
    print(f"  Patched cache: {len(X)} samples ({n_d} drones, {n_c} confusers)")
    print(f"  Added: {len(Xn_tp)} selcom drones, {len(Xn_fp)} selcom confusers")

    patched_cache = out_root / "training_data.npz"
    np.savez_compressed(patched_cache, X=X, y=y, w=w)
    with open(out_root / "training_meta.json", "w") as f:
        json.dump({
            "variant": args.variant,
            "selcom_source": "G:/drone/selcom_dataset (pure CCTV, minus 311 selcom_val files)",
            "selcom_weight_drone": w_drone,
            "selcom_weight_confuser": w_conf,
            "imgsz": args.imgsz,
            "n_total": int(len(X)),
            "n_drone": n_d,
            "n_confuser": n_c,
            "n_selcom_drones_added": int(len(Xn_tp)),
            "n_selcom_confusers_added": int(len(Xn_fp)),
            "parent_cache": str(args.existing_cache),
        }, f, indent=2)
    print(f"  Saved: {patched_cache}")

    # Step 4: retrain MLP V5 on patched cache
    print("\n[4/4] Retraining MLP V5 ...")
    artifact_path = out_root / "classifiers" / "mlp_v5.pt"
    cv_f1, cv_std = train_v5_mlp(X, y, w, artifact_path)

    print("\n" + "=" * 72)
    print(f"  Done. Variant={args.variant}, CV F1={cv_f1:.4f}")
    print(f"  Artifact: {artifact_path}")
    print(f"  Next: python eval/eval_v4_vs_patch.py --mlp-weights {artifact_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
