#!/usr/bin/env python3
"""
V5-IR distillation — multi-scale p3+p5 on IR detector (finetune_v3b).

Port of the RGB V5 pipeline (eval/distill_v5_p3p5_ft4.py) to the IR modality.
Same architecture (517-D input, MLP with BN+dropout), same training recipe
(focal loss, sample weights, 5-fold CV), different datasets.

Key differences from RGB V5:
  - Detector: finetune_v3b (IR-domain-trained)
  - No selcom (RGB CCTV only)
  - Primary dataset: IR_dset_final (107k train, the IR model's training set)
  - Confusers: IR_dset_final negatives + IR_video confusers + airplane/bird IR
  - §14 lesson applied: no single source > 30% effective drone-gradient share

Usage:
    python eval/distill_v5_p3p5_ir.py              # full run (~3-4h mining + 45min training)
    python eval/distill_v5_p3p5_ir.py --phase 1    # mine only
    python eval/distill_v5_p3p5_ir.py --phase 2    # train only (requires cached data)
"""
from __future__ import annotations

import gc


import argparse
import json
import pickle
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "eval"
OUT_DIR = EVAL_DIR / "results" / "_v5_ir_p3p5_v3b"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "classifiers").mkdir(parents=True, exist_ok=True)

# Reuse all shared machinery from the RGB V5 trainer
import sys
sys.path.insert(0, str(REPO / "eval"))
from distill_v5_p3p5_ft4 import (
    DetectInputHook, _extract_detection_features, _match_det_to_gt,
    _resolve_labels_dir, INPUT_DIM, META_DIM, P3_GRID, P5_GRID,
    _P3_DIM, _P5_DIM, YOLO_FEAT_DIM,
    MLPWrapper, LogRegWrapper, RFWrapper, XGBWrapper,
    cross_val_score_f1, is_jpg, IOU_THR, IOP_THR, CONF_THR, SEED,
    SourceConfig, collect_from_source,
)

from ultralytics import YOLO

# Override conf threshold for IR mining — the IR detector is much more
# selective than the RGB one, so we need a lower floor to mine enough
# confuser FPs.  The RGB default (0.25) yields only ~1.8k confusers;
# 0.10 captures weak false positives the MLP must learn to reject.
import distill_v5_p3p5_ft4
distill_v5_p3p5_ft4.CONF_THR = 0.10

# ── IR-specific paths ──────────────────────────────────────────────────────

# IR detector (production candidate)
IR_DETECTOR = str(REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt")

# Drone sources
SVANSTROM_IR      = Path("G:/drone/svanstrom_paired/IR/images")
ANTIUAV_VAL_IR    = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/val/IR/images")
IR_DSET_TRAIN     = Path("G:/drone/IR_dset_final/train/images")
IR_DSET_VAL       = Path("G:/drone/IR_dset_final/val/images")
IR_VIDEO_TRAIN    = Path("G:/drone/IR_video_ir_dataset/train/images")
IR_VIDEO_VAL      = Path("G:/drone/IR_video_ir_dataset/val/images")

# Dedicated confuser sources — REAL THERMAL only.
# NOTE (2026-05-30): airplane_ir (grayscale-RGB) and bird_ir (color-RGB) were
# REMOVED — they are wrong-modality and contaminate the IR feature space.
# Replaced with CBAM thermal dataset (bird/drone/plane, confirmed grayscale-
# thermal). See docs/analysis/2026-05-30_ir_mlp_v5_analysis_and_plan.md §2.1.
CBAM_TRAIN = Path("G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/train/images")

# ── Parameters ──────────────────────────────────────────────────────────────
IMGSZ_DEFAULT = 640
IMGSZ_SVAN = 1280   # Svanström native 640x480, drones unresolvable at 640

# ── Source quotas ───────────────────────────────────────────────────────────
#
# §14 lesson: no single source > 30% effective drone-gradient share.
# Effective drone-gradient = target_drones × weight_drone
#
# Source              | drones × weight | Effective | Share
# --------------------|-----------------|-----------|------
# Svanström IR        | 4,000 × 1.5     | 6,000     | 22%
# Anti-UAV val IR     | 4,000 × 1.5     | 6,000     | 22%
# IR_dset_final train | 6,000 × 1.0     | 6,000     | 22%
# IR_dset_final val   | 1,500 × 1.0     | 1,500     | 5%
# IR_video drone      | 4,800 × 1.5     | 7,200     | 26%
# CBAM thermal drone  | 2,000 × 1.5     | 3,000     | 11% (NEW, class D=1)
# TOTAL               | 22,300          | 29,700    |

SOURCES = [
    # --- Drone-rich datasets: mine both TPs and hard-neg FPs ---
    SourceConfig("svanstrom_ir",       SVANSTROM_IR,      stride=1,  kind="image_with_gt",
                 target_drones=4000,  target_confusers=5000,
                 weight_drone=1.5,    weight_confuser=1.5,
                 match_rule="iop",    # Svan GT boxes larger than drone
                 imgsz=IMGSZ_SVAN),

    SourceConfig("antiuav_val_ir",     ANTIUAV_VAL_IR,    stride=3,  kind="image_with_gt",
                 target_drones=4000,  target_confusers=1000,
                 weight_drone=1.5,    weight_confuser=1.5),

    SourceConfig("ir_dset_train",      IR_DSET_TRAIN,     stride=2,  kind="image_with_gt",
                 target_drones=6000,  target_confusers=4000,
                 weight_drone=1.0,    weight_confuser=1.0),

    SourceConfig("ir_dset_val",        IR_DSET_VAL,       stride=3,  kind="image_with_gt",
                 target_drones=1500,  target_confusers=0,
                 weight_drone=1.0,    weight_confuser=1.0),

    # --- IR video: split by filename prefix ---
    SourceConfig("ir_video_train_drone", IR_VIDEO_TRAIN,  stride=2,  kind="image_with_gt",
                 target_drones=4000,  target_confusers=0,
                 weight_drone=1.5,    weight_confuser=1.5,
                 filter_prefixes=("IR_DRONE_",)),

    SourceConfig("ir_video_val_drone",   IR_VIDEO_VAL,    stride=1,  kind="image_with_gt",
                 target_drones=800,   target_confusers=0,
                 weight_drone=1.5,    weight_confuser=1.5,
                 filter_prefixes=("IR_DRONE_",)),

    SourceConfig("ir_video_train_conf",  IR_VIDEO_TRAIN,  stride=1,  kind="image_with_gt",
                 target_drones=0,     target_confusers=3500,
                 weight_drone=1.5,    weight_confuser=1.5,
                 filter_prefixes=("IR_AIRPLANE_", "IR_BIRD_", "IR_HELICOPTER_")),

    SourceConfig("ir_video_val_conf",    IR_VIDEO_VAL,    stride=1,  kind="image_with_gt",
                 target_drones=0,     target_confusers=500,
                 weight_drone=1.5,    weight_confuser=1.5,
                 filter_prefixes=("IR_AIRPLANE_", "IR_BIRD_", "IR_HELICOPTER_")),

    # --- CBAM thermal (real thermal bird/drone/plane) ---
    # drone=class 1, bird=class 0, plane=class 2. Mines both real thermal
    # drone TPs AND real thermal aerial-confuser FPs from a single source.
    # 10,064 train images; target 2k drones + 3.5k aerial confusers.
    SourceConfig("cbam_thermal",       CBAM_TRAIN,        stride=1,  kind="image_with_gt",
                 target_drones=2000,  target_confusers=3500,
                 weight_drone=1.5,    weight_confuser=1.5,
                 drone_class=1),      # CBAM names=['B','D','P'] -> D=1
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V5-IR feature distillation")
    parser.add_argument("--phase", type=int, default=1,
                        help="Start from phase: 1=mine, 2=train")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Limit per-source images for smoke testing")
    args = parser.parse_args()

    phase1_cache = OUT_DIR / "training_data.npz"
    phase1_meta  = OUT_DIR / "training_meta.json"

    # ── Phase 1: Mine features ───────────────────────────────────────
    if args.phase <= 1:
        print("=" * 72)
        print("  V5-IR Phase 1: Feature Mining")
        print(f"  Detector: {IR_DETECTOR}")
        print(f"  Output:   {OUT_DIR}")
        print("=" * 72)

        # Load IR detector
        yolo = YOLO(IR_DETECTOR)
        hook = DetectInputHook()
        handle = hook.register(yolo)

        all_X, all_y, all_w = [], [], []
        source_summary = {}

        try:
            for src in SOURCES:
                print(f"\n── Source: {src.name} ──")
                X_tp, y_tp, w_tp, X_fp, y_fp, w_fp = collect_from_source(
                    yolo, hook, src)

                for arr in [X_tp, X_fp]:
                    if len(arr) > 0:
                        all_X.append(arr)
                for arr in [y_tp, y_fp]:
                    if len(arr) > 0:
                        all_y.append(arr)
                for arr in [w_tp, w_fp]:
                    if len(arr) > 0:
                        all_w.append(arr)

                source_summary[src.name] = {
                    "n_drone": int(len(X_tp)),
                    "n_confuser": int(len(X_fp)),
                    "target_drones": src.target_drones,
                    "target_confusers": src.target_confusers,
                    "weight_drone": src.weight_drone,
                    "weight_confuser": src.weight_confuser,
                }

                # Free memory between sources
                del X_tp, y_tp, w_tp, X_fp, y_fp, w_fp
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            handle.remove()

        # Concatenate and shuffle
        X_all = np.concatenate(all_X, axis=0)
        y_all = np.concatenate(all_y, axis=0)
        w_all = np.concatenate(all_w, axis=0)
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(X_all))
        X_all, y_all, w_all = X_all[perm], y_all[perm], w_all[perm]

        n_d = int((y_all == 1).sum())
        n_c = int((y_all == 0).sum())

        print(f"\n{'='*72}")
        print(f"  Phase 1 complete: {len(X_all)} samples "
              f"({n_d} drones, {n_c} confusers), dim={X_all.shape[1]}")
        print(f"  Weight stats: min={w_all.min():.2f} mean={w_all.mean():.2f} "
              f"max={w_all.max():.2f}")

        # Per-source summary
        print(f"\n  Per-source yields:")
        total_eff = 0
        for name, info in source_summary.items():
            eff_d = info["n_drone"] * info["weight_drone"]
            eff_c = info["n_confuser"] * info["weight_confuser"]
            total_eff += eff_d
            print(f"    {name}: {info['n_drone']}d/{info['n_confuser']}c "
                  f"(eff_drone={eff_d:.0f})")
        print(f"  Total effective drone gradient: {total_eff:.0f}")
        for name, info in source_summary.items():
            eff_d = info["n_drone"] * info["weight_drone"]
            if eff_d > 0:
                share = eff_d / total_eff * 100
                flag = " ⚠️ EXCEEDS 30%" if share > 30 else ""
                print(f"    {name}: {share:.1f}%{flag}")

        # Save
        np.savez_compressed(phase1_cache, X=X_all, y=y_all, w=w_all)
        with open(phase1_meta, "w") as f:
            json.dump({
                "modality": "ir",
                "detector": IR_DETECTOR,
                "n_total": int(len(X_all)),
                "n_drone": n_d,
                "n_confuser": n_c,
                "feature_dim": int(X_all.shape[1]),
                "sources": source_summary,
            }, f, indent=2)
        print(f"  Saved: {phase1_cache}")
        print(f"         {phase1_meta}")
    else:
        print("\n── Phase 1: Loading cached training data ──")
        z = np.load(phase1_cache)
        X_all, y_all = z["X"].astype(np.float32), z["y"].astype(np.float32)
        w_all = z["w"] if "w" in z.files else np.ones(len(y_all), dtype=np.float32)
        with open(phase1_meta) as f:
            meta_info = json.load(f)
        print(f"  Loaded: {meta_info['n_drone']} drone + {meta_info['n_confuser']} "
              f"confuser = {meta_info['n_total']} samples")
        print(f"  Feature dim: {meta_info['feature_dim']}")

    # ── Phase 2: Train MLP ───────────────────────────────────────────
    if args.phase <= 2:
        print("\n── Phase 2: Training IR MLP V5 ──")
        print(f"  Samples: {len(X_all)}, dim: {X_all.shape[1]}")
        print(f"  Class balance: {int((y_all==1).sum())} drones / "
              f"{int((y_all==0).sum())} confusers")
        print(f"  Weight stats: min={w_all.min():.2f} mean={w_all.mean():.2f} "
              f"max={w_all.max():.2f}")

        X_full = X_all
        t0 = time.time()
        mean_f1, std_f1, best_m = cross_val_score_f1(
            MLPWrapper, {"input_dim": X_full.shape[1]}, X_full, y_all,
            sample_weight=w_all, folds=5, seed=SEED)
        dt = time.time() - t0
        print(f"  CV F1 (mlp_meta+yolo, sample-weighted): {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"  Training time: {dt:.0f}s")

        # Save checkpoint
        artifact_path = OUT_DIR / "classifiers" / "mlp_v5_ir.pt"
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
            "base_detector": IR_DETECTOR,
            "modality": "ir",
        }, artifact_path)
        print(f"  Saved: {artifact_path}")

        print(f"\n{'='*72}")
        print(f"  Done. IR MLP V5 CV F1={mean_f1:.4f} (training took {dt:.0f}s)")
        print(f"  Artifact: {artifact_path}")
        print(f"\n  Next steps:")
        print(f"    1. Run PCA/LDA analysis:")
        print(f"       python scripts/visualize_v5_features_ir.py")
        print(f"    2. Evaluate:")
        print(f"       python eval/eval_v4_vs_patch.py --modality ir \\")
        print(f"           --mlp-weights {artifact_path}")
        print(f"{'='*72}")


if __name__ == "__main__":
    main()
