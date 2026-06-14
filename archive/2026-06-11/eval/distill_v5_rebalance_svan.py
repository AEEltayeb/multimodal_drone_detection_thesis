#!/usr/bin/env python3
"""
V5.1 — Svanstrom drone weight rebalance (Phase 2 retrain only).

Loads the existing production V5 cache (_v5_selcom_pure_1x8/training_data.npz),
reduces Svanstrom drone sample weights from 2.5 -> 1.5 (confuser weight stays
at 2.5), and retrains the MLP on the reweighted cache. No feature re-mining.

Rationale:
    V5 over-weights Svanstrom drones (40% effective gradient share), which biases
    the decision boundary toward Svanstrom sky-style prototypes and causes
    over-vetoing on diverse RGB benchmarks (rgb_dataset_test F1 0.792 vs bare 0.929).
    Reducing Svan drone weight to 1.5 brings Svan and rgb_dataset to roughly
    equal gradient shares (~33% vs ~36%), giving the AirBird scene prototype
    equal say as the Svanstrom sky prototype.

    Expected outcome:
      - rgb_dataset_test F1: 0.792 -> ~0.85
      - Svanstrom F1: 0.869 -> ~0.84-0.85 (still beats patch v2 by ~8 pp)
      - selcom + Anti-UAV + confuser: roughly unchanged

Usage:
    python eval/distill_v5_rebalance_svan.py

    Then evaluate:
    python eval/eval_v4_vs_patch.py \\
        --mlp-weights eval/results/_v5_rebalance_svan/classifiers/mlp_v5.pt \\
        --datasets rgb_dataset_test \\
        --out-suffix rebalance_svan \\
        --mlp-thrs 0.15,0.25,0.35,0.5,0.7
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO / "eval"))
from distill_v5_p3p5_ft4 import (
    INPUT_DIM, META_DIM, P3_GRID, P5_GRID, _P3_DIM, _P5_DIM,
    MODEL_PATHS, MLPWrapper, cross_val_score_f1, SEED,
)

# ── Config ──────────────────────────────────────────────────────────────────
# Source cache: production V5 with pure selcom
SRC_CACHE = REPO / "eval" / "results" / "_v5_selcom_pure_1x8" / "training_data.npz"
# Output
OUT_DIR = REPO / "eval" / "results" / "_v5_rebalance_svan"

# Svanstrom is the w=2.5 group. We only change DRONE weights (y==1).
OLD_SVAN_WEIGHT = 2.5
NEW_SVAN_DRONE_WEIGHT = 1.5   # was 2.5
# Confuser weight stays at 2.5 (no change)

WEIGHT_TOL = 0.01


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "classifiers").mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("  V5.1 — Svanstrom drone weight rebalance")
    print(f"  Svan drone weight: {OLD_SVAN_WEIGHT} -> {NEW_SVAN_DRONE_WEIGHT}")
    print(f"  Svan confuser weight: {OLD_SVAN_WEIGHT} (unchanged)")
    print(f"  Source cache: {SRC_CACHE}")
    print(f"  Output dir:   {OUT_DIR}")
    print("=" * 72)

    # ── Step 1: Load cache ──────────────────────────────────────────────
    print("\n[1/3] Loading production V5 cache ...")
    if not SRC_CACHE.exists():
        raise SystemExit(f"FATAL: {SRC_CACHE} not found. "
                          f"Run distill_v5_swap_selcom.py first.")
    z = np.load(SRC_CACHE)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.float32)
    w = z["w"].astype(np.float32)
    print(f"  Loaded: {len(X)} samples, dim {X.shape[1]}")
    print(f"  Drones: {int((y==1).sum())}  Confusers: {int((y==0).sum())}")

    # Show pre-rebalance weight distribution
    print("\n  Pre-rebalance weight distribution:")
    for ww in sorted(np.unique(w)):
        n = int((w == ww).sum())
        nd = int(((w == ww) & (y == 1)).sum())
        nc = int(((w == ww) & (y == 0)).sum())
        eff = nd * ww + nc * ww
        print(f"    w={ww:.2f}: {n} samples (drone={nd}, conf={nc}, eff_grad={eff:.0f})")

    # ── Step 2: Reweight Svanstrom drones ───────────────────────────────
    print(f"\n[2/3] Reweighting Svanstrom drones: {OLD_SVAN_WEIGHT} -> {NEW_SVAN_DRONE_WEIGHT} ...")
    is_svan = np.abs(w - OLD_SVAN_WEIGHT) < WEIGHT_TOL
    is_svan_drone = is_svan & (y == 1)
    is_svan_conf  = is_svan & (y == 0)
    n_svan_drone = int(is_svan_drone.sum())
    n_svan_conf  = int(is_svan_conf.sum())
    print(f"  Found {n_svan_drone} Svan drones + {n_svan_conf} Svan confusers at w={OLD_SVAN_WEIGHT}")

    w[is_svan_drone] = NEW_SVAN_DRONE_WEIGHT
    print(f"  Set {n_svan_drone} Svan drone weights to {NEW_SVAN_DRONE_WEIGHT}")
    print(f"  Svan confuser weights stay at {OLD_SVAN_WEIGHT}")

    # Show post-rebalance effective gradient shares (drone side only)
    print("\n  Post-rebalance effective gradient shares (drones):")
    drone_mask = (y == 1)
    total_drone_eff = float(np.sum(w[drone_mask]))
    for ww in sorted(np.unique(w[drone_mask])):
        mask = drone_mask & (np.abs(w - ww) < WEIGHT_TOL)
        nd = int(mask.sum())
        eff = float(nd * ww)
        pct = eff / total_drone_eff * 100
        print(f"    w={ww:.2f}: {nd} drones, eff={eff:.0f} ({pct:.1f}%)")
    print(f"    Total effective drone gradient: {total_drone_eff:.0f}")

    # Save reweighted cache
    cache_path = OUT_DIR / "training_data.npz"
    np.savez_compressed(cache_path, X=X, y=y, w=w)
    print(f"  Saved: {cache_path}")

    # Save metadata
    with open(OUT_DIR / "training_meta.json", "w") as f:
        json.dump({
            "variant": "v5.1_rebalance_svan",
            "change": f"Svan drone weight {OLD_SVAN_WEIGHT} -> {NEW_SVAN_DRONE_WEIGHT}",
            "svan_drone_weight_old": OLD_SVAN_WEIGHT,
            "svan_drone_weight_new": NEW_SVAN_DRONE_WEIGHT,
            "svan_confuser_weight": OLD_SVAN_WEIGHT,
            "n_svan_drones_reweighted": n_svan_drone,
            "n_svan_confusers_unchanged": n_svan_conf,
            "n_total": int(len(X)),
            "n_drone": int((y==1).sum()),
            "n_confuser": int((y==0).sum()),
            "parent_cache": str(SRC_CACHE),
        }, f, indent=2)

    # ── Step 3: Retrain MLP ─────────────────────────────────────────────
    print("\n[3/3] Retraining MLP V5.1 ...")
    print(f"  Samples: {len(X)}, dim: {X.shape[1]}")
    print(f"  Class balance: {int((y==1).sum())} drones / {int((y==0).sum())} confusers")
    print(f"  Weight stats: min={w.min():.2f} mean={w.mean():.2f} max={w.max():.2f}")

    t0 = time.time()
    mean_f1, std_f1, best_m = cross_val_score_f1(
        MLPWrapper, {"input_dim": X.shape[1]}, X, y, sample_weight=w,
        folds=5, seed=SEED)
    dt = time.time() - t0
    print(f"  CV F1 (mlp_meta+yolo, sample-weighted): {mean_f1:.4f} +- {std_f1:.4f}")
    print(f"  Training time: {dt:.0f}s")

    # Save checkpoint
    artifact_path = OUT_DIR / "classifiers" / "mlp_v5.pt"
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
        "variant": "v5.1_rebalance_svan",
        "svan_drone_weight": NEW_SVAN_DRONE_WEIGHT,
    }, artifact_path)
    print(f"  Saved: {artifact_path}")

    print("\n" + "=" * 72)
    print(f"  Done. V5.1 CV F1={mean_f1:.4f} (training took {dt:.0f}s)")
    print(f"  Artifact: {artifact_path}")
    print(f"\n  Evaluate on rgb_dataset_test:")
    print(f"    python eval/eval_v4_vs_patch.py \\")
    print(f"        --mlp-weights {artifact_path} \\")
    print(f"        --datasets rgb_dataset_test \\")
    print(f"        --out-suffix rebalance_svan \\")
    print(f"        --mlp-thrs 0.15,0.25,0.35,0.5,0.7")
    print("=" * 72)


if __name__ == "__main__":
    main()
