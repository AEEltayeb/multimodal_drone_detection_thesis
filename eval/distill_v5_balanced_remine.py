#!/usr/bin/env python3
"""distill_v5_balanced_remine.py — RGB filter re-mine that FIXES the
rgb_dataset_test small-drone veto (docs/analysis/2026-06-17_rgbtest_filter_regression.md).

Diagnosis: the shipped mlp_v5 vetoes 22% of real rgb_dataset_test drones because
the distill corpus collected rgb_dataset drones via an alphabetical, stride-8,
8000-drone quota that front-loaded large/early-alphabet drones and STARVED the
small-drone tail (wosdetc et al.) -> those drones are 3.3x OOD from the training
drone manifold, in a region owned by confusers, so the filter rejects them.

Fix (this script): mine rgb_dataset drones into a balanced (sub-source prefix x
detection size-bucket) quota grid, so the small-drone manifold is populated.
CONFUSERS ARE PROTECTED — confuser mining is unchanged (flat quotas, same OOD
hard-neg corpus) so precision on rgb_confuser / rgb_bird_confuser holds. All
OTHER sources (Anti-UAV, Svanstrom, Selcom, rgb_video, confuser sets) are mined
exactly as production via the original collect_from_source. Trains mlp_v5_balanced.pt
with the identical checkpoint schema (drop-in for the GUI / eval harness).

GPU. Reuses distill_v5_p3p5_ft4 primitives (feature extraction, MLPWrapper, CV).

  py eval/distill_v5_balanced_remine.py                       # full
  py eval/distill_v5_balanced_remine.py --per-cell-cap 1200   # tune balance
  py eval/distill_v5_balanced_remine.py --quick               # smoke (~10 min)
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval")); sys.path.insert(0, str(REPO / "classifier"))
import distill_v5_p3p5_ft4 as D                       # reuse all heavy primitives
from distill_v5_swap_selcom import mine_pure_selcom   # proven pure-CCTV selcom miner (blocklist + IoP + 1280)

OUT = REPO / "eval" / "results" / "_v5_balanced_remine"
(OUT / "classifiers").mkdir(parents=True, exist_ok=True)

SIZE_EDGES = (16, 32, 64)                              # detection short-side px
SIZE_NAMES = ["xs", "s", "m", "l"]


def size_bin(box) -> int:
    x1, y1, x2, y2 = box
    s = min(x2 - x1, y2 - y1)
    return 0 if s < SIZE_EDGES[0] else 1 if s < SIZE_EDGES[1] else 2 if s < SIZE_EDGES[2] else 3


def prefix_of(name: str) -> str:
    return name.replace("-", "_").split("_")[0]


def collect_rgbdataset_balanced(model, hook, img_dir: Path, stride: int, rule: str,
                                imgsz: int, per_cell_cap: int, target_confusers: int,
                                total_cap: int = 0, weight_drone=1.0, weight_confuser=1.0):
    """Full scan of one rgb_dataset split: drones cell-gated by (prefix x size),
    confusers flat to target_confusers. Returns the 6-tuple collect_from_source
    returns, plus a per-cell count dict."""
    if not img_dir.exists():
        e = np.empty((0, D.INPUT_DIM), np.float32)
        return e, np.empty(0), np.empty(0), e, np.empty(0), np.empty(0), {}
    labels_dir = D._resolve_labels_dir(img_dir)
    images = sorted(p for p in img_dir.iterdir() if D.is_jpg(p))[::stride]
    print(f"  [balanced] {img_dir}  {len(images)} imgs (stride {stride}, cap {per_cell_cap}/cell, "
          f"target_conf {target_confusers})  labels={labels_dir.exists()}")

    X_tp, X_fp = [], []
    cell = defaultdict(int)
    t0 = time.time(); n_proc = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        n_proc += 1
        ih, iw = img.shape[:2]
        gt = []
        lp = labels_dir / (img_path.stem + ".txt")
        if lp.exists():
            for ln in lp.read_text().splitlines():
                t = ln.split()
                if len(t) >= 5 and int(t[0]) == 0:
                    xc, yc, bw, bh = map(float, t[1:5])
                    gt.append(((xc - bw/2)*iw, (yc - bh/2)*ih, (xc + bw/2)*iw, (yc + bh/2)*ih))
        hook.clear()
        res = model.predict(img, imgsz=imgsz, conf=D.CONF_THR, verbose=False, device="cuda")
        bx = res[0].boxes
        if bx is None or len(bx) == 0:
            continue
        pre = prefix_of(img_path.name)
        for i in range(len(bx)):
            box = tuple(bx.xyxy[i].cpu().numpy()); conf = float(bx.conf[i])
            is_drone = D._match_det_to_gt(box, gt, rule)
            if is_drone:
                key = (pre, size_bin(box))
                if cell[key] >= per_cell_cap:
                    continue
                X_tp.append(D._extract_detection_features(hook, box, (ih, iw), conf)); cell[key] += 1
            else:
                if len(X_fp) >= target_confusers:
                    continue
                X_fp.append(D._extract_detection_features(hook, box, (ih, iw), conf))
    dt = max(time.time() - t0, .1)
    if total_cap and len(X_tp) > total_cap:           # ratio-preserving cap: redistribute, don't inflate
        rng = np.random.RandomState(D.SEED)
        X_tp = [X_tp[i] for i in sorted(rng.choice(len(X_tp), total_cap, replace=False).tolist())]
    print(f"    -> {len(X_tp)} drones (balanced{', capped '+str(total_cap) if total_cap else ''}) + {len(X_fp)} confusers  ({n_proc} imgs, {n_proc/dt:.1f} fps)")
    Xtp = np.array(X_tp, np.float32) if X_tp else np.empty((0, D.INPUT_DIM), np.float32)
    Xfp = np.array(X_fp, np.float32) if X_fp else np.empty((0, D.INPUT_DIM), np.float32)
    celld = {f"{p}_{SIZE_NAMES[s]}": c for (p, s), c in sorted(cell.items())}
    return (Xtp, np.ones(len(X_tp), np.float32), np.full(len(X_tp), weight_drone, np.float32),
            Xfp, np.zeros(len(X_fp), np.float32), np.full(len(X_fp), weight_confuser, np.float32), celld)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell-cap", type=int, default=1200,
                    help="max drones per (sub-source x size-bucket) cell for rgb_dataset")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cap = max(1, a.per_cell_cap // 8) if a.quick else a.per_cell_cap
    tr_stride = 40 if a.quick else 8
    va_stride = 9 if a.quick else 3

    print("=" * 72); print("  RGB filter BALANCED re-mine (size x source; confusers protected)"); print("=" * 72)
    model = YOLO_load()
    hook = D.DetectInputHook(); handle = hook.register(model)

    X_chunks, y_chunks, w_chunks, source_counts = [], [], [], []
    cell_meta = {}

    # 1) rgb_dataset train + val -> BALANCED drones, protected confusers
    # total_cap restores the original rgb_dataset budget (~8000 train + 1500 val = shipped 9500)
    # so the drone:confuser ratio matches shipped (~1.42) -> avoids the bird-FP dilution.
    for split, stride, tconf, tcap in (("train", tr_stride, 0 if a.quick else 3000, 8000), ("val", va_stride, 0, 1500)):
        img_dir = D.RGB_DATASET_TRAIN if split == "train" else D.RGB_DATASET_VAL
        Xtp, ytp, wtp, Xfp, yfp, wfp, cells = collect_rgbdataset_balanced(
            model, hook, img_dir, stride, "iou", 640, cap, tconf, total_cap=tcap)
        for Xc, yc, wc in ((Xtp, ytp, wtp), (Xfp, yfp, wfp)):
            if len(Xc):
                X_chunks.append(Xc); y_chunks.append(yc); w_chunks.append(wc)
        cell_meta[f"rgb_dataset_{split}"] = cells
        source_counts.append({"name": f"rgb_dataset_{split}_balanced", "n_drones": int(len(Xtp)), "n_confusers": int(len(Xfp))})

    # 2) every OTHER source EXACTLY as production, EXCEPT selcom (swapped to PURE below)
    others = [s for s in D.SOURCES if not s.name.startswith("rgb_dataset") and s.name != "selcom_train"]
    for src in (_quickify(others) if a.quick else others):
        Xtp, ytp, wtp, Xfp, yfp, wfp = D.collect_from_source(model, hook, src)
        for Xc, yc, wc in ((Xtp, ytp, wtp), (Xfp, yfp, wfp)):
            if len(Xc):
                X_chunks.append(Xc); y_chunks.append(yc); w_chunks.append(wc)
        source_counts.append({"name": src.name, "n_drones": int(len(Xtp)), "n_confusers": int(len(Xfp))})

    # 2b) PURE-CCTV selcom (matches the shipped recipe + selcom_val; fixes the -26pp selcom regression).
    #     mine_pure_selcom blocklists the 311 selcom_val files, mines at imgsz 1280 / IoP, weights 1.8/1.5.
    print("  pure selcom (G:/drone/selcom_dataset minus selcom_val):")
    sx_tp, sy_tp, sw_tp, sx_fp, sy_fp, sw_fp = mine_pure_selcom(
        model, hook, 1.8, 1.5, imgsz=(640 if a.quick else 1280))
    for Xc, yc, wc in ((sx_tp, sy_tp, sw_tp), (sx_fp, sy_fp, sw_fp)):
        if len(Xc):
            X_chunks.append(Xc); y_chunks.append(yc); w_chunks.append(wc)
    source_counts.append({"name": "selcom_pure", "n_drones": int(len(sx_tp)), "n_confusers": int(len(sx_fp))})

    X = np.concatenate(X_chunks); y = np.concatenate(y_chunks); w = np.concatenate(w_chunks)
    rng = np.random.RandomState(D.SEED); perm = rng.permutation(len(X))
    X, y, w = X[perm], y[perm], w[perm]
    np.savez_compressed(OUT / "training_data.npz", X=X, y=y, w=w)
    (OUT / "training_meta.json").write_text(json.dumps({
        "variant": "balanced_remine_v2_pureselcom_capped", "per_cell_cap": cap, "size_edges_px": SIZE_EDGES,
        "n_total": int(len(X)), "n_drone": int((y == 1).sum()), "n_confuser": int((y == 0).sum()),
        "per_source_counts": source_counts, "rgb_dataset_cells": cell_meta,
        "base_detector": D.MODEL_PATHS["ft4_r3"],
    }, indent=2))
    print(f"\n  corpus: {int((y==1).sum())} drone + {int((y==0).sum())} confuser = {len(X)}")

    # 3) train (reuse the production CV+MLP recipe) and save drop-in checkpoint
    print("  training mlp (V5 arch: focal+BN+sample weights, 5-fold CV best)...")
    cv_f1, cv_std, best = D.cross_val_score_f1(D.MLPWrapper, {"input_dim": X.shape[1]}, X, y, sample_weight=w)
    print(f"    CV F1 {cv_f1:.4f} ± {cv_std:.4f}")
    art = OUT / "classifiers" / "mlp_v5_balanced.pt"
    torch.save({
        "state_dict": best.net.state_dict(),
        "scaler_mean": torch.from_numpy(best.scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(best.scaler.scale_.astype(np.float32)),
        "input_dim": int(best.input_dim), "hidden_dims": list(best.hidden_dims),
        "threshold": 0.5, "cv_f1": float(cv_f1), "cv_std": float(cv_std),
        "metadata_order": ["conf", "log_area", "aspect", "rel_cx", "rel_cy"],
        "p3_grid": list(D.P3_GRID), "p5_grid": list(D.P5_GRID),
        "use_batchnorm": True, "dropout": 0.3, "base_detector": D.MODEL_PATHS["ft4_r3"],
        "note": "balanced re-mine (rgb_dataset drones balanced by prefix x size; confusers protected)",
    }, art)
    handle.remove()
    print(f"  saved {art}  (CV F1 {cv_f1:.4f})")
    print("\n  NEXT (zero-GPU): evaluate vs shipped mlp_v5 on rgb_dataset_test + confusers + svanstrom + selcom\n"
          "    (point eval/diagnose_rgbtest_veto_mechanism.py MLP_V5 at this checkpoint, and re-run\n"
          "     the offline verifier matrix / per-size recall).")


def YOLO_load():
    from ultralytics import YOLO
    p = D.MODEL_PATHS["ft4_r3"]
    print(f"  detector (FT4 R3): {p}")
    return YOLO(p)


def _quickify(srcs):
    return [D.SourceConfig(name=s.name, path=s.path, stride=max(1, s.stride * 5), kind=s.kind,
                           target_drones=max(0, s.target_drones // 20), target_confusers=max(0, s.target_confusers // 20),
                           weight_drone=s.weight_drone, weight_confuser=s.weight_confuser,
                           filter_prefixes=s.filter_prefixes, match_rule=s.match_rule, imgsz=s.imgsz) for s in srcs]


if __name__ == "__main__":
    main()
