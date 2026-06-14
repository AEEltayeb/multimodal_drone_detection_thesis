"""build_train_eval_optimal.py — End-to-end pipeline.

Steps:
  1. Generate full-featured CSV (56 features) from existing caches + 32feat scene data
  2. Run feature selection on the full dataset (greedy forward)
  3. Train XGBoost classifier with optimal features
  4. Evaluate vs sa32 baseline
  5. Save model + metrics + comparison report

Usage:
    python -u classifier/build_train_eval_optimal.py
    python -u classifier/build_train_eval_optimal.py --skip-feature-selection  # use pilot results
    python -u classifier/build_train_eval_optimal.py --sample 5000            # quick test
"""
import argparse, csv, json, re, sys, time
from pathlib import Path

import joblib
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent

# ── Paths ────────────────────────────────────────────────────────
LEAN19_CSV   = REPO / "models/routers/lean19/fusion_dataset_lean19.csv"
LEAN19C_CSV  = REPO / "models/routers/lean19_v2_C/fusion_dataset.csv"
FEAT32_CSV   = REPO / "models/routers/retrained_v2_32feat/fusion_dataset.csv"
AUV_CACHE    = REPO / "models/routers/lean19/cache_antiuav.json"
SVAN_CACHE   = REPO / "models/routers/lean19/cache_svanstrom.json"
VIDEO_CACHE  = REPO / "docs/analysis/full_pipeline_ablations/cache"
SA32_METRICS = REPO / "models/routers/scene_aware_v3more_32feat/metrics.json"
SA32_MODEL   = REPO / "models/routers/scene_aware_v3more_32feat/model.joblib"

OUTPUT_DIR   = REPO / "models/routers/optimal_v1"

# ── Feature groups ───────────────────────────────────────────────
LEAN19_FEATS = [
    "rgb_max_conf", "ir_max_conf",
    "rgb_best_log_bbox_area", "ir_best_log_bbox_area",
    "rgb_best_aspect_ratio", "ir_best_aspect_ratio",
    "rgb_best_pos_y", "ir_best_pos_y",
    "rgb_best_local_contrast", "ir_best_local_contrast",
    "rgb_img_mean", "ir_img_mean", "rgb_img_std",
    "rgb_best_pos_x", "ir_best_pos_x",
    "rgb_best_dist_to_center", "ir_best_dist_to_center",
    "rgb_best_target_bg_delta", "ir_best_target_bg_delta",
]
XMODAL_FEATS = ["xmodal_centroid_dist", "xmodal_scale_ratio", "xmodal_conf_ratio"]
CACHE_DERIVED = [
    "rgb_mean_conf", "ir_mean_conf",
    "rgb_n_dets", "ir_n_dets",
    "rgb_detected", "ir_detected",
    "both_detect", "neither_detect", "rgb_only_detect", "ir_only_detect",
]
SCENE_FEATS = [
    "ir_img_std",
    "rgb_img_dynamic_range", "ir_img_dynamic_range",
    "rgb_img_entropy", "ir_img_entropy",
    "rgb_sky_ground_ratio", "ir_sky_ground_ratio",
    "rgb_edge_density", "ir_edge_density",
    "rgb_blurriness", "ir_blurriness",
]
DERIVED_FEATS = [
    "conf_diff", "conf_product", "conf_sum",
    "det_count_diff", "total_dets",
    "pos_euclidean_diff",
    "area_diff", "aspect_ratio_diff",
    "contrast_diff", "bg_delta_diff",
    "scene_entropy_mean", "scene_entropy_diff",
    "brightness_ratio",
]

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)

CONF_THR = 0.25

# ── Pilot results (8 features, from the 1k sample) ──────────────
PILOT_OPTIMAL = [
    "area_diff", "rgb_blurriness", "ir_best_local_contrast",
    "ir_best_aspect_ratio", "rgb_mean_conf", "rgb_best_log_bbox_area",
    "ir_best_dist_to_center", "xmodal_centroid_dist",
]


def seq_id(stem, src):
    m = SEQ_RE.match(str(stem))
    base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


def compute_derived_features(df):
    """Compute 13 derived interaction features from existing columns."""
    df["conf_diff"] = (df["rgb_max_conf"] - df["ir_max_conf"]).abs()
    df["conf_product"] = df["rgb_max_conf"] * df["ir_max_conf"]
    df["conf_sum"] = df["rgb_max_conf"] + df["ir_max_conf"]
    df["det_count_diff"] = (df["rgb_n_dets"] - df["ir_n_dets"]).abs()
    df["total_dets"] = df["rgb_n_dets"] + df["ir_n_dets"]
    dx = df["rgb_best_pos_x"] - df["ir_best_pos_x"]
    dy = df["rgb_best_pos_y"] - df["ir_best_pos_y"]
    df["pos_euclidean_diff"] = np.sqrt(dx**2 + dy**2)
    df["area_diff"] = (df["rgb_best_log_bbox_area"] - df["ir_best_log_bbox_area"]).abs()
    df["aspect_ratio_diff"] = (df["rgb_best_aspect_ratio"] - df["ir_best_aspect_ratio"]).abs()
    df["contrast_diff"] = (df["rgb_best_local_contrast"] - df["ir_best_local_contrast"]).abs()
    df["bg_delta_diff"] = (df["rgb_best_target_bg_delta"] - df["ir_best_target_bg_delta"]).abs()
    if "rgb_img_entropy" in df.columns and "ir_img_entropy" in df.columns:
        df["scene_entropy_mean"] = (df["rgb_img_entropy"] + df["ir_img_entropy"]) / 2
        df["scene_entropy_diff"] = (df["rgb_img_entropy"] - df["ir_img_entropy"]).abs()
    else:
        df["scene_entropy_mean"] = 0.0
        df["scene_entropy_diff"] = 0.0
    df["brightness_ratio"] = df["rgb_img_mean"] / df["ir_img_mean"].clip(lower=1.0)
    return df


def cheap_baselines(df):
    """Evaluate trivial trust policies."""
    y_true = df["trust_label"].values
    results = {}

    # Policy 1: always trust IR
    preds = np.where(df["ir_detected"].values == 1, 2, 1)
    preds = np.where(
        (df["rgb_detected"].values == 0) & (df["ir_detected"].values == 0), 0, preds)
    results["always_ir"] = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
    }

    # Policy 2: trust higher confidence modality
    rgb_conf = df["rgb_max_conf"].values
    ir_conf = df["ir_max_conf"].values
    preds = np.where(rgb_conf >= ir_conf, 1, 2)
    preds = np.where((rgb_conf > 0) & (ir_conf > 0), 3, preds)
    preds = np.where((rgb_conf == 0) & (ir_conf == 0), 0, preds)
    results["higher_conf"] = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
    }

    return results


# ═══════════════════════════════════════════════════════════════════
#  STEP 1: Build full-featured DataFrame
# ═══════════════════════════════════════════════════════════════════
def build_full_dataframe():
    print("=" * 70)
    print("STEP 1: Building full-featured DataFrame")
    print("=" * 70)

    # 1a. Load lean19 + xmodal
    print("\n  [1a] Loading lean19 + xmodal ...")
    df = pd.read_csv(LEAN19_CSV)
    df_c = pd.read_csv(LEAN19C_CSV)
    xm = df_c[["stem", "source"] + XMODAL_FEATS]
    df = df.merge(xm, on=["stem", "source"], how="left")
    for c in XMODAL_FEATS:
        df[c] = df[c].fillna(0.0)
    print(f"       {len(df):,} rows, xmodal merged")

    # 1b. Merge scene features from 32feat CSV
    print("  [1b] Merging scene features from 32feat CSV ...")
    df32 = pd.read_csv(FEAT32_CSV)
    available_scene = [c for c in SCENE_FEATS if c in df32.columns]
    df32_scene = df32[["stem", "source"] + available_scene].drop_duplicates(
        subset=["stem", "source"])
    df = df.merge(df32_scene, on=["stem", "source"], how="left")
    n_matched = df[available_scene[0]].notna().sum() if available_scene else 0
    print(f"       scene coverage: {n_matched:,}/{len(df):,} ({100*n_matched/len(df):.1f}%)")
    for c in available_scene:
        df[c] = df[c].fillna(0.0)

    # 1c. Compute cache-derived features
    print("  [1c] Computing detection-derived features from caches ...")
    t0 = time.time()
    auv_cache = json.load(open(AUV_CACHE))
    svan_cache = json.load(open(SVAN_CACHE))
    video_caches = {}
    for src in df["source"].unique():
        if src.startswith("video_"):
            rc = VIDEO_CACHE / f"{src}_selcom_1280_sz1280.json"
            ic = VIDEO_CACHE / f"{src}_ir_grayscale_sz640.json"
            if rc.exists() and ic.exists():
                video_caches[src] = (
                    json.load(open(rc))["dets"],
                    json.load(open(ic))["dets"],
                )
    print(f"       caches loaded: auv={len(auv_cache):,} svan={len(svan_cache):,} video={len(video_caches)}")

    rgb_mean_confs, ir_mean_confs = [], []
    rgb_n_dets_arr, ir_n_dets_arr = [], []

    for idx, row in df.iterrows():
        src, stem = str(row["source"]), str(row["stem"])
        rgb_dets, ir_dets = [], []
        if src == "antiuav":
            e = auv_cache.get(stem)
            if e:
                rgb_dets = e.get("rgb_dets", [])
                ir_dets = e.get("ir_dets", [])
        elif src == "svanstrom":
            e = svan_cache.get(stem)
            if e:
                rgb_dets = e.get("rgb_dets", [])
                ir_dets = e.get("ir_dets", [])
        elif src in video_caches:
            rd_c, id_c = video_caches[src]
            prefix = src + "_"
            img_stem = stem[len(prefix):] if stem.startswith(prefix) else stem
            rgb_dets = rd_c.get(img_stem, [])
            ir_dets = id_c.get(img_stem, [])

        rc = [d[4] for d in rgb_dets if d[4] >= CONF_THR]
        ic = [d[4] for d in ir_dets if d[4] >= CONF_THR]
        rgb_mean_confs.append(round(float(np.mean(rc)), 6) if rc else 0.0)
        ir_mean_confs.append(round(float(np.mean(ic)), 6) if ic else 0.0)
        rgb_n_dets_arr.append(len(rc))
        ir_n_dets_arr.append(len(ic))

        if (idx + 1) % 10000 == 0:
            print(f"       {idx+1:,}/{len(df):,} ...", flush=True)

    df["rgb_mean_conf"] = rgb_mean_confs
    df["ir_mean_conf"] = ir_mean_confs
    df["rgb_n_dets"] = rgb_n_dets_arr
    df["ir_n_dets"] = ir_n_dets_arr
    df["rgb_detected"] = (df["rgb_n_dets"] > 0).astype(int)
    df["ir_detected"] = (df["ir_n_dets"] > 0).astype(int)
    df["both_detect"] = ((df["rgb_detected"] == 1) & (df["ir_detected"] == 1)).astype(int)
    df["neither_detect"] = ((df["rgb_detected"] == 0) & (df["ir_detected"] == 0)).astype(int)
    df["rgb_only_detect"] = ((df["rgb_detected"] == 1) & (df["ir_detected"] == 0)).astype(int)
    df["ir_only_detect"] = ((df["rgb_detected"] == 0) & (df["ir_detected"] == 1)).astype(int)
    print(f"       cache-derived done in {time.time()-t0:.0f}s")

    # 1d. Compute derived interaction features
    print("  [1d] Computing derived interaction features ...")
    df = compute_derived_features(df)

    # Build feature list
    ALL_FEATS = LEAN19_FEATS + XMODAL_FEATS + CACHE_DERIVED + available_scene + DERIVED_FEATS
    seen = set()
    feat_cols = []
    for f in ALL_FEATS:
        if f not in seen:
            feat_cols.append(f)
            seen.add(f)

    # Fill NaN / inf
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0.0)
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], 0.0)

    # Sequence IDs
    df["_seq"] = df.apply(lambda r: seq_id(r["stem"], r["source"]), axis=1)

    print(f"\n  Total: {len(df):,} rows, {len(feat_cols)} features")
    return df, feat_cols


# ═══════════════════════════════════════════════════════════════════
#  STEP 2: Feature selection (greedy forward)
# ═══════════════════════════════════════════════════════════════════
def run_feature_selection(df, feat_cols, sample_n=0):
    print("\n" + "=" * 70)
    print("STEP 2: Feature selection (greedy forward)")
    print("=" * 70)

    if sample_n > 0 and sample_n < len(df):
        print(f"\n  Sampling {sample_n} rows (stratified) ...")
        df["_strat"] = df["source"] + "_" + df["trust_label"].astype(str)
        counts = df["_strat"].value_counts()
        fracs = (counts / counts.sum() * sample_n).clip(lower=1).astype(int)
        sampled = []
        for grp, n in fracs.items():
            sub = df[df["_strat"] == grp]
            sampled.append(sub.sample(n=min(n, len(sub)), random_state=42))
        df_s = pd.concat(sampled).reset_index(drop=True)
        df_s = df_s.drop(columns=["_strat"], errors="ignore")
        print(f"  sampled {len(df_s):,} rows")
    else:
        df_s = df.copy()
        print(f"  Using all {len(df_s):,} rows")

    X = df_s[feat_cols].values
    y = df_s["trust_label"].values
    groups = df_s["_seq"].values

    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.25, random_state=42)
                          .split(X, y, groups=groups))
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    print(f"  train: {len(X_tr):,}  test: {len(X_te):,}")
    print(f"  class dist (train): {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    base_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, n_jobs=1,
    )

    remaining = list(range(len(feat_cols)))
    selected = []
    history = []
    best_score = -1
    t0 = time.time()

    print(f"\n  Running greedy forward selection ({len(feat_cols)} features) ...")
    for step in range(len(feat_cols)):
        scores = []
        for fi in remaining:
            trial = selected + [fi]
            X_trial = X_tr[:, trial]
            model = base_model.__class__(**base_model.get_params())
            model.fit(X_trial, y_tr, verbose=False)
            pred = model.predict(X_te[:, trial])
            f1m = float(f1_score(y_te, pred, average="macro", zero_division=0))
            scores.append((fi, f1m))

        best_fi, best_f1 = max(scores, key=lambda x: x[1])
        selected.append(best_fi)
        remaining.remove(best_fi)
        history.append({
            "step": step + 1,
            "feature_added": feat_cols[best_fi],
            "f1_macro": round(best_f1, 4),
            "features_so_far": [feat_cols[i] for i in selected],
        })
        delta = best_f1 - best_score if best_score >= 0 else 0
        best_score = max(best_score, best_f1)
        marker = " ★" if best_f1 >= best_score else ""
        elapsed = time.time() - t0
        print(f"    step {step+1:2d}: +{feat_cols[best_fi]:35s}  "
              f"F1={best_f1:.4f}  Δ={delta:+.4f}{marker}  ({elapsed:.0f}s)")
        sys.stdout.flush()

        if step >= 15 and all(h["f1_macro"] <= history[-6]["f1_macro"] for h in history[-5:]):
            print(f"    → Plateau at step {step+1}, stopping early")
            break

    best_step = max(history, key=lambda h: h["f1_macro"])
    optimal_features = best_step["features_so_far"]
    print(f"\n  OPTIMAL: {best_step['step']} features, F1={best_step['f1_macro']:.4f}")
    for i, f in enumerate(optimal_features):
        print(f"    {i+1:2d}. {f}")

    return optimal_features, history


# ═══════════════════════════════════════════════════════════════════
#  STEP 3: Train classifier
# ═══════════════════════════════════════════════════════════════════
def train_classifier(df, feat_cols, tag="optimal_v1"):
    print("\n" + "=" * 70)
    print(f"STEP 3: Training classifier ({len(feat_cols)} features)")
    print("=" * 70)

    X = df[feat_cols].values
    y = df["trust_label"].values
    groups = df["_seq"].values

    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.25, random_state=42)
                          .split(X, y, groups=groups))
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    src_te = df.iloc[te_idx]["source"].values

    print(f"  train: {len(X_tr):,}  test: {len(X_te):,}")
    for val, name in LABEL_NAMES.items():
        n = (y_tr == val).sum()
        print(f"  Class {val} ({name}): {n:,} ({100*n/len(y_tr):.1f}%)")

    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, n_jobs=1,
    )

    print("\n  Training ...")
    t0 = time.time()
    model.fit(X_tr, y_tr, verbose=False)
    print(f"  Trained in {time.time()-t0:.1f}s")

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    acc = float(accuracy_score(y_te, y_pred))
    f1_macro = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))

    # Per-class AUC
    per_class_auc = {}
    for c in range(4):
        if (y_te == c).sum() > 0 and (y_te != c).sum() > 0:
            binary = (y_te == c).astype(int)
            per_class_auc[LABEL_NAMES[c]] = float(roc_auc_score(binary, y_prob[:, c]))
        else:
            per_class_auc[LABEL_NAMES[c]] = float("nan")

    print(f"\n  Overall:")
    print(f"    Accuracy:      {acc:.4f}")
    print(f"    F1 (macro):    {f1_macro:.4f}")
    print(f"    F1 (weighted): {f1_weighted:.4f}")
    print(f"    Per-class AUC: {per_class_auc}")

    # Classification report
    target_names = [LABEL_NAMES[i] for i in range(4)]
    report = classification_report(y_te, y_pred, target_names=target_names, zero_division=0)
    print(f"\n{report}")

    # Per-dataset breakdown
    per_dataset = {}
    print(f"  Per-dataset:")
    print(f"    {'dataset':<25s} {'n':>7s} {'acc':>7s} {'f1_m':>7s} {'f1_w':>7s}")
    for ds in np.unique(src_te):
        mask = src_te == ds
        ys, yp = y_te[mask], y_pred[mask]
        a = float(accuracy_score(ys, yp))
        fm = float(f1_score(ys, yp, average="macro", zero_division=0))
        fw = float(f1_score(ys, yp, average="weighted", zero_division=0))
        per_dataset[str(ds)] = {"n": int(len(ys)), "acc": a, "f1_macro": fm, "f1_weighted": fw}
        print(f"    {ds:<25s} {len(ys):>7d} {a:>7.4f} {fm:>7.4f} {fw:>7.4f}")

    # Feature importance
    importance = dict(zip(feat_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print(f"\n  Feature importance:")
    for feat, val in importance.items():
        bar = "█" * int(val * 100)
        print(f"    {feat:<35s} {val:.4f} {bar}")

    # Cheap baselines on test set
    df_te = df.iloc[te_idx].copy()
    baselines = cheap_baselines(df_te)

    metrics = {
        "tag": tag,
        "n_features": len(feat_cols),
        "features": feat_cols,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_auc": per_class_auc,
        "per_dataset": per_dataset,
        "feature_importance": {k: float(v) for k, v in importance.items()},
        "baselines": baselines,
    }

    return model, metrics


# ═══════════════════════════════════════════════════════════════════
#  STEP 4: Compare with sa32
# ═══════════════════════════════════════════════════════════════════
def compare_with_sa32(new_metrics):
    print("\n" + "=" * 70)
    print("STEP 4: Comparison vs sa32 baseline")
    print("=" * 70)

    if not SA32_METRICS.exists():
        print("  [SKIP] sa32 metrics not found")
        return {}

    sa32 = json.load(open(SA32_METRICS))
    n_feat = new_metrics['n_features']
    opt_key = f"optimal ({n_feat} feat)"

    comparison = {
        "metric": [],
        "sa32 (32 feat)": [],
        opt_key: [],
        "delta": [],
    }

    for metric in ["accuracy", "f1_macro", "f1_weighted"]:
        old_val = sa32.get(metric, 0)
        new_val = new_metrics.get(metric, 0)
        comparison["metric"].append(metric)
        comparison["sa32 (32 feat)"].append(f"{old_val:.4f}")
        comparison[opt_key].append(f"{new_val:.4f}")
        comparison["delta"].append(f"{new_val - old_val:+.4f}")

    # Print table
    print(f"\n  {'Metric':<16s} {'sa32 (32f)':>12s} {'optimal':>12s} {'delta':>10s}")
    print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*10}")
    for i in range(len(comparison["metric"])):
        print(f"  {comparison['metric'][i]:<16s} "
              f"{comparison['sa32 (32 feat)'][i]:>12s} "
              f"{comparison[opt_key][i]:>12s} "
              f"{comparison['delta'][i]:>10s}")

    # Per-dataset comparison
    if "per_dataset" in sa32 and "per_dataset" in new_metrics:
        print(f"\n  Per-dataset F1-macro comparison:")
        print(f"    {'dataset':<25s} {'sa32':>8s} {'optimal':>8s} {'delta':>8s}")
        for ds in sorted(set(list(sa32["per_dataset"].keys()) +
                             list(new_metrics["per_dataset"].keys()))):
            old = sa32["per_dataset"].get(ds, {}).get("f1_macro", None)
            new = new_metrics["per_dataset"].get(ds, {}).get("f1_macro", None)
            old_s = f"{old:.4f}" if old is not None else "N/A"
            new_s = f"{new:.4f}" if new is not None else "N/A"
            delta_s = f"{new - old:+.4f}" if old is not None and new is not None else "N/A"
            print(f"    {ds:<25s} {old_s:>8s} {new_s:>8s} {delta_s:>8s}")

    # Per-class AUC comparison
    if "per_class_auc" in sa32:
        print(f"\n  Per-class AUC comparison:")
        for cls_name in LABEL_NAMES.values():
            old = sa32["per_class_auc"].get(cls_name, None)
            new = new_metrics["per_class_auc"].get(cls_name, None)
            old_s = f"{old:.4f}" if old is not None else "N/A"
            new_s = f"{new:.4f}" if new is not None else "N/A"
            delta_s = f"{new - old:+.4f}" if old is not None and new is not None else "N/A"
            print(f"    {cls_name:<15s}  sa32={old_s}  optimal={new_s}  Δ={delta_s}")

    return comparison


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="Sample N rows for feature selection (0=full dataset)")
    ap.add_argument("--skip-feature-selection", action="store_true",
                    help="Use pilot results instead of running full feature selection")
    ap.add_argument("--features", type=str, default="",
                    help="Comma-separated feature list (overrides selection)")
    args = ap.parse_args()

    t_total = time.time()

    # Step 1: Build full DataFrame
    df, all_feat_cols = build_full_dataframe()

    # Save the full CSV for future use
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "fusion_dataset_full56.csv"
    save_cols = all_feat_cols + ["trust_label", "stem", "source", "_seq"]
    df[save_cols].to_csv(csv_path, index=False)
    print(f"\n  Full CSV saved: {csv_path}")

    # Step 2: Feature selection
    if args.features:
        # User-specified features
        optimal_features = [f.strip() for f in args.features.split(",")]
        print(f"\n  Using user-specified features: {optimal_features}")
        history = None
    elif args.skip_feature_selection:
        # Use pilot results
        optimal_features = PILOT_OPTIMAL
        print(f"\n  Using pilot optimal features: {optimal_features}")
        history = None
    else:
        sample_n = args.sample if args.sample > 0 else 0
        optimal_features, history = run_feature_selection(df, all_feat_cols, sample_n)

    # Validate features exist
    missing = [f for f in optimal_features if f not in df.columns]
    if missing:
        print(f"\n  [ERROR] Missing features: {missing}")
        return

    # Step 3: Train with optimal features
    model, metrics = train_classifier(df, optimal_features)

    # Add selection history to metrics
    if history:
        metrics["feature_selection_history"] = history

    # Step 4: Compare with sa32
    comparison = compare_with_sa32(metrics)
    metrics["sa32_comparison"] = comparison

    # Save artifacts
    print("\n" + "=" * 70)
    print("Saving artifacts")
    print("=" * 70)

    joblib.dump({"model": model, "features": optimal_features},
                OUTPUT_DIR / "model.joblib")
    json.dump(metrics, open(OUTPUT_DIR / "metrics.json", "w"), indent=2)
    json.dump({"optimal_features": optimal_features, "all_features": all_feat_cols},
              open(OUTPUT_DIR / "config.json", "w"), indent=2)

    print(f"  model.joblib saved")
    print(f"  metrics.json saved")
    print(f"  config.json saved")

    total_time = time.time() - t_total
    print(f"\n{'=' * 70}")
    print(f"DONE — Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
