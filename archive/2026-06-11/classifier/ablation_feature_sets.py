"""ablation_feature_sets.py — Compare feature subsets on full 65k dataset.

Trains 4 models side-by-side to find the best cost/performance tradeoff:
  1. all56:       All 56 features (upper bound)
  2. sa32_feats:  sa32's exact 32 features (baseline on our data)
  3. sa32_lite:   sa32 minus 10 expensive scene features (22 feats)
  4. sa32_lite+:  sa32_lite + area_diff + xmodal_centroid_dist (24 feats)
"""
import json, re, sys, time
from pathlib import Path

import joblib
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parent.parent
CSV_PATH  = REPO / "models/routers/optimal_v1/fusion_dataset_full56.csv"
SA32_METRICS = REPO / "models/routers/scene_aware_v3more_32feat/metrics.json"
OUTPUT_DIR = REPO / "models/routers/optimal_v1"

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)

# ── Feature sets ─────────────────────────────────────────────────
SA32_FEATURES = [
    "rgb_max_conf", "rgb_mean_conf", "ir_max_conf", "ir_mean_conf",
    "rgb_img_mean", "rgb_img_std",
    "rgb_img_dynamic_range", "rgb_img_entropy", "rgb_sky_ground_ratio",
    "rgb_edge_density", "rgb_blurriness",
    "ir_img_mean", "ir_img_std",
    "ir_img_dynamic_range", "ir_img_entropy", "ir_sky_ground_ratio",
    "ir_edge_density", "ir_blurriness",
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
    "rgb_best_pos_x", "rgb_best_pos_y", "rgb_best_dist_to_center",
    "rgb_best_local_contrast", "rgb_best_target_bg_delta",
    "ir_best_log_bbox_area", "ir_best_aspect_ratio",
    "ir_best_pos_x", "ir_best_pos_y", "ir_best_dist_to_center",
    "ir_best_local_contrast", "ir_best_target_bg_delta",
]

# Expensive scene features (require full image read + heavy compute)
EXPENSIVE_SCENE = [
    "rgb_img_dynamic_range", "rgb_img_entropy", "rgb_sky_ground_ratio",
    "rgb_edge_density", "rgb_blurriness",
    "ir_img_dynamic_range", "ir_img_entropy", "ir_sky_ground_ratio",
    "ir_edge_density", "ir_blurriness",
]

SA32_LITE = [f for f in SA32_FEATURES if f not in EXPENSIVE_SCENE]

SA32_LITE_PLUS = SA32_LITE + ["area_diff", "xmodal_centroid_dist"]

# Minimal "output-scores-only" set: detector confidences + one cross-modal agreement (5 feats)
META5_SCORES = [
    "rgb_max_conf", "rgb_mean_conf", "ir_max_conf", "ir_mean_conf",
    "xmodal_centroid_dist",
]
# Minimal "scores + cheap geometry" set (5 feats); no EXPENSIVE_SCENE -> drift-resistant
META5_GEO = [
    "rgb_max_conf", "ir_max_conf",
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
    "xmodal_centroid_dist",
]

# All 56 features (will be read from CSV columns)


def seq_id(stem, src):
    m = SEQ_RE.match(str(stem))
    base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


def train_and_eval(X_tr, y_tr, X_te, y_te, src_te, feat_names, tag):
    """Train XGBoost and return metrics dict."""
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, n_jobs=1,
    )

    t0 = time.time()
    model.fit(X_tr, y_tr, verbose=False)
    train_time = time.time() - t0

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    acc = float(accuracy_score(y_te, y_pred))
    f1m = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    f1w = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))

    per_class_auc = {}
    for c in range(4):
        if (y_te == c).sum() > 0 and (y_te != c).sum() > 0:
            binary = (y_te == c).astype(int)
            per_class_auc[LABEL_NAMES[c]] = float(roc_auc_score(binary, y_prob[:, c]))

    # Per-dataset
    per_dataset = {}
    for ds in np.unique(src_te):
        mask = src_te == ds
        ys, yp = y_te[mask], y_pred[mask]
        per_dataset[str(ds)] = {
            "n": int(len(ys)),
            "acc": float(accuracy_score(ys, yp)),
            "f1_macro": float(f1_score(ys, yp, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(ys, yp, average="weighted", zero_division=0)),
        }

    importance = dict(zip(feat_names, [float(v) for v in model.feature_importances_]))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    return {
        "tag": tag,
        "n_features": len(feat_names),
        "features": feat_names,
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "per_class_auc": per_class_auc,
        "per_dataset": per_dataset,
        "feature_importance": importance,
        "train_time_s": round(train_time, 1),
    }, model


def main():
    print("=" * 70)
    print("Feature Set Ablation — Full 65k dataset")
    print("=" * 70)

    # Load data
    print("\nLoading full56 CSV ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    # Sequence IDs for group split
    df["_seq"] = df.apply(lambda r: seq_id(r["stem"], r["source"]), axis=1)

    # All numeric feature columns (exclude metadata)
    all_feat_cols = [c for c in df.columns
                     if c not in ("stem", "source", "_seq", "trust_label")]

    # Define ablation variants
    variants = {
        "all56":       all_feat_cols,
        "sa32_feats":  SA32_FEATURES,
        "sa32_lite":   SA32_LITE,
        "sa32_lite+":  SA32_LITE_PLUS,
        "meta5_scores": META5_SCORES,
        "meta5_geo":    META5_GEO,
    }

    # Single train/test split (shared across all variants)
    y = df["trust_label"].values
    groups = df["_seq"].values
    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.25, random_state=42)
                          .split(df, y, groups=groups))
    y_tr, y_te = y[tr_idx], y[te_idx]
    src_te = df.iloc[te_idx]["source"].values

    print(f"  train: {len(tr_idx):,}  test: {len(te_idx):,}")
    print(f"  class dist (train): { {LABEL_NAMES[k]: int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))} }")

    # Load sa32 reference metrics
    sa32_ref = {}
    if SA32_METRICS.exists():
        sa32_ref = json.load(open(SA32_METRICS))
        print(f"\n  sa32 reference: acc={sa32_ref['accuracy']:.4f}  "
              f"F1m={sa32_ref['f1_macro']:.4f}  F1w={sa32_ref['f1_weighted']:.4f}")

    # Run ablation
    results = {}
    best_model = None
    best_tag = None

    for tag, feat_cols in variants.items():
        print(f"\n{'─' * 60}")
        print(f"  Training: {tag} ({len(feat_cols)} features)")
        print(f"{'─' * 60}")

        # Validate features exist
        missing = [f for f in feat_cols if f not in df.columns]
        if missing:
            print(f"  SKIP — missing: {missing}")
            continue

        X_tr = df.iloc[tr_idx][feat_cols].values
        X_te = df.iloc[te_idx][feat_cols].values

        metrics, model = train_and_eval(X_tr, y_tr, X_te, y_te, src_te, feat_cols, tag)
        results[tag] = metrics

        # Print results
        print(f"  acc={metrics['accuracy']:.4f}  F1m={metrics['f1_macro']:.4f}  "
              f"F1w={metrics['f1_weighted']:.4f}  ({metrics['train_time_s']:.1f}s)")

        # Per-dataset breakdown for key datasets
        for ds in ["antiuav", "svanstrom"]:
            ds_metrics = [v for k, v in metrics["per_dataset"].items() if ds in k]
            if ds_metrics:
                dm = ds_metrics[0]
                print(f"    {ds}: acc={dm['acc']:.4f} F1m={dm['f1_macro']:.4f}")

        # Top 5 features
        top5 = list(metrics["feature_importance"].items())[:5]
        print(f"  Top 5: {', '.join(f'{f}({v:.3f})' for f, v in top5)}")

        # Track best
        if best_model is None or metrics["f1_macro"] > results.get(best_tag, {}).get("f1_macro", 0):
            best_model = model
            best_tag = tag

    # ── Summary table ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 70}")

    header = f"  {'variant':<16s} {'feats':>5s} {'acc':>7s} {'F1m':>7s} {'F1w':>7s} {'time':>6s}"
    if sa32_ref:
        header += f" {'vs sa32':>8s}"
    print(header)
    print(f"  {'-'*16} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*6}" +
          (f" {'-'*8}" if sa32_ref else ""))

    # sa32 reference line
    if sa32_ref:
        print(f"  {'sa32 (ref)':<16s} {'32':>5s} {sa32_ref['accuracy']:>7.4f} "
              f"{sa32_ref['f1_macro']:>7.4f} {sa32_ref['f1_weighted']:>7.4f} {'N/A':>6s} {'---':>8s}")

    for tag, m in results.items():
        line = (f"  {tag:<16s} {m['n_features']:>5d} {m['accuracy']:>7.4f} "
                f"{m['f1_macro']:>7.4f} {m['f1_weighted']:>7.4f} {m['train_time_s']:>5.1f}s")
        if sa32_ref:
            delta = m['f1_macro'] - sa32_ref['f1_macro']
            line += f" {delta:>+7.4f}"
        print(line)

    # Per-dataset comparison
    print(f"\n  Per-dataset F1-macro:")
    ds_order = ["antiuav", "svanstrom"]
    # Get all video datasets from results
    for tag, m in results.items():
        for ds in m["per_dataset"]:
            if ds.startswith("video_") and ds not in ds_order:
                ds_order.append(ds)
    
    hdr = f"    {'dataset':<40s}"
    for tag in results:
        hdr += f" {tag:>12s}"
    print(hdr)

    for ds in ds_order:
        line = f"    {ds:<40s}"
        for tag, m in results.items():
            val = m["per_dataset"].get(ds, {}).get("f1_macro", None)
            line += f" {val:>12.4f}" if val is not None else f" {'N/A':>12s}"
        print(line)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUTPUT_DIR / "ablation_results.json", "w"), indent=2)

    # Save best model
    if best_model and best_tag:
        best_feats = variants[best_tag]
        joblib.dump({"model": best_model, "features": best_feats},
                    OUTPUT_DIR / f"model_{best_tag}.joblib")
        print(f"\n  Best model saved: model_{best_tag}.joblib")

    print(f"\n  Results saved: ablation_results.json")
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
