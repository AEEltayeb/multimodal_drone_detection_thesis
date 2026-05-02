"""
train_failure_models.py — Train four XGBoost failure-prediction models:

  rgb_fn, rgb_fp, ir_fn, ir_fp

Inputs: {modality}_fn_dataset.csv and {modality}_fp_dataset.csv produced by
build_failure_datasets.py.

Splits: sequence-level train/test. Sequence ID is extracted from the image stem
using a regex (strip trailing frame-number suffix). Stratified by source_dataset
so each split has proportional dataset coverage.

Outputs (under runs/reliability/failure_models/):
  {modality}_{kind}_model.joblib
  {modality}_{kind}_metrics.json
  {modality}_{kind}_feature_importance.png

Usage:
    python train_failure_models.py
    python train_failure_models.py --only rgb --kind fn
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "failure_models"
OUT_DIR    = DATA_DIR


# ── FEATURE SETS ───────────────────────────────────────────────────

GLOBAL_FEATURES = [
    "img_mean", "img_std", "img_dynamic_range", "img_entropy",
    "sky_ground_ratio", "edge_density", "blurriness",
]

TARGET_FEATURES = [
    "log_bbox_area", "aspect_ratio", "pos_x", "pos_y", "dist_to_center",
    "local_contrast", "target_bg_delta",
]

FN_FEATURES = GLOBAL_FEATURES + TARGET_FEATURES  # 14 features
FP_FEATURES = GLOBAL_FEATURES                     # 7 features


# ── SEQUENCE ID EXTRACTION ─────────────────────────────────────────

SEQ_SUFFIX_RE = re.compile(
    r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared|_ir|_rgb)?$",
    re.IGNORECASE,
)


def extract_sequence_id(stem, source_dataset):
    """Extract a sequence ID from an image stem.

    Handles common patterns like:
      video_001_f000123                  -> video_001
      20190925_111757_1_9_visible_f000433 -> 20190925_111757_1_9_visible (Anti-UAV)
      IR_DRONE_001_f000000_visible       -> IR_DRONE_001  (Svanstrom — trailing modality)
      IR_DRONE_005_f000123_infrared      -> IR_DRONE_005  (Svanstrom — trailing modality)
      building_66_000001                 -> building_66   (CST / VTUAV)
      000001                              -> fallback (whole stem)
    """
    m = SEQ_SUFFIX_RE.match(stem)
    if m:
        base = m.group(1).rstrip("_")
        if base:
            return f"{source_dataset}::{base}"
    # Fallback: whole stem is one sequence (safe — may make split coarse)
    return f"{source_dataset}::{stem}"


# ── TRAINING ───────────────────────────────────────────────────────

def sequence_split(df, test_size=0.25, random_state=42):
    """Split rows by sequence ID, stratify by source_dataset implicitly via groups."""
    groups = df["sequence_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, df["label"], groups=groups))
    train_seqs = set(df.iloc[train_idx]["sequence_id"])
    test_seqs  = set(df.iloc[test_idx]["sequence_id"])
    assert len(train_seqs & test_seqs) == 0, "Sequence leakage!"
    return train_idx, test_idx


def find_best_threshold(y_true, y_prob):
    """Find threshold maximizing F1."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # F1 array aligned with thresholds (last element of P/R has no threshold)
    p = precisions[:-1]
    r = recalls[:-1]
    f1s = 2 * p * r / np.clip(p + r, 1e-9, None)
    if len(f1s) == 0 or thresholds is None or len(thresholds) == 0:
        return 0.5
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx])


def train_one(modality, kind, df, feature_cols):
    """Train one XGBoost model. kind in {'fn','fp'}."""
    tag = f"{modality}_{kind}"
    print(f"\n{'=' * 70}")
    print(f"Training {tag.upper()}  (features: {len(feature_cols)})")
    print(f"{'=' * 70}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Positives (label=1): {(df['label'] == 1).sum():,} "
          f"({(df['label'] == 1).mean() * 100:.2f}%)")

    # Sequence ID
    df = df.copy()
    df["sequence_id"] = df.apply(
        lambda r: extract_sequence_id(r["stem"], r["source_dataset"]), axis=1
    )
    n_seqs = df["sequence_id"].nunique()
    print(f"  Sequences: {n_seqs:,}")

    # Sequence-level split
    train_idx, test_idx = sequence_split(df, test_size=0.25, random_state=42)
    X_train = df.iloc[train_idx][feature_cols].values
    X_test  = df.iloc[test_idx][feature_cols].values
    y_train = df.iloc[train_idx]["label"].values
    y_test  = df.iloc[test_idx]["label"].values
    src_test = df.iloc[test_idx]["source_dataset"].values

    print(f"  Train rows: {len(X_train):,} "
          f"({y_train.sum():,} pos, {(y_train == 0).sum():,} neg)")
    print(f"  Test rows:  {len(X_test):,} "
          f"({y_test.sum():,} pos, {(y_test == 0).sum():,} neg)")

    # Handle class imbalance
    neg = (y_train == 0).sum()
    pos = max(1, (y_train == 1).sum())
    scale_pos_weight = neg / pos

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)

    # Eval
    y_prob = model.predict_proba(X_test)[:, 1]

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = float("nan")
    ap = float(average_precision_score(y_test, y_prob))
    thresh = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= thresh).astype(int)

    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall    = float(recall_score(y_test, y_pred, zero_division=0))
    f1        = float(f1_score(y_test, y_pred, zero_division=0))

    print(f"\n  Overall test metrics:")
    print(f"    AUC:       {auc:.4f}")
    print(f"    AP:        {ap:.4f}")
    print(f"    Threshold: {thresh:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1:        {f1:.4f}")

    # Per-dataset breakdown
    per_dataset = {}
    print(f"\n  Per-dataset (at threshold={thresh:.3f}):")
    print(f"    {'dataset':<25s} {'n':>7s} {'pos':>6s} {'AUC':>7s} "
          f"{'P':>7s} {'R':>7s} {'F1':>7s}")
    for tag_src in np.unique(src_test):
        mask = src_test == tag_src
        ys = y_test[mask]
        ps = y_prob[mask]
        n_ = len(ys)
        n_pos = int(ys.sum())
        if len(np.unique(ys)) < 2:
            auc_s = float("nan")
        else:
            auc_s = float(roc_auc_score(ys, ps))
        preds = (ps >= thresh).astype(int)
        p_s = float(precision_score(ys, preds, zero_division=0))
        r_s = float(recall_score(ys, preds, zero_division=0))
        f_s = float(f1_score(ys, preds, zero_division=0))
        per_dataset[str(tag_src)] = {
            "n": int(n_), "n_pos": n_pos,
            "auc": auc_s, "precision": p_s, "recall": r_s, "f1": f_s,
        }
        print(f"    {tag_src:<25s} {n_:>7d} {n_pos:>6d} "
              f"{auc_s:>7.4f} {p_s:>7.4f} {r_s:>7.4f} {f_s:>7.4f}")

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print(f"\n  Feature importance:")
    for feat, val in importance.items():
        print(f"    {feat:<22s} {val:.4f}")

    # Save artifacts
    model_path = OUT_DIR / f"{tag}_model.joblib"
    metrics_path = OUT_DIR / f"{tag}_metrics.json"
    fig_path = OUT_DIR / f"{tag}_feature_importance.png"

    joblib.dump({"model": model, "features": feature_cols, "threshold": thresh},
                model_path)

    metrics = {
        "modality": modality,
        "kind": kind,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_auc": auc,
        "test_ap": ap,
        "threshold": thresh,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "per_dataset": per_dataset,
        "feature_importance": {k: float(v) for k, v in importance.items()},
        "scale_pos_weight": float(scale_pos_weight),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(feature_cols) + 1))
    feats = list(importance.keys())
    vals = list(importance.values())
    ax.barh(feats[::-1], vals[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"{tag.upper()} feature importance (AUC={auc:.3f}, F1={f1:.3f})")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)

    print(f"\n  Saved: {model_path.name}, {metrics_path.name}, {fig_path.name}")
    return metrics


# ── MAIN ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["rgb", "ir"],
                        help="Train only one modality")
    parser.add_argument("--kind", choices=["fn", "fp"],
                        help="Train only one model kind")
    args = parser.parse_args()

    modalities = [args.only] if args.only else ["rgb", "ir"]
    kinds = [args.kind] if args.kind else ["fn", "fp"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for modality in modalities:
        for kind in kinds:
            csv_path = DATA_DIR / f"{modality}_{kind}_dataset.csv"
            if not csv_path.exists():
                print(f"[SKIP] {csv_path} not found. "
                      f"Run build_failure_datasets.py first.")
                continue
            print(f"\nLoading {csv_path.name}...", end="", flush=True)
            df = pd.read_csv(csv_path)
            print(f" {len(df):,} rows")

            feature_cols = FN_FEATURES if kind == "fn" else FP_FEATURES
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                print(f"  [ERROR] missing columns: {missing}")
                continue

            metrics = train_one(modality, kind, df, feature_cols)
            all_metrics[f"{modality}_{kind}"] = metrics

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'model':<12s} {'n_test':>8s} {'AUC':>7s} {'AP':>7s} "
          f"{'P':>7s} {'R':>7s} {'F1':>7s}")
    for name, m in all_metrics.items():
        print(f"  {name:<12s} {m['n_test']:>8d} {m['test_auc']:>7.4f} "
              f"{m['test_ap']:>7.4f} {m['test_precision']:>7.4f} "
              f"{m['test_recall']:>7.4f} {m['test_f1']:>7.4f}")


if __name__ == "__main__":
    main()
