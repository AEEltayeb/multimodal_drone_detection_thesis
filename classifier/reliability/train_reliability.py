"""
train_reliability.py — Train two XGBoost reliability classifiers,
one per modality, that predict P(detection is correct).

Reads:
  runs/reliability/rgb_reliability_dataset.csv
  runs/reliability/ir_reliability_dataset.csv

Outputs per modality:
  runs/reliability/{modality}_reliability_model.joblib
  runs/reliability/{modality}_feature_importance.png
  runs/reliability/{modality}_metrics.json

Usage:
    python train_reliability.py
    python train_reliability.py --only rgb
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, precision_recall_curve
)
from xgboost import XGBClassifier

# ── PATHS ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent / "runs" / "reliability"

FEATURE_COLS = [
    "conf", "box_area_norm", "aspect_ratio",
    "box_w_norm", "box_h_norm",
    "box_center_x", "box_center_y",
    "n_dets", "conf_rank",
    "conf_2nd", "conf_margin", "conf_mean_frame",
]

# Datasets that should be in the TEST set (never trained on)
# These are the most important for honest evaluation
HOLDOUT_DATASETS = {
    "rgb": ["antiuav_test_rgb", "svanstrom_rgb"],
    "ir":  ["antiuav_test_ir", "svanstrom_ir", "cst_antiuav_test"],
}

TRAIN_DATASETS = {
    "rgb": ["rgb_dataset_val", "rgb_dataset_test", "antiuav_val_rgb"],
    "ir":  ["ir_dset_final_val", "ir_dset_final_test", "antiuav_val_ir"],
}


def train_modality(modality):
    """Train reliability classifier for one modality."""
    csv_path = BASE_DIR / f"{modality}_reliability_dataset.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {csv_path} not found")
        return

    print(f"\n{'='*70}")
    print(f"Training {modality.upper()} reliability classifier")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} detections from {csv_path.name}")
    print(f"  TP: {(df['label']==1).sum()}, FP: {(df['label']==0).sum()}")

    # Split by dataset
    train_tags = TRAIN_DATASETS[modality]
    test_tags = HOLDOUT_DATASETS[modality]

    train_df = df[df["source_dataset"].isin(train_tags)].copy()
    test_df = df[df["source_dataset"].isin(test_tags)].copy()

    # Check for datasets not in either split
    all_tags = set(train_tags) | set(test_tags)
    orphan = df[~df["source_dataset"].isin(all_tags)]
    if len(orphan) > 0:
        print(f"  [WARN] {len(orphan)} rows from unassigned datasets -- adding to train")
        train_df = pd.concat([train_df, orphan])

    print(f"\n  Train: {len(train_df)} detections from {train_tags}")
    print(f"    TP: {(train_df['label']==1).sum()} "
          f"({(train_df['label']==1).mean()*100:.1f}%)")
    print(f"    FP: {(train_df['label']==0).sum()} "
          f"({(train_df['label']==0).mean()*100:.1f}%)")

    print(f"\n  Test:  {len(test_df)} detections from {test_tags}")
    print(f"    TP: {(test_df['label']==1).sum()} "
          f"({(test_df['label']==1).mean()*100:.1f}%)")
    print(f"    FP: {(test_df['label']==0).sum()} "
          f"({(test_df['label']==0).mean()*100:.1f}%)")

    # Prepare features
    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df["label"].values

    # Class balance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale = n_neg / max(n_pos, 1)
    print(f"\n  scale_pos_weight: {scale:.2f} (neg/pos ratio)")

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold via F1 on train
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_prob_train)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    print(f"\n  Optimal threshold (from train PR curve): {best_threshold:.4f}")

    # Evaluate on test
    y_pred_test = (y_prob_test >= best_threshold).astype(int)

    print(f"\n  === TEST SET RESULTS ===")
    print(classification_report(y_test, y_pred_test, target_names=["FP", "TP"], digits=4))

    test_f1 = f1_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test)
    test_rec = recall_score(y_test, y_pred_test)
    test_auc = roc_auc_score(y_test, y_prob_test) if len(np.unique(y_test)) > 1 else 0.0

    print(f"  AUC-ROC: {test_auc:.4f}")

    # Per-dataset breakdown
    print(f"\n  Per-dataset test performance:")
    print(f"    {'dataset':<25s} {'total':>7s} {'TP':>5s} {'FP':>5s} "
          f"{'Prec':>7s} {'Rec':>7s} {'F1':>7s}")
    per_dataset_metrics = {}
    for tag in test_tags:
        mask = test_df["source_dataset"] == tag
        if mask.sum() == 0:
            continue
        y_t = y_test[mask]
        y_p = y_pred_test[mask]
        tp = ((y_t == 1) & (y_p == 1)).sum()
        fp_pred = ((y_t == 0) & (y_p == 1)).sum()
        prec = tp / max(tp + fp_pred, 1)
        rec = tp / max((y_t == 1).sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"    {tag:<25s} {mask.sum():>7d} {(y_t==1).sum():>5d} "
              f"{(y_t==0).sum():>5d} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")
        per_dataset_metrics[tag] = {
            "total": int(mask.sum()),
            "actual_tp": int((y_t == 1).sum()),
            "actual_fp": int((y_t == 0).sum()),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  Feature importance:")
    for i in sorted_idx:
        bar = "#" * int(importances[i] * 50)
        print(f"    {FEATURE_COLS[i]:<20s} {importances[i]:.4f}  {bar}")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(FEATURE_COLS))
    ax.barh(y_pos, importances[sorted_idx[::-1]],
            color="#4CAF50" if modality == "rgb" else "#2196F3")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([FEATURE_COLS[i] for i in sorted_idx[::-1]])
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"{modality.upper()} Reliability Classifier - Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    fig_path = BASE_DIR / f"{modality}_feature_importance.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved feature importance -> {fig_path}")

    # Save model bundle
    bundle = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "threshold": best_threshold,
        "modality": modality,
        "train_datasets": train_tags,
        "test_datasets": test_tags,
        "test_f1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_auc": test_auc,
    }
    model_path = BASE_DIR / f"{modality}_reliability_model.joblib"
    joblib.dump(bundle, model_path)
    print(f"  Saved model -> {model_path}")

    # Save metrics
    metrics = {
        "modality": modality,
        "threshold": best_threshold,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "test_f1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_auc": test_auc,
        "per_dataset": per_dataset_metrics,
        "feature_importance": {FEATURE_COLS[i]: float(importances[i])
                               for i in sorted_idx},
    }
    metrics_path = BASE_DIR / f"{modality}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics -> {metrics_path}")

    return bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["rgb", "ir"],
                        help="Train only one modality")
    args = parser.parse_args()

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for modality in ["rgb", "ir"]:
        if args.only and modality != args.only:
            continue
        bundle = train_modality(modality)
        if bundle:
            results[modality] = bundle

    if len(results) == 2:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Metric':<20s} {'RGB':>10s} {'IR':>10s}")
        print(f"  {'-'*40}")
        for key in ["test_f1", "test_precision", "test_recall", "test_auc"]:
            label = key.replace("test_", "").upper()
            print(f"  {label:<20s} "
                  f"{results['rgb'][key]:>10.4f} "
                  f"{results['ir'][key]:>10.4f}")

    print(f"\n  Done. Ready for eval_fusion.py")


if __name__ == "__main__":
    main()
