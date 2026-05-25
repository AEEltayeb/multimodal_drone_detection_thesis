"""
train_lean13_classifier.py - Train the Lean-13 4-class XGBoost fusion classifier.

Same XGBoost config and sequence-split as train_retrained_v2_classifier.py
so the model is apples-to-apples comparable to retrained_v2_32feat.

Usage:
    python classifier/train_lean13_classifier.py
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


FEATURE_COLS = [
    "rgb_max_conf", "ir_max_conf",
    "rgb_best_log_bbox_area", "ir_best_log_bbox_area",
    "rgb_best_aspect_ratio", "ir_best_aspect_ratio",
    "rgb_best_pos_y", "ir_best_pos_y",
    "rgb_best_local_contrast", "ir_best_local_contrast",
    "rgb_img_mean", "ir_img_mean",
    "rgb_img_std",
]

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)


def extract_sequence_id(stem, source):
    m = SEQ_RE.match(stem)
    base = m.group(1).rstrip("_") if m else stem
    return f"{source}::{base}"


def sequence_split(df, test_size=0.25, seed=42):
    groups = df["sequence_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, df["trust_label"], groups=groups))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="classifier/fusion_models/lean13/fusion_dataset_lean13.csv")
    ap.add_argument("--output-dir", default="classifier/fusion_models/lean13")
    ap.add_argument("--detection-only", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = repo / csv_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns: {missing}")

    for t in sorted(df["trust_label"].unique()):
        n = (df["trust_label"] == t).sum()
        print(f"  {LABEL_NAMES.get(t,'?'):15s}: {n:,}")

    if args.detection_only:
        before = len(df)
        df = df[(df["rgb_max_conf"] > 0) | (df["ir_max_conf"] > 0)].copy()
        print(f"  detection-only: {before:,} -> {len(df):,}")

    df["sequence_id"] = df.apply(
        lambda r: extract_sequence_id(str(r["stem"]), str(r["source"])), axis=1)
    print(f"  sequences: {df['sequence_id'].nunique()}")

    train_idx, test_idx = sequence_split(df)
    X_train = df.iloc[train_idx][FEATURE_COLS].values
    X_test = df.iloc[test_idx][FEATURE_COLS].values
    y_train = df.iloc[train_idx]["trust_label"].values
    y_test = df.iloc[test_idx]["trust_label"].values
    src_test = df.iloc[test_idx]["source"].values
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    print("\nTraining XGBoost (400 trees, depth 6, lr 0.05)...")
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    print(f"\n  Accuracy:      {acc:.4f}")
    print(f"  F1 (macro):    {f1m:.4f}")
    print(f"  F1 (weighted): {f1w:.4f}")
    print()
    print(classification_report(y_test, y_pred,
                                target_names=[LABEL_NAMES[i] for i in range(4)],
                                zero_division=0))

    per_dataset = {}
    print("  Per-source:")
    for ds in np.unique(src_test):
        m = src_test == ds
        a = float(accuracy_score(y_test[m], y_pred[m]))
        fm = float(f1_score(y_test[m], y_pred[m], average="macro", zero_division=0))
        per_dataset[ds] = {"n": int(m.sum()), "acc": a, "f1_macro": fm}
        print(f"    {ds:<50s} n={m.sum():>6d}  acc={a:.4f}  f1m={fm:.4f}")

    importance = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\n  Feature importance:")
    for feat, imp in ranked:
        print(f"    {feat:<32s} {imp:.4f}  {'#' * int(imp * 50)}")

    joblib.dump({"model": model, "features": FEATURE_COLS},
                out_dir / "model.joblib")
    print(f"\n  Saved: {out_dir / 'model.joblib'}")

    metrics = {
        "tag": "fusion_lean13",
        "n_features": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "per_dataset": per_dataset,
        "feature_importance": importance,
        "detection_only": args.detection_only,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {out_dir / 'metrics.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    feats = [f for f, _ in ranked]; imps = [i for _, i in ranked]
    fig, ax = plt.subplots(figsize=(10, max(4, len(feats) * 0.45)))
    ax.barh(range(len(feats)), imps, color="#4C72B0", edgecolor="white")
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=11)
    ax.invert_yaxis(); ax.set_xlabel("Feature importance (gain)")
    ax.set_title(f"Lean-13 fusion - acc={acc:.3f}, F1m={f1m:.3f}")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(str(out_dir / "feature_importance.png"), dpi=200)
    plt.close(fig)
    print(f"  Saved: {out_dir / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
