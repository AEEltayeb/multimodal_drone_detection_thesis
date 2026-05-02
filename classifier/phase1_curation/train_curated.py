"""
train_curated.py — Train XGBoost on the curated Phase 1 frame-level dataset
with sequence-level splitting and per-stratum evaluation.

Key differences from classifier/train_classifier.py:
  * Sequence-level split (via split_sequences.py) instead of per-stem split.
  * Hard-coded honest feature set (no dataset-tag features).
  * Per-stratum test metrics: breakdown by source × {category | lighting}.
  * Ablation flags for Phase 1 experiments.

Usage:
    python train_curated.py
    python train_curated.py --tag baseline
    python train_curated.py --tag no_svan_drones --drop-svanstrom-drones
    python train_curated.py --tag with_brightness --include-brightness
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, average_precision_score
from xgboost import XGBClassifier

# Local imports
SCRIPT_DIR = Path(__file__).resolve().parent
CLASSIFIER_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from split_sequences import split_by_sequence, print_split_summary
from curate_dataset import FEATURE_COLS as HONEST_FEATURE_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sweep_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    if len(f1s) == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_idx = int(np.argmax(f1s))
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1": float(f1s[best_idx]),
    }


def per_stratum_metrics(df_test: pd.DataFrame, y_test, y_prob, threshold):
    """
    Break down test metrics by:
      - source × category (primary, works for Svanstrom)
      - source × lighting (primary, works for Anti-UAV)
    Each row lists: rows, pos, neg, TP, FP, FN, TN, precision, recall, F1,
    FP_rate.
    """
    preds = (y_prob >= threshold).astype(int)

    def _one(group_df, group_name):
        idx = group_df.index.values
        y = y_test[idx]
        p = preds[idx]
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return {
            "group": group_name,
            "rows": len(y),
            "pos": int(y.sum()),
            "neg": int(len(y) - y.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec, "f1": f1, "fp_rate": fpr,
        }

    # Reset index so positional indexing lines up with y_test
    df_test = df_test.reset_index(drop=True)
    y_test = np.asarray(y_test)
    y_prob = np.asarray(y_prob)
    preds = (y_prob >= threshold).astype(int)

    rows = []
    # source × category
    for (src, cat), sub in df_test.groupby(["source", "category"]):
        rows.append(_one(sub, f"{src} / {cat}"))
    # source × lighting
    for (src, lit), sub in df_test.groupby(["source", "lighting"]):
        rows.append(_one(sub, f"{src} / light={lit}"))

    return rows


def print_stratum_table(rows):
    header = f"  {'group':<35s} {'rows':>7s} {'pos':>7s} {'P':>7s} {'R':>7s} {'F1':>7s} {'FP/tot':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {r['group']:<35s} {r['rows']:>7d} {r['pos']:>7d} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
              f"{r['fp_rate']:>8.4f}")


def plot_feature_importance(feature_cols, importances, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    feats = [f for f, _ in ranked]
    imps = [i for _, i in ranked]

    fig, ax = plt.subplots(figsize=(10, max(4, len(feats) * 0.4)))
    y_pos = range(len(feats))
    bars = ax.barh(y_pos, imps, color="#4C72B0", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (gain)", fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, imp in zip(bars, imps):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{imp:.3f}", va="center", fontsize=9, color="#333")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CLASSIFIER_DIR / "config.yaml"))
    parser.add_argument("--csv", default="runs/curated_frame_dataset.csv")
    parser.add_argument("--tag", default="baseline",
                        help="Suffix for output artifacts (e.g. 'baseline', 'no_svan_drones')")
    parser.add_argument("--drop-svanstrom-drones", action="store_true",
                        help="Ablation: drop Svanstrom DRONE positives (tests IR leakage)")
    parser.add_argument("--svanstrom-only", action="store_true",
                        help="Ablation: train on Svanstrom rows only")
    parser.add_argument("--anti-uav-only", action="store_true",
                        help="Ablation: train on Anti-UAV rows only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = CLASSIFIER_DIR / cfg.get("output_dir", "runs") / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = CLASSIFIER_DIR / csv_path

    print("=" * 60)
    print(f"Training Phase 1 | tag={args.tag}")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Apply ablations
    if args.drop_svanstrom_drones:
        mask = ~((df["source"] == "svanstrom") & (df["category"] == "DRONE"))
        dropped = int((~mask).sum())
        df = df[mask].reset_index(drop=True)
        print(f"  Ablation: dropped {dropped} Svanstrom DRONE rows")
    if args.svanstrom_only:
        df = df[df["source"] == "svanstrom"].reset_index(drop=True)
        print(f"  Ablation: Svanstrom only, {len(df)} rows")
    if args.anti_uav_only:
        df = df[df["source"] == "anti_uav"].reset_index(drop=True)
        print(f"  Ablation: Anti-UAV only, {len(df)} rows")

    print(f"  After ablations: {len(df)} rows, "
          f"{df['label'].sum()} pos, {len(df) - df['label'].sum()} neg")

    # Sequence-level split
    masks = split_by_sequence(df, train_frac=0.70, val_frac=0.15, test_frac=0.15,
                              random_state=42, stratify_col="source")
    print()
    print_split_summary(df, masks)

    # Features
    feature_cols = HONEST_FEATURE_COLS
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    X_train, y_train = X[masks["train"]], y[masks["train"]]
    X_val, y_val = X[masks["val"]], y[masks["val"]]
    X_test, y_test = X[masks["test"]], y[masks["test"]]
    df_test = df[masks["test"]].reset_index(drop=True)

    print(f"\n  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    if len(y_train) == 0 or y_train.sum() == 0 or (len(y_train) - y_train.sum()) == 0:
        print("\n  ERROR: Train set has no positives or no negatives. Abort.")
        return

    # Train XGBoost
    xgb_cfg = cfg.get("xgboost", {})
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    model = XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 200),
        max_depth=xgb_cfg.get("max_depth", 4),
        learning_rate=xgb_cfg.get("learning_rate", 0.1),
        min_child_weight=xgb_cfg.get("min_child_weight", 10),
        subsample=xgb_cfg.get("subsample", 0.8),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
        scale_pos_weight=float(n_neg / n_pos) if n_pos > 0 else 1.0,
        eval_metric=xgb_cfg.get("eval_metric", "aucpr"),
        random_state=42,
        verbosity=0,
    )
    print("\n  Training XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Threshold on val
    y_prob_val = model.predict_proba(X_val)[:, 1]
    best = sweep_threshold(y_val, y_prob_val)
    print(f"  Val best: P={best['precision']:.4f} R={best['recall']:.4f} "
          f"F1={best['f1']:.4f} @ t={best['threshold']:.3f}")

    # Test
    y_prob_test = model.predict_proba(X_test)[:, 1]
    preds_test = (y_prob_test >= best["threshold"]).astype(int)

    tp = int(((preds_test == 1) & (y_test == 1)).sum())
    fp = int(((preds_test == 1) & (y_test == 0)).sum())
    fn = int(((preds_test == 0) & (y_test == 1)).sum())
    tn = int(((preds_test == 0) & (y_test == 0)).sum())
    test_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    test_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    test_f1 = 2 * test_p * test_r / (test_p + test_r) if (test_p + test_r) > 0 else 0.0
    aucpr = float(average_precision_score(y_test, y_prob_test)) if len(set(y_test)) > 1 else 0.0

    print(f"\n  Test (overall): P={test_p:.4f} R={test_r:.4f} F1={test_f1:.4f} "
          f"AUC-PR={aucpr:.4f}")
    print(f"    TP={tp} FP={fp} FN={fn} TN={tn}")

    # Per-stratum
    print(f"\n  Per-stratum test metrics (threshold={best['threshold']:.3f}):")
    strata = per_stratum_metrics(df_test, y_test, y_prob_test, best["threshold"])
    print_stratum_table(strata)

    # Feature importance
    importances = model.feature_importances_.tolist()
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature importance (ranked):")
    for feat, imp in ranked:
        bar = "█" * int(imp * 50)
        print(f"    {feat:<20s} {imp:.4f}  {bar}")

    # Save artifacts
    tag = args.tag
    model_path = out_dir / f"classifier_{tag}.joblib"
    metrics_path = out_dir / f"metrics_{tag}.json"
    plot_path = out_dir / f"feature_importance_{tag}.png"

    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "threshold": best["threshold"],
        "tag": tag,
    }, model_path)

    metrics_payload = {
        "tag": tag,
        "val": best,
        "test_overall": {
            "precision": test_p, "recall": test_r, "f1": test_f1,
            "aucpr": aucpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "threshold": best["threshold"],
        },
        "test_strata": strata,
        "feature_importance": dict(zip(feature_cols, importances)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "feature_cols": feature_cols,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    plot_feature_importance(feature_cols, importances, plot_path,
                            f"Phase 1 Feature Importance — {tag}")

    print(f"\n  Saved: {model_path.name}, {metrics_path.name}, {plot_path.name}")


if __name__ == "__main__":
    main()
