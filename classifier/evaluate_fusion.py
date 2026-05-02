"""
evaluate_fusion.py — Evaluate the trained fusion classifier and produce
comparison tables, PR curves, and feature importance plots.

Usage:
    python classifier/evaluate_fusion.py
    python classifier/evaluate_fusion.py --config classifier/config.yaml
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, average_precision_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_at_threshold(y_true, y_prob, threshold):
    preds = (y_prob >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def eval_rule_baselines(df, feature_cols):
    """Evaluate simple rule-based baselines for comparison."""
    y = df["label"].values
    results = {}

    # OR baseline: accept any candidate (all detections from union)
    results["OR-union"] = eval_at_threshold(y, np.ones(len(y)), 0.5)

    # AND baseline: accept only agreed detections
    agreement = df["agreement"].values
    results["AND-agreement"] = eval_at_threshold(y, agreement, 0.5)

    # conf_max > 0.5 (simple thresholding)
    results["conf_max>0.5"] = eval_at_threshold(y, df["conf_max"].values, 0.5)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pr_curve(y_true, y_prob, baselines, out_path):
    if not HAS_MPL:
        print("  matplotlib not available, skipping PR curve plot")
        return

    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)
    aucpr = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, "b-", linewidth=2,
            label=f"Fusion Classifier (AUC-PR={aucpr:.4f})")

    # Plot baselines as points
    markers = {"OR-union": "^", "AND-agreement": "s", "conf_max>0.5": "D"}
    colors = {"OR-union": "green", "AND-agreement": "red", "conf_max>0.5": "orange"}
    for name, m in baselines.items():
        ax.scatter(m["recall"], m["precision"], marker=markers.get(name, "o"),
                   c=colors.get(name, "gray"), s=100, zorder=5, label=name)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Fusion Classifier — Precision-Recall Curve", fontsize=14)
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  PR curve saved to {out_path}")


def plot_feature_importance(importances, out_path):
    if not HAS_MPL:
        return

    names = list(importances.keys())
    values = list(importances.values())
    order = np.argsort(values)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([names[i] for i in order], [values[i] for i in order])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (XGBoost)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Feature importance plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

def run_ablation(df, feature_cols, model_bundle, out_dir):
    """Remove one feature at a time and report impact."""
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    stems = df["stem"].unique()
    train_stems, test_stems = train_test_split(stems, test_size=0.2, random_state=42)
    train_mask = df["stem"].isin(train_stems)
    test_mask = df["stem"].isin(test_stems)

    y_train = df.loc[train_mask, "label"].values
    y_test = df.loc[test_mask, "label"].values

    threshold = model_bundle["threshold"]
    ablation = {}

    for drop in [None] + feature_cols:
        cols = [c for c in feature_cols if c != drop] if drop else feature_cols
        label = f"without_{drop}" if drop else "all_features"

        X_tr = df.loc[train_mask, cols].values
        X_te = df.loc[test_mask, cols].values

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        m = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            scale_pos_weight=float(n_neg / n_pos) if n_pos > 0 else 1.0,
            random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_train)
        y_prob = m.predict_proba(X_te)[:, 1]
        metrics = eval_at_threshold(y_test, y_prob, threshold)
        ablation[label] = {k: metrics[k] for k in ["precision", "recall", "f1"]}

    return ablation


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate fusion classifier")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--ablation", action="store_true",
                        help="Run feature ablation study")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    csv_path = Path(args.csv) if args.csv else out_dir / "fusion_dataset.csv"

    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Load trained model
    model_path = out_dir / "classifier.joblib"
    print(f"Loading model from {model_path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    threshold = bundle["threshold"]

    X = df[feature_cols].values
    y = df["label"].values
    y_prob = model.predict_proba(X)[:, 1]

    # Classifier results at frozen threshold
    clf_metrics = eval_at_threshold(y, y_prob, threshold)

    # Rule-based baselines
    baselines = eval_rule_baselines(df, feature_cols)

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Strategy':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 70)

    all_results = {**baselines, f"Classifier (t={threshold:.3f})": clf_metrics}
    for name, m in all_results.items():
        print(f"{name:<25} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6}")

    # Save results
    report = {
        "classifier": clf_metrics,
        "baselines": baselines,
        "threshold": threshold,
        "feature_cols": feature_cols,
        "aucpr": float(average_precision_score(y, y_prob)),
    }

    # Plots
    plot_pr_curve(y, y_prob, baselines, out_dir / "pr_curve.png")

    if bundle["model_type"] == "xgboost":
        importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
        report["feature_importance"] = importances
        plot_feature_importance(importances, out_dir / "feature_importance.png")

    # Ablation
    if args.ablation:
        print("\nRunning ablation study...")
        ablation = run_ablation(df, feature_cols, bundle, out_dir)
        report["ablation"] = ablation
        print("\nAblation results:")
        for name, m in ablation.items():
            print(f"  {name:<25} P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"F1={m['f1']:.4f}")

    eval_path = out_dir / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull evaluation saved to {eval_path}")


if __name__ == "__main__":
    main()
