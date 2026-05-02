"""
compare_classifiers.py — A/B comparison of v1.0 (original) vs v1.1 (retrained) fusion classifier.

Loads both models, runs on the same fusion dataset, reports:
  - Overall accuracy for each
  - Per-class agreement/disagreement
  - Feature analysis of disagreement frames (where v1.0 accepts but v1.1 rejects)

Usage:
    python classifier/reliability/fusion/compare_classifiers.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "fusion"

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}


def load_model(path):
    """Load a fusion model bundle and return (model, features, label_map)."""
    bundle = joblib.load(path)
    model = bundle["model"]
    features = bundle["features"]
    label_map = bundle.get("label_map", LABEL_NAMES)
    return model, features, label_map


def run_predictions(model, features, df):
    """Run model on df, return predicted labels and probabilities."""
    # Fill missing features with 0
    X = df.copy()
    for f in features:
        if f not in X.columns:
            X[f] = 0.0
    X = X[features].values.astype(float)
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    return preds, probs


def main():
    # Load models
    v10_path = DATA_DIR / "fusion_no_fn_model_original.joblib"
    v11_path = DATA_DIR / "fusion_no_fn_model_v1.1.joblib"

    if not v10_path.exists():
        print(f"[ERROR] Original model not found: {v10_path}")
        print("  Run: git show HEAD:classifier/runs/reliability/fusion/fusion_no_fn_model.joblib > <path>")
        return
    if not v11_path.exists():
        print(f"[ERROR] v1.1 model not found: {v11_path}")
        return

    print("Loading models...")
    m10, feat10, lm10 = load_model(v10_path)
    m11, feat11, lm11 = load_model(v11_path)
    print(f"  v1.0: {len(feat10)} features, {m10.n_estimators} trees")
    print(f"  v1.1: {len(feat11)} features, {m11.n_estimators} trees")

    # Show feature differences
    f10_set = set(feat10)
    f11_set = set(feat11)
    if f10_set != f11_set:
        print(f"\n  Features only in v1.0: {f10_set - f11_set}")
        print(f"  Features only in v1.1: {f11_set - f10_set}")
    else:
        print(f"  Same {len(feat10)} features in both models")

    # Load dataset
    csv_path = DATA_DIR / "fusion_dataset.csv"
    print(f"\nLoading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows")
    print(f"  Sources: {df['source_dataset'].value_counts().to_dict()}")

    # Run both models
    print("\nRunning predictions...")
    p10, pr10 = run_predictions(m10, feat10, df)
    p11, pr11 = run_predictions(m11, feat11, df)
    y_true = df["trust_label"].values

    # Overall accuracy
    acc10 = accuracy_score(y_true, p10)
    acc11 = accuracy_score(y_true, p11)
    print(f"\n{'=' * 60}")
    print(f"OVERALL ACCURACY")
    print(f"{'=' * 60}")
    print(f"  v1.0: {acc10:.4f} ({(p10 == y_true).sum():,}/{len(y_true):,})")
    print(f"  v1.1: {acc11:.4f} ({(p11 == y_true).sum():,}/{len(y_true):,})")

    # Per-source accuracy
    print(f"\n{'=' * 60}")
    print(f"PER-SOURCE ACCURACY")
    print(f"{'=' * 60}")
    for src in sorted(df["source_dataset"].unique()):
        mask = df["source_dataset"] == src
        n = mask.sum()
        a10 = accuracy_score(y_true[mask], p10[mask])
        a11 = accuracy_score(y_true[mask], p11[mask])
        delta = a11 - a10
        marker = "UP" if delta > 0.01 else ("DN" if delta < -0.01 else "==")
        print(f"  {src:<25s}  n={n:>6,}  v1.0={a10:.4f}  v1.1={a11:.4f}  {marker} {delta:+.4f}")

    # Disagreement analysis
    disagree = p10 != p11
    n_disagree = disagree.sum()
    print(f"\n{'=' * 60}")
    print(f"DISAGREEMENTS: {n_disagree:,} frames ({n_disagree/len(df)*100:.2f}%)")
    print(f"{'=' * 60}")

    if n_disagree > 0:
        # Who is right when they disagree?
        v10_right = ((p10 == y_true) & disagree).sum()
        v11_right = ((p11 == y_true) & disagree).sum()
        both_wrong = (((p10 != y_true) & (p11 != y_true)) & disagree).sum()
        print(f"  v1.0 correct: {v10_right:,}")
        print(f"  v1.1 correct: {v11_right:,}")
        print(f"  Both wrong:   {both_wrong:,}")

        # Key pattern: v1.0 says trust_both (3) but v1.1 says reject_both (0)
        critical = (p10 == 3) & (p11 == 0)
        n_critical = critical.sum()
        print(f"\n  CRITICAL: v1.0=trust_both -> v1.1=reject_both: {n_critical:,} frames")
        if n_critical > 0:
            crit_df = df[critical]
            print(f"  Sources: {crit_df['source_dataset'].value_counts().to_dict()}")
            # Ground truth for these frames
            gt_counts = crit_df["trust_label"].value_counts().to_dict()
            gt_named = {LABEL_NAMES.get(k, k): v for k, v in gt_counts.items()}
            print(f"  Ground truth: {gt_named}")

            # Feature comparison for critical frames
            all_feats = sorted(set(feat10) | set(feat11))
            print(f"\n  Feature means for CRITICAL frames vs ALL:")
            print(f"  {'feature':<35s}  {'critical':>10s}  {'all_data':>10s}  {'delta':>10s}")
            for feat in all_feats:
                if feat in df.columns:
                    crit_mean = crit_df[feat].mean()
                    all_mean = df[feat].mean()
                    delta = crit_mean - all_mean
                    if abs(delta) > 0.01:  # Only show meaningful differences
                        print(f"  {feat:<35s}  {crit_mean:>10.4f}  {all_mean:>10.4f}  {delta:>+10.4f}")

        # Reverse: v1.0 says reject but v1.1 says trust
        reverse = (p10 == 0) & (p11 == 3)
        n_reverse = reverse.sum()
        print(f"\n  REVERSE: v1.0=reject_both -> v1.1=trust_both: {n_reverse:,} frames")
        if n_reverse > 0:
            rev_df = df[reverse]
            print(f"  Sources: {rev_df['source_dataset'].value_counts().to_dict()}")
            gt_counts = rev_df["trust_label"].value_counts().to_dict()
            gt_named = {LABEL_NAMES.get(k, k): v for k, v in gt_counts.items()}
            print(f"  Ground truth: {gt_named}")

    # Per-source confusion: v1.1's predictions where v1.0 was correct
    print(f"\n{'=' * 60}")
    print(f"v1.1 REGRESSIONS BY SOURCE (frames where v1.0 correct, v1.1 wrong)")
    print(f"{'=' * 60}")
    regressed = (p10 == y_true) & (p11 != y_true)
    for src in sorted(df["source_dataset"].unique()):
        mask = (df["source_dataset"] == src) & regressed
        if mask.sum() > 0:
            src_df = df[mask]
            print(f"\n  {src}: {mask.sum():,} regressions")
            # What v1.1 predicted instead
            v11_preds = pd.Series(p11[mask]).map(LABEL_NAMES).value_counts().to_dict()
            gt_labs = pd.Series(y_true[mask]).map(LABEL_NAMES).value_counts().to_dict()
            print(f"    Ground truth: {gt_labs}")
            print(f"    v1.1 predicted: {v11_preds}")

    # Save detailed results
    out = DATA_DIR / "classifier_comparison.json"
    report = {
        "v10_accuracy": acc10,
        "v11_accuracy": acc11,
        "n_total": len(df),
        "n_disagree": int(n_disagree),
        "n_critical_trust_to_reject": int(n_critical) if n_disagree > 0 else 0,
        "per_source_v10": {},
        "per_source_v11": {},
    }
    for src in sorted(df["source_dataset"].unique()):
        mask = df["source_dataset"] == src
        report["per_source_v10"][src] = float(accuracy_score(y_true[mask], p10[mask]))
        report["per_source_v11"][src] = float(accuracy_score(y_true[mask], p11[mask]))

    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
