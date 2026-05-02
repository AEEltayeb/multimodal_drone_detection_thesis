r"""
failure_profile.py — Fit logistic regression on TP/FN features from error CSV.

Reads the per-object feature CSV produced by per_source_eval.py --feature-csv,
fits a logistic regression predicting FN vs TP, and outputs:
  1. Feature coefficients (thesis-ready table)
  2. Cross-validated AUC-ROC (ranking quality metric)
  3. Saved model pickle for use in select_samples.py

Usage:
    python scripts/analysis/failure_profile.py \
        --error-csv runs/IR_FT_dsetV6_aug1_s0/error_features_dsetV6_test.csv \
        --output-dir runs/IR_FT_dsetV6_aug1_s0/failure_profile/
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np


# ── Feature definitions ──────────────────────────────────────────
#
# RAW_FEATURES: columns read from the CSV.
# Transformed features are computed in build_feature_matrix().
#   - log_bbox_area = log(bbox_area + 1)      (linearises the scale)
#   - abs_contrast  = abs(local_contrast)      (sign depends on hot/cold)
#   - area_fraction is DROPPED (multicollinear with bbox_area)

RAW_FEATURES = [
    "bbox_area",        # → transformed to log_bbox_area
    "aspect_ratio",
    "pos_x",
    "pos_y",
    "dist_to_center",
    "local_contrast",   # → transformed to abs_contrast
    "img_mean",
    "img_std",
    "img_dynamic_range",
]

MODEL_FEATURES = [
    "log_bbox_area",
    "aspect_ratio",
    "pos_x",
    "pos_y",
    "dist_to_center",
    "abs_contrast",
    "img_mean",
    "img_std",
    "img_dynamic_range",
]


def build_feature_matrix(rows):
    """Build X matrix with engineered features from raw CSV rows."""
    X = np.zeros((len(rows), len(MODEL_FEATURES)))

    for i, row in enumerate(rows):
        bbox_area = float(row.get("bbox_area", 1))
        contrast = float(row.get("local_contrast", 0))

        X[i, 0] = np.log(bbox_area + 1)                  # log_bbox_area
        X[i, 1] = float(row.get("aspect_ratio", 1))
        X[i, 2] = float(row.get("pos_x", 0.5))
        X[i, 3] = float(row.get("pos_y", 0.5))
        X[i, 4] = float(row.get("dist_to_center", 0))
        X[i, 5] = abs(contrast)                           # abs_contrast
        X[i, 6] = float(row.get("img_mean", 0))
        X[i, 7] = float(row.get("img_std", 0))
        X[i, 8] = float(row.get("img_dynamic_range", 0))

    return X


def load_tp_fn_data(csv_path: str):
    """Load feature CSV, filter to TP/FN rows, return X matrix and y labels."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["outcome"] in ("TP", "FN"):
                rows.append(row)

    if not rows:
        print("[ERROR] No TP/FN rows found in CSV.")
        sys.exit(1)

    n_tp = sum(1 for r in rows if r["outcome"] == "TP")
    n_fn = sum(1 for r in rows if r["outcome"] == "FN")
    print(f"  Loaded {len(rows)} objects: {n_tp} TP, {n_fn} FN")
    print(f"  FN rate: {n_fn / len(rows) * 100:.1f}%")

    X = build_feature_matrix(rows)
    y = np.array([1 if r["outcome"] == "FN" else 0 for r in rows])

    return X, y, rows


def main():
    parser = argparse.ArgumentParser(
        description="Fit failure profile model on TP/FN features"
    )
    parser.add_argument("--error-csv", required=True,
                        help="Path to feature CSV from per_source_eval.py")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output artifacts")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FAILURE PROFILE — Logistic Regression (v2)")
    print(f"  Input:  {args.error_csv}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # ── Load data ──
    X, y, raw_rows = load_tp_fn_data(args.error_csv)

    # ── Import sklearn (late import — not needed for other scripts) ──
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("[ERROR] scikit-learn is required: pip install scikit-learn")
        sys.exit(1)

    # ── Scale features ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Fit logistic regression ──
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        penalty="l2",
    )
    model.fit(X_scaled, y)

    # ── Cross-validation with AUC-ROC (proper ranking metric) ──
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_auc = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    cv_acc = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")

    auc_mean, auc_std = cv_auc.mean(), cv_auc.std()
    acc_mean, acc_std = cv_acc.mean(), cv_acc.std()

    # Full-data AUC for reference
    y_prob = model.predict_proba(X_scaled)[:, 1]
    full_auc = roc_auc_score(y, y_prob)

    print(f"  5-fold CV AUC-ROC:  {auc_mean:.4f} ± {auc_std:.4f}  ← primary metric")
    print(f"  5-fold CV accuracy: {acc_mean:.4f} ± {acc_std:.4f}  (for reference)")
    print(f"  Full-data AUC-ROC:  {full_auc:.4f}")

    if auc_mean >= 0.70:
        print(f"  [GOOD] AUC ≥ 0.70 — model rankings are useful for sample selection")
    elif auc_mean >= 0.60:
        print(f"  [OK] AUC 0.60–0.70 — model provides some signal, consider combining with rules")
    else:
        print(f"  [WEAK] AUC < 0.60 — model rankings are weak, prefer rule-based selection")

    # ── Feature coefficients ──
    print(f"\n  {'Feature':<22s}  {'Coefficient':>12s}  {'Direction'}")
    print(f"  {'-'*60}")

    coef_data = []
    for feat, coef in zip(MODEL_FEATURES, model.coef_[0]):
        direction = "↑ = more FN" if coef > 0 else "↓ = more FN"
        print(f"  {feat:<22s}  {coef:>+12.4f}  {direction}")
        coef_data.append({
            "feature": feat,
            "coefficient": round(float(coef), 6),
        })

    print(f"\n  Intercept: {model.intercept_[0]:+.4f}")

    # ── Rank features by absolute importance ──
    ranked = sorted(coef_data, key=lambda d: abs(d["coefficient"]), reverse=True)
    print(f"\n  Feature importance ranking (|coefficient|):")
    for i, d in enumerate(ranked, 1):
        print(f"    {i}. {d['feature']:<22s}  |coef| = {abs(d['coefficient']):.4f}")

    # ── P(FN) distribution sanity check ──
    fn_probs = y_prob[y == 1]
    tp_probs = y_prob[y == 0]
    print(f"\n  P(FN) distribution:")
    print(f"    Actual FN objects: mean={fn_probs.mean():.3f}, median={np.median(fn_probs):.3f}")
    print(f"    Actual TP objects: mean={tp_probs.mean():.3f}, median={np.median(tp_probs):.3f}")
    print(f"    Separation: {fn_probs.mean() - tp_probs.mean():+.3f} "
          f"({'good' if fn_probs.mean() > tp_probs.mean() + 0.05 else 'weak'})")

    # ── Save outputs ──

    # 1. Feature importance CSV
    importance_path = out_dir / "feature_importance.csv"
    with open(importance_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "coefficient"])
        writer.writeheader()
        writer.writerows(ranked)
    print(f"\n  Saved: {importance_path}")

    # 2. Report JSON
    report = {
        "cv_auc_mean": round(auc_mean, 4),
        "cv_auc_std": round(auc_std, 4),
        "cv_accuracy_mean": round(acc_mean, 4),
        "cv_accuracy_std": round(acc_std, 4),
        "full_auc": round(full_auc, 4),
        "n_tp": int(sum(y == 0)),
        "n_fn": int(sum(y == 1)),
        "n_total": len(y),
        "fn_rate": round(sum(y == 1) / len(y) * 100, 2),
        "intercept": round(float(model.intercept_[0]), 6),
        "coefficients": {d["feature"]: d["coefficient"] for d in ranked},
        "feature_ranking": [d["feature"] for d in ranked],
        "p_fn_separation": {
            "fn_objects_mean": round(float(fn_probs.mean()), 4),
            "tp_objects_mean": round(float(tp_probs.mean()), 4),
        },
        "features_used": MODEL_FEATURES,
        "transforms": {
            "log_bbox_area": "log(bbox_area + 1)",
            "abs_contrast": "abs(local_contrast)",
            "dropped": ["area_fraction (multicollinear with bbox_area)"],
        },
    }
    report_path = out_dir / "failure_profile_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {report_path}")

    # 3. Model pickle (scaler + model + transforms info)
    model_path = out_dir / "failure_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "model": model,
            "features": MODEL_FEATURES,
            "raw_features": RAW_FEATURES,
            "transforms": {
                "log_bbox_area": "log(bbox_area + 1)",
                "abs_contrast": "abs(local_contrast)",
            },
        }, f)
    print(f"  Saved: {model_path}")

    print(f"\n  Done. Use --failure-model {model_path} in select_samples.py")


if __name__ == "__main__":
    main()
