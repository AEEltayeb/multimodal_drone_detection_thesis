"""
train_ir_suppressed.py — Option 2 experiment: synthetic IR-failure augmentation.

The Phase 1 finding: the classifier learned to depend almost entirely on
max_conf_ir because the IR model is effectively an oracle on both training
datasets. This script tests whether:

  (a) The baseline classifier collapses when IR is synthetically suppressed
      (which would confirm it never learned real fusion), and

  (b) A classifier trained with IR-suppression augmentation CAN learn to
      fall back on RGB when IR fails, while maintaining performance on
      normal frames.

Procedure:
  1. Load curated frame-level dataset + reuse Phase 1 sequence split.
  2. Evaluate the Phase 1 *baseline* classifier on three test variants:
        - normal (no suppression)
        - 30% suppressed (matches training rate)
        - 100% suppressed (worst case)
  3. Train a new classifier on train data where ~30% of rows have IR
     suppressed. Also keep a copy of the original rows (augmentation, not
     replacement) so the classifier sees both regimes.
  4. Evaluate the new classifier on the same three test variants.
  5. Compare F1 and feature importance.

Usage:
    python train_ir_suppressed.py
    python train_ir_suppressed.py --suppression-rate 0.3
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
PHASE1_DIR = CLASSIFIER_DIR / "phase1_curation"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PHASE1_DIR))

from ir_suppression import suppress_ir_features, random_suppression_mask
from split_sequences import split_by_sequence, print_split_summary
from curate_dataset import FEATURE_COLS as HONEST_FEATURE_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
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


def evaluate(model, feature_cols, df_test, threshold, tag):
    """Compute overall + per-stratum metrics on df_test."""
    X = df_test[feature_cols].values.astype(np.float32)
    y = df_test["label"].values.astype(np.int32)
    y_prob = model.predict_proba(X)[:, 1]
    preds = (y_prob >= threshold).astype(int)

    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    aucpr = float(average_precision_score(y, y_prob)) if len(set(y)) > 1 else 0.0

    # Per-stratum (source × category)
    strata = []
    df_r = df_test.reset_index(drop=True)
    preds_r = preds
    y_r = y
    for (src, cat), sub in df_r.groupby(["source", "category"]):
        idx = sub.index.values
        sy = y_r[idx]
        sp = preds_r[idx]
        stp = int(((sp == 1) & (sy == 1)).sum())
        sfp = int(((sp == 1) & (sy == 0)).sum())
        sfn = int(((sp == 0) & (sy == 1)).sum())
        stn = int(((sp == 0) & (sy == 0)).sum())
        spr = stp / (stp + sfp) if (stp + sfp) > 0 else 0.0
        sre = stp / (stp + sfn) if (stp + sfn) > 0 else 0.0
        sf1 = 2 * spr * sre / (spr + sre) if (spr + sre) > 0 else 0.0
        sfpr = sfp / (sfp + stn) if (sfp + stn) > 0 else 0.0
        strata.append({
            "group": f"{src} / {cat}", "rows": len(sy),
            "pos": int(sy.sum()), "neg": int(len(sy) - sy.sum()),
            "tp": stp, "fp": sfp, "fn": sfn, "tn": stn,
            "precision": spr, "recall": sre, "f1": sf1, "fp_rate": sfpr,
        })

    return {
        "tag": tag,
        "overall": {"precision": p, "recall": r, "f1": f1, "aucpr": aucpr,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": threshold},
        "strata": strata,
    }


def print_eval(result):
    o = result["overall"]
    print(f"\n  [{result['tag']}]")
    print(f"    Overall: P={o['precision']:.4f} R={o['recall']:.4f} "
          f"F1={o['f1']:.4f} AUC-PR={o['aucpr']:.4f}  "
          f"(TP={o['tp']} FP={o['fp']} FN={o['fn']} TN={o['tn']})")
    print(f"    {'group':<30s} {'rows':>6s} {'P':>7s} {'R':>7s} {'F1':>7s} {'FP/tot':>8s}")
    for s in result["strata"]:
        print(f"    {s['group']:<30s} {s['rows']:>6d} "
              f"{s['precision']:>7.4f} {s['recall']:>7.4f} "
              f"{s['f1']:>7.4f} {s['fp_rate']:>8.4f}")


def train_xgb(X_train, y_train, X_val, y_val, cfg):
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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def feature_importance_table(feature_cols, importances, label):
    ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print(f"\n  Feature importance ({label}):")
    for feat, imp in ranked:
        bar = "#" * int(imp * 50)
        print(f"    {feat:<20s} {imp:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CLASSIFIER_DIR / "config.yaml"))
    parser.add_argument("--csv", default="runs/curated_frame_dataset.csv")
    parser.add_argument("--suppression-rate", type=float, default=0.3,
                        help="Fraction of training rows to augment with IR suppression")
    parser.add_argument("--baseline-model",
                        default="runs/phase1/classifier_baseline.joblib")
    parser.add_argument("--out-dir", default="runs/phase2")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = CLASSIFIER_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = CLASSIFIER_DIR / csv_path
    baseline_path = Path(args.baseline_model)
    if not baseline_path.is_absolute():
        baseline_path = CLASSIFIER_DIR / baseline_path

    print("=" * 70)
    print("Option 2: Synthetic IR-failure augmentation")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Same sequence split as Phase 1 (same random_state)
    masks = split_by_sequence(df, 0.70, 0.15, 0.15, random_state=42,
                              stratify_col="source")
    df_train = df[masks["train"]].reset_index(drop=True)
    df_val = df[masks["val"]].reset_index(drop=True)
    df_test = df[masks["test"]].reset_index(drop=True)
    print(f"  Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    feature_cols = HONEST_FEATURE_COLS

    # -----------------------------------------------------------------------
    # Build three test variants
    # -----------------------------------------------------------------------
    df_test_normal = df_test.copy()
    mask_all = np.ones(len(df_test), dtype=bool)
    df_test_all_sup = suppress_ir_features(df_test, mask_all)
    mask_30 = random_suppression_mask(len(df_test), args.suppression_rate,
                                       random_state=123)
    df_test_30_sup = suppress_ir_features(df_test, mask_30)
    print(f"  Test variants: normal, 30%-suppressed ({int(mask_30.sum())} rows), "
          f"100%-suppressed ({len(df_test)} rows)")

    # -----------------------------------------------------------------------
    # PART 1: Evaluate the Phase 1 baseline on all three test variants
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PART 1: Baseline classifier (Phase 1) on suppressed test variants")
    print("=" * 70)

    if not baseline_path.exists():
        print(f"  ERROR: baseline model not found at {baseline_path}")
        print(f"  Run Phase 1 first: python phase1_curation/run_phase1.py")
        return

    baseline_bundle = joblib.load(baseline_path)
    baseline_model = baseline_bundle["model"]
    baseline_feats = baseline_bundle["feature_cols"]
    baseline_threshold = baseline_bundle["threshold"]
    print(f"  Loaded baseline: threshold={baseline_threshold:.3f}, "
          f"features={len(baseline_feats)}")

    # Feature column order must match
    if baseline_feats != feature_cols:
        print(f"  WARNING: baseline feature order differs, reordering.")
    feats_use = baseline_feats

    r_baseline_normal = evaluate(baseline_model, feats_use, df_test_normal,
                                 baseline_threshold, "baseline / normal")
    r_baseline_30 = evaluate(baseline_model, feats_use, df_test_30_sup,
                             baseline_threshold, "baseline / 30%-sup")
    r_baseline_all = evaluate(baseline_model, feats_use, df_test_all_sup,
                              baseline_threshold, "baseline / 100%-sup")

    print_eval(r_baseline_normal)
    print_eval(r_baseline_30)
    print_eval(r_baseline_all)

    # -----------------------------------------------------------------------
    # PART 2: Train a new classifier with IR-suppression augmentation
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"PART 2: Train with IR-suppression augmentation "
          f"(rate={args.suppression_rate})")
    print("=" * 70)

    # Augmentation: keep original train rows + add IR-suppressed copies of a
    # random subset. This way the classifier sees BOTH the "IR working"
    # regime and the "IR silent" regime.
    sup_mask_train = random_suppression_mask(len(df_train),
                                             args.suppression_rate,
                                             random_state=42)
    df_train_sup = suppress_ir_features(df_train, sup_mask_train)
    df_train_aug = pd.concat(
        [df_train, df_train_sup[sup_mask_train]],
        ignore_index=True,
    )
    print(f"  Augmented training set: {len(df_train)} original "
          f"+ {int(sup_mask_train.sum())} suppressed copies "
          f"= {len(df_train_aug)} total")

    # Also augment val set the same way, but ONLY the suppressed portion
    # (so threshold sweep reflects conditional-trust performance)
    sup_mask_val = random_suppression_mask(len(df_val),
                                           args.suppression_rate,
                                           random_state=7)
    df_val_sup = suppress_ir_features(df_val, sup_mask_val)
    df_val_aug = pd.concat(
        [df_val, df_val_sup[sup_mask_val]],
        ignore_index=True,
    )

    X_train = df_train_aug[feature_cols].values.astype(np.float32)
    y_train = df_train_aug["label"].values.astype(np.int32)
    X_val = df_val_aug[feature_cols].values.astype(np.float32)
    y_val = df_val_aug["label"].values.astype(np.int32)

    print(f"  Training XGBoost on {len(X_train)} rows "
          f"({int(y_train.sum())} pos, {int(len(y_train) - y_train.sum())} neg)...")
    new_model = train_xgb(X_train, y_train, X_val, y_val, cfg)

    y_prob_val = new_model.predict_proba(X_val)[:, 1]
    best = sweep_threshold(y_val, y_prob_val)
    new_threshold = best["threshold"]
    print(f"  Val best: P={best['precision']:.4f} R={best['recall']:.4f} "
          f"F1={best['f1']:.4f} @ t={new_threshold:.3f}")

    # Evaluate new model on all three test variants
    print()
    print("=" * 70)
    print("PART 3: Suppression-trained classifier on test variants")
    print("=" * 70)

    r_new_normal = evaluate(new_model, feature_cols, df_test_normal,
                            new_threshold, "sup-trained / normal")
    r_new_30 = evaluate(new_model, feature_cols, df_test_30_sup,
                        new_threshold, "sup-trained / 30%-sup")
    r_new_all = evaluate(new_model, feature_cols, df_test_all_sup,
                         new_threshold, "sup-trained / 100%-sup")
    print_eval(r_new_normal)
    print_eval(r_new_30)
    print_eval(r_new_all)

    # -----------------------------------------------------------------------
    # PART 4: Feature importance comparison
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("PART 4: Feature importance comparison")
    print("=" * 70)

    baseline_imp = dict(zip(feats_use, baseline_model.feature_importances_.tolist()))
    new_imp = dict(zip(feature_cols, new_model.feature_importances_.tolist()))

    all_feats = sorted(set(baseline_imp) | set(new_imp),
                       key=lambda f: -baseline_imp.get(f, 0.0))
    print(f"\n  {'feature':<20s} {'baseline':>12s} {'sup-trained':>14s} {'delta':>10s}")
    print("  " + "-" * 60)
    for f in all_feats:
        b = baseline_imp.get(f, 0.0)
        n = new_imp.get(f, 0.0)
        print(f"  {f:<20s} {b:>12.4f} {n:>14.4f} {n - b:>+10.4f}")

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    model_path = out_dir / "classifier_ir_suppressed.joblib"
    joblib.dump({
        "model": new_model,
        "feature_cols": feature_cols,
        "threshold": new_threshold,
        "suppression_rate": args.suppression_rate,
    }, model_path)

    summary = {
        "suppression_rate": args.suppression_rate,
        "baseline": {
            "normal": r_baseline_normal,
            "sup_30": r_baseline_30,
            "sup_100": r_baseline_all,
            "threshold": baseline_threshold,
            "feature_importance": baseline_imp,
        },
        "sup_trained": {
            "normal": r_new_normal,
            "sup_30": r_new_30,
            "sup_100": r_new_all,
            "threshold": new_threshold,
            "feature_importance": new_imp,
        },
    }
    metrics_path = out_dir / "metrics_ir_suppression.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {model_path.name}, {metrics_path.name}")

    # -----------------------------------------------------------------------
    # Final summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY: F1 across configurations")
    print("=" * 70)
    print(f"  {'model':<20s} {'normal':>10s} {'30%-sup':>10s} {'100%-sup':>10s}")
    print("  " + "-" * 52)
    print(f"  {'baseline':<20s} "
          f"{r_baseline_normal['overall']['f1']:>10.4f} "
          f"{r_baseline_30['overall']['f1']:>10.4f} "
          f"{r_baseline_all['overall']['f1']:>10.4f}")
    print(f"  {'sup-trained':<20s} "
          f"{r_new_normal['overall']['f1']:>10.4f} "
          f"{r_new_30['overall']['f1']:>10.4f} "
          f"{r_new_all['overall']['f1']:>10.4f}")


if __name__ == "__main__":
    main()
