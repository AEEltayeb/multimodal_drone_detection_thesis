"""
train_fusion.py — Train frame-level fusion classifier.

4-class trust decision:
  0 = REJECT_BOTH  (neither modality correct, or no drone)
  1 = TRUST_RGB    (only RGB correct)
  2 = TRUST_IR     (only IR correct)
  3 = TRUST_BOTH   (both correct)

Also trains binary "drone_present" classifier for comparison.

Ablation study (--ablation):
  baseline:    detector signals only (confs, n_dets, agreement)
  baseline+fn: adds FN model scores
  full:        all 46 features (scene + target + FN + detector)

Cheap sanity baselines reported automatically:
  - always_ir: trust IR whenever IR detects, else trust RGB
  - higher_conf: trust whichever modality has higher max conf
  - both_or_ir: trust both if both detect, else trust IR

Usage:
    python train_fusion.py
    python train_fusion.py --ablation
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
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "fusion"
OUT_DIR    = DATA_DIR


# ── FEATURE SETS ──────────────────────────────────────────────────

BASELINE_FEATURES = [
    # Detector signals only (18)
    "rgb_n_dets", "rgb_max_conf", "rgb_mean_conf", "rgb_detected",
    "ir_n_dets", "ir_max_conf", "ir_mean_conf", "ir_detected",
    "both_detect", "neither_detect", "rgb_only_detect", "ir_only_detect",
    # Best-det target features (size/shape — modality-safe)
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
    "ir_best_log_bbox_area", "ir_best_aspect_ratio",
    "rgb_best_dist_to_center", "ir_best_dist_to_center",
]

BASELINE_PLUS_FN = BASELINE_FEATURES + [
    "rgb_max_fn", "rgb_mean_fn", "rgb_min_fn",
    "ir_max_fn", "ir_mean_fn", "ir_min_fn",
]

FULL_FEATURES = [
    # RGB detection aggregates (7)
    "rgb_n_dets", "rgb_max_conf", "rgb_mean_conf",
    "rgb_max_fn", "rgb_mean_fn", "rgb_min_fn", "rgb_detected",
    # IR detection aggregates (7)
    "ir_n_dets", "ir_max_conf", "ir_mean_conf",
    "ir_max_fn", "ir_mean_fn", "ir_min_fn", "ir_detected",
    # RGB scene (7)
    "rgb_img_mean", "rgb_img_std", "rgb_img_dynamic_range",
    "rgb_img_entropy", "rgb_sky_ground_ratio", "rgb_edge_density",
    "rgb_blurriness",
    # IR scene (7)
    "ir_img_mean", "ir_img_std", "ir_img_dynamic_range",
    "ir_img_entropy", "ir_sky_ground_ratio", "ir_edge_density",
    "ir_blurriness",
    # RGB best-det target (7)
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
    "rgb_best_pos_x", "rgb_best_pos_y", "rgb_best_dist_to_center",
    "rgb_best_local_contrast", "rgb_best_target_bg_delta",
    # IR best-det target (7)
    "ir_best_log_bbox_area", "ir_best_aspect_ratio",
    "ir_best_pos_x", "ir_best_pos_y", "ir_best_dist_to_center",
    "ir_best_local_contrast", "ir_best_target_bg_delta",
    # Frame-level agreement (4)
    "both_detect", "neither_detect", "rgb_only_detect", "ir_only_detect",
]

ABLATION_SETS = {
    "baseline": BASELINE_FEATURES,
    "baseline+fn": BASELINE_PLUS_FN,
    "full": FULL_FEATURES,
}

NO_FN_FEATURES = [f for f in FULL_FEATURES
                   if f not in ("rgb_max_fn", "rgb_mean_fn", "rgb_min_fn",
                                "ir_max_fn", "ir_mean_fn", "ir_min_fn")]


# ── SEQUENCE SPLIT ────────────────────────────────────────────────

SEQ_SUFFIX_RE = re.compile(
    r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared|_ir|_rgb)?$",
    re.IGNORECASE,
)


def extract_sequence_id(base_stem, source_dataset):
    m = SEQ_SUFFIX_RE.match(base_stem)
    if m:
        base = m.group(1).rstrip("_")
        if base:
            return f"{source_dataset}::{base}"
    return f"{source_dataset}::{base_stem}"


def sequence_split(df, test_size=0.25, random_state=42,
                   force_train_patterns=None):
    groups = df["sequence_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, df["trust_label"], groups=groups))

    if force_train_patterns:
        test_seqs = set(df.iloc[test_idx]["sequence_id"].values)
        to_move = set()
        for pat in force_train_patterns:
            rgx = re.compile(pat, re.IGNORECASE)
            for s in test_seqs:
                if rgx.search(s):
                    to_move.add(s)
        if to_move:
            print(f"  Forcing {len(to_move)} sequence(s) into train:")
            for s in sorted(to_move):
                print(f"    {s}")
            test_mask_move = np.isin(groups[test_idx], list(to_move))
            train_idx = np.concatenate([train_idx, test_idx[test_mask_move]])
            test_idx = test_idx[~test_mask_move]

    train_seqs = set(df.iloc[train_idx]["sequence_id"])
    test_seqs = set(df.iloc[test_idx]["sequence_id"])
    assert len(train_seqs & test_seqs) == 0, "Sequence leakage!"
    return train_idx, test_idx


# ── CHEAP SANITY BASELINES ────────────────────────────────────────

def cheap_baselines(df):
    """Evaluate trivial trust policies. Returns dict of results."""
    y_true = df["trust_label"].values
    results = {}

    # Policy 1: always trust IR (if IR detects -> trust_ir, else trust_rgb)
    preds = np.where(df["ir_detected"].values == 1, 2, 1)
    # If neither detects, reject
    preds = np.where(
        (df["rgb_detected"].values == 0) & (df["ir_detected"].values == 0),
        0, preds
    )
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

    # Policy 3: trust both if both detect, else trust IR, else trust RGB
    preds = np.full(len(df), 0)
    preds = np.where(
        (df["rgb_detected"].values == 1) & (df["ir_detected"].values == 1),
        3, preds
    )
    preds = np.where(
        (df["rgb_detected"].values == 0) & (df["ir_detected"].values == 1),
        2, preds
    )
    preds = np.where(
        (df["rgb_detected"].values == 1) & (df["ir_detected"].values == 0),
        1, preds
    )
    results["both_or_ir"] = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
    }

    return results


# ── TRAINING ──────────────────────────────────────────────────────

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}


def train_and_eval(df, feature_cols, tag="fusion", force_train_patterns=None):
    print(f"\n{'=' * 60}")
    print(f"Training {tag.upper()} ({len(feature_cols)} features, 4-class)")
    print(f"{'=' * 60}")

    df = df.copy()
    df["sequence_id"] = df.apply(
        lambda r: extract_sequence_id(r["base_stem"], r["source_dataset"]), axis=1
    )
    n_seqs = df["sequence_id"].nunique()
    print(f"  Rows: {len(df):,}  Sequences: {n_seqs:,}")

    for val, name in LABEL_NAMES.items():
        n = (df["trust_label"] == val).sum()
        print(f"  Class {val} ({name}): {n:,} ({n / len(df) * 100:.1f}%)")

    train_idx, test_idx = sequence_split(
        df, force_train_patterns=force_train_patterns)
    X_train = df.iloc[train_idx][feature_cols].values
    X_test = df.iloc[test_idx][feature_cols].values
    y_train = df.iloc[train_idx]["trust_label"].values
    y_test = df.iloc[test_idx]["trust_label"].values
    src_test = df.iloc[test_idx]["source_dataset"].values

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Per-class AUC (one-vs-rest)
    per_class_auc = {}
    for c in range(4):
        if (y_test == c).sum() > 0 and (y_test != c).sum() > 0:
            binary = (y_test == c).astype(int)
            per_class_auc[LABEL_NAMES[c]] = float(roc_auc_score(binary, y_prob[:, c]))
        else:
            per_class_auc[LABEL_NAMES[c]] = float("nan")

    print(f"\n  Overall:")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    F1 (macro):  {f1_macro:.4f}")
    print(f"    F1 (weighted): {f1_weighted:.4f}")
    print(f"    Per-class AUC: {per_class_auc}")

    # Classification report
    target_names = [LABEL_NAMES[i] for i in range(4)]
    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )
    print(f"\n{report}")

    # Per-dataset breakdown
    per_dataset = {}
    print(f"  Per-dataset:")
    print(f"    {'dataset':<15s} {'n':>7s} {'acc':>7s} {'f1_m':>7s} {'f1_w':>7s}")
    for ds in np.unique(src_test):
        mask = src_test == ds
        ys, yp = y_test[mask], y_pred[mask]
        a = float(accuracy_score(ys, yp))
        fm = float(f1_score(ys, yp, average="macro", zero_division=0))
        fw = float(f1_score(ys, yp, average="weighted", zero_division=0))
        per_dataset[str(ds)] = {"n": int(len(ys)), "acc": a, "f1_macro": fm, "f1_weighted": fw}
        print(f"    {ds:<15s} {len(ys):>7d} {a:>7.4f} {fm:>7.4f} {fw:>7.4f}")

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print(f"\n  Top 10 features:")
    for i, (feat, val) in enumerate(importance.items()):
        if i >= 10:
            break
        print(f"    {feat:<35s} {val:.4f}")

    metrics = {
        "tag": tag,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_auc": per_class_auc,
        "per_dataset": per_dataset,
        "feature_importance": {k: float(v) for k, v in importance.items()},
    }

    return model, metrics, importance


def save_artifacts(model, metrics, importance, feature_cols, tag):
    model_path = OUT_DIR / f"{tag}_model.joblib"
    metrics_path = OUT_DIR / f"{tag}_metrics.json"
    fig_path = OUT_DIR / f"{tag}_feature_importance.png"

    joblib.dump({"model": model, "features": feature_cols}, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(feature_cols) + 1))
    feats = list(importance.keys())
    vals = list(importance.values())
    ax.barh(feats[::-1], vals[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"{tag.upper()} ({metrics['accuracy']:.3f} acc, "
                 f"{metrics['f1_macro']:.3f} F1-macro)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close(fig)

    print(f"  Saved: {model_path.name}, {metrics_path.name}, {fig_path.name}")


# ── MAIN ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--no-fn", action="store_true",
                        help="Train Approach #10: 40 features, no FN model scores")
    parser.add_argument("--force-train", nargs="*", default=None,
                        help="Regex patterns for sequence_ids to force into train "
                             "(moved out of the random test split).")
    parser.add_argument("--in-suffix", type=str, default="",
                        help="Suffix on input CSV: fusion_dataset{in_suffix}.csv")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix on saved model/metrics, e.g. _v3more")
    parser.add_argument("--max-rows-per-dataset", type=int, default=0,
                        help="cap each source_dataset at N rows (sequence-stratified). 0 = no cap")
    parser.add_argument("--exclude-features", type=str, default="",
                        help="comma-separated feature names to drop from training")
    args = parser.parse_args()

    csv_path = DATA_DIR / f"fusion_dataset{args.in_suffix}.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found")
        return

    print(f"Loading...", end="", flush=True)
    df = pd.read_csv(csv_path)
    print(f" {len(df):,} rows")

    # Lever B: per-dataset row cap (sequence-stratified to avoid leakage)
    if args.max_rows_per_dataset > 0:
        cap = args.max_rows_per_dataset
        kept = []
        for ds in df["source_dataset"].unique():
            sub = df[df["source_dataset"] == ds]
            if len(sub) <= cap:
                kept.append(sub)
                continue
            # Take whole sequences until cap is reached
            sub = sub.copy()
            sub["sequence_id"] = sub.apply(
                lambda r: extract_sequence_id(r["base_stem"], r["source_dataset"]),
                axis=1)
            seqs = sub["sequence_id"].unique()
            rng = np.random.RandomState(42)
            rng.shuffle(seqs)
            picked, n = [], 0
            for s in seqs:
                ssub = sub[sub["sequence_id"] == s]
                if n + len(ssub) > cap and picked:
                    break
                picked.append(ssub)
                n += len(ssub)
            kept.append(pd.concat(picked, ignore_index=True))
            print(f"  [cap] {ds}: {len(sub):,} -> {n:,} rows ({len(picked)} sequences)")
        df = pd.concat(kept, ignore_index=True)
        print(f"  After cap: {len(df):,} rows")

    # Lever C: drop named features (e.g. detection-flag shortcut)
    excluded = set(f.strip() for f in args.exclude_features.split(",") if f.strip())
    if excluded:
        print(f"  [excl] dropping {len(excluded)} feature(s): {sorted(excluded)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cheap baselines first
    print(f"\n{'=' * 60}")
    print("CHEAP SANITY BASELINES (on full data)")
    print(f"{'=' * 60}")
    baselines = cheap_baselines(df)
    print(f"  {'policy':<16s} {'acc':>7s} {'f1_m':>7s} {'f1_w':>7s}")
    for name, res in baselines.items():
        print(f"  {name:<16s} {res['accuracy']:>7.4f} "
              f"{res['f1_macro']:>7.4f} {res['f1_weighted']:>7.4f}")

    if args.ablation:
        ablation_results = {}
        for name, features in ABLATION_SETS.items():
            missing = [c for c in features if c not in df.columns]
            if missing:
                print(f"\n  [SKIP] {name}: missing {missing}")
                continue
            model, metrics, importance = train_and_eval(df, features, tag=name)
            metrics["baselines"] = baselines
            save_artifacts(model, metrics, importance, features, name)
            ablation_results[name] = metrics

        # Ablation summary
        print(f"\n{'=' * 70}")
        print("ABLATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"  {'model':<16s} {'feats':>5s} {'acc':>7s} {'f1_m':>7s} {'f1_w':>7s}")

        # Baselines first
        for name, res in baselines.items():
            print(f"  {name:<16s} {'--':>5s} {res['accuracy']:>7.4f} "
                  f"{res['f1_macro']:>7.4f} {res['f1_weighted']:>7.4f}")

        # Trained models
        for name, m in ablation_results.items():
            print(f"  {name:<16s} {m['n_features']:>5d} {m['accuracy']:>7.4f} "
                  f"{m['f1_macro']:>7.4f} {m['f1_weighted']:>7.4f}")

        with open(OUT_DIR / "ablation_summary.json", "w") as f:
            json.dump({"baselines": baselines, "models": ablation_results}, f, indent=2)
        print(f"\n  Saved: ablation_summary.json")

    elif args.no_fn:
        # Approach #10: 40 features, no FN model scores
        feat_cols = [f for f in NO_FN_FEATURES if f not in excluded]
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            print(f"[ERROR] Missing: {missing}")
            return
        tag = f"fusion_no_fn{args.out_suffix}"
        model, metrics, importance = train_and_eval(
            df, feat_cols, tag=tag,
            force_train_patterns=args.force_train)
        metrics["baselines"] = baselines
        save_artifacts(model, metrics, importance, feat_cols, tag)

    else:
        feat_cols = [f for f in FULL_FEATURES if f not in excluded]
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            print(f"[ERROR] Missing: {missing}")
            return
        tag = f"fusion{args.out_suffix}"
        model, metrics, importance = train_and_eval(
            df, feat_cols, tag=tag,
            force_train_patterns=args.force_train)
        metrics["baselines"] = baselines
        save_artifacts(model, metrics, importance, feat_cols, tag)

    print("\nDone.")


if __name__ == "__main__":
    main()
