"""
eval_all_fusion.py — Evaluate all 10 fusion approaches on the same dataset.

Loads fusion_dataset.csv, creates a shared sequence-level train/test split,
and evaluates every approach on the same test set for fair comparison.

Approaches:
  1. OR gate (simple union)
  2. AND gate (simple intersection)
  3. Confidence-weighted OR
  4. 4-class trust classifier (full 46 features)
  5. OR + binary FP suppressor
  6. Dynamic thresholding
  7. Hierarchical 2-stage
  8. Consensus + Solo ML
  9. Bayesian score fusion
  10. Plain frame-level classifier (no FN models, 40 features)

Outputs:
  fusion_comparison.json — all metrics
  fusion_comparison.png  — comparison bar chart
  per-approach confusion matrices

Usage:
    python eval_all_fusion.py
    python eval_all_fusion.py --ablation   # also run feature ablation
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
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "fusion"
OUT_DIR    = DATA_DIR / "comparison"

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}


# ── FEATURE COLUMN DEFINITIONS ────────────────────────────────────

FULL_FEATURES = [
    "rgb_n_dets", "rgb_max_conf", "rgb_mean_conf",
    "rgb_max_fn", "rgb_mean_fn", "rgb_min_fn", "rgb_detected",
    "ir_n_dets", "ir_max_conf", "ir_mean_conf",
    "ir_max_fn", "ir_mean_fn", "ir_min_fn", "ir_detected",
    "rgb_img_mean", "rgb_img_std", "rgb_img_dynamic_range",
    "rgb_img_entropy", "rgb_sky_ground_ratio", "rgb_edge_density",
    "rgb_blurriness",
    "ir_img_mean", "ir_img_std", "ir_img_dynamic_range",
    "ir_img_entropy", "ir_sky_ground_ratio", "ir_edge_density",
    "ir_blurriness",
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
    "rgb_best_pos_x", "rgb_best_pos_y", "rgb_best_dist_to_center",
    "rgb_best_local_contrast", "rgb_best_target_bg_delta",
    "ir_best_log_bbox_area", "ir_best_aspect_ratio",
    "ir_best_pos_x", "ir_best_pos_y", "ir_best_dist_to_center",
    "ir_best_local_contrast", "ir_best_target_bg_delta",
    "both_detect", "neither_detect", "rgb_only_detect", "ir_only_detect",
]

NO_FN_FEATURES = [f for f in FULL_FEATURES
                   if f not in ("rgb_max_fn", "rgb_mean_fn", "rgb_min_fn",
                                "ir_max_fn", "ir_mean_fn", "ir_min_fn")]

BASELINE_FEATURES = [
    "rgb_n_dets", "rgb_max_conf", "rgb_mean_conf", "rgb_detected",
    "ir_n_dets", "ir_max_conf", "ir_mean_conf", "ir_detected",
    "both_detect", "neither_detect", "rgb_only_detect", "ir_only_detect",
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
    "ir_best_log_bbox_area", "ir_best_aspect_ratio",
    "rgb_best_dist_to_center", "ir_best_dist_to_center",
]


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


def get_shared_split(df, test_size=0.25, random_state=42):
    """Create a single train/test split shared by all approaches."""
    df = df.copy()
    df["sequence_id"] = df.apply(
        lambda r: extract_sequence_id(r["base_stem"], r["source_dataset"]), axis=1
    )
    groups = df["sequence_id"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, df["trust_label"], groups=groups))

    train_seqs = set(df.iloc[train_idx]["sequence_id"])
    test_seqs = set(df.iloc[test_idx]["sequence_id"])
    assert len(train_seqs & test_seqs) == 0, "Sequence leakage!"

    return df, train_idx, test_idx


# ── EVALUATION HELPERS ────────────────────────────────────────────

def evaluate_predictions(y_true, y_pred, y_prob=None, src=None):
    """Compute all metrics for a set of predictions."""
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    per_class = {}
    for c in range(4):
        mask_true = y_true == c
        mask_pred = y_pred == c
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[LABEL_NAMES[c]] = {
            "n": int(mask_true.sum()),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    # Per-class AUC (if probabilities available)
    per_class_auc = {}
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 4:
        for c in range(4):
            binary = (y_true == c).astype(int)
            if binary.sum() > 0 and (1 - binary).sum() > 0:
                per_class_auc[LABEL_NAMES[c]] = round(
                    float(roc_auc_score(binary, y_prob[:, c])), 4
                )

    # Per-dataset breakdown
    per_dataset = {}
    if src is not None:
        for ds in np.unique(src):
            mask = src == ds
            ys, yp = y_true[mask], y_pred[mask]
            per_dataset[str(ds)] = {
                "n": int(len(ys)),
                "accuracy": round(float(accuracy_score(ys, yp)), 4),
                "f1_macro": round(float(f1_score(ys, yp, average="macro",
                                                  zero_division=0)), 4),
            }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class": per_class,
        "per_class_auc": per_class_auc,
        "per_dataset": per_dataset,
        "confusion_matrix": cm,
    }


def train_xgb(X_train, y_train, n_classes=4):
    """Train a standard XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


# ── APPROACH IMPLEMENTATIONS ──────────────────────────────────────

def approach_1_or_gate(df_test):
    """Simple OR gate: trust whoever detects."""
    rgb_det = df_test["rgb_detected"].values
    ir_det = df_test["ir_detected"].values

    preds = np.full(len(df_test), 0)
    preds[rgb_det.astype(bool) & ir_det.astype(bool)] = 3  # both → trust_both
    preds[rgb_det.astype(bool) & ~ir_det.astype(bool)] = 1  # rgb only → trust_rgb
    preds[~rgb_det.astype(bool) & ir_det.astype(bool)] = 2  # ir only → trust_ir
    # neither → reject (already 0)
    return preds, None


def approach_2_and_gate(df_test):
    """AND gate: only trust when both detect."""
    rgb_det = df_test["rgb_detected"].values.astype(bool)
    ir_det = df_test["ir_detected"].values.astype(bool)

    preds = np.full(len(df_test), 0)
    preds[rgb_det & ir_det] = 3  # both → trust_both
    return preds, None


def approach_3_conf_weighted(df_train, df_test):
    """Confidence-weighted OR gate with tuned parameters."""
    # Tune boost/penalty on train set
    best_f1 = -1
    best_params = (1.3, 0.7, 0.4)

    for boost in [1.1, 1.2, 1.3, 1.5]:
        for penalty in [0.5, 0.6, 0.7, 0.8]:
            for thresh in [0.3, 0.35, 0.4, 0.45, 0.5]:
                preds = _conf_weighted_predict(df_train, boost, penalty, thresh)
                f1 = f1_score(df_train["trust_label"].values, preds,
                             average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_params = (boost, penalty, thresh)

    preds = _conf_weighted_predict(df_test, *best_params)
    return preds, None


def _conf_weighted_predict(df, boost, penalty, thresh):
    rgb_conf = df["rgb_max_conf"].values
    ir_conf = df["ir_max_conf"].values
    rgb_det = rgb_conf > 0
    ir_det = ir_conf > 0

    # Adjusted confidences
    rgb_adj = np.where(ir_det, rgb_conf * boost, rgb_conf * penalty)
    ir_adj = np.where(rgb_det, ir_conf * boost, ir_conf * penalty)

    rgb_pass = rgb_adj >= thresh
    ir_pass = ir_adj >= thresh

    preds = np.full(len(df), 0)
    preds[rgb_pass & ir_pass] = 3
    preds[rgb_pass & ~ir_pass] = 1
    preds[~rgb_pass & ir_pass] = 2
    return preds


def approach_4_full_classifier(df_train, df_test, train_idx, test_idx, features):
    """4-class XGBoost with all features."""
    X_train = df_train[features].values
    X_test = df_test[features].values
    y_train = df_train["trust_label"].values

    model = train_xgb(X_train, y_train, n_classes=4)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    importance = dict(zip(features, model.feature_importances_))
    return preds, probs, importance


def approach_5_or_fp_suppressor(df_train, df_test):
    """OR gate + binary FP suppressor per modality (frame-level approximation).

    For each frame where only one modality detects:
      - Train binary classifier: is the solo detection a TP?
      - Features: solo det confidence, scene features, FN scores
    """
    # Separate frames by agreement pattern
    train_both = df_train[df_train["both_detect"] == 1]
    train_neither = df_train[df_train["neither_detect"] == 1]
    train_rgb_only = df_train[df_train["rgb_only_detect"] == 1]
    train_ir_only = df_train[df_train["ir_only_detect"] == 1]

    # For solo frames, label = is the solo detection a TP?
    # rgb_only: label = rgb_has_tp
    # ir_only: label = ir_has_tp
    solo_features = [
        "rgb_max_conf", "ir_max_conf", "rgb_mean_conf", "ir_mean_conf",
        "rgb_n_dets", "ir_n_dets",
        "rgb_max_fn", "ir_max_fn", "rgb_mean_fn", "ir_mean_fn",
        "rgb_img_mean", "rgb_img_std", "rgb_edge_density", "rgb_blurriness",
        "ir_img_mean", "ir_img_std", "ir_edge_density", "ir_blurriness",
        "rgb_best_log_bbox_area", "rgb_best_local_contrast",
        "ir_best_log_bbox_area", "ir_best_local_contrast",
    ]

    # Train separate models for rgb_only and ir_only scenarios
    rgb_only_model = None
    if len(train_rgb_only) >= 50:
        X = train_rgb_only[solo_features].values
        y = train_rgb_only["rgb_has_tp"].values
        if len(np.unique(y)) > 1:
            rgb_only_model = train_xgb(X, y, n_classes=2)

    ir_only_model = None
    if len(train_ir_only) >= 50:
        X = train_ir_only[solo_features].values
        y = train_ir_only["ir_has_tp"].values
        if len(np.unique(y)) > 1:
            ir_only_model = train_xgb(X, y, n_classes=2)

    # Predict on test set
    preds = np.full(len(df_test), 0)

    # Both detect → trust_both (OR gate keeps everything)
    both_mask = df_test["both_detect"].values == 1
    preds[both_mask] = 3

    # Neither → reject
    neither_mask = df_test["neither_detect"].values == 1
    preds[neither_mask] = 0

    # RGB only → suppressor decides
    rgb_only_mask = df_test["rgb_only_detect"].values == 1
    if rgb_only_model is not None and rgb_only_mask.sum() > 0:
        X = df_test.loc[rgb_only_mask, solo_features].values
        solo_preds = rgb_only_model.predict(X)
        preds_rgb = np.where(solo_preds == 1, 1, 0)  # trust_rgb if TP, reject if FP
        preds[rgb_only_mask] = preds_rgb
    else:
        preds[rgb_only_mask] = 1  # fallback: trust rgb

    # IR only → suppressor decides
    ir_only_mask = df_test["ir_only_detect"].values == 1
    if ir_only_model is not None and ir_only_mask.sum() > 0:
        X = df_test.loc[ir_only_mask, solo_features].values
        solo_preds = ir_only_model.predict(X)
        preds_ir = np.where(solo_preds == 1, 2, 0)  # trust_ir if TP, reject if FP
        preds[ir_only_mask] = preds_ir
    else:
        preds[ir_only_mask] = 2  # fallback: trust ir

    return preds, None


def approach_6_dynamic_threshold(df_train, df_test):
    """Dynamic thresholding based on scene features.

    Learn per-scene confidence threshold adjustment.
    """
    scene_features = [
        "rgb_img_mean", "rgb_img_std", "rgb_edge_density", "rgb_blurriness",
        "rgb_img_entropy", "rgb_sky_ground_ratio",
        "ir_img_mean", "ir_img_std", "ir_edge_density", "ir_blurriness",
        "ir_img_entropy", "ir_sky_ground_ratio",
    ]

    # Train: for frames with drone_present=1, 
    #   rgb_difficulty = 1 if rgb missed drone (rgb_has_tp=0), else 0
    #   ir_difficulty = 1 if ir missed drone (ir_has_tp=0), else 0
    drone_train = df_train[df_train["drone_present"] == 1]

    rgb_diff_model = None
    ir_diff_model = None

    if len(drone_train) >= 100:
        X = drone_train[scene_features].values
        # RGB difficulty
        y_rgb = (drone_train["rgb_has_tp"] == 0).astype(int).values
        if len(np.unique(y_rgb)) > 1:
            rgb_diff_model = train_xgb(X, y_rgb, n_classes=2)
        # IR difficulty
        y_ir = (drone_train["ir_has_tp"] == 0).astype(int).values
        if len(np.unique(y_ir)) > 1:
            ir_diff_model = train_xgb(X, y_ir, n_classes=2)

    preds = np.full(len(df_test), 0)
    X_test_scene = df_test[scene_features].values

    rgb_conf = df_test["rgb_max_conf"].values
    ir_conf = df_test["ir_max_conf"].values

    # Base threshold
    base_thresh = 0.4

    # Compute per-frame difficulty scores
    if rgb_diff_model is not None:
        rgb_difficulty = rgb_diff_model.predict_proba(X_test_scene)[:, 1]
    else:
        rgb_difficulty = np.full(len(df_test), 0.5)

    if ir_diff_model is not None:
        ir_difficulty = ir_diff_model.predict_proba(X_test_scene)[:, 1]
    else:
        ir_difficulty = np.full(len(df_test), 0.5)

    # Adjusted thresholds: if scene is hard for a modality, raise its threshold
    # (trust it less), lower the other's threshold (trust it more)
    rgb_thresh = base_thresh + 0.2 * rgb_difficulty - 0.1 * ir_difficulty
    ir_thresh = base_thresh + 0.2 * ir_difficulty - 0.1 * rgb_difficulty

    rgb_pass = rgb_conf >= rgb_thresh
    ir_pass = ir_conf >= ir_thresh

    preds[rgb_pass & ir_pass] = 3
    preds[rgb_pass & ~ir_pass] = 1
    preds[~rgb_pass & ir_pass] = 2
    # neither → 0 (already default)

    return preds, None


def approach_7_hierarchical(df_train, df_test):
    """Two-stage: (1) drone present? (2) which modality?"""
    features = NO_FN_FEATURES  # keep it without FN for cleaner comparison

    # Stage 1: binary drone_present
    X_train = df_train[features].values
    y_stage1 = df_train["drone_present"].values
    model_stage1 = train_xgb(X_train, y_stage1, n_classes=2)

    X_test = df_test[features].values
    stage1_pred = model_stage1.predict(X_test)

    # Stage 2: 3-class (trust_rgb=1, trust_ir=2, trust_both=3)
    # Only train on drone_present=1 frames
    drone_train = df_train[df_train["drone_present"] == 1].copy()
    # Remap trust labels: keep 1,2,3 only
    drone_labels = drone_train["trust_label"].values
    # Filter out class 0 (reject_both with drone present = very rare)
    valid = drone_labels > 0
    if valid.sum() >= 50:
        X_s2 = drone_train.loc[valid.astype(bool), features].values
        y_s2 = drone_labels[valid] - 1  # remap to 0,1,2
        model_stage2 = train_xgb(X_s2, y_s2, n_classes=3)

        # Apply stage 2 only where stage 1 says drone present
        drone_mask = stage1_pred == 1
        preds = np.zeros(len(df_test), dtype=int)
        if drone_mask.sum() > 0:
            s2_preds = model_stage2.predict(X_test[drone_mask]) + 1  # remap back
            preds[drone_mask] = s2_preds
    else:
        # Fallback to stage 1 only
        preds = np.where(stage1_pred == 1, 3, 0)

    return preds, None


def approach_8_consensus_solo(df_train, df_test):
    """Consensus + Solo ML: rules for easy cases, ML for disagreement."""
    solo_features = [
        "rgb_max_conf", "ir_max_conf", "rgb_mean_conf", "ir_mean_conf",
        "rgb_n_dets", "ir_n_dets",
        "rgb_max_fn", "ir_max_fn", "rgb_mean_fn", "ir_mean_fn",
        "rgb_img_mean", "rgb_img_std", "rgb_edge_density", "rgb_blurriness",
        "rgb_img_entropy", "rgb_sky_ground_ratio",
        "ir_img_mean", "ir_img_std", "ir_edge_density", "ir_blurriness",
        "ir_img_entropy", "ir_sky_ground_ratio",
        "rgb_best_log_bbox_area", "rgb_best_local_contrast",
        "rgb_best_target_bg_delta",
        "ir_best_log_bbox_area", "ir_best_local_contrast",
        "ir_best_target_bg_delta",
    ]

    # Only train on disagreement frames
    train_solo = df_train[
        (df_train["rgb_only_detect"] == 1) | (df_train["ir_only_detect"] == 1)
    ].copy()

    # Label for solo frames: what's the correct trust decision?
    # This is the original trust_label (0=reject, 1=trust_rgb, 2=trust_ir)
    # Note: trust_both (3) shouldn't appear in solo frames

    solo_model = None
    if len(train_solo) >= 50 and len(train_solo["trust_label"].unique()) > 1:
        X = train_solo[solo_features].values
        y = train_solo["trust_label"].values
        n_classes = len(np.unique(y))
        if n_classes > 1:
            solo_model = train_xgb(X, y, n_classes=4)  # keep 4-class for label compat

    # Predict
    preds = np.full(len(df_test), 0)

    # Mode A: both detect → trust_both (hard rule)
    both_mask = df_test["both_detect"].values == 1
    preds[both_mask] = 3

    # Mode B: neither detect → reject (hard rule)
    neither_mask = df_test["neither_detect"].values == 1
    preds[neither_mask] = 0

    # Mode C: disagreement → ML
    solo_mask = (df_test["rgb_only_detect"].values == 1) | \
                (df_test["ir_only_detect"].values == 1)
    if solo_model is not None and solo_mask.sum() > 0:
        X = df_test.loc[solo_mask, solo_features].values
        preds[solo_mask] = solo_model.predict(X)
    else:
        # Fallback: trust whoever detected
        rgb_only = df_test["rgb_only_detect"].values == 1
        ir_only = df_test["ir_only_detect"].values == 1
        preds[rgb_only] = 1
        preds[ir_only] = 2

    return preds, None


def approach_9_bayesian(df_train, df_test):
    """Bayesian score fusion using empirical likelihood estimation."""
    # Estimate P(rgb_conf_bin | drone) and P(rgb_conf_bin | no_drone) from train
    conf_bins = np.linspace(0, 1, 21)  # 20 bins

    drone_train = df_train[df_train["drone_present"] == 1]
    no_drone_train = df_train[df_train["drone_present"] == 0]

    prior_drone = len(drone_train) / len(df_train)

    def bin_likelihoods(values, bins):
        hist, _ = np.histogram(values, bins=bins)
        hist = hist.astype(float) + 1  # Laplace smoothing
        return hist / hist.sum()

    # P(rgb_conf | drone), P(rgb_conf | no_drone)
    rgb_lik_drone = bin_likelihoods(drone_train["rgb_max_conf"].values, conf_bins)
    rgb_lik_no = bin_likelihoods(no_drone_train["rgb_max_conf"].values, conf_bins)

    ir_lik_drone = bin_likelihoods(drone_train["ir_max_conf"].values, conf_bins)
    ir_lik_no = bin_likelihoods(no_drone_train["ir_max_conf"].values, conf_bins)

    # Predict on test
    rgb_conf_test = df_test["rgb_max_conf"].values
    ir_conf_test = df_test["ir_max_conf"].values

    rgb_bins = np.clip(np.digitize(rgb_conf_test, conf_bins) - 1, 0, len(rgb_lik_drone) - 1)
    ir_bins = np.clip(np.digitize(ir_conf_test, conf_bins) - 1, 0, len(ir_lik_drone) - 1)

    # P(drone | rgb_conf, ir_conf) ∝ P(rgb|drone)*P(ir|drone)*P(drone)
    log_p_drone = (np.log(rgb_lik_drone[rgb_bins] + 1e-10) +
                   np.log(ir_lik_drone[ir_bins] + 1e-10) +
                   np.log(prior_drone + 1e-10))
    log_p_no = (np.log(rgb_lik_no[rgb_bins] + 1e-10) +
                np.log(ir_lik_no[ir_bins] + 1e-10) +
                np.log(1 - prior_drone + 1e-10))

    p_drone = 1 / (1 + np.exp(log_p_no - log_p_drone))

    # Map to trust decision
    preds = np.full(len(df_test), 0)
    drone_pred = p_drone > 0.5

    rgb_det = df_test["rgb_detected"].values.astype(bool)
    ir_det = df_test["ir_detected"].values.astype(bool)

    # If Bayes says drone and both detect → trust_both
    preds[drone_pred & rgb_det & ir_det] = 3
    # If Bayes says drone and only rgb → trust_rgb
    preds[drone_pred & rgb_det & ~ir_det] = 1
    # If Bayes says drone and only ir → trust_ir
    preds[drone_pred & ~rgb_det & ir_det] = 2
    # If Bayes says no drone → reject (0)

    return preds, None


def approach_10_no_fn(df_train, df_test):
    """Plain frame-level classifier without FN model features."""
    X_train = df_train[NO_FN_FEATURES].values
    X_test = df_test[NO_FN_FEATURES].values
    y_train = df_train["trust_label"].values

    model = train_xgb(X_train, y_train, n_classes=4)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    importance = dict(zip(NO_FN_FEATURES, model.feature_importances_))
    return preds, probs, importance


# ── MAIN ──────────────────────────────────────────────────────────

APPROACHES = {
    "01_or_gate": {
        "name": "OR Gate",
        "type": "rule",
        "description": "Accept any detection from either modality",
    },
    "02_and_gate": {
        "name": "AND Gate",
        "type": "rule",
        "description": "Only trust when both modalities detect",
    },
    "03_conf_weighted": {
        "name": "Conf-Weighted OR",
        "type": "rule",
        "description": "OR gate with tuned confidence boost/penalty",
    },
    "04_full_classifier": {
        "name": "4-Class Full (46 feat)",
        "type": "ml",
        "description": "XGBoost 4-class with all 46 features incl FN models",
    },
    "05_or_fp_suppressor": {
        "name": "OR + FP Suppressor",
        "type": "ml",
        "description": "OR gate + binary FP suppressor for solo detections",
    },
    "06_dynamic_threshold": {
        "name": "Dynamic Threshold",
        "type": "ml",
        "description": "Scene-adaptive confidence thresholds",
    },
    "07_hierarchical": {
        "name": "Hierarchical 2-Stage",
        "type": "ml",
        "description": "Stage 1: drone present? Stage 2: which modality?",
    },
    "08_consensus_solo": {
        "name": "Consensus + Solo ML",
        "type": "ml",
        "description": "Rules for agreement, ML for disagreement only",
    },
    "09_bayesian": {
        "name": "Bayesian Fusion",
        "type": "rule",
        "description": "Bayesian score fusion with empirical likelihoods",
    },
    "10_no_fn_classifier": {
        "name": "Plain Classifier (40 feat)",
        "type": "ml",
        "description": "XGBoost 4-class with 40 features, no FN model scores",
    },
}


def run_approach(key, df, df_train, df_test, train_idx, test_idx):
    """Run one approach and return (preds, probs_or_none)."""
    if key == "01_or_gate":
        return approach_1_or_gate(df_test) + (None,)
    elif key == "02_and_gate":
        return approach_2_and_gate(df_test) + (None,)
    elif key == "03_conf_weighted":
        return approach_3_conf_weighted(df_train, df_test) + (None,)
    elif key == "04_full_classifier":
        return approach_4_full_classifier(df_train, df_test, train_idx, test_idx,
                                           FULL_FEATURES)
    elif key == "05_or_fp_suppressor":
        return approach_5_or_fp_suppressor(df_train, df_test) + (None,)
    elif key == "06_dynamic_threshold":
        return approach_6_dynamic_threshold(df_train, df_test) + (None,)
    elif key == "07_hierarchical":
        return approach_7_hierarchical(df_train, df_test) + (None,)
    elif key == "08_consensus_solo":
        return approach_8_consensus_solo(df_train, df_test) + (None,)
    elif key == "09_bayesian":
        return approach_9_bayesian(df_train, df_test) + (None,)
    elif key == "10_no_fn_classifier":
        return approach_10_no_fn(df_train, df_test)
    else:
        raise ValueError(f"Unknown approach: {key}")


# ── PLOTTING ──────────────────────────────────────────────────────

def plot_comparison(all_results, out_path):
    """Bar chart comparing all approaches."""
    names = [r["name"] for r in all_results]
    accs = [r["metrics"]["accuracy"] for r in all_results]
    f1_macros = [r["metrics"]["f1_macro"] for r in all_results]
    f1_weights = [r["metrics"]["f1_weighted"] for r in all_results]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    bars1 = ax.bar(x - width, accs, width, label="Accuracy", color="#3498db")
    bars2 = ax.bar(x, f1_macros, width, label="F1 Macro", color="#e74c3c")
    bars3 = ax.bar(x + width, f1_weights, width, label="F1 Weighted", color="#2ecc71")

    ax.set_ylabel("Score")
    ax.set_title("Fusion Approach Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", va="bottom", fontsize=6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_confusion_matrices(all_results, out_dir):
    """Plot confusion matrix for each approach."""
    n = len(all_results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    class_names = ["reject", "rgb", "ir", "both"]

    for i, res in enumerate(all_results):
        ax = axes[i]
        cm = np.array(res["metrics"]["confusion_matrix"])
        # Normalize by row
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(class_names, fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        ax.set_title(f"{res['name']}\nacc={res['metrics']['accuracy']:.3f}",
                     fontsize=8)

        # Annotate cells
        for ii in range(4):
            for jj in range(4):
                ax.text(jj, ii, f"{cm[ii, jj]}",
                       ha="center", va="center", fontsize=6,
                       color="white" if cm_norm[ii, jj] > 0.5 else "black")

    # Hide unused axes
    for i in range(len(all_results), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Confusion Matrices (rows=true, cols=pred)", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=120)
    plt.close(fig)
    print(f"  Saved: confusion_matrices.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approaches", nargs="*", default=None,
                        help="Specific approaches to run (e.g., 01 04 08)")
    args = parser.parse_args()

    csv_path = DATA_DIR / "fusion_dataset.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found. Run build_fusion_dataset.py first.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading...", end="", flush=True)
    df = pd.read_csv(csv_path)
    print(f" {len(df):,} rows")

    # Shared split
    print("Creating shared train/test split...")
    df, train_idx, test_idx = get_shared_split(df)
    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    y_true = df_test["trust_label"].values
    src_test = df_test["source_dataset"].values

    print(f"  Train: {len(df_train):,}  Test: {len(df_test):,}")
    print(f"  Sequences: {df['sequence_id'].nunique()} total")

    # Label distribution
    print(f"\n  Test set label distribution:")
    for val, name in LABEL_NAMES.items():
        n = (y_true == val).sum()
        print(f"    {val} ({name:<12s}): {n:>8,} ({n / len(y_true) * 100:>5.1f}%)")

    # Run all approaches
    all_results = []
    approach_keys = sorted(APPROACHES.keys())

    if args.approaches:
        approach_keys = [k for k in approach_keys
                        if any(a in k for a in args.approaches)]

    for key in approach_keys:
        info = APPROACHES[key]
        print(f"\n{'=' * 60}")
        print(f"Approach {key}: {info['name']}")
        print(f"  {info['description']}")
        print(f"{'=' * 60}")

        try:
            preds, probs, importance = run_approach(
                key, df, df_train, df_test, train_idx, test_idx
            )
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()
            continue

        metrics = evaluate_predictions(y_true, preds, probs, src_test)

        print(f"\n  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):   {metrics['f1_macro']:.4f}")
        print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")

        # Per-class
        print(f"\n  Per-class:")
        print(f"    {'class':<14s} {'n':>6s} {'prec':>7s} {'rec':>7s} {'f1':>7s}")
        for cname, cm in metrics["per_class"].items():
            print(f"    {cname:<14s} {cm['n']:>6,} {cm['precision']:>7.4f} "
                  f"{cm['recall']:>7.4f} {cm['f1']:>7.4f}")

        # Per-dataset
        if metrics["per_dataset"]:
            print(f"\n  Per-dataset:")
            print(f"    {'dataset':<15s} {'n':>7s} {'acc':>7s} {'f1_m':>7s}")
            for ds, dm in metrics["per_dataset"].items():
                print(f"    {ds:<15s} {dm['n']:>7,} {dm['accuracy']:>7.4f} "
                      f"{dm['f1_macro']:>7.4f}")

        if importance:
            sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
            print(f"\n  Top 5 features:")
            for feat, val in sorted_imp[:5]:
                print(f"    {feat:<35s} {val:.4f}")

        result = {
            "key": key,
            "name": info["name"],
            "type": info["type"],
            "description": info["description"],
            "metrics": metrics,
        }
        if importance:
            result["feature_importance"] = {k: round(float(v), 4)
                                            for k, v in sorted(importance.items(),
                                                               key=lambda x: -x[1])}
        all_results.append(result)

    # Summary table
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'#':<3s} {'Approach':<25s} {'Type':<5s} "
          f"{'Acc':>7s} {'F1_m':>7s} {'F1_w':>7s}")
    print(f"  {'-' * 55}")
    for r in sorted(all_results, key=lambda x: -x["metrics"]["f1_macro"]):
        print(f"  {r['key'][:2]:<3s} {r['name']:<25s} {r['type']:<5s} "
              f"{r['metrics']['accuracy']:>7.4f} "
              f"{r['metrics']['f1_macro']:>7.4f} "
              f"{r['metrics']['f1_weighted']:>7.4f}")

    # Save results
    json_path = OUT_DIR / "fusion_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {json_path.name}")

    # Plots
    plot_comparison(all_results, OUT_DIR / "fusion_comparison.png")
    plot_confusion_matrices(all_results, OUT_DIR)

    print("\nDone. All 10 approaches evaluated on the same test set.")


if __name__ == "__main__":
    main()
