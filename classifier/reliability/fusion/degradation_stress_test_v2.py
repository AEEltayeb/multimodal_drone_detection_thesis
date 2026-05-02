"""
Advanced Degradation Stress Test — Pushes scene features to in-distribution
extremes and sweeps degradation levels to find if/where scene features matter.

Tests:
  1. Per-feature percentile sweep (50th → 99th)
  2. Coordinated multi-feature extreme (all scene features at "bad" extremes)
  3. Real degraded frame subsets (genuinely bad conditions from the data)
  4. Feature ablation: remove scene features entirely vs keep them
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── PATHS ──
CLASSIFIER_DIR = Path(r"c:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace\ES_Drone_Detection\classifier")
FUSION_DIR = CLASSIFIER_DIR / "runs" / "reliability" / "fusion"

# ── LOAD ──
print("Loading model and data...")
bundle = joblib.load(FUSION_DIR / "fusion_no_fn_model.joblib")
model = bundle["model"]
feature_names = bundle["features"]

df = pd.read_csv(FUSION_DIR / "fusion_dataset.csv")
print(f"  {len(df):,} frames, {len(feature_names)} features")

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}

# Focus on trust_both frames
tb = df[df["trust_label"] == 3].copy()
X_tb = tb[feature_names].values
baseline_preds = model.predict(X_tb)
print(f"  {len(tb):,} trust_both frames")
print(f"  Baseline: {(baseline_preds == 3).sum():,} stay trust_both ({(baseline_preds==3).mean()*100:.2f}%)")

# Scene feature groups
RGB_SCENE = ["rgb_img_mean", "rgb_img_std", "rgb_img_dynamic_range",
             "rgb_img_entropy", "rgb_edge_density", "rgb_blurriness",
             "rgb_sky_ground_ratio"]
IR_SCENE = ["ir_img_mean", "ir_img_std", "ir_img_dynamic_range",
            "ir_img_entropy", "ir_edge_density", "ir_blurriness",
            "ir_sky_ground_ratio"]
RGB_TARGET = ["rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
              "rgb_best_pos_x", "rgb_best_pos_y", "rgb_best_dist_to_center",
              "rgb_best_local_contrast", "rgb_best_target_bg_delta"]
IR_TARGET = ["ir_best_log_bbox_area", "ir_best_aspect_ratio",
             "ir_best_pos_x", "ir_best_pos_y", "ir_best_dist_to_center",
             "ir_best_local_contrast", "ir_best_target_bg_delta"]
ALL_SCENE = RGB_SCENE + IR_SCENE
ALL_TARGET = RGB_TARGET + IR_TARGET

# Feature index map
feat_idx = {f: i for i, f in enumerate(feature_names)}

# Compute percentiles from ALL data (training distribution proxy)
percentiles = {}
for f in feature_names:
    vals = df[f].values
    percentiles[f] = {
        p: float(np.percentile(vals, p))
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
    }

# "Bad" direction for each scene feature (which extreme = degraded)
BAD_DIRECTION = {
    # Low blurriness (Laplacian variance) = blurry image
    "rgb_blurriness": "low", "ir_blurriness": "low",
    # Low std = low contrast
    "rgb_img_std": "low", "ir_img_std": "low",
    # Low dynamic range = compressed
    "rgb_img_dynamic_range": "low", "ir_img_dynamic_range": "low",
    # Low entropy = less information
    "rgb_img_entropy": "low", "ir_img_entropy": "low",
    # Low edge density = fewer features
    "rgb_edge_density": "low", "ir_edge_density": "low",
    # Extreme mean = over/underexposed — use low (dark)
    "rgb_img_mean": "low", "ir_img_mean": "low",
    # sky_ground_ratio — extreme high or low both bad, use low
    "rgb_sky_ground_ratio": "low", "ir_sky_ground_ratio": "low",
}


def switch_rate(preds):
    """Fraction of predictions that changed from trust_both."""
    return float((preds != 3).sum()) / len(preds) * 100


# ═══════════════════════════════════════════════════════════════
# TEST 1: Per-feature percentile sweep
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 1: Per-feature percentile sweep (set to in-distribution extreme)")
print("=" * 70)
print(f"  {'Feature':<30s}  {'P1':>6s}  {'P5':>6s}  {'P10':>6s}  {'P50':>6s}  {'P90':>6s}  {'P95':>6s}  {'P99':>6s}")
print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

for f in ALL_SCENE:
    direction = BAD_DIRECTION.get(f, "low")
    results = []

    if direction == "low":
        test_pcts = [50, 25, 10, 5, 1]  # progressively worse
        header_pcts = ["P50", "P25", "P10", "P5", "P1"]
    else:
        test_pcts = [50, 75, 90, 95, 99]
        header_pcts = ["P50", "P75", "P90", "P95", "P99"]

    for pct in test_pcts:
        X_mod = X_tb.copy()
        idx = feat_idx[f]
        X_mod[:, idx] = percentiles[f][pct]
        preds_mod = model.predict(X_mod)
        results.append(switch_rate(preds_mod))

    print(f"  {f:<30s}  " + "  ".join(f"{r:>5.2f}%" for r in results))


# ═══════════════════════════════════════════════════════════════
# TEST 2: Coordinated multi-feature extreme
# Set ALL RGB scene features to their "bad" extreme simultaneously
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: Coordinated multi-feature extremes")
print("=" * 70)

for label, features in [("All RGB scene", RGB_SCENE),
                         ("All IR scene", IR_SCENE),
                         ("All RGB+IR scene", ALL_SCENE),
                         ("All RGB target", RGB_TARGET),
                         ("All IR target", IR_TARGET),
                         ("ALL scene+target", ALL_SCENE + ALL_TARGET)]:
    for pct_label, pct_val in [("P5", 5), ("P1", 1)]:
        X_mod = X_tb.copy()
        for f in features:
            idx = feat_idx[f]
            direction = BAD_DIRECTION.get(f, "low")
            if direction == "low":
                X_mod[:, idx] = percentiles[f][pct_val]
            else:
                X_mod[:, idx] = percentiles[f][100 - pct_val]
        preds_mod = model.predict(X_mod)
        sr = switch_rate(preds_mod)
        n_switched = (preds_mod != 3).sum()
        # breakdown
        to_reject = (preds_mod == 0).sum()
        to_rgb = (preds_mod == 1).sum()
        to_ir = (preds_mod == 2).sum()
        print(f"  {label:<25s} @ {pct_label}: switched {sr:>5.2f}% ({n_switched:>5,})  "
              f"[reject={to_reject:,}, rgb={to_rgb:,}, ir={to_ir:,}]")


# ═══════════════════════════════════════════════════════════════
# TEST 3: Real degraded frame subsets
# Find frames with genuinely bad conditions
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: Real degraded frames from the dataset")
print("=" * 70)

conditions = [
    ("Dark RGB (mean < 30)", tb["rgb_img_mean"] < 30),
    ("Dark RGB (mean < 15)", tb["rgb_img_mean"] < 15),
    ("Low-contrast RGB (std < 10)", tb["rgb_img_std"] < 10),
    ("Low-contrast IR (std < 10)", tb["ir_img_std"] < 10),
    ("Very blurry RGB (blur < P5)", tb["rgb_blurriness"] < np.percentile(df["rgb_blurriness"], 5)),
    ("Very blurry IR (blur < P5)", tb["ir_blurriness"] < np.percentile(df["ir_blurriness"], 5)),
    ("Low IR dynamic range (< P5)", tb["ir_img_dynamic_range"] < np.percentile(df["ir_img_dynamic_range"], 5)),
    ("Low RGB entropy (< P5)", tb["rgb_img_entropy"] < np.percentile(df["rgb_img_entropy"], 5)),
    ("RGB only (ir_n_dets=0 in full df)", df[df["trust_label"] == 1]["rgb_img_mean"] < 999),  # skip
]

print(f"  {'Condition':<40s} {'N frames':>8s}  {'trust_both':>10s}  {'trust_ir':>8s}  {'trust_rgb':>9s}  {'reject':>6s}")
print(f"  {'-'*40} {'-'*8}  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*6}")

for label, mask in conditions[:8]:
    subset = tb[mask]
    n = len(subset)
    if n == 0:
        print(f"  {label:<40s} {0:>8,}  (empty)")
        continue
    X_sub = subset[feature_names].values
    preds = model.predict(X_sub)
    tb_pct = (preds == 3).sum() / n * 100
    ir_pct = (preds == 2).sum() / n * 100
    rgb_pct = (preds == 1).sum() / n * 100
    rej_pct = (preds == 0).sum() / n * 100
    print(f"  {label:<40s} {n:>8,}  {tb_pct:>9.1f}%  {ir_pct:>7.1f}%  {rgb_pct:>8.1f}%  {rej_pct:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# TEST 4: Feature ablation — what if scene features were random?
# Shuffle scene feature values across frames (destroys signal,
# keeps distribution). If accuracy doesn't change → features useless.
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: Feature ablation (shuffle scene features across frames)")
print("=" * 70)

np.random.seed(42)

for label, features in [("Shuffle RGB scene", RGB_SCENE),
                         ("Shuffle IR scene", IR_SCENE),
                         ("Shuffle ALL scene", ALL_SCENE),
                         ("Shuffle ALL target", ALL_TARGET),
                         ("Shuffle ALL scene+target", ALL_SCENE + ALL_TARGET),
                         ("Shuffle EVERYTHING except detected+agreement", 
                          [f for f in feature_names if f not in 
                           ["rgb_detected", "ir_detected", "both_detect", 
                            "neither_detect", "rgb_only_detect", "ir_only_detect",
                            "rgb_n_dets", "ir_n_dets"]])]:
    X_mod = X_tb.copy()
    for f in features:
        if f in feat_idx:
            idx = feat_idx[f]
            X_mod[:, idx] = np.random.permutation(X_mod[:, idx])
    preds_mod = model.predict(X_mod)
    sr = switch_rate(preds_mod)
    # Also compute agreement with baseline
    agree = (preds_mod == baseline_preds).mean() * 100
    print(f"  {label:<50s}  switched: {sr:>5.2f}%  agree_w_baseline: {agree:>5.1f}%")


# ═══════════════════════════════════════════════════════════════
# TEST 5: Detection-only model (7 features)
# What if we only used the 7 detection+agreement features?
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: How many features do we actually need?")
print("=" * 70)

core_features = ["rgb_detected", "ir_detected", "rgb_n_dets", "ir_n_dets",
                 "both_detect", "neither_detect", "rgb_only_detect", "ir_only_detect"]

# Train a minimal model on just these features
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

# Split same as original
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
groups = df["sequence_id"].values
train_idx, test_idx = next(gss.split(df, groups=groups))

df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]

X_train_core = df_train[core_features].values
X_test_core = df_test[core_features].values
y_train = df_train["trust_label"].values
y_test = df_test["trust_label"].values

# Full model prediction on test set
X_test_full = df_test[feature_names].values
full_preds = model.predict(X_test_full)

# Train minimal model
print("  Training 8-feature minimal model...")
mini_model = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="multi:softprob", num_class=4,
    eval_metric="mlogloss", tree_method="hist",
    random_state=42, n_jobs=-1,
)
mini_model.fit(X_train_core, y_train, verbose=False)
mini_preds = mini_model.predict(X_test_core)

from sklearn.metrics import accuracy_score, f1_score

full_acc = accuracy_score(y_test, full_preds)
full_f1m = f1_score(y_test, full_preds, average="macro", zero_division=0)
mini_acc = accuracy_score(y_test, mini_preds)
mini_f1m = f1_score(y_test, mini_preds, average="macro", zero_division=0)

print(f"\n  {'Model':<25s} {'Accuracy':>9s}  {'F1 macro':>9s}  {'Features':>8s}")
print(f"  {'-'*25} {'-'*9}  {'-'*9}  {'-'*8}")
print(f"  {'Full (40 features)':<25s} {full_acc:>9.4f}  {full_f1m:>9.4f}  {40:>8d}")
print(f"  {'Minimal (8 features)':<25s} {mini_acc:>9.4f}  {mini_f1m:>9.4f}  {8:>8d}")
print(f"  {'Delta':<25s} {full_acc - mini_acc:>+9.4f}  {full_f1m - mini_f1m:>+9.4f}")

# Agreement between models
agree = (full_preds == mini_preds).mean() * 100
print(f"\n  Agreement between 40-feature and 8-feature models: {agree:.2f}%")

# Per-class comparison
print(f"\n  {'Class':<14s} {'Full F1':>7s}  {'Mini F1':>7s}  {'Delta':>7s}")
for c in range(4):
    mask = y_test == c
    if mask.sum() == 0:
        continue
    full_f1c = f1_score(y_test == c, full_preds == c, zero_division=0)
    mini_f1c = f1_score(y_test == c, mini_preds == c, zero_division=0)
    print(f"  {LABEL_NAMES[c]:<14s} {full_f1c:>7.4f}  {mini_f1c:>7.4f}  {full_f1c - mini_f1c:>+7.4f}")


print("\n\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(f"  If Test 5 shows <0.5% accuracy drop with 8 features,")
print(f"  then the 32 scene+target features are decorative.")
print(f"  The classifier is fundamentally a detection-agreement gate.")
