"""Test 5 (FIXED): 8-feature vs 40-feature with PROPER sequence split from train_fusion.py."""
import re
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

FUSION_DIR = Path(__file__).resolve().parents[2] / "runs" / "reliability" / "fusion"   # resident classifier/ (was ES_Drone_Detection); legacy run data not shipped

# ── Load ──
bundle = joblib.load(FUSION_DIR / "fusion_no_fn_model.joblib")
model = bundle["model"]
feature_names = bundle["features"]
df = pd.read_csv(FUSION_DIR / "fusion_dataset.csv")
print(f"Loaded {len(df):,} frames")

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}

# ── Exact sequence ID extraction from train_fusion.py ──
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

df["sequence_id"] = df.apply(
    lambda r: extract_sequence_id(r["base_stem"], r["source_dataset"]), axis=1
)
n_seqs = df["sequence_id"].nunique()
print(f"Unique sequences: {n_seqs}")

# ── Exact split from train_fusion.py ──
groups = df["sequence_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(gss.split(df, df["trust_label"], groups=groups))

train_seqs = set(df.iloc[train_idx]["sequence_id"])
test_seqs = set(df.iloc[test_idx]["sequence_id"])
assert len(train_seqs & test_seqs) == 0, "Sequence leakage!"

df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]
y_train = df_train["trust_label"].values
y_test = df_test["trust_label"].values
print(f"Train: {len(df_train):,}  Test: {len(df_test):,}")
print(f"Train seqs: {len(train_seqs)}  Test seqs: {len(test_seqs)}")

# ── Full model on test ──
X_test_full = df_test[feature_names].values
full_preds = model.predict(X_test_full)

# ── Train minimal 8-feature model ──
core_features = ["rgb_detected", "ir_detected", "rgb_n_dets", "ir_n_dets",
                 "both_detect", "neither_detect", "rgb_only_detect", "ir_only_detect"]

print("\nTraining 8-feature model (same split)...")
mini = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="multi:softprob", num_class=4,
    eval_metric="mlogloss", tree_method="hist",
    random_state=42, n_jobs=-1,
)
mini.fit(df_train[core_features].values, y_train, verbose=False)
mini_preds = mini.predict(df_test[core_features].values)

# ── Also train with detection + confidence features (12 features) ──
conf_features = core_features + ["rgb_max_conf", "ir_max_conf", "rgb_mean_conf", "ir_mean_conf"]

print("Training 12-feature model (+ confidence)...")
mid = XGBClassifier(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective="multi:softprob", num_class=4,
    eval_metric="mlogloss", tree_method="hist",
    random_state=42, n_jobs=-1,
)
mid.fit(df_train[conf_features].values, y_train, verbose=False)
mid_preds = mid.predict(df_test[conf_features].values)

# ── Results ──
full_acc = accuracy_score(y_test, full_preds)
full_f1m = f1_score(y_test, full_preds, average="macro", zero_division=0)
mini_acc = accuracy_score(y_test, mini_preds)
mini_f1m = f1_score(y_test, mini_preds, average="macro", zero_division=0)
mid_acc = accuracy_score(y_test, mid_preds)
mid_f1m = f1_score(y_test, mid_preds, average="macro", zero_division=0)

print()
print("=" * 65)
print("FULL vs MINIMAL vs MID MODEL (proper sequence split)")
print("=" * 65)
print(f"  {'Model':<30s} {'Accuracy':>9s}  {'F1 macro':>9s}  {'# feat':>6s}")
print(f"  {'-'*30} {'-'*9}  {'-'*9}  {'-'*6}")
print(f"  {'Full (40 features)':<30s} {full_acc:>9.4f}  {full_f1m:>9.4f}  {40:>6d}")
print(f"  {'Mid (12: det+conf)':<30s} {mid_acc:>9.4f}  {mid_f1m:>9.4f}  {12:>6d}")
print(f"  {'Minimal (8: det+agree)':<30s} {mini_acc:>9.4f}  {mini_f1m:>9.4f}  {8:>6d}")
print(f"  {'-'*30} {'-'*9}  {'-'*9}  {'-'*6}")
print(f"  {'Full - Minimal':<30s} {full_acc - mini_acc:>+9.4f}  {full_f1m - mini_f1m:>+9.4f}")
print(f"  {'Full - Mid':<30s} {full_acc - mid_acc:>+9.4f}  {full_f1m - mid_f1m:>+9.4f}")

print(f"\n  Agreement:")
print(f"    Full vs Minimal: {(full_preds == mini_preds).mean() * 100:.2f}%")
print(f"    Full vs Mid:     {(full_preds == mid_preds).mean() * 100:.2f}%")

print(f"\n  Per-class F1:")
print(f"  {'Class':<14s} {'Full':>7s}  {'Mid':>7s}  {'Mini':>7s}  {'Full-Mini':>9s}")
print(f"  {'-'*14} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")
for c in range(4):
    f1_full = f1_score(y_test == c, full_preds == c, zero_division=0)
    f1_mid = f1_score(y_test == c, mid_preds == c, zero_division=0)
    f1_mini = f1_score(y_test == c, mini_preds == c, zero_division=0)
    n = (y_test == c).sum()
    print(f"  {LABEL_NAMES[c]:<14s} {f1_full:>7.4f}  {f1_mid:>7.4f}  {f1_mini:>7.4f}  {f1_full - f1_mini:>+9.4f}  (n={n:,})")
