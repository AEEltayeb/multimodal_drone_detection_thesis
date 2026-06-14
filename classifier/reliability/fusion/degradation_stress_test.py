"""
Synthetic Degradation Stress Test for the Fusion Classifier.

Takes real trust_both frames and simulates degradation effects on features
to test whether the classifier correctly switches trust.

Three scenarios:
  1. RGB failure (detection killed) — should switch to trust_ir
  2. IR failure (detection killed)  — should switch to trust_rgb
  3. Subtle degradation (detection survives, scene features change) — tests if scene features matter
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── PATHS ──
CLASSIFIER_DIR = Path(__file__).resolve().parents[2]   # resident classifier/ (was ES_Drone_Detection); legacy run data not shipped
FUSION_DIR = CLASSIFIER_DIR / "runs" / "reliability" / "fusion"

# ── LOAD ──
print("Loading model and data...")
bundle = joblib.load(FUSION_DIR / "fusion_no_fn_model.joblib")
model = bundle["model"]
feature_names = bundle["features"]

df = pd.read_csv(FUSION_DIR / "fusion_dataset.csv")
print(f"  {len(df):,} frames, {len(feature_names)} features")

# Focus on trust_both frames (both modalities correct)
tb = df[df["trust_label"] == 3].copy()
print(f"  {len(tb):,} trust_both frames to test")

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}


def classify(features_df):
    """Run classifier on a dataframe, return predictions."""
    X = features_df[feature_names].values
    return model.predict(X)


def summarize(preds, title):
    """Print prediction distribution."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    total = len(preds)
    for val in range(4):
        n = (preds == val).sum()
        pct = n / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {val} ({LABEL_NAMES[val]:12s}): {n:>7,} ({pct:>5.1f}%)  {bar}")
    # Key metric: did it switch away from trust_both?
    switched = (preds != 3).sum()
    print(f"\n  Switched from trust_both: {switched:,} / {total:,} ({switched/total*100:.1f}%)")
    return preds


# ═══════════════════════════════════════════════════════════════
# BASELINE: Original features, no degradation
# ═══════════════════════════════════════════════════════════════
baseline_preds = classify(tb)
summarize(baseline_preds, "BASELINE (no degradation)")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 1: RGB goes blind (motion blur kills RGB YOLO)
# ═══════════════════════════════════════════════════════════════
s1 = tb.copy()

# RGB YOLO produces no detections
s1["rgb_detected"] = 0
s1["rgb_n_dets"] = 0
s1["rgb_max_conf"] = 0.0
s1["rgb_mean_conf"] = 0.0

# Best-det target features zero out
for col in feature_names:
    if col.startswith("rgb_best_"):
        s1[col] = 0.0

# Agreement flags update
s1["both_detect"] = 0
s1["rgb_only_detect"] = 0
s1["ir_only_detect"] = s1["ir_detected"]  # IR still detects
s1["neither_detect"] = (s1["ir_detected"] == 0).astype(int)

# Scene features: simulate heavy blur
s1["rgb_blurriness"] = s1["rgb_blurriness"] * 0.1  # Laplacian variance drops when blurry
s1["rgb_img_std"] = s1["rgb_img_std"] * 0.5  # lower contrast
s1["rgb_edge_density"] = s1["rgb_edge_density"] * 0.3  # fewer edges

preds1 = classify(s1)
summarize(preds1, "SCENARIO 1: RGB goes blind (YOLO fails)")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 2: IR goes blind (thermal crossover kills IR YOLO)
# ═══════════════════════════════════════════════════════════════
s2 = tb.copy()

# IR YOLO produces no detections
s2["ir_detected"] = 0
s2["ir_n_dets"] = 0
s2["ir_max_conf"] = 0.0
s2["ir_mean_conf"] = 0.0

# Best-det target features zero out
for col in feature_names:
    if col.startswith("ir_best_"):
        s2[col] = 0.0

# Agreement flags update
s2["both_detect"] = 0
s2["ir_only_detect"] = 0
s2["rgb_only_detect"] = s2["rgb_detected"]
s2["neither_detect"] = (s2["rgb_detected"] == 0).astype(int)

# Scene features: simulate thermal crossover (low contrast IR)
s2["ir_img_dynamic_range"] = s2["ir_img_dynamic_range"] * 0.2
s2["ir_img_std"] = s2["ir_img_std"] * 0.3

preds2 = classify(s2)
summarize(preds2, "SCENARIO 2: IR goes blind (YOLO fails)")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 3: Subtle RGB degradation (YOLO still detects,
#             but scene features indicate poor quality)
# ═══════════════════════════════════════════════════════════════
s3 = tb.copy()

# RGB YOLO still detects! But scene is degraded
# detection signals UNCHANGED
s3["rgb_blurriness"] = s3["rgb_blurriness"] * 0.05  # extreme blur
s3["rgb_img_std"] = s3["rgb_img_std"] * 0.2  # very low contrast
s3["rgb_edge_density"] = s3["rgb_edge_density"] * 0.1  # almost no edges
s3["rgb_img_entropy"] = s3["rgb_img_entropy"] * 0.5  # low complexity
s3["rgb_img_dynamic_range"] = s3["rgb_img_dynamic_range"] * 0.2  # compressed range
s3["rgb_best_local_contrast"] = 0.0  # no target-bg contrast
s3["rgb_best_target_bg_delta"] = 0.0

preds3 = classify(s3)
summarize(preds3, "SCENARIO 3: Subtle RGB degradation (YOLO still fires, scene features degraded)")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 4: Subtle IR degradation (same but for IR side)
# ═══════════════════════════════════════════════════════════════
s4 = tb.copy()

s4["ir_blurriness"] = s4["ir_blurriness"] * 0.05
s4["ir_img_std"] = s4["ir_img_std"] * 0.2
s4["ir_edge_density"] = s4["ir_edge_density"] * 0.1
s4["ir_img_entropy"] = s4["ir_img_entropy"] * 0.5
s4["ir_img_dynamic_range"] = s4["ir_img_dynamic_range"] * 0.2
s4["ir_best_local_contrast"] = 0.0
s4["ir_best_target_bg_delta"] = 0.0

preds4 = classify(s4)
summarize(preds4, "SCENARIO 4: Subtle IR degradation (YOLO still fires, scene features degraded)")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 5: Both degraded subtly (neither modality good)
# ═══════════════════════════════════════════════════════════════
s5 = tb.copy()

for prefix in ["rgb", "ir"]:
    s5[f"{prefix}_blurriness"] = s5[f"{prefix}_blurriness"] * 0.05
    s5[f"{prefix}_img_std"] = s5[f"{prefix}_img_std"] * 0.2
    s5[f"{prefix}_edge_density"] = s5[f"{prefix}_edge_density"] * 0.1
    s5[f"{prefix}_img_entropy"] = s5[f"{prefix}_img_entropy"] * 0.5
    s5[f"{prefix}_best_local_contrast"] = 0.0
    s5[f"{prefix}_best_target_bg_delta"] = 0.0

preds5 = classify(s5)
summarize(preds5, "SCENARIO 5: Both subtly degraded (scene features trashed, detections intact)")


# ═══════════════════════════════════════════════════════════════
# SCENARIO 6: RGB confidence halved (but still detects)
# ═══════════════════════════════════════════════════════════════
s6 = tb.copy()
s6["rgb_max_conf"] = s6["rgb_max_conf"] * 0.5
s6["rgb_mean_conf"] = s6["rgb_mean_conf"] * 0.5

preds6 = classify(s6)
summarize(preds6, "SCENARIO 6: RGB confidence halved (still detects)")


# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 70}")
print("SUMMARY: What fraction of trust_both frames switched?")
print(f"{'=' * 70}")
print(f"  {'Scenario':<55s} {'Switched':>8s}")
print(f"  {'-'*55} {'-'*8}")

scenarios = [
    ("Baseline (no changes)", baseline_preds),
    ("1. RGB blind (YOLO fails)", preds1),
    ("2. IR blind (YOLO fails)", preds2),
    ("3. Subtle RGB degradation (YOLO still fires)", preds3),
    ("4. Subtle IR degradation (YOLO still fires)", preds4),
    ("5. Both subtly degraded (detections intact)", preds5),
    ("6. RGB confidence halved", preds6),
]

for name, preds in scenarios:
    switched = (preds != 3).sum()
    total = len(preds)
    pct = switched / total * 100
    print(f"  {name:<55s} {pct:>6.1f}%")

print(f"\n  Interpretation:")
print(f"  - Scenarios 1-2 test: 'does the classifier react to YOLO failure?'")
print(f"  - Scenarios 3-5 test: 'do scene features alone change the decision?'")
print(f"  - Scenario 6 tests: 'does confidence magnitude matter?'")
print(f"  - If 3-5 show ~0% change, scene features are cosmetic.")
print(f"  - If 3-5 show >0% change, scene features provide real signal.")
