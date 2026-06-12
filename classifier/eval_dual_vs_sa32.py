"""eval_dual_vs_sa32.py — Compare the new dual models vs original sa32 on all datasets."""
import json, sys
from pathlib import Path

import joblib
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
import re

sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parent.parent
CSV_FULL56 = REPO / "models/routers/optimal_v1/fusion_dataset_full56.csv"
CSV_32FEAT = REPO / "models/routers/retrained_v2_32feat/fusion_dataset.csv"

# Models
SA32_MODEL_PATH = REPO / "models/routers/scene_aware_v3more_32feat/model.joblib"
SA32_METRICS_PATH = REPO / "models/routers/scene_aware_v3more_32feat/metrics.json"

PAIRED_MODEL_PATH = REPO / "models/routers/split_v3/classifier_paired.joblib"
GRAYSCALE_MODEL_PATH = REPO / "models/routers/split_v3/classifier_grayscale.joblib"

PAIRED_SOURCES = ["antiuav", "svanstrom"]
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)

# sa32 expects 32 specific features, but we know it's SA32_FEATURES
SA32_FEATURES = [
    "rgb_max_conf", "rgb_mean_conf", "ir_max_conf", "ir_mean_conf",
    "rgb_img_mean", "rgb_img_std", "rgb_img_dynamic_range", "rgb_img_entropy", "rgb_sky_ground_ratio",
    "rgb_edge_density", "rgb_blurriness",
    "ir_img_mean", "ir_img_std", "ir_img_dynamic_range", "ir_img_entropy", "ir_sky_ground_ratio",
    "ir_edge_density", "ir_blurriness",
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio", "rgb_best_pos_x", "rgb_best_pos_y", "rgb_best_dist_to_center",
    "rgb_best_local_contrast", "rgb_best_target_bg_delta",
    "ir_best_log_bbox_area", "ir_best_aspect_ratio", "ir_best_pos_x", "ir_best_pos_y", "ir_best_dist_to_center",
    "ir_best_local_contrast", "ir_best_target_bg_delta",
]

def seq_id(stem, src):
    m = SEQ_RE.match(str(stem))
    base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"

def compute_derived(df):
    if "area_diff" not in df.columns:
        df["area_diff"] = (df["rgb_best_log_bbox_area"] - df["ir_best_log_bbox_area"]).abs()
    if "xmodal_centroid_dist" not in df.columns:
        df["xmodal_centroid_dist"] = 0.0
    return df

def main():
    print("Loading models...")
    sa32_data = joblib.load(SA32_MODEL_PATH)
    sa32_model = sa32_data if isinstance(sa32_data, type(joblib.load(PAIRED_MODEL_PATH)["model"])) else sa32_data.get("model", sa32_data)

    paired_data = joblib.load(PAIRED_MODEL_PATH)
    paired_model = paired_data["model"]
    paired_feats = paired_data["features"]

    gray_data = joblib.load(GRAYSCALE_MODEL_PATH)
    gray_model = gray_data["model"]
    gray_feats = gray_data["features"]

    print("Loading datasets (Test Sets Only)...")
    df_65k = pd.read_csv(CSV_FULL56)
    df_144k = pd.read_csv(CSV_32FEAT)
    df_144k = compute_derived(df_144k)
    df_65k = compute_derived(df_65k)

    # 1. Evaluate Paired datasets (from 65k)
    df_paired = df_65k[df_65k["source"].isin(PAIRED_SOURCES)].copy()
    df_paired["_seq"] = df_paired.apply(lambda r: seq_id(r["stem"], r["source"]), axis=1)
    
    y_p = df_paired["trust_label"].values
    groups_p = df_paired["_seq"].values
    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(df_paired, y_p, groups=groups_p))
    df_paired_test = df_paired.iloc[te_idx].copy()
    
    # 2. Evaluate Grayscale datasets (65k + 144k confusers)
    df_65_gray = df_65k[~df_65k["source"].isin(PAIRED_SOURCES)]
    df_144_gray = df_144k[~df_144k["source"].isin(PAIRED_SOURCES)]
    
    # Keep only necessary cols for grayscale eval to save memory
    all_needed = list(set(SA32_FEATURES + paired_feats + gray_feats + ["trust_label", "stem", "source"]))
    df_144_gray = df_144_gray[[c for c in all_needed if c in df_144_gray.columns]]
    for c in all_needed:
        if c not in df_144_gray.columns: df_144_gray[c] = 0.0
    
    df_65_gray = df_65_gray[[c for c in all_needed if c in df_65_gray.columns]]
    for c in all_needed:
        if c not in df_65_gray.columns: df_65_gray[c] = 0.0
        
    df_gray = pd.concat([df_65_gray, df_144_gray], ignore_index=True).drop_duplicates(subset=["stem", "source"])
    df_gray["_seq"] = df_gray.apply(lambda r: seq_id(r["stem"], r["source"]), axis=1)
    
    y_g = df_gray["trust_label"].values
    groups_g = df_gray["_seq"].values
    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(df_gray, y_g, groups=groups_g))
    df_gray_test = df_gray.iloc[te_idx].copy()

    # Predictions
    print("Running predictions...")
    
    # sa32 on everything
    sa32_preds_paired = sa32_model.predict(df_paired_test[SA32_FEATURES].values)
    # Filter missing sa32 features on grayscale just in case
    for f in SA32_FEATURES:
        if f not in df_gray_test.columns: df_gray_test[f] = 0.0
    sa32_preds_gray = sa32_model.predict(df_gray_test[SA32_FEATURES].values)

    # new models on their specific datasets
    new_preds_paired = paired_model.predict(df_paired_test[paired_feats].values)
    new_preds_gray = gray_model.predict(df_gray_test[gray_feats].values)

    # Calculate metrics per dataset
    results = []

    # Paired
    for ds in np.unique(df_paired_test["source"].values):
        mask = df_paired_test["source"].values == ds
        y_true = df_paired_test["trust_label"].values[mask]
        
        y_sa32 = sa32_preds_paired[mask]
        sa32_f1 = f1_score(y_true, y_sa32, average="macro", zero_division=0)
        
        y_new = new_preds_paired[mask]
        new_f1 = f1_score(y_true, y_new, average="macro", zero_division=0)
        
        results.append({
            "mode": "Paired", "dataset": ds, "n": len(y_true),
            "sa32_f1": sa32_f1, "new_f1": new_f1, "delta": new_f1 - sa32_f1
        })

    # Grayscale
    for ds in np.unique(df_gray_test["source"].values):
        mask = df_gray_test["source"].values == ds
        y_true = df_gray_test["trust_label"].values[mask]
        if len(y_true) < 10: continue # skip tiny splits
        
        y_sa32 = sa32_preds_gray[mask]
        sa32_f1 = f1_score(y_true, y_sa32, average="macro", zero_division=0)
        
        y_new = new_preds_gray[mask]
        new_f1 = f1_score(y_true, y_new, average="macro", zero_division=0)
        
        results.append({
            "mode": "Grayscale", "dataset": ds, "n": len(y_true),
            "sa32_f1": sa32_f1, "new_f1": new_f1, "delta": new_f1 - sa32_f1
        })

    # Print Table
    print("\n" + "="*80)
    print("Dual Classifier vs SA32 (F1-Macro on Test Sets)")
    print("="*80)
    print(f"{'Mode':<12s} | {'Dataset':<40s} | {'N':>6s} | {'sa32 (32f)':>10s} | {'New (22/24f)':>12s} | {'Delta':>8s}")
    print("-" * 100)
    
    last_mode = None
    for r in sorted(results, key=lambda x: (x['mode'], -x['n'])):
        if last_mode and r['mode'] != last_mode:
            print("-" * 100)
        print(f"{r['mode']:<12s} | {r['dataset']:<40s} | {r['n']:>6d} | "
              f"{r['sa32_f1']:>10.4f} | {r['new_f1']:>12.4f} | {r['delta']:>+8.4f}")
        last_mode = r['mode']
        
    print("="*80)

if __name__ == "__main__":
    main()
