"""ablation_split_v2.py — Train dual classifiers using the full gray_aug dataset.

- Paired:    152k paired rows from v3more_gray_aug
- Grayscale: 152k grayscale rows from v3more_gray_aug + confusers from 144k

Usage:
  python classifier/ablation_split_v2.py --mode paired
  python classifier/ablation_split_v2.py --mode grayscale
  python classifier/ablation_split_v2.py --mode both
"""
import argparse, json, re, sys, time
from pathlib import Path

import joblib
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parent.parent
CSV_GRAY_AUG = REPO / "classifier/runs/reliability/fusion/fusion_dataset_v3more_gray_aug.csv"
CSV_144K = REPO / "models/routers/retrained_v2_32feat/fusion_dataset.csv"
CSV_FULL56 = REPO / "models/routers/optimal_v1/fusion_dataset_full56.csv"
CSV_RGB_TEST = REPO / "docs/analysis/full_pipeline_ablations/cache/fusion_dataset_rgb_test.csv"
OUTPUT_DIR = REPO / "models/routers/split_v3"

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)

# ── Feature sets ─────────────────────────────────────────────────
EXPENSIVE_SCENE = [
    "rgb_img_dynamic_range", "rgb_img_entropy", "rgb_sky_ground_ratio",
    "rgb_edge_density", "rgb_blurriness",
    "ir_img_dynamic_range", "ir_img_entropy", "ir_sky_ground_ratio",
    "ir_edge_density", "ir_blurriness",
]

SA32_FEATURES = [
    "rgb_max_conf", "rgb_mean_conf", "ir_max_conf", "ir_mean_conf",
    "rgb_img_mean", "rgb_img_std",
    "rgb_img_dynamic_range", "rgb_img_entropy", "rgb_sky_ground_ratio",
    "rgb_edge_density", "rgb_blurriness",
    "ir_img_mean", "ir_img_std",
    "ir_img_dynamic_range", "ir_img_entropy", "ir_sky_ground_ratio",
    "ir_edge_density", "ir_blurriness",
    "rgb_best_log_bbox_area", "rgb_best_aspect_ratio",
    "rgb_best_pos_x", "rgb_best_pos_y", "rgb_best_dist_to_center",
    "rgb_best_local_contrast", "rgb_best_target_bg_delta",
    "ir_best_log_bbox_area", "ir_best_aspect_ratio",
    "ir_best_pos_x", "ir_best_pos_y", "ir_best_dist_to_center",
    "ir_best_local_contrast", "ir_best_target_bg_delta",
]

SA32_LITE = [f for f in SA32_FEATURES if f not in EXPENSIVE_SCENE]

SA32_LITE_PLUS = SA32_LITE + ["area_diff", "xmodal_centroid_dist"]


def seq_id(stem, src, idx=0):
    m = SEQ_RE.match(str(stem))
    base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


def compute_derived(df):
    if "area_diff" not in df.columns:
        df["area_diff"] = (df["rgb_best_log_bbox_area"] - df["ir_best_log_bbox_area"]).abs()
    if "xmodal_centroid_dist" not in df.columns:
        df["xmodal_centroid_dist"] = 0.0
    return df


def train_and_eval(X_tr, y_tr, X_te, y_te, src_te, feat_names, tag):
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, n_jobs=1,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, verbose=False)
    train_time = time.time() - t0

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    acc = float(accuracy_score(y_te, y_pred))
    f1m = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    f1w = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))

    per_class_auc = {}
    for c in range(4):
        if (y_te == c).sum() > 0 and (y_te != c).sum() > 0:
            binary = (y_te == c).astype(int)
            per_class_auc[LABEL_NAMES[c]] = float(roc_auc_score(binary, y_prob[:, c]))

    per_dataset = {}
    for ds in np.unique(src_te):
        mask = src_te == ds
        ys, yp = y_te[mask], y_pred[mask]
        per_dataset[str(ds)] = {
            "n": int(len(ys)),
            "acc": float(accuracy_score(ys, yp)),
            "f1_macro": float(f1_score(ys, yp, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(ys, yp, average="weighted", zero_division=0)),
        }

    importance = dict(zip(feat_names, [float(v) for v in model.feature_importances_]))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    return {
        "tag": tag,
        "n_features": len(feat_names),
        "features": feat_names,
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "per_class_auc": per_class_auc,
        "per_dataset": per_dataset,
        "feature_importance": importance,
        "train_time_s": round(train_time, 1),
    }, model


def run_mode(mode, df, variants, label="source"):
    print(f"\n{'=' * 70}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'=' * 70}")
    print(f"  Total rows: {len(df):,}")

    # Get stem column
    stem_col = "base_stem" if "base_stem" in df.columns else "stem"
    src_col = "source_dataset" if "source_dataset" in df.columns else "source"

    if src_col in df.columns:
        print(f"  Sources:\n{df[src_col].value_counts().to_string()}")

    df["_seq"] = [seq_id(row[stem_col], row.get(src_col, "unknown"), i) for i, row in df.iterrows()]

    y = df["trust_label"].values
    groups = df["_seq"].values
    tr_idx, te_idx = next(GroupShuffleSplit(1, test_size=0.25, random_state=42)
                          .split(df, y, groups=groups))

    # Fix Data Leakage: Force certain clips to train, hold out seagull_attack
    stem_col = "base_stem" if "base_stem" in df.columns else "stem"
    train_stems = ["video_drone_two_birds_drone", "video_drone_flock_of_seagulls_attack_drone_beach", "video_drone_drone_and_bird_sky_and_trees_short"]
    test_stems = ["video_drone_drone_seagull_attack"]
    
    tr_mask = np.zeros(len(df), dtype=bool)
    tr_mask[tr_idx] = True
    te_mask = np.zeros(len(df), dtype=bool)
    te_mask[te_idx] = True
    
    stems = df[stem_col].values
    for i, s in enumerate(stems):
        if s in train_stems:
            tr_mask[i] = True
            te_mask[i] = False
        elif s in test_stems:
            te_mask[i] = True
            tr_mask[i] = False
            
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    y_tr, y_te = y[tr_idx], y[te_idx]
    src_te = df.iloc[te_idx][src_col].values if src_col in df.columns else np.array(["all"] * len(te_idx))

    print(f"\n  train: {len(tr_idx):,}  test: {len(te_idx):,}")
    print(f"  class dist (train): { {LABEL_NAMES[k]: int(v) for k, v in zip(*np.unique(y_tr, return_counts=True))} }")

    results = {}
    best_model = None
    best_tag = None

    for tag, feat_cols in variants.items():
        print(f"\n{'─' * 60}")
        print(f"  Training: {tag} ({len(feat_cols)} features)")
        print(f"{'─' * 60}")

        missing = [f for f in feat_cols if f not in df.columns]
        if missing:
            print(f"  SKIP — missing: {missing}")
            continue

        X_tr = df.iloc[tr_idx][feat_cols].values
        X_te = df.iloc[te_idx][feat_cols].values

        metrics, model = train_and_eval(X_tr, y_tr, X_te, y_te, src_te, feat_cols, tag)
        results[tag] = metrics

        print(f"  acc={metrics['accuracy']:.4f}  F1m={metrics['f1_macro']:.4f}  "
              f"F1w={metrics['f1_weighted']:.4f}  ({metrics['train_time_s']:.1f}s)")

        for ds, dm in metrics["per_dataset"].items():
            print(f"    {ds}: acc={dm['acc']:.4f} F1m={dm['f1_macro']:.4f} (n={dm['n']})")

        top5 = list(metrics["feature_importance"].items())[:5]
        print(f"  Top 5: {', '.join(f'{f}({v:.3f})' for f, v in top5)}")

        if best_model is None or metrics["f1_macro"] > results.get(best_tag, {}).get("f1_macro", 0):
            best_model = model
            best_tag = tag

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"ABLATION SUMMARY — {mode.upper()}")
    print(f"{'=' * 70}")
    print(f"  {'variant':<16s} {'feats':>5s} {'acc':>7s} {'F1m':>7s} {'F1w':>7s} {'time':>6s}")
    print(f"  {'-'*16} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for tag, m in results.items():
        print(f"  {tag:<16s} {m['n_features']:>5d} {m['accuracy']:>7.4f} "
              f"{m['f1_macro']:>7.4f} {m['f1_weighted']:>7.4f} {m['train_time_s']:>5.1f}s")

    # Save
    outdir = OUTPUT_DIR / mode
    outdir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(outdir / "ablation_results.json", "w"), indent=2)

    if best_model and best_tag:
        best_feats = variants[best_tag]
        joblib.dump({"model": best_model, "features": best_feats},
                    outdir / f"model_{best_tag}.joblib")
        joblib.dump({"model": best_model, "features": best_feats},
                    OUTPUT_DIR / f"classifier_{mode}.joblib")
        print(f"\n  Best: {best_tag} → classifier_{mode}.joblib")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["paired", "grayscale", "both"])
    args = ap.parse_args()

    print("Loading datasets ...")
    df_aug = pd.read_csv(CSV_GRAY_AUG)
    df_aug = compute_derived(df_aug)
    print(f"  gray_aug: {len(df_aug):,} rows")

    df_144k = pd.read_csv(CSV_144K)
    df_144k = compute_derived(df_144k)
    print(f"  144k: {len(df_144k):,} rows")

    # Also load 65k for video tests
    df_65k = pd.read_csv(CSV_FULL56)
    df_65k = compute_derived(df_65k)
    print(f"  65k: {len(df_65k):,} rows")

    if CSV_RGB_TEST.exists():
        df_rgb = pd.read_csv(CSV_RGB_TEST)
        df_rgb = compute_derived(df_rgb)
        print(f"  rgb_test: {len(df_rgb):,} rows")
    else:
        df_rgb = pd.DataFrame()

    PAIRED_SOURCES = ["antiuav", "svanstrom"]
    PAIRED_SOURCES_AUG = ["antiuav_test", "antiuav_val", "svanstrom"]

    modes = [args.mode] if args.mode != "both" else ["paired", "grayscale"]

    for mode in modes:
        if mode == "paired":
            # Use the 152k paired rows from gray_aug
            df = df_aug[df_aug["modality_mode"] == "paired"].copy()
            # Rename columns for consistency
            if "source_dataset" in df.columns and "source" not in df.columns:
                df["source"] = df["source_dataset"]
            if "base_stem" in df.columns and "stem" not in df.columns:
                df["stem"] = df["base_stem"]

            variants = {
                "sa32_feats": SA32_FEATURES,
                "sa32_lite":  SA32_LITE,
                "sa32_lite+": SA32_LITE_PLUS,
            }
            run_mode(mode, df, variants)

        elif mode == "grayscale":
            # 152k grayscale rows from gray_aug
            df_gray_aug = df_aug[df_aug["modality_mode"] == "grayscale"].copy()
            if "source_dataset" in df_gray_aug.columns and "source" not in df_gray_aug.columns:
                df_gray_aug["source"] = df_gray_aug["source_dataset"]
            if "base_stem" in df_gray_aug.columns and "stem" not in df_gray_aug.columns:
                df_gray_aug["stem"] = df_gray_aug["base_stem"]

            # Add confusers from 144k
            confusers_144 = df_144k[~df_144k["source"].isin(PAIRED_SOURCES)].copy()

            # Add video tests from 65k
            videos_65k = df_65k[~df_65k["source"].isin(PAIRED_SOURCES)].copy()

            # Combine all columns we need
            all_feats = list(set(SA32_FEATURES + SA32_LITE_PLUS))
            meta_cols = ["trust_label", "stem", "source"]
            keep_cols = [c for c in all_feats + meta_cols if c in df_gray_aug.columns]

            parts = [df_gray_aug[keep_cols]]

            for extra, name in [(confusers_144, "confusers_144k"), (videos_65k, "videos_65k"), (df_rgb, "rgb_test")]:
                if extra.empty: continue
                sub = extra[[c for c in keep_cols if c in extra.columns]].copy()
                for c in keep_cols:
                    if c not in sub.columns:
                        sub[c] = 0.0
                parts.append(sub)
                print(f"  Adding {name}: {len(sub):,} rows")

            df = pd.concat(parts, ignore_index=True)
            df = df.drop_duplicates(subset=["stem", "source"])

            variants = {
                "sa32_feats": SA32_FEATURES,
                "sa32_lite":  SA32_LITE,
                "sa32_lite+": SA32_LITE_PLUS,
            }
            run_mode(mode, df, variants)

    print(f"\n{'=' * 70}")
    print("ALL DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
