"""Compare classifier_features modes (all / skip_expensive / detections_only).

Loads the cached 32-feature fusion dataset, simulates each mode by
mean-filling the features that the live GUI would NOT compute under
that mode, runs the production classifier, and reports per-class
precision/recall vs the held-out test split.

Strided: takes every Nth row (default N=10) for a cheap run.

No inference is needed — features are already cached. Mean-fills mirror
gui/fusion/features.py:_TRAIN_MEANS.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "classifier" / "runs" / "reliability" / "fusion" / "fusion_dataset_v3more.csv"
MODEL = ROOT / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib"

# Mirrors gui/fusion/features.py:_TRAIN_MEANS
TRAIN_MEANS = {
    "rgb": {"img_mean": 96.964, "img_std": 25.559, "img_dynamic_range": 71.883,
            "img_entropy": 4.943, "sky_ground_ratio": 1.288,
            "edge_density": 0.007581, "blurriness": 231.512},
    "ir":  {"img_mean": 85.290, "img_std": 50.707, "img_dynamic_range": 188.590,
            "img_entropy": 6.880, "sky_ground_ratio": 0.653,
            "edge_density": 0.025516, "blurriness": 882.696},
}

# What each mode SKIPS (= mean-fills) per modality.
# 'all' computes everything. 'skip_expensive' fills 4. 'detections_only' fills 7.
SKIP_EXPENSIVE = ["img_dynamic_range", "img_entropy", "edge_density", "blurriness"]
ALL_GLOBALS = ["img_mean", "img_std", "img_dynamic_range", "img_entropy",
               "sky_ground_ratio", "edge_density", "blurriness"]

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}


def apply_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    if mode == "all":
        return out
    if mode == "skip_expensive":
        skip = SKIP_EXPENSIVE
    elif mode == "detections_only":
        skip = ALL_GLOBALS
    else:
        raise ValueError(mode)
    for mod in ("rgb", "ir"):
        for f in skip:
            out[f"{mod}_{f}"] = TRAIN_MEANS[mod][f]
    return out


def per_class_metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    return {LABEL_NAMES[i]: {"P": float(p[i]), "R": float(r[i]),
                              "F1": float(f[i]), "n": int(s[i])}
            for i in range(4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10,
                    help="Take every Nth row (default 10)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {CSV.name} ...")
    df = pd.read_csv(CSV)
    df = df.iloc[::args.stride].reset_index(drop=True)
    print(f"  rows after stride={args.stride}: {len(df):,}")
    print(f"  by source: {df['source_dataset'].value_counts().to_dict()}")
    print(f"  by trust_label: {df['trust_label'].value_counts().sort_index().to_dict()}")

    print(f"\nLoading {MODEL.name} ...")
    bundle = joblib.load(MODEL)
    if isinstance(bundle, dict):
        model = bundle.get("model", bundle)
        feature_cols = bundle.get("features") or bundle.get("feature_cols")
    else:
        model = bundle
        feature_cols = None
    if feature_cols is None:
        meta = MODEL.parent / "metrics.json"
        feature_cols = json.loads(meta.read_text())["features"]
    print(f"  {len(feature_cols)} features")

    y_true = df["trust_label"].values

    results = {}
    for mode in ("all", "skip_expensive", "detections_only"):
        sim = apply_mode(df, mode)
        X = sim[feature_cols].values.astype(np.float32)
        y_pred = model.predict(X)
        acc = float((y_pred == y_true).mean())
        per = per_class_metrics(y_true, y_pred)
        results[mode] = {"accuracy": acc, "per_class": per}

        # Per-source breakdown
        per_src = {}
        for src in df["source_dataset"].unique():
            m = df["source_dataset"].values == src
            per_src[src] = {
                "n": int(m.sum()),
                "acc": float((y_pred[m] == y_true[m]).mean()),
            }
        results[mode]["per_source"] = per_src

    # Print tables
    print("\n" + "=" * 70)
    print(f"{'class':<14} {'mode':<18} {'P':>6} {'R':>6} {'F1':>6} {'n':>7}")
    print("-" * 70)
    for cls in [0, 1, 2, 3]:
        name = LABEL_NAMES[cls]
        for mode in ("all", "skip_expensive", "detections_only"):
            d = results[mode]["per_class"][name]
            print(f"{name:<14} {mode:<18} {d['P']:.3f} {d['R']:.3f} "
                  f"{d['F1']:.3f} {d['n']:>7}")
        print()

    print("=" * 70)
    print("Overall accuracy")
    for mode in ("all", "skip_expensive", "detections_only"):
        print(f"  {mode:<18} {results[mode]['accuracy']:.4f}")

    print("\nPer-source accuracy")
    print(f"{'source':<16} {'all':>8} {'skip_exp':>10} {'det_only':>10}")
    sources = list(results["all"]["per_source"].keys())
    for src in sources:
        n = results["all"]["per_source"][src]["n"]
        a = results["all"]["per_source"][src]["acc"]
        s = results["skip_expensive"]["per_source"][src]["acc"]
        d = results["detections_only"]["per_source"][src]["acc"]
        print(f"{src:<16} {a:.4f}   {s:.4f}     {d:.4f}   (n={n})")

    print("\nP/R drop vs 'all' (negative = worse)")
    print(f"{'class':<14} {'mode':<18} {'dP':>8} {'dR':>8} {'dF1':>8}")
    for cls in [0, 1, 2, 3]:
        name = LABEL_NAMES[cls]
        base = results["all"]["per_class"][name]
        for mode in ("skip_expensive", "detections_only"):
            d = results[mode]["per_class"][name]
            print(f"{name:<14} {mode:<18} "
                  f"{d['P'] - base['P']:+.3f}   "
                  f"{d['R'] - base['R']:+.3f}   "
                  f"{d['F1'] - base['F1']:+.3f}")
        print()


if __name__ == "__main__":
    main()
