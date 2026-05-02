"""
eval_on_vtuav.py — Evaluate trained classifiers on VTUAV (all-negative dataset).

Loads:
    runs/phase1/classifier_baseline.joblib       (Phase 1 baseline)
    runs/phase2/classifier_ir_suppressed.joblib  (Phase 2 sup-trained)
    runs/vtuav_frame_dataset.csv

Since every VTUAV frame is a negative (no drones in the dataset), only false
positives matter. We compute:
    - Overall FP rate
    - Per-sequence FP rate
    - Per-category FP rate (bus/car/etc.)
    - FP rate at multiple operating thresholds (model's own, plus a few others)

We also compare raw YOLO over-firing vs what survives the classifier, so we
can see how much "smart fusion" reduces the FP rate.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def evaluate_model(model_bundle, df):
    """
    Score df with the given classifier at its saved threshold.
    Returns a dict with overall and per-group FP counts.

    df must have all feature columns + sequence, category.
    """
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    threshold = model_bundle["threshold"]

    X = df[feature_cols].values.astype(np.float32)
    y_prob = model.predict_proba(X)[:, 1]
    preds = (y_prob >= threshold).astype(int)

    total = len(df)
    fp = int(preds.sum())
    fp_rate = fp / total if total > 0 else 0.0

    # Per sequence
    df_eval = df.copy()
    df_eval["pred"] = preds
    df_eval["prob"] = y_prob
    per_seq = df_eval.groupby("sequence")["pred"].agg(["size", "sum"]).rename(
        columns={"size": "rows", "sum": "fp"})
    per_seq["fp_rate"] = per_seq["fp"] / per_seq["rows"]

    # Per category
    per_cat = df_eval.groupby("category")["pred"].agg(["size", "sum"]).rename(
        columns={"size": "rows", "sum": "fp"})
    per_cat["fp_rate"] = per_cat["fp"] / per_cat["rows"]

    # FP rate at different operating thresholds (sanity curve)
    thresh_curve = []
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        fp_t = int((y_prob >= t).sum())
        thresh_curve.append({"threshold": t, "fp": fp_t,
                             "fp_rate": fp_t / total if total > 0 else 0.0})

    return {
        "threshold_used": float(threshold),
        "total_frames": total,
        "fp_total": fp,
        "fp_rate": fp_rate,
        "per_sequence": per_seq.to_dict("index"),
        "per_category": per_cat.to_dict("index"),
        "threshold_curve": thresh_curve,
        "preds": preds,
        "probs": y_prob,
    }


def print_result(name, result):
    print(f"\n  [{name}]")
    print(f"    Threshold: {result['threshold_used']:.3f}")
    print(f"    Total frames: {result['total_frames']}")
    print(f"    False positives: {result['fp_total']}")
    print(f"    FP rate: {result['fp_rate']:.4f}  "
          f"({result['fp_total']} / {result['total_frames']})")

    print(f"\n    Per category:")
    print(f"      {'category':<12s} {'rows':>6s} {'fp':>6s} {'fp_rate':>9s}")
    for cat, stats in sorted(result["per_category"].items(),
                             key=lambda x: -x[1]["fp_rate"]):
        print(f"      {cat:<12s} {stats['rows']:>6d} {stats['fp']:>6d} "
              f"{stats['fp_rate']:>9.4f}")

    print(f"\n    Per sequence:")
    print(f"      {'sequence':<12s} {'rows':>6s} {'fp':>6s} {'fp_rate':>9s}")
    for seq, stats in sorted(result["per_sequence"].items(),
                             key=lambda x: -x[1]["fp_rate"]):
        print(f"      {seq:<12s} {stats['rows']:>6d} {stats['fp']:>6d} "
              f"{stats['fp_rate']:>9.4f}")

    print(f"\n    FP rate by threshold:")
    print(f"      {'threshold':>10s} {'fp':>6s} {'fp_rate':>9s}")
    for entry in result["threshold_curve"]:
        print(f"      {entry['threshold']:>10.2f} {entry['fp']:>6d} "
              f"{entry['fp_rate']:>9.4f}")


def main():
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    classifier_dir = script_dir.parent
    parser.add_argument("--csv",
                        default=str(classifier_dir / "runs" / "vtuav_frame_dataset.csv"))
    parser.add_argument("--baseline",
                        default=str(classifier_dir / "runs" / "phase1" / "classifier_baseline.joblib"))
    parser.add_argument("--sup-trained",
                        default=str(classifier_dir / "runs" / "phase2" / "classifier_ir_suppressed.joblib"))
    parser.add_argument("--out-dir",
                        default=str(classifier_dir / "runs" / "phase3"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VTUAV evaluation (Phase 3) — hard negatives, FP rate test")
    print("=" * 70)

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run build_vtuav_frame_csv.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Sequences: {df['sequence'].nunique()}")
    print(f"  Categories: {sorted(df['category'].unique())}")

    # Raw YOLO over-fire baseline (no classifier, just "did either model fire?")
    n = len(df)
    raw_rgb_only = int(((df["max_conf_rgb"] > 0) & (df["max_conf_ir"] == 0)).sum())
    raw_ir_only = int(((df["max_conf_ir"] > 0) & (df["max_conf_rgb"] == 0)).sum())
    raw_both = int(df["both_detected"].sum())
    raw_either = int(((df["max_conf_rgb"] > 0) | (df["max_conf_ir"] > 0)).sum())

    print(f"\n  Raw YOLO over-firing (no classifier):")
    print(f"    RGB only fired:     {raw_rgb_only:>5d} ({raw_rgb_only/n:.4f})")
    print(f"    IR only fired:      {raw_ir_only:>5d} ({raw_ir_only/n:.4f})")
    print(f"    Both fired:         {raw_both:>5d} ({raw_both/n:.4f})")
    print(f"    At least one fired: {raw_either:>5d} ({raw_either/n:.4f})")
    print(f"    Neither fired:      {n - raw_either:>5d} ({(n - raw_either)/n:.4f})")
    print("  (These are frames where raw YOLO made *any* detection. The classifier")
    print("   then decides whether to promote them to a 'drone' call.)")

    results = {}

    for name, path in [("baseline", args.baseline),
                       ("sup-trained", args.sup_trained)]:
        p = Path(path)
        if not p.exists():
            print(f"\n  [{name}] model not found at {p} — skipping")
            continue
        print(f"\n{'=' * 70}")
        print(f"Evaluating {name} ({p.name})")
        print("=" * 70)
        bundle = joblib.load(p)
        res = evaluate_model(bundle, df)
        print_result(name, res)
        # Strip numpy arrays before saving
        res_serial = {k: v for k, v in res.items() if k not in ("preds", "probs")}
        results[name] = res_serial

    # Comparison table
    if len(results) >= 2:
        print()
        print("=" * 70)
        print("COMPARISON")
        print("=" * 70)
        names = list(results.keys())
        print(f"  {'metric':<30s}", end="")
        for n in names:
            print(f" {n:>15s}", end="")
        print()
        print("  " + "-" * (30 + 16 * len(names)))

        print(f"  {'overall FP rate':<30s}", end="")
        for n in names:
            print(f" {results[n]['fp_rate']:>15.4f}", end="")
        print()

        print(f"  {'overall FP count':<30s}", end="")
        for n in names:
            print(f" {results[n]['fp_total']:>15d}", end="")
        print()

        # Per-category comparison
        cats = sorted({cat for r in results.values() for cat in r["per_category"].keys()})
        for cat in cats:
            print(f"  {'  ' + cat + ' FP rate':<30s}", end="")
            for n in names:
                rate = results[n]["per_category"].get(cat, {}).get("fp_rate", 0.0)
                print(f" {rate:>15.4f}", end="")
            print()

    # Save
    out_path = out_dir / "vtuav_eval.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()
