"""
eval_aerial_negatives.py — Per-class FPR harness for the fusion classifier.

Splits evaluation rows by their aerial category (airplane / helicopter / bird /
drone / other), derived from the stem prefix (`IR_{CATEGORY}_...`). Reports,
at a threshold sweep and at the classifier's saved operational threshold:

  - drone_recall, drone_precision (drone rows with label=1 are true positives)
  - airplane_FPR, helicopter_FPR, bird_FPR, other_FPR (any accept on a
    label=0 aerial-negative row is a false positive)

Optionally splits results by `ir_is_real_thermal` so the grayscale-replicate
regime can be measured separately once Component 3 lands.

Usage (baseline measurement, P1):
    python classifier/eval_aerial_negatives.py \
        --csv classifier/runs/svanstrom_frame_dataset.csv \
        --model classifier/runs/classifier.joblib \
        --output classifier/runs/aerial_eval/v1.0_metrics.json \
        --label-tag v1.0
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


CATEGORY_PATTERN = re.compile(r"^IR_([A-Z]+)_", re.IGNORECASE)
KNOWN_CATEGORIES = {"airplane", "helicopter", "bird", "drone"}


def derive_category(stem: str) -> str:
    """Extract aerial category from a stem like `IR_AIRPLANE_001_f000000`."""
    m = CATEGORY_PATTERN.match(stem)
    if not m:
        return "other"
    cat = m.group(1).lower()
    return cat if cat in KNOWN_CATEGORIES else "other"


def align_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame with exactly `feature_cols` in order.
    Missing columns are filled with 0.0. `time_of_day_*` dummies are expanded
    from a raw `time_of_day` string column if present.
    """
    out = df.copy()

    if "time_of_day" in out.columns:
        tod_dummies = pd.get_dummies(out["time_of_day"], prefix="time_of_day",
                                     dtype=float)
        for col in tod_dummies.columns:
            out[col] = tod_dummies[col]

    for col in feature_cols:
        if col not in out.columns:
            out[col] = 0.0

    return out[feature_cols].astype(float)


def eval_at_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                      categories: np.ndarray, threshold: float) -> dict:
    """
    Compute per-category metrics at `threshold`.

    Drone rows with `y_true=1` are the positive population (for recall);
    all other label=0 rows (drone with label=0, airplane, helicopter, bird,
    other) contribute to their category's FPR.
    """
    preds = (y_prob >= threshold).astype(int)

    is_drone = categories == "drone"
    drone_pos = is_drone & (y_true == 1)
    drone_neg = is_drone & (y_true == 0)

    drone_tp = int((preds == 1)[drone_pos].sum())
    drone_fn = int((preds == 0)[drone_pos].sum())
    drone_pos_total = int(drone_pos.sum())
    drone_recall = drone_tp / drone_pos_total if drone_pos_total else 0.0

    result = {
        "threshold": round(float(threshold), 4),
        "drone_recall": round(drone_recall, 4),
        "drone_tp": drone_tp,
        "drone_fn": drone_fn,
        "drone_pos_total": drone_pos_total,
    }

    total_fp = 0
    total_neg = 0
    for cat in ["drone", "airplane", "helicopter", "bird", "other"]:
        if cat == "drone":
            mask = drone_neg
            metric_prefix = "drone_neg"
        else:
            mask = categories == cat
            metric_prefix = cat

        n = int(mask.sum())
        if n == 0:
            result[f"{metric_prefix}_n"] = 0
            result[f"{metric_prefix}_FP"] = 0
            result[f"{metric_prefix}_FPR"] = None
            continue

        fp = int((preds == 1)[mask].sum())
        fpr = fp / n
        result[f"{metric_prefix}_n"] = n
        result[f"{metric_prefix}_FP"] = fp
        result[f"{metric_prefix}_FPR"] = round(fpr, 4)

        total_fp += fp
        total_neg += n

    drone_precision = (drone_tp / (drone_tp + total_fp)
                       if (drone_tp + total_fp) > 0 else 0.0)
    result["drone_precision"] = round(drone_precision, 4)
    result["overall_FP"] = total_fp
    result["overall_neg"] = total_neg
    result["overall_FPR"] = round(total_fp / total_neg, 4) if total_neg else None

    return result


def print_summary_table(sweep: list[dict], frozen_idx: int):
    """Print a compact per-threshold table to stdout."""
    headers = ["thr", "drone_R", "drone_P", "plane_FPR", "heli_FPR",
               "bird_FPR", "other_FPR"]
    fmt = "{:<6} {:<8} {:<8} {:<10} {:<10} {:<10} {:<10}"
    print(fmt.format(*headers))
    print("-" * 72)
    for i, row in enumerate(sweep):
        marker = " *" if i == frozen_idx else ""
        print(fmt.format(
            f"{row['threshold']:.2f}",
            f"{row['drone_recall']:.3f}",
            f"{row['drone_precision']:.3f}",
            f"{row['airplane_FPR']:.3f}" if row["airplane_FPR"] is not None else "n/a",
            f"{row['helicopter_FPR']:.3f}" if row["helicopter_FPR"] is not None else "n/a",
            f"{row['bird_FPR']:.3f}" if row["bird_FPR"] is not None else "n/a",
            f"{row['other_FPR']:.3f}" if row["other_FPR"] is not None else "n/a",
        ) + marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True,
                        help="Fusion-row CSV (e.g. svanstrom_frame_dataset.csv)")
    parser.add_argument("--model", required=True,
                        help="Classifier joblib (e.g. classifier/runs/classifier.joblib)")
    parser.add_argument("--output", required=True,
                        help="Output JSON path (e.g. runs/aerial_eval/v1.0_metrics.json)")
    parser.add_argument("--label-tag", default="unnamed",
                        help="Free-text label stored in the output JSON")
    parser.add_argument("--thresholds", default="sweep",
                        help='"sweep" for 0.05..0.95 step 0.05, or a comma-separated list')
    parser.add_argument("--ir-mode-column", default=None,
                        help="Optional column name with the IR input mode "
                             "(e.g. 'ir_is_real_thermal'). When present, reports "
                             "per-mode metrics.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    model_path = Path(args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows")

    print(f"Loading model: {model_path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    frozen_threshold = float(bundle["threshold"])
    print(f"  model_type={bundle['model_type']} features={len(feature_cols)} "
          f"frozen_threshold={frozen_threshold:.4f}")

    df["category"] = df["stem"].apply(derive_category)
    cat_counts = df["category"].value_counts().to_dict()
    print(f"  category distribution: {cat_counts}")

    missing = [c for c in feature_cols if c not in df.columns
               and not c.startswith("time_of_day_")]
    if missing:
        print(f"  [WARN] {len(missing)} feature columns missing from CSV, "
              f"filling with 0.0: {missing}")

    X = align_features(df, feature_cols).values
    y = df["label"].values.astype(int)
    categories = df["category"].values

    print("Scoring rows...")
    y_prob = model.predict_proba(X)[:, 1]

    if args.thresholds == "sweep":
        thresholds = np.arange(0.05, 1.00, 0.05).tolist()
    else:
        thresholds = [float(t) for t in args.thresholds.split(",")]
    if frozen_threshold not in thresholds:
        thresholds.append(frozen_threshold)
    thresholds = sorted(set(round(t, 4) for t in thresholds))

    sweep = [eval_at_threshold(y, y_prob, categories, t) for t in thresholds]
    frozen_idx = thresholds.index(round(frozen_threshold, 4))

    print("\n=== Threshold sweep ===")
    print_summary_table(sweep, frozen_idx)
    print(f"\n * = frozen operational threshold ({frozen_threshold:.4f})")

    report = {
        "label_tag": args.label_tag,
        "csv": str(csv_path),
        "model": str(model_path),
        "frozen_threshold": frozen_threshold,
        "model_type": bundle["model_type"],
        "feature_cols": feature_cols,
        "category_counts": cat_counts,
        "n_rows": len(df),
        "sweep": sweep,
        "frozen_metrics": sweep[frozen_idx],
    }

    if args.ir_mode_column and args.ir_mode_column in df.columns:
        print(f"\n=== Per-IR-mode breakdown ({args.ir_mode_column}) ===")
        per_mode = {}
        for mode_val, mask in df.groupby(args.ir_mode_column).groups.items():
            mask_idx = df.index.isin(mask)
            sub = [eval_at_threshold(y[mask_idx], y_prob[mask_idx],
                                     categories[mask_idx], t)
                   for t in thresholds]
            per_mode[str(mode_val)] = {
                "n_rows": int(mask_idx.sum()),
                "frozen_metrics": sub[frozen_idx],
                "sweep": sub,
            }
            print(f"  {args.ir_mode_column}={mode_val}: "
                  f"{int(mask_idx.sum())} rows, "
                  f"frozen drone_R={sub[frozen_idx]['drone_recall']:.3f} "
                  f"plane_FPR={sub[frozen_idx]['airplane_FPR']} "
                  f"heli_FPR={sub[frozen_idx]['helicopter_FPR']} "
                  f"bird_FPR={sub[frozen_idx]['bird_FPR']}")
        report["per_ir_mode"] = per_mode

    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
