"""
validate_fn_models.py — Cross-dataset validation of rgb_fn and ir_fn models.

Loads the existing FN datasets (already have computed features + labels),
evaluates both models per-dataset, produces:
  - Per-dataset AUC, AP, precision/recall/F1 at stored threshold
  - Calibration analysis (reliability diagram)
  - Score distribution plots per dataset
  - Summary JSON

Usage:
    python validate_fn_models.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "failure_models"
OUT_DIR    = DATA_DIR / "validation"


# ── CALIBRATION ──────────────────────────────────────────────────

def calibration_curve(y_true, y_prob, n_bins=10):
    """Compute calibration curve (fraction of positives vs mean predicted)."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_true_fracs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if i == n_bins - 1:  # include right edge
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append(float(y_prob[mask].mean()))
        bin_true_fracs.append(float(y_true[mask].mean()))
        bin_counts.append(int(mask.sum()))

    return bin_centers, bin_true_fracs, bin_counts


# ── PER-DATASET EVALUATION ──────────────────────────────────────

def evaluate_model_on_dataset(model, feature_cols, threshold, df_subset, tag):
    """Evaluate FN model on one dataset subset. Returns metrics dict."""
    if len(df_subset) == 0:
        return None

    missing = [c for c in feature_cols if c not in df_subset.columns]
    if missing:
        return {"error": f"missing columns: {missing}"}

    X = df_subset[feature_cols].values.astype(np.float32)
    y_true = df_subset["label"].values

    n_pos = int(y_true.sum())
    n_neg = int((y_true == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return {
            "tag": tag,
            "n": len(df_subset),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "warning": "single-class — cannot compute AUC",
            "pos_rate": float(y_true.mean()),
        }

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, y_prob))
    ap = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Score statistics
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]

    # Calibration
    cal_centers, cal_fracs, cal_counts = calibration_curve(y_true, y_prob, n_bins=10)

    return {
        "tag": tag,
        "n": len(df_subset),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": round(float(y_true.mean()) * 100, 2),
        "auc": round(auc, 4),
        "ap": round(ap, 4),
        "brier": round(brier, 4),
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "pos_score_mean": round(float(pos_scores.mean()), 4),
        "pos_score_std": round(float(pos_scores.std()), 4),
        "neg_score_mean": round(float(neg_scores.mean()), 4),
        "neg_score_std": round(float(neg_scores.std()), 4),
        "calibration": {
            "centers": [round(c, 4) for c in cal_centers],
            "true_fracs": [round(f, 4) for f in cal_fracs],
            "counts": cal_counts,
        },
    }


# ── PLOTTING ─────────────────────────────────────────────────────

def plot_calibration(results, model_tag, out_path):
    """Plot calibration diagram for all datasets."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: calibration curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    for res in results:
        if "calibration" not in res:
            continue
        cal = res["calibration"]
        if len(cal["centers"]) < 2:
            continue
        ax.plot(cal["centers"], cal["true_fracs"],
                "o-", label=f"{res['tag']} (n={res['n']:,})", markersize=4)
    ax.set_xlabel("Mean predicted P(FN)")
    ax.set_ylabel("Fraction of actual FN")
    ax.set_title(f"{model_tag.upper()} Calibration")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # Right: AUC per dataset bar chart
    ax = axes[1]
    tags = [r["tag"] for r in results if "auc" in r]
    aucs = [r["auc"] for r in results if "auc" in r]
    n_pos_vals = [r["n_pos"] for r in results if "auc" in r]

    if tags:
        colors = ["#e74c3c" if n < 50 else "#2ecc71" if a > 0.85
                  else "#f39c12" for a, n in zip(aucs, n_pos_vals)]
        bars = ax.barh(range(len(tags)), aucs, color=colors)
        ax.set_yticks(range(len(tags)))
        ax.set_yticklabels(tags, fontsize=8)
        ax.set_xlabel("AUC")
        ax.set_title(f"{model_tag.upper()} AUC per Dataset")
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Random")
        ax.axvline(x=0.85, color="green", linestyle="--", alpha=0.3, label="Good")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)

        # Annotate with n_pos
        for i, (bar, n) in enumerate(zip(bars, n_pos_vals)):
            ax.text(bar.get_width() + 0.01, i, f"n+={n:,}",
                    va="center", fontsize=7, color="#555")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_score_distributions(results, df, model, feature_cols, model_tag, out_path):
    """Plot score distributions per dataset (FN vs TP)."""
    datasets = [r["tag"] for r in results if "auc" in r and r["n_pos"] >= 10]
    if not datasets:
        return

    n_ds = len(datasets)
    fig, axes = plt.subplots(min(n_ds, 6), 1,
                              figsize=(10, 2.5 * min(n_ds, 6)),
                              squeeze=False)

    for i, tag in enumerate(datasets[:6]):
        ax = axes[i, 0]
        subset = df[df["source_dataset"] == tag]
        X = subset[feature_cols].values.astype(np.float32)
        y = subset["label"].values
        scores = model.predict_proba(X)[:, 1]

        ax.hist(scores[y == 0], bins=50, alpha=0.6, density=True,
                label=f"TP (n={int((y==0).sum()):,})", color="#2ecc71")
        ax.hist(scores[y == 1], bins=50, alpha=0.6, density=True,
                label=f"FN (n={int((y==1).sum()):,})", color="#e74c3c")
        ax.axvline(x=results[i]["threshold"] if "threshold" in results[i] else 0.5,
                   color="black", linestyle="--", alpha=0.5, label="Threshold")
        ax.set_title(f"{tag}", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("P(FN) score")
        ax.set_ylabel("Density")

    plt.suptitle(f"{model_tag.upper()} Score Distributions", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── MAIN ─────────────────────────────────────────────────────────

def validate_one(model_tag, df, bundle):
    """Validate one FN model across all datasets in its CSV."""
    model = bundle["model"]
    feature_cols = bundle["features"]
    threshold = bundle["threshold"]

    print(f"\n{'=' * 70}")
    print(f"Validating {model_tag.upper()}")
    print(f"{'=' * 70}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Datasets: {df['source_dataset'].nunique()}")
    print()

    datasets = sorted(df["source_dataset"].unique())
    results = []

    # Per-dataset evaluation
    print(f"  {'dataset':<25s} {'n':>7s} {'n_pos':>6s} {'pos%':>6s} "
          f"{'AUC':>7s} {'AP':>7s} {'P':>7s} {'R':>7s} {'F1':>7s} {'Brier':>7s}")
    print(f"  {'-' * 88}")

    for tag in datasets:
        subset = df[df["source_dataset"] == tag]
        res = evaluate_model_on_dataset(model, feature_cols, threshold, subset, tag)
        results.append(res)

        if "auc" in res:
            print(f"  {tag:<25s} {res['n']:>7,} {res['n_pos']:>6,} "
                  f"{res['pos_rate']:>5.1f}% "
                  f"{res['auc']:>7.4f} {res['ap']:>7.4f} "
                  f"{res['precision']:>7.4f} {res['recall']:>7.4f} "
                  f"{res['f1']:>7.4f} {res['brier']:>7.4f}")
        elif "warning" in res:
            print(f"  {tag:<25s} {res['n']:>7,} {res['n_pos']:>6,} "
                  f"{res.get('pos_rate', 0) * 100:>5.1f}% "
                  f"  {'--- SINGLE CLASS ---':>40s}")

    # Overall evaluation (all data pooled)
    print(f"\n  {'OVERALL (pooled)':<25s}", end="")
    overall = evaluate_model_on_dataset(model, feature_cols, threshold, df, "OVERALL")
    if "auc" in overall:
        print(f" {overall['n']:>7,} {overall['n_pos']:>6,} "
              f"{overall['pos_rate']:>5.1f}% "
              f"{overall['auc']:>7.4f} {overall['ap']:>7.4f} "
              f"{overall['precision']:>7.4f} {overall['recall']:>7.4f} "
              f"{overall['f1']:>7.4f} {overall['brier']:>7.4f}")
    results.append(overall)

    # Risk assessment
    print(f"\n  Risk Assessment:")
    for res in results:
        if res.get("tag") == "OVERALL":
            continue
        if "auc" not in res:
            print(f"    [WARN] {res['tag']}: single class - cannot evaluate")
            continue
        if res["n_pos"] < 50:
            print(f"    [WARN] {res['tag']}: only {res['n_pos']} positives "
                  f"- AUC={res['auc']:.3f} is statistically unreliable")
        elif res["auc"] < 0.7:
            print(f"    [FAIL] {res['tag']}: AUC={res['auc']:.3f} - "
                  f"model fails on this dataset")
        elif res["auc"] < 0.85:
            print(f"    [WARN] {res['tag']}: AUC={res['auc']:.3f} - moderate performance")
        else:
            print(f"    [OK]   {res['tag']}: AUC={res['auc']:.3f} - good")

    # Plots
    plot_calibration(results[:-1], model_tag,
                     OUT_DIR / f"{model_tag}_calibration.png")
    plot_score_distributions(results[:-1], df, model, feature_cols, model_tag,
                             OUT_DIR / f"{model_tag}_score_distributions.png")

    return {"model": model_tag, "threshold": threshold,
            "per_dataset": results[:-1], "overall": overall}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FN Model Cross-Dataset Validation")
    print("=" * 70)

    all_results = {}

    for model_tag in ["rgb_fn", "ir_fn"]:
        model_path = DATA_DIR / f"{model_tag}_model.joblib"
        csv_path = DATA_DIR / f"{model_tag}_dataset.csv"

        if not model_path.exists():
            print(f"  [SKIP] {model_path} not found")
            continue
        if not csv_path.exists():
            print(f"  [SKIP] {csv_path} not found")
            continue

        print(f"\nLoading {model_tag} model...", end="", flush=True)
        bundle = joblib.load(model_path)
        print(f" done (features: {len(bundle['features'])})")

        print(f"Loading {csv_path.name}...", end="", flush=True)
        df = pd.read_csv(csv_path)
        print(f" {len(df):,} rows")

        results = validate_one(model_tag, df, bundle)
        all_results[model_tag] = results

    # Save summary JSON
    summary_path = OUT_DIR / "fn_validation_report.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {summary_path}")

    # Final comparison
    if len(all_results) == 2:
        print(f"\n{'=' * 70}")
        print("COMPARISON SUMMARY")
        print(f"{'=' * 70}")
        for model_tag, res in all_results.items():
            overall = res["overall"]
            if "auc" in overall:
                print(f"  {model_tag.upper():<10s} "
                      f"AUC={overall['auc']:.4f}  "
                      f"AP={overall['ap']:.4f}  "
                      f"F1={overall['f1']:.4f}  "
                      f"Brier={overall['brier']:.4f}")

            # Count risk levels
            good = sum(1 for r in res["per_dataset"]
                      if "auc" in r and r["auc"] >= 0.85 and r["n_pos"] >= 50)
            moderate = sum(1 for r in res["per_dataset"]
                          if "auc" in r and 0.7 <= r["auc"] < 0.85 and r["n_pos"] >= 50)
            poor = sum(1 for r in res["per_dataset"]
                      if "auc" in r and r["auc"] < 0.7 and r["n_pos"] >= 50)
            unreliable = sum(1 for r in res["per_dataset"]
                            if "auc" in r and r["n_pos"] < 50)
            print(f"    Datasets: {good} good, {moderate} moderate, "
                  f"{poor} poor, {unreliable} too-few-positives")

    print("\nDone.")


if __name__ == "__main__":
    main()
