"""
run_phase1.py — Orchestrate Phase 1: curate dataset + run baseline + ablations.

Steps:
  1. Run curate_dataset.py to build runs/curated_frame_dataset.csv
  2. Train baseline on curated data
  3. Ablation A: drop Svanstrom DRONE positives (tests IR leakage)
  4. Ablation B: Anti-UAV only (isolates classifier behavior on dataset A)
  5. Ablation C: Svanstrom only (isolates classifier behavior on dataset B)
  6. Print comparison summary of feature importances + overall F1

Usage:
    python run_phase1.py
"""

import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CLASSIFIER_DIR = SCRIPT_DIR.parent
PHASE1_OUT = CLASSIFIER_DIR / "runs" / "phase1"


def run(cmd, step_name):
    print()
    print("=" * 70)
    print(f"STEP: {step_name}")
    print("=" * 70)
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"\n  ERROR: '{step_name}' failed (exit {result.returncode})")
        sys.exit(result.returncode)


def load_metrics(tag):
    path = PHASE1_OUT / f"metrics_{tag}.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def summarize():
    print()
    print("=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)

    tags = ["baseline", "no_svan_drones", "anti_uav_only", "svanstrom_only"]
    rows = []
    for tag in tags:
        m = load_metrics(tag)
        if m is None:
            continue
        t = m["test_overall"]
        rows.append({
            "tag": tag,
            "P": t["precision"],
            "R": t["recall"],
            "F1": t["f1"],
            "AUC-PR": t["aucpr"],
            "threshold": t["threshold"],
            "n_test": m["n_test"],
        })

    if not rows:
        print("  No metrics files found.")
        return

    print(f"\n  Overall test metrics:")
    print(f"  {'tag':<20s} {'n_test':>7s} {'P':>8s} {'R':>8s} {'F1':>8s} {'AUC-PR':>8s} {'thresh':>8s}")
    for r in rows:
        print(f"  {r['tag']:<20s} {r['n_test']:>7d} {r['P']:>8.4f} "
              f"{r['R']:>8.4f} {r['F1']:>8.4f} {r['AUC-PR']:>8.4f} "
              f"{r['threshold']:>8.3f}")

    # Feature importance comparison
    print(f"\n  Feature importance comparison (top features across runs):")
    all_feats = set()
    for tag in tags:
        m = load_metrics(tag)
        if m:
            all_feats.update(m["feature_importance"].keys())

    # Sort by baseline importance if available, else by name
    baseline = load_metrics("baseline")
    if baseline:
        feat_order = sorted(
            all_feats,
            key=lambda f: -baseline["feature_importance"].get(f, 0.0),
        )
    else:
        feat_order = sorted(all_feats)

    header_tags = [t for t in tags if load_metrics(t) is not None]
    header = f"  {'feature':<22s}"
    for t in header_tags:
        header += f" {t[:14]:>15s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for feat in feat_order:
        line = f"  {feat:<22s}"
        for t in header_tags:
            m = load_metrics(t)
            imp = m["feature_importance"].get(feat, 0.0) if m else 0.0
            line += f" {imp:>15.4f}"
        print(line)

    # Per-stratum for baseline
    if baseline:
        print(f"\n  Baseline per-stratum test metrics:")
        print(f"  {'group':<35s} {'rows':>7s} {'P':>8s} {'R':>8s} {'F1':>8s} {'FP/tot':>8s}")
        print("  " + "-" * 75)
        for s in baseline["test_strata"]:
            print(f"  {s['group']:<35s} {s['rows']:>7d} "
                  f"{s['precision']:>8.4f} {s['recall']:>8.4f} "
                  f"{s['f1']:>8.4f} {s['fp_rate']:>8.4f}")


def main():
    # Step 1: Curate
    run([sys.executable, "curate_dataset.py"], "Curate dataset")

    # Step 2: Baseline
    run([sys.executable, "train_curated.py", "--tag", "baseline"],
        "Train baseline")

    # Step 3: Ablation — drop Svanstrom DRONE positives
    run([sys.executable, "train_curated.py",
         "--tag", "no_svan_drones", "--drop-svanstrom-drones"],
        "Ablation: drop Svanstrom DRONE positives")

    # Step 4: Ablation — Anti-UAV only
    run([sys.executable, "train_curated.py",
         "--tag", "anti_uav_only", "--anti-uav-only"],
        "Ablation: Anti-UAV only")

    # Step 5: Ablation — Svanstrom only
    run([sys.executable, "train_curated.py",
         "--tag", "svanstrom_only", "--svanstrom-only"],
        "Ablation: Svanstrom only")

    # Step 6: Summary
    summarize()


if __name__ == "__main__":
    main()
