"""Generate plots for both OLD and NEW pipelines into ablation_old_vs_new/."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from generate_all_plots import (
    compute_all_metrics, plot_metrics_bars, plot_confusion, plot_pr_curves,
    CONFIG_NAMES
)

ABLATION = SCRIPT_DIR / "runs" / "ablation_old_vs_new"

for label, src_dir in [
    ("old", SCRIPT_DIR / "runs" / "eval_six_configs"),
    ("new", SCRIPT_DIR / "runs" / "eval_six_configs_v3more_32feat"),
]:
    for ds in ["antiuav", "svanstrom"]:
        per_det = src_dir / ds / "per_det.jsonl"
        out_dir = ABLATION / label / ds
        out_dir.mkdir(parents=True, exist_ok=True)

        if not per_det.exists():
            print(f"  SKIP {label}/{ds} - no per_det.jsonl")
            continue

        print(f"\n[{label}/{ds}] Computing metrics + plots...")
        results = compute_all_metrics(per_det)
        if not results:
            print(f"  No data")
            continue

        for rule in ("iou", "iop"):
            title = f"{label.upper()} {ds} [{rule.upper()}]"
            plot_metrics_bars(results, out_dir, title, rule)
            plot_confusion(results, out_dir, title, rule)
            print(f"  {rule}: bars + confusion saved")

        plot_pr_curves(per_det, out_dir, f"{label}_{ds}")
        print(f"  PR curves saved")

print("\nAll plots generated!")
