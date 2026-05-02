"""Compute trust-scoped metrics for NEW pipeline from per_det.jsonl."""
import sys, csv
sys.path.insert(0, ".")
from classifier.generate_all_plots import compute_all_metrics, CONFIG_NAMES
from pathlib import Path

for ds in ["antiuav", "svanstrom"]:
    p = Path(f"classifier/runs/eval_six_configs_v3more_32feat/{ds}/per_det.jsonl")
    results = compute_all_metrics(p)
    if not results:
        print(f"{ds}: NO DATA")
        continue
    for rule in ["iou", "iop"]:
        print(f"\n=== {ds} {rule.upper()} ===")
        hdr = f"{'config':<28s} {'TP':>8s} {'FP':>8s} {'FN':>8s} {'TN':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}"
        print(hdr)
        print("-" * len(hdr))
        for c in CONFIG_NAMES:
            r = results[c][rule]
            line = "{:<28s} {:>8,} {:>8,} {:>8,} {:>8,} {:>8.4f} {:>8.4f} {:>8.4f}".format(
                c, r["TP"], r["FP"], r["FN"], r["TN"],
                r["precision"], r["recall"], r["f1"])
            print(line)

        # Also write CSV
        out = Path(f"classifier/runs/eval_six_configs_v3more_32feat/{ds}/metrics_scoped_{rule}.csv")
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["config","TP","FP","FN","TN","Precision","Recall","F1"])
            w.writeheader()
            for c in CONFIG_NAMES:
                r = results[c][rule]
                w.writerow({
                    "config": c, "TP": r["TP"], "FP": r["FP"],
                    "FN": r["FN"], "TN": r["TN"],
                    "Precision": round(r["precision"], 4),
                    "Recall": round(r["recall"], 4),
                    "F1": round(r["f1"], 4),
                })
        print(f"  -> saved {out}")
