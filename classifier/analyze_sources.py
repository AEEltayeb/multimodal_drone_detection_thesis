"""Quick analysis of ir_only vs rgb_only detection confidence distributions."""
import csv
from collections import defaultdict

rows = list(csv.DictReader(open("runs/fusion_dataset.csv", "r", encoding="utf-8")))

# Confidence bands
bands = [
    ("0.001-0.05", 0.001, 0.05),
    ("0.05-0.10",  0.05,  0.10),
    ("0.10-0.25",  0.10,  0.25),
    ("0.25-0.50",  0.25,  0.50),
    ("0.50-0.75",  0.50,  0.75),
    ("0.75-1.00",  0.75,  1.00),
]

for source in ["ir_only", "rgb_only", "both"]:
    src_rows = [r for r in rows if r["source"] == source]
    if not src_rows:
        continue

    print(f"\n{'='*65}")
    print(f"  {source.upper()} — {len(src_rows)} total detections")
    print(f"{'='*65}")
    print(f"  {'Conf band':<15} {'Total':>7} {'TP':>7} {'FP':>7} {'TP%':>7}")
    print(f"  {'-'*50}")

    for band_name, lo, hi in bands:
        if source == "ir_only":
            band_rows = [r for r in src_rows if lo <= float(r["conf_ir"]) < hi]
        elif source == "rgb_only":
            band_rows = [r for r in src_rows if lo <= float(r["conf_rgb"]) < hi]
        else:
            band_rows = [r for r in src_rows if lo <= float(r["conf_max"]) < hi]

        total = len(band_rows)
        tp = sum(1 for r in band_rows if int(r["label"]) == 1)
        fp = total - tp
        tp_pct = f"{tp/total*100:.1f}%" if total > 0 else "N/A"
        print(f"  {band_name:<15} {total:>7} {tp:>7} {fp:>7} {tp_pct:>7}")

    # Summary
    total_tp = sum(1 for r in src_rows if int(r["label"]) == 1)
    print(f"  {'-'*50}")
    print(f"  {'TOTAL':<15} {len(src_rows):>7} {total_tp:>7} {len(src_rows)-total_tp:>7} "
          f"{total_tp/len(src_rows)*100:.1f}%")

    # High-conf only (what you'd get at a normal threshold)
    if source == "ir_only":
        hi_conf = [r for r in src_rows if float(r["conf_ir"]) >= 0.25]
    elif source == "rgb_only":
        hi_conf = [r for r in src_rows if float(r["conf_rgb"]) >= 0.25]
    else:
        hi_conf = [r for r in src_rows if float(r["conf_max"]) >= 0.25]
    hi_tp = sum(1 for r in hi_conf if int(r["label"]) == 1)
    print(f"\n  At conf >= 0.25: {len(hi_conf)} dets, {hi_tp} TP, "
          f"{len(hi_conf)-hi_tp} FP")
