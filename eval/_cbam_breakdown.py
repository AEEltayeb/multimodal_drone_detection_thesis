"""Per-category FP breakdown for ir_mixed_cbam valid split."""
import csv
from pathlib import Path
from collections import Counter, defaultdict

cbam = Path("G:/drone/roboflow_eval/ir_mixed_cbam/valid")
labels_dir = cbam / "labels"
det_csv = Path("eval/results/roboflow_ood/ir_mixed_cbam/ir_model/valid/ir_model_frame_detections.csv")

# Map each stem to its GT categories
class_names = {0: "Bird", 1: "Drone", 2: "Airplane"}
stem_cats = {}
for f in labels_dir.glob("*.txt"):
    cats = set()
    for line in f.read_text().strip().split("\n"):
        if line.strip():
            cats.add(int(line.split()[0]))
    stem_cats[f.stem] = cats

# Read frame detections and tally FP by category
cat_stats = defaultdict(lambda: {"frames": 0, "raw_fp": 0, "filt_fp": 0, "tp": 0, "fn": 0})
with open(det_csv) as f:
    for row in csv.DictReader(f):
        stem = row["stem"]
        cats = stem_cats.get(stem, set())
        # Assign frame to primary category
        for c in sorted(cats):
            cat = class_names.get(c, f"cls{c}")
            cat_stats[cat]["frames"] += 1
            cat_stats[cat]["raw_fp"] += int(row["fp"])
            cat_stats[cat]["filt_fp"] += int(row["fp_f"])
            cat_stats[cat]["tp"] += int(row["tp"])
            cat_stats[cat]["fn"] += int(row["fn"])
            break  # assign to first (primary) category only
        if not cats:
            cat_stats["No GT"]["frames"] += 1
            cat_stats["No GT"]["raw_fp"] += int(row["fp"])

print(f"{'Category':<12s} {'Frames':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'fFP':>5s} {'FPPI':>7s} {'fFPPI':>7s}")
print("-" * 60)
for cat in ["Drone", "Bird", "Airplane", "No GT"]:
    v = cat_stats.get(cat)
    if not v:
        continue
    fppi = v["raw_fp"]/v["frames"]*100 if v["frames"] else 0
    ffppi = v["filt_fp"]/v["frames"]*100 if v["frames"] else 0
    print(f"{cat:<12s} {v['frames']:>6d} {v['tp']:>5d} {v['raw_fp']:>5d} {v['fn']:>5d} {v['filt_fp']:>5d} {fppi:>6.1f}% {ffppi:>6.1f}%")
