import json, numpy as np
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "runs" / "reliability" / "fusion" / "comparison" / "fusion_comparison.json"   # resident classifier/ (was ES_Drone_Detection); legacy run data not shipped
results = json.load(open(p))

print(f"  {'#':<3s} {'Approach':<28s} {'Det_P':>7s} {'Det_R':>7s} {'Det_F1':>7s} {'TP':>8s} {'FP':>6s} {'Missed':>7s}")
print(f"  {'-'*72}")

for r in sorted(results, key=lambda x: -x['metrics']['f1_macro']):
    cm = np.array(r['metrics']['confusion_matrix'])
    tp = int(cm[1:, 1:].sum())
    fp = int(cm[0, 1:].sum())
    fn = int(cm[1:, 0].sum())
    det_p = tp / (tp + fp) if (tp + fp) > 0 else 0
    det_r = tp / (tp + fn) if (tp + fn) > 0 else 0
    det_f1 = 2 * det_p * det_r / (det_p + det_r + 1e-9)
    print(f"  {r['key'][:2]:<3s} {r['name']:<28s} {det_p:>7.4f} {det_r:>7.4f} {det_f1:>7.4f} {tp:>8,} {fp:>6,} {fn:>7,}")

print()
print("  --- Individual Modality Baselines (all 152K frames, different denominator) ---")
print(f"  {'--':<3s} {'RGB YOLO alone':<28s} {'0.8762':>7s} {'0.8614':>7s} {'0.8687':>7s} {'114,725':>8s} {'16,217':>6s} {'18,454':>7s}")
print(f"  {'--':<3s} {'IR YOLO alone':<28s} {'0.9839':>7s} {'0.9449':>7s} {'0.9640':>7s} {'125,843':>8s} {' 2,059':>6s} {' 7,336':>7s}")
print(f"  {'--':<3s} {'OR gate (all 152K)':<28s} {'0.9293':>7s} {'0.9893':>7s} {'0.9584':>7s} {'131,756':>8s} {'10,029':>6s} {' 1,423':>7s}")
