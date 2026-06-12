"""test_size_aware_threshold.py — interim RGB fix: lenient MLP threshold for SMALL boxes.

Diagnosis: vetoed real drones are SMALLER (lower log_area). So keep if
  P(drone) >= base_thr  OR  (log_area < cutoff AND P >= small_thr)
i.e. relax the veto only for small detections (targeted), instead of fail-open's blanket
release. Risk: small CONFUSERS also get the lenient threshold -> precision cost. Test it.

log_area = feature index 513 (517-D: p3[0:256], p5[256:512], meta[512:517]=conf,log_area,...).

  py eval/test_size_aware_threshold.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np, sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
from metrics import score_detections, compute_prf
from eval_v4_vs_patch import MLPv4Verifier
CACHE = REPO / "eval/results/_offline_pipeline/cache"
mlp = MLPv4Verifier(REPO / "models/verifiers/rgb_v5/mlp_v5.pt", device="cpu")
LOG_AREA, BASE, SMALL_THR = 513, 0.25, 0.05


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0., x2-x1)*max(0., y2-y1); ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return i/(ua+ub-i) if ua+ub-i > 0 else 0.


def run(name, rule, has_dr, cutoff_pct):
    d = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
    # global cutoff on this surface's detection log_area
    all_la = np.concatenate([fr["feats"][:, LOG_AREA] for fr in d["frames"] if len(fr["feats"])]) if any(len(fr["feats"]) for fr in d["frames"]) else np.array([0.])
    cutoff = np.percentile(all_la, cutoff_pct)
    agg = {v: {"tp": 0, "fp": 0, "fn": 0} for v in ("mlp_v5", "size_aware")}
    for fr in d["frames"]:
        n = len(fr["feats"])
        if n:
            p = mlp.predict_drone_probs(fr["feats"]); la = fr["feats"][:, LOG_AREA]
        keeps = {
            "mlp_v5": (p >= BASE) if n else np.zeros(0, bool),
            "size_aware": ((p >= BASE) | ((la < cutoff) & (p >= SMALL_THR))) if n else np.zeros(0, bool),
        }
        for v, keep in keeps.items():
            kept = [(tuple(fr["boxes"][i]), float(fr["confs"][i])) for i in range(n) if keep[i]]
            if has_dr:
                t, f, fn = score_detections(kept, [tuple(g) for g in fr["gt_boxes"]], rule=rule, iou_thr=0.5, iop_thr=0.5)
                agg[v]["tp"] += t; agg[v]["fp"] += f; agg[v]["fn"] += fn
            else:
                agg[v]["fp"] += len(kept)
    print(f"== {name} ({rule}, drones={has_dr}, small-cutoff=p{cutoff_pct}={cutoff:.3f}) ==")
    for v, m in agg.items():
        if has_dr:
            prf = compute_prf(m["tp"], m["fp"], m["fn"]); print(f"   {v:<12} P={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f}")
        else:
            print(f"   {v:<12} FP={m['fp']}")
    print()


if __name__ == "__main__":
    for pct in (50, 70):
        print(f"########## small = log_area below p{pct} ##########")
        run("rgb_dataset_test", "iou", True, pct)
        run("svanstrom", "iop", True, pct)
        run("rgb_confuser", "iou", False, pct)
