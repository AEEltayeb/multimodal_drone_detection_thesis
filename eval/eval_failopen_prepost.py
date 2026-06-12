"""eval_failopen_prepost.py — clean PRE vs POST fail-open metrics for the thesis.

Three variants scored with the SAME score_detections used everywhere:
  bare            : keep all detections
  mlp_v5          : keep if P(drone) >= 0.25         (the recall drop)
  mlp_v5+failopen : keep if P>=0.25 OR ood_from_confuser > tau   (OOD-abstain)

tau is calibrated to 5% leak on the confuser surface's MLP-vetoed confusers (the knob).
Drone surfaces show recall recovery; the confuser surface shows the FP cost (the drawback).

  py eval/eval_failopen_prepost.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
from sklearn.neighbors import NearestNeighbors
from metrics import score_detections, compute_prf
from eval_v4_vs_patch import MLPv4Verifier
C = REPO / "eval/results/_offline_pipeline/cache"
mlp = MLPv4Verifier(REPO / "models/verifiers/rgb_v5/mlp_v5.pt", device="cpu")
THR, LEAK = 0.25, 0.05

# --- fit OOD-from-confuser on rgb_confuser dets; calibrate tau at 5% leak ---
cf = pickle.load(open(C / "rgb_confuser.pkl", "rb"))
Cf = np.array([f for fr in cf["frames"] for f in fr["feats"]]) if cf["frames"] else np.zeros((0, 517))
mu, sd = Cf.mean(0), Cf.std(0) + 1e-6
zc = lambda x: (x - mu) / sd
nn = NearestNeighbors(n_neighbors=5).fit(zc(Cf))
ood = lambda X: nn.kneighbors(zc(X))[0].mean(1) if len(X) else np.zeros(0)
# vetoed confusers -> tau at 95th pct (5% leak)
vet_conf = np.array([f for fr in cf["frames"] for f, p in zip(fr["feats"], mlp.predict_drone_probs(fr["feats"]) if len(fr["feats"]) else []) if p < THR])
TAU = float(np.quantile(ood(vet_conf), 1 - LEAK))
print(f"tau (5% leak) = {TAU:.2f}   |  fit on {len(Cf)} confuser dets, {len(vet_conf)} vetoed\n")


def score_surface(name, rule, has_dr):
    d = pickle.load(open(C / f"{name}.pkl", "rb"))
    agg = {v: {"tp": 0, "fp": 0, "fn": 0} for v in ("bare", "mlp_v5", "mlp_v5+failopen")}
    for fr in d["frames"]:
        n = len(fr["feats"]); gt = [tuple(g) for g in fr["gt_boxes"]]
        boxes = [tuple(b) for b in fr["boxes"]]; confs = [float(c) for c in fr["confs"]]
        if n:
            p = mlp.predict_drone_probs(fr["feats"]); o = ood(fr["feats"])
        keeps = {
            "bare": np.ones(n, bool) if n else np.zeros(0, bool),
            "mlp_v5": (p >= THR) if n else np.zeros(0, bool),
            "mlp_v5+failopen": ((p >= THR) | (o > TAU)) if n else np.zeros(0, bool),
        }
        for v, keep in keeps.items():
            kept = [(boxes[i], confs[i]) for i in range(n) if keep[i]]
            if has_dr:
                t, f, fn = score_detections(kept, gt, rule=rule, iou_thr=0.5, iop_thr=0.5)
                agg[v]["tp"] += t; agg[v]["fp"] += f; agg[v]["fn"] += fn
            else:
                agg[v]["fp"] += len(kept)
    print(f"== {name} ({rule}, drones={has_dr}, n={len(d['frames'])}) ==")
    for v, m in agg.items():
        if has_dr:
            prf = compute_prf(m["tp"], m["fp"], m["fn"])
            print(f"   {v:<18} P={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f}  (TP {m['tp']} FP {m['fp']} FN {m['fn']})")
        else:
            print(f"   {v:<18} FP={m['fp']}  halluc/img={m['fp']/max(len(d['frames']),1):.4f}")
    print()


if __name__ == "__main__":
    score_surface("rgb_dataset_test", "iou", True)
    score_surface("svanstrom", "iop", True)
    score_surface("rgb_confuser", "iou", False)
