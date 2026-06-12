"""retrain_v5_targeted.py — targeted re-mine fix for the mlp_v5 recall drop.

Diagnosis: mlp_v5 vetoes OOD real drones (small, atypical signature) that are SEPARABLE
from confusers (AUROC 0.876) -> the MLP CAN learn to keep them; the earlier untargeted
remine failed. So: up-weight, in the EXISTING V5 training pool, exactly the drones the
current MLP under-scores (P(drone)<0.5) — targeted at the failure mode — and retrain.
No new data, no test leakage (trains on training_data.npz, evals on held-out offline caches).

Then eval mlp_v5 vs mlp_v5_retargeted on rgb_dataset_test (recall recovery), svanstrom
(precision must HOLD — unlike fail-open which cratered it), rgb_confuser (FP cost).

  py eval/retrain_v5_targeted.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
from distill_v5_swap_selcom import train_v5_mlp
from metrics import score_detections, compute_prf
from eval_v4_vs_patch import MLPv4Verifier

NPZ = REPO / "eval/results/_v5_selcom_pure_1x8/training_data.npz"
OLD = REPO / "models/verifiers/rgb_v5/mlp_v5.pt"
NEW = REPO / "eval/results/_v5_retargeted/classifiers/mlp_v5.pt"
CACHE = REPO / "eval/results/_offline_pipeline/cache"
K = 4.0  # up-weight factor for under-scored drones

# --- build targeted weights ---
d = np.load(NPZ); X, y, w = d["X"].astype(np.float32), d["y"], d["w"].copy()
old = MLPv4Verifier(OLD, device="cpu")
probs = old.predict_drone_probs(X)
hard = (y == 1) & (probs < 0.5)          # real drones the current MLP under-scores
w[hard] *= K
print(f"targeted: up-weighted {int(hard.sum())} under-scored drones x{K} "
      f"(of {int((y==1).sum())} drones); weights mean {w.mean():.2f}")

# --- retrain (same V5 procedure + save schema) ---
train_v5_mlp(X, y, w, NEW)

# --- eval old vs new on held-out offline caches ---
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0., x2-x1)*max(0., y2-y1); ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return i/(ua+ub-i) if ua+ub-i > 0 else 0.

new = MLPv4Verifier(NEW, device="cpu")
def evalsurf(name, rule, has_dr, thr=0.25):
    dd = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
    res = {}
    for tag, vf in (("mlp_v5", old), ("mlp_v5_retargeted", new)):
        tp = fp = fn = 0
        for fr in dd["frames"]:
            n = len(fr["feats"]);
            if n:
                p = vf.predict_drone_probs(fr["feats"])
            keep = (p >= thr) if n else np.zeros(0, bool)
            kept = [(tuple(fr["boxes"][i]), float(fr["confs"][i])) for i in range(n) if keep[i]]
            if has_dr:
                gt = [tuple(g) for g in fr["gt_boxes"]]
                t, f, fnn = score_detections(kept, gt, rule=rule, iou_thr=0.5, iop_thr=0.5)
                tp += t; fp += f; fn += fnn
            else:
                fp += len(kept)
        res[tag] = compute_prf(tp, fp, fn) if has_dr else {"fp": fp}
    print(f"\n== {name} ({rule}, drones={has_dr}) ==")
    for tag, m in res.items():
        if has_dr:
            print(f"   {tag:<20} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
        else:
            print(f"   {tag:<20} FP={m['fp']}")

evalsurf("rgb_dataset_test", "iou", True)
evalsurf("svanstrom", "iop", True)
evalsurf("rgb_confuser", "iou", False)
print("\nDONE retarget+eval")
