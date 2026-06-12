"""diagnose_mlp_recall_drop.py — WHY does mlp_v5 veto real drones? (statistical, CPU, offline)

Uses the offline pipeline caches (517-D features per detection + GT). For each RGB drone
surface: find detections that MATCH a GT drone (= real drones), split into KEPT (mlp P>=thr)
vs FALSELY-VETOED (P<thr = recall loss), then run mri.stats (LDA / ANOVA-F / AUROC) on the
two groups. If they're separable and conf/log_area dominate, the MLP is killing a
characterizable sub-population (small/faint) -> fix = size/conf-aware threshold or targeted re-mine.

  py eval/diagnose_mlp_recall_drop.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
from mri.stats import anova_f, per_feature_auroc, lda_separability
from eval_v4_vs_patch import MLPv4Verifier

CACHE = REPO / "eval/results/_offline_pipeline/cache"
MLP_V5 = REPO / "models/verifiers/rgb_v5/mlp_v5.pt"
THR = 0.25
# 517-D layout: p3[0:256], p5[256:512], meta[512:517]=conf,log_area,aspect,cx,cy
META = {512: "conf", 513: "log_area", 514: "aspect", 515: "rel_cx", 516: "rel_cy"}
def fname(i): return META.get(i, f"p3_{i}" if i < 256 else f"p5_{i-256}")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0., x2-x1)*max(0., y2-y1); ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return i/(ua+ub-i) if ua+ub-i > 0 else 0.
def iop(d, g):
    x1, y1 = max(d[0], g[0]), max(d[1], g[1]); x2, y2 = min(d[2], g[2]), min(d[3], g[3])
    i = max(0., x2-x1)*max(0., y2-y1); da = (d[2]-d[0])*(d[3]-d[1])
    return i/da if da > 0 else 0.


def diagnose(name, rule):
    d = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
    mlp = MLPv4Verifier(MLP_V5, device="cpu")
    match = iop if rule == "iop" else iou
    feats, probs, kept = [], [], []
    for fr in d["frames"]:
        if len(fr["feats"]) == 0 or len(fr["gt_boxes"]) == 0:
            continue
        p = mlp.predict_drone_probs(fr["feats"])
        for i, box in enumerate(fr["boxes"]):
            if max((match(box, g) for g in fr["gt_boxes"]), default=0) >= 0.5:  # real drone
                feats.append(fr["feats"][i]); probs.append(float(p[i])); kept.append(p[i] >= THR)
    X = np.array(feats); y = np.array(kept, int); probs = np.array(probs)
    nk, nv = int(y.sum()), int((~y.astype(bool)).sum())
    print(f"\n=== {name} ({rule}) ===")
    print(f"  real drones matched: {len(y)}  | KEPT {nk}  FALSELY-VETOED {nv}  (recall loss {nv/max(len(y),1):.1%})")
    if nv < 10 or nk < 10:
        print("  too few in a group for stats"); return
    F = anova_f(X, y); auroc = per_feature_auroc(X, y); _, lda_acc, _ = lda_separability(X, y)
    top = np.argsort(F)[::-1][:8]
    print(f"  LDA separability kept-vs-vetoed: {lda_acc:.3f}  (1.0 = fully distinct sub-populations)")
    print(f"  top discriminating features (ANOVA F / AUROC):")
    for i in top:
        print(f"    {fname(int(i)):<12} F={F[i]:>9.1f}  AUROC={auroc[i]:.3f}")
    # interpretable: conf + log_area means, vetoed vs kept
    for mi, lbl in ((512, "conf"), (513, "log_area")):
        vk, vv = X[y == 1, mi].mean(), X[y == 0, mi].mean()
        print(f"  {lbl}: kept mean={vk:.3f}  vetoed mean={vv:.3f}  (Δ={vk-vv:+.3f})")


if __name__ == "__main__":
    for name, rule in [("rgb_dataset_test", "iou"), ("svanstrom", "iop"), ("selcom_val", "iop")]:
        try:
            diagnose(name, rule)
        except Exception as e:
            print(f"[{name} err: {e}]")
