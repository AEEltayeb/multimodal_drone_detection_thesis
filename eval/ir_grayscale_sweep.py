"""ir_grayscale_sweep.py — does the BALANCED IR filter's grayscale head hurt the
grayscale path vs the shipped gray head?  (the balanced retrain changed the SHARED
net, so the gray deploy head shifts too.)  ZERO-GPU.

gray drone recall (gray_svan) + gray confuser FP (gray_confuser), shipped vs balanced
gray head, swept over thresholds.

  py eval/ir_grayscale_sweep.py
"""
from __future__ import annotations
import pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from metrics import compute_prf, score_detections                      # noqa: E402
from pipeline_eval_offline import get_mlp, mlp_probs_per_frame, CACHE   # noqa: E402

SHIP_G = str(REPO / "models/verifiers/ir_aligned/mlp_aligned_gray.pt")
BAL_G = str(REPO / "mri/results/ir_aligned_balanced/classifiers/mlp_aligned_gray.pt")
DRONE_G = ["gray_svan"]; CONF_G = ["gray_confuser"]
THRS = [0.002, 0.01, 0.05, 0.10, 0.25]


def evalw(weight, thr):
    mlp = get_mlp(weight) if weight else None
    rec = {}
    for n in DRONE_G:
        d = pickle.load(open(CACHE / f"{n}.pkl", "rb")); fr = d["frames"]; rule = d["meta"]["rule"]
        probs = mlp_probs_per_frame(fr, mlp) if mlp else None
        tp = fp = fn = 0
        for fi, f in enumerate(fr):
            nn = len(f["confs"])
            keep = (probs[fi] >= thr) if (probs is not None and nn) else (np.ones(nn, bool) if nn else np.zeros(0, bool))
            kept = [(tuple(f["boxes"][i]), float(f["confs"][i])) for i in range(nn) if keep[i]]
            t, f_, n_ = score_detections(kept, [tuple(g) for g in f["gt_boxes"]], rule=rule, iou_thr=0.5, iop_thr=0.5)
            tp += t; fp += f_; fn += n_
        rec[n] = compute_prf(tp, fp, fn)["recall"]
    fps = {}
    for n in CONF_G:
        d = pickle.load(open(CACHE / f"{n}.pkl", "rb")); fr = d["frames"]
        probs = mlp_probs_per_frame(fr, mlp) if mlp else None
        fps[n] = sum(int(((probs[fi] >= thr) if (probs is not None and len(f["confs"])) else np.ones(len(f["confs"]), bool)).sum()) for fi, f in enumerate(fr))
    return rec, fps


def main():
    br, bf = evalw(None, 0)
    print(f"BARE (no gray filter): gray_svan recall={br['gray_svan']:.3f}  gray_confuser FP={bf['gray_confuser']}")
    for tag, w in (("shipped gray", SHIP_G), ("balanced gray", BAL_G)):
        print(f"\n-- {tag} --")
        print(f"{'thr':>6} {'gray_svan R':>12} {'gray_conf FP':>13}")
        for thr in THRS:
            r, fp = evalw(w, thr)
            print(f"{thr:>6.3f} {r['gray_svan']:>12.3f} {fp['gray_confuser']:>13}")
    print("\n(gray drone recall: higher=better; gray_confuser FP: lower=better. bare drone recall is the ceiling.)")


if __name__ == "__main__":
    main()
